"""Tests for the crude autoresearch-loop prototype — rulebook artifact, scoring, orchestration.

Like the rest of the suite these never spawn an agent. The loop's three agent boundaries
(``draft_skill``, ``run_matrix``, ``improve_rulebook``) are monkeypatched, so ``run_iteration``'s
own logic — seeding v1, the rulebook<->skill lockstep, scoring from disk, the moved calculation,
and resume — is exercised while the SDK is never touched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acumen import loop as loop_mod
from acumen import rulebooks as rb
from acumen.bench import PlannedRun
from acumen.config import Config
from acumen.env import Target
from acumen.loop import (
    DraftScores,
    RulebookResult,
    Score,
    score,
)
from acumen.paths import RESULT_FILE, RunKey, run_dir
from acumen.prompts import DRAFT_PROMPT, draft_prompt
from acumen.rulebooks import RulebookError, seed_default, validate_rulebook
from acumen.skills import load_skill, next_version
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"


# ── Rulebook artifact ───────────────────────────────────────────────────────────────────


def test_seed_default_is_idempotent_and_reproduces_the_draft_prompt(tmp_path: Path) -> None:
    assert seed_default(tmp_path) == "v1"
    assert seed_default(tmp_path) == "v1"  # idempotent — does not create v2
    assert rb.load_rulebook(tmp_path, "v1").text == DRAFT_PROMPT
    assert rb.next_version(tmp_path) == "v2"


def test_draft_prompt_with_v1_rulebook_is_byte_identical_to_the_default(tmp_path: Path) -> None:
    seed_default(tmp_path)
    kwargs = {
        "package": "squidpy",
        "version": "1.0",
        "src": Path("/s"),
        "python": Path("/p"),
        "out": Path("/o"),
        "skill_name": "squidpy",
    }
    assert draft_prompt(**kwargs) == draft_prompt(**kwargs, template=rb.load_rulebook(tmp_path, "v1").text)


def test_validate_rulebook_rejects_broken_templates_and_accepts_the_default() -> None:
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("no placeholders at all")
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("only {skill_name} here")  # missing {out}
    with pytest.raises(RulebookError, match="cannot fill"):
        validate_rulebook("{out} {skill_name} but also {bogus}")  # stray field
    validate_rulebook(DRAFT_PROMPT)  # the real template passes


def test_write_rulebook_is_immutable(tmp_path: Path) -> None:
    rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT, parent=None, rationale="seed")
    with pytest.raises(RulebookError, match="immutable"):
        rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT, parent=None, rationale="seed")


# ── Scoring ─────────────────────────────────────────────────────────────────────────────


def _planned(arm: str, split: str, task_id: str, rep: int = 1) -> PlannedRun:
    key = RunKey(arm=arm, split=split, model=MODEL, task_id=task_id, rep=rep)
    return PlannedRun(key=key, task=_task(task_id), model=MODEL, max_turns=1, max_usd=1.0)


def _write_result(runs_root: Path, planned: PlannedRun, *, success: bool, loaded: bool = True, skill=None) -> None:
    """A result.json as the runner writes it — including the hash resume checks it against."""
    directory = run_dir(runs_root, planned.key)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"success": success, "skill_loaded": loaded, "model": MODEL, "answer": "x", "reason": "ok"}
    payload["skill_hash"] = skill.hash if skill is not None else None
    (directory / RESULT_FILE).write_text(json.dumps(payload))


def test_score_counts_passes_and_loads_over_complete_runs(tmp_path: Path) -> None:
    passing = _planned("skill_v1", "test", "a")  # passed and loaded
    failing = _planned("skill_v1", "test", "b")  # failed but loaded
    noload = _planned("skill_v1", "test", "c")  # failed and never loaded
    missing = _planned("skill_v1", "test", "d")  # no result.json written
    _write_result(tmp_path, passing, success=True, loaded=True)
    _write_result(tmp_path, failing, success=False, loaded=True)
    _write_result(tmp_path, noload, success=False, loaded=False)

    result = score(tmp_path, [passing, failing, noload, missing])

    # pass and load are independent metrics: 1/3 passed, 2/3 loaded.
    assert (result.passed, result.total, result.loaded) == (1, 3, 2)
    assert result.rate == pytest.approx(1 / 3)
    assert result.load_rate == pytest.approx(2 / 3)


# ── Orchestration (agents mocked) ───────────────────────────────────────────────────────


def _task(task_id: str = "t") -> Task:
    return Task(id=task_id, train=TaskSplit("train goal", "TRAIN"), test=TaskSplit("test goal", "TEST"))


def _write_skill(skills_root: Path, version: str, name: str) -> None:
    directory = skills_root / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use {name} for its analyses.\n---\n\n# {name}\n\nCall it.\n"
    )


def _target(tmp_path: Path) -> Target:
    return Target(
        source="/pkg",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc123",
        pkg_name="pkg",
        pkg_version="1.0",
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch, cfg: Config, success_by_arm_split: dict, calls: dict) -> None:
    """Monkeypatch the loop's three agent boundaries with filesystem-writing fakes."""

    async def fake_draft(*, cfg, target, skills_root, rulebook, **_):
        calls["draft"] += 1
        version = next_version(skills_root)
        _write_skill(skills_root, version, cfg.skill_name)
        skill = load_skill(skills_root, version, expect_name=cfg.skill_name)
        return SimpleNamespace(skill=skill, cost_usd=0.1)

    async def fake_run_matrix(planned, *, runs_root, skill=None, **_):
        calls["bench"] += 1
        for item in planned:
            _write_result(runs_root, item, success=success_by_arm_split[(item.key.arm, item.key.split)], skill=skill)
        return []

    async def fake_improve(*, rulebooks_root, parent_version, **_):
        calls["improve"] += 1
        text = rb.load_rulebook(rulebooks_root, parent_version).text + "\n<!-- tweak -->\n"
        rb.write_rulebook(
            rulebooks_root, "v2", text, parent=parent_version, rationale="made the description guidance stronger"
        )
        return RulebookResult(
            version="v2",
            parent=parent_version,
            path=rb.rulebook_dir(rulebooks_root, "v2") / rb.RULEBOOK_FILE,
            rationale="made the description guidance stronger",
            changed=True,
            cost_usd=0.2,
            turns=1,
            n_train_runs=1,
            n_train_failures=1,
        )

    monkeypatch.setattr(loop_mod, "draft_skill", fake_draft)
    monkeypatch.setattr(loop_mod, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(loop_mod, "improve_rulebook", fake_improve)


def test_draft_refused_by_the_platform_is_a_pause_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acumen.draft import DraftError
    from acumen.transient import TransientLimitError

    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)

    async def refused(**_):
        raise DraftError("the drafting agent failed: ResultError: You've hit your session limit · resets 8:50am")

    monkeypatch.setattr(loop_mod, "draft_skill", refused)
    with pytest.raises(TransientLimitError, match="rerun to resume"):
        asyncio.run(
            loop_mod._ensure_skill(
                cfg=cfg,
                target=_target(tmp_path),
                skills_root=tmp_path / "skills",
                rulebook_text="draft it",
                expect_version="v1",
                auth_mode="api",
                rationale="test",
                log_dir=None,
                stream=False,
            )
        )
    assert not (tmp_path / "skills" / "v1").exists()  # nothing written; resume starts at the draft


# ── N-draft scoring (draft variance as a reported quantity) ─────────────────────────────


def test_draft_scores_mean_and_spread() -> None:
    ds = DraftScores(
        version="v2",
        scores=[Score(passed=27, total=36), Score(passed=22, total=36), Score(passed=29, total=36)],
        sizes=[100, 200, 300],
    )
    assert ds.mean_rate == pytest.approx((27 + 22 + 29) / (3 * 36))
    assert ds.spread == pytest.approx((29 - 22) / 36)
    empty = DraftScores(version="v1", scores=[], sizes=[])
    assert empty.mean_rate == 0.0 and empty.spread == 0.0 and empty.mean_load_rate == 0.0
