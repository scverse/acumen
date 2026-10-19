"""Tests for generational evolution — improve-from-best, screens, the confirmed ratchet, resume.

Same discipline as test_loop: the three agent boundaries (``draft_skill``, ``run_matrix``,
``improve_rulebook``) are monkeypatched *in acumen.loop* (evolve reuses loop's ensure/bench
helpers), so the decisions under test — who improvement builds from, what the screen decides, when
the ratchet promotes or reverts, what the journal records — run against real files and fake agents.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acumen import evolve as evolve_mod
from acumen import loop as loop_mod
from acumen import rulebooks as rb
from acumen.config import Config
from acumen.env import Target
from acumen.evolve import (
    DIRECTIVES,
    JOURNAL_FILE,
    EvolveRun,
    IslandResult,
    LoopError,
    _champion_chain,
    _write_island_materials,
    directive_for,
    run_archipelago,
    run_evolve,
    screen_subset,
)
from acumen.folds import write_lockbox
from acumen.loop import RulebookResult
from acumen.paths import RESULT_FILE, run_dir
from acumen.skills import load_skill, next_version
from acumen.taskgen import dump_tasks
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"


def test_directive_rotation_is_deterministic() -> None:
    assert directive_for(1) == DIRECTIVES[0]
    assert directive_for(len(DIRECTIVES) + 1) == DIRECTIVES[0]
    assert directive_for(2) == DIRECTIVES[1]
    with pytest.raises(LoopError):
        directive_for(0)


def test_screen_subset_is_seeded_stable_and_rotates_by_epoch() -> None:
    ids = [f"t{i}" for i in range(30)]
    a = screen_subset(ids, 12, epoch=0, seed=7)
    assert a == screen_subset(list(reversed(ids)), 12, epoch=0, seed=7)  # order-independent
    assert len(a) == 12 and a == sorted(a)
    assert screen_subset(ids, 12, epoch=1, seed=7) != a  # rotation
    assert screen_subset(ids, 50, epoch=0, seed=7) == sorted(ids)  # capped at the pool
    with pytest.raises(LoopError):
        screen_subset(ids, 0, epoch=0, seed=7)


# ── Faked-agent evolution runs ──────────────────────────────────────────────────────────


def _task(task_id: str) -> Task:
    split = TaskSplit(prompt=f"solve {task_id}", answer="42")
    return Task(id=task_id, train=split, test=split)


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


def _write_skill(skills_root: Path, version: str, name: str) -> None:
    d = skills_root / version
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: fake {version}\n---\nbody\n")


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Config,
    calls: dict,
    improves: list[dict],
    passes: dict[str, set[str]],
) -> None:
    """``passes[version]`` = the task ids that version's skill passes on the test split."""

    async def fake_draft(*, cfg, target, skills_root, rulebook, **_):
        calls["draft"] += 1
        version = next_version(skills_root)
        _write_skill(skills_root, version, cfg.skill_name)
        return SimpleNamespace(skill=load_skill(skills_root, version, expect_name=cfg.skill_name), cost_usd=0.1)

    async def fake_run_matrix(planned, *, runs_root, skill=None, **_):
        calls["bench"] += 1
        # Extra drafts each live in their own skills-root holding a single v1, so their arm is
        # always "skill_v1". The run tree is what says which version they are a draft OF:
        # runs/drafts/<version>/d<i>. Real drafts share the rulebook, so they share its behaviour.
        parts = Path(runs_root).parts
        variant = parts[parts.index("drafts") + 1] if "drafts" in parts else None
        for item in planned:
            version = variant or item.key.arm.removeprefix("skill_")
            ok = item.key.split == "train" or item.key.task_id in passes.get(version, set())
            d = run_dir(runs_root, item.key)
            d.mkdir(parents=True, exist_ok=True)
            payload = {"success": ok, "skill_loaded": True, "skill_hash": skill.hash if skill else None}
            (d / RESULT_FILE).write_text(json.dumps(payload))
        return []

    async def fake_improve(*, rulebooks_root, parent_version, tasks, feedback=None, deny_dirs=(), **_):
        calls["improve"] += 1
        improves.append(
            {
                "parent": parent_version,
                "tasks": sorted(t.id for t in tasks),
                "feedback": feedback,
                "deny": [Path(d) for d in deny_dirs],
            }
        )
        new = rb.next_version(rulebooks_root)
        text = rb.load_rulebook(rulebooks_root, parent_version).text + f"\n<!-- {new} from {parent_version} -->\n"
        rb.write_rulebook(rulebooks_root, new, text, parent=parent_version, rationale="tweak", feedback=feedback)
        return RulebookResult(
            version=new,
            parent=parent_version,
            path=rb.rulebook_dir(rulebooks_root, new) / rb.RULEBOOK_FILE,
            rationale="tweak",
            changed=True,
            cost_usd=0.2,
            turns=1,
            n_train_runs=len(tasks),
            n_train_failures=1,
        )

    monkeypatch.setattr(loop_mod, "draft_skill", fake_draft)
    monkeypatch.setattr(loop_mod, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(loop_mod, "improve_rulebook", fake_improve)


def _evolve(tmp_path: Path, cfg: Config, tasks: list[Task], **kw):
    return asyncio.run(
        run_evolve(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=tasks,
            auth_mode="session",
            **kw,
        )
    )


def test_evolve_improves_from_best_ratchets_and_journals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v2 wins its screen and is confirmed (new champion); v3 — improved FROM v2 — loses and is
    rejected; v4 — improved from v2 again, not from the rejected v3 — ties and is rejected too."""
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    ids = [f"t{i}" for i in range(4)]
    passes = {"v1": set(), "v2": set(ids), "v3": set(), "v4": set(ids)}
    _install_fakes(monkeypatch, cfg, calls, improves, passes)
    tasks = [_task(t) for t in ids]
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("z1"), _task("z2")], seed=0, fraction=0.3, dump=dump_tasks)
    passes["v2"].update({"z1", "z2"})  # the champion's lockbox showing
    seen: list[tuple[int, str, bool]] = []

    run = _evolve(
        tmp_path,
        cfg,
        tasks,
        generations=3,
        screen_size=4,
        accept_delta=2,
        confirm_every=1,
        n_drafts=1,
        lockbox_dir=lockbox.directory,
        on_generation=lambda gen: seen.append((gen.index, gen.parent, gen.accepted)),
    )

    # Gen 1: v1 -> v2 accepted + confirmed. Gen 2: parent is v2 (the best), v3 rejected.
    # Gen 3: parent is STILL v2 — never the rejected v3. That is improve-from-best.
    assert seen == [(1, "v1", True), (2, "v2", False), (3, "v2", False)]
    assert [c["parent"] for c in improves] == ["v1", "v2", "v2"]
    assert run.champion == "v2" and run.accepted == 1
    assert run.stopped_because.startswith("reached the generation budget")
    # v4 passed everything but tied v2 on the screen — accept_delta refused the churn.
    assert not run.generations[2].accepted

    # Every improve saw only screen-subset evidence and was denied the lockbox + drafts trees.
    for c in improves:
        assert set(c["tasks"]) <= set(ids) and len(c["tasks"]) == 4
        assert lockbox.directory in c["deny"]
        assert any(d.name == loop_mod.DRAFTS_DIRNAME for d in c["deny"])
    # The directive rotation reached the improve agent as feedback.
    assert improves[0]["feedback"] == DIRECTIVES[0] and improves[1]["feedback"] == DIRECTIVES[1]

    # Journal: one line per generation, replayable.
    lines = [json.loads(x) for x in (tmp_path / "runs" / JOURNAL_FILE).read_text().splitlines()]
    assert [x["generation"] for x in lines] == [1, 2, 3]
    assert lines[0]["accepted"] and lines[0]["confirm_promoted"] and lines[0]["confirmed"] == "v2"
    assert lines[1]["directive"] == DIRECTIVES[1]

    # Lockbox verdict: champion v2 vs seed v1, single draft.
    assert run.lockbox_baseline_drafts is not None and run.lockbox_baseline_drafts.scores[0].passed == 0
    assert run.lockbox_drafts is not None and run.lockbox_drafts.scores[0].passed == 2
    assert run.lockbox_mean_delta == 1.0


def test_evolve_reverts_a_failed_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate can win a small screen and still lose the full bench: the ratchet reverts it."""
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    ids = [f"t{i}" for i in range(6)]
    # v1 passes 4 of 6 overall; v2 sweeps the epoch-0 screen pair but fails everything else.
    subset = screen_subset(ids, 2, epoch=0, seed=0)
    passes = {"v1": set(ids) - set(subset), "v2": set(subset), "v3": set()}
    _install_fakes(monkeypatch, cfg, calls, improves, passes)
    tasks = [_task(t) for t in ids]

    run = _evolve(
        tmp_path,
        cfg,
        tasks,
        generations=2,
        screen_size=2,
        accept_delta=2,
        confirm_every=1,
        allow_no_lockbox=True,
        evaluate_lockbox=False,
        seed=0,
    )

    gen1 = run.generations[0]
    assert gen1.accepted and gen1.confirm_ran and gen1.confirm_promoted is False
    assert gen1.confirmed == "v1"
    # Generation 2 improved from v1 again — the reverted v2 is not the parent.
    assert run.generations[1].parent == "v1" and improves[1]["parent"] == "v1"
    assert run.champion == "v1"
    assert run.lockbox_drafts is None  # islands don't open the lockbox


def test_evolve_resumes_without_respawning_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    ids = [f"t{i}" for i in range(4)]
    passes = {"v1": set(), "v2": set(ids), "v3": set()}
    _install_fakes(monkeypatch, cfg, calls, [], passes)
    tasks = [_task(t) for t in ids]
    kw = {
        "generations": 2,
        "screen_size": 4,
        "accept_delta": 2,
        "confirm_every": 1,
        "allow_no_lockbox": True,
        "evaluate_lockbox": False,
    }

    first = _evolve(tmp_path, cfg, tasks, **kw)
    assert calls["draft"] == 3 and calls["improve"] == 2  # v1..v3 drafted, two improves
    calls.update(draft=0, bench=0, improve=0)
    second = _evolve(tmp_path, cfg, tasks, **kw)

    assert calls == {"draft": 0, "bench": 0, "improve": 0}
    assert second.champion == first.champion == "v2"
    assert [g.accepted for g in second.generations] == [g.accepted for g in first.generations]
    # The journal was not double-appended on replay.
    lines = (tmp_path / "runs" / JOURNAL_FILE).read_text().splitlines()
    assert len(lines) == 2


def test_evolve_requires_a_lockbox_and_sane_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    _install_fakes(monkeypatch, cfg, calls, [], {})
    tasks = [_task(t) for t in ("a", "b")]

    with pytest.raises(LoopError, match="no lockbox"):
        _evolve(tmp_path, cfg, tasks, generations=1)
    for bad in (
        {"generations": 0},
        {"generations": 1, "accept_delta": 0},
        {"generations": 1, "confirm_every": 0},
        {"generations": 1, "epoch_len": 0},
        {"generations": 1, "n_drafts": 0},
    ):
        with pytest.raises(LoopError):
            _evolve(tmp_path, cfg, tasks, allow_no_lockbox=True, **bad)
    assert calls == {"draft": 0, "bench": 0, "improve": 0}  # all refused before any agent ran


# ── Islands and cross-pollination ───────────────────────────────────────────────────────


def test_champion_chain_and_island_materials(tmp_path: Path) -> None:
    """The chain holds only the edits that survived; the materials lay them out per island."""
    root = tmp_path / "rulebooks"
    assert rb.seed_default(root) == "v1"
    base = rb.load_rulebook(root, "v1").text
    rb.write_rulebook(root, "v2", base + "\n<!-- e1 -->\n", parent="v1", rationale="e1")
    rb.write_rulebook(root, "v3", base + "\n<!-- e1 --><!-- e2 -->\n", parent="v2", rationale="e2")
    rb.write_rulebook(root, "v4", base + "\n<!-- rejected -->\n", parent="v2", rationale="dead end")

    assert _champion_chain(root, "v3") == [("v1", "v2"), ("v2", "v3")]
    assert _champion_chain(root, "v1") == []

    journal = tmp_path / "runs" / JOURNAL_FILE
    journal.parent.mkdir(parents=True)
    journal.write_text('{"generation": 1}\n')
    island = IslandResult(
        index=0,
        task_ids=("a",),
        run=EvolveRun(generations=[], champion="v3", stopped_because="test", journal_path=journal),
        rulebooks_root=root,
        skills_root=tmp_path / "skills",
        runs_root=tmp_path / "runs",
    )
    dest = tmp_path / "materials"
    _write_island_materials(dest, [island])
    d = dest / "island-0"
    assert (d / "champion.md").read_text() == rb.load_rulebook(root, "v3").text
    assert (d / "journal.jsonl").read_text() == journal.read_text()
    diffs = sorted(p.name for p in (d / "diffs").iterdir())
    assert diffs == ["v1-to-v2.diff", "v2-to-v3.diff"]  # the rejected v4 is on nobody's chain
    assert "<!-- e1 -->" in (d / "diffs" / "v1-to-v2.diff").read_text()


def test_archipelago_evolves_islands_pollinates_validates_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    ids = [f"t{i}" for i in range(6)]
    passes = {"v1": set(), "v2": set(ids) | {"z1", "z2"}}
    _install_fakes(monkeypatch, cfg, calls, improves, passes)
    tasks = [_task(t) for t in ids]
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("z1"), _task("z2")], seed=0, fraction=0.3, dump=dump_tasks)

    polls: list[dict] = []

    async def fake_pollinate(*, cfg, target, rulebooks_root, islands, parent_version="v1", deny_dirs=(), **_):
        polls.append(
            {
                "islands": [(i.index, i.run.champion, i.task_ids) for i in islands],
                "deny": [Path(d) for d in deny_dirs],
            }
        )
        text = rb.load_rulebook(rulebooks_root, parent_version).text + "\n<!-- merged -->\n"
        rb.write_rulebook(rulebooks_root, "v2", text, parent=parent_version, rationale="meta-rule: replicated")
        return RulebookResult(
            version="v2",
            parent=parent_version,
            path=rb.rulebook_dir(rulebooks_root, "v2") / rb.RULEBOOK_FILE,
            rationale="meta-rule: replicated",
            changed=True,
            cost_usd=0.3,
            turns=1,
            n_train_runs=0,
            n_train_failures=0,
        )

    monkeypatch.setattr(evolve_mod, "pollinate", fake_pollinate)
    kw = {
        "cfg": cfg,
        "target": _target(tmp_path),
        "skills_root": tmp_path / "skills",
        "rulebooks_root": tmp_path / "rulebooks",
        "runs_root": tmp_path / "runs",
        "tasks": tasks,
        "k": 2,
        "generations": 1,
        "screen_size": 3,
        "accept_delta": 2,
        "confirm_every": 1,
        "n_drafts": 1,
        "lockbox_dir": lockbox.directory,
        "auth_mode": "session",
    }

    run = asyncio.run(run_archipelago(**kw))

    # Partitions: disjoint, covering, one per island; each island evolved in its own trees.
    pools = [set(isl.task_ids) for isl in run.islands]
    assert pools[0] & pools[1] == set() and pools[0] | pools[1] == set(ids)
    for isl in run.islands:
        assert isl.run.champion == "v2" and isl.run.lockbox_drafts is None  # islands never open the lockbox
        assert evolve_mod.island_root(tmp_path / "rulebooks", isl.index).is_dir()
        assert (isl.runs_root / JOURNAL_FILE).is_file()
    # Every island improve agent was denied the ENTIRE main run tree (and thus every sibling).
    assert improves and all(any(d == tmp_path / "runs" for d in c["deny"]) for c in improves)

    # Pollination saw both champions; the merged version landed on the main chain.
    assert len(polls) == 1
    assert [(i, champ) for i, champ, _ in polls[0]["islands"]] == [(1, "v2"), (2, "v2")]  # fold indices are 1-based
    assert run.merged.version == "v2" and rb.available_versions(tmp_path / "rulebooks") == ["v1", "v2"]

    # Cross-island validation on the FULL working set, and the lockbox verdict.
    assert (run.seed_test.passed, run.seed_test.total) == (0, 6)
    assert (run.merged_test.passed, run.merged_test.total) == (6, 6)
    assert run.lockbox_baseline_drafts is not None and run.lockbox_baseline_drafts.scores[0].passed == 0
    assert run.lockbox_drafts is not None and run.lockbox_drafts.scores[0].passed == 2
    assert run.lockbox_mean_delta == 1.0

    # Resume: islands, pollination, benches all replay from disk — zero agents.
    calls.update(draft=0, bench=0, improve=0)
    polls.clear()
    again = asyncio.run(run_archipelago(**kw))
    assert calls == {"draft": 0, "bench": 0, "improve": 0} and polls == []
    assert again.merged.version == "v2" and again.merged.rationale == "meta-rule: replicated"
    assert again.lockbox_mean_delta == 1.0

    with pytest.raises(LoopError, match="k >= 2"):
        asyncio.run(run_archipelago(**{**kw, "k": 1}))


def test_screen_drafts_scores_each_version_over_n_drafts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With --screen-drafts N a version is drafted N times, benched in N trees, and judged on the sum.

    The accept bar scales with N, so ``accept_delta`` keeps meaning passes *per draft*: v2 passes
    everything and v1 nothing, so the summed gap is 4 * N against a bar of 2 * N.
    """
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    ids = [f"t{i}" for i in range(4)]
    passes = {"v1": set(), "v2": set(ids)}
    _install_fakes(monkeypatch, cfg, calls, [], passes)

    run = _evolve(
        tmp_path,
        cfg,
        [_task(t) for t in ids],
        generations=1,
        screen_size=4,
        screen_drafts=3,
        accept_delta=2,
        confirm_every=1,
        allow_no_lockbox=True,
        evaluate_lockbox=False,
    )

    gen = run.generations[0]
    # Summed over 3 drafts of a 4-task screen: the champion 0/12, the candidate 12/12.
    assert (gen.champion_screen.passed, gen.champion_screen.total) == (0, 12)
    assert (gen.candidate_screen.passed, gen.candidate_screen.total) == (12, 12)
    assert gen.accepted

    # Draft 1 stays the primary tree; drafts 2..N get their own, so no arm collides.
    for version in ("v1", "v2"):
        assert (tmp_path / "skills" / version / "SKILL.md").is_file()
        for i in (2, 3):
            assert (tmp_path / "skills" / "drafts" / version / f"d{i}" / "v1" / "SKILL.md").is_file()
            assert (tmp_path / "runs" / "drafts" / version / f"d{i}").is_dir()


def test_screen_drafts_bar_rejects_a_win_that_only_one_draft_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that beats the champion by the bar on ONE draft fails once N drafts are summed.

    This is the whole point: at ``screen_drafts=1`` a +2 win is indistinguishable from draft luck.
    Here the candidate gains 2 tasks per draft against a bar of 3 per draft, and is rejected.
    """
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    ids = [f"t{i}" for i in range(4)]
    passes = {"v1": {"t0"}, "v2": {"t0", "t1", "t2"}}
    _install_fakes(monkeypatch, cfg, calls, [], passes)

    run = _evolve(
        tmp_path,
        cfg,
        [_task(t) for t in ids],
        generations=1,
        screen_size=4,
        screen_drafts=2,
        accept_delta=3,
        confirm_every=1,
        allow_no_lockbox=True,
        evaluate_lockbox=False,
    )

    gen = run.generations[0]
    assert gen.candidate_screen.passed - gen.champion_screen.passed == 4  # 2 per draft, 2 drafts
    assert not gen.accepted  # the bar is 3 * 2 = 6
    assert run.champion == "v1"


def test_screen_drafts_resume_adds_only_the_missing_drafts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running at the same N spawns no agents; the run is replayed from disk."""
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    ids = [f"t{i}" for i in range(4)]
    passes = {"v1": set(), "v2": set(ids)}
    _install_fakes(monkeypatch, cfg, calls, [], passes)
    kw = {
        "generations": 1,
        "screen_size": 4,
        "screen_drafts": 2,
        "accept_delta": 2,
        "confirm_every": 1,
        "allow_no_lockbox": True,
        "evaluate_lockbox": False,
    }
    first = _evolve(tmp_path, cfg, [_task(t) for t in ids], **kw)
    drafted, improved = calls["draft"], calls["improve"]
    assert drafted > 0

    second = _evolve(tmp_path, cfg, [_task(t) for t in ids], **kw)

    assert calls["draft"] == drafted and calls["improve"] == improved  # no agent ran again
    assert second.champion == first.champion
    assert second.generations[0].candidate_screen.passed == first.generations[0].candidate_screen.passed


def test_screen_drafts_must_be_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    _install_fakes(monkeypatch, cfg, {"draft": 0, "bench": 0, "improve": 0}, [], {"v1": set()})
    with pytest.raises(LoopError, match="screen_drafts must be >= 1"):
        _evolve(tmp_path, cfg, [_task("t0")], generations=1, screen_drafts=0, allow_no_lockbox=True)
