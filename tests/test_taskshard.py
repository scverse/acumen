"""Tests for task generation — the pure sharding seams and the fan-out orchestration.

Like the rest of the suite these never spawn a real agent: the one place an agent would run
(:func:`acumen.taskgen._run_generation_agent`) is monkeypatched, so the orchestration around it
— notebook enumeration, per-shard resume, failure isolation, and the merge — is exercised
honestly while the SDK is never touched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from acumen import taskshard
from acumen.check import SCRIPTS_DIRNAME
from acumen.config import Config
from acumen.env import Target
from acumen.prompts import taskgen_shard_prompt
from acumen.taskgen import TaskGenError, dump_tasks
from acumen.tasks import Task, TaskSplit, load_tasks
from acumen.taskshard import _namespace_task, _shard_slug, generate_tasks_sharded, merge_shards, notebook_shards


def _task(task_id: str = "t") -> Task:
    return Task(id=task_id, train=TaskSplit("train goal", "TRAIN"), test=TaskSplit("test goal", "TEST"))


def _write_notebooks(src: Path, rels: list[str]) -> None:
    for rel in rels:
        nb = src / rel
        nb.parent.mkdir(parents=True, exist_ok=True)
        nb.write_text("{}")


def _target(src: Path) -> Target:
    return Target(
        source=str(src),
        ref="main",
        src_dir=src,
        venv_dir=src.parent / "venv",
        commit="abc123",
        pkg_name="pkg",
        pkg_version="1.0",
    )


# ── Pure sharding seams ─────────────────────────────────────────────────────────────────


def test_notebook_shards_enumerates_recursively_and_skips_checkpoints(tmp_path: Path) -> None:
    _write_notebooks(
        tmp_path,
        [
            "docs/notebooks/tutorials/a.ipynb",
            "docs/notebooks/examples/b.ipynb",
            ".ipynb_checkpoints/a.ipynb",
            "docs/.ipynb_checkpoints/c.ipynb",
            "docs/notes.md",
        ],
    )
    found = [nb.as_posix() for nb in notebook_shards(tmp_path)]
    assert found == ["docs/notebooks/examples/b.ipynb", "docs/notebooks/tutorials/a.ipynb"]


def test_shard_slug_disambiguates_same_basename_in_different_galleries() -> None:
    a = _shard_slug(Path("docs/notebooks/tutorials/plotting.ipynb"))
    b = _shard_slug(Path("docs/notebooks/examples/plotting.ipynb"))
    assert a != b
    assert all(ch.isalnum() or ch in "._-" for ch in a + b)


def test_merge_shards_namespaces_ids_and_enforces_uniqueness(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    # Two shards independently pick the same id 'foo' — the merge must not collide.
    (shards / "alpha.yaml").write_text(dump_tasks([_task("foo")]))
    (shards / "beta.yaml").write_text(dump_tasks([_task("foo")]))
    out = tmp_path / "tasks.yaml"

    merged = merge_shards(shards, out)

    assert sorted(t.id for t in merged) == ["alpha__foo", "beta__foo"]
    # The written file re-parses cleanly through the strict loader.
    assert sorted(t.id for t in load_tasks(out)) == ["alpha__foo", "beta__foo"]


def test_merge_shards_with_no_tasks_errors(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    with pytest.raises(TaskGenError, match="no tasks"):
        merge_shards(shards, tmp_path / "tasks.yaml")


def test_namespace_task_keeps_ids_filesystem_safe() -> None:
    out = _namespace_task(_task("some.id-1"), "docs-notebooks-a")
    assert out.id == "docs-notebooks-a__some.id-1"


def test_shard_prompt_scopes_to_one_notebook_and_stays_feedback_stable() -> None:
    kwargs = {
        "package": "pkg",
        "src": Path("/s"),
        "python": Path("/p"),
        "out": Path("/o/tasks.yaml"),
        "scripts_dir": Path("/o/scripts"),
    }
    prompt = taskgen_shard_prompt(**kwargs, notebook="docs/nb/tangram.ipynb")
    # feedback=None is byte-identical to no feedback (mirrors the draft-prompt guarantee).
    assert prompt == taskgen_shard_prompt(**kwargs, notebook="docs/nb/tangram.ipynb", feedback=None)
    # The one assigned notebook is named; the whole-package "enumerate them FIRST" scope is gone.
    assert "docs/nb/tangram.ipynb" in prompt
    assert "Your one tutorial" in prompt
    assert "enumerate them FIRST" not in prompt
    # No unfilled template placeholders leaked through.
    for placeholder in ("{scope}", "{coverage_check}", "{notebook}"):
        assert placeholder not in prompt


# ── Fan-out orchestration (agent monkeypatched) ─────────────────────────────────────────


class _FakeResult:
    """A stand-in for the agent result: only the fields the orchestration reads."""

    num_turns = 2


def _fake_agent(fail_slugs: set[str], seen: dict[str, str]):
    """Build a fake ``_run_shard_agent`` that records prompts, writes scripts, and fails chosen shards."""

    async def run(*, work_root: Path, make_prompt, **_: object):
        slug = work_root.name
        # Calling make_prompt proves each shard is handed ITS OWN notebook (no late-binding bug).
        scripts = work_root / "work" / SCRIPTS_DIRNAME
        scripts.mkdir(parents=True, exist_ok=True)
        seen[slug] = make_prompt(work_root / "work" / "tasks.yaml", scripts)
        if slug in fail_slugs:
            raise TaskGenError(f"boom in {slug}")
        for split in ("train", "test"):
            (scripts / f"t-{split}.py").write_text("import pkg\npkg.f()\n")
        return [_task("t")], _FakeResult(), 0.25

    return run


def _run_sharded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_slugs, seen, precreate=None):
    src = tmp_path / "src"
    _write_notebooks(src, ["docs/nb/a.ipynb", "docs/nb/b.ipynb", "docs/nb/c.ipynb"])
    shards = tmp_path / "shards"
    if precreate:
        shards.mkdir(parents=True, exist_ok=True)
        for slug in precreate:
            (shards / f"{slug}.yaml").write_text(dump_tasks([_task("cachedtask")]))
    monkeypatch.setattr(taskshard, "_run_shard_agent", _fake_agent(set(fail_slugs), seen))
    cfg = Config(repo=str(src), skill_name="pkg", max_concurrency=2)
    return asyncio.run(
        generate_tasks_sharded(
            cfg=cfg,
            target=_target(src),
            out_path=tmp_path / "tasks.yaml",
            shards_dir=shards,
        )
    )


def test_generate_tasks_sharded_merges_all_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    result = _run_sharded(tmp_path, monkeypatch, fail_slugs=[], seen=seen)

    assert result.n_ok == 3 and result.n_failed == 0
    assert len(result.tasks) == 3
    # ids are namespaced by shard slug, so the merged set is unique and traceable.
    assert sorted(t.id for t in result.tasks) == ["docs-nb-a__t", "docs-nb-b__t", "docs-nb-c__t"]
    # Each shard was prompted with its own notebook path.
    assert "docs/nb/a.ipynb" in seen["docs-nb-a"]
    assert "docs/nb/b.ipynb" in seen["docs-nb-b"]
    assert result.cost_usd == pytest.approx(0.75)
    # Confirmation scripts are collected under the SAME namespaced id as the tasks, so coverage
    # keys line up with tasks.yaml ids.
    scripts = tmp_path / SCRIPTS_DIRNAME
    names = {p.name for p in scripts.glob("*.py")}
    assert {"docs-nb-a__t-train.py", "docs-nb-b__t-train.py", "docs-nb-c__t-train.py"} <= names
    assert "import pkg" in (scripts / "docs-nb-a__t-train.py").read_text()


def test_generate_tasks_sharded_isolates_a_failed_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    result = _run_sharded(tmp_path, monkeypatch, fail_slugs=["docs-nb-b"], seen=seen)

    assert result.n_ok == 2 and result.n_failed == 1
    failed = [o for o in result.outcomes if o.status == "failed"]
    assert [o.slug for o in failed] == ["docs-nb-b"]
    assert "boom" in failed[0].error
    # The two survivors still merged; the failed shard wrote no file.
    assert sorted(t.id for t in result.tasks) == ["docs-nb-a__t", "docs-nb-c__t"]
    assert not (tmp_path / "shards" / "docs-nb-b.yaml").exists()


def test_generate_tasks_sharded_resumes_cached_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    result = _run_sharded(tmp_path, monkeypatch, fail_slugs=[], seen=seen, precreate=["docs-nb-a"])

    # The cached shard's agent is never invoked...
    assert "docs-nb-a" not in seen
    assert set(seen) == {"docs-nb-b", "docs-nb-c"}
    # ...but its tasks still land in the merge, under its slug namespace.
    cached = next(o for o in result.outcomes if o.slug == "docs-nb-a")
    assert cached.status == "cached"
    assert "docs-nb-a__cachedtask" in {t.id for t in result.tasks}


def test_generate_tasks_sharded_no_notebooks_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "readme.md").write_text("no notebooks here")
    monkeypatch.setattr(taskshard, "_run_shard_agent", _fake_agent(set(), {}))
    cfg = Config(repo=str(src), skill_name="pkg")
    with pytest.raises(TaskGenError, match="no notebooks"):
        asyncio.run(
            generate_tasks_sharded(
                cfg=cfg, target=_target(src), out_path=tmp_path / "tasks.yaml", shards_dir=tmp_path / "shards"
            )
        )


def test_generate_tasks_sharded_refuses_existing_out_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    _write_notebooks(src, ["docs/nb/a.ipynb"])
    out = tmp_path / "tasks.yaml"
    out.write_text("existing")
    monkeypatch.setattr(taskshard, "_run_shard_agent", _fake_agent(set(), {}))
    cfg = Config(repo=str(src), skill_name="pkg")
    with pytest.raises(TaskGenError, match="already exists"):
        asyncio.run(generate_tasks_sharded(cfg=cfg, target=_target(src), out_path=out, shards_dir=tmp_path / "shards"))


def test_mined_prompt_replaces_the_whole_package_scope() -> None:
    from acumen.prompts import taskgen_mined_prompt

    prompt = taskgen_mined_prompt(
        package="pkg",
        src=Path("/s"),
        python=Path("/p"),
        out=Path("/w/tasks.yaml"),
        scripts_dir=Path("/w/scripts"),
        analysis=Path("/w/analysis-x.py"),
    )
    assert "/w/analysis-x.py" in prompt and "Your one analysis" in prompt
    assert "enumerate them FIRST" not in prompt
    assert "at least one task per tutorial" not in prompt


def test_sharded_generation_pauses_after_a_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform limit on one shard must not fail every remaining shard against the same wall."""
    src = tmp_path / "src"
    _write_notebooks(src, ["docs/nb/a.ipynb", "docs/nb/b.ipynb", "docs/nb/c.ipynb"])
    ran: list[str] = []

    async def limited_agent(*, work_root: Path, make_prompt, **_):
        ran.append(work_root.name)
        if len(ran) == 1:
            return [_task("t")], _FakeResult(), 0.1  # the first shard lands
        raise TaskGenError("ResultError: You've hit your session limit · resets 7:20am")

    monkeypatch.setattr(taskshard, "_run_shard_agent", limited_agent)
    cfg = Config(repo=str(src), skill_name="pkg", max_concurrency=1)
    result = asyncio.run(
        generate_tasks_sharded(
            cfg=cfg, target=_target(src), out_path=tmp_path / "tasks.yaml", shards_dir=tmp_path / "shards"
        )
    )

    # One landed, one hit the wall, the third was skipped without an agent ever starting.
    assert len(ran) == 2
    assert result.paused and "session limit" in result.paused
    assert result.n_ok == 1 and result.n_failed == 2
    assert sum(1 for o in result.outcomes if (o.error or "").startswith("skipped:")) == 1
    assert len(result.tasks) == 1  # the merged file holds what landed
    assert len(list((tmp_path / "shards").glob("*.yaml"))) == 1  # nothing written for the refused/skipped shards
