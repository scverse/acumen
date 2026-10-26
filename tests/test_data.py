"""The prepared data cache: ``data:`` in tasks.yaml, ``acumen prepare``, and its delivery into runs."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import yaml

from acumen.agents import AgentResult
from acumen.check import check_tasks
from acumen.cli import main
from acumen.data import DataError, all_splits, data_sources, link_data, missing_data, require_data
from acumen.env import Target
from acumen.guard import find_escape
from acumen.paths import RunKey
from acumen.prepare import MARKER_DIRNAME, PrepareError, prepare_scripts, run_prepare, unfreeze
from acumen.prompts import benchmark_prompt
from acumen.runner import run_once
from acumen.sandbox import Sandbox
from acumen.taskgen import dump_tasks
from acumen.tasks import Task, TaskError, TaskSplit, parse_tasks


def _entry(**overrides) -> dict:
    entry = {
        "id": "t",
        "train": {"prompt": "p", "answer": "a"},
        "valid": {"prompt": "p", "answer": "b"},
        "test": {"prompt": "p", "answer": "c"},
    }
    entry.update(overrides)
    return entry


def _fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped directory whose interpreter is the one running the tests."""
    venv = tmp_path / "venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if os.name == "nt" else "python")).symlink_to(sys.executable)
    return venv


def _target(tmp_path: Path, venv: Path | None = None) -> Target:
    return Target(
        source="target",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=venv or tmp_path / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )


def _cache(tmp_path: Path) -> Path:
    """A data cache holding one file and one nested directory."""
    cache = tmp_path / "cache"
    (cache / "atlas").mkdir(parents=True)
    (cache / "atlas" / "cells.txt").write_text("CELLS")
    (cache / "pbmc.h5ad").write_text("PBMC")
    return cache


def _writable(path: Path) -> bool:
    return bool(path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


# --- tasks.yaml ------------------------------------------------------------------------


def test_split_data_adds_to_the_task_list() -> None:
    (task,) = parse_tasks(
        {
            "tasks": [
                _entry(
                    data=["net.csv", "atlas/"],
                    test={"prompt": "p", "answer": "c", "data": ["covid.h5ad"]},
                    valid={"prompt": "p", "answer": "b", "data": []},
                )
            ]
        }
    )

    assert task.data_for("train") == ("net.csv", "atlas")  # inherited; a trailing slash is dropped
    assert task.data_for("test") == ("net.csv", "atlas", "covid.h5ad")  # merged, task's first
    assert task.data_for("valid") == ("net.csv", "atlas")  # an empty list adds nothing
    assert parse_tasks({"tasks": [_entry()]})[0].data_for("train") == ()


@pytest.mark.parametrize(
    ("split_data", "match"),
    [(["net.csv"], "twice \\(the task's data: already has it\\)"), (["atlas/cells.txt"], "one inside the other")],
)
def test_split_data_may_not_repeat_the_task_list(split_data, match: str) -> None:
    with pytest.raises(TaskError, match=match):
        parse_tasks(
            {"tasks": [_entry(data=["net.csv", "atlas"], train={"prompt": "p", "answer": "a", "data": split_data})]}
        )


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ("pbmc.h5ad", "must be a list"),
        (["/abs/pbmc.h5ad"], "relative path"),
        (["../outside"], "relative path"),
        (["a/../../b"], "relative path"),
        ([".claude/skills"], "collide"),
        (["answer.md"], "collide"),
        (["x", "x"], "twice"),
        (["atlas", "atlas/cells.txt"], "one inside the other"),
        (["atlas/cells.txt", "atlas/"], "one inside the other"),
        ([""], "non-empty string"),
    ],
)
def test_data_entries_must_stay_inside_the_cache(data, match: str) -> None:
    with pytest.raises(TaskError, match=match):
        parse_tasks({"tasks": [_entry(data=data)]})
    with pytest.raises(TaskError, match=match):
        parse_tasks({"tasks": [_entry(train={"prompt": "p", "answer": "a", "data": data})]})


def test_dump_tasks_round_trips_data() -> None:
    (task,) = parse_tasks(
        {"tasks": [_entry(data=["pbmc.h5ad"], test={"prompt": "p", "answer": "c", "data": ["covid.h5ad"]})]}
    )

    (again,) = parse_tasks(yaml.safe_load(dump_tasks([task])))

    assert again == task


# --- the cache -------------------------------------------------------------------------


def test_missing_data_counts_only_the_splits_about_to_run(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    task = Task(
        id="t",
        train=TaskSplit("p", "a"),
        valid=TaskSplit("p", "b"),
        test=TaskSplit("p", "c", data=("covid.h5ad",)),
        data=("pbmc.h5ad", "atlas/cells.txt", "nope.csv"),
    )

    assert missing_data([(task, "train")], cache) == ["nope.csv"]
    assert missing_data(all_splits([task]), cache) == ["nope.csv", "covid.h5ad"]
    with pytest.raises(DataError, match=r"covid\.h5ad — run `acumen prepare`"):
        require_data([(task, "test")], cache)


def test_link_data_mirrors_each_entry_at_its_relative_path(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    link_data(work, data_sources(["pbmc.h5ad", "atlas"], cache))

    assert (work / "pbmc.h5ad").is_symlink()
    assert (work / "pbmc.h5ad").read_text() == "PBMC"
    assert (work / "atlas" / "cells.txt").read_text() == "CELLS"
    nested = tmp_path / "nested"
    link_data(nested, data_sources(["atlas/cells.txt"], cache))  # a nested entry gets real parents
    assert not (nested / "atlas").is_symlink()
    assert (nested / "atlas" / "cells.txt").is_symlink()
    with pytest.raises(DataError, match="nope"):
        data_sources(["nope"], cache)
    # A link through another link would write into the cache itself.
    with pytest.raises(DataError, match="inside another linked entry"):
        link_data(work, [("atlas/extra.txt", cache / "pbmc.h5ad")])
    assert not (cache / "atlas" / "extra.txt").exists()


def test_the_claude_guard_admits_linked_data_only_through_its_read_root(tmp_path: Path) -> None:
    """The guard resolves symlinks, so a link is judged by where it points: the cache entry."""
    cache = _cache(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    sources = data_sources(["pbmc.h5ad"], cache)
    link_data(work, sources)
    linked = {"file_path": str(work / "pbmc.h5ad")}

    assert find_escape(linked, [work], cwd=work) == str(work / "pbmc.h5ad")
    assert find_escape(linked, [work, *(source for _, source in sources)], cwd=work) is None
    # Naming one entry does not open the rest of the cache.
    other = {"file_path": str(cache / "atlas" / "cells.txt")}
    assert find_escape(other, [work, *(source for _, source in sources)], cwd=work) is not None


# --- delivery into runs ----------------------------------------------------------------


def test_run_once_links_the_split_data_and_allows_reading_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)
    seen: dict = {}
    box = Sandbox(
        root=tmp_path / "box", home=tmp_path / "home", config_dir=tmp_path / "cfg", env={}, authenticated=True
    )

    @asynccontextmanager
    async def fake_sandbox(*_args, **kwargs):
        seen["data"] = kwargs["data"]
        yield box

    async def fake_agent(prompt, *, options) -> AgentResult:
        seen["prompt"] = prompt
        seen["read_dirs"] = options.read_dirs
        return AgentResult(
            provider="claude",
            is_error=False,
            subtype="success",
            errors=[],
            session_id=None,
            result="",
            num_turns=1,
            total_cost_usd=0.0,
            duration_ms=10,
            usage={},
        )

    monkeypatch.setattr("acumen.runner.sandbox", fake_sandbox)
    monkeypatch.setattr("acumen.runner.run_agent", fake_agent)
    monkeypatch.setattr("acumen.runner._collect_artifacts", lambda *_args: None)
    monkeypatch.setattr("acumen.runner.agent_version", lambda *_args: "test")
    task = Task(
        id="t",
        train=TaskSplit("p", "a"),
        valid=TaskSplit("p", "b"),
        test=TaskSplit("p", "c", data=("atlas",)),
        data=("pbmc.h5ad",),
    )
    target = _target(tmp_path)

    asyncio.run(
        run_once(
            key=RunKey(arm="noskill", split="test", model="claude-haiku-4-5-20251001", task_id="t", rep=1),
            task=task,
            target=target,
            run_dir=tmp_path / "run",
            model="claude-haiku-4-5-20251001",
            max_turns=1,
            max_usd=1.0,
            data_root=cache,
        )
    )

    pbmc, atlas = (cache / "pbmc.h5ad").resolve(), (cache / "atlas").resolve()
    assert seen["data"] == [("pbmc.h5ad", pbmc), ("atlas", atlas)]
    assert seen["read_dirs"] == (target.venv_dir, pbmc, atlas)
    # The task's entries and the split's own are named together, and only as being there.
    assert (
        "- Already in your working directory (read-only; leave them in place): `pbmc.h5ad`, `atlas`.\n"
        in seen["prompt"]
    )


def test_a_prompt_without_data_is_unchanged() -> None:
    """Tasks that name no data must get exactly the preamble they got before the cache existed."""
    plain = benchmark_prompt("Do it.", sandbox=Path("/box"), python=Path("/py"), package="pkg")

    assert "Already in your working directory" not in plain
    assert "Use it if it helps.\n\n# What you must leave behind" in plain
    with_data = benchmark_prompt(
        "Do it.", sandbox=Path("/box"), python=Path("/py"), package="pkg", data=["a.h5ad", "net"]
    )
    assert "helps.\n- Already in your working directory (read-only; leave them in place): `a.h5ad`, `net`.\n\n#" in (
        with_data
    )


def test_check_links_the_same_data_into_each_reproducer(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    reproducer = 'from pathlib import Path\nPath("answer.md").write_text(Path("{}").read_text())\n'
    (scripts / "t-train.py").write_text(reproducer.format("pbmc.h5ad"))
    (scripts / "t-valid.py").write_text(reproducer.format("atlas/cells.txt"))
    task = Task(
        id="t",
        train=TaskSplit("p", "PBMC"),
        valid=TaskSplit("p", "CELLS", data=("atlas",)),
        test=TaskSplit("p", "c", data=("missing.csv",)),
        data=("pbmc.h5ad",),
    )
    (scripts / "t-test.py").write_text("pass\n")

    results = check_tasks(
        [task], scripts_root=scripts, python=_fake_venv(tmp_path) / "bin" / "python", timeout=30, data_root=cache
    )

    assert [(r.split, r.status) for r in results] == [("train", "ok"), ("valid", "ok"), ("test", "error")]
    assert "missing.csv" in results[2].error_tail


# --- acumen prepare --------------------------------------------------------------------

_WRITES_FILE = """\
import os, sys
from pathlib import Path
assert Path.cwd() == Path(sys.argv[1]) == Path(os.environ["ACUMEN_DATA_CACHE"])
Path("pbmc.h5ad").write_text("PBMC")
"""

_WRITES_DIR = """\
from pathlib import Path
Path("atlas").mkdir(exist_ok=True)
(Path("atlas") / "cells.txt").write_text(Path("pbmc.h5ad").read_text() + " CELLS")
"""


def test_prepare_builds_once_and_leaves_the_cache_read_only(tmp_path: Path) -> None:
    root = tmp_path / "prepare"
    root.mkdir()
    (root / "a_pbmc.py").write_text(_WRITES_FILE)
    (root / "b_atlas.py").write_text(_WRITES_DIR)  # filename order: it reads a_pbmc's output
    python = _fake_venv(tmp_path) / "bin" / "python"
    cache = tmp_path / "cache"

    first = run_prepare(prepare_scripts(root), python=python, cache=cache)

    assert [(r.name, r.status) for r in first] == [("a_pbmc", "ok"), ("b_atlas", "ok")], first
    assert (cache / "atlas" / "cells.txt").read_text() == "PBMC CELLS"
    assert (cache / MARKER_DIRNAME / "a_pbmc.json").is_file()
    assert not any(_writable(path) for path in [cache, *cache.rglob("*")])

    # Unchanged scripts are skipped; an edited one, or --force, runs again.
    assert {r.status for r in run_prepare(prepare_scripts(root), python=python, cache=cache)} == {"skipped"}
    unfreeze(root.parent / "cache")
    (root / "b_atlas.py").write_text(_WRITES_DIR + "# edited\n")
    again = run_prepare(prepare_scripts(root), python=python, cache=cache)
    assert [r.status for r in again] == ["skipped", "ok"]
    forced = run_prepare(prepare_scripts(root, ["a_pbmc"]), python=python, cache=cache, force=True)
    assert [(r.name, r.status) for r in forced] == [("a_pbmc", "ok")]
    unfreeze(cache)  # so pytest can clean the tmp dir


def test_prepare_records_no_marker_for_a_failing_script(tmp_path: Path) -> None:
    root = tmp_path / "prepare"
    root.mkdir()
    (root / "broken.py").write_text('import sys\nprint("download failed", file=sys.stderr)\nraise SystemExit(4)\n')
    cache = tmp_path / "cache"

    (result,) = run_prepare(prepare_scripts(root), python=_fake_venv(tmp_path) / "bin" / "python", cache=cache)

    assert (result.status, result.returncode) == ("error", 4)
    assert "download failed" in result.error_tail
    assert not (cache / MARKER_DIRNAME / "broken.json").exists()
    unfreeze(cache)


def test_prepare_scripts_rejects_an_unknown_name(tmp_path: Path) -> None:
    (tmp_path / "pbmc.py").write_text("pass\n")

    with pytest.raises(PrepareError, match="unknown prepare scripts"):
        prepare_scripts(tmp_path, ["pbcm"])
    with pytest.raises(PrepareError, match="no prepare scripts directory"):
        prepare_scripts(tmp_path / "nope")


# --- CLI -------------------------------------------------------------------------------


def test_bench_refuses_tasks_whose_data_is_missing_before_any_setup(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    text = (
        (project / "tasks.yaml")
        .read_text()
        .replace("  - id: example_task\n", "  - id: example_task\n    data: [pbmc.h5ad]\n")
    )
    (project / "tasks.yaml").write_text(text)

    def unreachable(*_args, **_kwargs):
        raise AssertionError("bench prepared the target for runs that cannot start")

    monkeypatch.setattr("acumen.cli.prepare_target", unreachable)
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", unreachable)
    argv = ["bench", "--config", str(project / "config.yaml"), "--tasks", str(project / "tasks.yaml")]
    argv += ["--runs", str(project / "runs"), "--no-skill", "--data-cache", str(project / "cache")]

    assert main(argv) == 2
    assert "pbmc.h5ad — run `acumen prepare`" in capsys.readouterr().err


def test_prepare_command_runs_the_scripts_and_reports_missing_data(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    (project / "prepare").mkdir()
    (project / "prepare" / "pbmc.py").write_text(_WRITES_FILE)
    text = (
        (project / "tasks.yaml")
        .read_text()
        .replace("  - id: example_task\n", "  - id: example_task\n    data: [pbmc.h5ad, gone.csv]\n")
    )
    (project / "tasks.yaml").write_text(text)
    target = _target(project, _fake_venv(project))
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_args, **_kwargs: target)
    argv = ["prepare", "--config", str(project / "config.yaml"), "--tasks", str(project / "tasks.yaml")]
    argv += ["--scripts", str(project / "prepare"), "--data-cache", str(project / "cache")]

    assert main(argv) == 0

    captured = capsys.readouterr()
    assert "pbmc  ok" in captured.out
    assert "gone.csv" in captured.err
    assert (project / "cache" / "pbmc.h5ad").read_text() == "PBMC"
    unfreeze(project / "cache")
