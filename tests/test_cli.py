"""Smoke tests for the ``acumen`` CLI — one per command.

These check that each command is wired up and does the right thing on the happy path
(or refuses cleanly), not that every flag works. Commands that spawn an agent are
covered only up to their pre-flight checks, which is everything that happens before a
single token is spent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from acumen.bench import BenchmarkInvalidError
from acumen.cli import _Progress, build_parser, main
from acumen.config import load_config
from acumen.env import Target
from acumen.paths import RunKey
from acumen.prices import RATES_AS_OF
from acumen.runner import RunOutcome
from acumen.tasks import load_tasks


def bench_args(project: Path, *extra: str) -> list[str]:
    """The paths every ``acumen bench`` invocation in these tests needs."""
    return [
        "bench",
        "--config",
        str(project / "config.yaml"),
        "--tasks",
        str(project / "tasks.yaml"),
        "--runs",
        str(project / "runs"),
        "--skills",
        str(project / "skills"),
        *extra,
    ]


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_progress_prints_unavailable_cost_without_casting_null(capsys: pytest.CaptureFixture[str], model: str) -> None:
    progress = _Progress(1)
    progress.running = 1
    progress.on_done(
        RunOutcome(
            key=RunKey(arm="noskill", split="test", model=model, task_id="task", rep=1),
            success=True,
            reason="ok",
            payload={
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": None,
                "cost_available": False,
                "duration_s": 0.1,
            },
        )
    )

    assert "cost n/a" in capsys.readouterr().out


def test_progress_prints_provider_exhaustion_as_invalid(capsys: pytest.CaptureFixture[str], model: str) -> None:
    progress = _Progress(1)
    progress.running = 1
    progress.on_done(
        RunOutcome(
            key=RunKey(arm="noskill", split="test", model=model, task_id="task", rep=1),
            success=False,
            reason="provider_exhausted",
            payload={
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": None,
                "cost_available": False,
                "duration_s": 0.1,
                "error": "usage limit reached",
            },
        )
    )

    captured = capsys.readouterr()
    assert "INVALID" in captured.out and "FAIL" not in captured.out
    assert "usage limit reached" in captured.err


def test_init_writes_files_the_loaders_accept(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["init", "--dir", str(tmp_path)]) == 0

    config, tasks = tmp_path / "config.yaml", tmp_path / "tasks.yaml"
    assert config.is_file() and tasks.is_file()
    # The scaffold is a placeholder, but it must still be a *valid* placeholder.
    loaded = load_config(config)
    assert loaded.repo.startswith("https://")
    assert loaded.models == [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ]
    assert [t.id for t in load_tasks(tasks)] == ["example_task"]
    assert "wrote" in capsys.readouterr().out


def test_init_refuses_to_clobber_without_force(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["init", "--dir", str(tmp_path)]) == 0
    (tmp_path / "tasks.yaml").write_text("tasks: []\n")

    assert main(["init", "--dir", str(tmp_path)]) == 2
    assert (tmp_path / "tasks.yaml").read_text() == "tasks: []\n"
    assert "already exist" in capsys.readouterr().err

    assert main(["init", "--dir", str(tmp_path), "--force"]) == 0
    assert "example_task" in (tmp_path / "tasks.yaml").read_text()


def test_tasks_generates_over_the_untouched_scaffold_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`acumen init` then `acumen tasks` is the documented start of the loop.

    The placeholder `init` writes is not user content, so refusing to generate over it made
    the first two steps of the quickstart contradict each other.
    """
    assert main(["init", "--dir", str(tmp_path)]) == 0
    out = tmp_path / "tasks.yaml"
    args = ["tasks", "--config", str(tmp_path / "config.yaml"), "--out", str(out)]

    # Stop at the first step past the clobber check, so no target is prepared and nothing spent.
    monkeypatch.setattr("acumen.cli.check_agent_cli", _boom)
    with pytest.raises(RuntimeError, match="reached preflight"):
        main(args)
    assert "replacing the untouched placeholder" in capsys.readouterr().out

    # An edited file is the user's, and is still protected.
    out.write_text(out.read_text() + "\n# mine now\n")
    assert main(args) == 2
    assert "pass --force to overwrite" in capsys.readouterr().err

    # …until --force, which reaches preflight again.
    with pytest.raises(RuntimeError, match="reached preflight"):
        main([*args, "--force"])


def _boom(*_: object, **__: object) -> None:
    raise RuntimeError("reached preflight")


def test_bench_dry_run_plans_the_matrix(project: Path, model: str, capsys: pytest.CaptureFixture) -> None:
    assert main(bench_args(project, "--dry-run")) == 0

    out = capsys.readouterr().out
    # 1 task x 2 splits x 1 model x 1 replicate, and nothing was executed.
    assert "2 runs planned" in out
    assert f"noskill/train/{model}/example_task/rep_1" in out
    assert not (project / "runs").exists()


def test_bench_dry_run_skips_completed_runs(
    project: Path, model: str, make_result, capsys: pytest.CaptureFixture
) -> None:
    make_result(
        project / "runs",
        RunKey(arm="noskill", split="train", model=model, task_id="example_task", rep=1),
    )
    assert main(bench_args(project, "--dry-run")) == 0

    out = capsys.readouterr().out
    assert "1 already complete, 1 to run" in out


def test_bench_dry_run_plans_every_arm_when_none_is_selected(
    project: Path, skills_root: Path, model: str, capsys: pytest.CaptureFixture
) -> None:
    assert main(bench_args(project, "--dry-run")) == 0

    out = capsys.readouterr().out
    # The baseline and every version on disk, each with its own counts, plus the run list.
    assert "arm noskill:" in out
    assert "arm skill_v1:" in out
    assert "skill v1: target" in out
    assert "total: 4 runs planned, 4 to run across 2 arms" in out
    assert f"noskill/train/{model}/example_task/rep_1" in out
    assert f"skill_v1/train/{model}/example_task/rep_1" in out


def test_bench_dry_run_stays_on_one_arm_when_the_baseline_is_explicit(
    project: Path, skills_root: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(bench_args(project, "--no-skill", "--dry-run")) == 0

    out = capsys.readouterr().out
    assert "arm noskill:" in out
    assert "skill_v1" not in out
    assert "total:" not in out


def test_bench_refuses_a_version_that_will_not_load_before_spending(
    project: Path, skills_root: Path, capsys: pytest.CaptureFixture
) -> None:
    # A broken version in skills/ is an arm that cannot run, so the whole pass stops at
    # planning rather than half-running and failing once money is on the line.
    (skills_root / "v2").mkdir()
    (skills_root / "v2" / "SKILL.md").write_text("no frontmatter here\n")
    assert main(bench_args(project, "--dry-run")) == 2
    assert "missing a non-empty 'name'" in capsys.readouterr().err

    # The healthy arms are still benchable by naming them.
    assert main(bench_args(project, "--skill", "v1", "--dry-run")) == 0


def test_bench_with_a_skill_loads_that_version(project: Path, skills_root: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(bench_args(project, "--skill", "v1", "--dry-run")) == 0

    out = capsys.readouterr().out
    assert "arm skill_v1" in out
    assert "skill v1: target" in out


def test_bench_reports_a_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["bench", "--config", str(tmp_path / "nope.yaml"), "--dry-run"]) == 2
    assert "error:" in capsys.readouterr().err


def test_bench_runs_every_arm_when_none_is_selected(
    project: Path, skills_root: Path, model: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: list[tuple[str | None, int]] = []

    async def fake_matrix(todo, **kwargs) -> list[RunOutcome]:
        skill = kwargs["skill"]
        seen.append((None if skill is None else skill.version, len(todo)))
        return [
            RunOutcome(key=item.key, success=True, reason="ok", payload={"cost_usd": 0.5, "skill_loaded": True})
            for item in todo
        ]

    target = Target(
        source="target",
        ref="main",
        src_dir=project / "src",
        venv_dir=project / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", lambda *_args, **_kwargs: "session")
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr("acumen.cli.run_matrix", fake_matrix)

    assert main(bench_args(project)) == 0

    # One matrix per arm, each carrying its own skill: the baseline, then v1.
    assert seen == [(None, 2), ("v1", 2)]
    out = capsys.readouterr().out
    assert "total: 4 runs planned, 4 to run across 2 arms" in out
    assert "=== arm skill_v1: 2 runs ===" in out
    assert "all 2 arms: 4/4 passed" in out


def test_bench_runs_only_the_named_arm(
    project: Path, skills_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: list[str | None] = []

    async def fake_matrix(todo, **kwargs) -> list[RunOutcome]:
        skill = kwargs["skill"]
        seen.append(None if skill is None else skill.version)
        return [RunOutcome(key=item.key, success=True, reason="ok", payload={}) for item in todo]

    target = Target(
        source="target",
        ref="main",
        src_dir=project / "src",
        venv_dir=project / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", lambda *_args, **_kwargs: "session")
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr("acumen.cli.run_matrix", fake_matrix)

    assert main(bench_args(project, "--no-skill")) == 0

    assert seen == [None]
    out = capsys.readouterr().out
    assert "skill_v1" not in out
    assert "total:" not in out


def test_bench_exits_nonzero_and_prints_provider_exhaustion(
    project: Path, model: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    key = RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1)
    outcome = RunOutcome(
        key=key,
        success=False,
        reason="provider_exhausted",
        payload={"agent": "claude", "auth_mode": "session", "error": "You've hit your usage limit"},
    )

    async def invalid_matrix(*_args: object, **_kwargs: object) -> list[RunOutcome]:
        raise BenchmarkInvalidError(outcome)

    target = Target(
        source="target",
        ref="main",
        src_dir=project / "src",
        venv_dir=project / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", lambda *_args, **_kwargs: "session")
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_args, **_kwargs: target)
    monkeypatch.setattr("acumen.cli.run_matrix", invalid_matrix)

    assert main(bench_args(project)) == 2
    err = capsys.readouterr().err
    assert "benchmark invalid" in err
    assert "You've hit your usage limit" in err
    assert "invalid and cancelled cells remain pending" in err


def test_report_writes_html_and_csv(project: Path, runs_root: Path, capsys: pytest.CaptureFixture) -> None:
    out_path = project / "report.html"
    exit_code = main(
        [
            "report",
            "--runs",
            str(runs_root),
            "--tasks",
            str(project / "tasks.yaml"),
            "--skills",
            str(project / "skills"),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    assert out_path.is_file()
    assert out_path.with_suffix(".csv").is_file()

    html = out_path.read_text()
    assert "noskill" in html
    # Both overview figures and the significance section are present, and each is reachable
    # from the contents.
    assert 'id="tradeoff"' in html and 'href="#tradeoff"' in html
    assert 'id="dominance"' in html and 'href="#dominance"' in html
    # Self-contained: figures are inlined, so nothing is fetched from the network.
    assert "http://" not in html and "https://" not in html
    assert "aggregated 2 runs" in capsys.readouterr().out


def test_report_csv_records_skill_loaded_as_a_plain_bool(
    project: Path, runs_root: Path, model: str, make_result
) -> None:
    """Undetermined (a transcript that could not be read) counts as not loaded in the CSV."""
    key = RunKey(arm="skill_v1", split="train", model=model, task_id="example_task", rep=1)
    make_result(runs_root, key, skill_loaded=None)
    out_path = project / "report.html"
    assert main(["report", "--runs", str(runs_root), "--out", str(out_path)]) == 0

    with out_path.with_suffix(".csv").open(newline="") as handle:
        loaded = [row["skill_loaded"] for row in csv.DictReader(handle)]
    assert set(loaded) == {"False"}  # two baseline runs and one undetermined skill run


def test_report_palette_recolours_and_validates(
    project: Path, runs_root: Path, model: str, capsys: pytest.CaptureFixture
) -> None:
    """--palette is accepted end to end; a key naming no benchmarked model is an error."""
    out_path = project / "report.html"
    argv = ["report", "--runs", str(runs_root), "--out", str(out_path)]

    assert main([*argv, "--palette", f"{model}=#3b7ea1"]) == 0
    assert out_path.is_file()

    assert main([*argv, "--palette", "gpt-4=#3b7ea1"]) == 2
    assert "no such model" in capsys.readouterr().err


def test_report_without_runs_errors(project: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["report", "--runs", str(project / "runs"), "--out", str(project / "report.html")]) == 2
    assert "error:" in capsys.readouterr().err


def test_draft_refuses_when_versions_exist(project: Path, skills_root: Path, capsys: pytest.CaptureFixture) -> None:
    """The guard fires before the target is prepared, so no agent runs."""
    exit_code = main(["draft", "--config", str(project / "config.yaml"), "--skills", str(skills_root)])

    assert exit_code == 2
    assert "skills already exist (v1)" in capsys.readouterr().err
    assert not (skills_root / "v2").exists()


def test_improve_without_a_skill_errors(project: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = main(
        [
            "improve",
            "--config",
            str(project / "config.yaml"),
            "--tasks",
            str(project / "tasks.yaml"),
            "--skills",
            str(project / "skills"),
            "--runs",
            str(project / "runs"),
        ]
    )

    assert exit_code == 2
    assert "no skill versions" in capsys.readouterr().err


def test_tasks_refuses_to_overwrite_without_force(project: Path, capsys: pytest.CaptureFixture) -> None:
    before = (project / "tasks.yaml").read_text()
    exit_code = main(["tasks", "--config", str(project / "config.yaml"), "--out", str(project / "tasks.yaml")])

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err
    assert (project / "tasks.yaml").read_text() == before


def test_ship_rejects_an_unknown_version(project: Path, skills_root: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["ship", "--skill", "v9", "--config", str(project / "config.yaml"), "--skills", str(skills_root)])

    assert exit_code == 2
    assert "no such skill version" in capsys.readouterr().err


def test_prices_lists_the_rate_table_without_touching_the_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`acumen prices` is the offline view; only --refresh reaches for the network."""
    assert main(["prices", "--config", str(tmp_path / "absent.yaml")]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5" in out and "gpt-5.6-sol" in out
    assert RATES_AS_OF in out
