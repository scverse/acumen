"""Smoke tests for the ``acumen`` CLI — one per command.

These check that each command is wired up and does the right thing on the happy path
(or refuses cleanly), not that every flag works. Commands that spawn an agent are
covered only up to their pre-flight checks, which is everything that happens before a
single token is spent.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

from acumen.agents import AgentError
from acumen.bench import BenchmarkInvalidError
from acumen.cli import _agent_prices, _print_run_summary, _Progress, build_parser, main
from acumen.config import load_config
from acumen.env import Target
from acumen.paths import RunKey
from acumen.pricefeed import PriceFeedError
from acumen.prices import Rates
from acumen.review import ReviewError, ReviewResult, ReviewVerdict
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


@pytest.fixture
def stub_prices(monkeypatch: pytest.MonkeyPatch) -> dict[str, Rates]:
    """Stand in for the live pricing pages, which ``bench`` reads before every pass.

    Returns the fetched table so a test can assert on what the pass was priced by, or
    mutate it to model a price change between passes.
    """
    fetched = {"claude-haiku-4-5-20251001": Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=5.0)}
    monkeypatch.setattr("acumen.pricefeed.refresh", lambda **_kwargs: fetched)
    return fetched


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


def test_console_costs_are_the_figure_the_report_plots(capsys: pytest.CaptureFixture[str], model: str) -> None:
    """The per-run line and the pass tally must not disagree with the report.

    A Claude run's SDK figure can dwarf what its own usage block prices — nested subagents —
    so a console reading that figure would tally a pass at one number and report it at another.
    """
    outcome = RunOutcome(
        key=RunKey(arm="noskill", split="test", model=model, task_id="task", rep=1),
        success=True,
        reason="ok",
        payload={
            "input_tokens": 10,
            "output_tokens": 2,
            "cost_usd": 0.20,
            "cost_available": True,
            "provider_cost_usd": 0.90,
            "inferred_cost_usd": 0.20,
            "duration_s": 0.1,
        },
    )
    progress = _Progress(1)
    progress.running = 1
    progress.on_done(outcome)
    _print_run_summary([outcome], 1.0)

    out = capsys.readouterr().out
    assert "$0.20" in out
    assert "$0.90" not in out


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
    project: Path,
    skills_root: Path,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    stub_prices,
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
    project: Path, skills_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, stub_prices
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
    project: Path, model: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, stub_prices
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


def test_prices_reads_the_live_pages_because_nothing_ships_with_the_package(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no offline view to show: rates exist only on the providers' pages."""
    fetched = {
        "claude-haiku-4-5-20251001": Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=5.0),
        "gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0),
    }
    monkeypatch.setattr("acumen.cli.refresh", lambda **_kwargs: fetched)

    assert main(["prices", "--config", str(project / "config.yaml")]) == 0

    out = capsys.readouterr().out
    assert "claude-haiku-4-5-20251001" in out and "(fetched)" in out
    assert date.today().isoformat() in out


def test_prices_exits_nonzero_when_the_pages_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no shipped table there is nothing to print, so this is a failure, not a warning."""
    monkeypatch.setattr(
        "acumen.cli.refresh",
        lambda **_kwargs: (_ for _ in ()).throw(PriceFeedError("could not fetch: timed out")),
    )

    assert main(["prices", "--config", str(tmp_path / "absent.yaml")]) == 2
    assert "could not fetch" in capsys.readouterr().err


def test_prices_refresh_flags_a_pin_that_has_drifted_from_the_published_price(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pinned rates are the only ones that can go stale, so they are what --refresh checks."""
    (project / "config.yaml").write_text(
        (project / "config.yaml").read_text() + "prices:\n  gpt-5.6-sol:\n    input: 5.0\n    output: 30.0\n"
    )
    published = {"gpt-5.6-sol": Rates(input=8.0, cached_input=0.8, cache_write=10.0, output=48.0)}
    monkeypatch.setattr("acumen.cli.refresh", lambda **_kwargs: published)

    assert main(["prices", "--refresh", "--config", str(project / "config.yaml")]) == 0

    out = capsys.readouterr().out
    assert "1 pinned rate(s) differ" in out
    assert "input $5.0→$8.0" in out


def test_prices_refresh_says_nothing_can_drift_without_pins(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unpinned rates are read live each run, so there is no drift to report."""
    monkeypatch.setattr("acumen.cli.refresh", lambda **_kwargs: {})

    assert main(["prices", "--refresh", "--config", str(project / "config.yaml")]) == 0
    assert "nothing can drift" in capsys.readouterr().out


def _stub_bench(project: Path, monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    """Stand in for everything ``bench`` does after it has resolved its rates."""

    async def fake_matrix(todo, **kwargs) -> list[RunOutcome]:
        seen.append(kwargs["prices"])
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
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", lambda *_a, **_k: "session")
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_a, **_k: None)
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_a, **_k: target)
    monkeypatch.setattr("acumen.cli.run_matrix", fake_matrix)


def test_bench_prices_its_runs_from_the_live_pages(
    project: Path, skills_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass takes its rates from what the providers publish on the day it runs.

    Nothing ships with the package to fall back on, and the figure produced is frozen into
    every result, so this fetch is the only thing standing between the report and a cost
    column that means nothing.
    """
    fetched = {"claude-haiku-4-5-20251001": Rates(input=9.0, cached_input=0.9, cache_write=11.25, output=45.0)}
    monkeypatch.setattr("acumen.pricefeed.refresh", lambda **_kwargs: fetched)
    seen: list = []
    _stub_bench(project, monkeypatch, seen)

    assert main(bench_args(project)) == 0

    assert seen and all(table is seen[0] for table in seen), "every arm must price by one table"
    lookup = seen[0].lookup("claude-haiku-4-5-20251001")
    assert lookup.rates.input == 9.0
    assert lookup.source == "fetched"
    assert lookup.as_of == date.today().isoformat()
    assert "prices: 1 model(s) resolved" in capsys.readouterr().out


def test_bench_refuses_to_run_when_the_pricing_pages_are_unreachable(
    project: Path, skills_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreachable feed stops the pass before it spends anything.

    There is no table to fall back to, so continuing would store a run with no cost at all
    while still charging for it — the benchmark would spend real money to produce a report
    missing one of its headline columns.
    """

    def unreachable(**_kwargs):
        raise PriceFeedError("could not fetch https://example.invalid: timed out")

    monkeypatch.setattr("acumen.pricefeed.refresh", unreachable)
    seen: list = []
    _stub_bench(project, monkeypatch, seen)

    assert main(bench_args(project)) == 2
    assert seen == [], "no run may start once pricing is known to be unresolvable"
    err = capsys.readouterr().err
    assert "could not fetch" in err
    assert "'prices:' block in config.yaml" in err, "the error must name the way forward"


def test_bench_warns_when_a_benched_model_is_not_published(
    project: Path, skills_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A successful fetch that does not cover your model still leaves it unpriced.

    The pass runs — tokens are still recorded — but it must say so before the spend rather
    than let a blank cost column read as a free model afterwards.
    """
    monkeypatch.setattr("acumen.pricefeed.refresh", lambda **_kwargs: {"some-other-model": _RATE})
    seen: list = []
    _stub_bench(project, monkeypatch, seen)

    assert main(bench_args(project)) == 0

    assert seen[0].lookup("claude-haiku-4-5-20251001") is None
    err = capsys.readouterr().err
    assert "no token rates for claude-haiku-4-5-20251001" in err
    assert "record tokens but no cost" in err


_RATE = Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=5.0)


def test_agent_commands_keep_working_when_pricing_is_unavailable(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``draft`` and friends report cost as progress, so an outage must not stop the work.

    It is still announced, and for Codex the real consequence is named: ``max_usd`` is
    derived from these rates, so an unpriced model has no enforceable budget cap.
    """
    monkeypatch.setattr(
        "acumen.pricefeed.refresh",
        lambda **_kwargs: (_ for _ in ()).throw(PriceFeedError("could not fetch: timed out")),
    )
    cfg = load_config(project / "config.yaml")

    table = _agent_prices(cfg, model="gpt-5.6-sol")

    assert table.lookup("gpt-5.6-sol") is None, "unpriced, rather than guessed from a shipped default"
    err = capsys.readouterr().err
    assert "reports no cost" in err
    assert "max_usd cannot be enforced" in err, "the Codex budget-cap consequence must be named"


def _stub_target(project: Path, monkeypatch: pytest.MonkeyPatch, *, pkg_name: str = "pyyaml") -> Target:
    """Point `acumen check` at a venv-shaped directory holding the test interpreter.

    ``pkg_name`` is the distribution the import probe looks for: ``pyyaml`` is installed here
    and imports as ``yaml``, so it also covers the name mismatch. A name nothing provides
    models a target whose install went wrong.
    """
    venv = project / "venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True, exist_ok=True)
    interpreter = bin_dir / ("python.exe" if os.name == "nt" else "python")
    if not interpreter.exists():
        interpreter.symlink_to(sys.executable)
    target = Target(
        source="target",
        ref="main",
        src_dir=project / "target",
        venv_dir=venv,
        commit="abc1234def",
        pkg_name=pkg_name,
        pkg_version="1.0",
    )
    monkeypatch.setattr("acumen.cli.prepare_target", lambda *_args, **_kwargs: target)
    return target


def check_args(project: Path, *extra: str) -> list[str]:
    """The paths every ``acumen check`` invocation in these tests needs.

    ``--log-dir`` is pinned to the scratch project: the review phase opens a real
    :class:`~acumen.logs.LiveLog` before the agent is stubbed out, and its default is ``logs/``
    relative to the *cwd* — so leaving it out has the suite writing agent logs into whatever
    directory pytest was started from.
    """
    return [
        "check",
        "--config",
        str(project / "config.yaml"),
        "--tasks",
        str(project / "tasks.yaml"),
        "--scripts",
        str(project / "tasks"),
        "--log-dir",
        str(project / "logs"),
        *extra,
    ]


def _write_reproducer(project: Path, split: str, answer: str) -> None:
    scripts = project / "tasks"
    scripts.mkdir(exist_ok=True)
    (scripts / f"example_task-{split}.py").write_text(
        f'from pathlib import Path\n\nPath("answer.md").write_text({answer!r})\n'
    )


def _squash(text: str) -> str:
    """Collapse the table's column padding, so a test asserts content and not alignment."""
    return " ".join(text.split())


def test_check_passes_when_every_answer_reproduces(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _write_reproducer(project, "test", "TEST_ANSWER")

    assert main(check_args(project, "--no-review", "--jobs", "1")) == 0

    out = capsys.readouterr().out
    assert "imports as yaml" in out, "the probe reports the module it actually imported"
    assert "reproduced 2/2 (100%)" in _squash(out)
    assert "tasks fully reproduced 1/1 (100%)" in _squash(out)
    assert "every task's ground truth reproduced" in out


def test_check_exits_nonzero_and_names_what_is_wrong(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd answer and a missing reproducer are different repairs, so both are named."""
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "SOMETHING_ELSE")

    assert main(check_args(project, "--no-review", "--jobs", "1")) == 1

    captured = capsys.readouterr()
    assert "got SOMETHING_ELSE / want TRAIN_ANSWER" in captured.out
    assert "no reproducer at" in captured.out
    squashed = _squash(captured.out)
    assert "reproduced 0/2 (0%)" in squashed
    assert "wrong_answer 1" in squashed and "missing 1" in squashed
    assert "fix the reproducers and answers above" in captured.err


def test_check_refuses_to_run_anything_when_the_package_will_not_import(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One broken install would otherwise be reported once per task, at full runtime cost."""
    _stub_target(project, monkeypatch, pkg_name="never-installed-anywhere")
    _write_reproducer(project, "train", "TRAIN_ANSWER")

    assert main(check_args(project, "--no-review")) == 2

    err = capsys.readouterr().err
    assert "does not import in the target venv" in err
    assert "Every task would fail the same way, so nothing was run" in err


def test_check_filters_to_one_cell_and_writes_json(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    report = project / "check.json"

    assert (
        main(check_args(project, "--no-review", "--task", "example_task", "--split", "train", "--json", str(report)))
        == 0
    )

    payload = json.loads(report.read_text())
    assert [(r["task_id"], r["split"], r["status"]) for r in payload["results"]] == [("example_task", "train", "ok")]
    assert payload["summary"]["ok"] is True
    assert payload["target"]["commit"] == "abc1234def"
    # The held-out split was not asked for, so its absent reproducer is not a failure here.
    assert "no reproducer" not in capsys.readouterr().out


def test_check_rejects_an_unknown_task_before_preparing_a_target(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _unreachable(*_args, **_kwargs):
        raise AssertionError("a typo'd --task must be caught before the costly target prep")

    monkeypatch.setattr("acumen.cli.prepare_target", _unreachable)

    assert main(check_args(project, "--no-review", "--task", "typo")) == 2
    assert "unknown task ids: ['typo']" in capsys.readouterr().err


def test_check_warns_about_a_reproducer_no_task_claims(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _write_reproducer(project, "test", "TEST_ANSWER")
    (project / "tasks" / "renamed_away-train.py").write_text("pass\n")

    assert main(check_args(project, "--no-review", "--jobs", "1")) == 0

    assert "renamed_away-train.py matches no task and split" in capsys.readouterr().err


def _stub_review(monkeypatch: pytest.MonkeyPatch, verdicts: dict[tuple[str, str], tuple[str, str | None]]):
    """Stand in for the review agent, returning a fixed verdict per (task, split)."""
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_a, **_k: None)
    monkeypatch.setattr("acumen.cli.resolve_auth_mode", lambda *_a, **_k: "session")
    monkeypatch.setattr("acumen.cli._agent_prices", lambda *_a, **_k: None)

    async def fake_review(**kwargs):
        built = {}
        for result in kwargs["results"]:
            key = (result.task_id, result.split)
            status, issue = verdicts.get(key, ("ok", None))
            built[key] = ReviewVerdict(
                task_id=result.task_id,
                split=result.split,
                status=status,
                issue=issue,
                fix="say descending in the prompt" if status == "mismatch" else None,
            )
        return ReviewResult(verdicts=built, cost_usd=0.31, turns=9)

    monkeypatch.setattr("acumen.cli.review_tasks", fake_review)


def test_check_no_review_spawns_nothing_and_never_reaches_a_credential(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deterministic phase must stay free, and usable in a loop while fixing a script."""
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _write_reproducer(project, "test", "TEST_ANSWER")
    monkeypatch.setattr("acumen.cli.check_agent_cli", _boom)
    monkeypatch.setattr("acumen.cli.review_tasks", _boom)

    assert main(check_args(project, "--no-review", "--jobs", "1")) == 0

    out = capsys.readouterr().out
    assert "auth:" not in out, "no credential is resolved when no agent runs"
    assert "reviewing" not in out
    assert "reviewed" not in out, "the summary has no review line to report"
    assert "every task's ground truth reproduced" in out


def test_check_flags_a_coherent_looking_task_whose_prompt_asks_for_something_else(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this phase exists for: the code reproduces, but the prompt disagrees with it.

    The deterministic column says ok, so nothing else in acumen would ever notice.
    """
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _write_reproducer(project, "test", "TEST_ANSWER")
    _stub_review(
        monkeypatch,
        {("example_task", "test"): ("mismatch", "prompt says ascending; script and answer are descending")},
    )

    assert main(check_args(project, "--jobs", "1")) == 1

    captured = capsys.readouterr()
    squashed = _squash(captured.out)
    assert "reproduced 2/2 (100%)" in squashed, "the deterministic phase saw nothing wrong"
    assert "reviewed 2/2 (100%)" in squashed
    assert "mismatch 1" in squashed
    # The reason and the fix are both shown, and briefly.
    assert "prompt says ascending; script and answer are descending" in captured.out
    assert "fix: say descending in the prompt" in captured.out
    assert "review flagged" in captured.err or "review flagged" in captured.out
    assert "$0.31 over 9 turns" in captured.out


def test_check_reports_a_failed_review_as_an_incomplete_check(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A review that could not run must never read as a green check, but must not lose the runs."""
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _write_reproducer(project, "test", "TEST_ANSWER")
    _stub_review(monkeypatch, {})

    async def die(**_kwargs):
        raise ReviewError("the review agent did not write review.json")

    monkeypatch.setattr("acumen.cli.review_tasks", die)

    assert main(check_args(project, "--jobs", "1")) == 2

    captured = capsys.readouterr()
    assert "reproduced 2/2 (100%)" in _squash(captured.out), "the deterministic results survive"
    assert "did not write review.json" in captured.err
    assert "not a clean check" in captured.err


def test_check_needs_the_agent_cli_before_it_prepares_a_target(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Half an hour of analyses must not be spent to discover the reviewer cannot run."""
    monkeypatch.setattr("acumen.cli.prepare_target", _boom)
    monkeypatch.setattr("acumen.cli.check_agent_cli", lambda *_a, **_k: (_ for _ in ()).throw(AgentError("no cli")))

    assert main(check_args(project)) == 2


def test_check_json_carries_the_review_verdicts(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _stub_review(monkeypatch, {("example_task", "train"): ("mismatch", "asks for a count, answer is a name")})
    report = project / "check.json"

    assert main(check_args(project, "--task", "example_task", "--split", "train", "--json", str(report))) == 1

    payload = json.loads(report.read_text())
    (row,) = payload["results"]
    assert row["review"] == {
        "verdict": "mismatch",
        "issue": "asks for a count, answer is a name",
        "fix": "say descending in the prompt",
    }
    assert payload["review"]["flagged"] == 1
    assert payload["review"]["turns"] == 9
    capsys.readouterr()


def test_check_writes_its_review_log_where_it_was_told_to(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--log-dir is honoured, and nothing lands in the cwd.

    The review phase opens its log before the agent runs, so a default-valued --log-dir writes
    into whatever directory the command was started from. Pinning it here is what keeps the
    suite from dropping agent logs into the repository.
    """
    _stub_target(project, monkeypatch)
    _write_reproducer(project, "train", "TRAIN_ANSWER")
    _stub_review(monkeypatch, {})
    before = set(Path.cwd().iterdir())

    assert main(check_args(project, "--task", "example_task", "--split", "train")) == 0

    logs = sorted((project / "logs").glob("acumen-check-*.jsonl"))
    assert len(logs) == 1, "the review phase writes exactly one live log"
    assert set(Path.cwd().iterdir()) == before, "the command wrote into the cwd"
    capsys.readouterr()
