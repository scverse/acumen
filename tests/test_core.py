"""Unit tests for the pure parts the CLI sits on: grader, schemas, paths, skills.

Deliberately thin — one test per behaviour that would silently corrupt a benchmark if it
broke, not an exhaustive sweep of each validator.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

import acumen
from acumen.agents import (
    AgentError,
    AgentOptions,
    AgentResult,
    _codex_terminal,
    _install_codex_guard,
    check_agent_cli,
    provider_for_model,
    run_agent,
)
from acumen.bench import build_matrix, pending
from acumen.config import Config, ConfigError, derive_skill_name, load_config, parse_config
from acumen.env import (
    AUTH_ENV_VARS,
    EnvError,
    Target,
    _validate_deps,
    api_auth_available,
    build_agent_env,
    cache_key,
    resolve_auth_mode,
    scrubbed_env,
    session_auth_available,
)
from acumen.grade import grade_answer, grade_run
from acumen.improve import _write_material, collect_train_runs, load_rates
from acumen.logs import LiveLog
from acumen.paths import RunKey, arm_name, is_complete, parse_run_dir, run_dir
from acumen.pricefeed import PriceFeedError, diff_rates, parse_anthropic, parse_openai
from acumen.prices import DEFAULT_RATES, Rates, Usage, normalize_usage, price_run, price_usage, pricer, resolve_rates
from acumen.procs import label_env, reap, supported, survivors
from acumen.prompts import draft_prompt, feedback_block, improve_prompt
from acumen.report import (
    ReportError,
    _arm_marker,
    _best_cells,
    _holm,
    _integrity_notes,
    _loaded_rank,
    _pareto_front,
    _pareto_steps,
    _runs_table_html,
    _skill_diff_html,
    _split_diff_rows,
    _tests_table_html,
    arm_metrics,
    load_results,
    loaded_only_rates,
    metrics_figure,
    resolve_palette,
    skill_tests,
    tradeoff_figure,
)
from acumen.runner import StderrFilter, _skill_fired, _terminal_reason, run_once
from acumen.sandbox import Sandbox
from acumen.ship import _ship_env
from acumen.skills import SkillError, load_skill, read_meta, skill_hash, write_meta
from acumen.tasks import Task, TaskError, TaskSplit, load_tasks, parse_tasks
from acumen.transcript import render_agent_transcript, render_codex_transcript

# --- grading ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected", "success", "reason"),
    [
        ("SPI1", "SPI1", True, "ok"),
        ("  SPI1\n", "SPI1", True, "ok"),  # graded after strip()
        ("STAT1", "SPI1", False, "wrong_answer"),
        ("**SPI1**", "SPI1", False, "format_error"),  # right content, forbidden formatting
        (None, "SPI1", False, "no_answer_file"),
    ],
)
def test_grade_answer(answer: str | None, expected: str, success: bool, reason: str) -> None:
    grade = grade_answer(answer, expected)
    assert (grade.success, grade.reason) == (success, reason)


def test_grade_run_reads_answer_md(tmp_path: Path) -> None:
    (tmp_path / "answer.md").write_text("SPI1\n")
    assert grade_run(tmp_path, "SPI1").success
    assert grade_run(tmp_path / "empty", "SPI1").reason == "no_answer_file"


# --- run paths -------------------------------------------------------------------------


def test_run_dir_round_trips(tmp_path: Path) -> None:
    key = RunKey(arm="skill_v2", split="test", model="claude-opus-5", task_id="tf_activity", rep=3)
    directory = run_dir(tmp_path, key)

    assert directory == tmp_path / "skill_v2/test/claude-opus-5/tf_activity/rep_3"
    assert parse_run_dir(tmp_path, directory) == key
    assert key.skill == "v2"


def test_arm_name() -> None:
    assert arm_name(None) == "noskill"
    assert arm_name("v1") == "skill_v1"


def test_is_complete_needs_a_result_file(tmp_path: Path) -> None:
    assert not is_complete(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    assert is_complete(tmp_path)


# --- config / tasks --------------------------------------------------------------------


def test_config_defaults_and_derived_skill_name() -> None:
    cfg = parse_config({"repo": "https://github.com/scverse/scanpy"})

    assert cfg.skill_name == "scanpy"
    assert cfg.ref == "main"
    assert cfg.n_replicates == 3
    assert cfg.env_passthrough == []
    assert not cfg.is_local
    assert derive_skill_name("git@github.com:scverse/scanpy.git") == "scanpy"

    cfg2 = parse_config({"repo": "https://github.com/scverse/scanpy", "env_passthrough": ["OMP_NUM_THREADS"]})
    assert cfg2.env_passthrough == ["OMP_NUM_THREADS"]


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config({"repo": "https://example.com/pkg", "modles": ["x"]})


def test_config_dependency_selection() -> None:
    cfg = parse_config({"repo": "https://example.com/pkg"})
    assert cfg.dependency_groups == []
    assert cfg.pip_packages == []

    cfg2 = parse_config({"repo": "https://example.com/pkg", "dependency_groups": ["full"], "pip_packages": ["numpy<2"]})
    assert cfg2.dependency_groups == ["full"]
    assert cfg2.pip_packages == ["numpy<2"]

    with pytest.raises(ConfigError, match="list of non-empty strings"):
        parse_config({"repo": "https://example.com/pkg", "dependency_groups": "full"})


# --- target dependency selection -------------------------------------------------------


def _target(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text(f'[project]\nname = "pkg"\n{body}')
    return src


def _cfg(**kwargs: list[str]) -> Config:
    return parse_config({"repo": "https://example.com/pkg", **kwargs})


def test_validate_deps_points_at_the_right_key(tmp_path: Path) -> None:
    """A group asked for as an extra is the failure that silently empties the venv."""
    src = _target(tmp_path, '\n[dependency-groups]\nfull = ["scanpy"]\n')

    _validate_deps(src, _cfg(dependency_groups=["full"]))  # the correct spelling passes

    with pytest.raises(EnvError, match=r"'full' is a dependency group, not an extra"):
        _validate_deps(src, _cfg(extras=["full"]))


def test_validate_deps_reports_what_is_declared(tmp_path: Path) -> None:
    src = _target(tmp_path, '\n[project.optional-dependencies]\ntest = ["pytest"]\n')

    with pytest.raises(EnvError, match="the target declares no dependency groups"):
        _validate_deps(src, _cfg(dependency_groups=["nope"]))
    with pytest.raises(EnvError, match="available extras: test"):
        _validate_deps(src, _cfg(extras=["nope"]))
    with pytest.raises(EnvError, match=r"'test' is an extra, not a dependency group"):
        _validate_deps(src, _cfg(dependency_groups=["test"]))


def test_validate_deps_ignores_pip_packages(tmp_path: Path) -> None:
    """Arbitrary package names can't be checked against the source tree; uv rejects bad ones."""
    src = _target(tmp_path, "")

    _validate_deps(src, _cfg(pip_packages=["harmonypy", "numpy<2"]))


def test_cache_key_tracks_the_dependency_selection() -> None:
    base = cache_key("https://example.com/pkg", "main")

    assert cache_key("https://example.com/pkg", "main", extras=["test"]) != base
    assert cache_key("https://example.com/pkg", "main", dependency_groups=["test"]) != base
    assert cache_key("https://example.com/pkg", "main", pip_packages=["test"]) != base
    # ... and the three are distinct from each other, not just from the base
    assert (
        len(
            {
                cache_key("https://example.com/pkg", "main", extras=["x"]),
                cache_key("https://example.com/pkg", "main", dependency_groups=["x"]),
                cache_key("https://example.com/pkg", "main", pip_packages=["x"]),
            }
        )
        == 3
    )


def test_cache_key_is_stable_under_reordering() -> None:
    assert cache_key("https://example.com/pkg", "main", extras=["a", "b"]) == cache_key(
        "https://example.com/pkg", "main", extras=["b", "a"]
    )


def test_load_config_resolves_a_local_repo(project: Path) -> None:
    cfg = load_config(project / "config.yaml")

    assert cfg.is_local
    assert Path(cfg.repo) == (project / "target").resolve()


def test_load_tasks(project: Path) -> None:
    (task,) = load_tasks(project / "tasks.yaml")

    assert task.id == "example_task"
    assert task.split("train").answer == "TRAIN_ANSWER"
    assert task.split("test").prompt.startswith("Do the same")


def test_tasks_reject_duplicate_ids() -> None:
    entry = {"id": "dup", "train": {"prompt": "p", "answer": "a"}, "test": {"prompt": "p", "answer": "a"}}
    with pytest.raises(TaskError, match="duplicate task id"):
        parse_tasks({"tasks": [entry, dict(entry)]})


# --- matrix ----------------------------------------------------------------------------


def test_build_matrix_and_resume(project: Path, model: str, make_result) -> None:
    cfg = load_config(project / "config.yaml")
    tasks = load_tasks(project / "tasks.yaml")

    planned = build_matrix(cfg, tasks, skill="v1")
    assert len(planned) == 2  # 1 task x 2 splits x 1 model x 1 replicate
    assert {p.key.split for p in planned} == {"train", "test"}
    assert all(p.key.arm == "skill_v1" for p in planned)

    runs = project / "runs"
    make_result(runs, RunKey(arm="skill_v1", split="train", model=model, task_id="example_task", rep=1))
    assert [p.key.split for p in pending(planned, runs)] == ["test"]
    assert len(pending(planned, runs, resume=False)) == 2


def test_skill_fired_matches_the_skill_under_test_only(tmp_path: Path) -> None:
    def transcript(*skills: str) -> Path:
        path = tmp_path / f"{'-'.join(skills) or 'none'}.jsonl"
        records = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Skill", "input": {"skill": name}}]},
            }
            for name in skills
        ]
        path.write_text("".join(f"{json.dumps(r)}\n" for r in records))
        return path

    assert _skill_fired(transcript("target"), "target") is True
    # Skills bundled with the CLI are reachable in either arm and are not what is measured.
    assert _skill_fired(transcript("dataviz", "init"), "target") is False
    assert _skill_fired(transcript("dataviz", "target"), "target") is True
    assert _skill_fired(transcript(), "target") is False
    # A missing transcript is unknown, not a miss.
    assert _skill_fired(tmp_path / "absent.jsonl", "target") is None


def test_codex_skill_fired_matches_project_skill_read(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "sed -n '1,200p' .agents/skills/target/SKILL.md",
                },
            }
        )
        + "\n"
    )
    assert _skill_fired(path, "target", provider="codex") is True
    assert _skill_fired(path, "another", provider="codex") is False


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-opus-5", "claude"),
        ("anthropic/claude-sonnet-5", "claude"),
        ("gpt-5.6-sol", "codex"),
        ("openai/gpt-5.6-terra", "codex"),
        ("o4-mini", "codex"),
    ],
)
def test_provider_for_model(model: str, provider: str) -> None:
    assert provider_for_model(model) == provider


def test_provider_for_model_rejects_ambiguous_ids() -> None:
    with pytest.raises(AgentError, match="cannot infer"):
        provider_for_model("custom-model")


def test_codex_terminal_normalizes_jsonl() -> None:
    result = _codex_terminal(
        [
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "done"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 60,
                    "output_tokens": 20,
                },
            },
        ],
        0,
        1234,
    )
    assert result.provider == "codex"
    assert result.session_id == "thread-1"
    assert result.result == "done"
    assert result.num_turns == 1
    assert result.usage["input_tokens"] == 100
    assert result.total_cost_usd is None
    assert not result.is_error


def test_codex_adapter_runs_jsonl_cli(tmp_path: Path) -> None:
    fake = tmp_path / "codex"
    fake.write_text(
        """#!/bin/sh
printf '%s\\n' \\
  '{"type":"thread.started","thread_id":"thread-1"}' \\
  '{"type":"turn.started"}' \\
  '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}' \\
  '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}'
"""
    )
    fake.chmod(0o755)
    seen: list[dict] = []
    result = asyncio.run(
        run_agent(
            "write the answer",
            options=AgentOptions(
                cwd=tmp_path,
                env={"PATH": f"{tmp_path}:/usr/bin", "HOME": str(tmp_path)},
                model="gpt-5.6-sol",
            ),
            on_event=seen.append,
        )
    )
    assert result.result == "done"
    assert result.usage == {"input_tokens": 12, "output_tokens": 3}
    assert len(seen) == 4


def test_codex_guard_denies_isolated_paths(tmp_path: Path) -> None:
    denied = tmp_path / "runs"
    denied.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    codex_home = tmp_path / "codex-home"
    options = AgentOptions(
        cwd=work,
        env={"CODEX_HOME": str(codex_home)},
        model="gpt-5.6-sol",
        deny_paths=(denied,),
    )
    _install_codex_guard(options)

    hooks = json.loads((work / ".codex" / "hooks.json").read_text())
    assert "acumen_guard.py" in hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    payload = json.dumps(
        {
            "cwd": str(work),
            "tool_name": "Bash",
            "tool_input": {"command": f"cat {denied / 'skill_v1/test/result.json'}"},
        }
    )
    proc = subprocess.run(
        [sys.executable, str(codex_home / "acumen_guard.py")],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    decision = json.loads(proc.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def _fake_codex(path: Path, events: list[dict]) -> Path:
    """Write a stub ``codex`` CLI that replays ``events`` as JSONL and exits 0."""
    lines = "".join(f"  '{json.dumps(event)}' \\\n" for event in events).rstrip(" \\\n")
    cli = path / "codex"
    cli.write_text(f"#!/bin/sh\nprintf '%s\\n' \\\n{lines}\n")
    cli.chmod(0o755)
    return cli


def _codex_options(tmp_path: Path, **kwargs: object) -> AgentOptions:
    return AgentOptions(
        cwd=tmp_path,
        env={"PATH": f"{tmp_path}:/usr/bin", "HOME": str(tmp_path)},
        model="gpt-5.6-sol",
        **kwargs,  # type: ignore[arg-type]
    )


def test_codex_turns_count_model_actions_not_exec_invocations() -> None:
    """`codex exec` is one turn however much work happens inside it.

    Counting ``turn.started`` would record 1 for every Codex run, which makes the turns column
    meaningless and leaves ``max_turns`` with nothing to bite on. Completed items are the unit.
    """
    result = _codex_terminal(
        [
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "echo one"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "echo two"}},
            {"type": "item.completed", "item": {"type": "todo_list", "items": []}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1}},
        ],
        0,
        10,
    )
    # Two commands and the message; reasoning and bookkeeping are not actions.
    assert result.num_turns == 3


def test_codex_stops_at_the_turn_cap(tmp_path: Path) -> None:
    _fake_codex(
        tmp_path,
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "a", "type": "command_execution", "command": "echo one"}},
            {"type": "item.completed", "item": {"id": "b", "type": "command_execution", "command": "echo two"}},
            {"type": "item.completed", "item": {"id": "c", "type": "agent_message", "text": "never reached"}},
            {"type": "turn.completed", "usage": {"input_tokens": 999}},
        ],
    )
    result = asyncio.run(run_agent("go", options=_codex_options(tmp_path, max_turns=2)))

    assert result.is_error
    assert result.subtype == "error_max_turns"
    assert result.num_turns == 2
    # The agent is stopped at the cap, so the events after it are never recorded…
    assert result.result == ""
    # …and the run is a cap breach, not a crashed CLI.
    assert result.errors == ["acumen stopped the run at its turn cap"]


def test_codex_stops_at_the_budget_cap(tmp_path: Path) -> None:
    _fake_codex(
        tmp_path,
        [
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1_000_000, "output_tokens": 0}},
        ],
    )
    # gpt-5.6-sol bills $5/M input, so a million fresh input tokens is $5 against a $1 cap.
    options = _codex_options(tmp_path, max_usd=1.0, price_usd=pricer("gpt-5.6-sol"))
    result = asyncio.run(run_agent("go", options=options))

    assert result.is_error
    assert result.subtype == "error_max_budget_usd"


def test_codex_budget_cap_is_inert_without_a_rate_for_the_model(tmp_path: Path) -> None:
    """An unpriced model must not be capped as if it were free."""
    _fake_codex(
        tmp_path,
        [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1_000_000}},
        ],
    )
    options = _codex_options(tmp_path, max_usd=0.000_001, price_usd=pricer("gpt-unpriced-model"))
    result = asyncio.run(run_agent("go", options=options))

    assert not result.is_error
    assert result.result == "done"


@pytest.mark.parametrize(
    ("subtype", "reason"),
    [("error_max_turns", "max_turns"), ("error_max_budget_usd", "budget")],
)
def test_codex_cap_subtypes_map_onto_the_same_reasons_as_claude(subtype: str, reason: str) -> None:
    """A cap breach must record the same reason whichever provider hit it."""
    result = _codex_terminal([], -15, 10, subtype)
    assert _terminal_reason(result) == reason


def test_codex_transcript_renders_its_own_html(tmp_path: Path) -> None:
    """`claude-code-log` reads the SDK format only — given Codex events it silently exits 0
    and writes an empty page, so Codex is rendered here instead."""
    jsonl = tmp_path / "transcript.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "thread.started", "thread_id": "thread-9"},
                {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "<b>hi</b>"}},
                {
                    "type": "item.completed",
                    "item": {
                        "id": "b",
                        "type": "command_execution",
                        "command": "python -c 'print(1)'",
                        "aggregated_output": "1\n",
                        "exit_code": 0,
                    },
                },
                # Started but never completed — what a turn-capped run leaves behind.
                {"type": "item.started", "item": {"id": "c", "type": "command_execution", "command": "sleep 60"}},
                {"type": "turn.completed", "usage": {"input_tokens": 41_214, "output_tokens": 122}},
            ]
        )
        + "\n"
    )
    html = tmp_path / "transcript.html"
    assert render_codex_transcript(jsonl, html) is True

    body = html.read_text()
    assert "thread-9" in body
    assert "python -c &#x27;print(1)&#x27;" in body
    assert "exit 0" in body
    # Agent text is escaped, never injected as markup.
    assert "&lt;b&gt;hi&lt;/b&gt;" in body and "<b>hi</b>" not in body
    # The unfinished item survives, flagged as such.
    assert "sleep 60" in body and "unfinished" in body
    assert "41214" in body


def test_codex_transcript_renders_without_events(tmp_path: Path) -> None:
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("not json\n")
    html = tmp_path / "empty.html"
    assert render_codex_transcript(jsonl, html) is True
    assert "no events" in html.read_text()
    assert render_codex_transcript(tmp_path / "missing.jsonl", tmp_path / "missing.html") is False


def test_render_agent_transcript_dispatches_on_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}) + "\n")
    calls: list[str] = []
    monkeypatch.setattr("acumen.transcript.render_transcript", lambda *_: calls.append("claude") or True)

    assert render_agent_transcript(jsonl, tmp_path / "codex.html", provider="codex") is True
    assert calls == []
    assert render_agent_transcript(jsonl, tmp_path / "claude.html", provider="claude") is True
    assert calls == ["claude"]


def test_backends_are_optional_at_import_time() -> None:
    """A Codex-only install has no Claude SDK, so nothing may import it at module scope.

    This is the invariant that keeps ``import acumen`` working with either backend installed
    alone; the SDK may only be reached inside a function or under ``TYPE_CHECKING``.
    """
    offenders: list[str] = []
    for path in sorted(Path(acumen.__file__).parent.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
                continue  # annotations only — never executed
            for child in ast.walk(node) if isinstance(node, ast.If | ast.Try) else [node]:
                module = getattr(child, "module", None) if isinstance(child, ast.ImportFrom) else None
                names = [alias.name for alias in getattr(child, "names", [])] if isinstance(child, ast.Import) else []
                if (module or "").startswith("claude_agent_sdk") or any(
                    name.startswith("claude_agent_sdk") for name in names
                ):
                    offenders.append(f"{path.name}:{child.lineno}")
    assert offenders == []


def test_missing_backend_fails_before_any_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both backends are optional, so both are preflighted the same way."""
    monkeypatch.setattr("acumen.agents.claude_sdk_available", lambda: False)
    with pytest.raises(AgentError, match=r"acumen\[claude\]"):
        check_agent_cli("claude")

    monkeypatch.setattr("acumen.agents.claude_sdk_available", lambda: True)
    check_agent_cli("claude")

    monkeypatch.setattr("acumen.agents.shutil.which", lambda _: None)
    with pytest.raises(AgentError, match="codex is not on PATH"):
        check_agent_cli("codex")


def test_live_log_records_codex_without_the_claude_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live JSONL is provider-neutral: Codex events are dicts, recognized without the SDK."""
    monkeypatch.setattr("acumen.logs._sdk", lambda: None)
    with LiveLog(tmp_path / "log.jsonl") as log:
        log.append({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
        log.append(object())  # an SDK message on an install that cannot recognize it
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["assistant"]
    assert events[0]["text"] == "done"


def test_live_log_streams_codex_terminal_events_without_missing_field_crashes(tmp_path: Path) -> None:
    lines: list[str] = []
    with LiveLog(tmp_path / "log.jsonl", stream=True, echo=lines.append) as log:
        log.append(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 12, "cached_input_tokens": 3, "output_tokens": 4},
            }
        )
        log.append({"type": "turn.failed", "error": {"message": "boom"}})

    assert lines == [
        "● done: turn.completed · 1 turns · cost n/a",
        "✗ error: turn.failed · cost n/a",
    ]


def test_usage_normalizes_both_providers_without_collapsing_the_cache_split() -> None:
    """The cache classes are priced up to 10x apart, so they must survive normalization.

    The two providers report the same information differently: Codex's ``input_tokens``
    is the total with ``cached_input_tokens`` inside it, while the Claude SDK reports the
    three classes side by side. Both normalize to a total ``input`` plus its parts.
    """
    claude = normalize_usage(
        {
            "input_tokens": 5_000,
            "cache_read_input_tokens": 180_000,
            "cache_creation_input_tokens": 15_000,
            "output_tokens": 4_000,
        }
    )
    assert (claude.input, claude.cache_read, claude.cache_write, claude.output) == (200_000, 180_000, 15_000, 4_000)
    assert claude.fresh_input == 5_000

    # Captured verbatim from a live `codex exec --json` turn.completed event.
    codex = normalize_usage(
        {
            "input_tokens": 12_051,
            "cached_input_tokens": 8_960,
            "cache_write_input_tokens": 0,
            "output_tokens": 5,
            "reasoning_output_tokens": 0,
        },
        provider="codex",
    )
    assert (codex.input, codex.cache_read, codex.cache_write, codex.output) == (12_051, 8_960, 0, 5)
    # Cached input is a subset of input_tokens, not an addition to it.
    assert codex.fresh_input == 3_091
    # reasoning_output_tokens is a subset of output_tokens, so it must not be added on.
    assert codex.total == 12_056

    written = normalize_usage(
        {"input_tokens": 12_051, "cached_input_tokens": 8_000, "cache_write_input_tokens": 960, "output_tokens": 5},
        provider="codex",
    )
    assert (written.cache_write, written.fresh_input) == (960, 3_091)

    empty = normalize_usage(None)
    assert (empty.input, empty.output, empty.total) == (0, 0, 0)


def test_price_run_bills_each_cache_class_at_its_own_rate() -> None:
    """Collapsing the split and billing it all at the base input rate overcharges ~3.6x.

    That is the whole reason the breakdown is carried: a benchmark agent's input is
    mostly cache reads, billed at a tenth of the base rate.
    """
    usage = Usage(input=200_000, cache_read=180_000, cache_write=15_000, output=4_000)
    rates = resolve_rates("claude-opus-5")
    assert rates is not None
    # 5k @ $5 + 180k @ $0.50 + 15k @ $6.25 + 4k @ $25, per million.
    assert price_run(usage, rates) == pytest.approx(0.30875)
    collapsed = (usage.input * rates.input + usage.output * rates.output) / 1_000_000
    assert collapsed > price_run(usage, rates) * 3


def test_price_usage_prices_codex_meta_agent_tokens_when_provider_cost_is_absent() -> None:
    cost = price_usage(
        {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000},
        model="gpt-5.6-sol",
        provider="codex",
    )

    assert cost == pytest.approx(35.0)


def test_price_run_leaves_an_unpriced_model_unpriced_rather_than_free() -> None:
    """``None`` is not ``0.0`` — a model with no rates must not read as a free one."""
    assert resolve_rates("some-local-llm") is None
    assert price_run(Usage(input=1, cache_read=0, cache_write=0, output=1), None) is None


def test_benchmark_persists_unavailable_cost_as_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box_root = tmp_path / "box"
    box_root.mkdir()
    box = Sandbox(
        root=box_root,
        home=tmp_path / "home",
        config_dir=tmp_path / "codex-home",
        env={},
        authenticated=True,
        provider="codex",
    )

    @contextmanager
    def fake_sandbox(*_args, **_kwargs):
        yield box

    async def fake_run_agent(*_args, **_kwargs) -> AgentResult:
        return AgentResult(
            provider="codex",
            is_error=False,
            subtype="success",
            errors=None,
            session_id="thread-1",
            result="done",
            num_turns=1,
            total_cost_usd=None,
            duration_ms=100,
            usage={"input_tokens": 10, "output_tokens": 2},
        )

    def collect(_box: Sandbox, directory: Path) -> None:
        (directory / "answer.md").write_text("OK")

    monkeypatch.setattr("acumen.runner.sandbox", fake_sandbox)
    monkeypatch.setattr("acumen.runner.run_agent", fake_run_agent)
    monkeypatch.setattr("acumen.runner._collect_artifacts", collect)
    monkeypatch.setattr("acumen.runner.render_agent_transcript", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("acumen.runner.agent_version", lambda _provider: "test")

    run_dir = tmp_path / "run"
    outcome = asyncio.run(
        run_once(
            key=RunKey(arm="noskill", split="test", model="gpt-unpriced", task_id="task", rep=1),
            task=Task(id="task", train=TaskSplit("prompt", "OK"), test=TaskSplit("prompt", "OK")),
            target=Target(
                source="target",
                ref="main",
                src_dir=tmp_path / "src",
                venv_dir=tmp_path / "venv",
                commit="abc",
                pkg_name="target",
                pkg_version="1",
            ),
            run_dir=run_dir,
            model="gpt-unpriced",
            max_turns=1,
            max_usd=1.0,
        )
    )

    persisted = json.loads((run_dir / "result.json").read_text())
    assert outcome.payload["cost_usd"] is None
    assert persisted["cost_usd"] is None
    assert persisted["cost_available"] is False


def test_resolve_rates_accepts_provider_qualified_ids_and_config_overrides() -> None:
    assert resolve_rates("openai/gpt-5.6-sol") == resolve_rates("gpt-5.6-sol")
    assert resolve_rates("Claude-Opus-5") == resolve_rates("claude-opus-5")

    overrides = {"claude-opus-5": Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=2.0)}
    assert resolve_rates("claude-opus-5", overrides).input == 1.0
    # An override for one model doesn't shadow the built-in table for the others.
    assert resolve_rates("gpt-5.6-luna", overrides) == DEFAULT_RATES["gpt-5.6-luna"]


def test_config_prices_block_validates_and_defaults_the_cache_rates() -> None:
    cfg = parse_config(
        {
            "repo": "https://github.com/o/r",
            "prices": {"my-gateway-model": {"input": 4.0, "output": 20.0}},
        }
    )
    rates = cfg.prices["my-gateway-model"]
    # Cache rates are optional; they default to the standard 0.1x / 1.25x of input.
    assert (rates.input, rates.output, rates.cached_input, rates.cache_write) == (4.0, 20.0, 0.4, 5.0)

    for bad in ({"input": 4.0}, {"input": -1.0, "output": 2.0}, {"input": 1.0, "output": 2.0, "nope": 3.0}, []):
        with pytest.raises(ConfigError, match="prices"):
            parse_config({"repo": "https://github.com/o/r", "prices": {"m": bad}})


_ANTHROPIC_MD = """
| Model | Base Input Tokens | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output Tokens |
|---|---|---|---|---|---|
| Claude Opus 5 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Sonnet 5 [through August 31, 2026](/docs/pricing#intro) | $2 / MTok | $2.50 / MTok | $4 / MTok | $0.20 / MTok | $10 / MTok |
| Claude Sonnet 5 starting September 1, 2026 | $3 / MTok | $3.75 / MTok | $6 / MTok | $0.30 / MTok | $15 / MTok |
| Claude Haiku 4.5 | $1 / MTok | $1.25 / MTok | $2 / MTok | $0.10 / MTok | $5 / MTok |

| Model | Batch input | Batch output |
|---|---|---|
| Claude Opus 5 | $2.50 / MTok | $12.50 / MTok |
"""

_OPENAI_MD = """
### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
|---|---|---|---|---|---|---|---|---|
| gpt-5.6-sol | $5.00 | $0.50 | $6.25 | $30.00 | $10.00 | $1.00 | $12.50 | $45.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |
| gpt-5.5 (<272K context length) | $5.00 | $0.50 | - | $30.00 | $10.00 | $1.00 | - | $45.00 |

### Batch pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
|---|---|---|---|---|---|---|---|---|
| gpt-5.6-sol | $2.50 | $0.25 | $3.125 | $15.00 | $5.00 | $0.50 | $6.25 | $22.50 |
"""


def test_price_feed_picks_the_rate_in_effect_on_a_dated_row() -> None:
    """A promotional rate is published as two rows; taking the first would misprice runs."""
    intro = parse_anthropic(_ANTHROPIC_MD, today=date(2026, 8, 3))
    assert intro["claude-sonnet-5"] == Rates(input=2.0, cached_input=0.2, cache_write=2.5, output=10.0)

    after = parse_anthropic(_ANTHROPIC_MD, today=date(2026, 9, 1))
    assert after["claude-sonnet-5"] == Rates(input=3.0, cached_input=0.3, cache_write=3.75, output=15.0)

    # Display names become API model IDs, and the batch table is not mistaken for the base one.
    assert intro["claude-haiku-4-5"].input == 1.0
    assert intro["claude-opus-5"].output == 25.0


def test_price_feed_reads_standard_short_context_and_ignores_the_other_tiers() -> None:
    """Batch is half price and long context roughly double — picking either misprices runs."""
    rates = parse_openai(_OPENAI_MD)
    assert rates["gpt-5.6-sol"] == Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)
    assert rates["gpt-5.6-luna"].output == 1.2
    # A row whose cache columns are "-" falls back to the standard multiples of input.
    assert rates["gpt-5.5"] == Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)


def test_price_feed_reports_a_changed_cache_rate_not_just_input_and_output() -> None:
    """On a cached workload the cache rate is most of the bill, so a diff must surface it."""
    current = {"claude-opus-5": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=25.0)}
    same = diff_rates(current, {"claude-opus-5": current["claude-opus-5"]})
    assert same == []

    cheaper_cache = Rates(input=5.0, cached_input=0.25, cache_write=6.25, output=25.0)
    (change,) = diff_rates(current, {"claude-opus-5": cheaper_cache})
    assert change.kind == "changed"
    assert "cached_input $0.5→$0.25" in change.describe()
    assert "input $" not in change.describe().replace("cached_input $", "")


def test_price_feed_parsers_fail_loudly_on_an_unrecognized_page() -> None:
    """A silent empty parse would leave every model unpriced with no explanation."""
    for parse in (lambda md: parse_anthropic(md, today=date(2026, 8, 3)), parse_openai):
        with pytest.raises(PriceFeedError, match="layout may have changed"):
            parse("# Pricing\n\nWe have moved our prices to a new page.\n")


def test_stderr_filter_keeps_first_of_each_line() -> None:
    import io

    sink = io.StringIO()
    emit = StderrFilter(sink=sink)
    warn = "⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY is set"

    for _ in range(4):  # the same per-spawn warning fires once per run in a real pass
        emit(warn)
    emit("a distinct line")
    emit(warn)  # a later repeat is still dropped

    assert sink.getvalue() == f"{warn}\na distinct line\n"


# --- skills ----------------------------------------------------------------------------


def test_load_skill(skills_root: Path) -> None:
    skill = load_skill(skills_root, "v1", expect_name="target")

    assert (skill.version, skill.name, skill.number) == ("v1", "target", 1)
    assert skill.hash.startswith("sha256:")


def test_load_skill_rejects_a_name_mismatch(skills_root: Path) -> None:
    with pytest.raises(SkillError, match="config.skill_name"):
        load_skill(skills_root, "v1", expect_name="something_else")


def test_skill_hash_ignores_meta_json(skills_root: Path) -> None:
    """``meta.json`` carries the hash, so hashing it would be circular."""
    directory = skills_root / "v1"
    before = skill_hash(directory)

    (directory / "meta.json").write_text('{"rationale": "rewritten"}')
    assert skill_hash(directory) == before

    (directory / "SKILL.md").write_text("---\nname: target\ndescription: d\n---\nnew body\n")
    assert skill_hash(directory) != before


def test_feedback_block_absent_is_empty_and_present_is_subordinated() -> None:
    """No ``--feedback`` must leave the prompt byte-identical; present text is subordinated."""
    assert feedback_block(None) == ""
    assert feedback_block("   ") == ""

    base_kwargs = {
        "package": "p",
        "version": "1",
        "src": Path("/s"),
        "python": Path("/py"),
        "out": Path("/o"),
        "skill_name": "p",
    }
    assert draft_prompt(**base_kwargs) == draft_prompt(**base_kwargs, feedback=None)

    steered = draft_prompt(**base_kwargs, feedback="skip the plotting API")
    assert "skip the plotting API" in steered
    assert "does NOT override" in steered
    # Guidance sits after the how-to rules but before the closing deliverable reminder.
    assert steered.index("skip the plotting API") < steered.index("When you are done")


def test_write_meta_persists_feedback_but_omits_it_when_absent(tmp_path: Path) -> None:
    directory = tmp_path / "v1"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: target\ndescription: d\n---\nbody\n")

    write_meta(directory, parent=None, rationale="initial draft")
    assert "feedback" not in (directory / "meta.json").read_text()
    assert read_meta(directory).feedback is None

    write_meta(directory, parent="v1", rationale="fixed", feedback="  emphasise pseudobulk  ")
    assert read_meta(directory).feedback == "emphasise pseudobulk"


# --- improve evidence ------------------------------------------------------------------


def _train_evidence(project: Path, make_result, loaded: list[bool | None]) -> list:
    """Write one train run per entry of ``loaded``, then collect them as improver evidence."""
    runs = project / "runs"
    models = ("model_a", "model_b")
    for i, flag in enumerate(loaded):
        key = RunKey(arm="skill_v1", split="train", model=models[i % 2], task_id="example_task", rep=i + 1)
        make_result(runs, key, success=flag is True, skill_loaded=flag)
    return collect_train_runs(runs, "skill_v1", load_tasks(project / "tasks.yaml"))


def test_collect_train_runs_carries_load_status(project: Path, make_result) -> None:
    """A run's ``skill_loaded`` must reach the improver, with undetermined kept distinct."""
    runs = _train_evidence(project, make_result, [True, False, None])

    assert sorted(r.skill_loaded is True for r in runs).count(True) == 1
    assert [r.skill_loaded for r in runs].count(False) == 1
    # Unreadable transcripts stay None — undetermined is not the same as "did not load".
    assert [r.skill_loaded for r in runs].count(None) == 1


def test_load_rates_split_by_model_and_count_undetermined(project: Path, make_result) -> None:
    """Rates are per model, since the load rate varies more by model than by skill version."""
    rates = load_rates(_train_evidence(project, make_result, [True, True, False, None]))

    # model_a took reps 1 and 3 (True, False); model_b took reps 2 and 4 (True, None).
    assert rates == {"model_a": (1, 2, 0), "model_b": (1, 2, 1)}


def test_written_evidence_reports_load_rate_and_marks_each_run(project: Path, make_result, tmp_path: Path) -> None:
    """The improver reads SUMMARY.md, so the load signal has to survive into the file."""
    runs = _train_evidence(project, make_result, [True, False, None])
    train_dir = tmp_path / "train"

    _write_material(train_dir, runs)

    summary = (train_dir / "SUMMARY.md").read_text()
    assert "Did the skill load at all?" in summary
    assert "| `model_a` |" in summary and "| `model_b` |" in summary
    assert "skill LOADED" in summary
    assert "skill NOT LOADED" in summary
    assert "skill load UNDETERMINED" in summary

    # Every per-run page states it too, so a reader who opens one run isn't left guessing.
    pages = [p.read_text() for p in train_dir.rglob("run.md")]
    assert len(pages) == 3
    assert all("- Skill: skill " in page for page in pages)


def test_improve_prompt_separates_loading_from_the_body() -> None:
    """The prompt must not let a never-loaded run be read as evidence against the body."""
    prompt = improve_prompt(
        package="p",
        version="1",
        python=Path("/py"),
        skill_dir=Path("/skill"),
        train_dir=Path("/train"),
        rationale_path=Path("/r.md"),
        skill_name="p",
        parent_version="v1",
        new_version="v2",
    )

    assert "The skill never loaded" in prompt
    assert "description" in prompt
    # Raising the load rate must not become licence to name the train tasks in it.
    assert prompt.index("Do not overfit it to the train tasks") < prompt.index("When you are done")


# --- auth preflight --------------------------------------------------------------------


def _clear_auth(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    """Isolate auth detection: empty credential dir, every auth variable stripped."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    for var in AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_oauth_credentials(config_dir: Path) -> None:
    """Write a credentials file shaped like a real `claude` subscription login."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "x"}}')


def test_session_and_api_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)

    # Nothing present → neither mode is available.
    assert session_auth_available() is False
    assert api_auth_available() is False

    # An API key is API auth, not a subscription.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert api_auth_available() is True
    assert session_auth_available() is False

    # The OAuth token is a subscription credential, not API auth.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    assert session_auth_available() is True
    assert api_auth_available() is False

    # A bare credentials file with no OAuth block is not a subscription…
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    (tmp_path / ".credentials.json").write_text("{}")
    assert session_auth_available() is False
    # …but a real `claude` login (claudeAiOauth) is.
    _write_oauth_credentials(tmp_path)
    assert session_auth_available() is True


def test_resolve_auth_mode_for_meta_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)

    # No credentials at all → auto cannot resolve.
    with pytest.raises(EnvError, match="no Claude credentials"):
        resolve_auth_mode("auto")

    # auto prefers the subscription when a login exists.
    _write_oauth_credentials(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_auth_mode("auto") == "session"

    # auto falls back to the API when there is no subscription.
    (tmp_path / ".credentials.json").unlink()
    assert resolve_auth_mode("auto") == "api"

    # Forcing a mode requires that mode's credential.
    assert resolve_auth_mode("api") == "api"
    with pytest.raises(EnvError, match="--auth session"):
        resolve_auth_mode("session")
    _write_oauth_credentials(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_auth_mode("session") == "session"
    with pytest.raises(EnvError, match="--auth api"):
        resolve_auth_mode("api")


def test_bench_may_bill_the_subscription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cost comes from token counts, which both billing modes report.

    bench used to reject ``session`` on the grounds that only metered API billing yields a
    real per-run cost. Once cost became a function of tokens that stopped being true, so the
    mode is a choice the run records rather than one it refuses.
    """
    _clear_auth(monkeypatch, tmp_path)
    _write_oauth_credentials(tmp_path)

    assert resolve_auth_mode("auto") == "session"
    assert resolve_auth_mode("session") == "session"
    # Forcing the API still requires an API credential.
    with pytest.raises(EnvError, match="--auth api"):
        resolve_auth_mode("api")


def test_codex_auth_resolution_and_isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_home = tmp_path / "real-codex"
    real_home.mkdir()
    (real_home / "auth.json").write_text('{"tokens": {}}')
    monkeypatch.setenv("CODEX_HOME", str(real_home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    assert session_auth_available("codex")
    assert api_auth_available("codex")
    assert resolve_auth_mode("auto", provider="codex") == "session"
    assert resolve_auth_mode("api", provider="codex") == "api"

    isolated = tmp_path / "isolated-codex"
    env = build_agent_env(
        config_dir=isolated,
        home=tmp_path / "home",
        auth_mode="session",
        provider="codex",
    )
    assert (isolated / "auth.json").is_file()
    assert env["CODEX_HOME"] == str(isolated)
    assert env["CODEX_API_KEY"] == ""
    assert env["OPENAI_API_KEY"] == ""

    api_home = tmp_path / "api-codex"
    api_env = build_agent_env(
        config_dir=api_home,
        home=tmp_path / "api-home",
        auth_mode="api",
        provider="codex",
    )
    assert not (api_home / "auth.json").exists()
    assert api_env["CODEX_API_KEY"] == "sk-openai"


def test_codex_persisted_api_key_is_api_auth_not_a_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real-codex"
    real_home.mkdir()
    (real_home / "auth.json").write_text('{"auth_mode": "apikey", "OPENAI_API_KEY": "sk-stored"}')
    monkeypatch.setenv("CODEX_HOME", str(real_home))
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert not session_auth_available("codex")
    assert api_auth_available("codex")
    assert resolve_auth_mode("auto", provider="codex") == "api"
    assert resolve_auth_mode("api", provider="codex") == "api"
    with pytest.raises(EnvError, match="--auth session"):
        resolve_auth_mode("session", provider="codex")

    isolated = tmp_path / "isolated-codex"
    env = build_agent_env(
        config_dir=isolated,
        home=tmp_path / "home",
        auth_mode="api",
        provider="codex",
    )
    assert json.loads((isolated / "auth.json").read_text())["auth_mode"] == "apikey"
    assert env["CODEX_API_KEY"] == ""
    assert not env.get("OPENAI_API_KEY")


def test_scrubbed_env_never_carries_the_other_providers_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run authenticates with one provider, so the other's keys must not ride the allowlist in.

    ``CODEX_API_KEY``/``OPENAI_API_KEY`` and the Anthropic variables are all allowlisted, so
    without an explicit blank an operator with both configured hands every Claude agent their
    OpenAI key and every Codex agent their Anthropic key — ambient secrets in a web-enabled
    agent, and two live credentials where the module promises exactly one.
    """
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    home = tmp_path / "home"

    for mode in ("session", "api", None):
        claude = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode=mode, provider="claude")
        assert claude["OPENAI_API_KEY"] == ""
        assert claude["CODEX_API_KEY"] == ""

        codex = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode=mode, provider="codex")
        assert codex["ANTHROPIC_API_KEY"] == ""
        assert codex["CLAUDE_CODE_OAUTH_TOKEN"] == ""

    # The selected provider's own credential still survives its mode.
    api = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api", provider="codex")
    assert api["CODEX_API_KEY"] == "sk-openai"
    session = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session", provider="codex")
    assert session["CODEX_API_KEY"] == ""


def test_scrubbed_env_auth_mode_filters_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    home = tmp_path / "home"

    # The unwanted credential is set to "" (not omitted) so it overrides the inherited value
    # under the SDK's {**os.environ, **options.env} merge — see the merge regression below.
    # session keeps only the subscription token; api keeps only the API key.
    session_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session")
    assert session_env["ANTHROPIC_API_KEY"] == ""
    assert session_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"

    api_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api")
    assert api_env["ANTHROPIC_API_KEY"] == "sk-test"
    assert api_env["CLAUDE_CODE_OAUTH_TOKEN"] == ""

    # No mode leaves both credentials in place (the historical behavior).
    both = scrubbed_env(config_dir=tmp_path / "cfg", home=home)
    assert both["ANTHROPIC_API_KEY"] == "sk-test"
    assert both["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


def test_session_mode_neutralizes_the_api_key_under_the_sdk_env_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential drop must survive the SDK's env merge, not just the returned dict.

    The SDK builds the agent subprocess env as ``{**os.environ, **options.env}``, so a
    credential we *omit* from our mapping falls back through from ``os.environ`` and the run
    bills the wrong path. Setting it to "" is what actually neutralizes it. This guards the
    session-mode meta-agents (draft/improve/tasks and the unscrubbed ship env) from silently
    billing the API when ``ANTHROPIC_API_KEY`` is present in the environment.
    """
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    home = tmp_path / "home"
    target = Target(
        source="/pkg",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc123",
        pkg_name="pkg",
        pkg_version="1.0",
    )

    # scrubbed_env (draft/improve/tasks) and the unscrubbed ship env must both hold up.
    for agent_env in (
        scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session"),
        _ship_env(target, "session"),
    ):
        merged = {**os.environ, **agent_env}  # what the SDK actually hands the subprocess
        assert not merged["ANTHROPIC_API_KEY"], "API key leaked into a session-mode agent"
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


def test_scrubbed_env_blanks_ambient_vars_under_the_sdk_env_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-allowlisted ambient variable must not reach the agent through the env merge.

    The SDK builds the agent env as ``{**os.environ, **options.env}``, so a variable
    scrubbed_env merely omits falls straight through from the operator's shell into the
    web-enabled agent. scrubbed_env therefore blanks every inherited variable it did not
    keep, and the check that matters is on the *merged* mapping, not the returned dict.
    """
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # allowlisted — must survive
    monkeypatch.setenv("MY_APP_SECRET", "hunter2")  # ambient secret — must be neutralized
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # target-needed — kept only via env_passthrough
    home = tmp_path / "home"

    env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api")
    merged = {**os.environ, **env}  # what the SDK actually hands the subprocess
    assert merged["ANTHROPIC_API_KEY"] == "sk-test"  # allowlisted credential preserved
    assert not merged["MY_APP_SECRET"], "ambient secret leaked into a benchmark agent"
    assert not merged["OMP_NUM_THREADS"], "non-allowlisted var leaked without env_passthrough"
    # Our throwaway overrides still land.
    assert merged["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert merged["HOME"] == str(home)


def test_env_passthrough_carries_declared_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable named in env_passthrough survives the scrub; an undeclared one does not."""
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MY_APP_SECRET", "hunter2")
    home = tmp_path / "home"

    env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api", extra_allow=["OMP_NUM_THREADS"])
    merged = {**os.environ, **env}
    assert merged["OMP_NUM_THREADS"] == "8", "declared passthrough var was dropped"
    assert not merged["MY_APP_SECRET"], "undeclared var leaked"

    # A declared var that isn't actually set in the shell is simply absent, not blanked to "".
    env2 = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api", extra_allow=["NOT_SET_ANYWHERE"])
    assert "NOT_SET_ANYWHERE" not in env2


def test_build_agent_env_seeds_only_in_session_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _clear_auth(monkeypatch, source)
    _write_oauth_credentials(source)  # the user's real login, discovered via CLAUDE_CONFIG_DIR
    home = tmp_path / "home"

    session_cfg = tmp_path / "session_cfg"
    build_agent_env(config_dir=session_cfg, home=home, auth_mode="session")
    assert (session_cfg / ".credentials.json").is_file()  # seeded

    api_cfg = tmp_path / "api_cfg"
    build_agent_env(config_dir=api_cfg, home=home, auth_mode="api")
    assert not (api_cfg / ".credentials.json").exists()  # not seeded


def test_resolve_palette_overrides_by_id_and_by_label() -> None:
    """An override wins over the tier default, keyed by the raw id or its display form."""
    models = ["claude-opus-5", "claude-haiku-4-5-20251001"]

    colors = resolve_palette(models, {"claude-opus-5": "#3b7ea1", "claude-haiku-4-5": "rebeccapurple"})

    assert colors["claude-opus-5"] == "#3b7ea1"
    # The legend strips the snapshot date, so that form is accepted as a key too.
    assert colors["claude-haiku-4-5-20251001"] == "rebeccapurple"
    # Every model gets an entry, overridden or not.
    assert resolve_palette(models, None) == resolve_palette(models, {})
    assert set(resolve_palette(models, None)) == set(models)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"claude-opus-5": "not-a-colour"}, "not a colour"),
        ({"gpt-4": "#3b7ea1"}, "no such model"),
    ],
)
def test_resolve_palette_rejects_bad_input(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ReportError, match=message):
        resolve_palette(["claude-opus-5"], overrides)


def test_metrics_figure_paints_bars_with_the_palette(runs_root: Path, model: str) -> None:
    """The resolved colour reaches the bars themselves, not just the legend."""
    df = load_results(runs_root)
    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        faces = {to_hex(patch.get_facecolor()) for ax in figure.axes for patch in ax.patches}
        assert "#3b7ea1" in faces
    finally:
        plt.close(figure)


def test_metrics_figure_pools_every_model_into_a_last_grey_bar(runs_root: Path, model: str, make_result) -> None:
    """With several models, each cell ends in a grey bar over all of them pooled.

    It is the rate across every run, not the mean of the per-model rates — an under-sampled
    model must not weigh as much as a well-sampled one.
    """
    other = "claude-opus-5"
    key = RunKey(arm="noskill", split="test", model=other, task_id="example_task", rep=1)
    make_result(runs_root, key, success=False)
    make_result(runs_root, replace(key, rep=2), success=False)
    df = load_results(runs_root)  # the fixture's one passing test run, plus two failing ones

    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model, other]))
    try:
        (rate_ax,) = [ax for ax in figure.axes if ax.get_title() == "Success rate"]
        bars = rate_ax.patches
        widths = [bar.get_width() for bar in bars]
        pooled_color = to_hex(bars[-1].get_facecolor())
    finally:
        plt.close(figure)

    # Most potent model first, then the pooled bar last: opus 0/2, haiku 1/1, overall 1/3.
    assert widths == pytest.approx([0.0, 1.0, 1 / 3])
    assert pooled_color not in {to_hex(c) for c in resolve_palette([model, other]).values()}


def test_skill_loaded_column_counts_undetermined_runs_as_not_loaded(runs_root: Path, model: str, make_result) -> None:
    """The load-rate bar is the share of runs that loaded the skill under test.

    A run whose transcript could not be read is undetermined, not evidence of a load, so it
    counts against the rate — the same reading the runs CSV takes.
    """
    for rep, loaded in ((1, True), (2, None)):
        key = RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=rep)
        make_result(runs_root, key, skill_loaded=loaded)
    df = load_results(runs_root)

    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model]))
    try:
        # Only the top row is titled, so the title locates the column; the grid is
        # row-major with one row per arm — noskill first, then skill v1.
        columns = [i for i, ax in enumerate(figure.axes) if ax.get_title() == "Skill loaded"]
        assert len(columns) == 1
        n_cols = len(figure.axes) // 2
        widths = [figure.axes[columns[0] + row * n_cols].patches[0].get_width() for row in range(2)]
    finally:
        plt.close(figure)

    assert widths == [0.0, 0.5]  # the baseline loaded nothing; one of two skill runs is determined


@pytest.mark.parametrize(("loads", "warned"), [((True, True, False), False), ((True, False, False), True)])
def test_load_warning_needs_most_of_the_arm_to_miss(
    runs_root: Path, model: str, make_result, loads: tuple[bool, ...], warned: bool
) -> None:
    """A skill arm is only flagged once fewer than half its runs loaded the skill.

    Misses in a minority of runs are ordinary, and the per-run table already marks each one,
    so warning about them at the top of the report would cry wolf on a healthy arm.
    """
    for rep, loaded in enumerate(loads, start=1):
        key = RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=rep)
        make_result(runs_root, key, skill_loaded=loaded)

    notes = _integrity_notes(load_results(runs_root))

    assert bool(notes) is warned


def test_unpriced_runs_are_unknown_in_reports_not_zero_cost(
    runs_root: Path, model: str, make_result, tmp_path: Path
) -> None:
    key = RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1)
    make_result(runs_root, key, cost_usd=0.0, cost_available=False)

    df = load_results(runs_root)
    test = df[df["split"] == "test"]
    assert test["cost_usd"].isna().all()
    assert pd.isna(arm_metrics(test).loc[0, "cost"])
    assert any("cost unavailable" in note for note in _integrity_notes(df))
    assert "&mdash;" in _runs_table_html(test, tmp_path)

    tests = skill_tests(df)
    assert tests.cost_unavailable
    assert not tests.usable
    assert "Cost comparison unavailable" in _tests_table_html(tests)

    figure = tradeoff_figure(df)
    try:
        assert _pooled_marks(figure) == []
        assert _model_marks(figure) == []
    finally:
        plt.close(figure)


# --- the cost/success trade-off figure --------------------------------------------------

#: The two markers matplotlib gives an error bar's caps; neither is a data point.
_CAP_MARKERS = {"|", "_"}


def _pooled_marks(figure: plt.Figure) -> list[tuple[float, float]]:
    """``(cost, rate)`` of each pooled mark, in the order the arms are drawn.

    The pooled marks are the only ones carrying error bars, so they are exactly the axes'
    error-bar containers.
    """
    return [(float(c.lines[0].get_xdata()[0]), float(c.lines[0].get_ydata()[0])) for c in figure.axes[0].containers]


def _model_marks(figure: plt.Figure) -> list[plt.Line2D]:
    """Every per-model point mark — no pooled marks, no error-bar caps, no reference lines."""
    ax = figure.axes[0]
    pooled = {id(c.lines[0]) for c in ax.containers}
    return [
        line for line in ax.lines if str(line.get_marker()) not in _CAP_MARKERS | {"None"} and id(line) not in pooled
    ]


def _frontier(figure: plt.Figure) -> list[tuple[float, float]]:
    """The vertices of the drawn Pareto staircase — the only marker-less line on the axes."""
    (line,) = [ln for ln in figure.axes[0].lines if str(ln.get_marker()) == "None"]
    return [(float(x), float(y)) for x, y in zip(*line.get_data(), strict=True)]


def test_tradeoff_pooled_mark_averages_runs_not_per_model_means(runs_root: Path, model: str, make_result) -> None:
    """The pooled mark sits where an average *run* lands, whichever model drew it.

    Averaging the per-model points instead would let a model with two runs count for no more
    than one with a single run — the same trap the grid's grey bar avoids.
    """
    other = "claude-opus-5"
    key = RunKey(arm="noskill", split="test", model=other, task_id="example_task", rep=1)
    make_result(runs_root, key, success=False, cost_usd=0.30)
    make_result(runs_root, replace(key, rep=2), success=False, cost_usd=0.30)
    df = load_results(runs_root)  # the fixture's one passing $0.12 run, plus two failing $0.30 ones

    figure = tradeoff_figure(df)
    try:
        (pooled,) = _pooled_marks(figure)
    finally:
        plt.close(figure)

    # Over runs: ($0.12 + $0.30 + $0.30)/3 and 1 of 3 passing. Over per-model means it would
    # have been $0.21 and 50%.
    assert pooled == pytest.approx((0.24, 1 / 3))


def test_tradeoff_shape_carries_the_arm_and_colour_carries_the_model(runs_root: Path, model: str, make_result) -> None:
    """Two channels, two meanings: an ✕ then a widening polygon per version, hue left to the model."""
    for arm in ("skill_v1", "skill_v2"):
        make_result(runs_root, RunKey(arm=arm, split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    figure = tradeoff_figure(df, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        markers = {str(line.get_marker()) for line in _model_marks(figure)}
        colors = {to_hex(line.get_color()) for line in _model_marks(figure)}
    finally:
        plt.close(figure)

    # noskill is off the ladder; v1 and v2 are the 3- and 4-sided polygons.
    assert markers == {"X", "(3, 0, 0)", "(4, 0, 0)"}
    assert colors == {"#3b7ea1"}  # the override reaches the marks, not just the legend


def test_tradeoff_frontier_steps_between_the_marks_nothing_beats(runs_root: Path, model: str, make_result) -> None:
    """The staircase holds each rate until the next frontier point's price, then steps up.

    A dominated arm — dearer *and* worse — must leave no trace on the line, and the path must
    never cut a diagonal between two points, which would claim a result nobody measured.
    """
    other = "claude-opus-5"
    # Cheap and good, dear and bad, dear and best: only the first and last are non-dominated.
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=other, task_id="example_task", rep=1))
    make_result(
        runs_root,
        RunKey(arm="skill_v2", split="test", model=other, task_id="example_task", rep=1),
        cost_usd=0.50,
        success=False,
    )
    df = load_results(runs_root)  # plus the fixture's $0.12 passing baseline run

    figure = tradeoff_figure(df)
    try:
        vertices = _frontier(figure)
        y_min = figure.axes[0].get_ylim()[0]
    finally:
        plt.close(figure)

    # Both $0.12 runs pass, so the cheapest 100% mark alone survives; the $0.50 failure is
    # dominated and the riser starts at the axis floor.
    assert vertices == [(pytest.approx(0.12), pytest.approx(y_min)), (pytest.approx(0.12), pytest.approx(1.0))]


def test_tradeoff_handles_an_arm_where_nothing_succeeded(runs_root: Path, model: str, make_result) -> None:
    """A 0% rate is a real result, not a hole: the mark is plotted and the frontier still draws."""
    make_result(
        runs_root,
        RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1),
        success=False,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        assert _pooled_marks(figure) == [(pytest.approx(0.12), 0.0)]
        assert _frontier(figure)  # degenerate but drawn, rather than crashing on the empty case
    finally:
        plt.close(figure)


def test_tradeoff_plots_the_test_split_only(runs_root: Path, model: str, make_result) -> None:
    """Train runs feed the improver; a report measures held-out performance and must not mix them."""
    make_result(
        runs_root,
        RunKey(arm="noskill", split="train", model=model, task_id="example_task", rep=2),
        cost_usd=99.0,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        (pooled,) = _pooled_marks(figure)
    finally:
        plt.close(figure)

    assert pooled == pytest.approx((0.12, 1.0))  # the fixture's test run alone


def test_tradeoff_keeps_the_two_corner_tick_labels_apart(runs_root: Path, model: str, make_result) -> None:
    """The cost label at the origin is centred on the corner the rate floor label also sits on.

    Left to itself that puts half of one under the other, which is unreadable exactly where the
    reader looks to learn the rate axis is truncated.
    """
    make_result(
        runs_root,
        RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1),
        success=False,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        figure.canvas.draw()  # tick labels have no position until the figure has been laid out
        ax = figure.axes[0]
        # The locators run past the view, so the corner pair are the innermost *visible* labels.
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        cost = min(
            (t for t in ax.get_xticklabels() if x_lo <= t.get_position()[0] <= x_hi),
            key=lambda t: t.get_position()[0],
        )
        rate = min(
            (t for t in ax.get_yticklabels() if y_lo <= t.get_position()[1] <= y_hi),
            key=lambda t: t.get_position()[1],
        )
        renderer = figure.canvas.get_renderer()
        overlaps = cost.get_window_extent(renderer).overlaps(rate.get_window_extent(renderer))
    finally:
        plt.close(figure)

    assert not overlaps


def test_tradeoff_skill_key_is_unfilled_and_the_model_key_is_not(runs_root: Path, model: str, make_result) -> None:
    """Only one of the two keys is about colour, and the other must not look like it is."""
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    figure = tradeoff_figure(df, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        keys = {legend.get_title().get_text(): legend.legend_handles for legend in figure.legends}
        skill = {handle.get_markerfacecolor() for handle in keys["skill"]}
        models = {to_hex(handle.get_markerfacecolor()) for handle in keys["model"]}
    finally:
        plt.close(figure)

    assert skill == {"none"}  # shape is the channel, so an outline is the whole handle
    assert models == {"#3b7ea1"}  # the model key is the colour key, and keeps its fill


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # Dearer and worse than (0.1, 0.8), so nothing would make you pick it.
        ([(0.1, 0.8), (0.2, 0.7)], [(0.1, 0.8)]),
        # Dearer but better: a real choice, so both stay.
        ([(0.1, 0.8), (0.2, 0.9)], [(0.1, 0.8), (0.2, 0.9)]),
        # Cheaper and better at once — one point dominates the whole set.
        ([(0.2, 0.7), (0.1, 0.9), (0.3, 0.5)], [(0.1, 0.9)]),
        # Same price, so only the better rate survives.
        ([(0.1, 0.8), (0.1, 0.6)], [(0.1, 0.8)]),
        ([], []),
    ],
)
def test_pareto_front_keeps_only_what_nothing_beats_on_both_counts(
    points: list[tuple[float, float]], expected: list[tuple[float, float]]
) -> None:
    """Cheapest first, and a point survives only when nothing is both no dearer and no worse."""
    assert _pareto_front(points) == expected


def test_pareto_steps_never_cuts_a_diagonal() -> None:
    """Between frontier points the best rate on offer is the cheaper one's, so the path holds it.

    Sloping straight from one point to the next would assert results at prices nobody ran.
    """
    xs, ys = _pareto_steps([(0.1, 0.8), (0.2, 0.9)], y_min=0.5)

    assert list(zip(xs, ys, strict=True)) == [(0.1, 0.5), (0.1, 0.8), (0.2, 0.8), (0.2, 0.9)]


# --- significance -----------------------------------------------------------------------

#: Enough tasks that a resample can actually resolve a difference. The bootstrap draws whole
#: tasks, so an effect carried by a single task is invisible in the (n-1 / n)**n of resamples
#: that happen to miss it -- with four tasks that is a third of them.
_TASKS = ("alpha", "beta", "gamma", "delta", "epsilon")


def _arena(runs_root: Path, model: str, make_result, spec: dict[str, tuple[float, list[bool]]]) -> pd.DataFrame:
    """A run tree of ``arm -> (cost per run, one success flag per task)``, one rep each."""
    for arm, (cost, outcomes) in spec.items():
        for task, ok in zip(_TASKS, outcomes, strict=True):
            key = RunKey(arm=arm, split="test", model=model, task_id=task, rep=1)
            make_result(runs_root, key, cost_usd=cost, success=ok)
    return load_results(runs_root)


def test_cheaper_but_worse_does_not_dominate(runs_root: Path, model: str, make_result) -> None:
    """The behaviour the whole design exists for: price cannot buy past a worse success rate.

    A single combined score would rank the cheap arm first — it is a fraction of the cost and only
    slightly less accurate. Requiring *both* axes to improve refuses it, because the evidence for
    dominance is only as strong as its weaker half.
    """
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, True, False, False, False]),
            "skill_v1": (0.10, [True, False, False, False, False]),  # far cheaper, strictly worse
            "skill_v2": (0.10, [True, True, True, True, True]),  # cheaper and better everywhere
        },
    )

    tests = skill_tests(df, resamples=4000)
    by_arm = tests.comparisons.set_index("challenger")

    # Overwhelming evidence on cost, none on rate -> the max is large and the claim fails.
    assert by_arm.loc["skill_v1", "p_cost"] < 0.05
    assert by_arm.loc["skill_v1", "p_rate"] > 0.5
    assert by_arm.loc["skill_v1", "p"] == by_arm.loc["skill_v1", "p_rate"]
    # Better on both axes, so the same rule passes it.
    assert by_arm.loc["skill_v2", "p"] < 0.05


def test_frontier_probability_agrees_with_the_plotted_frontier(runs_root: Path, model: str, make_result) -> None:
    """An arm that dominates every resample is never off the frontier, and a dominated one never on."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.10, [True, True, True, True, True]),  # cheaper and better, always
        },
    )

    tests = skill_tests(df, resamples=2000)
    frontier = tests.arms.set_index("arm")["frontier"]
    observed = _pareto_front(list(zip(tests.arms["cost"], tests.arms["rate"], strict=True)))

    assert frontier["skill_v1"] == 1.0
    assert frontier["noskill"] == 0.0
    # The column and the plot's staircase must pick out the same arm on the observed data.
    assert observed == [(pytest.approx(0.10), pytest.approx(1.0))]


def test_skill_tests_compares_every_version_with_the_baseline_only(runs_root: Path, model: str, make_result) -> None:
    """One family, one correction.

    Version-against-version claims are deliberately absent: correcting them alongside the
    baseline ones spends the budget on comparisons nobody makes and can bury the real result.
    """
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.50, [True, True, False, False, False]),
            "skill_v2": (0.10, [True, True, True, True, True]),
        },
    )

    tests = skill_tests(df, resamples=2000)

    assert tests.baseline == "noskill"
    assert set(tests.comparisons["reference"]) == {"noskill"}
    assert list(tests.comparisons["challenger"]) == ["skill_v1", "skill_v2"]
    assert (tests.comparisons["p_adjusted"] >= tests.comparisons["p"]).all()  # adjusting only costs
    assert tests.comparisons["p_adjusted"].notna().all()  # every row is covered by the correction


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Smallest p times the family size, then one fewer, and so on.
        ([0.01, 0.04, 0.30], [0.03, 0.08, 0.30]),
        # Monotone: 0.021 alone would adjust to 0.021, but the running maximum lifts it to
        # match the more significant result above it, so neither can overtake the other.
        ([0.02, 0.021], [0.04, 0.04]),
        ([0.5, 0.9], [1.0, 1.0]),  # clamped at 1
    ],
)
def test_holm_is_monotone_and_scales_by_remaining_tests(raw: list[float], expected: list[float]) -> None:
    assert _holm(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "highest", "expected"),
    [
        ([0.2, 0.9, 0.5], True, {1}),
        ([0.2, 0.9, 0.5], False, {0}),
        ([None, None, 0.8], True, set()),  # a lone value has beaten nothing, so no emphasis
        ([0.3, 0.3, 0.8], False, {0, 1}),  # a tie marks both rather than crowning the first
        ([None, 0.4, 0.1], False, {2}),  # the baseline's blank cells never win a column
    ],
)
def test_best_cells_follows_each_column_own_direction(
    values: list[float | None], highest: bool, expected: set[int]
) -> None:
    """Best means highest for the rates and lowest for cost and p, so the caller states which."""
    assert _best_cells(values, highest=highest) == expected


def test_tests_table_bolds_the_winner_in_each_column(runs_root: Path, model: str, make_result) -> None:
    """The bold cells must track the column's direction, not simply the largest number."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.10, [True, True, True, True, True]),
        },
    )

    html = _tests_table_html(skill_tests(df, resamples=2000))

    assert "<strong>100.0%</strong>" in html  # highest success rate wins its column
    assert "<strong>$0.100</strong>" in html  # *lowest* cost wins its own
    assert "<strong>$1.000</strong>" not in html
    # The comparison columns sit under one header, so it is clear they all refer to the baseline.
    assert 'colspan="4">Compared with No skill' in html


def test_skill_tests_is_deterministic(runs_root: Path, model: str, make_result) -> None:
    """A rebuilt report must reach the same conclusion — a p-value that drifts is worse than none."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {"noskill": (1.00, [True, False, True, False, False]), "skill_v1": (0.20, [True, True, True, True, False])},
    )

    first, second = skill_tests(df, resamples=2000), skill_tests(df, resamples=2000)

    assert first.comparisons["p"].tolist() == second.comparisons["p"].tolist()
    assert first.arms["frontier"].tolist() == second.arms["frontier"].tolist()


def test_skill_tests_declines_to_test_too_few_tasks(runs_root: Path, model: str, make_result) -> None:
    """With one task every resample is identical, so p would be 0 or 1 by construction.

    The fixture tree has a single task, which is exactly the case that must not silently produce
    confident-looking numbers.
    """
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    tests = skill_tests(df)

    assert tests.n_clusters == 1
    assert not tests.usable
    assert tests.comparisons.empty
    assert "at least" in _tests_table_html(tests)  # a note, not a table of numbers


def test_skill_tests_uses_the_test_split_only(runs_root: Path, model: str, make_result) -> None:
    """Train runs feed the improver, so letting them into the test would be marking its own work."""
    spec = {"noskill": (1.00, [True, False, True, False, False]), "skill_v1": (0.20, [True, True, True, True, False])}
    before = skill_tests(_arena(runs_root, model, make_result, spec), resamples=2000)

    for task in _TASKS:  # a pile of cheap, always-passing train runs for the baseline
        make_result(runs_root, RunKey(arm="noskill", split="train", model=model, task_id=task, rep=9), cost_usd=0.001)
    after = skill_tests(load_results(runs_root), resamples=2000)

    assert after.comparisons["p"].tolist() == before.comparisons["p"].tolist()
    assert after.arms["cost"].tolist() == before.arms["cost"].tolist()


def test_arm_marker_widens_with_the_version() -> None:
    """The baseline is off the ladder; each version adds a side, so the order is legible."""
    assert _arm_marker("noskill") == "X"
    assert [_arm_marker(f"skill_v{n}") for n in (1, 2, 3, 9)] == [(3, 0, 0), (4, 0, 0), (5, 0, 0), (11, 0, 0)]


# --- the runs table --------------------------------------------------------------------


def test_runs_table_marks_every_column_sortable_but_the_transcript_link(runs_root: Path, tmp_path: Path) -> None:
    """The link column has no ordering, so offering to sort on it would be a dead control."""
    table = _runs_table_html(load_results(runs_root), tmp_path)
    headings = re.findall(r"<th([^>]*)>([^<]+)</th>", table)

    sortable = {text for attrs, text in headings if "data-sortable" in attrs}
    inert = {text for attrs, text in headings if "data-sortable" not in attrs}
    assert inert == {"transcript"}
    assert "cost $" in sortable and "total tok" in sortable


def test_runs_table_sorts_formatted_numbers_by_their_value(
    runs_root: Path, model: str, make_result, tmp_path: Path
) -> None:
    """Displayed text is formatted for reading, so the sort key has to carry the raw number.

    Sorting on the text would put 9 above 10 and a 3-minute run above a 40-second one.
    """
    make_result(
        runs_root,
        RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=2),
        turns=9,
        input_tokens=1_200_000,
        output_tokens=34_567,
        cost_usd=12.5,
        duration_s=185.0,
    )
    table = _runs_table_html(load_results(runs_root), tmp_path)
    (row,) = [r for r in table.split("<tr") if "1,234,567" in r]

    keys = {text: key for key, text in re.findall(r'<td data-sort="([^"]+)">([^<]*)</td>', row)}
    # The separators, the padded cost and the humanised duration all read differently to
    # the values they stand for.
    assert keys["1,234,567"] == "1234567.0"
    assert keys["12.500"] == "12.5"
    assert float(keys["3.1m"]) == 185.0
    assert keys["9"] == "9.0"


def test_runs_table_ranks_a_skill_that_failed_to_load_above_one_that_loaded() -> None:
    """Sorting on 'skill loaded' is how you find the runs that did not measure their skill."""
    baseline = pd.Series({"arm": "noskill", "skill_loaded": None})
    loaded = pd.Series({"arm": "skill_v1", "skill_loaded": True})
    missed = pd.Series({"arm": "skill_v1", "skill_loaded": False})
    unknown = pd.Series({"arm": "skill_v1", "skill_loaded": None})

    ranks = [_loaded_rank(r) for r in (missed, loaded, unknown, baseline)]

    assert ranks == sorted(ranks, reverse=True)


def _run(arm: str, model: str, success: bool, loaded: object, split: str = "test") -> dict:
    return {"arm": arm, "model": model, "split": split, "success": success, "skill_loaded": loaded}


def test_loaded_only_rates_pools_over_one_model_mix_on_both_sides() -> None:
    """A model that never loaded the skill must leave the pooled baseline too.

    Otherwise a model that fails every run — an outage, an unavailable id — depresses the
    baseline while contributing nothing to the loaded side, manufacturing a gain from nothing.
    """
    rows = (
        # A working model: loads the skill, same rate in both arms.
        [_run("noskill", "good", True, None) for _ in range(4)]
        + [_run("skill_v1", "good", True, True) for _ in range(4)]
        # A broken model: fails everything and never loads the skill.
        + [_run("noskill", "broken", False, None) for _ in range(4)]
        + [_run("skill_v1", "broken", False, False) for _ in range(4)]
    )
    table = loaded_only_rates(pd.DataFrame(rows))
    pooled = table[table["scope"] == "matched"].iloc[0]
    raw = table[table["scope"] == "all"].iloc[0]

    # Only the loading model is pooled, on both sides — so the skill shows no gain…
    assert pooled["loaded"] == 4 and pooled["runs"] == 4
    assert pooled["baseline"] == 1.0 and pooled["rate"] == 1.0
    assert pooled["delta"] == 0.0
    # …while the raw all-models row keeps the mix as it ran, and shows the artifact the
    # matched row exists to expose: a +50% that is entirely the broken model leaving.
    assert raw["runs"] == 8 and raw["baseline"] == 0.5 and raw["rate"] == 1.0
    assert raw["delta"] == 0.5


def test_loaded_only_rates_reports_a_model_that_never_loaded() -> None:
    """The row still appears — a 0% load rate is the finding, not a reason to hide the model."""
    rows = [_run("noskill", "m", True, None), _run("skill_v1", "m", False, False)]
    table = loaded_only_rates(pd.DataFrame(rows))

    row = table[table["model"] == "m"].iloc[0]
    assert row["loaded"] == 0
    assert row["load_rate"] == 0.0
    assert pd.isna(row["rate"]) and pd.isna(row["delta"])


# --- split diff ------------------------------------------------------------------------


def test_split_diff_pairs_a_rewrite_and_leaves_the_other_side_empty() -> None:
    """A changed line sits opposite its replacement; where one side runs out, it faces nothing."""
    rows = _split_diff_rows(["keep", "old"], ["keep", "new", "extra"])
    assert [(r.left, r.right) for r in rows] == [("keep", "keep"), ("old", "new"), (None, "extra")]
    assert [(r.left_no, r.right_no) for r in rows] == [(1, 1), (2, 2), (None, 3)]
    assert [r.kind for r in rows[1:]] == ["replace", "replace"]  # one run, so both rows tint


def test_split_diff_collapses_untouched_stretches_but_keeps_context() -> None:
    """Three lines of context survive on each side of a change; the rest becomes one gap row."""
    before = [f"line {n}" for n in range(20)]
    after = [*before[:10], "inserted", *before[10:]]
    rows = _split_diff_rows(before, after)
    assert [r.kind for r in rows] == ["gap", *["equal"] * 3, "insert", *["equal"] * 3, "gap"]
    assert [r.left for r in rows if r.kind == "equal"] == [f"line {n}" for n in (7, 8, 9, 10, 11, 12)]


def test_skill_diff_marks_only_the_words_that_changed() -> None:
    """Word-level highlighting inside a rewritten line, on both sides, without touching the rest."""
    diff = _skill_diff_html({"SKILL.md": "use ulm here\n"}, {"SKILL.md": "use mlm here\n"}, ("v001", "v002"))
    assert "<mark>ulm</mark>" in diff and "<mark>mlm</mark>" in diff
    assert "<mark>here</mark>" not in diff


def test_skill_diff_reports_an_unchanged_version_and_a_first_draft() -> None:
    """Neither case has a diff to show, and each says which case it is."""
    content = {"SKILL.md": "same\n"}
    assert "No content change" in _skill_diff_html(content, content, ("v001", "v002"))
    assert "no parent" in _skill_diff_html(None, content, ("", "v001"))


# --- orphan reaping --------------------------------------------------------------------

#: A child that outlives its parent, and a parent that exits immediately after starting it.
#: Reproduces the leak: the SDK terminates the agent, and the commands it started live on
#: with no parent left to find them by.
_ORPHAN = "import time; time.sleep(60)"
_ABANDONING_PARENT = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {_ORPHAN!r}])"

#: Linux, macOS and Windows all qualify; this only skips where psutil cannot read a
#: process's environment, which is the one thing the reaper is built on.
pytestmark_procs = pytest.mark.skipif(not supported(), reason="psutil cannot read process environments here")


def _spawn_orphan(holder: Path) -> None:
    """Start a labelled process whose parent then exits, leaving it running and unparented."""
    parent = subprocess.Popen(
        [sys.executable, "-c", _ABANDONING_PARENT],
        env=label_env(dict(os.environ), holder),
    )
    parent.wait()


@pytestmark_procs
def test_reap_kills_a_command_that_outlived_its_agent(tmp_path: Path) -> None:
    """The whole point: a process whose parent is gone is still found, by its marker, and killed."""
    holder = tmp_path / "run"
    holder.mkdir()
    _spawn_orphan(holder)

    orphans = survivors(holder)
    assert orphans, "the orphaned command was not found after its parent exited"

    assert set(reap(holder)) == set(orphans)
    deadline = time.monotonic() + 5
    while survivors(holder) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not survivors(holder)


@pytestmark_procs
def test_reap_leaves_everything_outside_the_run_alone(tmp_path: Path) -> None:
    """Only the marked run is reaped — not the test process, and not another run's agent."""
    mine = tmp_path / "mine"
    other = tmp_path / "other"
    for path in (mine, other):
        path.mkdir()
    _spawn_orphan(other)
    bystander = subprocess.Popen([sys.executable, "-c", _ORPHAN])
    try:
        assert os.getpid() not in survivors(mine)
        assert bystander.pid not in survivors(mine)
        assert reap(mine) == []
        assert survivors(other), "reaping one run killed another run's processes"
        assert bystander.poll() is None
    finally:
        bystander.kill()
        reap(other)


@pytestmark_procs
def test_label_env_marks_a_run_without_disturbing_the_rest(tmp_path: Path) -> None:
    """Stamping is additive — an agent's carefully built environment is otherwise untouched."""
    holder = tmp_path / "run"
    base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-test"}

    marked = label_env(dict(base), holder)

    assert marked.items() >= base.items()
    assert str(holder) in marked.values()
