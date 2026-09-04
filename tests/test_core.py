"""Unit tests for the pure parts the CLI sits on: grader, schemas, paths, skills.

Deliberately thin — one test per behaviour that would silently corrupt a benchmark if it
broke, not an exhaustive sweep of each validator.
"""

from __future__ import annotations

import ast
import asyncio
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch

import acumen
from acumen.agents import (
    AgentError,
    AgentOptions,
    AgentResult,
    _claude_path_rule,
    _claude_settings,
    _codex_command,
    _codex_terminal,
    _install_codex_guard,
    check_agent_cli,
    provider_for_model,
    run_agent,
)
from acumen.bench import BenchmarkInvalidError, PlannedRun, build_matrix, pending, run_matrix
from acumen.check import (
    CheckError,
    CheckResult,
    check_tasks,
    import_probe,
    orphan_scripts,
    script_path,
    select_tasks,
    summarize_checks,
)
from acumen.config import Config, ConfigError, derive_skill_name, load_config, parse_config
from acumen.env import (
    AUTH_ENV_VARS,
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_TIMEOUT_MS,
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
from acumen.grade import INVALID_REASONS, grade_answer, grade_run
from acumen.guard import SYSTEM_ROOTS, find_escape
from acumen.improve import ImproveError, _write_material, collect_train_runs, load_rates
from acumen.logs import LiveLog
from acumen.paths import RunKey, arm_name, is_complete, parse_run_dir, run_dir
from acumen.pricefeed import PriceFeedError, diff_rates, fetch_table, parse_anthropic, parse_openai
from acumen.prices import (
    PriceTable,
    Rates,
    Usage,
    normalize_usage,
    price_provenance,
    price_run,
    price_usage,
    pricer,
    resolve_cost,
)
from acumen.procs import label_env, reap, supported, survivors
from acumen.prompts import draft_prompt, feedback_block, improve_prompt
from acumen.report import (
    ReportError,
    _arm_marker,
    _best_cells,
    _holm,
    _integrity_notes,
    _loaded_flags,
    _loaded_rank,
    _pareto_front,
    _pareto_steps,
    _runs_table_html,
    _skill_diff_html,
    _split_diff_rows,
    _tests_table_html,
    arm_metrics,
    build_report,
    load_results,
    loaded_only_rates,
    metrics_figure,
    render_report,
    resolve_palette,
    skill_tests,
    tradeoff_figure,
)
from acumen.review import (
    MAX_NOTE_CHARS,
    PACKET_DIGEST,
    PACKET_DIRNAME,
    PACKET_SCRIPTS,
    REVIEW_FILE,
    ReviewError,
    parse_reviews,
    review_tasks,
    write_packet,
)
from acumen.runner import (
    RunOutcome,
    StderrFilter,
    _provider_exhaustion_error,
    _sandbox_denial,
    _skill_fired,
    _terminal_reason,
    run_once,
)
from acumen.sandbox import Sandbox
from acumen.ship import _ship_env
from acumen.skills import SkillError, load_skill, read_meta, skill_hash, write_meta
from acumen.taskgen import dump_tasks, harvest_scripts
from acumen.tasks import Task, TaskError, TaskSplit, load_tasks, parse_tasks
from acumen.trajectory import (
    Metrics,
    Observation,
    Step,
    ToolCall,
    Trajectory,
    from_claude_records,
    from_codex_events,
    render_trajectory,
    write_trajectory_json,
)
from acumen.transcript import render_agent_transcript, render_codex_events, render_codex_transcript

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

    (tmp_path / "result.json").write_text('{"valid": false}')
    assert not is_complete(tmp_path)


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


def _codex_read(name: str, status: str, exit_code: int | None) -> str:
    return (
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": f"sed -n '1,200p' .agents/skills/{name}/SKILL.md",
                    "exit_code": exit_code,
                    "status": status,
                },
            }
        )
        + "\n"
    )


def test_codex_skill_fired_matches_project_skill_read(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    path.write_text(_codex_read("target", "completed", 0))
    assert _skill_fired(path, "target", provider="codex") is True
    assert _skill_fired(path, "another", provider="codex") is False


def test_codex_skill_fired_ignores_a_read_that_never_ran(tmp_path: Path) -> None:
    """A command the sandbox refused is an attempt to load the skill, not a load.

    Counting the attempt reports the skill as loaded-and-ineffective when its body was
    never seen, sending the improver to rewrite prose the agent never read.
    """
    path = tmp_path / "codex.jsonl"
    path.write_text(_codex_read("target", "failed", 1))
    assert _skill_fired(path, "target", provider="codex") is False

    started = tmp_path / "started.jsonl"
    started.write_text(_codex_read("target", "in_progress", None))
    assert _skill_fired(started, "target", provider="codex") is False


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


def test_agent_policies_are_workspace_scoped(tmp_path: Path) -> None:
    work = tmp_path / "work"
    home = tmp_path / "home"
    venv = tmp_path / "venv"
    source = tmp_path / "source"
    for path in (work, home / "tmp", home / ".codex", venv, source):
        path.mkdir(parents=True)
    options = AgentOptions(
        cwd=work,
        env={"PATH": os.environ["PATH"], "HOME": str(home), "CODEX_HOME": str(home / ".codex")},
        model="gpt-5.6-sol",
        read_dirs=(venv, source),
    )

    claude = _claude_settings(options)
    # No OS sandbox for Claude: enabling it also interposes an egress proxy whose allowlist
    # cannot be set from --settings, which is what killed every download in a whole pass.
    # Filesystem containment is the guard hook instead; see test_guard_* below.
    assert "sandbox" not in claude
    assert _claude_path_rule("Read", venv.resolve()) in claude["permissions"]["allow"]
    assert _claude_path_rule("Edit", work.resolve()) in claude["permissions"]["allow"]
    assert _claude_path_rule("Edit", source.resolve()) not in claude["permissions"]["allow"]

    command = _codex_command(options, "do it")
    overrides = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-c"]
    assert 'default_permissions="acumen"' in overrides
    assert 'web_search="disabled"' in overrides
    assert "tools.web_search=false" in overrides
    assert "features.browser_use=false" in overrides
    assert "features.apps=false" in overrides
    assert "features.network_proxy=true" in overrides
    filesystem = next(value for value in overrides if value.startswith("permissions.acumen.filesystem="))
    assert '":root"="deny"' in filesystem
    assert '":minimal"="read"' in filesystem
    assert f'{json.dumps(str(work.resolve()))}="write"' in filesystem
    assert f'{json.dumps(str(venv.resolve()))}="read"' in filesystem
    codex_dir = Path(shutil.which("codex", path=options.env["PATH"])).resolve().parent
    assert f'{json.dumps(str(codex_dir))}="read"' in filesystem
    assert '"/bin"="read"' in filesystem
    assert "permissions.acumen.network.enabled=true" in overrides
    assert "permissions.acumen.network.allow_local_binding=false" in overrides
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert str(Path.home() / ".codex" / "config.toml") not in "\n".join(command)
    assert str(Path.home() / ".codex" / "rules") not in "\n".join(command)


def test_egress_is_unrestricted_for_both_providers(tmp_path: Path) -> None:
    """Nothing in a sandbox may refuse an outbound host.

    A target decides its own hosts at runtime — which mirror a loader falls back to, which
    server a prior comes from — so no list written before a pass can be complete. A refused
    host does not surface to the agent as a policy decision either: it arrives as an ordinary
    connection error, the agent improvises from memory, and the run scores as a wrong answer.
    """
    work, home = tmp_path / "work", tmp_path / "home"
    for path in (work, home / ".codex"):
        path.mkdir(parents=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "CODEX_HOME": str(home / ".codex")}
    options = AgentOptions(cwd=work, env=env, model="claude-opus-5")

    # Claude gets no sandbox at all: its proxy cannot be opened from --settings, so the only
    # configuration that reaches the network is the one that does not configure it.
    settings = _claude_settings(options)
    assert "sandbox" not in settings
    assert not any(rule.startswith("WebFetch(domain:") for rule in settings["permissions"]["allow"])
    assert "deny" not in settings["permissions"]

    command = _codex_command(replace(options, model="gpt-5.6-sol"), "probe")
    overrides = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-c"]
    # ``limited`` is not merely a narrower domain table: its proxy also answers anything
    # outside GET/HEAD/OPTIONS with 403, which breaks every POST a target's loaders make.
    assert 'permissions.acumen.network.mode="full"' in overrides
    assert 'permissions.acumen.network.mode="limited"' not in overrides
    domains = next(value for value in overrides if value.startswith("permissions.acumen.network.domains="))
    assert domains == 'permissions.acumen.network.domains={"*"="allow"}'


def test_sandbox_denial_is_infrastructure_not_evidence(tmp_path: Path) -> None:
    """A host the sandbox refused says nothing about the model, so it must not be scored."""
    jsonl = tmp_path / "transcript.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "ProxyError('Unable to connect to proxy', "
                            "OSError('Tunnel connection failed: 403 Forbidden'))",
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    evidence = _sandbox_denial(jsonl)
    assert evidence is not None and "403" in evidence
    assert "sandbox_blocked" in INVALID_REASONS

    # An origin-server 403 or a DNS failure is the target's own trouble, not the harness's.
    for innocuous in ("HTTPError: 403 Client Error: Forbidden for url: https://example.org/x", "NameResolutionError"):
        other = tmp_path / "other.jsonl"
        other.write_text(json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": innocuous}]}}))
        assert _sandbox_denial(other) is None

    assert _sandbox_denial(tmp_path / "absent.jsonl") is None


def test_codex_runtime_roots_cover_the_vendored_native_binary(tmp_path: Path) -> None:
    """The npm shim execs a binary in a sibling package, not one beside itself.

    Allowing only the shim's own directory leaves the real binary outside the command
    sandbox's namespace, and every command the agent runs fails with
    ``bwrap: execvp …: No such file or directory``.
    """
    from acumen.agents import _codex_runtime_roots

    package = tmp_path / "lib" / "node_modules" / "@openai" / "codex"
    shim = package / "bin" / "codex.js"
    shim.parent.mkdir(parents=True)
    shim.touch()
    platform_package = package / "node_modules" / "@openai" / "codex-linux-arm64"
    native = platform_package / "vendor" / "aarch64-unknown-linux-musl" / "bin" / "codex"
    native.parent.mkdir(parents=True)
    native.touch()

    roots = _codex_runtime_roots(shim)
    assert shim.parent in roots
    assert any(native.is_relative_to(root) for root in roots)

    # A standalone install is a real binary; its own directory is the whole requirement.
    standalone = tmp_path / "usr" / "bin" / "codex"
    standalone.parent.mkdir(parents=True)
    standalone.touch()
    assert _codex_runtime_roots(standalone) == [standalone.parent]


def test_codex_filesystem_permissions_reach_the_binary_bwrap_execs(tmp_path: Path) -> None:
    """The emitted allowlist, not just the helper, has to name the native binary.

    Reproduces the layout of a benchmark host where every Codex command failed with
    ``bwrap: execvp …/vendor/aarch64-unknown-linux-musl/bin/codex: No such file or
    directory``: npm puts a symlink on PATH, pointing at a Node shim, whose native
    binary lives in a sibling package the shim's own directory does not contain.
    """
    npm = tmp_path / "npm"
    package = npm / "lib" / "node_modules" / "@openai" / "codex"
    shim = package / "bin" / "codex.js"
    native = (
        package
        / "node_modules"
        / "@openai"
        / "codex-linux-arm64"
        / "vendor"
        / "aarch64-unknown-linux-musl"
        / "bin"
        / "codex"
    )
    for path in (shim, native):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.chmod(0o755)
    binroot = npm / "bin"
    binroot.mkdir()
    (binroot / "codex").symlink_to(os.path.relpath(shim, binroot))

    work = tmp_path / "work"
    work.mkdir()
    options = AgentOptions(
        cwd=work,
        env={"PATH": str(binroot), "HOME": str(tmp_path / "home"), "CODEX_HOME": str(tmp_path / "home" / ".codex")},
        model="gpt-5.6-sol",
    )
    command = _codex_command(options, "probe")
    overrides = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-c"]
    filesystem = next(value for value in overrides if value.startswith("permissions.acumen.filesystem="))

    allowed = [
        json.loads(entry.rpartition("=")[0])
        for entry in filesystem.split("=", 1)[1].strip("{}").split(",")
        if json.loads(entry.rpartition("=")[2]) in {"read", "write"}
    ]
    # The sandbox resolves symlinks, so compare real paths on both sides.
    target = native.resolve()
    assert any(target.is_relative_to(Path(entry).resolve()) for entry in allowed)


def test_claude_options_write_run_local_settings_file(tmp_path: Path) -> None:
    pytest.importorskip("claude_agent_sdk")
    from acumen.agents import _claude_options

    work = tmp_path / "work"
    config_dir = tmp_path / "claude-config"
    work.mkdir()
    options = AgentOptions(
        cwd=work,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "CLAUDE_CONFIG_DIR": str(config_dir),
        },
        model="claude-haiku-4-5",
    )

    sdk_options = _claude_options(options)

    settings_path = Path(sdk_options.settings)
    assert settings_path == config_dir / "acumen-settings.json"
    settings = json.loads(settings_path.read_text())
    assert settings == _claude_settings(options)


def _claude_result(**kwargs: object) -> object:
    from claude_agent_sdk import ResultMessage

    fields: dict[str, object] = {
        "subtype": "error_max_turns",
        "duration_ms": 91_000,
        "duration_api_ms": 88_000,
        "is_error": True,
        "num_turns": 40,
        "session_id": "session-1",
        "total_cost_usd": 0.42,
        "usage": {"input_tokens": 1200, "output_tokens": 340},
        "errors": ["Reached maximum number of turns (40)"],
    }
    fields.update(kwargs)
    return ResultMessage(**fields)  # type: ignore[arg-type]


class _FakeClaudeClient:
    """A stand-in for ``ClaudeSDKClient`` that replays a scripted message stream.

    ``script`` is the response to the graded prompt, replayed until (and including) the first
    ResultMessage. ``after`` is what the CLI goes on emitting once that result has landed: the
    task lifecycle events of a run whose background work is still going, and the extra results a
    re-entered session produces. ``calls`` records the control-protocol calls in order, which is
    what the teardown is asserted on.
    """

    instances: list[_FakeClaudeClient] = []
    script: list[object] = []
    after: list[object] = []
    error: Exception | None = None

    def __init__(self, options: object = None, transport: object = None) -> None:
        self.options = options
        self.calls: list[tuple[str, object]] = []
        self.pending = list(type(self).after)
        type(self).instances.append(self)

    async def connect(self, prompt: object = None) -> None:
        self.calls.append(("connect", prompt))

    async def query(self, prompt: object, session_id: str = "default") -> None:
        self.calls.append(("query", prompt))

    async def receive_response(self):
        from claude_agent_sdk import ResultMessage

        for message in type(self).script:
            yield message
            if isinstance(message, ResultMessage):
                break
        if type(self).error is not None:
            raise type(self).error

    async def receive_messages(self):
        while self.pending:
            yield self.pending.pop(0)

    async def stop_task(self, task_id: str) -> None:
        self.calls.append(("stop_task", task_id))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))

    async def disconnect(self) -> None:
        self.calls.append(("disconnect", None))


def _patch_claude_client(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[object],
    error: Exception | None = None,
    after: list[object] | None = None,
) -> None:
    import claude_agent_sdk

    monkeypatch.setattr(_FakeClaudeClient, "instances", [])
    monkeypatch.setattr(_FakeClaudeClient, "script", messages)
    monkeypatch.setattr(_FakeClaudeClient, "after", after or [])
    monkeypatch.setattr(_FakeClaudeClient, "error", error)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClaudeClient)


def _claude_agent_options(tmp_path: Path, **kwargs: object) -> AgentOptions:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return AgentOptions(
        cwd=work,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / "cfg")},
        model="claude-opus-5",
        **kwargs,  # type: ignore[arg-type]
    )


def _live_tasks(*task_ids: str) -> object:
    """The CLI's level signal for the set of background tasks currently running."""
    from claude_agent_sdk import SystemMessage

    return SystemMessage(
        subtype="background_tasks_changed",
        data={
            "tasks": [
                {"task_id": task_id, "task_type": "local_bash", "description": "python script.py"}
                for task_id in task_ids
            ]
        },
    )


def test_claude_cap_breach_keeps_the_result_the_cli_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI exits non-zero *after* reporting a cap breach, and the SDK re-raises that.

    The structured result has already arrived, so treating the trailing exception as the
    outcome throws away the turns, cost, session id and transcript of a run that really
    happened, and files a turn cap as an unexplained crash.
    """
    pytest.importorskip("claude_agent_sdk")

    _patch_claude_client(
        monkeypatch,
        [_claude_result()],
        Exception("Claude Code returned an error result: Reached maximum number of turns (40)"),
    )

    result = asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path, max_turns=40)))

    assert result.subtype == "error_max_turns"
    assert result.num_turns == 40
    assert result.session_id == "session-1"
    assert result.total_cost_usd == 0.42
    # The runner maps the subtype onto its reason taxonomy; a discarded result cannot.
    assert _terminal_reason(result) == "max_turns"
    # Even a run that ends in an exception must not leave its session live.
    assert ("disconnect", None) in _FakeClaudeClient.instances[-1].calls


def test_claude_crash_after_a_clean_result_still_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a deliberate non-zero exit after an error result is expected; a break is not."""
    pytest.importorskip("claude_agent_sdk")

    _patch_claude_client(
        monkeypatch,
        [_claude_result(subtype="success", is_error=False, errors=None)],
        Exception("transport closed unexpectedly"),
    )

    with pytest.raises(Exception, match="transport closed unexpectedly"):
        asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path)))


def test_claude_records_the_first_result_not_a_re_entered_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backgrounded command finishing re-enters the session and produces a second result.

    That second result describes the re-entry alone. One measured run took 1117 seconds over 41
    turns and hit its cap; the trailing re-entry reported 2 turns in 4 seconds, and that is what
    got recorded — a capped run filed as a clean success, with inferred cost 96% low. Only the
    result the graded prompt produced may be recorded.
    """
    pytest.importorskip("claude_agent_sdk")

    graded = _claude_result(
        num_turns=41,
        duration_ms=1_117_000,
        usage={"input_tokens": 297_709, "output_tokens": 2287},
    )
    re_entry = _claude_result(
        subtype="success",
        is_error=False,
        errors=None,
        num_turns=2,
        duration_ms=4_180,
        usage={"input_tokens": 95_045, "output_tokens": 229},
    )
    _patch_claude_client(monkeypatch, [graded], after=[re_entry])

    result = asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path)))

    assert result.num_turns == 41
    assert result.duration_ms == 1_117_000
    assert result.usage == {"input_tokens": 297_709, "output_tokens": 2287}
    assert result.subtype == "error_max_turns"


def test_claude_teardown_stops_background_tasks_and_closes_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outstanding background work is stopped, and the session really closed, before grading.

    Left alone the CLI queues each finishing task as a fresh prompt and re-enters the model past
    its turn cap, where it answers every tool call with a cancelled-permission denial — the agent
    waiting on an operator who does not exist while the harness has already moved on to grading.
    """
    pytest.importorskip("claude_agent_sdk")

    seen: list[object] = []
    stopped = _live_tasks()
    _patch_claude_client(monkeypatch, [_live_tasks("bda1dmuki"), _claude_result()], after=[stopped])

    asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path), on_event=seen.append))

    client = _FakeClaudeClient.instances[-1]
    assert [name for name, _ in client.calls] == ["connect", "query", "stop_task", "interrupt", "disconnect"]
    assert ("stop_task", "bda1dmuki") in client.calls
    # Waiting for the stop to be confirmed is what lets the CLI flush the task's output file
    # before the run's artifacts are collected, so those events are consumed, not skipped.
    assert stopped in seen


def test_claude_teardown_reads_task_lifecycle_edges_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI that reports only start/stop edges must still be tracked, and a finished task left alone."""
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import TaskStartedMessage, TaskUpdatedMessage

    def started(task_id: str) -> object:
        return TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id=task_id,
            description="python script.py",
            uuid="u",
            session_id="s",
        )

    finished = TaskUpdatedMessage(
        subtype="task_updated",
        data={},
        task_id="done",
        patch={"status": "completed"},
        status="completed",
    )
    _patch_claude_client(monkeypatch, [started("done"), finished, started("live"), _claude_result()])

    asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path)))

    calls = _FakeClaudeClient.instances[-1].calls
    assert ("stop_task", "live") in calls
    assert ("stop_task", "done") not in calls


def test_claude_teardown_survives_a_control_protocol_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's result is in hand before teardown; nothing that goes wrong tidying up may lose it."""
    pytest.importorskip("claude_agent_sdk")

    _patch_claude_client(
        monkeypatch, [_live_tasks("b1"), _claude_result(subtype="success", is_error=False, errors=None)]
    )

    async def boom(self: object, task_id: str) -> None:
        raise RuntimeError("control request failed")

    monkeypatch.setattr(_FakeClaudeClient, "stop_task", boom)

    result = asyncio.run(run_agent("go", options=_claude_agent_options(tmp_path)))

    assert result.subtype == "success"
    assert ("disconnect", None) in _FakeClaudeClient.instances[-1].calls


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


def test_guard_blocks_exploration_outside_the_run_roots(tmp_path: Path) -> None:
    """The agent may work in its sandbox and use system paths, and go nowhere else."""
    work = tmp_path / "work"
    venv = tmp_path / "venv"
    elsewhere = tmp_path / "elsewhere"
    for path in (work, venv, elsewhere):
        path.mkdir()
    roots = [work, venv, *SYSTEM_ROOTS]

    # Allowed: the workspace, the target venv, and the system paths every command touches.
    assert find_escape({"command": "python analysis.py"}, roots, cwd=work) is None
    assert find_escape({"command": "curl -sS -o /dev/null https://zenodo.org/"}, roots, cwd=work) is None
    assert find_escape({"file_path": str(work / "answer.md")}, roots, cwd=work) is None
    assert find_escape({"command": f"ls {venv}/lib"}, roots, cwd=work) is None
    assert find_escape({"command": "cat /etc/hosts"}, roots, cwd=work) is None

    # Blocked: anywhere else on the host, however the path is spelled.
    assert find_escape({"command": f"ls {elsewhere}"}, roots, cwd=work) is not None
    assert find_escape({"file_path": str(elsewhere / "secret.txt")}, roots, cwd=work) is not None
    assert find_escape({"command": "ls ../.."}, roots, cwd=work) is not None

    # ``~`` means the agent's throwaway home, not the operator's. Judging it against the
    # harness process's own HOME would deny the agent its own scratch directory and permit
    # nothing useful in exchange.
    agent_home = work / "home"
    (agent_home / "tmp").mkdir(parents=True)
    assert find_escape({"command": "ls ~/tmp"}, [*roots, agent_home], cwd=work, home=agent_home) is None
    assert find_escape({"command": "cat ~/.ssh/id_rsa"}, roots, cwd=work, home=Path("/home/someone")) is not None
    # Nested inputs are walked, so a path cannot hide inside a structured tool argument.
    assert find_escape({"edits": [{"path": str(elsewhere / "x")}]}, roots, cwd=work) is not None


def test_guard_denies_paths_win_over_allowed_roots(tmp_path: Path) -> None:
    """A meta-agent's hidden evidence stays hidden even when it sits inside the workspace."""
    work = tmp_path / "work"
    runs = work / "runs"
    runs.mkdir(parents=True)
    assert find_escape({"command": f"ls {runs}"}, [work], deny_roots=[runs], cwd=work) is not None
    assert find_escape({"command": f"ls {work}"}, [work], deny_roots=[runs], cwd=work) is None


def test_codex_sandbox_failure_is_caught_in_command_output_not_only_stderr() -> None:
    """A sandbox that cannot start reports itself where the command was meant to write.

    bubblewrap's ``execvp`` failure arrives as the command's ``aggregated_output``, not on
    codex's own stderr. Scanning stderr alone leaves a run whose every command died before it
    began looking like a model that answered badly, and it enters the report as evidence.
    """
    result = _codex_terminal(
        [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/usr/bin/bash -lc 'python script.py'",
                    "aggregated_output": "bwrap: execvp /opt/codex/vendor/bin/codex: No such file or directory\n",
                    "exit_code": 1,
                    "status": "failed",
                },
            },
            {"type": "item.completed", "item": {"type": "agent_message", "text": "here is my answer"}},
        ],
        0,
        10,
    )
    assert result.is_error is True
    assert result.subtype == "error_sandbox"
    assert result.errors is not None
    assert "bwrap" in "\n".join(result.errors)


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
    # Without a rollout to read there is nothing to recover, and that stays a quiet no-op.
    assert result.usage == {}


def _fake_codex_with_rollout(
    path: Path,
    codex_home: Path,
    thread_id: str,
    steps: list[tuple[dict | None, dict | None]],
) -> Path:
    """Write a stub ``codex`` that records rollout usage as it emits stdout events.

    Each step is ``(running_total, event)``: the cumulative token usage Codex would append to
    its rollout session file, and the JSONL event it would print next. Either half may be
    ``None``. This is the only source of usage for a run that never reaches ``turn.completed``.
    """
    session = codex_home / "sessions" / "2026" / "08" / "19"
    rollout = session / f"rollout-2026-08-19T00-00-00-{thread_id}.jsonl"
    lines = [f"mkdir -p {shlex.quote(str(session))}"]
    for total, event in steps:
        if total is not None:
            record = {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": total}}}
            lines.append(f"printf '%s\\n' {shlex.quote(json.dumps(record))} >> {shlex.quote(str(rollout))}")
        if event is not None:
            lines.append(f"printf '%s\\n' {shlex.quote(json.dumps(event))}")
    cli = path / "codex"
    cli.write_text("#!/bin/sh\n" + "\n".join(lines) + "\n")
    cli.chmod(0o755)
    return cli


def _codex_home_options(tmp_path: Path, codex_home: Path, **kwargs: object) -> AgentOptions:
    options = _codex_options(tmp_path, **kwargs)
    return replace(options, env={**options.env, "CODEX_HOME": str(codex_home)})


def test_codex_recovers_usage_from_the_rollout_when_the_turn_cap_kills_the_run(tmp_path: Path) -> None:
    """A capped run spent real tokens, and its rollout is the only place they survive.

    ``codex exec --json`` reports usage once, in ``turn.completed``, which a run stopped at the
    cap never reaches. Recording zero there priced a run that did minutes of work at $0.00.
    """
    codex_home = tmp_path / "codex-home"
    _fake_codex_with_rollout(
        tmp_path,
        codex_home,
        "thread-1",
        [
            (None, {"type": "thread.started", "thread_id": "thread-1"}),
            (None, {"type": "turn.started"}),
            (
                {"input_tokens": 900, "cached_input_tokens": 400, "output_tokens": 30},
                {"type": "item.completed", "item": {"id": "a", "type": "command_execution", "command": "echo one"}},
            ),
            (
                {"input_tokens": 1_800, "cached_input_tokens": 900, "output_tokens": 70, "total_tokens": 1_870},
                {"type": "item.completed", "item": {"id": "b", "type": "command_execution", "command": "echo two"}},
            ),
            (None, {"type": "item.completed", "item": {"id": "c", "type": "agent_message", "text": "never reached"}}),
        ],
    )
    result = asyncio.run(run_agent("go", options=_codex_home_options(tmp_path, codex_home, max_turns=2)))

    assert result.subtype == "error_max_turns"
    assert result.num_turns == 2
    # The running total, not a sum of the reports, and without the derived total_tokens key.
    assert result.usage == {"input_tokens": 1_800, "cached_input_tokens": 900, "output_tokens": 70}
    assert normalize_usage(result.usage, provider="codex") == Usage(
        input=1_800, cache_read=900, cache_write=0, output=70
    )


def test_codex_prefers_turn_completed_usage_over_the_rollout(tmp_path: Path) -> None:
    """The rollout fills a gap; it never overrides the figure Codex reports itself."""
    codex_home = tmp_path / "codex-home"
    _fake_codex_with_rollout(
        tmp_path,
        codex_home,
        "thread-1",
        [
            (None, {"type": "thread.started", "thread_id": "thread-1"}),
            (
                {"input_tokens": 10, "output_tokens": 1},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            ),
            ({"input_tokens": 11, "output_tokens": 2}, {"type": "turn.completed", "usage": {"input_tokens": 12}}),
        ],
    )
    result = asyncio.run(run_agent("go", options=_codex_home_options(tmp_path, codex_home)))

    assert not result.is_error
    assert result.usage == {"input_tokens": 12}


def test_codex_budget_cap_bites_before_the_turn_ends(tmp_path: Path) -> None:
    """The rollout total is what lets max_usd stop a Codex run rather than only report it."""
    codex_home = tmp_path / "codex-home"
    _fake_codex_with_rollout(
        tmp_path,
        codex_home,
        "thread-1",
        [
            (None, {"type": "thread.started", "thread_id": "thread-1"}),
            (
                {"input_tokens": 1_000_000, "output_tokens": 0},
                {"type": "item.completed", "item": {"id": "a", "type": "command_execution", "command": "echo one"}},
            ),
            (None, {"type": "item.completed", "item": {"id": "b", "type": "agent_message", "text": "never reached"}}),
        ],
    )
    # gpt-5.6-sol bills $5/M input, so a million fresh input tokens is $5 against a $1 cap.
    prices = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)},
        fetched_as_of="2026-08-04",
    )
    options = _codex_home_options(tmp_path, codex_home, max_usd=1.0, price_usd=pricer("gpt-5.6-sol", prices))
    result = asyncio.run(run_agent("go", options=options))

    assert result.subtype == "error_max_budget_usd"
    # Stopped on the first report past the cap, before the agent's next action.
    assert result.result == ""
    assert result.usage == {"input_tokens": 1_000_000, "output_tokens": 0}


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
    prices = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)},
        fetched_as_of="2026-08-04",
    )
    options = _codex_options(tmp_path, max_usd=1.0, price_usd=pricer("gpt-5.6-sol", prices))
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


@pytest.mark.parametrize(
    "detail",
    [
        "You've hit your usage limit · resets at 5pm",
        "insufficient_quota: exceeded your current quota",
        "Your credit balance is too low to access the Anthropic API",
        "HTTP 402 payment required; purchase more credits",
    ],
)
def test_provider_quota_and_credit_exhaustion_is_infrastructure_invalid(detail: str) -> None:
    result = AgentResult(
        provider="codex",
        is_error=True,
        subtype="turn.failed",
        errors=[detail],
        session_id=None,
        result="",
        num_turns=0,
        total_cost_usd=None,
        duration_ms=1,
        usage={},
    )
    assert _provider_exhaustion_error(result) is not None


def test_transient_rate_limit_and_acumen_caps_are_not_provider_exhaustion() -> None:
    throttled = AgentResult(
        provider="claude",
        is_error=True,
        subtype="error",
        errors=["429 rate limit exceeded; retry after 2 seconds"],
        session_id=None,
        result="",
        num_turns=0,
        total_cost_usd=None,
        duration_ms=1,
        usage={},
    )
    capped = replace(throttled, subtype="error_max_budget_usd", errors=["acumen stopped at its spending limit"])
    assert _provider_exhaustion_error(throttled) is None
    assert _provider_exhaustion_error(capped) is None


def test_provider_exhaustion_result_is_diagnostic_not_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    @asynccontextmanager
    async def fake_sandbox(*_args, **_kwargs):
        yield box

    async def exhausted(*_args, **_kwargs) -> AgentResult:
        return AgentResult(
            provider="codex",
            is_error=True,
            subtype="turn.failed",
            errors=["insufficient_quota: purchase more credits"],
            session_id=None,
            result="",
            num_turns=0,
            total_cost_usd=None,
            duration_ms=10,
            usage={},
        )

    monkeypatch.setattr("acumen.runner.sandbox", fake_sandbox)
    monkeypatch.setattr("acumen.runner.run_agent", exhausted)
    monkeypatch.setattr("acumen.runner._collect_artifacts", lambda *_args: None)
    monkeypatch.setattr("acumen.runner.agent_version", lambda _provider: "test")
    directory = tmp_path / "run"
    outcome = asyncio.run(
        run_once(
            key=RunKey(arm="noskill", split="test", model="gpt-5.6-sol", task_id="task", rep=1),
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
            run_dir=directory,
            model="gpt-5.6-sol",
            max_turns=1,
            max_usd=1.0,
        )
    )

    persisted = json.loads((directory / "result.json").read_text())
    assert outcome.reason == "provider_exhausted"
    assert persisted["valid"] is False
    assert not is_complete(directory)


def _run_once_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: AgentResult,
    answer: str | None,
    prices: PriceTable | None = None,
) -> dict:
    """Drive ``run_once`` against a canned agent result and return the persisted payload."""
    box_root = tmp_path / "box"
    box_root.mkdir(exist_ok=True)
    box = Sandbox(
        root=box_root,
        home=tmp_path / "home",
        config_dir=tmp_path / "agent-home",
        env={},
        authenticated=True,
        provider=result.provider,
    )

    @asynccontextmanager
    async def fake_sandbox(*_args, **_kwargs):
        yield box

    async def fake_run_agent(*_args, **_kwargs) -> AgentResult:
        return result

    def collect(_box: Sandbox, directory: Path) -> None:
        if answer is not None:
            (directory / "answer.md").write_text(answer)

    monkeypatch.setattr("acumen.runner.sandbox", fake_sandbox)
    monkeypatch.setattr("acumen.runner.run_agent", fake_run_agent)
    monkeypatch.setattr("acumen.runner._collect_artifacts", collect)
    monkeypatch.setattr("acumen.runner.render_trajectory", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("acumen.runner.agent_version", lambda _provider: "test")

    model = "gpt-5.6-sol" if result.provider == "codex" else "claude-opus-5"
    run_dir = tmp_path / "run"
    asyncio.run(
        run_once(
            key=RunKey(arm="noskill", split="test", model=model, task_id="task", rep=1),
            task=Task(id="task", train=TaskSplit("prompt", "SPI1"), test=TaskSplit("prompt", "SPI1")),
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
            model=model,
            max_turns=1,
            max_usd=1.0,
            prices=prices,
        )
    )
    return json.loads((run_dir / "result.json").read_text())


def _capped_result(subtype: str, provider: str = "codex") -> AgentResult:
    return AgentResult(
        provider=provider,  # type: ignore[arg-type]
        is_error=True,
        subtype=subtype,
        errors=["acumen stopped the run at its turn cap"],
        session_id="thread-1",
        result="",
        num_turns=1,
        total_cost_usd=None,
        duration_ms=10,
        usage={"input_tokens": 100, "output_tokens": 10},
    )


@pytest.mark.parametrize("subtype", ["error_max_turns", "error_max_budget_usd"])
@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_capped_run_is_graded_on_the_answer_it_managed_to_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subtype: str, provider: str
) -> None:
    """A cap says the run was cut short, not that the answer it already wrote is worthless.

    Discarding it scored a correct answer as a failure, which is the opposite of what the run
    measured. The cap stays on the record in ``subtype`` and in the agent's errors.
    """
    payload = _run_once_with(tmp_path, monkeypatch, result=_capped_result(subtype, provider), answer="SPI1\n")

    assert payload["success"] is True
    assert payload["reason"] == "ok"
    assert payload["subtype"] == subtype
    assert payload["valid"] is True


@pytest.mark.parametrize(("answer", "reason"), [("PU.1", "wrong_answer"), (None, "max_turns"), ("", "max_turns")])
def test_capped_run_without_a_usable_answer_still_fails_on_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str | None, reason: str
) -> None:
    """No answer.md, or an empty one, leaves nothing to grade — the cap is the whole story."""
    payload = _run_once_with(tmp_path, monkeypatch, result=_capped_result("error_max_turns"), answer=answer)

    assert payload["success"] is False
    assert payload["reason"] == reason


def test_a_broken_run_is_not_rescued_by_an_answer_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only caps defer to the grade. A failed sandbox means the run itself cannot be trusted."""
    broken = replace(
        _capped_result("error_max_turns"),
        subtype="error_sandbox",
        errors=["the Codex sandbox could not complete the run's file operations: bwrap: ..."],
    )
    payload = _run_once_with(tmp_path, monkeypatch, result=broken, answer="SPI1\n")

    assert payload["success"] is False
    assert payload["reason"] == "error"


def test_capped_codex_run_records_the_tokens_it_spent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The usage recovered from the rollout has to reach result.json, not just AgentResult."""
    prices = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)},
        fetched_as_of="2026-08-04",
    )
    payload = _run_once_with(
        tmp_path, monkeypatch, result=_capped_result("error_max_turns"), answer="SPI1\n", prices=prices
    )

    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 10
    # 100 fresh input at $5/M plus 10 output at $30/M.
    assert payload["cost_usd"] == pytest.approx(100 * 5.0e-6 + 10 * 30.0e-6)
    assert payload["cost_available"] is True


async def _no_preflight(*_args: object, **_kwargs: object) -> dict[str, str]:
    """Stand in for `preflight_models`: no model is unreachable, so the matrix proceeds."""
    return {}


def test_run_matrix_cancels_remaining_cells_when_provider_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(id="quota", train=TaskSplit("prompt", "OK"), test=TaskSplit("prompt", "OK"))
    planned = [
        PlannedRun(
            key=RunKey(arm="noskill", split="test", model="gpt-5.6-sol", task_id=f"task_{index}", rep=1),
            task=task,
            model="gpt-5.6-sol",
            max_turns=1,
            max_usd=1.0,
        )
        for index in range(3)
    ]
    cancelled: list[str] = []

    async def fake_run_once(*, key: RunKey, **_kwargs: object) -> RunOutcome:
        if key.task_id == "task_0":
            await asyncio.sleep(0)  # let sibling Codex cells become in-flight
            return RunOutcome(
                key=key,
                success=False,
                reason="provider_exhausted",
                payload={"agent": "codex", "auth_mode": "session", "error": "usage limit reached"},
            )
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(key.task_id)

    monkeypatch.setattr("acumen.bench.run_once", fake_run_once)
    monkeypatch.setattr("acumen.bench.preflight_models", _no_preflight)
    target = Target(
        source="target",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )

    with pytest.raises(BenchmarkInvalidError, match="benchmark invalid"):
        asyncio.run(run_matrix(planned, target=target, runs_root=tmp_path / "runs", max_concurrency=3))
    assert set(cancelled) == {"task_1", "task_2"}


def test_run_matrix_continues_other_provider_after_one_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(id="mixed", train=TaskSplit("prompt", "OK"), test=TaskSplit("prompt", "OK"))

    def planned(model: str, task_id: str) -> PlannedRun:
        return PlannedRun(
            key=RunKey(arm="noskill", split="test", model=model, task_id=task_id, rep=1),
            task=task,
            model=model,
            max_turns=1,
            max_usd=1.0,
        )

    matrix = [
        planned("gpt-5.6-sol", "codex_quota"),
        planned("claude-opus-5", "claude_first"),
        planned("gpt-5.6-sol", "codex_queued"),
        planned("claude-opus-5", "claude_queued"),
    ]
    completed: list[str] = []
    cancelled: list[str] = []

    async def fake_run_once(*, key: RunKey, **_kwargs: object) -> RunOutcome:
        if key.task_id == "codex_quota":
            return RunOutcome(
                key=key,
                success=False,
                reason="provider_exhausted",
                payload={"agent": "codex", "auth_mode": "session", "error": "usage limit reached"},
            )
        if key.model.startswith("gpt-"):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(key.task_id)
        await asyncio.sleep(0)
        completed.append(key.task_id)
        return RunOutcome(key=key, success=True, reason="ok", payload={})

    monkeypatch.setattr("acumen.bench.run_once", fake_run_once)
    monkeypatch.setattr("acumen.bench.preflight_models", _no_preflight)
    target = Target(
        source="target",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc",
        pkg_name="target",
        pkg_version="1",
    )

    with pytest.raises(BenchmarkInvalidError, match="other providers continued"):
        asyncio.run(run_matrix(matrix, target=target, runs_root=tmp_path / "runs", max_concurrency=2))
    assert completed == ["claude_first", "claude_queued"]
    assert cancelled == []  # the queued Codex cell was cancelled before it was submitted


def test_codex_transcript_renders_its_own_html(tmp_path: Path) -> None:
    """A saved Codex event stream maps to a trajectory and renders through the unified renderer."""
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


def test_capped_codex_transcript_shows_the_usage_the_run_recovered(tmp_path: Path) -> None:
    """A capped run has no turn.completed to read usage from, and its footer said nothing.

    It is the run whose spend is most worth seeing, so the caller passes in what it recorded.
    """
    jsonl = tmp_path / "capped.jsonl"
    jsonl.write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread-9"})
        + "\n"
        + json.dumps({"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "hi"}})
        + "\n"
    )
    html = tmp_path / "capped.html"
    assert render_codex_transcript(jsonl, html, {"input_tokens": 1_234, "output_tokens": 56}) is True
    assert "1234" in html.read_text()

    bare = tmp_path / "bare.html"
    assert render_codex_transcript(jsonl, bare) is True
    assert "1234" not in bare.read_text()


def test_codex_transcript_renders_without_events(tmp_path: Path) -> None:
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("not json\n")
    html = tmp_path / "empty.html"
    assert render_codex_transcript(jsonl, html) is True
    assert "no events" in html.read_text()
    assert render_codex_transcript(tmp_path / "missing.jsonl", tmp_path / "missing.html") is False


def test_codex_transcript_renders_the_prompt_as_a_leading_block(tmp_path: Path) -> None:
    """Codex's event stream never echoes the prompt, so the caller carries it back to be shown."""
    events = [
        {"type": "thread.started", "thread_id": "thread-9"},
        {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "done"}},
    ]
    html = tmp_path / "with_prompt.html"
    assert render_codex_events(events, html, prompt="Report the top <gene>.") is True
    body = html.read_text()
    # The prompt is rendered, labelled, and ahead of the agent's first message.
    assert '<div class="item prompt">' in body
    assert body.index("Report the top") < body.index("done")
    # It is escaped, never injected as markup.
    assert "&lt;gene&gt;" in body and "<gene>" not in body

    # An empty prompt renders no block at all, so a run without one is unchanged.
    bare = tmp_path / "no_prompt.html"
    assert render_codex_events(events, bare, prompt="   ") is True
    assert '<div class="item prompt">' not in bare.read_text()


def test_from_codex_events_maps_items_to_a_trajectory() -> None:
    """Each Codex item becomes an agent step; a command carries its output as an observation."""
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"id": "a", "type": "reasoning", "text": "planning"}},
        {"type": "item.completed", "item": {"id": "b", "type": "agent_message", "text": "on it"}},
        {
            "type": "item.completed",
            "item": {
                "id": "c",
                "type": "command_execution",
                "command": "ls",
                "aggregated_output": "x\n",
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "d",
                "type": "command_execution",
                "command": "boom",
                "aggregated_output": "no",
                "exit_code": 2,
            },
        },
        {"type": "item.started", "item": {"id": "e", "type": "command_execution", "command": "sleep 9"}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
    ]
    traj = from_codex_events(events, prompt="do it")
    assert traj.harness == "codex" and traj.session_id == "thread-1" and traj.prompt == "do it"
    # reasoning, message, two commands, and the unfinished command all survive as agent steps.
    assert [s.source for s in traj.steps] == ["agent"] * 5
    assert traj.steps[0].reasoning == "planning"
    assert traj.steps[1].text == "on it"
    ok = traj.steps[2]
    assert ok.tool_calls[0].name == "command_execution" and ok.tool_calls[0].arguments["command"] == "ls"
    assert ok.observations[0].exit_code == 0 and ok.observations[0].is_error is False
    assert traj.steps[3].observations[0].is_error is True  # non-zero exit
    assert traj.steps[4].incomplete is True  # started but never completed
    assert traj.usage == Metrics(input_tokens=100, output_tokens=20)


def test_from_codex_events_prefers_completed_over_started_and_carries_errors() -> None:
    """An item that started then completed renders once (completed), and errors are collected."""
    events = [
        {"type": "item.started", "item": {"id": "a", "type": "command_execution", "command": "go"}},
        {"type": "item.completed", "item": {"id": "a", "type": "command_execution", "command": "go", "exit_code": 0}},
        {"type": "turn.failed", "message": "provider exploded"},
    ]
    traj = from_codex_events(events)
    assert len(traj.steps) == 1 and traj.steps[0].incomplete is False
    assert traj.errors == ("provider exploded",)


def test_from_claude_records_maps_messages_and_attaches_observations() -> None:
    """Assistant blocks become an agent step; a tool_result attaches to the call that made it."""
    records = [
        {"type": "user", "message": {"content": "the prompt"}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5",
                "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 2},
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"file_path": "x.py"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "file body"}]},
        },
    ]
    traj = from_claude_records(records)
    assert traj.harness == "claude-code" and traj.model == "claude-opus-5"
    # The first plain-string user message is the prompt, not a step.
    assert traj.prompt == "the prompt"
    assert len(traj.steps) == 1
    step = traj.steps[0]
    assert step.reasoning == "hmm" and step.text == "reading"
    assert step.tool_calls[0].name == "Read" and step.tool_calls[0].arguments == {"file_path": "x.py"}
    # The observation landed on the step that issued the matching call.
    assert step.observations[0].call_id == "call-1" and step.observations[0].content == "file body"
    assert traj.usage == Metrics(input_tokens=10, output_tokens=3, cached_tokens=2)


def test_render_trajectory_and_json_roundtrip(tmp_path: Path) -> None:
    """One renderer serves any trajectory, and to_dict/write_trajectory_json produce the artifact."""
    traj = Trajectory(
        harness="codex",
        session_id="s1",
        model="gpt-5.6-sol",
        prompt="find the <gene>",
        steps=(
            Step(index=1, source="agent", text="working"),
            Step(
                index=2,
                source="agent",
                tool_calls=(ToolCall(call_id="c", name="command_execution", arguments={"command": "run"}),),
                observations=(Observation(call_id="c", content="oops", is_error=True, exit_code=1),),
            ),
        ),
        usage=Metrics(input_tokens=7),
    )
    html = tmp_path / "t.html"
    assert render_trajectory(traj, html) is True
    body = html.read_text()
    assert "find the &lt;gene&gt;" in body and "<gene>" not in body  # prompt shown and escaped
    assert "working" in body and "run" in body and "exit 1" in body and "failed" in body
    assert "gpt-5.6-sol" in body and "thread s1" in body and "7" in body

    out = tmp_path / "trajectory.json"
    assert write_trajectory_json(traj, out) is True
    data = json.loads(out.read_text())
    assert data["schema"] == traj.schema and data["harness"] == "codex" and data["prompt"] == "find the <gene>"
    assert data["steps"][1]["observations"][0]["exit_code"] == 1
    assert data["usage"] == {"input_tokens": 7}


def test_render_trajectory_toggles_tool_calls_and_renders_markdown(tmp_path: Path) -> None:
    """Tool calls collapse into a <details> toggle; agent text renders as markdown."""
    traj = Trajectory(
        harness="codex",
        steps=(
            Step(index=1, source="agent", text="Here is a **bold** claim and `code`.\n\n- one\n- two"),
            Step(
                index=2,
                source="agent",
                reasoning="thinking hard",
                tool_calls=(ToolCall(call_id="c1", name="command_execution", arguments={"command": "ls -la"}),),
                observations=(Observation(call_id="c1", content="files", exit_code=0),),
            ),
            Step(
                index=3,
                source="agent",
                tool_calls=(ToolCall(call_id="c2", name="command_execution", arguments={"command": "boom"}),),
                observations=(Observation(call_id="c2", content="nope", is_error=True, exit_code=1),),
            ),
        ),
    )
    html = tmp_path / "t.html"
    assert render_trajectory(traj, html) is True
    body = html.read_text()
    # Markdown: bold, inline code, and a list are rendered as HTML, not shown as raw syntax.
    assert "<strong>bold</strong>" in body and "<code>code</code>" in body
    assert "<li>one</li>" in body and "<li>two</li>" in body
    # The successful command lives in a collapsed toggle that pairs it with its output.
    assert '<details class="tool"><summary>' in body
    assert "ls -la" in body and "files" in body
    # Reasoning is tucked into its own toggle.
    assert "<summary>reasoning</summary>" in body and "thinking hard" in body
    # A failed command's toggle is opened so the error is visible without a click.
    assert '<details class="tool" open>' in body and "nope" in body


def test_render_trajectory_renders_gfm_tables(tmp_path: Path) -> None:
    """A GitHub-flavored pipe table in agent text renders as an HTML table, not raw pipes."""
    text = "Summary:\n\n| id | score |\n| :-- | --: |\n| `a` | 1 |\n| `b` | 2 |\n\nDone."
    traj = Trajectory(harness="codex", steps=(Step(index=1, source="agent", text=text),))
    html = tmp_path / "t.html"
    assert render_trajectory(traj, html) is True
    body = html.read_text()
    assert "<table>" in body and "<th" in body
    assert '<th style="text-align:left">id</th>' in body  # alignment from :--
    assert '<td style="text-align:right"><code>' not in body  # code col is the first (left)
    assert '<td style="text-align:right">1</td>' in body  # score col is right-aligned from --:
    assert "<code>a</code>" in body and "<code>b</code>" in body  # inline markdown inside cells
    assert "<p>| id" not in body  # the pipe rows are not left as paragraph text


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
    package_root = Path(acumen.__file__).parent
    for path in sorted(package_root.rglob("*.py")):
        if any(part.startswith(".") for part in path.relative_to(package_root).parts):
            continue  # editor caches and other hidden paths are not package source
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


def test_claude_cache_writes_retain_five_minute_and_one_hour_classes() -> None:
    usage = normalize_usage(
        {
            "input_tokens": 100,
            "cache_creation_input_tokens": 3_000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1_000,
                "ephemeral_1h_input_tokens": 2_000,
            },
        }
    )
    rates = Rates(
        input=4,
        cached_input=0.4,
        cache_write=5,
        output=20,
        cache_write_5m=5,
        cache_write_1h=8,
    )

    assert usage.cache_write == 3_000
    assert (usage.cache_write_5m, usage.cache_write_1h) == (1_000, 2_000)
    assert price_run(usage, rates) == pytest.approx((100 * 4 + 1_000 * 5 + 2_000 * 8) / 1_000_000)


def test_inference_is_canonical_while_the_provider_figure_and_its_delta_survive() -> None:
    # The SDK total includes nested agents while the parent usage block can be smaller, so the
    # two figures are not measuring the same work; the delta is what makes that visible.
    cost = resolve_cost(provider_cost_usd=1.25, inferred_cost_usd=0.40)
    assert cost.cost_usd == 0.40
    assert cost.cost_source == "inferred"
    assert cost.provider_cost_usd == 1.25
    assert cost.cost_delta_usd == pytest.approx(0.85)
    assert cost.cost_delta_pct == pytest.approx(2.125)


def test_a_provider_figure_never_substitutes_for_missing_inference() -> None:
    """An unpriced model is unpriced, however many dollars the provider reported.

    Taking the provider's figure would put that one run on a basis no other run in the pass
    is on, which is a silent wrong number rather than a visible gap.
    """
    unpriced = resolve_cost(1.25, None)
    assert unpriced.cost_usd is None
    assert unpriced.cost_source == "unavailable"
    assert unpriced.available is False
    assert unpriced.provider_cost_usd == 1.25
    assert unpriced.cost_delta_usd is None

    codex = resolve_cost(None, 0.40)
    assert (codex.cost_usd, codex.cost_source, codex.available) == (0.40, "inferred", True)

    unavailable = resolve_cost(None, None)
    assert unavailable.cost_usd is None
    assert unavailable.cost_source == "unavailable"
    assert unavailable.available is False


def test_price_run_bills_each_cache_class_at_its_own_rate() -> None:
    """Collapsing the split and billing it all at the base input rate overcharges ~3.6x.

    That is the whole reason the breakdown is carried: a benchmark agent's input is
    mostly cache reads, billed at a tenth of the base rate.
    """
    usage = Usage(input=200_000, cache_read=180_000, cache_write=15_000, output=4_000)
    rates = Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=25.0)
    # 5k @ $5 + 180k @ $0.50 + 15k @ $6.25 + 4k @ $25, per million.
    assert price_run(usage, rates) == pytest.approx(0.30875)
    collapsed = (usage.input * rates.input + usage.output * rates.output) / 1_000_000
    assert collapsed > price_run(usage, rates) * 3


def test_price_usage_prices_codex_meta_agent_tokens_when_provider_cost_is_absent() -> None:
    prices = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)},
        fetched_as_of="2026-08-04",
    )
    cost = price_usage(
        {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000},
        model="gpt-5.6-sol",
        provider="codex",
        prices=prices,
    )

    assert cost == pytest.approx(35.0)


def test_price_run_leaves_an_unpriced_model_unpriced_rather_than_free() -> None:
    """``None`` is not ``0.0`` — a model with no rates must not read as a free one."""
    assert PriceTable().lookup("some-local-llm") is None
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

    @asynccontextmanager
    async def fake_sandbox(*_args, **_kwargs):
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
    monkeypatch.setattr("acumen.runner.render_trajectory", lambda *_args, **_kwargs: False)
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
    assert persisted["cost_source"] == "unavailable"
    assert persisted["provider_cost_usd"] is None
    assert persisted["inferred_cost_usd"] is None
    assert persisted["provider"] == "openai"
    assert persisted["backend"] == "codex_cli"
    assert persisted["agent"] == "codex"


def test_benchmark_persists_inferred_cost_and_records_the_claude_sdk_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box_root = tmp_path / "box"
    box_root.mkdir()
    box = Sandbox(
        root=box_root,
        home=tmp_path / "home",
        config_dir=tmp_path / "claude-home",
        env={},
        authenticated=True,
        provider="claude",
    )

    @asynccontextmanager
    async def fake_sandbox(*_args, **_kwargs):
        yield box

    async def fake_run_agent(*_args, **_kwargs) -> AgentResult:
        return AgentResult(
            provider="claude",
            is_error=False,
            subtype="success",
            errors=None,
            session_id=None,
            result="done",
            num_turns=3,
            # SDK totals include nested agents; the parent usage block below does not.
            total_cost_usd=1.25,
            duration_ms=100,
            usage={"input_tokens": 1_000, "output_tokens": 0},
        )

    def collect(_box: Sandbox, directory: Path) -> None:
        (directory / "answer.md").write_text("OK")

    monkeypatch.setattr("acumen.runner.sandbox", fake_sandbox)
    monkeypatch.setattr("acumen.runner.run_agent", fake_run_agent)
    monkeypatch.setattr("acumen.runner._collect_artifacts", collect)
    monkeypatch.setattr("acumen.runner.agent_version", lambda _provider: "test")

    run_dir = tmp_path / "run"
    asyncio.run(
        run_once(
            key=RunKey(arm="noskill", split="test", model="claude-opus-5", task_id="task", rep=1),
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
            model="claude-opus-5",
            max_turns=5,
            max_usd=2.0,
            prices=PriceTable(
                fetched={"claude-opus-5": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=25.0)},
                fetched_as_of="2026-08-04",
            ),
        )
    )

    persisted = json.loads((run_dir / "result.json").read_text())
    assert persisted["cost_usd"] == pytest.approx(0.005)
    assert persisted["cost_source"] == "inferred"
    assert persisted["provider_cost_usd"] == 1.25
    assert persisted["inferred_cost_usd"] == pytest.approx(0.005)
    assert persisted["cost_delta_usd"] == pytest.approx(1.245)
    assert persisted["cost_delta_pct"] == pytest.approx(249.0)
    assert persisted["provider"] == "anthropic"
    assert persisted["backend"] == "claude_agent_sdk"
    assert persisted["agent"] == "claude"


def test_price_table_matches_provider_qualified_and_differently_cased_ids() -> None:
    """A gateway or proxy names the same model differently; it is still that model."""
    published = Rates(input=7.0, cached_input=0.7, cache_write=8.75, output=35.0)
    table = PriceTable(fetched={"gpt-5.6-sol": published}, fetched_as_of="2026-09-15")

    assert table.rates("openai/gpt-5.6-sol") is published
    assert table.rates("GPT-5.6-Sol") is published
    assert table.lookup("no-such-model") is None


def test_price_table_prices_a_dated_snapshot_at_its_family_rate() -> None:
    """Providers publish one rate per family and never list the dated snapshot IDs.

    Pinning a snapshot is the reproducible thing to do, and ``acumen init`` scaffolds one,
    so without this fallback a default project would bench an entirely unpriced model.
    """
    published = Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=5.0)
    table = PriceTable(fetched={"claude-haiku-4-5": published}, fetched_as_of="2026-09-15")

    assert table.rates("claude-haiku-4-5-20251001") is published
    assert table.rates("anthropic/claude-haiku-4-5-20251001") is published
    # A version suffix that is not a date is part of the name, not a snapshot.
    assert table.lookup("claude-haiku-4-5-turbo") is None


def test_price_table_ranks_config_over_fetched() -> None:
    """A pinned rate is the operator stating what *they* are billed, which no page knows.

    With nothing baked into the package, these are the only two sources there are: what
    the provider publishes today, and what the operator says overrides it.
    """
    negotiated = Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=2.0)
    published = Rates(input=7.0, cached_input=0.7, cache_write=8.75, output=35.0)
    table = PriceTable(
        overrides={"claude-opus-5": negotiated},
        fetched={"claude-opus-5": published, "gpt-5.6-sol": published},
        fetched_as_of="2026-09-15",
    )

    assert table.lookup("claude-opus-5").rates is negotiated
    assert table.lookup("gpt-5.6-sol").rates is published
    # A model no layer prices is unpriced, never guessed at from a shipped default.
    assert table.lookup("claude-sonnet-4-6") is None


def test_an_empty_price_table_prices_nothing_rather_than_guessing() -> None:
    """No rates ship with the package, so an unfetched table is honestly empty.

    The alternative — a compiled-in table — is wrong from whatever date the providers next
    move their prices, and since each run's cost is frozen when written, that wrongness is
    stored rather than corrected on the next run.
    """
    assert PriceTable().lookup("claude-opus-5") is None
    assert price_usage({"input_tokens": 10_000, "output_tokens": 1_000}, model="claude-opus-5") is None


def test_price_table_dates_a_fetched_rate_but_not_an_operators_own() -> None:
    """A config rate has no verification date, and inventing one would claim currency."""
    table = PriceTable(
        overrides={"my-gateway": Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=2.0)},
        fetched={"gpt-5.6-sol": Rates(input=7.0, cached_input=0.7, cache_write=8.75, output=35.0)},
        fetched_as_of="2026-09-15",
    )

    assert price_provenance(table.lookup("my-gateway")) == {
        "price_rates": {
            "input": 1.0,
            "cached_input": 0.1,
            "cache_write": 1.25,
            "output": 2.0,
            "cache_write_5m": None,
            "cache_write_1h": None,
        },
        "price_source": "config",
        "price_rates_as_of": None,
    }
    fetched = price_provenance(table.lookup("gpt-5.6-sol"))
    assert (fetched["price_source"], fetched["price_rates_as_of"]) == ("fetched", "2026-09-15")
    # An unpriced model still produces the keys, so results have one shape to read.
    assert price_provenance(None) == {"price_rates": None, "price_source": None, "price_rates_as_of": None}


def test_two_passes_months_apart_keep_the_rates_each_was_priced_by() -> None:
    """The point of recording rates per run: a later pass must not restate an earlier one.

    Same model, same tokens, two months and a price rise apart. Both figures stay valid
    because each carries the rate that produced it, which is what lets one report mix them.
    """
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    august = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=30.0)},
        fetched_as_of="2026-08-04",
    )
    october = PriceTable(
        fetched={"gpt-5.6-sol": Rates(input=8.0, cached_input=0.8, cache_write=10.0, output=48.0)},
        fetched_as_of="2026-10-04",
    )

    then = price_usage(usage, model="gpt-5.6-sol", provider="codex", prices=august)
    now = price_usage(usage, model="gpt-5.6-sol", provider="codex", prices=october)

    assert then == pytest.approx(5.0 + 3.0)
    assert now == pytest.approx(8.0 + 4.8)
    assert price_provenance(august.lookup("gpt-5.6-sol"))["price_rates_as_of"] == "2026-08-04"
    assert price_provenance(october.lookup("gpt-5.6-sol"))["price_rates_as_of"] == "2026-10-04"


def test_fetch_table_dates_the_fetch_and_keeps_config_on_top() -> None:
    """What ``bench`` builds: live rates, dated, with the operator's own still winning."""
    pages = {"anthropic": _ANTHROPIC_MD, "openai": _OPENAI_MD}
    mine = {"gpt-5.6-sol": Rates(input=1.0, cached_input=0.1, cache_write=1.25, output=2.0)}

    table = fetch_table(
        mine,
        today=date(2026, 9, 15),
        fetcher=lambda url: pages["anthropic" if "claude" in url else "openai"],
    )

    assert table.fetched_as_of == "2026-09-15"
    # The page priced this one at $5 input; the operator's own rate still wins.
    assert table.fetched["gpt-5.6-sol"].input == 5.0
    assert table.lookup("gpt-5.6-sol").source == "config"
    assert table.lookup("gpt-5.6-sol").rates.input == 1.0
    # Everything else comes from the fetch, dated by it.
    sonnet = table.lookup("claude-sonnet-5")
    assert (sonnet.source, sonnet.as_of) == ("fetched", "2026-09-15")
    # 2026-09-15 is past the introductory window, so the standard rate applies.
    assert sonnet.rates.input == 3.0


def test_fetch_table_raises_rather_than_pricing_a_pass_from_a_stale_table() -> None:
    """Bench turns this into a refusal: a wrong cost, once frozen into a result, stays wrong."""

    def unreachable(url, **_kwargs):
        raise PriceFeedError(f"could not fetch {url}: timed out")

    with pytest.raises(PriceFeedError, match="could not fetch"):
        fetch_table({}, today=date(2026, 9, 15), fetcher=unreachable)


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
    assert intro["claude-sonnet-5"] == Rates(
        input=2.0,
        cached_input=0.2,
        cache_write=2.5,
        output=10.0,
        cache_write_5m=2.5,
        cache_write_1h=4.0,
    )

    after = parse_anthropic(_ANTHROPIC_MD, today=date(2026, 9, 1))
    assert after["claude-sonnet-5"] == Rates(
        input=3.0,
        cached_input=0.3,
        cache_write=3.75,
        output=15.0,
        cache_write_5m=3.75,
        cache_write_1h=6.0,
    )

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


def test_improver_refuses_provider_exhaustion_evidence(project: Path, make_result) -> None:
    key = RunKey(arm="skill_v1", split="train", model="gpt-5.6-sol", task_id="example_task", rep=1)
    make_result(project / "runs", key, valid=False, reason="provider_exhausted", success=False)

    with pytest.raises(ImproveError, match="infrastructure-invalid benchmark result"):
        collect_train_runs(project / "runs", "skill_v1", load_tasks(project / "tasks.yaml"))


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


def test_sandbox_bash_timeout_outlasts_a_dataset_download(tmp_path: Path) -> None:
    """A fetch that outruns the Bash timeout is backgrounded, not failed, and that is the bug.

    The CLI's 120s default backgrounded nearly every dataset download a benchmark target makes
    (one measured over 300s), and a backgrounded command finishing is what re-enters an agent
    after its run is over. Raising the timeout is the half of the fix that prevents it.
    """
    env = scrubbed_env(config_dir=tmp_path / "cfg", home=tmp_path / "home", auth_mode="api")

    assert int(env["BASH_DEFAULT_TIMEOUT_MS"]) == BASH_DEFAULT_TIMEOUT_MS
    assert int(env["BASH_MAX_TIMEOUT_MS"]) == BASH_MAX_TIMEOUT_MS
    assert int(env["BASH_DEFAULT_TIMEOUT_MS"]) > 300_000
    # An agent can still ask for longer than the default on a fit or a slow test suite.
    assert int(env["BASH_MAX_TIMEOUT_MS"]) > int(env["BASH_DEFAULT_TIMEOUT_MS"])


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


def test_reports_refuse_provider_exhaustion_results(runs_root: Path, model: str, make_result) -> None:
    key = RunKey(arm="noskill", split="train", model=model, task_id="quota", rep=1)
    make_result(runs_root, key, valid=False, reason="provider_exhausted", success=False)

    with pytest.raises(ReportError, match="infrastructure-invalid result"):
        load_results(runs_root)


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


def test_loaded_flags_fill_missing_values_without_object_downcast_warning() -> None:
    values = pd.DataFrame({"skill_loaded": pd.Series([True, None, False], dtype=object)})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        flags = _loaded_flags(values)

    assert flags.dtype == bool
    assert flags.tolist() == [True, False, False]


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


def test_arms_priced_on_different_dates_are_flagged_as_not_cost_comparable(
    runs_root: Path, model: str, make_result
) -> None:
    """Benching arms weeks apart confounds the skill's effect with a change in list price.

    Rates are resolved live and frozen per run, so each figure is true to what it cost —
    but the difference *between* arms then contains both effects, and nothing downstream
    can separate them. The report says so rather than presenting the gap as the skill's.
    """
    for arm, as_of in (("noskill", "2026-08-04"), ("skill_v1", "2026-10-04")):
        for split in ("train", "test"):
            key = RunKey(arm=arm, split=split, model=model, task_id="example_task", rep=1)
            make_result(runs_root, key, price_rates_as_of=as_of, price_source="fetched")

    (note,) = [n for n in _integrity_notes(load_results(runs_root)) if "priced on different dates" in n]
    assert "2026-08-04" in note and "2026-10-04" in note
    assert "not only the skill's effect" in note


def test_one_pass_priced_on_one_date_is_not_flagged(runs_root: Path, model: str, make_result) -> None:
    """The note must stay quiet on the ordinary case, or it teaches people to ignore it."""
    for arm in ("noskill", "skill_v1"):
        for split in ("train", "test"):
            key = RunKey(arm=arm, split=split, model=model, task_id="example_task", rep=1)
            make_result(runs_root, key, price_rates_as_of="2026-08-04", price_source="fetched")

    notes = _integrity_notes(load_results(runs_root))

    assert not any("priced on different dates" in note for note in notes)


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


def test_reports_show_inferred_cost_and_csv_keeps_the_recorded_one(
    runs_root: Path, model: str, make_result, tmp_path: Path
) -> None:
    """A result recorded before inference became canonical still reports on that basis.

    The provider figure stays auditable in the CSV, but nothing displayed is drawn from it:
    it covers one provider only, so a report mixing the two would compare unlike numbers.
    """
    key = RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1)
    make_result(
        runs_root,
        key,
        cost_usd=0.90,
        cost_available=True,
        cost_source="provider",
        provider_cost_usd=0.90,
        inferred_cost_usd=0.20,
    )

    df = load_results(runs_root)
    test = df[df["split"] == "test"]
    assert test.iloc[0]["cost_usd"] == pytest.approx(0.20)
    assert arm_metrics(test).loc[0, "cost"] == pytest.approx(0.20)
    assert "0.200" in _runs_table_html(test, tmp_path)
    assert "0.900" not in _runs_table_html(test, tmp_path)

    out_path = tmp_path / "results.html"
    build_report(runs_root, out_path)
    exported = pd.read_csv(out_path.with_suffix(".csv"))
    test_row = exported[exported["split"] == "test"].iloc[0]
    assert test_row["cost_usd"] == pytest.approx(0.20)
    assert test_row["inferred_cost_usd"] == pytest.approx(0.20)
    assert test_row["recorded_cost_usd"] == pytest.approx(0.90)
    assert "provider_cost_usd" not in exported.columns


def test_modern_result_without_inferred_cost_stays_unpriced_in_report(runs_root: Path, model: str, make_result) -> None:
    """A recorded provider estimate must never substitute for missing inference."""
    key = RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1)
    make_result(
        runs_root,
        key,
        cost_usd=0.90,
        cost_available=True,
        cost_source="provider",
        provider_cost_usd=0.90,
        inferred_cost_usd=None,
    )

    test = load_results(runs_root).query("split == 'test'")
    assert pd.isna(test.iloc[0]["cost_usd"])
    assert pd.isna(arm_metrics(test).loc[0, "cost"])


def test_the_unpriced_note_names_the_model_that_could_not_be_priced(
    runs_root: Path, model: str, make_result, tmp_path: Path
) -> None:
    """The fix is a ``prices:`` entry, so the warning has to say which model needs one."""
    for rep, cost in ((1, None), (2, 0.20)):
        key = RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=rep)
        make_result(runs_root, key, cost_usd=cost, cost_available=cost is not None, inferred_cost_usd=cost)

    df = load_results(runs_root)
    note = next(n for n in _integrity_notes(df) if "cost unavailable" in n)
    assert f"no token rates for {model}" in note

    # The warning belongs above the figures it qualifies, not buried beside them.
    rendered = render_report(df, tmp_path)
    assert rendered.index(html.escape(note)) < rendered.index('<section id="overview">')


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


def _frontier_line(figure: plt.Figure) -> plt.Line2D:
    """The drawn Pareto staircase — the only marker-less line on the axes."""
    (line,) = [ln for ln in figure.axes[0].lines if str(ln.get_marker()) == "None"]
    return line


def _frontier(figure: plt.Figure) -> list[tuple[float, float]]:
    """The vertices of the drawn Pareto staircase."""
    return [(float(x), float(y)) for x, y in zip(*_frontier_line(figure).get_data(), strict=True)]


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
    """Two channels, two meanings: an ✕ against a disc for skill, hue left to the model."""
    for arm in ("skill_v1", "skill_v2"):
        make_result(runs_root, RunKey(arm=arm, split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    figure = tradeoff_figure(df, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        markers = {str(line.get_marker()) for line in _model_marks(figure)}
        colors = {to_hex(line.get_color()) for line in _model_marks(figure)}
    finally:
        plt.close(figure)

    # Both versions are the same disc: the arrows tell them apart, not the shape.
    assert markers == {"X", "o"}
    assert colors == {"#3b7ea1"}  # the override reaches the marks, not just the legend


def _arrows(figure: plt.Figure) -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    """Every trajectory arrow as ``(tail, head, colour)`` in data coordinates.

    The arrows are the axes' only patches; the marks and the frontier are lines. Their
    endpoints come off the patch's private pair because the public path is the *shrunk*
    outline, which no longer touches the marks the arrow was asked to join.
    """
    return [
        (patch._posA_posB[0], patch._posA_posB[1], to_hex(patch.get_edgecolor(), keep_alpha=False))
        for patch in figure.axes[0].patches
        if isinstance(patch, FancyArrowPatch)
    ]


def test_tradeoff_arrows_walk_each_model_in_version_order_and_never_cross_models(
    runs_root: Path, model: str, make_result
) -> None:
    """The chain is what says which disc is which version, so it must run within one model."""
    other = "claude-opus-5"
    costs = {(model, "skill_v1"): 0.40, (other, "noskill"): 0.60, (other, "skill_v1"): 0.90}
    for (who, arm), cost in costs.items():
        key = RunKey(arm=arm, split="test", model=who, task_id="example_task", rep=1)
        make_result(runs_root, key, cost_usd=cost)
    df = load_results(runs_root)  # plus the fixture's $0.12 noskill run for `model`

    figure = tradeoff_figure(df, colors=resolve_palette([model, other], {model: "#3b7ea1", other: "#a1553b"}))
    try:
        arrows = _arrows(figure)
    finally:
        plt.close(figure)

    # One hop per model — baseline to v1 — plus the pooled chain's own hop. Nothing joins the
    # $0.12 and $0.60 baselines, or either baseline to the other model's version.
    assert sorted((tail[0], head[0], color) for tail, head, color in arrows) == [
        (pytest.approx(0.12), pytest.approx(0.40), "#3b7ea1"),
        (pytest.approx(0.36), pytest.approx(0.65), "#9b968d"),  # pooled: mean of each arm's runs
        (pytest.approx(0.60), pytest.approx(0.90), "#a1553b"),
    ]


def test_tradeoff_joins_versions_that_landed_on_the_same_result(runs_root: Path, model: str, make_result) -> None:
    """Every hop is drawn and every hop is straight, however close the two marks are.

    Marks this close overlap and hide the arrow between them, which is the honest reading: one
    place, two versions. Bending the hop out to make it visible would draw a path through
    results nobody measured.
    """
    for arm, cost in (("skill_v1", 0.1201), ("skill_v2", 0.1202)):
        key = RunKey(arm=arm, split="test", model=model, task_id="example_task", rep=1)
        make_result(runs_root, key, cost_usd=cost)
    df = load_results(runs_root)  # three arms within a hundredth of a cent of the fixture's run

    figure = tradeoff_figure(df)
    try:
        arrows = figure.axes[0].patches
        curved = [patch for patch in arrows if patch.get_connectionstyle().rad]
    finally:
        plt.close(figure)

    assert len(arrows) == 4  # two hops for the model's own chain, two more for the pooled one
    assert curved == []


def test_tradeoff_arrow_reaches_the_mark_it_points_at(runs_root: Path, model: str, make_result) -> None:
    """The head must land on the target's rim, as squarely as the tail leaves the one behind it.

    matplotlib insets a filled head from the end of its own path, so an arrow given the same
    standoff at both ends stops visibly short of what it points at while its tail sits flush.
    """
    key = RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1)
    make_result(runs_root, key, cost_usd=0.60)
    df = load_results(runs_root)  # the fixture's $0.12 baseline, then this

    figure = tradeoff_figure(df)
    try:
        arrow = next(patch for patch in figure.axes[0].patches if isinstance(patch, FancyArrowPatch))
        drawn = figure.axes[0].transData.transform(arrow.get_path().vertices)
        tail, head = figure.axes[0].transData.transform(arrow._posA_posB)
        reach = [min(float(abs(complex(*(point - centre)))) for point in drawn) for centre in (tail, head)]
    finally:
        plt.close(figure)

    assert reach[1] == pytest.approx(reach[0], abs=0.2)  # both ends stop on their mark's rim


def test_tradeoff_labels_no_mark_in_the_plot(runs_root: Path, model: str, make_result) -> None:
    """Order rides the arrows, so no version has to be named on the panel itself."""
    for arm, cost in (("skill_v1", 0.40), ("skill_v2", 0.80)):
        key = RunKey(arm=arm, split="test", model=model, task_id="example_task", rep=1)
        make_result(runs_root, key, cost_usd=cost)
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        assert _arrows(figure)  # the arms are far enough apart to be joined, so this is a real test
        texts = {text.get_text() for text in figure.axes[0].texts}
    finally:
        plt.close(figure)

    assert texts == {"Pareto frontier"}  # the frontier's own caption, and nothing else


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
        dashes = _frontier_line(figure).get_linestyle()
        y_min, _y_max = figure.axes[0].get_ylim()
        _x_min, x_max = figure.axes[0].get_xlim()
    finally:
        plt.close(figure)

    # Dashed: no run lies along the line, unlike every other path on the panel.
    assert dashes != "-"

    # Both $0.12 runs pass, so the cheapest 100% mark alone survives; the $0.50 failure is
    # dominated and the riser starts at the axis floor. The last leg carries that rate out to
    # the right edge, since nothing dearer beat it either.
    assert vertices == [
        (pytest.approx(0.12), pytest.approx(y_min)),
        (pytest.approx(0.12), pytest.approx(1.0)),
        (pytest.approx(x_max), pytest.approx(1.0)),
    ]


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
        labels = {
            legend.get_title().get_text(): [text.get_text() for text in legend.get_texts()] for legend in figure.legends
        }
        marks = [handle for handle in keys["skill"] if isinstance(handle, plt.Line2D)]
        skill = {handle.get_markerfacecolor() for handle in marks}
        models = {to_hex(handle.get_markerfacecolor()) for handle in keys["model"]}
    finally:
        plt.close(figure)

    assert skill == {"none"}  # shape is the channel, so an outline is the whole handle
    assert models == {"#3b7ea1"}  # the model key is the colour key, and keeps its fill
    # Two shapes and the arrow that orders them, whatever the version count.
    assert labels["skill"] == ["No skill", "Skill", "next version"]


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
    xs, ys = _pareto_steps([(0.1, 0.8), (0.2, 0.9)], y_min=0.5, x_max=0.4)

    # The last leg runs flat to the right edge: past the dearest frontier point, paying more
    # cannot buy less, so that stretch is dominated ground rather than nothing at all.
    assert list(zip(xs, ys, strict=True)) == [(0.1, 0.5), (0.1, 0.8), (0.2, 0.8), (0.2, 0.9), (0.4, 0.9)]


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


def test_arm_marker_separates_skill_from_no_skill_only() -> None:
    """Shape says skill or not; which version is the arrows' job, so it never runs out of shapes."""
    assert _arm_marker("noskill") == "X"
    assert {_arm_marker(f"skill_v{n}") for n in (1, 2, 3, 9, 30)} == {"o"}


# --- the runs table --------------------------------------------------------------------


def test_runs_table_marks_every_column_sortable_but_the_transcript_link(runs_root: Path, tmp_path: Path) -> None:
    """The link column has no ordering, so offering to sort on it would be a dead control."""
    table = _runs_table_html(load_results(runs_root), tmp_path)
    headings = re.findall(r"<th([^>]*)>([^<]+)</th>", table)

    sortable = {text for attrs, text in headings if "data-sortable" in attrs}
    inert = {text for attrs, text in headings if "data-sortable" not in attrs}
    assert inert == {"transcript"}
    assert "cost / run $" in sortable and "total tok" in sortable


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


# --- checking task ground truth --------------------------------------------------------


def _fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped directory whose interpreter is the one running the tests."""
    venv = tmp_path / "venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if os.name == "nt" else "python")).symlink_to(sys.executable)
    return venv


def _task(task_id: str, train: str = "TRAIN", test: str = "TEST", **kwargs) -> Task:
    return Task(
        id=task_id,
        train=TaskSplit(prompt="do the thing", answer=train),
        test=TaskSplit(prompt="do the other thing", answer=test),
        **kwargs,
    )


#: One reproducer per failure mode the check has to tell apart, keyed by its filename.
_REPRODUCERS = {
    "good-train.py": 'from pathlib import Path\nprint("noise on stdout")\nPath("answer.md").write_text("TRAIN")\n',
    "good-test.py": 'from pathlib import Path\nPath("answer.md").write_text("TEST")\n',
    "shaky-train.py": 'from pathlib import Path\nPath("answer.md").write_text("SOMETHING ELSE")\n',
    "shaky-test.py": 'from pathlib import Path\nPath("answer.md").write_text("**TEST**")\n',
    "broken-train.py": 'import sys\nprint("no module named scanpy", file=sys.stderr)\nraise SystemExit(3)\n',
    "broken-test.py": "pass\n",  # runs clean, writes nothing
    "slow-train.py": "import time\n\ntime.sleep(30)\n",
}


def test_needs_script_defaults_to_true_and_must_be_a_bool() -> None:
    entry = {"id": "t", "train": {"prompt": "p", "answer": "a"}, "test": {"prompt": "p", "answer": "b"}}

    (task,) = parse_tasks({"tasks": [entry]})
    assert task.needs_script is True

    (opted_out,) = parse_tasks({"tasks": [{**entry, "needs_script": False}]})
    assert opted_out.needs_script is False

    # `1` and `"no"` would silently mean something the author did not write.
    with pytest.raises(TaskError, match="needs_script must be true or false"):
        parse_tasks({"tasks": [{**entry, "needs_script": 1}]})


def test_dump_tasks_writes_needs_script_only_when_it_is_off() -> None:
    text = dump_tasks([_task("code"), _task("prose", needs_script=False)])

    assert text.count("needs_script") == 1
    assert "needs_script: false" in text
    # Round-tripping is what the generator's output is validated by, so it must survive.
    assert [t.needs_script for t in parse_tasks(yaml.safe_load(text))] == [True, False]


def test_check_tells_every_failure_mode_apart(tmp_path: Path) -> None:
    """The point of the command: name *why* a task's ground truth is not trustworthy.

    A wrong recorded answer, a script that crashes, one that writes nothing, one that hangs and
    a missing script all mean different repairs, so they must not collapse into "failed".
    """
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    for name, body in _REPRODUCERS.items():
        (scripts / name).write_text(body)
    tasks = [
        _task("good"),
        _task("shaky"),
        _task("broken"),
        _task("slow"),
        _task("prose", needs_script=False),
    ]

    results = check_tasks(
        tasks,
        scripts_root=scripts,
        python=_fake_venv(tmp_path).joinpath("bin", "python"),
        timeout=1,
        jobs=4,
    )

    assert {(r.task_id, r.split): r.status for r in results} == {
        ("good", "train"): "ok",
        ("good", "test"): "ok",
        ("shaky", "train"): "wrong_answer",
        ("shaky", "test"): "format_error",  # right content, formatting the exact match rejects
        ("broken", "train"): "error",
        ("broken", "test"): "no_answer",
        ("slow", "train"): "timeout",
        ("slow", "test"): "missing",
        ("prose", "train"): "skipped",
        ("prose", "test"): "skipped",
    }
    wrong = next(r for r in results if (r.task_id, r.split) == ("shaky", "train"))
    assert (wrong.answer, wrong.expected) == ("SOMETHING ELSE", "TRAIN")
    crashed = next(r for r in results if (r.task_id, r.split) == ("broken", "train"))
    assert crashed.returncode == 3
    assert "scanpy" in crashed.error_tail
    # stdout is diagnostics, never the answer: a passing script may print whatever it likes.
    assert next(r for r in results if (r.task_id, r.split) == ("good", "train")).answer == "TRAIN"


def test_check_invokes_the_venv_interpreter_by_its_own_path(tmp_path: Path) -> None:
    """The target venv's ``bin/python`` must not be resolved to the interpreter it links to.

    Python finds its ``pyvenv.cfg`` — and therefore the venv's site-packages — from the path it
    was invoked by. Following the symlink runs the base interpreter, where the target package is
    not installed, so every single reproducer would fail with ModuleNotFoundError.
    """
    venv = _fake_venv(tmp_path)
    interpreter = venv / "bin" / "python"
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    (scripts / "venvcheck-train.py").write_text(
        "import sys\nfrom pathlib import Path\n\nPath('answer.md').write_text(sys.executable)\n"
    )

    results = check_tasks(
        [_task("venvcheck", train=str(interpreter))], scripts_root=scripts, python=interpreter, splits=["train"]
    )

    assert [r.status for r in results] == ["ok"], f"ran {results[0].answer} instead of {interpreter}"


def test_check_runs_each_reproducer_in_its_own_empty_directory(tmp_path: Path) -> None:
    """Two scripts writing answer.md must not read each other's, and neither may litter."""
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    for split, answer in (("train", "TRAIN"), ("test", "TEST")):
        (scripts / f"solo-{split}.py").write_text(
            "from pathlib import Path\n"
            "assert not list(Path.cwd().iterdir()), f'not an empty directory: {list(Path.cwd().iterdir())}'\n"
            f'Path("answer.md").write_text({answer!r})\n'
            'Path("scratch.csv").write_text("stray output")\n'
        )
    before = sorted(p.name for p in scripts.iterdir())

    results = check_tasks(
        [_task("solo")],
        scripts_root=scripts,
        python=_fake_venv(tmp_path).joinpath("bin", "python"),
        jobs=2,
    )

    assert [r.status for r in results] == ["ok", "ok"]
    assert sorted(p.name for p in scripts.iterdir()) == before
    assert not (tmp_path / "scratch.csv").exists()


def test_summarize_checks_scores_only_the_tasks_that_need_a_script(tmp_path: Path) -> None:
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    (scripts / "good-train.py").write_text(_REPRODUCERS["good-train.py"])
    (scripts / "good-test.py").write_text(_REPRODUCERS["good-test.py"])
    (scripts / "shaky-train.py").write_text(_REPRODUCERS["shaky-train.py"])
    tasks = [_task("good"), _task("shaky"), _task("prose", needs_script=False)]

    results = check_tasks(tasks, scripts_root=scripts, python=_fake_venv(tmp_path).joinpath("bin", "python"), jobs=1)
    summary = summarize_checks(results, tasks)

    assert (summary.n_splits, summary.n_code_splits, summary.n_non_code_splits) == (6, 4, 2)
    assert (summary.n_with_script, summary.pct_with_script) == (3, 75.0)
    assert (summary.n_ok, summary.pct_reproduced) == (2, 50.0)
    # `good` reproduced on both splits; `shaky` on neither. `prose` is not counted either way.
    assert (summary.n_code_tasks, summary.n_tasks_reproduced) == (2, 1)
    assert summary.ok is False
    assert summary.by_status["skipped"] == 2


def test_summary_of_an_all_prose_task_set_is_not_reported_as_zero_percent() -> None:
    """An empty denominator must read as "nothing to measure", never as "nothing worked"."""
    tasks = [_task("prose", needs_script=False)]
    results = check_tasks(tasks, scripts_root=Path("nowhere"), python=Path(sys.executable))

    summary = summarize_checks(results, tasks)

    assert summary.pct_reproduced is None
    assert summary.pct_tasks_reproduced is None
    assert summary.ok is True


def test_orphan_scripts_flags_a_reproducer_no_task_claims(tmp_path: Path) -> None:
    """A renamed task id leaves a script that looks like coverage but is never run."""
    scripts = tmp_path / "tasks"
    scripts.mkdir()
    for name in ("kept-train.py", "kept-test.py", "renamed_away-train.py", "notes.txt"):
        (scripts / name).write_text("pass\n")

    assert orphan_scripts([_task("kept")], scripts) == [scripts / "renamed_away-train.py"]
    assert orphan_scripts([_task("kept")], tmp_path / "absent") == []


def test_script_path_builds_the_name_rather_than_splitting_it() -> None:
    """Task ids may contain '-', so a filename can never be parsed back apart safely."""
    assert script_path(Path("tasks"), "bulk-tf-activity", "train") == Path("tasks/bulk-tf-activity-train.py")


def test_select_tasks_rejects_an_unknown_id() -> None:
    tasks = [_task("a"), _task("b")]

    assert [t.id for t in select_tasks(tasks, ["b"])] == ["b"]
    assert [t.id for t in select_tasks(tasks, None)] == ["a", "b"]
    with pytest.raises(CheckError, match="unknown task ids"):
        select_tasks(tasks, ["c"])


def test_import_probe_resolves_the_import_name_from_the_distribution() -> None:
    """`prepare_target` proves the distribution is installed; only an import proves it works.

    The two names differ often enough to matter (`pyyaml` imports as `yaml`), so probing the
    distribution name verbatim would report a healthy target as broken.
    """
    ok, detail = import_probe(Path(sys.executable), "pyyaml")
    assert ok
    assert detail.startswith("yaml ")

    broken, why = import_probe(Path(sys.executable), "not-installed-anywhere")
    assert not broken
    assert "No module named" in why


def test_harvest_keeps_only_the_reproducers_the_tasks_declare(tmp_path: Path) -> None:
    """What the generator leaves behind is not automatically ground truth worth keeping."""
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in ("code-train.py", "code-test.py", "prose-train.py", "leftover.py"):
        (staged / name).write_text("pass\n")
    scripts_root = tmp_path / "project" / "tasks"
    tasks = [_task("code"), _task("prose", needs_script=False), _task("nocode")]

    harvest = harvest_scripts(tasks, staged, scripts_root)

    assert sorted(p.name for p in harvest.scripts) == ["code-test.py", "code-train.py"]
    assert sorted(p.name for p in scripts_root.iterdir()) == ["code-test.py", "code-train.py"]
    # A split that expected a script and got none is a reported gap, not a silent pass.
    assert harvest.missing == (("nocode", "train"), ("nocode", "test"))
    # Neither a script for a prose task nor a file matching no task is kept.
    assert harvest.unexpected == ("leftover.py", "prose-train.py")


def test_harvest_leaves_no_directory_behind_when_there_is_nothing_to_keep(tmp_path: Path) -> None:
    """The project is only touched once there is a reproducer to put in it.

    ``generate_tasks`` harvests after the generated tasks validate, so a rejected generation
    must leave the project exactly as it was — not an empty ``tasks/`` implying coverage.
    """
    scripts_root = tmp_path / "project" / "tasks"

    harvest = harvest_scripts([_task("code")], tmp_path / "staged-that-never-existed", scripts_root)

    assert harvest.scripts == ()
    assert harvest.missing == (("code", "train"), ("code", "test"))
    assert not scripts_root.exists()


# --- reviewing task coherence ----------------------------------------------------------


def _result(task_id: str, split: str, status: str = "ok", **kwargs) -> CheckResult:
    """A CheckResult as `check_task_split` builds them: `expected` is always the recorded answer."""
    default = "TRAIN" if split == "train" else "TEST"
    return CheckResult(
        task_id=task_id,
        split=split,
        status=status,
        script=kwargs.pop("script", None),
        # `answer` is what the script wrote, so it is None when none ran; `expected` comes from
        # tasks.yaml and is therefore always present.
        answer=kwargs.pop("answer", default),
        expected=kwargs.pop("expected", default),
        seconds=1.0,
        **kwargs,
    )


def test_parse_reviews_maps_verdicts_onto_the_rows_that_were_reviewed() -> None:
    rows = [_result("bulk", "train"), _result("bulk", "test")]
    raw = {
        "reviews": [
            {"task": "bulk", "split": "train", "verdict": "ok"},
            {
                "task": "bulk",
                "split": "test",
                "verdict": "mismatch",
                "issue": "prompt says ascending; script and answer are descending",
                "fix": "say descending in the prompt",
            },
        ]
    }

    verdicts, warnings = parse_reviews(raw, rows)

    assert [(v.task_id, v.split, v.status) for v in verdicts] == [
        ("bulk", "train", "ok"),
        ("bulk", "test", "mismatch"),
    ]
    assert verdicts[1].issue.startswith("prompt says ascending")
    assert verdicts[1].fix == "say descending in the prompt"
    assert warnings == []


def test_parse_reviews_enforces_brevity_rather_than_trusting_it() -> None:
    """The prompt asks for one short clause; the table stays readable when it does not comply."""
    rows = [_result("bulk", "train")]
    raw = {
        "reviews": [
            {
                "task": "bulk",
                "split": "train",
                "verdict": "mismatch",
                "issue": "the prompt\n  spans several\n  lines " + "x" * 400,
                "fix": "short",
            }
        ]
    }

    (verdict,), _warnings = parse_reviews(raw, rows)

    assert "\n" not in verdict.issue
    assert len(verdict.issue) <= MAX_NOTE_CHARS
    assert verdict.issue.startswith("the prompt spans several lines")


def test_parse_reviews_degrades_instead_of_discarding_a_good_deterministic_run() -> None:
    """A review is a judgement layered on results that already stand on their own.

    So every way the agent can be sloppy has to survive as a warning plus an `unreviewed` row,
    never as an exception that throws away half an hour of reproducer runs.
    """
    rows = [_result("bulk", "train"), _result("bulk", "test"), _result("scell", "train")]
    raw = {
        "reviews": [
            {"task": "bulk", "split": "train", "verdict": "mismatch"},  # flagged, no reason
            {"task": "bulk", "split": "test", "verdict": "ok"},
            {"task": "bulk", "split": "test", "verdict": "mismatch"},  # duplicate
            {"task": "ghost", "split": "train", "verdict": "ok"},  # not under review
            {"task": "scell", "split": "train", "verdict": "probably fine"},  # not a verdict
            "not a mapping at all",
        ]
    }

    verdicts, warnings = parse_reviews(raw, rows)

    assert [v.status for v in verdicts] == ["mismatch", "ok", "unreviewed"]
    joined = " | ".join(warnings)
    assert "bulk/train was flagged with no reason given" in joined
    assert "reviewed twice; keeping the first" in joined
    assert "'ghost'" in joined and "not under review" in joined
    assert "not 'ok' or 'mismatch'" in joined
    assert "is not a mapping" in joined
    assert "no verdict for scell/train" in joined


def test_parse_reviews_treats_a_verdict_file_with_no_reviews_as_nothing_reviewed() -> None:
    rows = [_result("bulk", "train")]

    for raw in ({}, {"reviews": "nope"}, []):
        verdicts, warnings = parse_reviews(raw, rows)
        assert [v.status for v in verdicts] == ["unreviewed"]
        assert "wrote no 'reviews' list" in warnings[0]


def test_parse_reviews_drops_commentary_on_a_task_that_holds_together() -> None:
    """An `ok` verdict has nothing to say, so a stray note on one is not shown."""
    rows = [_result("bulk", "train")]
    raw = {"reviews": [{"task": "bulk", "split": "train", "verdict": "ok", "issue": "looks nice", "fix": "none"}]}

    (verdict,), warnings = parse_reviews(raw, rows)

    assert (verdict.issue, verdict.fix) == (None, None)
    assert warnings == []


def test_review_packet_copies_everything_and_points_nowhere_near_the_project(tmp_path: Path) -> None:
    """The reviewer must not be able to reach tasks.yaml or the real tasks/ tree.

    It knows the test-split answers, so a path back into the project is the one thing that
    could turn it into a leak. Copies, and a digest that names scripts by filename only.
    """
    project = tmp_path / "project"
    scripts = project / "tasks"
    scripts.mkdir(parents=True)
    (scripts / "bulk-train.py").write_text("# the real reproducer\nprint('hi')\n")
    (project / "tasks.yaml").write_text("tasks: []\n")
    tasks = [_task("bulk", train="FOXD1", test="FOXO1"), _task("prose", needs_script=False)]
    results = [
        _result("bulk", "train", script=scripts / "bulk-train.py", answer="FOXD1", expected="FOXD1"),
        _result("bulk", "test", "wrong_answer", answer="SOMETHING", expected="FOXO1"),
        _result("prose", "train", "skipped", answer=None, expected="BSD"),
    ]
    packet = tmp_path / "work" / "review"

    written = write_packet(packet, tasks, results)

    assert [(r.task_id, r.split) for r in written] == [("bulk", "train"), ("bulk", "test"), ("prose", "train")]
    digest = (packet / PACKET_DIGEST).read_text()
    # Every split's prompt, recorded answer and outcome are in the digest.
    assert "## bulk / train" in digest and "## prose / train" in digest
    assert "do the thing" in digest and "FOXD1" in digest and "BSD" in digest
    assert "reproduced the recorded answer exactly" in digest
    assert "produced `SOMETHING`, NOT the recorded answer" in digest
    # The script is copied in, and named relatively.
    assert (packet / PACKET_SCRIPTS / "bulk-train.py").read_text() == "# the real reproducer\nprint('hi')\n"
    assert f"`{PACKET_SCRIPTS}/bulk-train.py`" in digest
    # Nothing anywhere in the packet points back at the project.
    for path in packet.rglob("*"):
        if path.is_file():
            assert str(project) not in path.read_text(), f"{path} leaks a path into the project"
    # Train and test differ on purpose; the digest has to say so or the reviewer flags the design.
    assert "differ on purpose" in digest


def test_review_packet_explains_a_split_with_no_script(tmp_path: Path) -> None:
    """A missing or unrunnable reproducer still leaves prompt vs answer worth reviewing."""
    packet = tmp_path / "review"
    tasks = [_task("gone"), _task("broken")]
    results = [
        _result("gone", "train", "missing", answer=None),
        _result("broken", "train", "error", answer=None, returncode=1),
    ]

    write_packet(packet, tasks, results)

    digest = (packet / PACKET_DIGEST).read_text()
    assert "There is no reproducer for this split" in digest
    assert "The script failed to run" in digest
    assert not list((packet / PACKET_SCRIPTS).iterdir())


def _review_target(tmp_path: Path) -> Target:
    """A target whose source and venv exist, so the reviewer's read roots resolve."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    return Target(
        source="target",
        ref="main",
        src_dir=src,
        venv_dir=_fake_venv(tmp_path),
        commit="abc1234",
        pkg_name="target",
        pkg_version="1.0",
    )


def _agent_result() -> AgentResult:
    return AgentResult(
        is_error=False,
        subtype="success",
        result="done",
        num_turns=6,
        duration_ms=1000,
        total_cost_usd=0.25,
        usage={"input_tokens": 20_000, "output_tokens": 1_000},
        session_id="s",
        provider="claude",
        errors=None,
    )


#: Rates for the review agent's model, so its run has a cost to infer at all.
_REVIEW_PRICES = PriceTable(
    fetched={"claude-opus-5": Rates(input=5.0, cached_input=0.5, cache_write=6.25, output=25.0)},
    fetched_as_of="2026-08-04",
)


def test_review_tasks_hands_the_agent_a_packet_and_reads_its_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    async def fake_run_agent(prompt, *, options, on_event=None):
        seen["prompt"] = prompt
        seen["read_dirs"] = options.read_dirs
        seen["discover_skills"] = options.discover_skills
        # The packet the agent is told to read must actually be there before it runs.
        digest = options.cwd / PACKET_DIRNAME / PACKET_DIGEST
        seen["digest"] = digest.read_text()
        (options.cwd / REVIEW_FILE).write_text(
            json.dumps(
                {
                    "reviews": [
                        {"task": "bulk", "split": "train", "verdict": "ok"},
                        {
                            "task": "bulk",
                            "split": "test",
                            "verdict": "mismatch",
                            "issue": "prompt says ascending; script is descending",
                            "fix": "say descending",
                        },
                    ]
                }
            )
        )
        return _agent_result()

    monkeypatch.setattr("acumen.review.run_agent", fake_run_agent)
    monkeypatch.setattr("acumen.review.build_agent_env", lambda **_kwargs: {})
    target = _review_target(tmp_path)
    tasks = [_task("bulk")]
    results = [_result("bulk", "train"), _result("bulk", "test")]

    review = asyncio.run(
        review_tasks(
            cfg=parse_config({"repo": "/tmp/target-demo", "meta_model": "claude-opus-5"}),
            target=target,
            tasks=tasks,
            results=results,
            prices=_REVIEW_PRICES,
        )
    )

    assert review.status_for("bulk", "train") == "ok"
    assert review.status_for("bulk", "test") == "mismatch"
    assert [v.task_id for v in review.flagged] == ["bulk"]
    # The agent's own $0.25 is recorded but not reported: the review is priced from the table,
    # like every other run, so one basis covers both providers.
    assert review.cost_usd == pytest.approx(20_000 * 5.0e-6 + 1_000 * 25.0e-6)
    assert review.turns == 6
    assert review.warnings == ()
    # The prompt points at the staged packet, and the reviewer gets the package like the drafter.
    assert PACKET_DIRNAME in str(seen["prompt"])
    assert seen["read_dirs"] == (target.src_dir, target.venv_dir)
    # A skill shipped by the target must not colour its own tasks' review.
    assert seen["discover_skills"] is False
    assert "## bulk / train" in seen["digest"]


def test_review_tasks_refuses_to_call_an_unreviewed_task_set_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that writes nothing must raise, not return a table of silent passes."""

    async def writes_nothing(prompt, *, options, on_event=None):
        return _agent_result()

    monkeypatch.setattr("acumen.review.run_agent", writes_nothing)
    monkeypatch.setattr("acumen.review.build_agent_env", lambda **_kwargs: {})

    with pytest.raises(ReviewError, match=f"did not write {REVIEW_FILE}"):
        asyncio.run(
            review_tasks(
                cfg=parse_config({"repo": "/tmp/target-demo"}),
                target=_review_target(tmp_path),
                tasks=[_task("bulk")],
                results=[_result("bulk", "train")],
            )
        )


def test_review_tasks_reports_unparseable_output_as_a_failed_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def writes_junk(prompt, *, options, on_event=None):
        (options.cwd / REVIEW_FILE).write_text("this is not JSON {")
        return _agent_result()

    monkeypatch.setattr("acumen.review.run_agent", writes_junk)
    monkeypatch.setattr("acumen.review.build_agent_env", lambda **_kwargs: {})

    with pytest.raises(ReviewError, match="not readable JSON"):
        asyncio.run(
            review_tasks(
                cfg=parse_config({"repo": "/tmp/target-demo"}),
                target=_review_target(tmp_path),
                tasks=[_task("bulk")],
                results=[_result("bulk", "train")],
            )
        )


def test_meta_model_defaults_to_the_first_benchmark_model() -> None:
    """The meta-agent model defaults to the first benchmark model, so one `models:` line configures the lot."""
    cfg = parse_config({"repo": "/tmp/target-demo", "models": ["gpt-5.6-sol", "claude-opus-5"]})
    assert cfg.meta_model == "gpt-5.6-sol"

    named = parse_config({"repo": "/tmp/target-demo", "meta_model": "claude-haiku-4-5-20251001"})
    assert named.meta_model == "claude-haiku-4-5-20251001"
