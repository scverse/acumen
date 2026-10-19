"""Tests for transient-failure handling: a platform limit must never become a recorded run.

When the subscription session limit (or a rate limit) hits mid-pass, the first live CV loop wrote
54 "failed" result.json files in minutes — runs that said nothing about the tasks, that resume would
then skip forever, and that the improver would read as evidence. These tests pin the fix: such an
outcome is recorded invalid (so resume retries it), and the loop turns the pause into a
``TransientLimitError`` so no caller scores a partial matrix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from acumen import loop as loop_mod
from acumen.bench import BenchmarkInvalidError, build_matrix
from acumen.config import Config
from acumen.env import Target
from acumen.runner import RunOutcome
from acumen.tasks import Task, TaskSplit
from acumen.transient import TransientLimitError, is_transient

MODEL = "claude-sonnet-5"


def test_is_transient_recognises_platform_limits() -> None:
    assert is_transient(
        "ResultError: Claude Code returned an error result: You've hit your session limit · resets 8:40pm"
    )
    assert is_transient("API rate limit exceeded")
    assert is_transient("overloaded_error: Overloaded")
    assert is_transient("HTTP 529 upstream")
    assert not is_transient("ValueError: no answer file")
    assert not is_transient("the agent wrote the wrong answer")


def test_is_transient_recognises_a_dropped_network() -> None:
    """A machine that loses the network says nothing about the task (round 2, draft v4/d3)."""
    assert is_transient(
        "ResultError: Claude Code returned an error result: API Error: Can't reach the API "
        "server — check your internet or DNS (ENOTFOUND) (exit code: 1)"
    )
    assert is_transient("FetchError: request to https://api.anthropic.com failed, reason: ECONNRESET")
    assert is_transient("Error: socket hang up")
    assert is_transient("HTTP 503 Service Unavailable")
    assert not is_transient("the notebook could not be found in the repository")


def _tasks(n: int) -> list[Task]:
    return [Task(id=f"t{i}", train=TaskSplit("p", "A"), test=TaskSplit("p", "B")) for i in range(n)]


def test_platform_refusal_reads_errors_never_the_agents_answer() -> None:
    from acumen.agents import AgentResult
    from acumen.runner import _platform_refusal

    def result(*, text: str = "", errors: list[str] | None = None) -> AgentResult:
        return AgentResult(
            provider="claude",
            is_error=bool(errors),
            subtype="error_during_execution" if errors else "success",
            errors=errors,
            session_id=None,
            result=text,
            num_turns=1,
            total_cost_usd=None,
            duration_ms=1,
            usage={},
        )

    assert _platform_refusal(result(errors=["API Error: 529 overloaded_error"])) is not None
    assert _platform_refusal(None, "Claude AI session limit reached") is not None
    # A task *about* rate limiting must not be misread as the platform refusing the run.
    assert _platform_refusal(result(text="the endpoint returns 429 rate limit headers")) is None


def test_bench_turns_a_provider_pause_into_a_transient_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider-scoped invalid pass is a pause (rerun resumes it); a refused sandbox host is not."""
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=1)
    target = Target(
        source="x", ref="main", src_dir=tmp_path, venv_dir=tmp_path / "venv", commit="c", pkg_name="p", pkg_version="1"
    )
    skill_stub = type("S", (), {"version": "v1", "hash": "h"})()
    key = build_matrix(cfg, _tasks(1), skill=None, splits=["test"])[0].key

    def raising(reason: str):
        async def fake_run_matrix(*_a, **_k):
            raise BenchmarkInvalidError(RunOutcome(key=key, success=False, reason=reason, payload={}))

        return fake_run_matrix

    for reason, expect in (("connection_error", TransientLimitError), ("sandbox_blocked", BenchmarkInvalidError)):
        monkeypatch.setattr(loop_mod, "run_matrix", raising(reason))
        with pytest.raises(expect):
            asyncio.run(
                loop_mod._bench(
                    cfg=cfg,
                    target=target,
                    skill=skill_stub,
                    tasks=_tasks(1),
                    runs_root=tmp_path / "runs",
                    splits=["test"],
                    auth_mode="api",
                    task_ids=None,
                    max_concurrency=1,
                    on_start=None,
                    on_done=None,
                )
            )
