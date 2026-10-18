"""Tests for the experiment health check and the verdict: the sign test is the one piece of real logic."""

from __future__ import annotations

import json
from pathlib import Path

from acumen.cli import main
from acumen.evolve_analyze import sign_test, voided
from acumen.evolve_status import main as status_main


def test_sign_test_counts_only_paired_disagreements() -> None:
    base = {f"t{i}": i < 5 for i in range(10)}  # 5 pass, 5 fail
    assert sign_test(base, dict(base)) == (0, 0, 1.0, 10)
    better = {**base, **{f"t{i}": True for i in range(10)}}  # recovers all 5 failures
    gained, lost, p, n = sign_test(base, better)
    assert (gained, lost, n) == (5, 0, 10) and p < 0.07
    # One lost, one gained is pure noise; a voided run drops out of the pairing.
    assert sign_test(base, {**base, "t0": False, "t9": True})[2] == 1.0
    assert sign_test(base, {**base, "t9": None})[3] == 9


def test_platform_killed_runs_are_voided_but_real_failures_are_not() -> None:
    assert voided({"reason": "error", "error": "API Error: Can't reach the API server (ENOTFOUND)"})
    assert voided({"reason": "connection_error", "valid": False})
    assert not voided({"reason": "error", "error": "Reached maximum number of turns (40)"})
    assert not voided({"reason": "wrong_answer", "error": None})


def _result(root: Path, name: str, **fields: object) -> None:
    path = root / "skill_v1" / "test" / "m" / name / "rep_1" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_id": name, **fields}))


def test_status_separates_pending_invalid_runs_from_mislabelled_failures(tmp_path: Path, capsys) -> None:
    _result(tmp_path, "ok", success=True, skill_name="s", skill_loaded=True)
    _result(tmp_path, "paused", success=False, reason="connection_error", valid=False)
    assert status_main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "invalid-and-pending 1" in out and "platform-killed-but-recorded 0" in out

    # A platform error recorded as an ordinary failure is the silent one, and is flagged.
    _result(tmp_path, "mislabelled", success=False, reason="error", error="429 rate limit")
    assert status_main([str(tmp_path)]) == 1
    assert "PROBLEM" in capsys.readouterr().out


def test_status_and_verdict_are_cli_subcommands(tmp_path: Path, capsys) -> None:
    assert main(["status", str(tmp_path / "missing")]) == 2
    arm = tmp_path / "arm"
    _result(arm, "a", success=True)
    _result(arm, "b", success=False, reason="wrong_answer")
    assert main(["verdict", f"base={arm / 'skill_v1'}", f"other={arm / 'skill_v1'}"]) == 0
    assert "per arm" in capsys.readouterr().out
