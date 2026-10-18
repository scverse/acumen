"""Tests for the hold-out guard — the structural boundary every improve agent runs behind.

The guard is pure (:func:`find_test_access`), so the three things it must deny are checked
directly: the test split of any task, every split of a held-out task (a CV fold), and any denied
directory (the lockbox, other folds). Paths reach it as structured fields or as tokens of a shell
command, and both routes are covered.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from acumen.evidence import find_test_access, make_test_guard


def _runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for arm in ("noskill", "skill_v1"):
        for split in ("train", "test"):
            for task in ("keep", "held"):
                (runs / arm / split / "m" / task / "rep_1").mkdir(parents=True)
    return runs


def test_test_split_is_always_denied(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    assert find_test_access("Read", {"file_path": str(runs / "skill_v1/test/m/keep/rep_1/result.json")}, runs)
    assert find_test_access("Read", {"file_path": str(runs / "skill_v1/train/m/keep/rep_1/result.json")}, runs) is None
    assert find_test_access("Bash", {"command": f"cat {runs}/noskill/test/m/keep/rep_1/answer.md"}, runs)
    assert find_test_access("Bash", {"command": f"ls {runs}/noskill/train"}, runs) is None


def test_held_out_task_is_denied_in_every_split(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    held = frozenset({"held"})
    train_held = str(runs / "skill_v1/train/m/held/rep_1/result.json")
    train_keep = str(runs / "skill_v1/train/m/keep/rep_1/result.json")
    # Without a fold, train runs of any task are readable; with the fold, the held-out task's are not.
    assert find_test_access("Read", {"file_path": train_held}, runs) is None
    assert find_test_access("Read", {"file_path": train_held}, runs, held_out_ids=held) == train_held
    assert find_test_access("Read", {"file_path": train_keep}, runs, held_out_ids=held) is None
    # Through a shell command too, and via a relative path resolved against cwd-independent root.
    assert find_test_access(
        "Bash", {"command": f"grep -r answer {runs}/skill_v1/train/m/held"}, runs, held_out_ids=held
    )
    # Listing the split root is not itself a held-out path (the task component is absent).
    assert find_test_access("Bash", {"command": f"ls {runs}/skill_v1/train/m"}, runs, held_out_ids=held) is None


def test_denied_directories_are_off_limits_wholesale(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    lockbox = tmp_path / "lockbox"
    (lockbox / "sub").mkdir(parents=True)
    deny = (lockbox.resolve(),)
    assert find_test_access("Read", {"file_path": str(lockbox / "tasks.yaml")}, runs, deny_dirs=deny)
    assert find_test_access("Read", {"file_path": str(lockbox / "sub" / "x")}, runs, deny_dirs=deny)
    assert find_test_access("Bash", {"command": f"cat {lockbox}/manifest.json"}, runs, deny_dirs=deny)
    # A token that merely resembles the name but resolves elsewhere is fine.
    assert find_test_access("Read", {"file_path": str(tmp_path / "lockbox2" / "x")}, runs, deny_dirs=deny) is None
    # And an unrelated Bash command with no paths at all passes.
    assert find_test_access("Bash", {"command": "echo hello"}, runs, deny_dirs=deny) is None


def test_hook_denies_with_a_reason(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    hook = make_test_guard(runs, held_out_ids=["held"], deny_dirs=[tmp_path / "lockbox"]).hooks[0]
    denied = asyncio.run(
        hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(runs / "skill_v1/train/m/held/rep_1/x")}}, None, None
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "held-out" in denied["hookSpecificOutput"]["permissionDecisionReason"]
    allowed = asyncio.run(
        hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(runs / "skill_v1/train/m/keep/rep_1/x")}}, None, None
        )
    )
    assert allowed == {}
