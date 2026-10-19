"""Tests for CV folds and the lockbox — pure partitioning, determinism, and the write-once boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from acumen.folds import (
    FoldError,
    check_disjoint,
    load_lockbox_tasks,
    make_folds,
    read_lockbox,
    split_lockbox,
    write_lockbox,
)
from acumen.taskgen import dump_tasks
from acumen.tasks import Task, TaskSplit

IDS = [f"t{i}" for i in range(10)]


def _task(task_id: str) -> Task:
    return Task(id=task_id, train=TaskSplit("p", "A"), test=TaskSplit("p", "B"))


def test_make_folds_partitions_every_task_exactly_once() -> None:
    folds = make_folds(IDS, 3, seed=0)
    assert [f.index for f in folds] == [1, 2, 3]
    held = [t for f in folds for t in f.held_out]
    assert sorted(held) == sorted(IDS)  # each task held out exactly once
    assert {len(f.held_out) for f in folds} == {3, 4}  # 10 tasks over 3 folds
    for f in folds:
        assert set(f.optimize) | set(f.held_out) == set(IDS)
        assert not set(f.optimize) & set(f.held_out)


def test_make_folds_is_deterministic_in_seed_and_order_independent() -> None:
    a = make_folds(IDS, 3, seed=7)
    b = make_folds(list(reversed(IDS)), 3, seed=7)  # file order must not matter
    c = make_folds(IDS, 3, seed=8)
    assert a == b
    assert a != c


def test_make_folds_rejects_degenerate_requests() -> None:
    with pytest.raises(FoldError, match="at least 2"):
        make_folds(IDS, 1)
    with pytest.raises(FoldError, match="needs a held-out task"):
        make_folds(["a", "b"], 3)
    with pytest.raises(FoldError, match="unique"):
        make_folds(["a", "a", "b"], 2)


def test_split_lockbox_holds_back_a_fraction_deterministically() -> None:
    working, lock = split_lockbox(IDS, 0.2, seed=0)
    assert len(lock) == 2 and len(working) == 8
    assert not set(working) & set(lock) and set(working) | set(lock) == set(IDS)
    assert split_lockbox(IDS, 0.2, seed=0) == (working, lock)
    assert split_lockbox(IDS, 0.2, seed=1) != (working, lock)
    # At least one task is always locked, and never all of them.
    assert len(split_lockbox(["a", "b", "c"], 0.01)[1]) == 1
    with pytest.raises(FoldError, match="no working task"):
        split_lockbox(["a", "b"], 0.9)
    with pytest.raises(FoldError, match="fraction"):
        split_lockbox(IDS, 1.5)


def test_lockbox_is_written_once_and_verified_on_read(tmp_path: Path) -> None:
    box_dir = tmp_path / "lockbox"
    tasks = [_task("t1"), _task("t2")]
    box = write_lockbox(box_dir, tasks, seed=0, fraction=0.2, dump=dump_tasks)
    assert box.task_ids == ("t1", "t2") and box.digest.startswith("sha256:")

    again = read_lockbox(box_dir)
    assert again == box
    assert [t.id for t in load_lockbox_tasks(again)] == ["t1", "t2"]

    with pytest.raises(FoldError, match="written once"):
        write_lockbox(box_dir, tasks, seed=1, fraction=0.2, dump=dump_tasks)

    # Editing the held-back tasks after the fact is detected.
    box.tasks_path.write_text(dump_tasks([_task("t1")]))
    with pytest.raises(FoldError, match="modified since"):
        read_lockbox(box_dir)


def test_check_disjoint_refuses_overlap(tmp_path: Path) -> None:
    box = write_lockbox(tmp_path / "lb", [_task("t9")], seed=0, fraction=0.1, dump=dump_tasks)
    check_disjoint([_task("t1"), _task("t2")], box)  # fine
    with pytest.raises(FoldError, match="t9"):
        check_disjoint([_task("t1"), _task("t9")], box)


def test_read_lockbox_missing(tmp_path: Path) -> None:
    with pytest.raises(FoldError, match="acumen lockbox"):
        read_lockbox(tmp_path / "nope")
