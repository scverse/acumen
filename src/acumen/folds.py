"""Cross-validation folds over tasks, and the lockbox — the two boundaries that make a score honest.

Today's train/test split is *within* a task: two variants of one analysis. That measures whether a
skill memorised an answer, not whether the rulebook generalises to analyses it never saw. Two
structures fix that, both defined here as pure functions over task ids so they can be tested
without an agent in sight:

* **Folds.** The working tasks are partitioned into ``k`` folds. In each fold the rulebook is
  improved from the *optimize* tasks' evidence only and scored on the *held-out* tasks — analyses
  the improve agent never saw. Averaged over folds, that is a cross-validated estimate of what the
  improvement procedure buys on unseen analyses. The partition is deterministic in ``seed`` so a
  resumed or re-run loop reproduces the same folds.

* **The lockbox.** A fraction of tasks set aside *before* any of this starts, written to their own
  directory, and scored exactly once at the very end (P7). Every selection the loop makes — which
  rulebook version to carry, when to stop — is made on CV scores, so those scores are optimistic by
  construction (selection leakage). The lockbox is the number that was never selected on. Its
  directory is denied to every improve agent by a ``PreToolUse`` guard, and the loop refuses to run
  on a working set that overlaps it, so the boundary is structural rather than a promise.

Assignment is by task **id** (sorted, then shuffled by seed), never by position in the file, so
reordering ``tasks.yaml`` does not silently move a task across a boundary.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen.tasks import Task, TaskError, load_tasks

LOCKBOX_TASKS = "tasks.yaml"
LOCKBOX_MANIFEST = "manifest.json"


class FoldError(ValueError):
    """Raised when folds or a lockbox cannot be built or verified."""


@dataclass(frozen=True)
class Fold:
    """One CV fold: the tasks the rulebook may learn from, and the ones it is scored on."""

    index: int
    optimize: tuple[str, ...]
    held_out: tuple[str, ...]


def _shuffled(task_ids: Sequence[str], seed: int) -> list[str]:
    ids = sorted(set(task_ids))
    if len(ids) != len(task_ids):
        raise FoldError("task ids must be unique")
    random.Random(seed).shuffle(ids)
    return ids


def make_folds(task_ids: Sequence[str], k: int, seed: int = 0) -> list[Fold]:
    """Partition task ids into ``k`` folds, deterministically in ``seed``.

    Ids are sorted then shuffled with a seeded RNG and dealt round-robin, so every fold's held-out
    set has ``n // k`` or ``n // k + 1`` tasks and the assignment depends only on the id set and
    the seed. Each fold's ``optimize`` set is every other task.

    Raises
    ------
    FoldError
        If ``k < 2`` (no held-out possible) or there are fewer tasks than folds.
    """
    if k < 2:
        raise FoldError(f"k must be at least 2, got {k}")
    ids = _shuffled(task_ids, seed)
    if len(ids) < k:
        raise FoldError(f"cannot make {k} folds from {len(ids)} tasks — each fold needs a held-out task")
    buckets: list[list[str]] = [[] for _ in range(k)]
    for i, task_id in enumerate(ids):
        buckets[i % k].append(task_id)
    folds: list[Fold] = []
    for i, held in enumerate(buckets):
        held_set = set(held)
        folds.append(
            Fold(
                index=i + 1,
                optimize=tuple(t for t in sorted(ids) if t not in held_set),
                held_out=tuple(sorted(held)),
            )
        )
    return folds


def split_lockbox(task_ids: Sequence[str], fraction: float, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split ids into ``(working, lockbox)``; the lockbox gets ``round(fraction * n)``, at least 1.

    Deterministic in ``seed`` like :func:`make_folds` (and independent of it: a different seed for
    the folds does not move the lockbox). Both halves come back sorted.
    """
    if not 0 < fraction < 1:
        raise FoldError(f"lockbox fraction must be in (0, 1), got {fraction}")
    ids = _shuffled(task_ids, seed)
    n_lock = max(1, round(fraction * len(ids)))
    if n_lock >= len(ids):
        raise FoldError(f"a lockbox of {n_lock} would leave no working task out of {len(ids)}")
    lock = sorted(ids[:n_lock])
    lock_set = set(lock)
    return [t for t in sorted(ids) if t not in lock_set], lock


@dataclass(frozen=True)
class Lockbox:
    """A written lockbox: where it lives and which task ids it holds back."""

    directory: Path
    task_ids: tuple[str, ...]
    seed: int
    fraction: float
    #: sha256 of the lockbox ``tasks.yaml`` as written — so a later read can tell it was not edited.
    digest: str

    @property
    def tasks_path(self) -> Path:
        """The held-back ``tasks.yaml`` inside the lockbox directory."""
        return self.directory / LOCKBOX_TASKS


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_lockbox(directory: Path, tasks: Sequence[Task], *, seed: int, fraction: float, dump) -> Lockbox:
    """Write the held-back tasks and a manifest into ``directory``; refuse if one is already there.

    Written once: a lockbox that could be regenerated with a different seed after the loop has
    run would be no lockbox at all. ``dump`` is the tasks serializer
    (:func:`acumen.taskgen.dump_tasks`), passed in so this module stays free of the agent stack.
    """
    if directory.exists():
        raise FoldError(f"{directory} already exists — a lockbox is written once and never replaced")
    directory.mkdir(parents=True)
    path = directory / LOCKBOX_TASKS
    path.write_text(dump(list(tasks)))
    box = Lockbox(
        directory=directory,
        task_ids=tuple(t.id for t in tasks),
        seed=seed,
        fraction=fraction,
        digest=_sha256(path),
    )
    (directory / LOCKBOX_MANIFEST).write_text(
        json.dumps({"task_ids": list(box.task_ids), "seed": seed, "fraction": fraction, "digest": box.digest}, indent=2)
        + "\n"
    )
    return box


def read_lockbox(directory: Path) -> Lockbox:
    """Load a lockbox and verify its ``tasks.yaml`` still matches the manifest's digest."""
    manifest = directory / LOCKBOX_MANIFEST
    path = directory / LOCKBOX_TASKS
    if not manifest.is_file() or not path.is_file():
        raise FoldError(f"no lockbox under {directory} — run `acumen lockbox` first")
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError) as err:
        raise FoldError(f"cannot read {manifest}: {err}") from err
    digest = _sha256(path)
    if data.get("digest") != digest:
        raise FoldError(
            f"{path} has been modified since the lockbox was written (recorded {data.get('digest')}, now "
            f"{digest}) — a lockbox is never edited"
        )
    return Lockbox(
        directory=directory,
        task_ids=tuple(data.get("task_ids", ())),
        seed=int(data.get("seed", 0)),
        fraction=float(data.get("fraction", 0.0)),
        digest=digest,
    )


def load_lockbox_tasks(box: Lockbox) -> list[Task]:
    """The held-back tasks themselves — for the one final evaluation, and nothing else."""
    try:
        return load_tasks(box.tasks_path)
    except TaskError as err:
        raise FoldError(str(err)) from err


def check_disjoint(working: Sequence[Task], box: Lockbox) -> None:
    """Refuse a working set that overlaps the lockbox — the structural half of the boundary.

    Raises
    ------
    FoldError
        Naming the overlapping ids.
    """
    overlap = sorted({t.id for t in working} & set(box.task_ids))
    if overlap:
        raise FoldError(
            f"{len(overlap)} working task(s) are in the lockbox and may not be optimized on: {', '.join(overlap)}"
        )
