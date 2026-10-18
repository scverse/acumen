"""Task difficulty — baseline pass rate as the only honest difficulty signal — and task selection.

A skill (and the rulebook that writes it) can only be shown to help on tasks the **baseline** — an
agent with no skill — does not already solve. There is no intrinsic difficulty label on a task, and
guessing one is dishonest, so we *measure* it: run the baseline (`noskill`) arm and read the per-task
pass rate. A task the baseline always passes has no headroom (a skill cannot improve a perfect
score); a task the baseline always fails, or passes only sometimes, is where a skill's guidance can
move the number — those are the tasks the loop must be scored on.

Difficulty is **model-dependent** — an easy task for one model is hard for another — so screening
can be done per reference model (``by_model=True``), which records which model each reading is
relative to, and :func:`select_headroom` judges headroom against the models a loop actually benches
with. Pooling across models (the default) is only honest when one model was screened.

This module only *reads* baseline runs; produce them first with ``acumen bench --no-skill``. Kept
read-only and pure so it composes with the existing run tree and resume, rather than launching a
second, subtly different benchmark path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen.paths import NOSKILL_ARM, RESULT_FILE, SPLITS, Split, parse_run_dir
from acumen.tasks import Task

#: Difficulty strata, by baseline pass rate. ``solved`` has no headroom for the loop; ``hard`` and
#: ``flaky`` do (a skill can move a score the baseline does not already max out).
SOLVED = "solved"
FLAKY = "flaky"
HARD = "hard"
UNSCREENED = "unscreened"


@dataclass(frozen=True)
class Difficulty:
    """The baseline outcome for one (task, split[, model]): how often the no-skill arm passed."""

    task_id: str
    split: Split
    passed: int
    total: int
    #: The reference model these runs came from, or ``None`` when pooled over every model found.
    #: Difficulty is relative to a model; a pooled reading is only meaningful if one model ran.
    model: str | None = None

    @property
    def pass_rate(self) -> float:
        """Baseline pass rate in ``[0, 1]``; ``0.0`` when nothing was screened."""
        return self.passed / self.total if self.total else 0.0

    @property
    def stratum(self) -> str:
        """The difficulty bucket: ``solved`` / ``flaky`` / ``hard`` / ``unscreened``."""
        if self.total == 0:
            return UNSCREENED
        if self.passed == 0:
            return HARD
        if self.passed == self.total:
            return SOLVED
        return FLAKY

    @property
    def has_headroom(self) -> bool:
        """Whether a skill could improve on the baseline here (baseline does not already pass all)."""
        return self.total > 0 and self.passed < self.total


def screen(
    runs_root: Path,
    tasks: Sequence[Task],
    *,
    arm: str = NOSKILL_ARM,
    splits: Sequence[Split] = SPLITS,
    by_model: bool = False,
) -> list[Difficulty]:
    """Read the baseline arm's runs and tally per-(task, split[, model]) pass rate.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    tasks
        The tasks to screen; a run for a task not in this list is ignored (stale).
    arm
        The arm to read as the baseline — ``"noskill"`` by default. Pass a skill arm to screen
        that skill's difficulty instead.
    splits
        Which splits to screen.
    by_model
        Keep each reference model's runs apart and record the model on every reading. Off, runs
        are pooled over every model found — only honest when a single model was screened, since a
        task hard for one model and trivial for another would read as merely "flaky".

    Returns
    -------
    One :class:`Difficulty` per (task, split[, model]) that has at least one baseline run, sorted
    by task id, split, then model. The model is the canonical name recorded in ``result.json``,
    falling back to the run path's slug for a result that predates the field.
    """
    known = {task.id for task in tasks}
    tally: dict[tuple[str, Split, str | None], tuple[int, int]] = {}
    for split in splits:
        split_root = runs_root / arm / split
        if not split_root.is_dir():
            continue
        for result_path in sorted(split_root.rglob(RESULT_FILE)):
            key = parse_run_dir(runs_root, result_path.parent)
            if key.task_id not in known:
                continue
            try:
                data = json.loads(result_path.read_text())
            except (OSError, ValueError):
                continue
            model = (data.get("model") or key.model) if by_model else None
            passed, total = tally.get((key.task_id, split, model), (0, 0))
            tally[(key.task_id, split, model)] = (passed + bool(data.get("success")), total + 1)
    return [
        Difficulty(task_id=task_id, split=split, passed=passed, total=total, model=model)
        for (task_id, split, model), (passed, total) in sorted(
            tally.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")
        )
    ]


@dataclass(frozen=True)
class HeadroomSelection:
    """Which tasks a loop should score on, and why the rest were left out."""

    #: Tasks with headroom, in the caller's original order.
    selected: list[Task]
    #: Screened, and the baseline passed every run — a skill cannot improve a perfect score.
    solved: list[str]
    #: No baseline run for the split (and models) asked about — cannot be placed, so not selected;
    #: ``acumen bench --no-skill`` them first.
    unscreened: list[str]


def select_headroom(
    diffs: Sequence[Difficulty],
    tasks: Sequence[Task],
    *,
    split: Split = "test",
    models: Sequence[str] | None = None,
) -> HeadroomSelection:
    """Keep the tasks the baseline does not already max out on ``split``.

    A task is selected when it has headroom for **any** of ``models`` (or any model present when
    ``models`` is ``None``): a skill helping even one of the configured models can move the
    pooled score, which is what the loop reports. A task with no reading at all for the split (and
    those models) is ``unscreened`` — structurally excluded rather than guessed at, since the whole
    point is to score only where a measured baseline says a skill *could* matter.

    Parameters
    ----------
    diffs
        Readings from :func:`screen` — by model if ``models`` is given, else pooled or by model.
    tasks
        The candidate tasks; order is preserved in ``selected``.
    split
        The split headroom is judged on. The loop scores the held-out ``test`` split, so that is
        the default: a task the baseline aces on test cannot show movement there whatever train says.
    models
        Restrict to readings for these reference models (the loop's ``config.models``).
    """
    wanted = None if models is None else set(models)
    by_task: dict[str, list[Difficulty]] = {}
    for d in diffs:
        if d.split != split or (wanted is not None and d.model not in wanted):
            continue
        by_task.setdefault(d.task_id, []).append(d)

    selected: list[Task] = []
    solved: list[str] = []
    unscreened: list[str] = []
    for task in tasks:
        readings = by_task.get(task.id, [])
        if not readings:
            unscreened.append(task.id)
        elif any(d.has_headroom for d in readings):
            selected.append(task)
        else:
            solved.append(task.id)
    return HeadroomSelection(selected=selected, solved=solved, unscreened=unscreened)
