"""Task schema and loader for ``tasks.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acumen.paths import Split, is_safe_component


class TaskError(ValueError):
    """Raised when ``tasks.yaml`` is malformed."""


@dataclass(frozen=True)
class TaskSplit:
    """One half of a task: a prompt and the answer it must produce."""

    prompt: str
    answer: str


@dataclass(frozen=True)
class Task:
    """A single benchmark task, with a train/test pair and optional overrides."""

    id: str
    train: TaskSplit
    test: TaskSplit
    max_turns: int | None = None
    max_usd: float | None = None
    model: str | None = None

    def split(self, split: Split) -> TaskSplit:
        """Return the ``train`` or ``test`` half of this task."""
        if split not in ("train", "test"):
            raise TaskError(f"no such split: {split!r}")
        return self.train if split == "train" else self.test


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _parse_split(raw: Any, where: str) -> TaskSplit:
    if not isinstance(raw, dict):
        raise TaskError(f"{where} must be a mapping with 'prompt' and 'answer', got {type(raw).__name__}")
    unknown = set(raw) - {"prompt", "answer"}
    if unknown:
        raise TaskError(f"{where} has unknown keys: {sorted(unknown)}")
    for key in ("prompt", "answer"):
        if key not in raw:
            raise TaskError(f"{where} is missing '{key}'")
    return TaskSplit(
        prompt=_require_str(raw["prompt"], f"{where}.prompt"), answer=_require_str(raw["answer"], f"{where}.answer")
    )


def _parse_task(raw: Any, index: int) -> Task:
    where = f"tasks[{index}]"
    if not isinstance(raw, dict):
        raise TaskError(f"{where} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"id", "train", "test", "max_turns", "max_usd", "model"}
    if unknown:
        raise TaskError(f"{where} has unknown keys: {sorted(unknown)}")
    if "id" not in raw:
        raise TaskError(f"{where} is missing 'id'")
    task_id = _require_str(raw["id"], f"{where}.id")
    if not is_safe_component(task_id):
        raise TaskError(f"{where}.id {task_id!r} is not filesystem-safe — use only letters, digits, '.', '_', '-'")
    where = f"task {task_id!r}"
    for key in ("train", "test"):
        if key not in raw:
            raise TaskError(f"{where} is missing the '{key}' split — both splits are required")
    max_turns = raw.get("max_turns")
    if max_turns is not None and (not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1):
        raise TaskError(f"{where}.max_turns must be a positive integer, got {max_turns!r}")
    max_usd = raw.get("max_usd")
    if max_usd is not None:
        if isinstance(max_usd, bool) or not isinstance(max_usd, int | float) or max_usd <= 0:
            raise TaskError(f"{where}.max_usd must be a positive number, got {max_usd!r}")
        max_usd = float(max_usd)
    model = raw.get("model")
    if model is not None:
        model = _require_str(model, f"{where}.model")
    return Task(
        id=task_id,
        train=_parse_split(raw["train"], f"{where}.train"),
        test=_parse_split(raw["test"], f"{where}.test"),
        max_turns=max_turns,
        max_usd=max_usd,
        model=model,
    )


def parse_tasks(raw: Any) -> list[Task]:
    """Validate an already-parsed ``tasks.yaml`` document.

    Parameters
    ----------
    raw
        The mapping loaded from YAML.

    Returns
    -------
    The tasks, in file order.
    """
    if not isinstance(raw, dict):
        raise TaskError(f"tasks.yaml must be a mapping with a 'tasks' key, got {type(raw).__name__}")
    if "tasks" not in raw:
        raise TaskError("tasks.yaml is missing the top-level 'tasks' key")
    entries = raw["tasks"]
    if not isinstance(entries, list) or not entries:
        raise TaskError("'tasks' must be a non-empty list")
    tasks = [_parse_task(entry, i) for i, entry in enumerate(entries)]
    seen: dict[str, int] = {}
    for i, task in enumerate(tasks):
        if task.id in seen:
            raise TaskError(
                f"duplicate task id {task.id!r} (tasks[{seen[task.id]}] and tasks[{i}]) — ids must be unique"
            )
        seen[task.id] = i
    return tasks


def load_tasks(path: Path) -> list[Task]:
    """Load and validate ``tasks.yaml`` from disk."""
    try:
        text = path.read_text()
    except OSError as err:
        raise TaskError(f"cannot read tasks file {path}: {err}") from err
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise TaskError(f"{path} is not valid YAML: {err}") from err
    try:
        return parse_tasks(raw)
    except TaskError as err:
        raise TaskError(f"{path}: {err}") from err
