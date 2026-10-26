"""Task schema and loader for ``tasks.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acumen.paths import ANSWER_FILE, SCRIPT_FILE, Split, is_safe_component


class TaskError(ValueError):
    """Raised when ``tasks.yaml`` is malformed."""


@dataclass(frozen=True)
class TaskSplit:
    """One half of a task: a prompt and the answer it must produce."""

    prompt: str
    answer: str
    #: Entries of the prepared data cache to link into this split's sandbox, on top of the
    #: task-level list.
    data: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    """A single benchmark task, with a train/valid/test triple and optional overrides."""

    id: str
    train: TaskSplit
    valid: TaskSplit
    test: TaskSplit
    max_turns: int | None = None
    max_usd: float | None = None
    model: str | None = None
    #: Whether this task's answers are reproducible by running code. ``acumen check`` expects
    #: a reproducer script per split for such a task and reports a missing one as a gap; a task
    #: that needs none (a licence, a species name, a fact from the docs) sets this ``False``.
    #: Task-level, not per-split: every split of one task is the same analysis.
    needs_script: bool = True
    #: Entries of the prepared data cache (see :mod:`acumen.data`) linked into every split's
    #: sandbox, alongside any the split adds.
    data: tuple[str, ...] = ()

    def split(self, split: Split) -> TaskSplit:
        """Return the ``train``, ``valid`` or ``test`` half of this task."""
        halves = {"train": self.train, "valid": self.valid, "test": self.test}
        if split not in halves:
            raise TaskError(f"no such split: {split!r}")
        return halves[split]

    def data_for(self, split: Split) -> tuple[str, ...]:
        """Return the data-cache entries one split's runs see: the task's, then its own."""
        return self.data + self.split(split).data


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskError(f"{where} must be a non-empty string, got {value!r}")
    return value


#: Names a data entry may not take at the sandbox root, because the run itself writes there.
_RESERVED_DATA = frozenset({ANSWER_FILE, SCRIPT_FILE})


def _parse_data(raw: Any, where: str, inherited: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Validate a ``data:`` list of relative paths into the prepared data cache.

    ``inherited`` is the task-level list a split's entries are added to, checked against them so
    the split's runs never get one entry twice or one inside another.
    """
    if not isinstance(raw, list):
        raise TaskError(f"{where} must be a list of paths inside the data cache, got {type(raw).__name__}")
    entries: list[str] = list(inherited)
    for i, item in enumerate(raw):
        entry = _require_str(item, f"{where}[{i}]").strip().rstrip("/")
        parts = entry.split("/")
        # Each component must be plain: no absolute path, no '..' climbing out of the cache, and
        # no hidden first component, which could land on the sandbox's own .claude/ or .agents/.
        if entry.startswith("/") or not all(is_safe_component(part) for part in parts):
            raise TaskError(f"{where}[{i}] {item!r} must be a relative path inside the data cache, with no '..'")
        if parts[0].startswith(".") or entry in _RESERVED_DATA:
            raise TaskError(f"{where}[{i}] {item!r} would collide with a file the run writes in its sandbox")
        if entry in entries:
            also = " (the task's data: already has it)" if entry in inherited else ""
            raise TaskError(f"{where} lists {entry!r} twice{also}")
        # Each entry becomes one symlink, so one inside another would be linked *through* the
        # outer link, into the shared cache itself. Name the directory alone.
        nested = next(
            (other for other in entries if f"{other}/".startswith(f"{entry}/") or f"{entry}/".startswith(f"{other}/")),
            None,
        )
        if nested is not None:
            raise TaskError(f"{where} lists {entry!r} and {nested!r}, one inside the other — name only the outer one")
        entries.append(entry)
    return tuple(entries[len(inherited) :])


def _parse_split(raw: Any, where: str, inherited: tuple[str, ...] = ()) -> TaskSplit:
    if not isinstance(raw, dict):
        raise TaskError(f"{where} must be a mapping with 'prompt' and 'answer', got {type(raw).__name__}")
    unknown = set(raw) - {"prompt", "answer", "data"}
    if unknown:
        raise TaskError(f"{where} has unknown keys: {sorted(unknown)}")
    for key in ("prompt", "answer"):
        if key not in raw:
            raise TaskError(f"{where} is missing '{key}'")
    return TaskSplit(
        prompt=_require_str(raw["prompt"], f"{where}.prompt"),
        answer=_require_str(raw["answer"], f"{where}.answer"),
        data=_parse_data(raw["data"], f"{where}.data", inherited) if "data" in raw else (),
    )


def _parse_task(raw: Any, index: int) -> Task:
    where = f"tasks[{index}]"
    if not isinstance(raw, dict):
        raise TaskError(f"{where} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"id", "train", "valid", "test", "max_turns", "max_usd", "model", "needs_script", "data"}
    if unknown:
        raise TaskError(f"{where} has unknown keys: {sorted(unknown)}")
    if "id" not in raw:
        raise TaskError(f"{where} is missing 'id'")
    task_id = _require_str(raw["id"], f"{where}.id")
    if not is_safe_component(task_id):
        raise TaskError(f"{where}.id {task_id!r} is not filesystem-safe — use only letters, digits, '.', '_', '-'")
    where = f"task {task_id!r}"
    for key in ("train", "valid", "test"):
        if key not in raw:
            raise TaskError(f"{where} is missing the '{key}' split — train, valid and test are all required")
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
    needs_script = raw.get("needs_script", True)
    # Strictly a bool: `needs_script: 1` or `"no"` would silently mean something the author
    # did not write, and this key decides whether a missing reproducer is a gap or expected.
    if not isinstance(needs_script, bool):
        raise TaskError(f"{where}.needs_script must be true or false, got {needs_script!r}")
    data = _parse_data(raw["data"], f"{where}.data") if "data" in raw else ()
    return Task(
        id=task_id,
        train=_parse_split(raw["train"], f"{where}.train", data),
        valid=_parse_split(raw["valid"], f"{where}.valid", data),
        test=_parse_split(raw["test"], f"{where}.test", data),
        max_turns=max_turns,
        max_usd=max_usd,
        model=model,
        needs_script=needs_script,
        data=data,
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
