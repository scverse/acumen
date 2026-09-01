"""Run-directory layout — the single source of truth for path construction.

The layout is ``runs/{arm}/{split}/{model}/{task_id}/rep_{n}/`` and is consumed by the
runner, the resume check, the improver's allow/deny list, and the report. Anything that
needs to know where a run lives asks this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Split = Literal["train", "test"]

SPLITS: tuple[Split, ...] = ("train", "test")

NOSKILL_ARM = "noskill"

ANSWER_FILE = "answer.md"
SCRIPT_FILE = "script.py"
TRANSCRIPT_JSONL = "transcript.jsonl"
TRANSCRIPT_HTML = "transcript.html"
RESULT_FILE = "result.json"

#: Every file a completed run leaf is expected to contain.
RUN_FILES = (ANSWER_FILE, SCRIPT_FILE, TRANSCRIPT_JSONL, TRANSCRIPT_HTML, RESULT_FILE)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_ARM_RE = re.compile(r"^skill_v(\d+)$")
_REP_RE = re.compile(r"^rep_(\d+)$")


class PathError(ValueError):
    """Raised when a path cannot be constructed or parsed."""


def slugify(text: str) -> str:
    """Make ``text`` safe to use as a single path component.

    Parameters
    ----------
    text
        Arbitrary text, e.g. a model name such as ``anthropic/claude-opus-5``.

    Returns
    -------
    A string containing only ``[A-Za-z0-9._-]``.
    """
    slug = _UNSAFE.sub("-", text).strip("-")
    if not slug or slug in {".", ".."}:
        raise PathError(f"{text!r} has no filesystem-safe representation")
    return slug


def is_safe_component(text: str) -> bool:
    """Return whether ``text`` is already usable as a path component verbatim."""
    return bool(text) and text not in {".", ".."} and not _UNSAFE.search(text)


def arm_name(skill: str | None) -> str:
    """Return the run-directory arm name for a skill version.

    Parameters
    ----------
    skill
        A skill version such as ``"v1"``, or ``None`` for the baseline arm.

    Returns
    -------
    ``"noskill"`` for the baseline, otherwise ``"skill_v1"``-style.
    """
    if skill is None:
        return NOSKILL_ARM
    version = skill[1:] if skill.startswith("v") else skill
    if not version.isdigit():
        raise PathError(f"skill version must look like 'v1', got {skill!r}")
    return f"skill_v{int(version)}"


def skill_from_arm(arm: str) -> str | None:
    """Invert :func:`arm_name` — return the skill version an arm refers to."""
    if arm == NOSKILL_ARM:
        return None
    match = _ARM_RE.match(arm)
    if not match:
        raise PathError(f"not an arm name: {arm!r}")
    return f"v{int(match.group(1))}"


@dataclass(frozen=True)
class RunKey:
    """Identifies exactly one benchmark run — one leaf of the run tree."""

    arm: str
    split: Split
    model: str
    task_id: str
    rep: int

    @property
    def skill(self) -> str | None:
        """The skill version this run exercises, or ``None`` for the baseline."""
        return skill_from_arm(self.arm)


def run_dir(runs_root: Path, key: RunKey) -> Path:
    """Build the directory for a single run.

    Parameters
    ----------
    runs_root
        The ``runs/`` root directory.
    key
        Which run to locate.

    Returns
    -------
    ``runs_root/{arm}/{split}/{model}/{task_id}/rep_{n}``.
    """
    if key.split not in SPLITS:
        raise PathError(f"split must be one of {SPLITS}, got {key.split!r}")
    if key.rep < 1:
        raise PathError(f"rep must be >= 1, got {key.rep}")
    if not is_safe_component(key.task_id):
        raise PathError(f"task id {key.task_id!r} is not filesystem-safe")
    return runs_root / key.arm / key.split / slugify(key.model) / key.task_id / f"rep_{key.rep}"


def parse_run_dir(runs_root: Path, path: Path) -> RunKey:
    """Recover a :class:`RunKey` from a run directory.

    The model is returned in its slugified form, which is what the path carries; the
    canonical model name lives in ``result.json``.
    """
    try:
        rel = path.resolve().relative_to(runs_root.resolve())
    except ValueError as err:
        raise PathError(f"{path} is not under {runs_root}") from err
    parts = rel.parts
    if len(parts) != 5:
        raise PathError(f"expected {runs_root}/arm/split/model/task/rep_n, got {rel}")
    arm, split, model, task_id, rep = parts
    skill_from_arm(arm)  # validates the arm name
    if split not in SPLITS:
        raise PathError(f"split must be one of {SPLITS}, got {split!r}")
    match = _REP_RE.match(rep)
    if not match:
        raise PathError(f"expected 'rep_<n>', got {rep!r}")
    return RunKey(arm=arm, split=split, model=model, task_id=task_id, rep=int(match.group(1)))  # type: ignore[arg-type]


def is_complete(directory: Path) -> bool:
    """Return whether a run directory holds a complete run.

    A readable ``result.json`` is what normally marks a run done. Infrastructure-invalid
    results explicitly carry ``valid: false`` and remain pending, so fixing the provider
    quota and rerunning retries them without requiring ``--no-resume``.
    """
    result = directory / RESULT_FILE
    try:
        if not result.is_file() or result.stat().st_size == 0:
            return False
        data = json.loads(result.read_text())
        return isinstance(data, dict) and data.get("valid", True) is not False
    except (OSError, ValueError):
        return False
