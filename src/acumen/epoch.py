"""Resolve which training epoch an ``acumen epoch`` invocation is on, from on-disk state.

An epoch benches the current arm on ``train``, distils it into the wiki, produces the next skill
version, and benches that version on ``valid``. The resolver here decides — purely from what is on
disk — whether a fresh invocation *starts* a new epoch or *resumes* one that crashed partway, so a
crash between "improve wrote the version" and "valid bench finished" re-enters the SAME epoch and
completes it rather than starting another.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from acumen.skills import available_versions, next_version


@dataclass(frozen=True)
class EpochPlan:
    """What one ``acumen epoch`` invocation will do, resolved from state."""

    #: The arm learned from this epoch: a skill version, or ``None`` for the ``noskill`` baseline.
    parent_version: str | None
    #: The version this epoch produces, e.g. ``"v1"`` or ``"v2"``.
    new_version: str
    #: Whether this is the first epoch (``parent_version is None``), which also benches the
    #: ``noskill`` baseline on ``valid`` for the report.
    first: bool
    #: Whether ``new_version`` already exists on disk (a resumed epoch), so ``improve`` is skipped.
    resumed: bool


def resolve_epoch(skills_root, *, valid_complete: Callable[[str], bool]) -> EpochPlan:
    """Decide the epoch to run from the skills on disk and each version's valid-bench state.

    Parameters
    ----------
    skills_root
        The ``skills/`` root.
    valid_complete
        Predicate: does this version have a complete ``valid`` bench? Injected so the resolver
        stays pure and unit-testable (the caller builds it from ``build_matrix``/``pending``).

    Returns
    -------
    The resolved :class:`EpochPlan`.

    Notes
    -----
    - No versions yet → first epoch: ``parent=None`` (noskill), ``new=v1``.
    - Latest version exists but its valid bench is unfinished → resume that epoch: ``new=latest``,
      ``parent`` is the version below it (or ``None`` when latest is ``v1``), ``resumed=True``.
    - Otherwise → fresh epoch: ``parent=latest``, ``new=next``.
    """
    versions = available_versions(skills_root)
    latest = versions[-1] if versions else None

    if latest is not None and not valid_complete(latest):
        parent = versions[-2] if len(versions) >= 2 else None
        return EpochPlan(parent_version=parent, new_version=latest, first=parent is None, resumed=True)

    new = next_version(skills_root)
    return EpochPlan(parent_version=latest, new_version=new, first=latest is None, resumed=False)
