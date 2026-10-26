"""The prepared data cache: inputs built once by ``acumen prepare`` and linked into runs.

A target's tasks often start from the same dataset or prior-knowledge resource, and fetching it
inside every run is the slowest (and flakiest) part of the run. ``acumen prepare`` builds such
inputs once into a project-local directory (``cache/`` by default); a task, or one of its splits,
then names the entries its runs start with under ``data:`` in ``tasks.yaml``.

Each named entry, a file or a whole directory, is symlinked into the run's working directory at
the same relative path. The links are what the agent sees; nothing in its prompt mentions them.
The cache is left read-only by ``acumen prepare``, because the link is shared by every run and the
Claude path guard cannot tell a read from a write.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from acumen.paths import SPLITS, Split
from acumen.tasks import Task

#: Default data cache directory, relative to where acumen is run, like ``runs/`` and ``skills/``.
DATA_DIRNAME = "cache"


class DataError(RuntimeError):
    """Raised when a task names data the cache does not hold."""


def all_splits(tasks: Iterable[Task], splits: Sequence[Split] = SPLITS) -> list[tuple[Task, Split]]:
    """Pair every task with every split, the shape :func:`missing_data` takes."""
    return [(task, split) for task in tasks for split in splits]


def missing_data(cells: Iterable[tuple[Task, Split]], data_root: Path) -> list[str]:
    """Return every entry the given ``(task, split)`` cells name that ``data_root`` does not hold.

    Parameters
    ----------
    cells
        The task splits about to run; an entry only a skipped split names is not required.
    data_root
        The prepared data cache.

    Returns
    -------
    The missing entries, deduplicated, in first-mention order.
    """
    missing: dict[str, None] = {}
    for task, split in cells:
        for name in task.data_for(split):
            if not (data_root / name).exists():
                missing[name] = None
    return list(missing)


def require_data(cells: Iterable[tuple[Task, Split]], data_root: Path) -> None:
    """Raise :class:`DataError` naming what is missing, before any run starts without it."""
    missing = missing_data(cells, data_root)
    if missing:
        listed = ", ".join(missing)
        raise DataError(f"missing from the data cache {data_root}: {listed} — run `acumen prepare` to build it")


def data_sources(names: Sequence[str], data_root: Path) -> list[tuple[str, Path]]:
    """Pair each entry with the resolved path it links to.

    Resolved, because that is the path both providers' confinement judges a read by: the Claude
    guard resolves symlinks before checking a path, and Codex's kernel sandbox sees the real file.
    """
    sources = []
    for name in names:
        source = (data_root / name).resolve()
        if not source.exists():
            raise DataError(f"missing from the data cache {data_root}: {name} — run `acumen prepare` to build it")
        sources.append((name, source))
    return sources


def link_data(dest_root: Path, sources: Sequence[tuple[str, Path]]) -> None:
    """Symlink each ``(name, source)`` pair at ``dest_root / name``, creating parent directories.

    Refuses a link whose parent is itself a link: creating it would write into whatever that
    outer link points at, which is the shared cache. ``tasks.yaml`` parsing already rejects
    nested entries; this keeps the API honest for any other caller.
    """
    for name, source in sources:
        link = dest_root / name
        if any(parent.is_symlink() for parent in link.parents if parent != dest_root and dest_root in parent.parents):
            raise DataError(f"data entry {name!r} lies inside another linked entry")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source, target_is_directory=source.is_dir())
