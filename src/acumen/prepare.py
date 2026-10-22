"""``acumen prepare``: build the data cache that task sandboxes start from.

Each ``prepare/<name>.py`` is one of the operator's own scripts. It runs in the target venv with
the data cache as its working directory, the cache's absolute path as ``argv[1]`` and as
``$ACUMEN_DATA_CACHE``, and writes whatever files or directories its tasks need there. A task
then names those entries under ``data:`` in ``tasks.yaml`` (see :mod:`acumen.data`).

A script that succeeded is recorded in ``<cache>/.acumen-prepared/<name>.json`` with its content
hash, and skipped on later runs until the script changes or ``--force`` asks again. Scripts run
in filename order, one at a time by default, so a later script may build on an earlier one's
output.

When the pass ends the cache is made read-only. Runs reach it through symlinks, and nothing at
the agent's side stops a write through one: Claude's path guard treats a readable root as
writable, so the file mode is what keeps one run from corrupting every later run's inputs.
"""

from __future__ import annotations

import hashlib
import json
import stat
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from acumen.check import run_script

#: Default directory holding the prepare scripts, a sibling of ``tasks.yaml`` like ``tasks/``.
PREPARE_DIRNAME = "prepare"

#: Where the per-script success markers live inside the cache. Hidden, so a ``data:`` entry can
#: never name it (see :func:`acumen.tasks._parse_data`).
MARKER_DIRNAME = ".acumen-prepared"

#: Environment variable that tells a prepare script where the cache is.
CACHE_ENV = "ACUMEN_DATA_CACHE"

#: Default seconds one prepare script may run for. Downloads of large atlases are slow.
DEFAULT_TIMEOUT = 3600.0

PrepareStatus = Literal["ok", "skipped", "error", "timeout"]

#: Statuses that mean a script did not leave its data behind.
FAILED: frozenset[PrepareStatus] = frozenset({"error", "timeout"})

_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class PrepareError(RuntimeError):
    """Raised when the prepare pass cannot be carried out at all."""


@dataclass(frozen=True)
class PrepareResult:
    """What one prepare script did."""

    name: str
    script: Path
    status: PrepareStatus
    seconds: float = 0.0
    returncode: int | None = None
    error_tail: str | None = None


def prepare_scripts(scripts_root: Path, names: Sequence[str] | None = None) -> list[Path]:
    """Return the prepare scripts to run, in filename order.

    Parameters
    ----------
    scripts_root
        The ``prepare/`` directory.
    names
        Restrict to these script names (the filename stem, e.g. ``pbmc3k``). A name that
        matches nothing is an error, so a typo never reads as "nothing to do".
    """
    if not scripts_root.is_dir():
        raise PrepareError(f"no prepare scripts directory at {scripts_root}")
    scripts = sorted(path for path in scripts_root.glob("*.py") if path.is_file())
    if names:
        known = {path.stem for path in scripts}
        unknown = set(names) - known
        if unknown:
            raise PrepareError(f"unknown prepare scripts: {sorted(unknown)} (known: {sorted(known)})")
        scripts = [path for path in scripts if path.stem in set(names)]
    return scripts


def _script_hash(script: Path) -> str:
    return hashlib.sha256(script.read_bytes()).hexdigest()


def _marker(cache: Path, script: Path) -> Path:
    return cache / MARKER_DIRNAME / f"{script.stem}.json"


def is_prepared(cache: Path, script: Path) -> bool:
    """Return whether ``script`` already ran successfully, in its current form, into ``cache``."""
    try:
        recorded = json.loads(_marker(cache, script).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(recorded, dict) and recorded.get("sha256") == _script_hash(script)


def _set_writable(root: Path, writable: bool) -> None:
    """Add or remove every write bit under ``root``, never following a symlink out of it."""
    if not root.exists():
        return
    paths = [root, *root.rglob("*")]
    for path in paths:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR if writable else mode & ~_WRITE_BITS)


def unfreeze(cache: Path) -> None:
    """Make the cache writable again, as a prepare script (or ``rm -r``) needs it to be."""
    _set_writable(cache, True)


def freeze(cache: Path) -> None:
    """Make the cache read-only, so no run can write through its symlinks into it."""
    _set_writable(cache, False)


def run_prepare(
    scripts: Sequence[Path],
    *,
    python: Path,
    cache: Path,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    jobs: int = 1,
    on_done: Callable[[PrepareResult], None] | None = None,
) -> list[PrepareResult]:
    """Run the prepare scripts that are out of date, then leave the cache read-only.

    Parameters
    ----------
    scripts
        The scripts to consider, from :func:`prepare_scripts`.
    python
        The target venv interpreter, so the data is built with the package version under test.
    cache
        The data cache to build into; created when absent.
    force
        Re-run every script, even one already prepared in its current form.
    timeout
        Seconds any one script may run for.
    jobs
        How many scripts to run at once. The default of one keeps filename order meaningful.
    on_done
        Called with each result as it completes, for progress output.

    Returns
    -------
    One result per script, in the order given.
    """
    if jobs < 1:
        raise PrepareError(f"jobs must be >= 1, got {jobs}")
    cache.mkdir(parents=True, exist_ok=True)
    cache = cache.resolve()
    todo = [script for script in scripts if force or not is_prepared(cache, script)]

    def run(script: Path) -> PrepareResult:
        if script not in todo:
            result = PrepareResult(name=script.stem, script=script, status="skipped")
        else:
            result = _run_one(script, python=python, cache=cache, timeout=timeout)
        if on_done is not None:
            on_done(result)
        return result

    if todo:
        unfreeze(cache)
    try:
        if jobs == 1:
            return [run(script) for script in scripts]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(run, scripts))
    finally:
        freeze(cache)


def _run_one(script: Path, *, python: Path, cache: Path, timeout: float) -> PrepareResult:
    marker = _marker(cache, script)
    # A failed re-run must not leave the old marker vouching for whatever half-written output
    # the script left behind.
    marker.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        proc = run_script(
            script,
            python=python,
            cwd=cache,
            timeout=timeout,
            args=(str(cache),),
            extra_env={CACHE_ENV: str(cache)},
        )
    except OSError as err:
        return PrepareResult(
            name=script.stem,
            script=script,
            status="error",
            seconds=time.monotonic() - started,
            error_tail=f"cannot run {script}: {err}",
        )
    if proc.timed_out or proc.returncode != 0:
        return PrepareResult(
            name=script.stem,
            script=script,
            status="timeout" if proc.timed_out else "error",
            seconds=proc.seconds,
            returncode=None if proc.timed_out else proc.returncode,
            error_tail=proc.error_tail,
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "script": script.name,
        "sha256": _script_hash(script),
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(proc.seconds, 1),
    }
    marker.write_text(json.dumps(record, indent=2) + "\n")
    return PrepareResult(name=script.stem, script=script, status="ok", seconds=proc.seconds, returncode=proc.returncode)
