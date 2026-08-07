"""Killing the processes an agent leaves behind.

An agent is a ``claude`` CLI subprocess, and that CLI spawns processes of its own: every
Bash tool call is a shell, and a shell can start a pytest, a fit that runs for an hour, or
something backgrounded with ``&``. The SDK terminates the CLI and nothing else — so does a
crash, a cap breach, or a Ctrl-C — and the descendants survive. They keep burning CPU
alongside the runs still going, and they keep writing into a directory that is about to be
deleted.

The parent link cannot find them, on any platform. On Unix an orphan is reparented to init,
so the tree that identified it is gone; on Windows it is not reparented at all and its
recorded parent id dangles, free to be reused by an unrelated process. What survives in
both cases is the environment: :func:`label_env` stamps every agent with :data:`RUN_MARKER`,
set to that run's private temp directory, and a process inherits its environment from
whatever started it. So the marker identifies a descendant even after the parent is gone,
the process has renamed itself, or it has changed directory. :func:`reap` finds everything
carrying a given run's marker and kills it; the sandbox and the meta-agent staging dirs call
it on the way out, before their directory is removed, on every exit path including
cancellation.

The marker is what makes this exact rather than a guess at which paths a run touches. Every
agent has a throwaway ``HOME`` and ``TMPDIR`` under its run directory, but the explicit marker
still identifies descendants that change directory or sanitize part of their environment.

Reading *another* process's environment is the one step with no portable stdlib spelling:
it is a file read on Linux, a ``KERN_PROCARGS2`` sysctl on macOS, and a ``ReadProcessMemory``
walk of the PEB on Windows. ``psutil`` implements all three natively, which is why it is a
dependency. :func:`supported` reports whether the running platform is one of them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import psutil

#: Environment variable naming the run a process belongs to, set to the run's temp directory.
#: Stamped on every agent by :func:`label_env` and inherited by everything the agent spawns.
#: Uppercase deliberately: Windows environment keys are case-insensitive and psutil upper-cases
#: them when reading a process block, so an all-caps name is the one spelling that round-trips
#: identically on every platform.
RUN_MARKER = "ACUMEN_RUN"

#: How long to let a signalled process exit before escalating to a hard kill. An upper bound,
#: not a fixed pause — the wait ends the moment the processes are gone, which is the common
#: case. It still needs to stay modest: the callers are synchronous teardown paths inside
#: async runs, so this blocks the event loop and every other run on it.
GRACE_S = 2.0

#: Raised by psutil when a process is gone, is another user's, or is unreadable for a
#: platform-specific reason (on Windows a 32-bit interpreter cannot read a 64-bit process's
#: environment). Every one of these means "not identifiable as ours", never "abort the scan".
_UNREADABLE = (psutil.Error, OSError, ValueError)


def supported() -> bool:
    """Whether orphan reaping works on this platform.

    True on Linux, macOS and Windows. False only where psutil cannot read another process's
    environment, in which case :func:`reap` does nothing and orphans are left alone.
    """
    return hasattr(psutil.Process, "environ")


def label_env(env: dict[str, str], holder: Path) -> dict[str, str]:
    """Stamp an agent's environment with the run it belongs to, and return it.

    Call this on the mapping handed to the SDK, with the same ``holder`` later passed to
    :func:`reap`. Without the stamp a run's orphans are unidentifiable and :func:`reap` has
    nothing to match on.

    Parameters
    ----------
    env
        The agent's environment, modified in place.
    holder
        The run's private temp directory, which is unique per run and so serves as its id.

    Returns
    -------
    The same mapping, for chaining onto the call that built it.
    """
    env[RUN_MARKER] = str(holder)
    return env


def _protected() -> set[int]:
    """Return acumen's own pid and every ancestor of it.

    A belt-and-braces guard. Matching is on a variable acumen itself never has in its
    environment, so these should never come up as candidates — but the cost of a false
    positive here is killing the process doing the killing, or the operator's shell.
    """
    me = psutil.Process()
    protected = {me.pid}
    try:
        protected.update(parent.pid for parent in me.parents())
    except _UNREADABLE:  # pragma: no cover - an ancestor exiting mid-walk
        pass
    return protected


def _labelled(proc: psutil.Process, holder: Path) -> bool:
    """Whether ``proc`` belongs to the run rooted at ``holder``.

    The environment psutil reports is the one the process was *executed* with, which is why
    the marker still identifies a descendant whose parent has since died.

    ``cwd`` is checked as a second signal, for the rare process started with a wiped
    environment that is nonetheless running inside the run's directory. Path comparison is
    case-insensitive on Windows, which :class:`~pathlib.Path` already handles.
    """
    try:
        if proc.environ().get(RUN_MARKER) == str(holder):
            return True
    except _UNREADABLE:
        pass
    try:
        cwd = proc.cwd()
    except _UNREADABLE:
        return False
    return bool(cwd) and Path(cwd).is_relative_to(holder)


def _survivor_procs(holder: Path) -> Iterator[psutil.Process]:
    """Yield live processes descended from the agent that ran in ``holder``."""
    if not supported():
        return
    protected = _protected()
    for proc in psutil.process_iter():
        # Processes come and go mid-scan; one that vanished is simply not a survivor.
        if proc.pid not in protected and _labelled(proc, holder):
            yield proc


def survivors(holder: Path) -> list[int]:
    """Return the pids of every live process descended from the agent that ran in ``holder``.

    Parameters
    ----------
    holder
        The run's temp directory, the same one passed to :func:`label_env`.

    Returns
    -------
    The matching pids, excluding acumen and its ancestors. Empty on an unsupported platform.
    """
    return [proc.pid for proc in _survivor_procs(holder)]


def _quietly(action: Callable[[], None]) -> None:
    """Run a kill that may lose its race with the process exiting on its own."""
    try:
        action()
    except _UNREADABLE:
        pass


def reap(holder: Path, *, grace: float = GRACE_S) -> list[int]:
    """Kill everything the agent that ran in ``holder`` left behind.

    Asks politely first (``SIGTERM``; on Windows, where there is no such signal, psutil
    terminates outright), then kills whatever is still standing after ``grace``. The second
    round re-scans rather than reusing the first round's list, so a process spawned while
    the first signals were landing is caught too.

    Safe to call when nothing leaked, when the agent exited cleanly, and on a platform
    psutil cannot read — in each case it does nothing and returns an empty list.

    Parameters
    ----------
    holder
        The run's temp directory, as passed to :func:`survivors`.
    grace
        Seconds to wait before the hard kill. Blocks the caller, but only until the
        signalled processes are actually gone.

    Returns
    -------
    The pids that were signalled, for logging. An empty list means nothing was orphaned.
    """
    doomed = list(_survivor_procs(holder))
    if not doomed:
        return []
    for proc in doomed:
        _quietly(proc.terminate)
    # Waiting on a process that is our own direct child consumes its exit status, which
    # would otherwise be the SDK's to collect. Harmless here: callers reap only once the
    # SDK has closed its transport, so the agent process has already been waited on.
    psutil.wait_procs(doomed, timeout=grace)
    for proc in _survivor_procs(holder):
        _quietly(proc.kill)
    return [proc.pid for proc in doomed]
