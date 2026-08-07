"""Tool-layer filesystem containment for isolated Claude agents.

Claude Code's OS sandbox is all-or-nothing on the axes acumen cares about. Enabling it
confines the filesystem but also interposes a proxy on every outbound connection, and the
allowlist that proxy enforces cannot be set from the ``--settings`` file the SDK passes:
``sandbox.network.allowedDomains`` is ignored there, whether it names ``"*"`` or an explicit
host, and so are ``WebFetch(domain:...)`` allow rules. Verified against CLI 2.1.224, where a
run with the sandbox on cannot reach zenodo.org, ftp.ebi.ac.uk or omnipathdb.org, and the same
run with the sandbox off reaches all three.

Benchmark targets download their own datasets and priors, so egress has to stay open, which
means the sandbox has to stay off, which means the filesystem boundary has to be enforced
somewhere else. This module is that somewhere: a ``PreToolUse`` hook that refuses any tool
call naming a path outside the run's roots.

What this does and does not buy is worth being precise about. It stops the agent *exploring*
the host: no reading the operator's home, no walking sibling checkouts, no grepping for
credentials. It does not stop a program the agent legitimately starts from opening whatever
it likes, because the kernel is no longer in the loop -- ``python -c "open('/etc/shadow')"``
is a string the hook cannot see through. That is the boundary acumen needs (a benchmark agent
must not discover another task's data or the operator's files) rather than a security
boundary against an adversarial agent.

Codex needs none of this: its permission profile confines the filesystem at the OS level and
leaves egress alone, so it keeps kernel enforcement and only uses the narrower
``deny_paths`` guard in :mod:`acumen.agents`.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: System directories every ordinary command touches, on top of the run's own roots. An agent
#: reading these learns nothing about the operator or about another run: they are the same on
#: any machine with the same packages installed. Leaving them out would deny ``/dev/null``,
#: which is in half the shell commands an agent writes, and read the resulting failures as the
#: model being bad at bash.
SYSTEM_ROOTS: tuple[Path, ...] = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
    Path("/opt"),
    Path("/var"),
)

#: Characters that end one shell word and start another. Splitting on them first means a
#: command like ``cd /tmp && cat ~/.ssh/id_rsa`` is examined as its parts rather than as one
#: opaque string.
_SHELL_BREAKS = "|&;()<>`\n\r\t"


def _candidates(value: str) -> list[str]:
    """Return the substrings of ``value`` that might be paths."""
    parts = [value]
    flattened = value.translate(str.maketrans(dict.fromkeys(_SHELL_BREAKS, " ")))
    try:
        parts.extend(shlex.split(flattened))
    except ValueError:
        # Unbalanced quotes: fall back to whitespace splitting rather than skipping the
        # command, since a command we cannot parse is exactly one we should not wave through.
        parts.extend(flattened.split())
    return parts


def _strings(value: Any) -> list[str]:
    """Flatten every string reachable inside a tool input."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return []


def _within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _expand(token: str, home: Path | None) -> Path:
    """Expand a leading ``~`` against the *agent's* home, not the harness process's.

    ``Path.expanduser`` reads the ``HOME`` of whatever process calls it, which here is acumen
    itself. An agent's home is the throwaway directory in its own environment, so expanding
    with the default would judge the agent's own ``~/tmp`` against the operator's real home
    and deny it.
    """
    if home is not None:
        if token == "~":
            return home
        if token.startswith("~/"):
            return home / token[2:]
    return Path(token).expanduser()


def find_escape(
    tool_input: Any,
    allow_roots: Sequence[Path],
    deny_roots: Sequence[Path] = (),
    cwd: Path | None = None,
    home: Path | None = None,
) -> str | None:
    """Return the first path in ``tool_input`` that leaves the run's roots, or ``None``.

    A candidate escapes when it resolves outside every entry in ``allow_roots``, or inside any
    entry in ``deny_roots``. Deny wins: those are the paths a specific agent must not see even
    though they sit inside its workspace, such as the benchmark ``runs/`` tree an improver is
    reasoning about.

    Only candidates that *look* like paths are judged. A bare word with no separator is
    skipped, because command names and flags would otherwise resolve against ``cwd`` and every
    command would be an escape.
    """
    base = (cwd or Path.cwd()).resolve()
    allowed = [root.resolve() for root in allow_roots]
    denied = [root.resolve() for root in deny_roots]
    for value in _strings(tool_input):
        for raw in _candidates(value):
            token = raw.strip(" ,;()[]{}'\"")
            if not token or ("/" not in token and not token.startswith("~")):
                continue
            if "://" in token:
                continue  # a URL, not a path
            try:
                expanded = _expand(token, home)
                resolved = (expanded if expanded.is_absolute() else base / expanded).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if _within(resolved, denied):
                return token
            if not _within(resolved, allowed):
                return token
    return None


def containment_hook(
    allow_roots: Sequence[Path],
    deny_roots: Sequence[Path] = (),
    cwd: Path | None = None,
    home: Path | None = None,
) -> Any:
    """Build the ``PreToolUse`` hook that keeps a Claude agent inside its roots.

    ``matcher=None`` fires for every tool, so a path cannot be smuggled through whichever tool
    the hook forgot to name.
    """
    # Imported here, not at module scope: the Claude SDK is an optional dependency and a
    # Codex-only install never builds an SDK hook.
    from claude_agent_sdk import HookMatcher

    allowed = [*(root.resolve() for root in allow_roots), *SYSTEM_ROOTS]
    denied = [root.resolve() for root in deny_roots]

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_escape(input_data.get("tool_input") or {}, allowed, denied, cwd, home)
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen confines this run to its own sandbox directory, and {hit} is outside it. "
                    "Work from the files in your working directory and the installed package; do not "
                    "read or list paths elsewhere on the host."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])
