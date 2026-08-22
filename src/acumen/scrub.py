"""Hide a target's own agent guidance, so the skill under test is the only one in play.

A target package may ship agent-facing guidance of its own: a first-party skill inside the
installed package (``<pkg>/_skills/data/SKILL.md`` plus a ``references/`` tree — the shape
``acumen ship`` itself produces), a repo-level ``.claude/skills/`` or ``.agents/skills/``, a
``CLAUDE.md``, an ``AGENTS.md``, Copilot instructions. None of it is registered anywhere acumen
looks and no agent is told about it, which is exactly why it went unnoticed for so long: agents
find it by grepping the venv they were handed.

It breaks both comparisons acumen exists to make. A baseline run that read the target's own
skill is not skill-free and a skill-arm run that read both is not measuring ours — across four
measured passes, 18.2% of baseline runs and 12.2% of skill-arm runs had done one or the other,
concentrated in exactly the tasks that discriminate between arms. And a skill drafted while
reading the maintainer's skill, or a task set mined from it, is not the independent artifact the
report presents it as.

So the guidance goes before any agent starts, one of two ways depending on what that agent is
allowed to read:

* the target **venv** is scrubbed in place (:func:`scrub_venv`) — every benchmark run reads it,
  and it is the only part of the target a benchmark run can reach;
* an agent that legitimately reads the target **source** (``draft``, ``tasks``) is pointed at a
  filtered copy (:func:`build_filtered_source`) and denied the original
  (:func:`make_skill_guard`). The real checkout stays byte-identical: ``ship`` commits from it,
  and for a local target it is the user's own working tree.

The vocabulary below is the single definition all of them use.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import HookMatcher

from acumen.skills import REFERENCES_DIR, SKILL_FILE

#: Directory names that hold agent guidance rather than package code. Removed wherever they
#: appear, and denied wherever they appear on a resolved path (``.agents/skills``,
#: ``.claude/skills``, ``.cursor/rules``, ``.claude-plugin``, …).
GUIDANCE_DIRS = frozenset({".claude", ".agents", ".codex", ".cursor", ".claude-plugin"})

#: Directory names that *may* hold a packaged skill. Unlike :data:`GUIDANCE_DIRS` these are
#: also plausible module names, so one is only treated as guidance when it actually contains a
#: ``SKILL.md`` (see :func:`_holds_skill`) — a package whose own code lives in ``skills/`` keeps it.
SKILL_DIRS = frozenset({"_skills", "skills"})

#: Basenames anywhere in a tree that are agent-facing skill/guidance artifacts.
GUIDANCE_FILES = frozenset({SKILL_FILE, "CLAUDE.md", "AGENTS.md", "copilot-instructions.md"})

#: tool_input keys carrying a filesystem path — the coarse set the improver's guard also uses.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))

#: Largest ``bin/`` entry read while looking for a dead console script. A generated entry point
#: is a few hundred bytes of Python; the cap keeps us from reading a vendored binary.
_SCRIPT_MAX_BYTES = 64 * 1024


# ── Finding it ─────────────────────────────────────────────────────────────────────────


def _holds_skill(directory: Path) -> bool:
    """Whether ``directory`` contains a ``SKILL.md`` at any depth.

    This is what separates a shipped skill from a package that happens to have a module called
    ``skills``: the marker file is the thing an agent would read, and code never carries one.
    """
    return any(True for _ in directory.rglob(SKILL_FILE))


def find_guidance(root: Path) -> list[Path]:
    """Return the agent-guidance paths under ``root``, outermost first.

    Nothing is removed. A hit's own contents are never reported separately, so the result can be
    deleted (or listed) as it stands. A ``references/`` directory beside a reported ``SKILL.md``
    comes with it: that pairing is the standard skill bundle, and taking the entry point without
    the body would leave an agent the whole reference tree to read.

    Parameters
    ----------
    root
        A venv or a source tree.

    Returns
    -------
    Directories and files to remove, each outside every other entry.
    """
    hits: list[Path] = []
    for parent, dirs, files in os.walk(root):
        here = Path(parent)
        keep: list[str] = []
        for name in sorted(dirs):
            child = here / name
            if name in GUIDANCE_DIRS or (name in SKILL_DIRS and _holds_skill(child)):
                hits.append(child)
            else:
                keep.append(name)
        # Never descend into something we are about to remove: its contents are already covered.
        dirs[:] = keep
        for name in sorted(files):
            if name not in GUIDANCE_FILES:
                continue
            hits.append(here / name)
            if name == SKILL_FILE and REFERENCES_DIR in dirs:
                hits.append(here / REFERENCES_DIR)
                dirs.remove(REFERENCES_DIR)
    return hits


def _dead_scripts(venv_dir: Path, doomed: Sequence[Path]) -> list[Path]:
    """Return console scripts pointing into a package directory that is about to go.

    A packaged skill usually ships an installer entry point beside it, and ``uv pip install``
    writes that as a script in the venv's ``bin`` — which is on every agent's PATH, so it is the
    most visible pointer to a skill whose data has just been removed, as well as being
    unimportable once it has. Matched on the module path it names rather than on its own
    filename, so it holds whatever a given package calls its installer.
    """
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    modules = set()
    for path in doomed:
        parts = path.parts
        if path.is_dir() and "site-packages" in parts:
            modules.add(".".join(parts[parts.index("site-packages") + 1 :]))
    if not modules or not bin_dir.is_dir():
        return []
    dead: list[Path] = []
    for script in sorted(bin_dir.iterdir()):
        try:
            if not script.is_file() or script.stat().st_size > _SCRIPT_MAX_BYTES:
                continue
            text = script.read_text(errors="ignore")
        except OSError:
            continue
        if any(module in text for module in modules):
            dead.append(script)
    return dead


def _remove(path: Path) -> bool:
    """Delete one path, reporting whether it is gone. Never raises.

    A single unreadable or already-vanished entry must not fail a whole pass: what matters is
    that the guidance an agent could have read is gone, and the caller's returned list says
    which paths that was.
    """
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        return not path.exists()
    return True


def scrub_venv(venv_dir: Path) -> list[Path]:
    """Remove the target's own agent guidance from an installed venv, in place.

    Called by :func:`acumen.env.prepare_target` after the package is installed and before any
    agent can reach the venv, which is the one part of the target a benchmark run reads. Only
    guidance is touched, so the package still imports and behaves identically — the removed
    files are data an agent reads, never code the package runs.

    Idempotent, so it can run on a cache hit as well as on a fresh build: a second call over a
    scrubbed venv finds nothing and returns an empty list.

    Returns
    -------
    The paths removed, outermost first.
    """
    targets = find_guidance(venv_dir)
    targets += _dead_scripts(venv_dir, targets)
    return [path for path in targets if _remove(path)]


# ── Hiding it from an agent that reads the source ───────────────────────────────────────


def _copy_ignore(dir_path: str, names: list[str]) -> set[str]:
    """A ``shutil.copytree`` ignore callback that drops guidance artifacts (and ``.git``).

    Skill directories and agent-instruction files are left out of the copy so an agent reading
    it physically cannot reach them. ``.git`` goes too: it holds every stripped file in its
    object store, and it is not what any of these agents are here to read.
    """
    drop = set()
    here = Path(dir_path)
    for name in names:
        if name == ".git" or name in GUIDANCE_DIRS or name in GUIDANCE_FILES:
            drop.add(name)
        elif name in SKILL_DIRS and _holds_skill(here / name):
            drop.add(name)
    if SKILL_FILE in drop and REFERENCES_DIR in names:
        drop.add(REFERENCES_DIR)
    return drop


def build_filtered_source(src: Path, dest: Path) -> Path:
    """Copy ``src`` to ``dest`` with skills and agent guidance stripped out.

    This is the structural half of the isolation: an agent's read policy points at the returned
    copy, not the real checkout, so existing skills are simply absent from what it can read. The
    checkout itself is never modified — ``ship`` commits from it, and a local target's checkout
    is the user's own working tree.

    Parameters
    ----------
    src
        The real target source checkout.
    dest
        Where to write the filtered copy; must not already exist.

    Returns
    -------
    ``dest``.
    """
    shutil.copytree(src, dest, ignore=_copy_ignore, symlinks=True)
    return dest


def _artifact_hit(candidate: str, original_src: Path) -> str | None:
    """Return ``candidate`` if it resolves to a skill/guidance artifact or the original tree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved.name in GUIDANCE_FILES:
        return candidate
    if set(resolved.parts) & GUIDANCE_DIRS:
        return candidate
    # The unfiltered source tree is off-limits — the agent must read the filtered copy, so any
    # path back into the original checkout (which still holds the stripped artifacts) is denied.
    try:
        resolved.relative_to(original_src)
    except ValueError:
        return None
    return candidate


def find_skill_access(tool_name: str, tool_input: dict[str, Any], original_src: Path) -> str | None:
    """Return the first path in a tool call that reaches a skill/guidance artifact, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without standing up
    an agent (mirrors :func:`acumen.improve.find_test_access`). Checks the path-bearing
    tool_input keys and — for shell tools — the metacharacter-split command tokens, since a Bash
    call can name a path no structured field would.

    Parameters
    ----------
    tool_name
        The tool being invoked; unused today but kept so the guard can special-case tools.
    tool_input
        The tool's arguments.
    original_src
        The real (unfiltered) source checkout, resolved by the caller.

    Returns
    -------
    The offending path string, or ``None`` if the call touches no artifact.
    """
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            hit = _artifact_hit(value, original_src)
            if hit is not None:
                return hit
    command = tool_input.get("command")
    if isinstance(command, str):
        for raw in command.translate(_SHELL_SPLIT).split():
            token = raw.rstrip(",;")
            if not token:
                continue
            hit = _artifact_hit(token, original_src)
            if hit is not None:
                return hit
    return None


def make_skill_guard(original_src: Path) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies an agent any existing skill/guidance.

    ``matcher=None`` fires the hook for every tool. Paths are resolved against the real source
    checkout, so the guard holds regardless of the agent's ``cwd``.
    """
    # Imported here, not at module scope: the Claude SDK is an optional dependency and a
    # Codex-only install never builds an SDK hook.
    from claude_agent_sdk import HookMatcher

    root = original_src.resolve()

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_skill_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen hides the target's existing skills and agent-instruction files so they "
                    f"cannot bias what this agent writes ({hit}). Work from the package's API and its "
                    "user-facing docs only, using the provided source copy."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])
