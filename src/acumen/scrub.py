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


def _within(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` or nested under it. Both are expected already resolved."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _artifact_hit(candidate: str, original_src: Path, exempt: Sequence[Path] = ()) -> str | None:
    """Return ``candidate`` if it resolves to a skill/guidance artifact or the original tree.

    ``exempt`` names the agent's own writable work tree. A path under one of those roots is never
    an artifact to hide, *even when it is named* ``SKILL.md``: that is exactly what the improver
    must write into its staging directory (and the seeded parent skill it edits, and the skill
    body a wiki run reads) — all of which live under the agent's ``work`` dir. Without this the
    guard, whose job is to hide the *target's* shipped guidance, would also deny the agent the one
    file it exists to produce. Exemption is checked first, so it wins over the name/dir matches.
    """
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if any(_within(resolved, root) for root in exempt):
        return None
    if resolved.name in GUIDANCE_FILES:
        return candidate
    if set(resolved.parts) & GUIDANCE_DIRS:
        return candidate
    # The unfiltered source tree is off-limits — the agent must read the filtered copy, so any
    # path back into the original checkout (which still holds the stripped artifacts) is denied.
    if _within(resolved, original_src):
        return candidate
    return None


def find_skill_access(
    tool_name: str, tool_input: dict[str, Any], original_src: Path, exempt: Sequence[Path] = ()
) -> str | None:
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
    exempt
        The agent's own writable work roots, resolved by the caller. Paths under any of them are
        never flagged, so the agent can write and read its own staging skill (see
        :func:`_artifact_hit`).

    Returns
    -------
    The offending path string, or ``None`` if the call touches no artifact.
    """
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            hit = _artifact_hit(value, original_src, exempt)
            if hit is not None:
                return hit
    command = tool_input.get("command")
    if isinstance(command, str):
        for raw in command.translate(_SHELL_SPLIT).split():
            token = raw.rstrip(",;")
            if not token:
                continue
            hit = _artifact_hit(token, original_src, exempt)
            if hit is not None:
                return hit
    return None


def _iter_strings(value: Any) -> Any:
    """Yield every string anywhere inside a tool_input (dict/list/scalar), provider-agnostic.

    Claude carries a Bash command under ``command`` and a read path under ``file_path``; Codex
    nests the argv differently. Walking all string leaves means one matcher covers both without
    knowing either schema.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _source_needles(repo: str | None) -> frozenset[str]:
    """Lowercased substrings that identify the target's own repository.

    ``owner/repo`` is the load-bearing one — every clone/fetch/archive URL and every
    ``gh repo clone`` names it — with ``host/owner/repo`` added for extra specificity. A local
    ``repo`` (a filesystem path, not a URL) yields nothing: its source is not fetchable over the
    network and the sandbox never contains it, so containment already covers it.
    """
    if not repo:
        return frozenset()
    text = repo.strip().lower().rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    for prefix in ("https://", "http://", "ssh://", "git://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace(":", "/")  # git@github.com:owner/repo -> git@github.com/owner/repo
    head = text.split("/", 1)[0]
    if "@" in head:  # strip userinfo like git@
        text = text.split("@", 1)[1]
    parts = [part for part in text.split("/") if part]
    needles: set[str] = set()
    if len(parts) >= 3:
        host, owner, name = parts[0], parts[-2], parts[-1]
        needles.add(f"{owner}/{name}")
        needles.add(f"{host}/{owner}/{name}")
    return frozenset(needles)


def _normalise_token(value: str) -> str:
    token = value.strip().lower().rstrip("/")
    if token.endswith(".git"):
        token = token[:-4]
    return token.replace(":", "/")


_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".whl")


def find_source_fetch(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    repo: str | None = None,
    pkg_name: str | None = None,
) -> str | None:
    """Return the first token that fetches the target's own source repo/distribution, else ``None``.

    Pure and side-effect free, so it is unit-testable without an agent (mirrors
    :func:`find_skill_access`). It flags ``git clone``/``fetch``, ``gh repo clone``,
    ``git+<repo>`` installs and any ``curl``/``wget`` of a ``github.com``/``codeload``/
    ``raw.githubusercontent`` URL — all of which must name ``owner/repo`` — plus a narrow
    check for downloading the package's own source archive from an index.

    It deliberately does **not** block reads of skill/guidance files: in the skill arm the agent
    legitimately reads its own installed skill under ``<sandbox>/.claude/skills/``. Blocking the
    fetch is what matters — without the clone there is no external skill tree to read.
    """
    needles = _source_needles(repo)
    pkg = pkg_name.lower() if pkg_name else None
    if not needles and not pkg:
        return None
    for value in _iter_strings(tool_input or {}):
        token = _normalise_token(value)
        if any(needle in token for needle in needles):
            return value
        if pkg and pkg in token:
            # A source archive of the package (``…/pkg-1.2.3.tar.gz``) …
            if any(suffix in token for suffix in _ARCHIVE_SUFFIXES):
                return value
            # … or a pip/uv install/download of the package. The benchmark forbids installing
            # anything (the package is already present), so naming it here is only ever an
            # attempt to fetch a fresh, unscrubbed copy.
            if ("pip" in token or "uv " in token) and ("install" in token or "download" in token):
                return value
    return None


def make_source_guard(repo: str | None, pkg_name: str | None) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies a benchmark agent the target's own source.

    ``matcher=None`` fires the hook for every tool. Used only by benchmark runs; meta-agents get
    :func:`make_skill_guard` instead, and ``ship`` gets neither.
    """
    from claude_agent_sdk import HookMatcher

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_source_fetch(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            repo=repo,
            pkg_name=pkg_name,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "acumen benchmarks against the already-installed package only; fetching the "
                    f"target's source repository or distribution is not permitted ({hit}). Work "
                    "from the installed package."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])


def make_skill_guard(original_src: Path, exempt: Sequence[Path] = ()) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies an agent any existing skill/guidance.

    ``matcher=None`` fires the hook for every tool. Paths are resolved against the real source
    checkout, so the guard holds regardless of the agent's ``cwd``.

    ``exempt`` names the agent's own writable work tree (its ``work`` dir). Paths under it are
    never denied — the guard hides the *target's* shipped guidance, not the skill the agent is
    itself writing or editing there, which is legitimately named ``SKILL.md``.
    """
    # Imported here, not at module scope: the Claude SDK is an optional dependency and a
    # Codex-only install never builds an SDK hook.
    from claude_agent_sdk import HookMatcher

    root = original_src.resolve()
    exempt_roots = tuple(dict.fromkeys(path.resolve() for path in exempt))

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_skill_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
            exempt_roots,
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
