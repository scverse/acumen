"""Locating agent transcripts and rendering them — a thin layer over :mod:`acumen.trajectory`.

Every harness records a run in its own native format, but acumen renders them all through one
path: map the native record into the harness-neutral :class:`~acumen.trajectory.Trajectory`, then
hand it to the single renderer. This module keeps the per-run plumbing — finding the Claude
SDK-native transcript on disk, reading a saved Codex event stream — and the provider dispatch;
the model, mappers and renderer live in :mod:`acumen.trajectory`.

Claude is no longer handed to an external ``claude-code-log`` CLI: both providers map into the
same model and render through the same code, so the two reports finally look alike. Nothing here
imports the Claude SDK at module scope, so a Codex-only install still renders its own runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acumen.trajectory import (
    Trajectory,
    from_claude_transcript,
    from_codex_events,
    render_trajectory,
    write_trajectory_json,
)


def locate_transcript(config_dir: Path, work_dir: Path, session_id: str) -> Path | None:
    """Find the SDK-native transcript for a finished Claude run.

    The ``claude`` CLI writes transcripts to
    ``<config_dir>/projects/<project_key_for(work_dir)>/<session_id>.jsonl``. That encoding is
    deterministic, but fall back to a glob rather than lose the transcript if the CLI ever keys
    the directory differently.

    Parameters
    ----------
    config_dir
        The ``CLAUDE_CONFIG_DIR`` the agent ran under (throwaway, for isolated agents; the
        run-local one used by the agent).
    work_dir
        The agent's ``cwd`` — what the project key is computed from.
    session_id
        ``ResultMessage.session_id`` for the run.

    Returns
    -------
    The transcript path, or ``None`` if it could not be found.
    """
    # Imported lazily: a Codex-only install has no Claude SDK, and this module is imported by
    # the runner on every run regardless of provider.
    from claude_agent_sdk import project_key_for_directory

    transcript_root = config_dir / "projects"
    key = project_key_for_directory(str(work_dir))
    direct = transcript_root / key / f"{session_id}.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(transcript_root.glob(f"**/{session_id}.jsonl"))
    return matches[0] if matches else None


def _read_events(jsonl: Path) -> list[dict[str, Any]] | None:
    """Read a saved Codex event stream (one JSON object per line). ``None`` if unreadable."""
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def render_transcript(jsonl: Path, html: Path, usage: dict[str, Any] | None = None, *, prompt: str = "") -> bool:
    """Render a Claude SDK-native transcript to HTML through the unified renderer.

    ``usage`` is the run's authoritative ``ResultMessage.usage`` for the footer; without it the
    footer falls back to the session file's per-message usage, which overcounts.
    """
    traj = from_claude_transcript(jsonl, prompt=prompt, usage=usage)
    if traj is None:
        return False
    return render_trajectory(traj, html)


def render_codex_events(
    events: list[dict[str, Any]],
    html: Path,
    usage: dict[str, Any] | None = None,
    prompt: str = "",
) -> bool:
    """Render a ``codex exec --json`` event stream (already parsed) to HTML."""
    return render_trajectory(from_codex_events(events, prompt=prompt, usage=usage), html)


def render_codex_transcript(
    jsonl: Path,
    html: Path,
    usage: dict[str, Any] | None = None,
    prompt: str = "",
) -> bool:
    """Render a saved Codex event stream (one JSON object per line) to HTML."""
    events = _read_events(jsonl)
    if events is None:
        return False
    return render_codex_events(events, html, usage, prompt)


def render_agent_transcript(
    jsonl: Path,
    html: Path,
    *,
    provider: str,
    usage: dict[str, Any] | None = None,
    prompt: str = "",
) -> bool:
    """Render a run's transcript with the mapper that understands its harness, one renderer for all."""
    if provider == "codex":
        return render_codex_transcript(jsonl, html, usage, prompt)
    # Claude's own transcript already carries the prompt as its first user message, so it needs
    # no injected copy — call positionally so a monkeypatched stub with *args stays satisfied.
    return render_transcript(jsonl, html, usage)


def build_trajectory(
    jsonl: Path,
    *,
    provider: str,
    prompt: str = "",
    usage: dict[str, Any] | None = None,
) -> Trajectory | None:
    """Map a saved run transcript into a :class:`Trajectory` for rendering and ``trajectory.json``.

    ``usage`` is the run's authoritative usage. For Codex it is the only source on a capped run
    whose stream carries no ``turn.completed``; for Claude it is the billed ``ResultMessage.usage``,
    which the footer must report rather than re-summing the session file's per-message usage (that
    overcounts, since each turn re-counts the cached context).
    """
    if provider == "codex":
        events = _read_events(jsonl)
        return None if events is None else from_codex_events(events, prompt=prompt, usage=usage)
    return from_claude_transcript(jsonl, prompt=prompt, usage=usage)


__all__ = [
    "Trajectory",
    "build_trajectory",
    "locate_transcript",
    "render_agent_transcript",
    "render_codex_events",
    "render_codex_transcript",
    "render_transcript",
    "render_trajectory",
    "write_trajectory_json",
]
