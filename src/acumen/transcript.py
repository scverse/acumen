"""Locating and rendering SDK-native transcripts — shared by the runner and the meta-agents.

The benchmark runner has always done this per run: after an agent finishes, find the
transcript the ``claude`` CLI wrote (under ``<CLAUDE_CONFIG_DIR>/projects/<key>/<session>.jsonl``),
copy it into the run dir, and render it to HTML with ``claude-code-log``. M8 factors that
machinery out here so the four single-agent commands (``draft``/``improve``/``tasks``/``ship``)
can render an HTML log of their own agent the same way, instead of losing the transcript to the
``rmtree`` of a throwaway config dir.

The transcript path is fully determined by the throwaway ``CLAUDE_CONFIG_DIR`` and the agent's
``cwd`` (both of which acumen sets) plus the ``session_id`` on the ``ResultMessage`` — see §3.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import project_key_for_directory


def locate_transcript(config_dir: Path, work_dir: Path, session_id: str) -> Path | None:
    """Find the SDK-native transcript for a finished agent run.

    The ``claude`` CLI writes transcripts to
    ``<config_dir>/projects/<project_key_for(work_dir)>/<session_id>.jsonl``. That encoding is
    deterministic, but fall back to a glob rather than lose the transcript if the CLI ever keys
    the directory differently.

    Parameters
    ----------
    config_dir
        The ``CLAUDE_CONFIG_DIR`` the agent ran under (throwaway, for isolated agents; the
        user's real one for the unisolated ship agent).
    work_dir
        The agent's ``cwd`` — what the project key is computed from.
    session_id
        ``ResultMessage.session_id`` for the run.

    Returns
    -------
    The transcript path, or ``None`` if it could not be found.
    """
    transcript_root = config_dir / "projects"
    key = project_key_for_directory(str(work_dir))
    direct = transcript_root / key / f"{session_id}.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(transcript_root.glob(f"**/{session_id}.jsonl"))
    return matches[0] if matches else None


def claude_code_log() -> str | None:
    """Locate the ``claude-code-log`` CLI.

    It is a dependency of acumen, so it lives next to the interpreter running us — look there
    before PATH, which won't contain the venv's ``bin`` when acumen is invoked by absolute path
    rather than through an activated venv.
    """
    local = Path(sys.executable).parent / "claude-code-log"
    if local.is_file():
        return str(local)
    return shutil.which("claude-code-log")


def render_transcript(jsonl: Path, html: Path) -> bool:
    """Render an SDK-native transcript to a standalone HTML file via ``claude-code-log``.

    Returns whether the render succeeded and produced the file.
    """
    cli = claude_code_log()
    if cli is None:
        return False
    proc = subprocess.run([cli, str(jsonl), "-o", str(html)], capture_output=True, text=True)
    return proc.returncode == 0 and html.is_file()
