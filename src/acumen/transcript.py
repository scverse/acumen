"""Locating and rendering agent transcripts — shared by the runner and the meta-agents.

The benchmark runner has always done this per run: after an agent finishes, find the
transcript the ``claude`` CLI wrote (under ``<CLAUDE_CONFIG_DIR>/projects/<key>/<session>.jsonl``),
copy it into the run dir, and render it to HTML with ``claude-code-log``. This module factors
that machinery out here so the four single-agent commands (``draft``/``improve``/``tasks``/``ship``)
can render an HTML log of their own agent the same way, instead of losing the transcript to the
``rmtree`` of a throwaway config dir.

The Claude transcript path is fully determined by the throwaway ``CLAUDE_CONFIG_DIR`` and the
agent's ``cwd`` (both of which acumen sets) plus the ``session_id`` on the ``ResultMessage``.

Codex is rendered here rather than handed to ``claude-code-log``: that tool reads the SDK-native
format only, and given a Codex event stream it skips every line, **still exits 0**, and writes an
empty page — a silent wrong answer. :func:`render_codex_transcript` renders the event stream
acumen already captures, so both providers produce a readable HTML log from their own format.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any


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


def claude_code_log() -> str | None:
    """Locate the ``claude-code-log`` CLI.

    It ships with acumen's ``claude`` extra, so it lives next to the interpreter running us —
    look there before PATH, which won't contain the venv's ``bin`` when acumen is invoked by
    absolute path rather than through an activated venv. ``None`` when it is not installed,
    which is the normal state of a Codex-only install.
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


_CODEX_CSS = """\
:root { color-scheme: light dark; }
body { font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0 auto; max-width: 60rem; padding: 2rem 1rem; }
h1 { font-size: 1.25rem; margin: 0 0 .25rem; }
.meta { color: #6b7280; font-size: .8rem; margin-bottom: 1.5rem; }
.item { border-left: 3px solid #d1d5db; margin: 0 0 1rem; padding: .25rem 0 .25rem .75rem; }
.item > .label { color: #6b7280; font-size: .7rem; letter-spacing: .04em; text-transform: uppercase; }
.agent_message { border-left-color: #6b8f71; }
.reasoning { border-left-color: #b6a8c9; color: #6b7280; }
.command_execution { border-left-color: #7f9cb5; }
.file_change { border-left-color: #d0a05a; }
.failed { border-left-color: #c0685c; }
pre { background: #00000010; border-radius: .25rem; margin: .35rem 0 0; overflow-x: auto; padding: .5rem .65rem; white-space: pre-wrap; word-break: break-word; }
.text { white-space: pre-wrap; }
.exit { color: #6b7280; font-size: .75rem; }
table { border-collapse: collapse; font-size: .8rem; margin-top: .5rem; }
td { border-top: 1px solid #d1d5db; padding: .2rem .75rem .2rem 0; }
td.n { text-align: right; }
"""

#: How much of one command's captured output the page keeps. A benchmark agent can print a
#: whole dataframe; the transcript is for reading, and the full output is in the JSONL beside it.
_OUTPUT_CAP = 20_000


def _codex_block(item: dict[str, Any], *, incomplete: bool) -> str:
    """Render one Codex item as an HTML block."""
    kind = str(item.get("type") or "item")
    label = kind.replace("_", " ") + (" (unfinished)" if incomplete else "")
    parts = [
        f'<div class="item {escape(kind)}{" failed" if incomplete else ""}">',
        f'<div class="label">{escape(label)}</div>',
    ]
    if kind in {"agent_message", "reasoning"}:
        parts.append(f'<div class="text">{escape(str(item.get("text") or ""))}</div>')
    elif kind == "command_execution":
        parts.append(f"<pre>{escape(str(item.get('command') or ''))}</pre>")
        output = str(item.get("aggregated_output") or "")
        if output.strip():
            clipped = output[:_OUTPUT_CAP]
            suffix = "" if len(output) <= _OUTPUT_CAP else f"\n… {len(output) - _OUTPUT_CAP} more characters"
            parts.append(f"<pre>{escape(clipped + suffix)}</pre>")
        code = item.get("exit_code")
        if code is not None:
            parts.append(f'<div class="exit">exit {escape(str(code))}</div>')
    else:
        # file_change, mcp_tool_call, web_search, todo_list, and anything Codex adds later:
        # dump the item verbatim rather than silently drop what we have no template for.
        body = {key: value for key, value in item.items() if key not in {"id", "type"}}
        parts.append(f"<pre>{escape(json.dumps(body, indent=2, default=str))}</pre>")
    parts.append("</div>")
    return "".join(parts)


def _codex_usage_table(usage: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td class='n'>{escape(str(value))}</td></tr>" for key, value in usage.items()
    )
    return f"<table>{rows}</table>"


def render_codex_events(events: Iterable[dict[str, Any]], html: Path) -> bool:
    """Render a ``codex exec --json`` event stream to a standalone HTML file.

    Items are rendered in the order Codex completed them. An item that only ever ``started`` —
    what a run terminated at its turn cap leaves behind — is rendered too, marked unfinished, so
    a capped run still shows what it was doing when acumen stopped it.

    Returns
    -------
    Whether the file was written.
    """
    started: dict[str, dict[str, Any]] = {}
    blocks: list[str] = []
    session = ""
    usage: dict[str, Any] = {}
    errors: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "thread.started":
            session = str(event.get("thread_id") or "")
        elif kind == "turn.completed":
            raw = event.get("usage")
            if isinstance(raw, dict):
                usage = raw
        elif kind in {"error", "turn.failed"}:
            errors.append(str(event.get("message") or event.get("error") or event))
        elif kind in {"item.started", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or len(blocks))
            if kind == "item.started":
                started[key] = item
            else:
                started.pop(key, None)
                blocks.append(_codex_block(item, incomplete=False))
    blocks.extend(_codex_block(item, incomplete=True) for item in started.values())
    if errors:
        joined = escape("\n".join(errors))
        blocks.append(f'<div class="item failed"><div class="label">error</div><pre>{joined}</pre></div>')

    meta = f"thread {escape(session)}" if session else "no thread id"
    body = "".join(blocks) or '<div class="item"><div class="label">no events</div></div>'
    footer = _codex_usage_table(usage) if usage else ""
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>codex transcript</title><style>{_CODEX_CSS}</style></head><body>"
        f'<h1>codex transcript</h1><div class="meta">{meta}</div>{body}{footer}'
        "</body></html>\n",
        encoding="utf-8",
    )
    return html.is_file()


def render_codex_transcript(jsonl: Path, html: Path) -> bool:
    """Render a saved Codex event stream (one JSON object per line) to HTML."""
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return render_codex_events(events, html)


def render_agent_transcript(jsonl: Path, html: Path, *, provider: str) -> bool:
    """Render a run's transcript with the renderer that understands its format."""
    return render_codex_transcript(jsonl, html) if provider == "codex" else render_transcript(jsonl, html)
