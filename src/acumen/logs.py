"""Live agent logs for the meta-agents.

``draft``/``improve``/``tasks``/``ship`` each spin up one long, autonomous agent whose transcript
today is written into a throwaway ``CLAUDE_CONFIG_DIR`` and ``rmtree``'d on return — a black box.
:class:`LiveLog` makes each run observable:

* **A live JSONL**, appended and flushed *one normalized event per SDK message, during the run*,
  at ``logs/acumen-<command>-<datetime>.jsonl``. This is the contract for a master agent watching
  a run by **reading the file** on its own schedule — never a flood of the command's stdout into
  its context ("print, don't read"). The feed is deliberately compact: it does **not** inline huge
  tool results (file dumps, command output), only their status and size.
* **A terminal mirror** when ``--stream`` is on (off by default): each turn's assistant text and a
  one-line-per-tool-call summary printed live for a human watching.
* **An HTML log** rendered at the end from the SDK-native transcript (:func:`finalize`), via the
  same ``claude-code-log`` path the benchmark runner uses. The rich post-hoc artifact; the JSONL
  is the lightweight live feed.

The class is stateful (it correlates tool results back to the tool that produced them) but has no
agent dependency, so it is unit-testable on a synthetic message sequence — like ``cli._Progress``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from acumen.transcript import locate_transcript, render_transcript

#: tool_input keys, in priority order, whose value best summarizes a tool call in one line.
_SUMMARY_KEYS = ("command", "file_path", "path", "notebook_path", "pattern", "url", "query", "description", "prompt")

#: Thinking blocks can be huge; the live feed keeps only a short preview of them.
_THINKING_CAP = 240
#: How long a one-line tool-input summary may get before it is truncated.
_SUMMARY_CAP = 160


def _truncate(text: str, cap: int) -> str:
    text = text.strip()
    return text if len(text) <= cap else text[:cap] + "…"


def _tool_summary(tool_input: Any) -> str:
    """A one-line summary of a tool call's input — the salient argument, whitespace-collapsed."""
    if not isinstance(tool_input, dict):
        return ""
    for key in _SUMMARY_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(" ".join(value.split()), _SUMMARY_CAP)
    parts = [f"{key}={tool_input[key]!r}" for key in list(tool_input)[:3]]
    return _truncate(", ".join(parts), _SUMMARY_CAP)


def _content_len(content: Any) -> int:
    """Character length of a tool result's content — recorded instead of inlining the content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(str(block.get("text", "")))
            else:
                total += len(str(block))
        return total
    return 0


class LiveLog:
    """A live, append-only JSONL log of one meta-agent run, with optional terminal streaming.

    Open one on ``logs/acumen-<command>-<datetime>.jsonl`` with :meth:`open`, feed every SDK
    message to :meth:`append` inside the ``async for`` loop, and call :meth:`finalize` once the
    run is done (before the throwaway config dir is deleted) to render the HTML log. Close it to
    flush and release the file.
    """

    def __init__(
        self,
        jsonl_path: Path,
        *,
        stream: bool = False,
        echo: Callable[[str], None] = print,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.html_path = self.jsonl_path.with_suffix(".html")
        self.stream = stream
        self.html_rendered = False
        self.n_events = 0
        self._echo = echo
        self._clock = clock or (lambda: datetime.now().isoformat(timespec="seconds"))
        self._tool_names: dict[str, str] = {}
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.jsonl_path.open("w", encoding="utf-8")

    @classmethod
    def open(
        cls,
        log_dir: Path,
        command: str,
        *,
        stream: bool = False,
        now: datetime | None = None,
        echo: Callable[[str], None] = print,
    ) -> LiveLog:
        """Open a log for ``command`` under ``log_dir`` with the fixed filename pattern.

        The filename is ``acumen-<command>-<YYYYMMDD-HHMMSS>.jsonl``; the directory is the only
        thing a caller overrides.
        """
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        return cls(Path(log_dir) / f"acumen-{command}-{stamp}.jsonl", stream=stream, echo=echo)

    def append(self, message: Any) -> None:
        """Normalize one SDK message to a compact event, append it (flushed), and maybe echo it."""
        event = self._normalize(message)
        if event is None:
            return
        record = {"ts": self._clock(), **event}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.n_events += 1
        if self.stream:
            for line in self._render_lines(record):
                self._echo(line)

    def finalize(self, *, config_dir: Path, work_dir: Path, result: ResultMessage | None) -> bool:
        """Locate the run's SDK-native transcript and render the HTML log from it.

        Call before the throwaway config dir is removed — the native transcript still lives under
        it at this point. A no-op that returns ``False`` if the run produced no result.
        """
        if result is None:
            return False
        native = locate_transcript(config_dir, work_dir, result.session_id)
        if native is None or not native.is_file():
            return False
        self.html_rendered = render_transcript(native, self.html_path)
        return self.html_rendered

    def close(self) -> None:
        """Flush and release the underlying JSONL file. Idempotent."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> LiveLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Normalization ──────────────────────────────────────────────────────────────────

    def _normalize(self, message: Any) -> dict[str, Any] | None:
        if isinstance(message, AssistantMessage):
            return self._assistant_event(message)
        if isinstance(message, UserMessage):
            return self._user_event(message)
        if isinstance(message, SystemMessage):
            return {"type": "system", "subtype": message.subtype}
        if isinstance(message, ResultMessage):
            return {
                "type": "result",
                "is_error": bool(message.is_error),
                "subtype": message.subtype,
                "turns": message.num_turns,
                "cost_usd": round(message.total_cost_usd or 0.0, 4),
                "duration_s": round((message.duration_ms or 0) / 1000, 2),
                "session_id": message.session_id,
            }
        return None

    def _assistant_event(self, message: AssistantMessage) -> dict[str, Any]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tools: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text.strip():
                    text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thinking_parts.append(block.thinking)
            elif isinstance(block, ToolUseBlock):
                self._tool_names[block.id] = block.name
                tools.append({"id": block.id, "name": block.name, "summary": _tool_summary(block.input)})
        event: dict[str, Any] = {"type": "assistant"}
        if message.model:
            event["model"] = message.model
        text = "\n".join(text_parts).strip()
        if text:
            event["text"] = text
        if thinking_parts:
            event["thinking"] = _truncate("\n".join(thinking_parts), _THINKING_CAP)
        if tools:
            event["tools"] = tools
        return event

    def _user_event(self, message: UserMessage) -> dict[str, Any] | None:
        content = message.content
        if not isinstance(content, list):
            return None  # a plain user turn (our own prompt) — nothing worth an event
        results: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, ToolResultBlock):
                results.append(
                    {
                        "tool_use_id": block.tool_use_id,
                        "name": self._tool_names.get(block.tool_use_id),
                        "status": "error" if block.is_error else "ok",
                        "chars": _content_len(block.content),
                    }
                )
        return {"type": "tool_result", "results": results} if results else None

    # ── Terminal mirror (--stream) ─────────────────────────────────────────────────────

    def _render_lines(self, event: dict[str, Any]) -> list[str]:
        kind = event.get("type")
        lines: list[str] = []
        if kind == "assistant":
            if event.get("thinking"):
                lines.append(f"  · thinking: {event['thinking']}")
            if event.get("text"):
                lines.append(str(event["text"]).strip())
            for tool in event.get("tools", []):
                summary = tool.get("summary") or ""
                lines.append(f"  → {tool['name']}: {summary}".rstrip())
        elif kind == "tool_result":
            for res in event["results"]:
                mark = "✗" if res["status"] == "error" else "✓"
                name = res.get("name") or "tool"
                lines.append(f"  {mark} {name} ({res['chars']} chars)")
        elif kind == "result":
            mark = "✗ error" if event["is_error"] else "● done"
            lines.append(
                f"{mark}: {event['subtype']} · {event['turns']} turns · "
                f"${event['cost_usd']:.2f} · {event['duration_s']}s"
            )
        return lines
