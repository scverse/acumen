"""A harness-neutral transcript model, one mapper per harness, and a single HTML renderer.

acumen drives more than one agent harness — Claude Code and Codex today, more later — and each
records a run in its own native format. Rather than carry a renderer per harness, a run is mapped
once into the small :class:`Trajectory` model below, and one renderer turns any trajectory into
HTML. Adding a harness then costs one mapper, not a new renderer.

The model is acumen-native and deliberately lean, but its shape follows harbor's ATIF (Agent
Trajectory Interchange Format): a trajectory is an ordered list of steps, each originating from
the system, the user, or the agent, carrying text, reasoning, the tool calls the agent made, and
the observations those calls produced. A command and its output live on one step, so they render
together regardless of which harness produced them.

Nothing here imports the Claude SDK: the Claude mapper reads the SDK-native session file as plain
JSON, so a Codex-only install renders its own transcripts without the SDK installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal

#: Bumped when the on-disk ``trajectory.json`` shape changes in a way a reader must notice.
SCHEMA = "acumen-transcript-v1"

Source = Literal["system", "user", "agent"]


@dataclass(frozen=True)
class Metrics:
    """Token and cost accounting for a step or a whole trajectory."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None

    def is_empty(self) -> bool:
        """Whether nothing was recorded — used to drop an empty usage footer."""
        return all(v is None for v in (self.input_tokens, self.output_tokens, self.cached_tokens, self.cost_usd))

    def to_dict(self) -> dict[str, Any]:
        """The non-``None`` fields, for ``trajectory.json``."""
        keys = ("input_tokens", "output_tokens", "cached_tokens", "cost_usd")
        return {key: getattr(self, key) for key in keys if getattr(self, key) is not None}


@dataclass(frozen=True)
class ToolCall:
    """One action the agent took — a command, a file edit, an MCP call."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict of this call."""
        return {"call_id": self.call_id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class Observation:
    """The result a tool call produced, or environment feedback after an action."""

    call_id: str | None = None
    content: str = ""
    is_error: bool = False
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict, omitting absent fields."""
        out: dict[str, Any] = {"content": self.content, "is_error": self.is_error}
        if self.call_id is not None:
            out["call_id"] = self.call_id
        if self.exit_code is not None:
            out["exit_code"] = self.exit_code
        return out


@dataclass(frozen=True)
class Step:
    """One turn in the interaction: who spoke, what they said, what they did and observed."""

    index: int
    source: Source
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    observations: tuple[Observation, ...] = ()
    incomplete: bool = False

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict, omitting empty fields."""
        out: dict[str, Any] = {"index": self.index, "source": self.source}
        if self.text:
            out["text"] = self.text
        if self.reasoning:
            out["reasoning"] = self.reasoning
        if self.tool_calls:
            out["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.observations:
            out["observations"] = [obs.to_dict() for obs in self.observations]
        if self.incomplete:
            out["incomplete"] = True
        return out


@dataclass(frozen=True)
class Trajectory:
    """A complete run, mapped from one harness's native record into acumen's neutral model."""

    harness: str
    session_id: str = ""
    model: str = ""
    prompt: str = ""
    steps: tuple[Step, ...] = ()
    usage: Metrics | None = None
    errors: tuple[str, ...] = ()
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict for the portable ``trajectory.json`` artifact."""
        out: dict[str, Any] = {"schema": self.schema, "harness": self.harness}
        if self.session_id:
            out["session_id"] = self.session_id
        if self.model:
            out["model"] = self.model
        if self.prompt:
            out["prompt"] = self.prompt
        out["steps"] = [step.to_dict() for step in self.steps]
        if self.usage is not None and not self.usage.is_empty():
            out["usage"] = self.usage.to_dict()
        if self.errors:
            out["errors"] = list(self.errors)
        return out


# ── Builders (mutable, used only while mapping) ──────────────────────────────────────────


@dataclass
class _StepBuilder:
    index: int
    source: Source
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    incomplete: bool = False

    def build(self) -> Step:
        return Step(
            index=self.index,
            source=self.source,
            text=self.text,
            reasoning=self.reasoning,
            tool_calls=tuple(self.tool_calls),
            observations=tuple(self.observations),
            incomplete=self.incomplete,
        )


def _metrics_from(usage: dict[str, Any] | None) -> Metrics | None:
    """Map a provider usage dict onto :class:`Metrics`, tolerating each provider's key names."""
    if not isinstance(usage, dict):
        return None

    def pick(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return int(value)
        return None

    cost = usage.get("cost_usd")
    metrics = Metrics(
        input_tokens=pick("input_tokens", "prompt_tokens"),
        output_tokens=pick("output_tokens", "completion_tokens"),
        cached_tokens=pick("cached_tokens", "cache_read_input_tokens", "cached_input_tokens"),
        cost_usd=float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None,
    )
    return None if metrics.is_empty() else metrics


# ── Codex mapper ─────────────────────────────────────────────────────────────────────────


def from_codex_events(
    events: list[dict[str, Any]],
    *,
    prompt: str = "",
    usage: dict[str, Any] | None = None,
) -> Trajectory:
    """Map a ``codex exec --json`` event stream into a :class:`Trajectory`.

    Each item becomes an agent step. An item that only ever ``started`` — what a run stopped at
    its turn cap leaves behind — is kept and flagged ``incomplete``. ``usage`` is the tally the
    caller recorded; a capped run has no ``turn.completed`` to read one from, so passing it keeps
    the footer populated on exactly the runs whose spend is most worth seeing.
    """
    session = ""
    tally: dict[str, Any] | None = dict(usage) if usage else None
    errors: list[str] = []
    started: dict[str, _StepBuilder] = {}
    steps: list[_StepBuilder] = []
    counter = 0

    def add(item: dict[str, Any], *, incomplete: bool) -> _StepBuilder:
        nonlocal counter
        counter += 1
        builder = _codex_step(item, counter, incomplete=incomplete)
        steps.append(builder)
        return builder

    for event in events:
        kind = event.get("type")
        if kind == "thread.started":
            session = str(event.get("thread_id") or "")
        elif kind == "turn.completed":
            raw = event.get("usage")
            if isinstance(raw, dict):
                tally = raw
        elif kind in {"error", "turn.failed"}:
            errors.append(str(event.get("message") or event.get("error") or event))
        elif kind in {"item.started", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or f"_{counter}")
            if kind == "item.started":
                started[key] = add(item, incomplete=True)
            else:
                pending = started.pop(key, None)
                if pending is not None:
                    steps.remove(pending)
                    counter -= 1
                add(item, incomplete=False)

    return Trajectory(
        harness="codex",
        session_id=session,
        prompt=prompt,
        steps=tuple(builder.build() for builder in steps),
        usage=_metrics_from(tally),
        errors=tuple(errors),
    )


def _codex_step(item: dict[str, Any], index: int, *, incomplete: bool) -> _StepBuilder:
    """Map one Codex item onto a step builder."""
    builder = _StepBuilder(index=index, source="agent", incomplete=incomplete)
    item_type = str(item.get("type") or "item")
    call_id = str(item.get("id") or f"item-{index}")
    if item_type == "agent_message":
        builder.text = str(item.get("text") or "")
    elif item_type == "reasoning":
        builder.reasoning = str(item.get("text") or "")
    elif item_type == "command_execution":
        command = str(item.get("command") or "")
        builder.tool_calls.append(ToolCall(call_id=call_id, name="command_execution", arguments={"command": command}))
        output = str(item.get("aggregated_output") or "")
        exit_code = item.get("exit_code")
        exit_code = int(exit_code) if isinstance(exit_code, int) else None
        if output.strip() or exit_code is not None:
            builder.observations.append(
                Observation(
                    call_id=call_id,
                    content=output,
                    is_error=exit_code not in (0, None),
                    exit_code=exit_code,
                )
            )
    else:
        # file_change, mcp_tool_call, web_search, todo_list, and anything Codex adds later: keep
        # the whole item as a tool call rather than silently drop what we have no template for.
        arguments = {key: value for key, value in item.items() if key not in {"id", "type"}}
        builder.tool_calls.append(ToolCall(call_id=call_id, name=item_type, arguments=arguments))
    return builder


# ── Claude Code mapper ─────────────────────────────────────────────────────────────────────


def from_claude_records(records: list[dict[str, Any]], *, prompt: str = "") -> Trajectory:
    """Map Claude's SDK-native session records (already parsed) into a :class:`Trajectory`.

    Assistant messages become agent steps (text, thinking as reasoning, ``tool_use`` blocks as
    tool calls); a user message's ``tool_result`` blocks become observations attached to the step
    that issued the matching call. The run's first plain-string user message is the prompt.
    """
    session = ""
    model = ""
    discovered_prompt = ""
    errors: list[str] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    saw_usage = False
    by_call: dict[str, _StepBuilder] = {}
    steps: list[_StepBuilder] = []

    for record in records:
        session = session or str(record.get("sessionId") or record.get("session_id") or "")
        rtype = record.get("type")
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        content = message.get("content")
        if rtype == "assistant":
            model = model or str(message.get("model") or "")
            usage = message.get("usage")
            if isinstance(usage, dict):
                saw_usage = True
                usage_totals["input_tokens"] += int(usage.get("input_tokens") or 0)
                usage_totals["output_tokens"] += int(usage.get("output_tokens") or 0)
                usage_totals["cached_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
            builder = _StepBuilder(index=len(steps) + 1, source="agent")
            for block in content if isinstance(content, list) else []:
                _apply_assistant_block(block, builder, by_call)
            if builder.text or builder.reasoning or builder.tool_calls:
                steps.append(builder)
        elif rtype == "user":
            if isinstance(content, str):
                if not discovered_prompt and content.strip():
                    discovered_prompt = content
                continue
            _apply_user_results(content, steps, by_call)
        elif rtype == "result" and record.get("is_error"):
            detail = record.get("subtype") or record.get("result") or record.get("error")
            errors.append(str(detail or "run reported an error"))

    metrics = _metrics_from(usage_totals) if saw_usage else None
    return Trajectory(
        harness="claude-code",
        session_id=session,
        model=model,
        prompt=discovered_prompt or prompt,
        steps=tuple(builder.build() for builder in steps),
        usage=metrics,
        errors=tuple(errors),
    )


def _apply_assistant_block(block: Any, builder: _StepBuilder, by_call: dict[str, _StepBuilder]) -> None:
    if not isinstance(block, dict):
        return
    btype = block.get("type")
    if btype == "text":
        text = str(block.get("text") or "")
        builder.text = f"{builder.text}\n{text}".strip() if builder.text else text.strip()
    elif btype == "thinking":
        thought = str(block.get("thinking") or "")
        builder.reasoning = f"{builder.reasoning}\n{thought}".strip() if builder.reasoning else thought.strip()
    elif btype == "tool_use":
        call_id = str(block.get("id") or f"call-{builder.index}-{len(builder.tool_calls)}")
        arguments = block.get("input")
        builder.tool_calls.append(
            ToolCall(
                call_id=call_id,
                name=str(block.get("name") or "tool"),
                arguments=arguments if isinstance(arguments, dict) else {"input": arguments},
            )
        )
        by_call[call_id] = builder


def _apply_user_results(content: Any, steps: list[_StepBuilder], by_call: dict[str, _StepBuilder]) -> None:
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        call_id = str(block.get("tool_use_id") or "")
        observation = Observation(
            call_id=call_id or None,
            content=_result_text(block.get("content")),
            is_error=bool(block.get("is_error")),
        )
        owner = by_call.get(call_id)
        if owner is not None:
            owner.observations.append(observation)
        else:
            step = _StepBuilder(index=len(steps) + 1, source="user")
            step.observations.append(observation)
            steps.append(step)


def _result_text(content: Any) -> str:
    """Flatten a tool_result's content — a string, or a list of text/blocks — to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return "" if content is None else str(content)


def from_claude_transcript(jsonl: Path, *, prompt: str = "") -> Trajectory | None:
    """Read a Claude SDK-native session file and map it. ``None`` if the file cannot be read."""
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return from_claude_records(records, prompt=prompt)


# ── The one renderer ─────────────────────────────────────────────────────────────────────

_CSS = """\
:root { color-scheme: light dark; }
body { font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0 auto; max-width: 60rem; padding: 2rem 1rem; }
h1 { font-size: 1.25rem; margin: 0 0 .25rem; }
.meta { color: #6b7280; font-size: .8rem; margin-bottom: 1.5rem; }
.item { border-left: 3px solid #d1d5db; margin: 0 0 1rem; padding: .25rem 0 .25rem .75rem; }
.item > .label { color: #6b7280; font-size: .7rem; letter-spacing: .04em; text-transform: uppercase; }
.prompt { border-left-color: #4b8b9b; }
.agent { border-left-color: #6b8f71; }
.user { border-left-color: #7f9cb5; }
.system { border-left-color: #b6a8c9; }
.failed { border-left-color: #c0685c; }
details.tool, details.reasoning { margin-top: .4rem; }
details.reasoning { color: #6b7280; }
summary { cursor: pointer; color: #6b7280; font-size: .8rem; }
summary .name { color: inherit; font-weight: 600; }
summary .preview { color: #9ca3af; font-weight: 400; }
pre { background: #00000010; border-radius: .25rem; margin: .35rem 0 0; overflow-x: auto; padding: .5rem .65rem; white-space: pre-wrap; word-break: break-word; }
.md { overflow-wrap: anywhere; }
.md > :first-child { margin-top: .2rem; }
.md > :last-child { margin-bottom: 0; }
.md p { margin: .5rem 0; }
.md h3, .md h4, .md h5, .md h6 { margin: .8rem 0 .3rem; font-size: 1rem; }
.md ul, .md ol { margin: .4rem 0; padding-left: 1.4rem; }
.md code { background: #00000010; border-radius: .2rem; padding: .05rem .3rem; font-size: .9em; }
.md pre code { background: none; padding: 0; }
.md a { color: #4b8b9b; }
.md table { border-collapse: collapse; margin: .6rem 0; font-size: .9em; }
.md th, .md td { border: 1px solid #d1d5db; padding: .25rem .55rem; text-align: left; }
.md th { background: #00000010; }
.exit { color: #6b7280; font-size: .75rem; }
table { border-collapse: collapse; font-size: .8rem; margin-top: .5rem; }
td { border-top: 1px solid #d1d5db; padding: .2rem .75rem .2rem 0; }
td.n { text-align: right; }
"""

#: How much of one tool observation the page keeps. The full output is in the JSONL beside it.
_OUTPUT_CAP = 20_000
#: How much of a tool call to show in its collapsed summary before the reader expands it.
_PREVIEW_CAP = 90


def _clip(text: str) -> str:
    if len(text) <= _OUTPUT_CAP:
        return text
    return text[:_OUTPUT_CAP] + f"\n… {len(text) - _OUTPUT_CAP} more characters"


def _md_inline(text: str) -> str:
    """Render inline markdown (code, bold, italic, links) on one line, escaping HTML first."""
    codes: list[str] = []

    def stash(match: re.Match[str]) -> str:
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{escape(codes[int(m.group(1))])}</code>", text)


def _table_cells(line: str) -> list[str]:
    """Split one pipe-table row into cells, dropping the optional outer pipes."""
    stripped = line.strip()
    stripped = stripped.removeprefix("|").removesuffix("|")
    return stripped.split("|")


def _is_table_delimiter(line: str) -> bool:
    """Whether ``line`` is a GFM table delimiter row (``| --- | :--: |``)."""
    cells = _table_cells(line)
    return "|" in line and bool(cells) and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells)


def _table_align(spec: str) -> str:
    spec = spec.strip()
    left, right = spec.startswith(":"), spec.endswith(":")
    if left and right:
        return "center"
    if right:
        return "right"
    return "left" if left else ""


def _render_table(header: list[str], aligns: list[str], rows: list[list[str]]) -> str:
    def cell(tag: str, text: str, index: int) -> str:
        align = aligns[index] if index < len(aligns) else ""
        style = f' style="text-align:{align}"' if align else ""
        return f"<{tag}{style}>{_md_inline(text.strip())}</{tag}>"

    head = "<tr>" + "".join(cell("th", value, i) for i, value in enumerate(header)) + "</tr>"
    body = "".join("<tr>" + "".join(cell("td", value, i) for i, value in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _markdown(text: str) -> str:
    """A small, safe markdown-to-HTML renderer for agent messages — headings, lists, code, inline.

    Deliberately a common-case subset (not full CommonMark): agent prose is paragraphs, bullet
    lists, fenced code and inline emphasis. Everything is HTML-escaped, so no message can inject
    markup, and anything unrecognized falls through as plain text.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []
    list_tag = ""

    def flush_para() -> None:
        if para:
            out.append("<p>" + "<br>".join(_md_inline(line) for line in para) + "</p>")
            para.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if items:
            out.append(f"<{list_tag}>" + "".join(f"<li>{_md_inline(it)}</li>" for it in items) + f"</{list_tag}>")
            items.clear()
            list_tag = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            flush_list()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # skip the closing fence
            out.append(f"<pre><code>{escape(_clip(chr(10).join(code)))}</code></pre>")
            continue
        if "|" in stripped and i + 1 < len(lines) and _is_table_delimiter(lines[i + 1]):
            flush_para()
            flush_list()
            header = _table_cells(line)
            aligns = [_table_align(spec) for spec in _table_cells(lines[i + 1])]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(_table_cells(lines[i]))
                i += 1
            out.append(_render_table(header, aligns, rows))
            continue
        if not stripped:
            flush_para()
            flush_list()
        elif heading := re.match(r"(#{1,6})\s+(.*)", stripped):
            flush_para()
            flush_list()
            level = min(len(heading.group(1)) + 2, 6)  # start at h3 so a message never out-shouts the page
            out.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
        elif bullet := re.match(r"[-*+]\s+(.*)", stripped):
            flush_para()
            if list_tag and list_tag != "ul":
                flush_list()
            list_tag = "ul"
            items.append(bullet.group(1))
        elif ordered := re.match(r"\d+[.)]\s+(.*)", stripped):
            flush_para()
            if list_tag and list_tag != "ol":
                flush_list()
            list_tag = "ol"
            items.append(ordered.group(1))
        else:
            flush_list()
            para.append(stripped)
        i += 1
    flush_para()
    flush_list()
    return "".join(out)


def _call_preview(call: ToolCall) -> str:
    if call.name == "command_execution":
        body = str(call.arguments.get("command") or "")
    elif list(call.arguments) == ["input"] and isinstance(call.arguments["input"], str):
        body = call.arguments["input"]
    else:
        body = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    preview = " ".join(body.split())
    return preview if len(preview) <= _PREVIEW_CAP else preview[:_PREVIEW_CAP] + "…"


def _call_body(call: ToolCall) -> str:
    if call.name == "command_execution":
        return str(call.arguments.get("command") or "")
    if list(call.arguments) == ["input"] and isinstance(call.arguments["input"], str):
        return call.arguments["input"]
    return json.dumps(call.arguments, indent=2, default=str) if call.arguments else ""


def _observation_html(obs: Observation) -> str:
    parts: list[str] = []
    if obs.content.strip():
        parts.append(f"<pre>{escape(_clip(obs.content))}</pre>")
    if obs.exit_code is not None:
        parts.append(f'<div class="exit">exit {escape(str(obs.exit_code))}</div>')
    return "".join(parts)


def _tool_details(call: ToolCall, observations: list[Observation]) -> str:
    """Render one tool call and its results in a collapsed ``<details>`` toggle."""
    errored = any(obs.is_error for obs in observations)
    preview = _call_preview(call)
    summary = f'<span class="name">{escape(call.name)}</span>'
    if preview:
        summary += f' <span class="preview">{escape(preview)}</span>'
    body = _call_body(call)
    inner = f"<pre>{escape(_clip(body))}</pre>" if body.strip() else ""
    inner += "".join(_observation_html(obs) for obs in observations)
    return f'<details class="tool"{" open" if errored else ""}><summary>{summary}</summary>{inner}</details>'


def _step_html(step: Step) -> str:
    classes = f"item {step.source}"
    if step.incomplete or any(obs.is_error for obs in step.observations):
        classes += " failed"
    label = step.source + (" (unfinished)" if step.incomplete else "")
    parts = [f'<div class="{classes}"><div class="label">{escape(label)}</div>']
    if step.reasoning.strip():
        parts.append(
            f'<details class="reasoning"><summary>reasoning</summary>'
            f'<div class="md">{_markdown(step.reasoning)}</div></details>'
        )
    if step.text.strip():
        parts.append(f'<div class="md">{_markdown(step.text)}</div>')
    call_ids = {call.call_id for call in step.tool_calls}
    matched: dict[str, list[Observation]] = {}
    orphans: list[Observation] = []
    for obs in step.observations:
        if obs.call_id is not None and obs.call_id in call_ids:
            matched.setdefault(obs.call_id, []).append(obs)
        else:
            orphans.append(obs)
    for call in step.tool_calls:
        parts.append(_tool_details(call, matched.get(call.call_id, [])))
    for obs in orphans:
        parts.append(
            f'<details class="tool"{" open" if obs.is_error else ""}>'
            f'<summary><span class="name">result</span></summary>{_observation_html(obs)}</details>'
        )
    parts.append("</div>")
    return "".join(parts)


def _usage_html(usage: Metrics | None) -> str:
    if usage is None or usage.is_empty():
        return ""
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td class='n'>{escape(str(value))}</td></tr>"
        for key, value in usage.to_dict().items()
    )
    return f"<table>{rows}</table>"


def render_trajectory(traj: Trajectory, html: Path) -> bool:
    """Render any :class:`Trajectory` to a standalone HTML file — the one centralized renderer."""
    blocks: list[str] = []
    if traj.prompt.strip():
        blocks.append(
            f'<div class="item prompt"><div class="label">prompt</div>'
            f'<div class="md">{_markdown(traj.prompt)}</div></div>'
        )
    blocks.extend(_step_html(step) for step in traj.steps)
    if traj.errors:
        joined = escape("\n".join(traj.errors))
        blocks.append(f'<div class="item failed"><div class="label">error</div><pre>{joined}</pre></div>')

    meta = f"{escape(traj.harness)}"
    if traj.model:
        meta += f" · {escape(traj.model)}"
    meta += f" · thread {escape(traj.session_id)}" if traj.session_id else " · no thread id"
    body = "".join(blocks) or '<div class="item"><div class="label">no events</div></div>'
    footer = _usage_html(traj.usage)
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(traj.harness)} transcript</title><style>{_CSS}</style></head><body>"
        f'<h1>{escape(traj.harness)} transcript</h1><div class="meta">{meta}</div>{body}{footer}'
        "</body></html>\n",
        encoding="utf-8",
    )
    return html.is_file()


def write_trajectory_json(traj: Trajectory, path: Path) -> bool:
    """Write the portable ``trajectory.json`` artifact for ``traj``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(traj.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    return path.is_file()
