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
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal

from acumen.htmldiff import split_diff_table
from acumen.markdown import clip as _clip
from acumen.markdown import render_markdown as _markdown
from acumen.theme import BAR, DIFF_CSS, INK, PALETTE_ROOT_CSS

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


def _claude_metrics(usage: dict[str, Any] | None) -> Metrics | None:
    """Footer metrics for a Claude run from its authoritative usage (``ResultMessage.usage``).

    This is the one usage figure the whole run is billed on, and it is what ``result.json`` records.
    It must NOT be reconstructed by summing the session file's per-message usage: each Claude turn
    re-sends the whole conversation, so every turn's ``usage`` already counts the cached prefix, and
    summing them across turns overcounts (the same context, billed once, added once per turn). The
    numbers are normalized so the footer matches ``result.json``: ``input`` is the total input
    (fresh + cache read + cache write), ``cached`` is the cache-read count, output as reported.
    """
    if not usage:
        return None
    from acumen.prices import normalize_usage

    norm = normalize_usage(usage, provider="claude")
    metrics = Metrics(input_tokens=norm.input, output_tokens=norm.output, cached_tokens=norm.cache_read)
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


def from_claude_records(
    records: list[dict[str, Any]], *, prompt: str = "", usage: dict[str, Any] | None = None
) -> Trajectory:
    """Map Claude's SDK-native session records (already parsed) into a :class:`Trajectory`.

    Assistant messages become agent steps (text, thinking as reasoning, ``tool_use`` blocks as
    tool calls); a user message's ``tool_result`` blocks become observations attached to the step
    that issued the matching call. The run's first plain-string user message is the prompt.

    ``usage`` is the run's authoritative ``ResultMessage.usage``; when given, the footer reports it
    (matching ``result.json``). It is only reconstructed by summing the session file's per-message
    usage — which overcounts, since each turn re-counts the cached context — when no authoritative
    usage is available (e.g. rendering a bare transcript file with no result to hand).
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
            msg_usage = message.get("usage")
            if isinstance(msg_usage, dict):
                saw_usage = True
                usage_totals["input_tokens"] += int(msg_usage.get("input_tokens") or 0)
                usage_totals["output_tokens"] += int(msg_usage.get("output_tokens") or 0)
                usage_totals["cached_tokens"] += int(msg_usage.get("cache_read_input_tokens") or 0)
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

    metrics = _claude_metrics(usage) if usage else (_metrics_from(usage_totals) if saw_usage else None)
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


def from_claude_transcript(jsonl: Path, *, prompt: str = "", usage: dict[str, Any] | None = None) -> Trajectory | None:
    """Read a Claude SDK-native session file and map it. ``None`` if the file cannot be read.

    ``usage`` is the run's authoritative ``ResultMessage.usage`` for the footer (see
    :func:`from_claude_records`).
    """
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
    return from_claude_records(records, prompt=prompt, usage=usage)


# ── The one renderer ─────────────────────────────────────────────────────────────────────

_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

#: The transcript's stylesheet. It shares the report's palette tokens and the exact split-diff
#: rules (see :mod:`acumen.theme`), so a file edit here reads the same as a skill diff there, while
#: the page stays a single self-contained file with no external stylesheet.
_CSS = f"""\
{PALETTE_ROOT_CSS}
* {{ box-sizing: border-box; }}
body {{ font: 14px/1.6 system-ui, -apple-system, Segoe UI, sans-serif; margin: 0 auto; max-width: 62rem;
        padding: 2rem 1.2rem; background: var(--page); color: var(--ink); }}
h1 {{ font-size: 1.25rem; margin: 0 0 .25rem; }}
.meta {{ color: {BAR}; font-size: .8rem; margin-bottom: 1.5rem; }}
.item {{ border-left: 3px solid {INK}22; margin: 0 0 1rem; padding: .25rem 0 .25rem .85rem; }}
.item > .label {{ color: {BAR}; font-size: .7rem; letter-spacing: .04em; text-transform: uppercase; }}
.prompt {{ border-left-color: #4b8b9b; }}
.agent {{ border-left-color: #6b8f71; }}
.user {{ border-left-color: #7f9cb5; }}
.system {{ border-left-color: #b6a8c9; }}
.failed {{ border-left-color: #a4432b; }}
details.tool, details.reasoning {{ margin-top: .4rem; }}
details.reasoning {{ color: {BAR}; }}
summary {{ cursor: pointer; color: {BAR}; font-size: .8rem; }}
summary .name {{ color: inherit; font-weight: 600; }}
summary .preview {{ color: {BAR}; opacity: .8; font-weight: 400; font-family: {_MONO}; }}
.call {{ margin-top: .35rem; }}
.call .path {{ font-family: {_MONO}; font-size: .8rem; color: {INK}; overflow-wrap: anywhere; }}
.call .note {{ color: {BAR}; font-size: .74rem; margin-top: .15rem; }}
.call a {{ color: var(--bar); overflow-wrap: anywhere; }}
dl.kv {{ margin: .35rem 0 0; display: grid; grid-template-columns: max-content 1fr; gap: .12rem .7rem;
        font-size: .82rem; }}
dl.kv dt {{ color: {BAR}; font-family: {_MONO}; }}
dl.kv dd {{ margin: 0; overflow-wrap: anywhere; }}
pre {{ background: var(--surface); border: 1px solid {INK}22; border-radius: .25rem; margin: .35rem 0 0;
        overflow-x: auto; padding: .5rem .65rem; white-space: pre-wrap; word-break: break-word; font-family: {_MONO}; }}
.md {{ overflow-wrap: anywhere; }}
.md > :first-child {{ margin-top: .2rem; }}
.md > :last-child {{ margin-bottom: 0; }}
.md p {{ margin: .5rem 0; }}
.md h3, .md h4, .md h5, .md h6 {{ margin: .8rem 0 .3rem; font-size: 1rem; }}
.md ul, .md ol {{ margin: .4rem 0; padding-left: 1.4rem; }}
.md code {{ background: var(--surface); border: 1px solid {INK}22; border-radius: .2rem;
        padding: .05rem .3rem; font-size: .9em; }}
.md pre code {{ background: none; border: 0; padding: 0; }}
.md a {{ color: var(--bar); }}
.md table {{ border-collapse: collapse; margin: .6rem 0; font-size: .9em; }}
.md th, .md td {{ border: 1px solid {INK}22; padding: .25rem .55rem; text-align: left; }}
.md th {{ background: {INK}0a; }}
.exit {{ color: {BAR}; font-size: .75rem; }}
table {{ border-collapse: collapse; font-size: .8rem; margin-top: .5rem; }}
td {{ border-top: 1px solid {INK}22; padding: .2rem .75rem .2rem 0; }}
td.n {{ text-align: right; }}
{DIFF_CSS}
"""

#: How much of a tool call to show in its collapsed summary before the reader expands it.
_PREVIEW_CAP = 90


#: Lines of a written file or long command shown inline before the rest folds into a toggle.
_INLINE_LINES = 30


def _clamp(text: str) -> str:
    """A one-line summary preview: whitespace collapsed, clamped to :data:`_PREVIEW_CAP`."""
    preview = " ".join(text.split())
    return preview if len(preview) <= _PREVIEW_CAP else preview[:_PREVIEW_CAP] + "…"


def _tool_kind(call: ToolCall) -> str:
    """Classify a tool call into a render kind, working across harnesses.

    Keyed on the tool *name* first, then the argument-key signature, so Claude Code's native names
    (``Bash``, ``Edit``, ``Write``, …), Codex's names (``command_execution``, ``file_change``) and
    an unfamiliar future tool whose arguments match a known shape all land on the same rendering.
    Returns one of ``command``, ``read``, ``write``, ``edit``, ``file_change``, ``web_fetch``,
    ``web_search`` or ``generic``.
    """
    name = call.name.lower()
    keys = set(call.arguments)
    if name in {"bash", "command_execution", "shell", "bashoutput"} or "command" in keys:
        return "command"
    if name in {"edit", "multiedit", "notebookedit"} or {"old_string", "new_string"} <= keys:
        return "edit"
    if name == "write" or {"file_path", "content"} <= keys:
        return "write"
    if name == "read" or ("file_path" in keys and not keys & {"content", "old_string", "new_string"}):
        return "read"
    if name == "file_change" or keys & {"diff", "unified_diff", "patch", "changes"}:
        return "file_change"
    if name in {"webfetch", "web_fetch"} or {"url", "prompt"} <= keys:
        return "web_fetch"
    if name in {"websearch", "web_search", "toolsearch"} or ("query" in keys and keys <= {"query", "max_results"}):
        return "web_search"
    return "generic"


def _code_block(text: str, *, cls: str = "") -> str:
    """A ``<pre><code>`` block, clipped, folded into a nested toggle when it runs long."""
    clipped = _clip(text)
    attr = f' class="{cls}"' if cls else ""
    block = f"<pre{attr}><code>{escape(clipped)}</code></pre>"
    n_lines = clipped.count("\n") + 1
    if n_lines > _INLINE_LINES:
        return f"<details><summary>{n_lines} lines</summary>{block}</details>"
    return block


def _kv_list(args: dict[str, Any]) -> str:
    """A tool's raw arguments as a definition list — the readable fallback for unmodelled tools."""
    if not args:
        return ""
    rows: list[str] = []
    for key, value in args.items():
        if isinstance(value, str) and "\n" in value:
            rendered = _code_block(value)
        elif isinstance(value, dict | list):
            rendered = f"<pre><code>{escape(_clip(json.dumps(value, indent=2, default=str)))}</code></pre>"
        else:
            rendered = escape(str(value))
        rows.append(f"<dt>{escape(str(key))}</dt><dd>{rendered}</dd>")
    return f'<dl class="kv">{"".join(rows)}</dl>'


def _range_note(offset: object, limit: object) -> str:
    """A compact 'lines a–b' note for a windowed file read, tolerating missing bounds."""
    try:
        off = int(offset) if offset is not None else None
        lim = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        return ""
    if off is not None and lim is not None:
        return f"lines {off}–{off + lim}"
    if lim is not None:
        return f"first {lim} lines"
    if off is not None:
        return f"from line {off}"
    return ""


def _diff_tables(path: str, pairs: list[tuple[str, str]]) -> str:
    """One split diff per (old, new) pair — a plain Edit is one pair, a MultiEdit several."""
    return "".join(
        split_diff_table(path, _clip(old).splitlines(), _clip(new).splitlines(), ("before", "after"))
        for old, new in pairs
    )


def _edit_body(call: ToolCall) -> str:
    """A file edit as a red/green split diff — the same renderer the report uses for skill diffs."""
    args = call.arguments
    path = str(args.get("file_path") or args.get("notebook_path") or call.name)
    pairs: list[tuple[str, str]] = []
    edits = args.get("edits")
    if isinstance(edits, list):  # MultiEdit: a list of {old_string, new_string}
        pairs = [
            (str(e["old_string"]), str(e["new_string"]))
            for e in edits
            if isinstance(e, dict) and "old_string" in e and "new_string" in e
        ]
    elif "old_string" in args and "new_string" in args:
        pairs = [(str(args["old_string"]), str(args["new_string"]))]
    if not pairs:  # a shape we don't model (e.g. NotebookEdit cell ops) — show the raw arguments
        return f'<div class="path">{escape(path)}</div>{_kv_list(args)}'
    note = '<div class="note">replace all</div>' if args.get("replace_all") else ""
    return f'<div class="path">{escape(path)}</div>{_diff_tables(path, pairs)}{note}'


def _file_change_body(args: dict[str, Any]) -> str:
    """A Codex ``file_change`` item, best-effort: a unified diff if present, else per-file entries."""
    for key in ("diff", "unified_diff", "patch"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _code_block(value)
    changes = args.get("changes")
    parts: list[str] = []
    if isinstance(changes, list):
        for change in changes:
            if not isinstance(change, dict):
                continue
            path = str(change.get("path") or change.get("file") or "")
            if "old" in change and "new" in change:
                parts.append(_diff_tables(path, [(str(change["old"]), str(change["new"]))]))
            else:
                what = str(change.get("kind") or change.get("type") or "change")
                parts.append(f'<div class="path">{escape(what)}: {escape(path)}</div>')
    return "".join(parts) or _kv_list(args)


def _file_change_preview(args: dict[str, Any]) -> str:
    changes = args.get("changes")
    if isinstance(changes, list) and changes:
        paths = [str(c.get("path") or c.get("file") or "") for c in changes if isinstance(c, dict)]
        if any(paths):
            return ", ".join(p for p in paths if p)
    if any(isinstance(args.get(k), str) and args[k].strip() for k in ("diff", "unified_diff", "patch")):
        return "patch"
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _render_call(call: ToolCall, kind: str) -> tuple[str, str]:
    """The ``(summary preview, body HTML)`` for one tool call, dispatched on its kind."""
    args = call.arguments
    if kind == "command":
        command = str(args.get("command") or args.get("input") or "")
        return _clamp(command), _code_block(command, cls="sh")
    if kind == "read":
        path = str(args.get("file_path") or "")
        note = _range_note(args.get("offset"), args.get("limit"))
        note_html = f'<div class="note">{escape(note)}</div>' if note else ""
        return _clamp(path), f'<div class="path">{escape(path)}</div>{note_html}'
    if kind == "write":
        path = str(args.get("file_path") or "")
        content = str(args.get("content") or "")
        return _clamp(path), f'<div class="path">{escape(path)}</div>{_code_block(content)}'
    if kind == "edit":
        path = str(args.get("file_path") or args.get("notebook_path") or "")
        return _clamp(path or call.name), _edit_body(call)
    if kind == "file_change":
        return _clamp(_file_change_preview(args)), _file_change_body(args)
    if kind == "web_fetch":
        url = str(args.get("url") or "")
        prompt = str(args.get("prompt") or "")
        body = f'<div class="path"><a href="{escape(url)}">{escape(url)}</a></div>'
        if prompt:
            body += f'<div class="note">{escape(prompt)}</div>'
        return _clamp(url), body
    if kind == "web_search":
        query = str(args.get("query") or "")
        results = args.get("max_results")
        note = f'<div class="note">max results: {escape(str(results))}</div>' if results is not None else ""
        return _clamp(query), f'<div class="path">{escape(query)}</div>{note}'
    # generic: a single string argument reads as a body; anything else as a key/value list.
    if list(args) == ["input"] and isinstance(args["input"], str):
        return _clamp(args["input"]), _code_block(args["input"])
    return _clamp(", ".join(f"{key}={value!r}" for key, value in args.items())), _kv_list(args)


def _observation_html(obs: Observation) -> str:
    parts: list[str] = []
    if obs.content.strip():
        parts.append(f"<pre>{escape(_clip(obs.content))}</pre>")
    if obs.exit_code is not None:
        parts.append(f'<div class="exit">exit {escape(str(obs.exit_code))}</div>')
    return "".join(parts)


def _tool_details(call: ToolCall, observations: list[Observation]) -> str:
    """Render one tool call and its results in a collapsed ``<details>`` toggle.

    The call is classified by :func:`_tool_kind` and rendered per kind (a command as a shell
    block, a file edit as a red/green diff, and so on); the kind rides on an inner ``div`` so the
    outer ``<details class="tool">`` tag stays stable for callers keying on it.
    """
    errored = any(obs.is_error for obs in observations)
    kind = _tool_kind(call)
    preview, body = _render_call(call, kind)
    summary = f'<span class="name">{escape(call.name)}</span>'
    if preview:
        summary += f' <span class="preview">{escape(preview)}</span>'
    obs_html = "".join(_observation_html(obs) for obs in observations)
    inner = f'<div class="call call-{kind}">{body}{obs_html}</div>' if (body or obs_html) else ""
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
