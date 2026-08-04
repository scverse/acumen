"""Provider-neutral agent execution for Claude Code and Codex.

The rest of acumen deals in one small result shape. Claude is driven through its
Python SDK; Codex is driven through its documented non-interactive JSONL interface.
Keeping both adapters here prevents provider-specific message types and subprocess
details from leaking into the benchmark and meta-agent workflows.

Neither backend is required to use acumen. The Claude SDK is an optional dependency
(``pip install acumen[claude]``) and Codex is an external CLI, so a Claude-only and a
Codex-only install are both complete. Every import of a backend is therefore made inside
the function that needs it, and :func:`check_agent_cli` turns a missing one into an
actionable error before a command does any expensive work.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

AgentProvider = Literal["claude", "codex"]


class AgentError(RuntimeError):
    """Raised when an agent model or provider cannot be resolved."""


def provider_for_model(model: str) -> AgentProvider:
    """Infer the agent provider from a model ID.

    Anthropic model IDs begin with ``claude``. OpenAI's coding-capable IDs use
    ``gpt-*``, ``o*``, or ``codex-*``. Provider-qualified IDs are accepted too,
    which keeps proxy/router configurations readable.
    """
    value = model.strip().lower()
    tail = value.rsplit("/", 1)[-1]
    if tail.startswith("claude"):
        return "claude"
    if tail.startswith(("gpt-", "o1", "o3", "o4", "codex-")):
        return "codex"
    raise AgentError(
        f"cannot infer an agent for model {model!r}; use a Claude model ID beginning "
        "with 'claude' or an OpenAI model ID beginning with 'gpt-', 'o1', 'o3', "
        "'o4', or 'codex-'"
    )


def claude_sdk_available() -> bool:
    """Whether the optional Claude Agent SDK is installed."""
    return importlib.util.find_spec("claude_agent_sdk") is not None


def check_agent_cli(provider: AgentProvider) -> None:
    """Fail before target preparation when the selected backend is not installed.

    Each backend is optional and they are checked the same way: acumen refuses to prepare a
    target, build a sandbox, or spend a token for a provider it cannot actually drive.
    """
    if provider == "codex" and shutil.which("codex") is None:
        raise AgentError("codex is not on PATH — install Codex CLI before running an OpenAI model")
    if provider == "claude" and not claude_sdk_available():
        raise AgentError(
            "the Claude Agent SDK is not installed — run `pip install acumen[claude]` before "
            "running a Claude model, or select a Codex model"
        )


@dataclass(frozen=True)
class AgentResult:
    """The provider-neutral terminal result of one autonomous agent run."""

    provider: AgentProvider
    is_error: bool
    subtype: str
    errors: list[str] | None
    session_id: str | None
    result: str
    num_turns: int
    total_cost_usd: float | None
    duration_ms: int
    usage: dict[str, int]
    stop_reason: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentOptions:
    """Provider-neutral options shared by benchmark and meta-agent runs."""

    cwd: Path
    env: dict[str, str]
    model: str
    max_turns: int | None = None
    max_usd: float | None = None
    add_dirs: tuple[Path, ...] = ()
    discover_skills: bool = True
    claude_hooks: dict[str, Any] | None = None
    deny_paths: tuple[Path, ...] = ()
    stderr: Callable[[str], None] | None = None
    #: Prices one provider usage block in USD. Claude enforces ``max_usd`` itself against the
    #: figure it bills; Codex reports tokens and no dollar amount, so enforcing a budget there
    #: means pricing its usage ourselves — pass the caller's rate table in as this hook.
    #: Without it ``max_usd`` cannot be enforced for a Codex run.
    price_usd: Callable[[dict[str, Any]], float | None] | None = None


def _claude_options(options: AgentOptions) -> ClaudeAgentOptions:
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        cwd=str(options.cwd),
        env=options.env,
        model=options.model,
        max_turns=options.max_turns,
        max_budget_usd=options.max_usd,
        add_dirs=[str(path) for path in options.add_dirs],
        setting_sources=["project"] if options.discover_skills else [],
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code"},
        hooks=options.claude_hooks,
        stderr=options.stderr,
    )


def _from_claude(result: ResultMessage) -> AgentResult:
    return AgentResult(
        provider="claude",
        is_error=bool(result.is_error),
        subtype=result.subtype or "",
        errors=result.errors,
        session_id=result.session_id,
        result=result.result or "",
        num_turns=result.num_turns,
        total_cost_usd=result.total_cost_usd,
        duration_ms=result.duration_ms or 0,
        usage=dict(result.usage or {}),
        stop_reason=result.stop_reason,
    )


async def _run_claude(
    prompt: str,
    options: AgentOptions,
    on_event: Callable[[Any], None] | None,
) -> AgentResult:
    from claude_agent_sdk import ResultMessage, query

    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=_claude_options(options)):
        if on_event is not None:
            on_event(message)
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise AgentError("the Claude agent produced no result message")
    return _from_claude(result)


def _codex_command(options: AgentOptions, prompt: str) -> list[str]:
    cli = shutil.which("codex", path=options.env.get("PATH"))
    if cli is None:
        raise AgentError("codex is not on PATH — install Codex CLI before running an OpenAI model")
    command = [
        cli,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--sandbox",
        "workspace-write",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "project_doc_fallback_filenames=[]",
        "-c",
        "sandbox_workspace_write.network_access=true",
        "--cd",
        str(options.cwd),
        "--model",
        options.model,
    ]
    if options.deny_paths:
        command.append("--dangerously-bypass-hook-trust")
    for path in options.add_dirs:
        command.extend(("--add-dir", str(path)))
    command.append(prompt)
    return command


def _codex_guard_source(denied: Sequence[Path]) -> str:
    """Return a standalone Codex PreToolUse guard for absolute denied roots."""
    roots = repr([str(path.resolve()) for path in denied])
    return f"""\
import json
import shlex
import sys
from pathlib import Path

ROOTS = [Path(value) for value in {roots}]


def blocked(value, cwd):
    if not isinstance(value, str) or not value.strip():
        return None
    candidates = [value]
    expanded = value.translate(str.maketrans({{char: " " for char in "|&;()<>`\\n\\r\\t"}}))
    try:
        candidates.extend(shlex.split(expanded))
    except ValueError:
        candidates.extend(expanded.split())
    for raw in candidates:
        raw = raw.strip(" ,;()[]{{}}'\\\"")
        if not raw:
            continue
        try:
            path = Path(raw).expanduser()
            resolved = (path if path.is_absolute() else cwd / path).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        for root in ROOTS:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            return raw
    return None


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


payload = json.load(sys.stdin)
cwd = Path(payload.get("cwd") or ".").resolve()
for value in strings(payload.get("tool_input") or {{}}):
    hit = blocked(value, cwd)
    if hit is not None:
        print(json.dumps({{
            "hookSpecificOutput": {{
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "acumen blocks access to isolated benchmark data: " + hit,
            }}
        }}))
        raise SystemExit(0)
"""


def _install_codex_guard(options: AgentOptions) -> None:
    """Install a trusted, run-local Codex guard when isolation needs a deny boundary."""
    if not options.deny_paths:
        return
    codex_home = Path(options.env["CODEX_HOME"])
    script = codex_home / "acumen_guard.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_codex_guard_source(options.deny_paths))
    hooks_dir = options.cwd / ".codex"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hooks = {
        "description": "acumen run isolation",
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
                            "timeout": 10,
                        }
                    ],
                }
            ]
        },
    }
    (hooks_dir / "hooks.json").write_text(json.dumps(hooks, indent=2) + "\n")


#: Item types that count as one model action. ``codex exec`` runs the whole prompt as a single
#: turn — one ``turn.started``/``turn.completed`` pair however much work happens inside it — so
#: counting turns would record 1 for every run. A completed item is the closest analogue to a
#: Claude turn, and it is the unit :attr:`AgentOptions.max_turns` is enforced in. ``reasoning``
#: and ``todo_list`` items are the model's own narration and bookkeeping, not actions, so they
#: do not count.
_TURN_ITEM_TYPES = frozenset({"agent_message", "command_execution", "file_change", "mcp_tool_call", "web_search"})

#: Terminal subtypes for a cap acumen enforced itself. They deliberately reuse the strings the
#: Claude CLI reports for its own caps, so a breach maps onto the same reason for both providers.
_MAX_TURNS_SUBTYPE = "error_max_turns"
_MAX_BUDGET_SUBTYPE = "error_max_budget_usd"


def _is_turn_item(event: dict[str, Any]) -> bool:
    """Whether ``event`` is one completed model action — the unit Codex turns are counted in."""
    if event.get("type") != "item.completed":
        return False
    item = event.get("item")
    return isinstance(item, dict) and item.get("type") in _TURN_ITEM_TYPES


def _codex_terminal(
    events: Sequence[dict[str, Any]],
    returncode: int,
    duration_ms: int,
    capped: str | None = None,
) -> AgentResult:
    session_id: str | None = None
    final = ""
    usage: dict[str, int] = {}
    errors: list[str] = []
    subtype = "success"
    turns = 0
    for event in events:
        kind = event.get("type")
        if _is_turn_item(event):
            turns += 1
        if kind == "thread.started":
            value = event.get("thread_id")
            session_id = str(value) if value else None
        elif kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final = text
        elif kind == "turn.completed":
            raw = event.get("usage")
            if isinstance(raw, dict):
                usage = {str(key): int(value or 0) for key, value in raw.items() if isinstance(value, int | float)}
        elif kind in {"error", "turn.failed"}:
            subtype = str(event.get("type"))
            detail = event.get("message") or event.get("error")
            errors.append(str(detail or event))

    if capped is not None:
        # We terminated the process on purpose, so its non-zero status is ours and must not be
        # reported as a codex failure. The partial transcript is kept: it is the evidence of
        # what the run spent the cap on.
        is_error = True
        subtype = capped
        errors.append(
            "acumen stopped the run at its turn cap"
            if capped == _MAX_TURNS_SUBTYPE
            else "acumen stopped the run at its budget cap"
        )
    else:
        is_error = returncode != 0 or bool(errors)
        if returncode != 0 and not errors:
            errors.append(f"codex exited with status {returncode}")
    return AgentResult(
        provider="codex",
        is_error=is_error,
        subtype=subtype if is_error else "success",
        errors=errors or None,
        session_id=session_id,
        result=final,
        num_turns=turns,
        # Codex's JSONL protocol reports tokens, not a billed dollar amount.
        total_cost_usd=None,
        duration_ms=duration_ms,
        usage=usage,
        transcript=list(events),
    )


async def _drain_stderr(
    stream: asyncio.StreamReader,
    callback: Callable[[str], None] | None,
) -> None:
    while line := await stream.readline():
        text = line.decode(errors="replace").rstrip("\r\n")
        if callback is not None:
            callback(text)
        else:
            print(text, file=sys.stderr, flush=True)


def _cap_breach(event: dict[str, Any], options: AgentOptions, turns: int) -> str | None:
    """Return the cap ``event`` breached, or ``None``.

    ``codex exec`` has no turn or budget cap of its own, so acumen enforces both against the
    event stream and terminates the process on a breach. The two caps are enforced at different
    resolutions, and the difference matters:

    * ``max_turns`` is checked as each model action completes, so the run really is stopped at
      the cap — the agent gets ``max_turns`` actions and no more. The cost of stopping mid-turn
      is that ``turn.completed`` never arrives, so a turn-capped run reports no usage and is
      priced at zero; it is a failed run either way, but its tokens are not recoverable.
    * ``max_usd`` can only be checked when usage is reported, and Codex reports it once, in
      ``turn.completed``, at the end of the turn. A single-turn ``codex exec`` therefore has
      already spent the money by the time the breach is visible: the cap makes the *outcome*
      honest — the run is recorded as a budget failure, exactly as it would be for Claude —
      but it cannot prevent the overspend. Bound Codex spend with ``max_turns`` instead.
    """
    if options.max_turns is not None and _is_turn_item(event) and turns >= options.max_turns:
        return _MAX_TURNS_SUBTYPE
    if options.max_usd is None or options.price_usd is None or event.get("type") != "turn.completed":
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    spent = options.price_usd(usage)
    return _MAX_BUDGET_SUBTYPE if spent is not None and spent > options.max_usd else None


async def _run_codex(
    prompt: str,
    options: AgentOptions,
    on_event: Callable[[Any], None] | None,
) -> AgentResult:
    _install_codex_guard(options)
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *_codex_command(options, prompt),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=options.env,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(_drain_stderr(process.stderr, options.stderr))
    events: list[dict[str, Any]] = []
    turns = 0
    capped: str | None = None
    try:
        while line := await process.stdout.readline():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "error", "message": line.decode(errors="replace").strip()}
            if not isinstance(event, dict):
                continue
            events.append(event)
            if on_event is not None:
                on_event(event)
            if _is_turn_item(event):
                turns += 1
            capped = _cap_breach(event, options, turns)
            if capped is not None:
                process.terminate()
                break
        returncode = await process.wait()
        await stderr_task
    except BaseException:
        process.terminate()
        await process.wait()
        stderr_task.cancel()
        raise
    duration_ms = round((time.monotonic() - started) * 1000)
    return _codex_terminal(events, returncode, duration_ms, capped)


async def run_agent(
    prompt: str,
    *,
    options: AgentOptions,
    on_event: Callable[[Any], None] | None = None,
) -> AgentResult:
    """Run one Claude or Codex agent, selected from ``options.model``."""
    provider = provider_for_model(options.model)
    if provider == "claude":
        return await _run_claude(prompt, options, on_event)
    return await _run_codex(prompt, options, on_event)
