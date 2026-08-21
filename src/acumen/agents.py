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
import signal
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
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
    read_dirs: tuple[Path, ...] = ()
    write_dirs: tuple[Path, ...] = ()
    discover_skills: bool = True
    claude_hooks: dict[str, Any] | None = None
    deny_paths: tuple[Path, ...] = ()
    #: Whether to install the :mod:`acumen.guard` containment hook, which refuses a Claude tool
    #: call naming a path outside ``read_dirs``/``write_dirs``/``cwd``. True for every isolated
    #: agent. The shipper sets it False: it is the one agent that runs in the operator's real
    #: environment on purpose, and it needs the git and ``gh`` credentials that live there.
    confine: bool = True
    stderr: Callable[[str], None] | None = None
    #: Prices one provider usage block in USD. Claude enforces ``max_usd`` itself against the
    #: figure it bills; Codex reports tokens and no dollar amount, so enforcing a budget there
    #: means pricing its usage ourselves — pass the caller's rate table in as this hook.
    #: Without it ``max_usd`` cannot be enforced for a Codex run.
    price_usd: Callable[[dict[str, Any]], float | None] | None = None


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in result:
            result.append(resolved)
    return result


def _runtime_read_dirs() -> list[Path]:
    """Host paths required to start ordinary Python and shell subprocesses."""
    candidates = [
        Path(sys.base_prefix),
        Path(sys.prefix),
        Path(sys.executable).resolve().parent,
        Path("/bin"),
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ssl"),
        Path("/etc/hosts"),
        Path("/etc/resolv.conf"),
        Path("/System"),
        Path("/Library"),
    ]
    return _dedupe_paths(candidates)


def _access_roots(options: AgentOptions) -> tuple[list[Path], list[Path]]:
    env_paths = [
        Path(value)
        for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "CLAUDE_CONFIG_DIR", "CODEX_HOME")
        if (value := options.env.get(key))
    ]
    writes = _dedupe_paths([options.cwd, *options.write_dirs, *env_paths])
    reads = _dedupe_paths([*_runtime_read_dirs(), *options.read_dirs, *writes])
    return reads, writes


def _codex_runtime_roots(cli: Path) -> list[Path]:
    """Return the directories the Codex CLI must read to re-execute itself.

    Codex's command sandbox re-runs the Codex binary inside a bubblewrap namespace, so
    whatever ``codex`` resolves to on ``PATH`` has to stay readable. For a standalone
    install (Homebrew, cargo, a downloaded release) that is a real binary and its own
    directory is enough.

    The npm package is not: ``codex`` there is a Node shim at ``<pkg>/bin/codex.js`` that
    execs a native binary shipped in a *separate* optional dependency,
    ``@openai/codex-<platform>-<arch>/vendor/<triple>/bin/codex``. That package lives
    outside ``<pkg>/bin``, so allowing only the shim's directory leaves the real binary
    outside the namespace and every command dies with ``bwrap: execvp …: No such file or
    directory`` — the agent keeps its tools but can no longer run a single one.

    The shim finds the platform package by ordinary Node resolution, walking up from its
    own directory through each ancestor's ``node_modules``. Mirroring that walk locates
    the package wherever the install layout put it (npm, yarn, or pnpm's symlinked store)
    without hard-coding a target triple: npm installs only the optional dependency that
    matches the host, so a glob finds the one that is actually present.
    """
    roots = [cli.parent]
    for ancestor in cli.parents:
        scope = ancestor / "node_modules" / "@openai"
        if not scope.is_dir():
            continue
        # ``codex-*`` cannot match the shim package itself, only its platform siblings.
        roots.extend(package for package in scope.glob("codex-*") if (package / "vendor").is_dir())
    # The shim falls back to a vendor directory inside its own package when the optional
    # dependency was flattened away by the installer.
    bundled = cli.parent.parent / "vendor"
    if bundled.is_dir():
        roots.append(bundled)
    # pnpm reaches the platform package through a symlink into its content-addressed
    # store. Landlock and bubblewrap both evaluate the real path, so allow both.
    return _dedupe_paths([*roots, *(path.resolve() for path in roots)])


def _claude_path_rule(tool: str, path: Path) -> str:
    return f"{tool}(//{str(path).lstrip('/')}/**)"


def _claude_settings(options: AgentOptions) -> dict[str, Any]:
    """Build run-local Claude settings for unattended execution.

    Deliberately carries no ``sandbox`` block. Claude Code's OS sandbox confines the
    filesystem but also puts a proxy in front of every outbound connection, and that proxy's
    allowlist cannot be set from the ``--settings`` file the SDK passes: it ignores
    ``sandbox.network.allowedDomains`` whether the entry is ``"*"`` or an explicit host, and
    ignores ``WebFetch(domain:...)`` allow rules too. Measured on CLI 2.1.224: with the
    sandbox on, zenodo.org, ftp.ebi.ac.uk and omnipathdb.org are all unreachable; with it off,
    all three answer. A benchmark target downloads its own datasets and priors, so egress has
    to win, and the filesystem boundary moves to :mod:`acumen.guard`.

    The file is still written even now that it carries only permissions, because passing
    ``--settings`` is also what stops the operator's own ``~/.claude/settings.json`` from
    being inherited by a run.
    """
    reads, writes = _access_roots(options)
    return {
        "permissions": {
            "allow": [
                "Bash",
                "WebFetch",
                "WebSearch",
                "Skill",
                *(_claude_path_rule("Read", path) for path in reads),
                *(_claude_path_rule("Edit", path) for path in writes),
            ],
        },
    }


def _claude_hooks(options: AgentOptions) -> dict[str, Any]:
    """Combine the caller's own hooks with the filesystem containment guard.

    Composed rather than replaced: ``improve`` and ``taskgen`` each install a ``PreToolUse``
    hook of their own to hide evidence a meta-agent must not see, and those still have to fire
    alongside containment.
    """
    hooks: dict[str, Any] = {key: list(value) for key, value in (options.claude_hooks or {}).items()}
    if not options.confine:
        return hooks

    from acumen.guard import containment_hook

    reads, _ = _access_roots(options)
    agent_home = options.env.get("HOME")
    guard = containment_hook(
        reads,
        options.deny_paths,
        cwd=options.cwd,
        home=Path(agent_home) if agent_home else None,
    )
    hooks.setdefault("PreToolUse", [])
    hooks["PreToolUse"] = [guard, *hooks["PreToolUse"]]
    return hooks


def _claude_options(options: AgentOptions) -> ClaudeAgentOptions:
    from claude_agent_sdk import ClaudeAgentOptions

    config_dir_value = options.env.get("CLAUDE_CONFIG_DIR")
    if not config_dir_value:
        raise AgentError("an isolated Claude run requires a run-local CLAUDE_CONFIG_DIR")
    settings_path = Path(config_dir_value) / "acumen-settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(_claude_settings(options)), encoding="utf-8")

    return ClaudeAgentOptions(
        cwd=str(options.cwd),
        env=options.env,
        model=options.model,
        max_turns=options.max_turns,
        max_budget_usd=options.max_usd,
        add_dirs=[],
        setting_sources=["project"] if options.discover_skills else [],
        permission_mode="dontAsk",
        # The SDK's ``settings`` field is a path passed to Claude Code's
        # ``--settings`` flag, not an in-memory settings object.  Keeping this
        # file in the isolated config directory also prevents user settings
        # from being inherited by the run.
        settings=str(settings_path),
        system_prompt={"type": "preset", "preset": "claude_code"},
        hooks=_claude_hooks(options),
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


#: Subtype of the CLI system message that reports the run's live background tasks as a whole
#: set rather than as start/stop edges. Its payload is authoritative and replaces whatever we
#: were tracking, so a bookend we never saw cannot leave a phantom task behind. The Python SDK
#: has no typed class for it, so it arrives as a plain ``SystemMessage``.
_LIVE_TASKS_SUBTYPE = "background_tasks_changed"

#: How long teardown waits for the CLI to confirm the background tasks it was told to stop are
#: really gone. Bounded because it sits between the agent finishing and the run being graded: a
#: task that ignores the stop must not hold up the pass, and :func:`acumen.procs.reap` kills
#: whatever is left when the sandbox goes. Long enough for the CLI to flush each task's output
#: file first, which is what stops a run losing an ``answer.md`` written moments before the end.
_TEARDOWN_GRACE_S = 10.0


def _note_tasks(message: Any, live: dict[str, None], sdk: ModuleType) -> None:
    """Fold one message into ``live``, the ids of the run's still-running background tasks.

    Kept as an insertion-ordered dict so the ids are stopped in the order they started, which
    makes a teardown log read in the order the agent created the work.

    Three signals, deliberately all three. ``background_tasks_changed`` carries the whole live
    set and so is authoritative — the CLI documents replace semantics for it precisely so a
    consumer that missed an edge still converges. The typed edge messages are the fallback for a
    CLI build that does not emit the level signal: ``task_started``/``task_progress`` add an id,
    and a terminal status retires it. A terminal status can arrive on *either* of
    ``task_notification`` and ``task_updated`` — a task stopped via ``TaskStop`` reports
    ``killed`` only on the latter — so both are read the same way.
    """
    if isinstance(message, sdk.TaskStartedMessage | sdk.TaskProgressMessage):
        live[message.task_id] = None
    elif isinstance(message, sdk.TaskNotificationMessage | sdk.TaskUpdatedMessage):
        if (message.status or "") in sdk.TERMINAL_TASK_STATUSES:
            live.pop(message.task_id, None)
    elif isinstance(message, sdk.SystemMessage) and message.subtype == _LIVE_TASKS_SUBTYPE:
        tasks = message.data.get("tasks")
        if isinstance(tasks, list):
            live.clear()
            live.update(
                dict.fromkeys(str(task["task_id"]) for task in tasks if isinstance(task, dict) and task.get("task_id"))
            )


async def _quiesce_claude(
    client: Any,
    live: dict[str, None],
    on_event: Callable[[Any], None] | None,
    sdk: ModuleType,
) -> None:
    """End the run's session for good, before anything it wrote is graded.

    The terminal result is not the end of a Claude session. A Bash command the agent left
    running keeps the CLI alive, and when it finishes the CLI queues the ``task-notification``
    as a *new* prompt and re-enters the model. That re-entry produces its own result message,
    and it happens after the turn cap, so the CLI answers every tool call the agent then makes
    with a cancelled-permission denial — leaving the agent politely waiting for an operator who
    does not exist while the harness has already moved on to grading. Measured on one such run:
    18 minutes and 100 model calls past the answer, recorded as a 4-second, two-turn success.

    So the background work is stopped first, which is what stops another notification being
    queued at all; then the CLI is given a bounded moment to confirm each task is really gone,
    so its output file is flushed before the run's artifacts are collected; then the turn is
    interrupted, in case a notification queued before the first step already started one.

    Every step is best-effort, and called from a ``finally`` so it covers a crashed or capped
    run as well as a clean one: those can have left background work running too. Whatever the
    run produced has already been read by the time this is reached, and no failure to tidy up
    afterwards may be allowed to change or discard it.
    """
    for task_id in list(live):
        with suppress(Exception):
            await client.stop_task(task_id)
    if live:
        with suppress(Exception):
            async with asyncio.timeout(_TEARDOWN_GRACE_S), aclosing(client.receive_messages()) as stream:
                async for message in stream:
                    if on_event is not None:
                        on_event(message)
                    _note_tasks(message, live, sdk)
                    if not live:
                        break
    with suppress(Exception):
        await client.interrupt()


async def _run_claude(
    prompt: str,
    options: AgentOptions,
    on_event: Callable[[Any], None] | None,
) -> AgentResult:
    """Run one Claude agent to its terminal result, then shut its session down.

    Driven through ``ClaudeSDKClient`` rather than the SDK's one-shot ``query`` helper. The
    session has to be *steered* at the end of a run — the background tasks stopped, the turn
    interrupted, the transport closed — and ``query`` is documented as offering none of that.
    See :func:`_quiesce_claude` for what that teardown is for.

    ``receive_response`` stops at the first result message, which is also what freezes the
    figures the run is judged on. A re-entered session reports turns, duration and usage for
    the re-entry alone, so a loop that kept assigning the latest result recorded a 1117-second,
    41-turn run that hit its cap as a 4-second, 2-turn success, and priced it 96% low. Reading
    exactly one result makes that unrepresentable rather than guarded against.
    """
    import claude_agent_sdk as sdk

    result: ResultMessage | None = None
    noise: list[str] = []
    live: dict[str, None] = {}

    def capture_stderr(line: str) -> None:
        noise.append(line)
        if options.stderr is not None:
            options.stderr(line)
        else:
            print(line, file=sys.stderr, flush=True)

    client = sdk.ClaudeSDKClient(options=_claude_options(replace(options, stderr=capture_stderr)))
    try:
        await client.connect()
        await client.query(prompt)
        async for message in client.receive_response():
            if on_event is not None:
                on_event(message)
            _note_tasks(message, live, sdk)
            if isinstance(message, sdk.ResultMessage):
                result = message
    except Exception as err:
        # The CLI exits non-zero on purpose after reporting an ``is_error`` result — a turn
        # or budget cap, say — and the SDK surfaces that trailing ProcessError as an
        # exception carrying the result text. The structured ResultMessage has already
        # arrived by then, so raising would throw away the turns, cost, session id and
        # transcript of a run that really happened, and file a cap breach as an
        # unexplained crash. Return the result the CLI gave us and let the caller classify
        # it. A result that is *not* an error means the stream broke after a clean finish,
        # which is a genuine failure and still raises.
        if result is not None and result.is_error:
            return _from_claude(result)
        # SDK connection/process exceptions can be generic while the actionable quota or
        # billing message was emitted only on stderr. Preserve it so the benchmark runner can
        # classify infrastructure exhaustion instead of blaming the agent.
        detail = "\n".join(line.strip() for line in noise if line.strip())
        if detail:
            raise AgentError(f"{err}\nClaude stderr:\n{detail}") from err
        raise
    finally:
        # Both steps run on every path, a cap breach and a crash included: a run that died
        # mid-turn can have left background work running just as a clean one can. Nested so the
        # disconnect cannot be skipped — a client connected with no input stream keeps the CLI
        # subprocess alive until it is disconnected, so losing this leaks a live agent per run,
        # and teardown under cancellation raises straight past a plain ``suppress(Exception)``.
        try:
            await _quiesce_claude(client, live, on_event, sdk)
        finally:
            with suppress(Exception):
                await client.disconnect()
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
        # Deliberately not --ephemeral. The JSONL stream reports usage exactly once, in
        # ``turn.completed``, so a run acumen stops at a cap — or one whose CLI dies mid-turn —
        # would report no tokens at all and be priced at zero. Codex's rollout session file
        # records the running total as the turn proceeds, and it is the only source for it.
        # It is written inside the run-local CODEX_HOME and discarded with the sandbox.
        # See :class:`_CodexUsageTail`.
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "project_doc_fallback_filenames=[]",
        # Built-in web/browser/app surfaces do not run under local permission
        # profiles. Disable them so every network request made by an isolated
        # run goes through the profile-controlled command proxy.
        "-c",
        'web_search="disabled"',
        "-c",
        "tools.web_search=false",
        "-c",
        "features.browser_use=false",
        "-c",
        "features.browser_use_external=false",
        "-c",
        "features.browser_use_full_cdp_access=false",
        "-c",
        "features.apps=false",
        "-c",
        "features.remote_plugin=false",
        "-c",
        "features.network_proxy=true",
        "-c",
        'approval_policy="never"',
        "--cd",
        str(options.cwd),
        "--model",
        options.model,
    ]
    if options.deny_paths:
        command.append("--dangerously-bypass-hook-trust")
    reads, writes = _access_roots(options)
    # The Linux command sandbox re-executes the Codex binary through bubblewrap.
    # Its resolved standalone install is outside the usual /usr runtime roots,
    # so make the containing directories visible without exposing user config.
    reads = [*reads, *_codex_runtime_roots(Path(cli).resolve())]
    # Keep lexical system aliases as well as their resolved targets.  Landlock
    # evaluates the command path it is given, so allowing /usr/bin does not make
    # an invocation through the /bin -> /usr/bin symlink readable.
    reads.extend(path for path in (Path("/bin"), Path("/lib"), Path("/lib64")) if path.exists())
    reads = list(dict.fromkeys(reads))
    command.extend(("-c", 'default_permissions="acumen"'))
    filesystem = {":root": "deny", ":minimal": "read"}
    for path in reads:
        filesystem[str(path)] = "read"
    for path in writes:
        filesystem[str(path)] = "write"
    # One inline TOML table keeps paths opaque. Passing each path as a dotted
    # ``-c`` key misparses versioned and hidden directories (for example
    # ``0.146.0/...`` or ``.venv``) as nested configuration keys.
    filesystem_toml = ",".join(f"{json.dumps(path)}={json.dumps(access)}" for path, access in filesystem.items())
    command.extend(("-c", f"permissions.acumen.filesystem={{{filesystem_toml}}}"))
    command.extend(("-c", "permissions.acumen.network.enabled=true"))
    # ``full``, not ``limited``: the two accepted modes differ in more than their domain
    # table. Under ``limited`` the proxy enforces a method policy as well, answering anything
    # outside GET/HEAD/OPTIONS with 403 (verified against ``codex sandbox`` with
    # ``features.network_proxy=true``), which silently breaks every POST a target's loaders
    # make. Egress here is meant to be unrestricted, so the mode has to say so.
    command.extend(("-c", 'permissions.acumen.network.mode="full"'))
    command.extend(("-c", 'permissions.acumen.network.domains={"*"="allow"}'))
    command.extend(("-c", "permissions.acumen.network.allow_local_binding=false"))
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

#: Subtype for a run whose *sandbox* failed, as distinct from a run that failed the task.
_SANDBOX_SUBTYPE = "error_sandbox"

#: Stderr fragments that mean Codex's own sandbox could not carry out the agent's file writes.
#: On a host where unprivileged user namespaces are restricted — Ubuntu 24.04's AppArmor
#: default, a container without the capability — bubblewrap cannot start, and every write the
#: agent attempts fails. The agent still does the whole task and still spends the tokens; it
#: just cannot save the answer, which would otherwise be graded ``no_answer_file`` and read as
#: the model failing. It is the harness that failed, so the run is recorded as an error and
#: leaves the comparison rather than being counted against the model.
_SANDBOX_FAILURES = (
    "fs sandbox helper failed",
    "bwrap:",
    "failed to write file",
    "sandbox helper",
    "sandbox setup failed",
    "failed to apply sandbox",
    "landlock",
    "unprivileged user namespace",
)


def _sandbox_failure(lines: Sequence[str]) -> str | None:
    """Return the first stderr line showing a sandbox failure, or ``None``."""
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in _SANDBOX_FAILURES):
            return line.strip()
    return None


def _is_turn_item(event: dict[str, Any]) -> bool:
    """Whether ``event`` is one completed model action — the unit Codex turns are counted in."""
    if event.get("type") != "item.completed":
        return False
    item = event.get("item")
    return isinstance(item, dict) and item.get("type") in _TURN_ITEM_TYPES


#: Token counters acumen reads out of a rollout ``token_count`` event. They are the same names
#: ``turn.completed`` uses, which is what lets :func:`acumen.prices.normalize_usage` price either
#: source without knowing which one it got. ``total_tokens`` is dropped: it is the sum of the
#: others and ``turn.completed`` does not report it.
_ROLLOUT_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _rollout_usage(line: bytes) -> dict[str, int] | None:
    """Return the running token total carried by a rollout ``token_count`` line, or ``None``."""
    if b"token_count" not in line:
        return None
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict) or payload.get("type") != "token_count":
        return None
    info = payload.get("info")
    # ``info`` is null on the token_count events that only refresh rate limits.
    total = info.get("total_token_usage") if isinstance(info, dict) else None
    if not isinstance(total, dict):
        return None
    return {key: int(total[key] or 0) for key in _ROLLOUT_USAGE_KEYS if isinstance(total.get(key), int | float)}


class _CodexUsageTail:
    """Follow the running token total Codex writes to its rollout session file.

    ``codex exec --json`` reports usage once, in ``turn.completed``. A run acumen terminates at
    a cap never reaches that event, and neither does a run whose CLI dies mid-turn, so both used
    to record zero tokens and price at zero. Codex separately appends a ``token_count`` event to
    ``$CODEX_HOME/sessions/**/rollout-*.jsonl`` after every model response, carrying the turn's
    cumulative usage, and this reads that file as it grows.

    Best-effort by construction. The rollout layout is Codex's own and is not a documented
    interface, so every failure to find, open, or parse it is a no-op that leaves the last known
    total in place. ``turn.completed`` remains authoritative whenever it does arrive; this only
    fills the gap when it does not.

    The recovered figure is a lower bound on a stopped run. Codex writes each record slightly
    behind the stdout event it belongs to, so the very last response can be cut off before it is
    recorded even with the interrupt :func:`_stop_codex` sends. Under-reporting the tail of a
    capped run is the honest failure direction; reporting nothing was not.
    """

    def __init__(self, codex_home: str | None) -> None:
        self._home = Path(codex_home) if codex_home else None
        self._thread_id: str | None = None
        self._path: Path | None = None
        self._offset = 0
        self.usage: dict[str, int] = {}

    def attach(self, thread_id: str) -> None:
        """Note the thread whose rollout to follow, as reported by ``thread.started``."""
        self._thread_id = thread_id or None

    def _locate(self) -> Path | None:
        """Resolve the rollout path, retrying until Codex has created the file."""
        if self._path is not None:
            return self._path
        if self._home is None or self._thread_id is None:
            return None
        sessions = self._home / "sessions"
        try:
            matches = sorted(sessions.glob(f"**/rollout-*-{self._thread_id}.jsonl"))
            if not matches:
                # A CODEX_HOME shared across runs can hold older sessions, so the thread-id
                # match is what we want and the newest file is only a fallback for a naming
                # scheme we did not anticipate. Rollout names lead with an ISO timestamp, so
                # sorting by name sorts by age.
                matches = sorted(sessions.glob("**/rollout-*.jsonl"))[-1:]
        except OSError:
            return None
        self._path = matches[-1] if matches else None
        return self._path

    def poll(self) -> None:
        """Consume whatever Codex has appended since the last poll."""
        path = self._locate()
        if path is None:
            return
        try:
            with path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return
        # Stop at the last newline: the tail of the file can be a half-written record, and
        # parsing it would drop the event it belongs to.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self._offset += cut + 1
        for line in chunk[: cut + 1].splitlines():
            usage = _rollout_usage(line)
            if usage is not None:
                # The event carries the turn's running total, so it replaces rather than adds.
                self.usage = usage


def _event_usage(event: dict[str, Any]) -> dict[str, int]:
    """Return the usage block a ``turn.completed`` event reports, or ``{}``."""
    if event.get("type") != "turn.completed":
        return {}
    raw = event.get("usage")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value or 0) for key, value in raw.items() if isinstance(value, int | float)}


def _codex_terminal(
    events: Sequence[dict[str, Any]],
    returncode: int,
    duration_ms: int,
    capped: str | None = None,
    sandbox_failure: str | None = None,
    stderr_lines: Sequence[str] = (),
    streamed_usage: dict[str, int] | None = None,
) -> AgentResult:
    """Fold a Codex event stream into the provider-neutral result.

    ``streamed_usage`` is the running total read off the rollout file by
    :class:`_CodexUsageTail`. It is used only when ``turn.completed`` never arrived, which is
    exactly the case a cap breach or a mid-turn crash leaves behind.
    """
    session_id: str | None = None
    final = ""
    usage: dict[str, int] = {}
    errors: list[str] = []
    subtype = "success"
    turns = 0
    command_output: list[str] = []
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
            elif isinstance(item, dict) and item.get("type") == "command_execution":
                output = item.get("aggregated_output")
                if isinstance(output, str):
                    command_output.extend(output.splitlines())
        elif kind == "turn.completed":
            usage = _event_usage(event) or usage
        elif kind in {"error", "turn.failed"}:
            subtype = str(event.get("type"))
            detail = event.get("message") or event.get("error")
            errors.append(str(detail or event))

    # A sandbox that cannot start reports itself where the *command* was supposed to write,
    # not on codex's own stderr: bubblewrap's ``execvp`` failure arrives as the command's
    # ``aggregated_output``. Scanning stderr alone therefore misses the case where every
    # command in the run died before it began, which grades as the model answering badly
    # rather than as the harness failing.
    if sandbox_failure is None:
        sandbox_failure = _sandbox_failure(command_output)
    if not usage and streamed_usage:
        usage = dict(streamed_usage)

    if sandbox_failure is not None:
        # Checked before the cap and before the exit status: when the sandbox cannot carry out
        # the agent's writes, nothing downstream of it means anything. The run did the work and
        # spent the tokens, so the usage is kept, but the outcome is the harness's failure and
        # is recorded as such — never as the model declining to answer.
        is_error = True
        subtype = _SANDBOX_SUBTYPE
        errors.append(f"the Codex sandbox could not complete the run's file operations: {sandbox_failure}")
    elif capped is not None:
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
        if returncode != 0:
            errors.extend(line.strip() for line in stderr_lines if line.strip() and line.strip() not in errors)
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
    sink: list[str] | None = None,
) -> None:
    while line := await stream.readline():
        text = line.decode(errors="replace").rstrip("\r\n")
        if sink is not None:
            sink.append(text)
        if callback is not None:
            callback(text)
        else:
            print(text, file=sys.stderr, flush=True)


def _cap_breach(event: dict[str, Any], options: AgentOptions, turns: int, usage: dict[str, int]) -> str | None:
    """Return the cap ``event`` breached, or ``None``.

    ``codex exec`` has no turn or budget cap of its own, so acumen enforces both against the
    event stream and stops the process on a breach (see :func:`_stop_codex`). The two caps are
    enforced at different resolutions, and the difference matters:

    * ``max_turns`` is checked as each model action completes, so the run really is stopped at
      the cap — the agent gets ``max_turns`` actions and no more.
    * ``max_usd`` can only be checked when usage is reported. The JSONL stream reports it once,
      at the end of the turn, but Codex records a running total after every model response in
      its rollout file, and ``usage`` here is whatever :class:`_CodexUsageTail` has read so far.
      The budget therefore bites during the turn rather than after it, to the resolution of one
      model response: a single response can still overshoot the cap, and a run whose rollout
      cannot be read falls back to being caught at ``turn.completed``.
    """
    if options.max_turns is not None and _is_turn_item(event) and turns >= options.max_turns:
        return _MAX_TURNS_SUBTYPE
    if options.max_usd is None or options.price_usd is None or not usage:
        return None
    spent = options.price_usd(usage)
    return _MAX_BUDGET_SUBTYPE if spent is not None and spent > options.max_usd else None


#: How long a capped run is given to shut down on its own before it is killed. Codex writes the
#: usage record for the response that has just landed as it exits, and losing that record costs
#: the run tokens it really paid for. A clean exit takes milliseconds; the ceiling is only there
#: so a build that ignores the interrupt cannot hang the benchmark.
_STOP_GRACE_S = 5.0


async def _stop_codex(process: asyncio.subprocess.Process) -> None:
    """Stop a capped run, giving Codex the chance to flush its last usage record.

    SIGINT is the operator interrupt Codex handles, and it exits cleanly enough to finish
    writing the rollout. SIGTERM does not: measured against the same capped run, the hard kill
    lost the tokens of the last model responses, which is exactly the accounting this is here to
    get right. A run that ignores the interrupt is killed rather than left hanging.
    """
    if process.returncode is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.wait(), timeout=_STOP_GRACE_S)
    except (ProcessLookupError, TimeoutError):
        with suppress(ProcessLookupError):
            process.terminate()


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
        limit=10 * 1024 * 1024,  # 10 MB; default 64 KB is too small for large Codex events
    )
    assert process.stdout is not None
    assert process.stderr is not None
    noise: list[str] = []
    stderr_task = asyncio.create_task(_drain_stderr(process.stderr, options.stderr, noise))
    events: list[dict[str, Any]] = []
    turns = 0
    capped: str | None = None
    tail = _CodexUsageTail(options.env.get("CODEX_HOME"))
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
            if event.get("type") == "thread.started":
                tail.attach(str(event.get("thread_id") or ""))
            if _is_turn_item(event):
                turns += 1
            tail.poll()
            capped = _cap_breach(event, options, turns, _event_usage(event) or tail.usage)
            if capped is not None:
                await _stop_codex(process)
                break
        returncode = await process.wait()
        # Codex flushes the turn's last token_count after its last stdout line, and a
        # terminated run's final counts land only once the process is gone.
        tail.poll()
        await stderr_task
    except BaseException:
        process.terminate()
        await process.wait()
        stderr_task.cancel()
        raise
    duration_ms = round((time.monotonic() - started) * 1000)
    return _codex_terminal(events, returncode, duration_ms, capped, _sandbox_failure(noise), noise, tail.usage)


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
