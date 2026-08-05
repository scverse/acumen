"""Provider-neutral invocation for a single benchmark run.

Runs one agent in one sandbox, captures its transcript, grades what it wrote, and emits
``result.json`` — the unit of record. Writing ``result.json`` last is deliberate: its
presence is what marks a run complete, which is what makes ``--resume`` safe.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.env import AuthMode, Target, agent_version
from acumen.grade import Grade, Reason, grade_run
from acumen.paths import (
    ANSWER_FILE,
    RESULT_FILE,
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    TRANSCRIPT_JSONL,
    RunKey,
)
from acumen.prices import Rates, normalize_usage, price_run, pricer, rates_payload, resolve_cost, resolve_rates
from acumen.prompts import benchmark_prompt
from acumen.sandbox import Sandbox, sandbox
from acumen.skills import Skill
from acumen.tasks import Task
from acumen.transcript import locate_transcript, render_agent_transcript

#: Result subtypes the CLI reports on a cap breach, mapped to our reason taxonomy.
_SUBTYPE_REASONS: dict[str, Reason] = {
    "error_max_budget_usd": "budget",
    "error_max_turns": "max_turns",
}

# Provider/account exhaustion is a failure of the benchmark infrastructure, not evidence
# about the model. Keep these deliberately narrower than generic 429/rate-limit wording:
# transient throttling is not proof that the account has no remaining usage or credit.
_PROVIDER_EXHAUSTION_MARKERS = (
    "usage limit",
    "usage_limit",
    "hit your limit",
    "quota exceeded",
    "quota_exceeded",
    "exceeded your current quota",
    "insufficient_quota",
    "billing_hard_limit",
    "credit balance",
    "out of credits",
    "no credits remaining",
    "credits are exhausted",
    "purchase more credits",
    "payment required",
    "spending limit",
    "spend limit",
    "token quota",
)


@dataclass(frozen=True)
class RunOutcome:
    """What a run produced, as recorded in ``result.json``."""

    key: RunKey
    success: bool
    reason: Reason
    payload: dict


def _terminal_reason(message: AgentResult) -> Reason | None:
    """Map a cap breach or hard error onto a reason, or ``None`` if the agent finished.

    ``subtype`` is a free-form string from the CLI, so match defensively rather than
    against an exhaustive list we cannot see from here.
    """
    subtype = (message.subtype or "").lower()
    if subtype in _SUBTYPE_REASONS:
        return _SUBTYPE_REASONS[subtype]
    if "budget" in subtype:
        return "budget"
    if "max_turns" in subtype:
        return "max_turns"
    if message.is_error or subtype.startswith("error"):
        return "error"
    return None


def _provider_exhaustion_error(message: AgentResult | None, error: str | None = None) -> str | None:
    """Return provider quota/credit evidence, excluding Acumen's intentional caps.

    Both adapters preserve provider error text, but their subtype vocabularies differ and
    change over time. Matching the stable billing/quota concepts keeps the classification
    provider-neutral while avoiding a false match on ordinary transient rate limiting.
    """
    if message is not None and (message.subtype or "").lower() in _SUBTYPE_REASONS:
        return None
    parts = [error or ""]
    if message is not None:
        parts.extend([message.subtype or "", *(message.errors or [])])
    detail = "\n".join(str(part) for part in parts if part).strip()
    lowered = detail.lower()
    if any(marker in lowered for marker in _PROVIDER_EXHAUSTION_MARKERS):
        return detail
    return None


def _find_transcript(box: Sandbox, session_id: str) -> Path | None:
    return locate_transcript(box.config_dir, box.root, session_id)


def _skill_fired(jsonl: Path, name: str, *, provider: str = "claude") -> bool | None:
    """Return whether the agent loaded the skill called ``name``, read from the transcript.

    This is the evidence the skill arm actually used the skill — the skill arm must show it
    loading and the baseline must not — so it is recorded per run rather than left to
    hand-inspection. ``None`` means the transcript was unavailable, which is not the same
    as the skill not loading.

    The match is on the skill *identity*, not on the tool: a ``Skill`` call carries the
    requested skill in ``input.skill``, and the CLI ships bundled skills of its own that the
    agent can reach in either arm. Loading one of those says nothing about the skill under
    test, so only ``input.skill == name`` counts.
    """
    try:
        lines = jsonl.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if provider == "codex":
            item = record.get("item")
            if not isinstance(item, dict):
                continue
            # Codex loads repository skills from .agents/skills. The JSONL
            # protocol exposes the command used to read the selected SKILL.md.
            command = item.get("command")
            if isinstance(command, str):
                normalized = command.replace("\\", "/")
                if f".agents/skills/{name}/SKILL.md" in normalized:
                    return True
            continue
        if record.get("type") != "assistant":
            continue
        content = record.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use" or block.get("name") != "Skill":
                continue
            payload = block.get("input")
            if isinstance(payload, dict) and payload.get("skill") == name:
                return True
    return False


def _collect_artifacts(box: Sandbox, run_dir: Path) -> None:
    for name in (ANSWER_FILE, SCRIPT_FILE):
        src = box.root / name
        if src.is_file():
            shutil.copyfile(src, run_dir / name)


class StderrFilter:
    """Dedupe the CLI subprocess stderr the SDK forwards, keeping the first of each line.

    A benchmark pass spawns one CLI subprocess per run, and each reprints the same
    startup diagnostics — most visibly the ``claude.ai connectors are disabled`` notice,
    which fires on every spawn because an auth source is present in the agents' env. Left
    alone that is one identical warning per run.

    Share a single instance across a whole pass and hand it to :func:`run_matrix` as its
    ``stderr`` callback. The SDK pipes subprocess stderr only when a callback is set and
    then invokes it one line at a time on the event loop, so this becomes the sole sink:
    the first time a given line is seen it is forwarded to our own stderr, and every later
    repeat is dropped. Distinct lines still all pass through; only exact repeats are cut.

    Called on a single-threaded event loop, so the seen-set needs no locking.
    """

    def __init__(self, sink: TextIO | None = None) -> None:
        self._seen: set[str] = set()
        self._sink = sink if sink is not None else sys.stderr

    def __call__(self, line: str) -> None:
        """Forward ``line`` the first time it is seen; drop it on every later repeat."""
        if line in self._seen:
            return
        self._seen.add(line)
        print(line, file=self._sink, flush=True)


def _build_options(
    *,
    box: Sandbox,
    model: str,
    max_turns: int,
    max_usd: float,
    read_dirs: tuple[Path, ...] = (),
    price_overrides: dict[str, Rates] | None = None,
    stderr: Callable[[str], None] | None = None,
) -> AgentOptions:
    """Assemble the SDK options for one run.

    These options are **identical for every arm** — baseline parity means the
    noskill and skill arms differ only in whether a skill directory exists inside the
    sandbox, never in how the agent is configured.

    ``setting_sources`` is always set explicitly, never left to default: the default of
    ``None`` loads user settings *and* memories, and leaving it unset while
    passing ``skills`` silently re-enables the ``"user"`` source.
    ``["project"]`` scopes discovery to ``<sandbox>/.claude/``, which is exactly where
    the skill is installed — empirically verified, along with the fact that
    ``[]`` disables skill discovery entirely.

    ``skills`` is deliberately left ``None``: with ``setting_sources=["project"]`` the
    skill is discovered from the sandbox anyway (verified), whereas naming it here would
    append ``Skill(<name>)`` to ``allowed_tools`` in the skill arm only — an options-level
    difference between arms, which is the one thing parity forbids.
    """
    return AgentOptions(
        cwd=box.root,
        env=box.env,
        model=model,
        max_turns=max_turns,
        max_usd=max_usd,
        discover_skills=True,
        price_usd=pricer(model, price_overrides),
        read_dirs=read_dirs,
        stderr=stderr,
    )


async def run_once(
    *,
    key: RunKey,
    task: Task,
    target: Target,
    run_dir: Path,
    model: str,
    max_turns: int,
    max_usd: float,
    auth_mode: AuthMode = "api",
    skill: Skill | None = None,
    skill_name: str | None = None,
    sandbox_base: Path | None = None,
    keep_sandbox: bool = False,
    stderr: Callable[[str], None] | None = None,
    env_passthrough: Sequence[str] | None = None,
    price_overrides: dict[str, Rates] | None = None,
) -> RunOutcome:
    """Execute one benchmark run end to end and write its ``result.json``.

    Parameters
    ----------
    key
        Identifies the run; also determines ``run_dir`` upstream.
    task
        The task being run; ``key.split`` selects which half.
    target
        The prepared target, supplying the interpreter and fingerprint.
    run_dir
        Where the five run files are written.
    model, max_turns, max_usd
        Already resolved against per-task overrides by the caller.
    auth_mode
        Which credential the benchmark run authenticates with. Under ``"session"``, recorded
        Claude's SDK cost is API-equivalent rather than necessarily an invoice charge;
        Codex falls back to frozen-table token inference.
    skill
        The skill to install, or ``None`` for the baseline arm. Must agree with
        ``key.arm``, else the run would be filed under an arm it doesn't belong to.
    skill_name
        Name of the skill under test, used to recognise it in the transcript. Defaults to
        the installed skill's name; pass it in the baseline arm too, where nothing is
        installed but a stray load of that same skill is exactly what must be caught.
        Without either, ``skill_loaded`` is left ``None`` — unattributable, not absent.
    sandbox_base
        Parent directory for the throwaway sandbox.
    keep_sandbox
        Leave the sandbox on disk — useful when hand-checking a failure.
    stderr
        Callback for the CLI subprocess's stderr, one line at a time. Pass a shared
        :class:`StderrFilter` across a pass to collapse the per-spawn startup warnings;
        ``None`` (the default) lets the subprocess stderr inherit the terminal unfiltered.
    env_passthrough
        Extra environment variable names to carry into the sandbox on top of the built-in
        allowlist (the operator's ``config.env_passthrough``).

    Returns
    -------
    The outcome, matching what was written to disk.
    """
    if (key.skill is None) != (skill is None):
        # Filing a skill run under noskill/ (or vice versa) would silently corrupt the
        # comparison the whole benchmark rests on, so refuse rather than record it.
        raise ValueError(
            f"arm {key.arm!r} expects skill {key.skill!r} but was given {skill.version if skill else None!r}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    split = task.split(key.split)

    provider = provider_for_model(model)
    result: AgentResult | None = None
    error: str | None = None

    with sandbox(
        target,
        auth_mode=auth_mode,
        base=sandbox_base,
        keep=keep_sandbox,
        skill=skill,
        env_passthrough=env_passthrough,
        provider=provider,
    ) as box:
        prompt = benchmark_prompt(
            split.prompt,
            sandbox=box.root,
            python=target.python,
            package=target.pkg_name,
        )
        options = _build_options(
            box=box,
            model=model,
            max_turns=max_turns,
            max_usd=max_usd,
            read_dirs=(target.venv_dir,),
            price_overrides=price_overrides,
            stderr=stderr,
        )
        try:
            result = await run_agent(prompt, options=options)
        except Exception as err:  # noqa: BLE001 - a crashed agent is a recorded failure, not a crashed pass
            error = f"{type(err).__name__}: {err}"

        _collect_artifacts(box, run_dir)
        if result is not None and provider == "claude" and result.session_id is not None:
            jsonl = _find_transcript(box, result.session_id)
            if jsonl is not None:
                shutil.copyfile(jsonl, run_dir / TRANSCRIPT_JSONL)
        elif result is not None and provider == "codex":
            (run_dir / TRANSCRIPT_JSONL).write_text("".join(f"{json.dumps(event)}\n" for event in result.transcript))

    rendered = False
    skill_loaded: bool | None = None
    expected_skill = skill.name if skill is not None else skill_name
    if (run_dir / TRANSCRIPT_JSONL).is_file():
        rendered = render_agent_transcript(run_dir / TRANSCRIPT_JSONL, run_dir / TRANSCRIPT_HTML, provider=provider)
        if expected_skill is not None:
            skill_loaded = _skill_fired(run_dir / TRANSCRIPT_JSONL, expected_skill, provider=provider)

    grade: Grade = grade_run(run_dir, split.answer)
    exhaustion = _provider_exhaustion_error(result, error)
    if exhaustion is not None:
        success, reason = False, "provider_exhausted"
    elif error is not None or result is None:
        success, reason = False, "error"
    else:
        terminal = _terminal_reason(result)
        if terminal is not None:
            # A cap breach is a failure regardless of what landed in answer.md.
            success, reason = False, terminal
        else:
            success, reason = grade.success, grade.reason

    usage = normalize_usage(result.usage if result else None, provider=provider)
    # Preserve frozen-table inference for both providers, but prefer the SDK's value when it
    # exists. For session authentication that provider figure is API-equivalent, not
    # necessarily money charged on an invoice.
    rates = resolve_rates(model, price_overrides)
    inferred = price_run(usage, rates)
    provider_cost = result.total_cost_usd if result else None
    cost = resolve_cost(provider_cost, inferred)
    payload = {
        "task_id": key.task_id,
        "split": key.split,
        "skill": key.skill,
        "arm": key.arm,
        "model": model,
        "agent": provider,
        "provider": "anthropic" if provider == "claude" else "openai",
        "backend": "claude_agent_sdk" if provider == "claude" else "codex_cli",
        # Which credential paid for the run. Under "session", provider cost is an
        # API-equivalent SDK estimate rather than necessarily metered spend.
        "auth_mode": auth_mode,
        "rep": key.rep,
        "success": success,
        "reason": reason,
        # Provider quota/credit exhaustion invalidates the measurement. Keeping a diagnostic
        # result is useful, but reports and resume must never treat it as benchmark evidence.
        "valid": reason != "provider_exhausted",
        "turns": result.num_turns if result else 0,
        "cost_usd": cost.cost_usd,
        "cost_available": cost.available,
        "cost_source": cost.cost_source,
        "provider_cost_usd": cost.provider_cost_usd,
        "inferred_cost_usd": cost.inferred_cost_usd,
        "cost_delta_usd": cost.cost_delta_usd,
        "cost_delta_pct": cost.cost_delta_pct,
        "price_rates": rates_payload(rates),
        "input_tokens": usage.input,
        "output_tokens": usage.output,
        "cache_read_tokens": usage.cache_read,
        "cache_write_tokens": usage.cache_write,
        "cache_write_5m_tokens": usage.cache_write_5m,
        "cache_write_1h_tokens": usage.cache_write_1h,
        "duration_s": round((result.duration_ms if result else 0) / 1000, 2),
        "answer": grade.answer,
        "expected": split.answer.strip(),
        "pkg_version": target.fingerprint,
        "commit": target.commit,
        "skill_hash": skill.hash if skill else None,
        "skill_name": skill.name if skill else None,
        # Evidence, not configuration: did the agent actually invoke the Skill tool?
        "skill_loaded": skill_loaded,
        "sdk_version": agent_version(provider),
        "session_id": result.session_id if result else None,
        "stop_reason": result.stop_reason if result else None,
        "subtype": result.subtype if result else None,
        "transcript_html": rendered,
        "error": exhaustion or error or (result.errors if result else None),
    }
    (run_dir / RESULT_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    return RunOutcome(key=key, success=success, reason=reason, payload=payload)
