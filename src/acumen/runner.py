"""SDK invocation for a single benchmark run.

Runs one agent in one sandbox, captures its transcript, grades what it wrote, and emits
``result.json`` — the unit of record. Writing ``result.json`` last is deliberate: its
presence is what marks a run complete, which is what makes ``--resume`` safe.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, project_key_for_directory, query

from acumen.env import Target, sdk_version
from acumen.grade import Grade, Reason, grade_run
from acumen.paths import (
    ANSWER_FILE,
    RESULT_FILE,
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    TRANSCRIPT_JSONL,
    RunKey,
)
from acumen.prompts import benchmark_prompt
from acumen.sandbox import Sandbox, sandbox
from acumen.tasks import Task

#: Result subtypes the CLI reports on a cap breach, mapped to our reason taxonomy.
_SUBTYPE_REASONS: dict[str, Reason] = {
    "error_max_budget_usd": "budget",
    "error_max_turns": "max_turns",
}


@dataclass(frozen=True)
class RunOutcome:
    """What a run produced, as recorded in ``result.json``."""

    key: RunKey
    success: bool
    reason: Reason
    payload: dict


def _terminal_reason(message: ResultMessage) -> Reason | None:
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


def _find_transcript(box: Sandbox, session_id: str) -> Path | None:
    key = project_key_for_directory(str(box.root))
    direct = box.transcript_root / key / f"{session_id}.jsonl"
    if direct.is_file():
        return direct
    # The encoding is deterministic, but fall back to a search rather than lose the
    # transcript if the CLI ever keys the directory differently.
    matches = sorted(box.transcript_root.glob(f"**/{session_id}.jsonl"))
    return matches[0] if matches else None


def _claude_code_log() -> str | None:
    """Locate the ``claude-code-log`` CLI.

    It is a dependency of acumen, so it lives next to the interpreter running us — look
    there before PATH, which won't contain the venv's ``bin`` when acumen is invoked by
    absolute path rather than through an activated venv.
    """
    local = Path(sys.executable).parent / "claude-code-log"
    if local.is_file():
        return str(local)
    return shutil.which("claude-code-log")


def _render_transcript(jsonl: Path, html: Path) -> bool:
    cli = _claude_code_log()
    if cli is None:
        return False
    proc = subprocess.run([cli, str(jsonl), "-o", str(html)], capture_output=True, text=True)
    return proc.returncode == 0 and html.is_file()


def _collect_artifacts(box: Sandbox, run_dir: Path) -> None:
    for name in (ANSWER_FILE, SCRIPT_FILE):
        src = box.root / name
        if src.is_file():
            shutil.copyfile(src, run_dir / name)


def _usage_tokens(usage: dict | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    inp = int(usage.get("input_tokens", 0) or 0)
    inp += int(usage.get("cache_read_input_tokens", 0) or 0)
    inp += int(usage.get("cache_creation_input_tokens", 0) or 0)
    return inp, int(usage.get("output_tokens", 0) or 0)


def _build_options(
    *,
    box: Sandbox,
    model: str,
    max_turns: int,
    max_usd: float,
) -> ClaudeAgentOptions:
    """Assemble the SDK options for one run.

    ``setting_sources=[]`` is set explicitly, never left to default: the default of
    ``None`` loads user settings *and* ``CLAUDE.md`` memories, which silently breaks the
    isolation requirement (§3, trap 1).
    """
    return ClaudeAgentOptions(
        cwd=str(box.root),
        env=box.env,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_usd,
        setting_sources=[],
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code"},
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
    sandbox_base: Path | None = None,
    keep_sandbox: bool = False,
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
    sandbox_base
        Parent directory for the throwaway sandbox.
    keep_sandbox
        Leave the sandbox on disk — useful when hand-checking a failure.

    Returns
    -------
    The outcome, matching what was written to disk.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    split = task.split(key.split)

    result: ResultMessage | None = None
    error: str | None = None

    with sandbox(target, base=sandbox_base, keep=keep_sandbox) as box:
        prompt = benchmark_prompt(
            split.prompt,
            sandbox=box.root,
            python=target.python,
            package=target.pkg_name,
        )
        options = _build_options(box=box, model=model, max_turns=max_turns, max_usd=max_usd)
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # noqa: BLE001 - a crashed agent is a recorded failure, not a crashed pass
            error = f"{type(err).__name__}: {err}"

        _collect_artifacts(box, run_dir)
        if result is not None:
            jsonl = _find_transcript(box, result.session_id)
            if jsonl is not None:
                shutil.copyfile(jsonl, run_dir / TRANSCRIPT_JSONL)

    rendered = False
    if (run_dir / TRANSCRIPT_JSONL).is_file():
        rendered = _render_transcript(run_dir / TRANSCRIPT_JSONL, run_dir / TRANSCRIPT_HTML)

    grade: Grade = grade_run(run_dir, split.answer)
    if error is not None or result is None:
        success, reason = False, "error"
    else:
        terminal = _terminal_reason(result)
        if terminal is not None:
            # A cap breach is a failure regardless of what landed in answer.md (§2).
            success, reason = False, terminal
        else:
            success, reason = grade.success, grade.reason

    inp, out = _usage_tokens(result.usage if result else None)
    payload = {
        "task_id": key.task_id,
        "split": key.split,
        "skill": key.skill,
        "arm": key.arm,
        "model": model,
        "rep": key.rep,
        "success": success,
        "reason": reason,
        "turns": result.num_turns if result else 0,
        "cost_usd": (result.total_cost_usd if result else None) or 0.0,
        "input_tokens": inp,
        "output_tokens": out,
        "duration_s": round((result.duration_ms if result else 0) / 1000, 2),
        "answer": grade.answer,
        "expected": split.answer.strip(),
        "pkg_version": target.fingerprint,
        "commit": target.commit,
        "skill_hash": None,
        "sdk_version": sdk_version(),
        "session_id": result.session_id if result else None,
        "stop_reason": result.stop_reason if result else None,
        "subtype": result.subtype if result else None,
        "transcript_html": rendered,
        "error": error or (result.errors if result else None),
    }
    (run_dir / RESULT_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    return RunOutcome(key=key, success=success, reason=reason, payload=payload)
