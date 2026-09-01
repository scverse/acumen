"""The improving agent: read train evidence, write ``skills/v{n+1}/``.

Where the drafter reads the package *source* to write the first skill, the improver
reads how the current skill *performed* — on the TRAIN split only — and edits it. Versions
are immutable: ``improve`` always writes a new directory, never mutates an existing one.

The load-bearing constraint is that the improver must never see test results, or the whole
benchmark is worthless. This is enforced two ways, not one:

1. **Structurally.** The agent's ``cwd`` is a throwaway work dir and its readable material
   is a *curated copy* of train-split runs. The real ``runs/`` tree is not on its ``cwd``,
   not in the read-only roots, and not in scope — so there is no test directory to reach in the
   first place.
2. **Belt-and-braces.** A ``PreToolUse`` hook denies any tool call whose path resolves under
   a ``runs/*/test/`` subtree, so even a call that names an absolute path back into the
   project runs tree is refused. See :func:`find_test_access`.

Prompt-level instruction alone is explicitly *not* sufficient.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import HookMatcher

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import (
    ANSWER_FILE,
    RESULT_FILE,
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    TRANSCRIPT_JSONL,
    arm_name,
    parse_run_dir,
)
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import improve_prompt
from acumen.skills import (
    SKILL_FILE,
    Skill,
    SkillError,
    content_files,
    load_skill,
    next_version,
    skill_dir,
    write_meta,
)
from acumen.tasks import Task

#: The subtree component that marks a held-out split. A path under ``runs/<arm>/test/…`` is
#: what the improver must never reach.
_TEST_SPLIT = "test"

#: tool_input keys that carry a filesystem path across the default Claude Code toolset.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens. Coarse on
#: purpose — the structural isolation is the real guarantee; this only has to catch an
#: absolute path named directly in a command.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))


class ImproveError(RuntimeError):
    """Raised when a skill could not be improved."""


@dataclass(frozen=True)
class TrainRun:
    """One train-split benchmark run of the parent skill, as evidence for the improver.

    ``success`` and ``skill_loaded`` are independent outcomes with independent fixes: a run
    that never loaded the skill says nothing about the skill's body, only about its
    ``description``. Both reach the improver so it can tell the two apart.
    """

    task_id: str
    model: str
    rep: int
    prompt: str
    expected: str
    answer: str | None
    reason: str
    success: bool
    directory: Path
    #: Whether the agent actually invoked the skill. ``None`` when the transcript could not
    #: be read — undetermined, which is not the same as "did not load".
    skill_loaded: bool | None = None


@dataclass(frozen=True)
class ImproveResult:
    """A newly improved skill version and what it cost to produce."""

    skill: Skill
    parent: str
    cost_usd: float | None
    turns: int
    n_train_runs: int
    n_train_failures: int
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Train evidence ─────────────────────────────────────────────────────────────────────


def collect_train_runs(runs_root: Path, arm: str, tasks: list[Task]) -> list[TrainRun]:
    """Collect the parent arm's train-split runs into evidence for the improver.

    Walks ``runs_root/<arm>/train`` for ``result.json`` files and pairs each with its task's
    train prompt and ground-truth answer. Only train runs are ever collected — the test
    subtree is never even visited.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    arm
        The arm whose train runs are the improvement signal, e.g. ``"skill_v1"``.
    tasks
        The loaded tasks, to recover each run's prompt and expected answer.

    Returns
    -------
    The train runs, failures first then by task/rep — so a reader sees what went wrong up
    front.
    """
    train_root = runs_root / arm / "train"
    by_id = {task.id: task for task in tasks}
    runs: list[TrainRun] = []
    if not train_root.is_dir():
        return runs
    for result_path in sorted(train_root.rglob(RESULT_FILE)):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError) as err:
            raise ImproveError(f"cannot read {result_path}: {err}") from err
        if data.get("valid", True) is False:
            raise ImproveError(
                "cannot improve from an infrastructure-invalid benchmark result: "
                f"{result_path} ({data.get('reason')}). Fix the harness — replenish the provider "
                "credential, or report the sandbox refusal as a bug if it refused a host — "
                "and resume the benchmark first."
            )
        key = parse_run_dir(runs_root, result_path.parent)
        task = by_id.get(key.task_id)
        if task is None:
            # A run for a task no longer in tasks.yaml is stale evidence; skip it rather
            # than guess a prompt for it.
            continue
        runs.append(
            TrainRun(
                task_id=key.task_id,
                model=str(data.get("model", key.model)),
                rep=key.rep,
                prompt=task.train.prompt.strip(),
                expected=task.train.answer.strip(),
                answer=data.get("answer"),
                reason=str(data.get("reason", "")),
                success=bool(data.get("success", False)),
                directory=result_path.parent,
                skill_loaded=_loaded(data.get("skill_loaded")),
            )
        )
    runs.sort(key=lambda r: (r.success, r.task_id, r.model, r.rep))
    return runs


def _loaded(value: Any) -> bool | None:
    """Coerce a ``result.json`` ``skill_loaded`` field, preserving "undetermined" as ``None``."""
    return None if value is None else bool(value)


def load_rates(runs: list[TrainRun]) -> dict[str, tuple[int, int, int]]:
    """Count how often the skill actually loaded, per model.

    Reported per model rather than pooled because the load rate varies far more across models
    than across skill versions — a pooled number invites the improver to chase a model whose
    behaviour no wording can reach.

    Parameters
    ----------
    runs
        The train runs to tally.

    Returns
    -------
    ``{model: (loaded, total, undetermined)}``, where ``undetermined`` counts runs whose
    transcript could not be read and which are therefore excluded from ``loaded``.
    """
    rates: dict[str, tuple[int, int, int]] = {}
    for run in runs:
        loaded, total, unknown = rates.get(run.model, (0, 0, 0))
        rates[run.model] = (
            loaded + (run.skill_loaded is True),
            total + 1,
            unknown + (run.skill_loaded is None),
        )
    return dict(sorted(rates.items()))


def _load_mark(loaded: bool | None) -> str:
    """The phrase naming a run's load outcome, used identically in SUMMARY.md and run.md."""
    if loaded is None:
        return "skill load UNDETERMINED"
    return "skill LOADED" if loaded else "skill NOT LOADED"


def _load_section(runs: list[TrainRun]) -> list[str]:
    """The SUMMARY.md section reporting per-model load rates."""
    rates = load_rates(runs)
    lines = [
        "## Did the skill load at all?",
        "",
        "A run where the skill never loaded is not evidence about the skill's body — the agent",
        "never read it. Only the `description` in the frontmatter decides whether a skill loads.",
        "",
        "| model | loaded | runs | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_loaded = total_runs = total_unknown = 0
    for model, (loaded, total, unknown) in rates.items():
        total_loaded += loaded
        total_runs += total
        total_unknown += unknown
        lines.append(f"| `{model}` | {loaded} | {total} | {loaded / total:.0%} |")
    if total_runs:
        lines.extend(["", f"**Overall: {total_loaded}/{total_runs} ({total_loaded / total_runs:.0%}).**"])
    if total_unknown:
        noun = "run" if total_unknown == 1 else "runs"
        lines.append(
            f"({total_unknown} {noun} had no readable transcript, so loading is undetermined and is "
            "counted as not loaded above.)"
        )
    lines.append("")
    return lines


def _write_material(train_dir: Path, runs: list[TrainRun]) -> None:
    """Lay out the curated train evidence the improver reads.

    A ``SUMMARY.md`` digest (per-model load rates, then runs with failures first) plus one
    directory per run holding the run's ``script.py``, transcript, and a ``run.md`` stating
    prompt/expected/actual/outcome and whether the skill loaded. Copied, not linked, so the
    agent never has a path back into the real runs tree.
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for r in runs if not r.success)
    noun = "run" if len(runs) == 1 else "runs"
    lines = [
        "# Train-split evidence for the current skill",
        "",
        "Each run below benchmarked the current skill on one train task. Two things were",
        "recorded per run, and they fail for different reasons and take different fixes:",
        "whether the agent got the answer RIGHT, and whether the agent LOADED the skill at",
        "all. Read the load rates first, then the failing runs. Open a run's `run.md`,",
        "`script.py`, and `transcript.html` for the detail.",
        "",
        f"**{len(runs)} {noun}, {n_fail} failing.**",
        "",
        *_load_section(runs),
        "## Runs",
        "",
    ]
    for run in runs:
        slug = f"{run.task_id}__{run.model}__rep_{run.rep}"
        mark = "PASS" if run.success else "FAIL"
        lines.append(
            f"- `{slug}/` — **{mark}** ({run.reason}) — {_load_mark(run.skill_loaded)} — "
            f"task `{run.task_id}` on `{run.model}` — expected `{run.expected}`, got `{run.answer!r}`"
        )
        run_out = train_dir / slug
        run_out.mkdir(parents=True, exist_ok=True)
        (run_out / "run.md").write_text(
            f"# {run.task_id} — {mark} ({run.reason})\n\n"
            f"- Model: `{run.model}`\n"
            f"- Skill: {_load_mark(run.skill_loaded)}\n\n"
            f"## Task prompt\n\n{run.prompt}\n\n"
            f"## Expected answer\n\n{run.expected}\n\n"
            f"## Answer the agent gave\n\n{run.answer!r}\n\n"
            f"## Files\n\n- `script.py` — the script the agent wrote\n"
            f"- `transcript.html` — the full run transcript\n"
        )
        for name in (SCRIPT_FILE, ANSWER_FILE, TRANSCRIPT_HTML, TRANSCRIPT_JSONL):
            src = run.directory / name
            if src.is_file():
                shutil.copyfile(src, run_out / name)
    (train_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ── Test-access guard ──────────────────────────────────────────────────────────────────


def _blocks(candidate: str, runs_root: Path) -> bool:
    """Return whether ``candidate`` resolves under a ``runs/*/test/`` subtree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        rel = resolved.relative_to(runs_root)
    except ValueError:
        return False
    parts = rel.parts
    # runs_root/<arm>/test/... — the split is the second component.
    return len(parts) >= 2 and parts[1] == _TEST_SPLIT


def find_test_access(tool_name: str, tool_input: dict[str, Any], runs_root: Path) -> str | None:
    """Return the first path in a tool call that reaches held-out test results, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without
    standing up an agent. Checks the path-bearing tool_input keys, and — for shell tools —
    the whitespace/metacharacter-split tokens of the command, since a Bash call can name a
    path no structured field would.

    Parameters
    ----------
    tool_name
        The tool being invoked; unused today but kept so the guard can special-case tools.
    tool_input
        The tool's arguments.
    runs_root
        The project ``runs/`` root, resolved by the caller.

    Returns
    -------
    The offending path string, or ``None`` if the call touches no test results.
    """
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and _blocks(value, runs_root):
            return value
    command = tool_input.get("command")
    if isinstance(command, str):
        for token in command.translate(_SHELL_SPLIT).split():
            token = token.rstrip(",;")
            if ("runs" in token or token.startswith("/") or token.startswith("~")) and _blocks(token, runs_root):
                return token
    return None


def make_test_guard(runs_root: Path) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies any access to held-out test results.

    ``matcher=None`` fires the hook for every tool. The hook resolves paths against the real
    project ``runs/`` root, so it holds regardless of the agent's ``cwd``.
    """
    # Imported here, not at module scope: the Claude SDK is an optional dependency and a
    # Codex-only install never builds an SDK hook.
    from claude_agent_sdk import HookMatcher

    root = runs_root.resolve()

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_test_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen blocks the improver from reading held-out test results ({hit}). "
                    "Only train-split evidence is available to you."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])


# ── Orchestration ──────────────────────────────────────────────────────────────────────


def _seed_staging(staging: Path, parent: Skill) -> None:
    """Pre-fill the staging dir with the parent skill's content, so the agent edits in place.

    Only content files are copied — ``meta.json`` is acumen bookkeeping and is written
    fresh once the new version is promoted.
    """
    staging.mkdir(parents=True, exist_ok=True)
    for src in content_files(parent.directory):
        dest = staging / src.relative_to(parent.directory)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def _validate_staged(staging: Path, skill_name: str) -> None:
    """Fail loudly if the agent's output isn't a usable skill, before it becomes a version."""
    if not (staging / SKILL_FILE).is_file():
        raise ImproveError(
            f"the improving agent left no {SKILL_FILE} in the staging directory — nothing to "
            "promote. Inspect the run log or the prompt."
        )
    try:
        load_skill(staging.parent, staging.name, expect_name=skill_name)
    except SkillError as err:
        raise ImproveError(f"the improved skill is not valid: {err}") from err


async def improve_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    runs_root: Path,
    tasks: list[Task],
    auth_mode: AuthMode = "session",
    prices: PriceTable | None = None,
    parent_version: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    feedback: str | None = None,
    log: LiveLog | None = None,
) -> ImproveResult:
    """Improve the current skill into the next version from its train-split evidence.

    Parameters
    ----------
    cfg
        The pass config; supplies ``skill_name`` and ``improve_model``.
    target
        The prepared target, supplying the interpreter (for verifying claims). Its source
        is **not** exposed — the improver works from evidence, not code.
    skills_root
        The ``skills/`` root; the new version is the next unused one.
    runs_root
        The ``runs/`` root, read for train evidence and guarded against test access.
    tasks
        The loaded tasks, to pair runs with their prompts and expected answers.
    auth_mode
        Which credential the improving agent authenticates with — ``"session"`` (the Claude
        subscription) or ``"api"`` (see :func:`acumen.env.build_agent_env`).
    parent_version
        The version to improve; defaults to the latest present.
    model
        Override for the improving model; defaults to the config's.
    max_turns, max_usd
        Turn and budget caps for the improving agent. **Unbounded by default** — the config's
        ``max_turns``/``max_usd`` cap benchmark agents only; pass explicit caps to bound the
        improver.
    feedback
        Optional maintainer guidance, injected into the improve prompt as a subordinated block
        and recorded in ``meta.json`` as provenance. It is prompt text only — it cannot reach
        the held-out test split, which stays blocked structurally and by the guard hook.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The improved skill, loaded and validated, with its parent and cost.
    """
    parent = load_skill(skills_root, parent_version or _latest_or_error(skills_root), expect_name=cfg.skill_name)
    new_version = next_version(skills_root)
    dest = skill_dir(skills_root, new_version)
    if dest.exists():
        raise ImproveError(f"{dest} already exists — skill versions are immutable and never overwritten")

    arm = arm_name(parent.version)
    train_runs = collect_train_runs(runs_root, arm, tasks)
    if not train_runs:
        raise ImproveError(
            f"no train-split runs found for {parent.version} under {runs_root / arm / 'train'} — "
            f"run `acumen bench --skill {parent.version}` first so the improver has evidence to work from"
        )
    n_failures = sum(1 for r in train_runs if not r.success)

    holder = Path(tempfile.mkdtemp(prefix="acumen-improve-"))
    try:
        work = holder / "work"
        staging = work / new_version
        train_dir = work / "train"
        rationale_path = work / "rationale.md"
        home = holder / "home"
        selected_model = model or cfg.improve_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        _seed_staging(staging, parent)
        _write_material(train_dir, train_runs)

        # Marks the agent's processes so the teardown below can find what it leaves running.
        env = label_env(
            build_agent_env(
                config_dir=config_dir,
                home=home,
                extra_path=[target.bin_dir],
                auth_mode=auth_mode,
                extra_allow=cfg.env_passthrough,
                provider=provider,
            ),
            holder,
        )

        prompt = improve_prompt(
            package=target.pkg_name,
            version=target.pkg_version,
            python=target.python,
            skill_dir=staging,
            train_dir=train_dir,
            rationale_path=rationale_path,
            skill_name=cfg.skill_name,
            parent_version=parent.version,
            new_version=new_version,
            feedback=feedback,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            # No default turn or budget cap: only bound the agent if the caller asked. The
            # config's ``max_turns``/``max_usd`` cap benchmark agents only.
            max_turns=max_turns,
            max_usd=max_usd,
            # Codex reports no billed figure, so a budget cap needs the run's own rate table.
            price_usd=pricer(selected_model, table),
            read_dirs=(target.venv_dir,),
            write_dirs=(work,),
            discover_skills=True,
            # Belt-and-braces over the structural isolation: refuse any call that reaches a
            # held-out test result, wherever the agent points it. Built only for Claude — the
            # hook is an SDK object, and Codex gets the equivalent from ``deny_paths`` below.
            claude_hooks={"PreToolUse": [make_test_guard(runs_root)]} if provider == "claude" else None,
            # Codex gets only the staged train evidence and is denied the original
            # run tree wholesale, which includes every held-out test artifact.
            deny_paths=(runs_root.resolve(),),
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(
                prompt,
                options=options,
                on_event=log.append if log is not None else None,
            )
        except Exception as err:  # noqa: BLE001 - a failed improve is an error to report, re-raised below
            agent_error = err
        finally:
            # Render the HTML log while the throwaway config dir still holds the native
            # transcript — in a finally so an aborted run (the SDK raises on a cap breach,
            # after yielding the result) is still inspectable.
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            raise ImproveError(
                f"the improving agent failed: {type(agent_error).__name__}: {agent_error}"
            ) from agent_error
        if result is None:
            raise ImproveError("the improving agent produced no result message")
        if result.is_error:
            raise ImproveError(f"the improving agent errored: {result.subtype} {result.errors or ''}".strip())

        _validate_staged(staging, cfg.skill_name)

        rationale = _read_rationale(rationale_path, result, parent.version, new_version)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, dest)
        write_meta(dest, parent=parent.version, rationale=rationale, feedback=feedback)
        skill = load_skill(skills_root, new_version, expect_name=cfg.skill_name)
        return ImproveResult(
            skill=skill,
            parent=parent.version,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(
                    result.usage,
                    model=selected_model,
                    provider=result.provider,
                    prices=table,
                ),
            ).cost_usd,
            turns=result.num_turns,
            n_train_runs=len(train_runs),
            n_train_failures=n_failures,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        # Kill anything the agent left running before removing the directory it runs in.
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


def _latest_or_error(skills_root: Path) -> str:
    from acumen.skills import latest_version

    latest = latest_version(skills_root)
    if latest is None:
        raise ImproveError(f"no skill versions found under {skills_root} — run `acumen draft` first, then bench it")
    return latest


def _read_rationale(path: Path, result: AgentResult, parent: str, new_version: str) -> str:
    """Recover the agent's rationale for ``meta.json``.

    Prefers the ``rationale.md`` the prompt asks for; falls back to the agent's final
    message, then to a plain provenance note, so a version always records why it exists.
    """
    if path.is_file():
        text = path.read_text().strip()
        if text:
            return text
    final = (result.result or "").strip()
    if final:
        return final
    return f"improved from {parent} to {new_version}"
