"""The improving agent: read the knowledge wiki, write the next ``skills/vN/``.

The improver no longer reads a raw dump of the parent skill's runs. It reads the **wiki** — the
distilled, per-task ``observations.md``/``hypothesis.md`` notes that :mod:`acumen.wiki` builds from
train-split runs and accumulates across versions — plus the parent skill (in improve mode), the
raw train transcripts (for drill-down), and the *filtered* package source. On the first epoch there
is no parent skill: the wiki holds only the ``noskill`` baseline and the improver **creates** v1.

Two load-bearing isolations, both structural rather than by prompt:

1. **No held-out (valid) results.** The improver's readable material is the wiki + staged train
   transcripts — the real ``runs/`` tree is denied wholesale, and a ``PreToolUse`` hook refuses any
   call that resolves under ``runs/*/valid/`` wherever it is pointed (see :func:`find_valid_access`).
2. **No skill bias from the target.** A package may already ship its own skill/agent guidance; the
   improver reads the *filtered* source (:func:`acumen.scrub.build_filtered_source`) with those
   artifacts stripped, and the raw checkout denied, so the optimization never re-serves a skill the
   package already carries. This matters precisely because acumen's job is to ship skills into
   packages.
"""

from __future__ import annotations

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
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    arm_name,
)
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import improve_prompt
from acumen.scrub import build_filtered_source, make_skill_guard
from acumen.skills import (
    SKILL_FILE,
    Skill,
    SkillError,
    content_files,
    latest_version,
    load_skill,
    next_version,
    skill_dir,
    write_meta,
)
from acumen.tasks import Task
from acumen.wiki import WIKI_DIRNAME, collect_arm_runs

#: The subtree component that marks the held-out split. A path under ``runs/<arm>/valid/…`` is
#: what the improver must never reach.
_VALID_SPLIT = "valid"

#: tool_input keys that carry a filesystem path across the default Claude Code toolset.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens. Coarse on
#: purpose — the structural isolation is the real guarantee; this only has to catch an
#: absolute path named directly in a command.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))


class ImproveError(RuntimeError):
    """Raised when a skill could not be improved."""


@dataclass(frozen=True)
class ImproveResult:
    """A newly created or improved skill version and what it cost to produce."""

    skill: Skill
    #: The parent version, or ``None`` when this is the first (created) skill.
    parent: str | None
    cost_usd: float | None
    turns: int
    n_train_runs: int
    n_train_failures: int
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Staging ──────────────────────────────────────────────────────────────────────────────


def _seed_staging(staging: Path, parent: Skill) -> None:
    """Pre-fill the staging dir with the parent skill's content, so the agent edits in place.

    Only content files are copied — ``meta.json`` is acumen bookkeeping and is written fresh once
    the new version is promoted.
    """
    staging.mkdir(parents=True, exist_ok=True)
    for src in content_files(parent.directory):
        dest = staging / src.relative_to(parent.directory)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def _stage_wiki(wiki_root: Path, dest: Path) -> None:
    """Copy the whole wiki into the agent's work dir, so it never reaches back into the project."""
    dest.mkdir(parents=True, exist_ok=True)
    if wiki_root.is_dir():
        shutil.copytree(wiki_root, dest, dirs_exist_ok=True)


def _stage_transcripts(runs_root: Path, arm: str, tasks: list[Task], dest: Path) -> int:
    """Copy the parent arm's train transcripts into the work dir for drill-down.

    Returns the number of runs staged. Copies transcript/script/answer per run into
    ``dest/<task>/<model>__rep_<n>/`` — copies, so there is no path back into the real tree.
    """
    runs = collect_arm_runs(runs_root, arm, tasks, split="train")
    for run in runs:
        run_out = dest / run.task_id / f"{run.model}__rep_{run.rep}"
        run_out.mkdir(parents=True, exist_ok=True)
        for name in (TRANSCRIPT_HTML, SCRIPT_FILE, ANSWER_FILE):
            src = run.directory / name
            if src.is_file():
                shutil.copyfile(src, run_out / name)
    return len(runs)


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


# ── Valid-access guard ─────────────────────────────────────────────────────────────────────


def _blocks(candidate: str, runs_root: Path) -> bool:
    """Return whether ``candidate`` resolves under a ``runs/*/valid/`` subtree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        rel = resolved.relative_to(runs_root)
    except ValueError:
        return False
    parts = rel.parts
    # runs_root/<arm>/valid/... — the split is the second component.
    return len(parts) >= 2 and parts[1] == _VALID_SPLIT


def find_valid_access(tool_name: str, tool_input: dict[str, Any], runs_root: Path) -> str | None:
    """Return the first path in a tool call that reaches held-out valid results, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without standing up an
    agent. Checks the path-bearing tool_input keys, and — for shell tools — the whitespace/
    metacharacter-split tokens of the command, since a Bash call can name a path no structured
    field would.

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
    The offending path string, or ``None`` if the call touches no valid results.
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


def make_valid_guard(runs_root: Path) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies any access to held-out valid results.

    ``matcher=None`` fires the hook for every tool. The hook resolves paths against the real
    project ``runs/`` root, so it holds regardless of the agent's ``cwd``.
    """
    # Imported here, not at module scope: the Claude SDK is an optional dependency and a
    # Codex-only install never builds an SDK hook.
    from claude_agent_sdk import HookMatcher

    root = runs_root.resolve()

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_valid_access(
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
                    f"acumen blocks the improver from reading held-out valid results ({hit}). "
                    "Only train-split evidence is available to you."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])


# ── Orchestration ──────────────────────────────────────────────────────────────────────────


async def improve_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    runs_root: Path,
    wiki_root: Path,
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
    """Create or improve a skill from the knowledge wiki.

    When ``parent_version`` is ``None`` and no skill versions exist, the wiki holds only the
    ``noskill`` baseline and the agent CREATES v1 from it + the filtered source. Otherwise it
    improves ``parent_version`` (default: latest) into the next version, editing a pre-filled copy.

    Parameters
    ----------
    cfg
        The pass config; supplies ``skill_name`` and ``meta_model``.
    target
        The prepared target. Its source is exposed only as a FILTERED copy (bundled skills
        stripped); the interpreter is available for verifying claims.
    skills_root
        The ``skills/`` root; the new version is the next unused one.
    runs_root
        The ``runs/`` root, read for train transcripts and guarded against valid access.
    wiki_root
        The ``wiki/`` root, the improver's primary distilled signal.
    tasks
        The loaded tasks, to locate the parent arm's train transcripts.
    auth_mode
        Which credential the improving agent authenticates with.
    parent_version
        The version to improve; defaults to the latest present, or ``None`` (create) when none
        exist.
    model
        Override for the improving model; defaults to the config's.
    max_turns, max_usd
        Turn and budget caps. Unbounded by default — the config's caps bound benchmark agents only.
    feedback
        Optional maintainer guidance, subordinated below the hard rules and recorded in
        ``meta.json``. It cannot reach the held-out valid split.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The created/improved skill, loaded and validated, with its parent and cost.
    """
    # Resolve parent: an explicit version, else the latest present, else None (create the first).
    resolved_parent = parent_version if parent_version is not None else latest_version(skills_root)
    parent = None if resolved_parent is None else load_skill(skills_root, resolved_parent, expect_name=cfg.skill_name)
    parent_ver = parent.version if parent is not None else None

    new_version = next_version(skills_root)
    dest = skill_dir(skills_root, new_version)
    if dest.exists():
        raise ImproveError(f"{dest} already exists — skill versions are immutable and never overwritten")

    parent_arm = arm_name(parent_ver)
    train_runs = collect_arm_runs(runs_root, parent_arm, tasks, split="train")
    if not train_runs:
        hint = (
            "acumen bench --no-skill --split train"
            if parent_ver is None
            else f"acumen bench --skill {parent_ver} --split train"
        )
        raise ImproveError(
            f"no train-split runs found for {parent_arm} under {runs_root / parent_arm / 'train'} — "
            f"run `{hint}` first so the wiki and improver have evidence to work from"
        )
    n_failures = sum(1 for r in train_runs if not r.success)

    holder = Path(tempfile.mkdtemp(prefix="acumen-improve-"))
    try:
        work = holder / "work"
        staging = work / new_version
        wiki_dir = work / WIKI_DIRNAME
        transcripts_dir = work / "transcripts"
        rationale_path = work / "rationale.md"
        home = holder / "home"
        selected_model = model or cfg.meta_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        if parent is not None:
            _seed_staging(staging, parent)
        else:
            staging.mkdir(parents=True, exist_ok=True)
        _stage_wiki(wiki_root, wiki_dir)
        _stage_transcripts(runs_root, parent_arm, tasks, transcripts_dir)
        # The filtered source: a package's own shipped skill must not bias the optimization.
        source_copy = build_filtered_source(target.src_dir, holder / "source")

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
            src=source_copy,
            python=target.python,
            skill_dir=staging,
            wiki_dir=wiki_dir,
            transcripts_dir=transcripts_dir,
            rationale_path=rationale_path,
            skill_name=cfg.skill_name,
            new_version=new_version,
            parent_version=parent_ver,
            feedback=feedback,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            # No default turn or budget cap: only bound the agent if the caller asked.
            max_turns=max_turns,
            max_usd=max_usd,
            # Codex reports no billed figure, so a budget cap needs the run's own rate table.
            price_usd=pricer(selected_model, table),
            # The filtered source + venv, for verifying claims. Never the raw checkout.
            read_dirs=(source_copy, target.venv_dir),
            write_dirs=(work,),
            discover_skills=True,
            # Belt-and-braces over the structural isolation: refuse any call that reaches a
            # held-out valid result, or an unfiltered source artifact. Built only for Claude.
            claude_hooks=(
                {"PreToolUse": [make_valid_guard(runs_root), make_skill_guard(target.src_dir)]}
                if provider == "claude"
                else None
            ),
            # Codex gets only the staged evidence and filtered source; deny the run tree and the
            # original checkout wholesale.
            deny_paths=(runs_root.resolve(), target.src_dir.resolve()),
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

        rationale = _read_rationale(rationale_path, result, parent_ver, new_version)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, dest)
        write_meta(dest, parent=parent_ver, rationale=rationale, feedback=feedback)
        skill = load_skill(skills_root, new_version, expect_name=cfg.skill_name)
        return ImproveResult(
            skill=skill,
            parent=parent_ver,
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


def _read_rationale(path: Path, result: AgentResult, parent: str | None, new_version: str) -> str:
    """Recover the agent's rationale for ``meta.json``.

    Prefers the ``rationale.md`` the prompt asks for; falls back to the agent's final message, then
    to a plain provenance note, so a version always records why it exists.
    """
    if path.is_file():
        text = path.read_text().strip()
        if text:
            return text
    final = (result.result or "").strip()
    if final:
        return final
    return f"created {new_version}" if parent is None else f"improved from {parent} to {new_version}"
