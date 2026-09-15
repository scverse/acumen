"""The knowledge wiki: distil each arm's train-split runs into durable, per-task notes.

The wiki is a persistent directory (sibling of ``runs/`` and ``skills/``) that *accumulates*
across epochs. For each task it holds two files:

- ``wiki/<task_id>/observations.md`` — one short entry per ``[version][model]`` saying what the
  agents did across replicates, how often the skill loaded, and whether loading led to success.
- ``wiki/<task_id>/hypothesis.md`` — one short entry per ``[version][model]`` saying *why* it did
  or did not work.

Each ``acumen epoch`` benches one arm on the ``train`` split and appends a new block per arm —
starting from ``noskill``. A per-task ``.arms`` marker records which arms have been written, so a
re-run never double-appends and the update is resume-safe.

Brevity is the whole point: the improver reads the entire wiki every epoch, and it grows one block
per ``[version][model]`` forever, so a verbose wiki is actively harmful. The wiki agent is told to
keep entries to a line or two; :func:`_overlong_entries` warns (does not fail) when it does not.

The wiki is built from ``train`` runs ONLY. It is the sole distilled channel from benchmark
results to the improver, so it must never carry anything from the held-out ``valid`` split. Wiki
agents read *staged copies* of the train runs, never the live ``runs/`` tree, and — like every
agent that now reads the package source — read the *filtered* source (bundled skills / agent
guidance stripped) so a package's own shipped skill cannot colour the notes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import (
    ANSWER_FILE,
    RESULT_FILE,
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    Split,
    arm_name,
    parse_run_dir,
)
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import wiki_prompt
from acumen.scrub import build_filtered_source, make_skill_guard
from acumen.skills import content_files
from acumen.tasks import Task

#: The wiki root directory name, a sibling of ``runs/`` and ``skills/``.
WIKI_DIRNAME = "wiki"

#: Per-task files the wiki holds.
OBSERVATIONS_FILE = "observations.md"
HYPOTHESIS_FILE = "hypothesis.md"

#: Per-task marker listing the arms already recorded, one per line. Internal bookkeeping that
#: makes the append idempotent: an arm already listed here is skipped on a re-run.
RECORDED_FILE = ".arms"

#: The staged evidence directory inside a wiki agent's work dir.
EVIDENCE_DIRNAME = "evidence"

#: A soft length budget per wiki entry, in characters. Exceeding it warns, never fails — the
#: note still stands; the warning tells the operator the agent is being verbose.
MAX_ENTRY_CHARS = 400


class WikiError(RuntimeError):
    """Raised when the wiki could not be built."""


@dataclass(frozen=True)
class RunRecord:
    """One benchmark run of an arm on one split, as evidence for the wiki (and the improver).

    ``success`` and ``skill_loaded`` are independent outcomes with independent fixes: a run that
    never loaded the skill says nothing about the skill's body, only about its ``description``.
    Both are carried so a reader can tell the two apart.
    """

    task_id: str
    model: str
    rep: int
    split: str
    prompt: str
    expected: str
    answer: str | None
    reason: str
    success: bool
    directory: Path
    #: Whether the agent actually invoked the skill. ``None`` when the transcript could not be
    #: read — undetermined, which is not the same as "did not load".
    skill_loaded: bool | None = None


@dataclass(frozen=True)
class TaskWikiResult:
    """What one task's wiki update produced."""

    task_id: str
    version: str
    cost_usd: float | None
    turns: int
    #: Entries the agent wrote that exceeded the soft length budget, as human-readable warnings.
    warnings: tuple[str, ...] = ()
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Paths / bookkeeping ──────────────────────────────────────────────────────────────────


def wiki_task_dir(wiki_root: Path, task_id: str) -> Path:
    """Return the wiki directory for one task."""
    return wiki_root / task_id


def recorded_arms(task_dir: Path) -> set[str]:
    """Return the set of arms already recorded for a task, read from its ``.arms`` marker."""
    marker = task_dir / RECORDED_FILE
    if not marker.is_file():
        return set()
    return {line.strip() for line in marker.read_text().splitlines() if line.strip()}


def mark_recorded(task_dir: Path, arm: str) -> None:
    """Record that ``arm`` has been written into a task's wiki, idempotently."""
    task_dir.mkdir(parents=True, exist_ok=True)
    arms = recorded_arms(task_dir)
    if arm in arms:
        return
    arms.add(arm)
    (task_dir / RECORDED_FILE).write_text("\n".join(sorted(arms)) + "\n")


# ── Train evidence ───────────────────────────────────────────────────────────────────────


def _loaded(value: Any) -> bool | None:
    """Coerce a ``result.json`` ``skill_loaded`` field, preserving "undetermined" as ``None``."""
    return None if value is None else bool(value)


def collect_arm_runs(
    runs_root: Path,
    arm: str,
    tasks: Sequence[Task],
    *,
    split: Split = "train",
    task_id: str | None = None,
) -> list[RunRecord]:
    """Collect one arm's runs on one split into evidence, paired with each task's prompt/answer.

    Walks ``runs_root/<arm>/<split>`` for ``result.json`` files and pairs each with its task's
    prompt and ground-truth answer for that split. Only the requested split is ever visited, so
    calling with ``split="train"`` never touches the held-out ``valid`` subtree.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    arm
        The arm to collect, e.g. ``"noskill"`` or ``"skill_v1"``.
    tasks
        The loaded tasks, to recover each run's prompt and expected answer.
    split
        Which split to collect. Defaults to ``"train"``.
    task_id
        Restrict to one task; ``None`` collects every task's runs.

    Returns
    -------
    The runs, failures first then by task/model/rep — so a reader sees what went wrong up front.
    """
    root = runs_root / arm / split
    by_id = {task.id: task for task in tasks}
    runs: list[RunRecord] = []
    if not root.is_dir():
        return runs
    for result_path in sorted(root.rglob(RESULT_FILE)):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError) as err:
            raise WikiError(f"cannot read {result_path}: {err}") from err
        if data.get("valid", True) is False:
            raise WikiError(
                "cannot summarise an infrastructure-invalid benchmark result: "
                f"{result_path} ({data.get('reason')}). Fix the harness — replenish the provider "
                "credential, or report the sandbox refusal as a bug — and resume the benchmark first."
            )
        key = parse_run_dir(runs_root, result_path.parent)
        if task_id is not None and key.task_id != task_id:
            continue
        task = by_id.get(key.task_id)
        if task is None:
            # A run for a task no longer in tasks.yaml is stale evidence; skip it rather than
            # guess a prompt for it.
            continue
        chosen = task.split(key.split)
        runs.append(
            RunRecord(
                task_id=key.task_id,
                model=str(data.get("model", key.model)),
                rep=key.rep,
                split=key.split,
                prompt=chosen.prompt.strip(),
                expected=chosen.answer.strip(),
                answer=data.get("answer"),
                reason=str(data.get("reason", "")),
                success=bool(data.get("success", False)),
                directory=result_path.parent,
                skill_loaded=_loaded(data.get("skill_loaded")),
            )
        )
    runs.sort(key=lambda r: (r.success, r.task_id, r.model, r.rep))
    return runs


def _load_mark(loaded: bool | None) -> str:
    """The phrase naming a run's load outcome."""
    if loaded is None:
        return "skill load UNDETERMINED"
    return "skill LOADED" if loaded else "skill NOT LOADED"


def stage_task_evidence(work: Path, runs: list[RunRecord], *, version: str) -> Path:
    """Lay out one task's staged run evidence for a wiki agent, and return the evidence dir.

    Writes an ``INDEX.md`` digest (pass/fail and load per run) plus one directory per run holding
    the run's ``script.py``, ``answer.md`` and ``transcript.html``. Everything is copied, not
    linked, so the agent never has a path back into the real ``runs/`` tree.
    """
    evidence = work / EVIDENCE_DIRNAME
    evidence.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for r in runs if not r.success)
    noun = "run" if len(runs) == 1 else "runs"
    task_id = runs[0].task_id if runs else "?"
    lines = [
        f"# Train-split runs of `[{version}]` on task `{task_id}`",
        "",
        f"**{len(runs)} {noun} across models and replicates, {n_fail} failing.** Each run recorded",
        "two independent things: whether the agent got the answer RIGHT, and whether it LOADED the",
        "skill at all. Open a run's `transcript.html` for what the agent actually did.",
        "",
    ]
    for run in runs:
        slug = f"{run.model}__rep_{run.rep}"
        mark = "PASS" if run.success else "FAIL"
        lines.append(
            f"- `{slug}/` — **{mark}** ({run.reason}) — {_load_mark(run.skill_loaded)} — "
            f"`{run.model}` — expected `{run.expected}`, got `{run.answer!r}`"
        )
        run_out = evidence / slug
        run_out.mkdir(parents=True, exist_ok=True)
        for name in (SCRIPT_FILE, ANSWER_FILE, TRANSCRIPT_HTML):
            src = run.directory / name
            if src.is_file():
                shutil.copyfile(src, run_out / name)
    (evidence / "INDEX.md").write_text("\n".join(lines) + "\n")
    return evidence


# ── Overlong-entry check ─────────────────────────────────────────────────────────────────


def _overlong_entries(text: str, version: str, *, kind: str) -> list[str]:
    """Return a warning per ``[version]…`` bullet in ``text`` that exceeds the soft length budget.

    Only entries for THIS version are checked, so an older verbose block written by a previous
    epoch is not re-reported every time.
    """
    tag = f"[{version}]"
    warnings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") and tag in stripped and len(stripped) > MAX_ENTRY_CHARS:
            warnings.append(f"{kind} entry for {tag} is {len(stripped)} chars (soft budget {MAX_ENTRY_CHARS})")
    return warnings


# ── Orchestration ────────────────────────────────────────────────────────────────────────


async def _update_one_task(
    *,
    task: Task,
    runs: list[RunRecord],
    version: str,
    arm: str,
    cfg: Config,
    target: Target,
    source_copy: Path,
    runs_root: Path,
    wiki_root: Path,
    skill_body_src: Path | None,
    auth_mode: AuthMode,
    table: PriceTable,
    model: str,
    max_turns: int | None,
    max_usd: float | None,
    log_dir: Path | None,
    stream: bool,
) -> TaskWikiResult:
    """Run one wiki agent for one task+arm, append its block, and mark the arm recorded.

    ``skill_body_src`` is the arm's skill content directory (``skills/vN``) for a skill arm, or
    ``None`` for ``noskill``.
    """
    task_dir = wiki_task_dir(wiki_root, task.id)
    task_dir.mkdir(parents=True, exist_ok=True)

    holder = Path(tempfile.mkdtemp(prefix=f"acumen-wiki-{task.id}-"))
    log: LiveLog | None = None
    try:
        work = holder / "work"
        home = holder / "home"
        provider = provider_for_model(model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        stage_task_evidence(work, runs, version=version)

        # Seed the agent with the current wiki files so it APPENDS rather than clobbers; harvest
        # them back after. New tasks start from empty files.
        obs_path = work / OBSERVATIONS_FILE
        hyp_path = work / HYPOTHESIS_FILE
        existing_obs = task_dir / OBSERVATIONS_FILE
        existing_hyp = task_dir / HYPOTHESIS_FILE
        obs_path.write_text(existing_obs.read_text() if existing_obs.is_file() else "")
        hyp_path.write_text(existing_hyp.read_text() if existing_hyp.is_file() else "")

        # The skill body under test, so hypotheses can point at what the skill actually said.
        # ``noskill`` has none.
        skill_body: Path | None = None
        if skill_body_src is not None:
            skill_body = work / "skill"
            skill_body.mkdir(parents=True, exist_ok=True)
            for src in content_files(skill_body_src):
                dest = skill_body / src.relative_to(skill_body_src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)

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

        prompt = wiki_prompt(
            package=target.pkg_name,
            version=version,
            src=source_copy,
            python=target.python,
            task_id=task.id,
            evidence_dir=work / EVIDENCE_DIRNAME,
            observations_path=obs_path,
            hypothesis_path=hyp_path,
            skill_dir=skill_body,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=model,
            max_turns=max_turns,
            max_usd=max_usd,
            price_usd=pricer(model, table),
            # The filtered source + venv, for grounding hypotheses. Never the raw checkout.
            read_dirs=(source_copy, target.venv_dir),
            write_dirs=(work,),
            # A skill the target itself ships must not colour the notes.
            discover_skills=False,
            claude_hooks={"PreToolUse": [make_skill_guard(target.src_dir)]} if provider == "claude" else None,
            # Wiki agents read staged copies; deny the live runs tree and the raw checkout.
            deny_paths=(runs_root.resolve(), target.src_dir.resolve()),
        )

        if log_dir is not None:
            log = LiveLog.open(log_dir, f"wiki-{task.id}", stream=stream)

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(prompt, options=options, on_event=log.append if log is not None else None)
        except Exception as err:  # noqa: BLE001 - a failed wiki update is reported, re-raised below
            agent_error = err
        finally:
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            raise WikiError(
                f"the wiki agent for {task.id} failed: {type(agent_error).__name__}: {agent_error}"
            ) from agent_error
        if result is None:
            raise WikiError(f"the wiki agent for {task.id} produced no result message")
        if result.is_error:
            raise WikiError(f"the wiki agent for {task.id} errored: {result.subtype} {result.errors or ''}".strip())

        new_obs = obs_path.read_text() if obs_path.is_file() else ""
        new_hyp = hyp_path.read_text() if hyp_path.is_file() else ""
        if not new_obs.strip():
            raise WikiError(f"the wiki agent for {task.id} left {OBSERVATIONS_FILE} empty — nothing to record")
        existing_obs.write_text(new_obs if new_obs.endswith("\n") else new_obs + "\n")
        existing_hyp.write_text(new_hyp if new_hyp.endswith("\n") or not new_hyp else new_hyp + "\n")
        mark_recorded(task_dir, arm)

        warnings = tuple(
            _overlong_entries(new_obs, version, kind="observations")
            + _overlong_entries(new_hyp, version, kind="hypothesis")
        )
        return TaskWikiResult(
            task_id=task.id,
            version=version,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(result.usage, model=model, provider=result.provider, prices=table),
            ).cost_usd,
            turns=result.num_turns,
            warnings=warnings,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        if log is not None:
            log.close()
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


async def update_wiki(
    *,
    cfg: Config,
    target: Target,
    runs_root: Path,
    wiki_root: Path,
    skills_root: Path,
    tasks: Sequence[Task],
    version: str,
    prices: PriceTable | None = None,
    auth_mode: AuthMode = "session",
    model: str | None = None,
    max_concurrency: int = 4,
    max_turns: int | None = None,
    max_usd: float | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    on_plan: Callable[[int], None] | None = None,
    on_task_done: Callable[[TaskWikiResult], None] | None = None,
) -> list[TaskWikiResult]:
    """Append one arm's train-split notes to the wiki, one agent per task, in parallel.

    Idempotent: a task whose ``.arms`` marker already lists this arm is skipped, so a re-run
    after a crash never double-appends.

    Parameters
    ----------
    cfg
        The pass config; supplies ``skill_name``, ``meta_model`` and ``env_passthrough``.
    target
        The prepared target, for the interpreter and (filtered) source used to ground hypotheses.
    runs_root
        The ``runs/`` root, read for train evidence (never the ``valid`` subtree).
    wiki_root
        The ``wiki/`` root to append into.
    skills_root
        The ``skills/`` root, so a skill arm's body can be shown to the agent.
    tasks
        The tasks whose runs are summarised.
    version
        The version label recorded in each entry: ``"noskill"`` or ``"v1"``/``"v2"``…
    max_concurrency
        Ceiling on simultaneous wiki agents.

    Returns
    -------
    One :class:`TaskWikiResult` per task actually updated (skipped tasks are omitted).
    """
    from acumen.skills import skill_dir as _skill_dir

    noskill = arm_name(None)
    arm = noskill if version == noskill else arm_name(version)
    selected_model = model or cfg.meta_model
    table = prices if prices is not None else PriceTable(overrides=cfg.prices)
    # The arm's skill content directory, shown to the agent so hypotheses can cite it. ``noskill``
    # has none. Resolved once here; the leaf agent only copies from it.
    skill_body_src = None if arm == noskill else _skill_dir(skills_root, version)

    # Announce how many tasks will actually run (recorded ones are skipped in ``one``), so a
    # progress bar has an accurate denominator without duplicating the skip predicate.
    if on_plan is not None:
        on_plan(sum(1 for task in tasks if arm not in recorded_arms(wiki_task_dir(wiki_root, task.id))))

    holder = Path(tempfile.mkdtemp(prefix="acumen-wiki-source-"))
    results: list[TaskWikiResult] = []
    try:
        source_copy = build_filtered_source(target.src_dir, holder / "source")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def one(task: Task) -> TaskWikiResult | None:
            task_dir = wiki_task_dir(wiki_root, task.id)
            if arm in recorded_arms(task_dir):
                return None
            runs = collect_arm_runs(runs_root, arm, tasks, split="train", task_id=task.id)
            if not runs:
                raise WikiError(
                    f"no train-split runs for arm {arm!r} on task {task.id!r} under "
                    f"{runs_root / arm / 'train'} — bench that arm on train first"
                )
            async with semaphore:
                res = await _update_one_task(
                    task=task,
                    runs=runs,
                    version=version,
                    arm=arm,
                    cfg=cfg,
                    target=target,
                    source_copy=source_copy,
                    runs_root=runs_root,
                    wiki_root=wiki_root,
                    skill_body_src=skill_body_src,
                    auth_mode=auth_mode,
                    table=table,
                    model=selected_model,
                    max_turns=max_turns,
                    max_usd=max_usd,
                    log_dir=log_dir,
                    stream=stream,
                )
            if on_task_done is not None:
                on_task_done(res)
            return res

        gathered = await asyncio.gather(*(one(task) for task in tasks))
        results = [r for r in gathered if r is not None]
        return results
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
