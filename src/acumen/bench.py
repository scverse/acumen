"""Benchmark orchestration: build the matrix, run it concurrently, resume what's done."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen.agents import AgentProvider, provider_for_model
from acumen.config import Config
from acumen.env import AuthMode, Target
from acumen.paths import SPLITS, RunKey, Split, arm_name, is_complete, run_dir
from acumen.prices import Rates
from acumen.runner import RunOutcome, run_once
from acumen.skills import Skill
from acumen.tasks import Task


@dataclass(frozen=True)
class PlannedRun:
    """One cell of the benchmark matrix, with its caps already resolved."""

    key: RunKey
    task: Task
    model: str
    max_turns: int
    max_usd: float


def models_for(cfg: Config, task: Task) -> list[str]:
    """Return the models a task runs on — its own override, else the config's list."""
    return [task.model] if task.model else list(cfg.models)


def build_matrix(
    cfg: Config,
    tasks: Sequence[Task],
    *,
    skill: str | None = None,
    splits: Iterable[Split] = SPLITS,
    task_ids: Sequence[str] | None = None,
) -> list[PlannedRun]:
    """Expand config and tasks into the full list of runs for one arm.

    A pass is models x tasks x replicates x splits. Both splits always run; only
    train results are ever shown to the improver, and that is enforced downstream.

    Parameters
    ----------
    cfg
        The pass config.
    tasks
        The tasks to run.
    skill
        Skill version for this arm, or ``None`` for the baseline.
    splits
        Which splits to run. Defaults to both.
    task_ids
        Restrict to these task ids; ``None`` runs all of them.

    Returns
    -------
    The planned runs, in a stable order.
    """
    splits = list(splits)
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    wanted = set(task_ids) if task_ids else None
    if wanted:
        known = {t.id for t in tasks}
        missing = wanted - known
        if missing:
            raise ValueError(f"unknown task ids: {sorted(missing)} (known: {sorted(known)})")

    arm = arm_name(skill)
    planned: list[PlannedRun] = []
    for task in tasks:
        if wanted and task.id not in wanted:
            continue
        for split in splits:
            for model in models_for(cfg, task):
                for rep in range(1, cfg.n_replicates + 1):
                    planned.append(
                        PlannedRun(
                            key=RunKey(arm=arm, split=split, model=model, task_id=task.id, rep=rep),
                            task=task,
                            model=model,
                            max_turns=task.max_turns or cfg.max_turns,
                            max_usd=task.max_usd or cfg.max_usd,
                        )
                    )
    return planned


def pending(planned: Sequence[PlannedRun], runs_root: Path, *, resume: bool = True) -> list[PlannedRun]:
    """Drop runs that already have a complete ``result.json``."""
    if not resume:
        return list(planned)
    return [p for p in planned if not is_complete(run_dir(runs_root, p.key))]


async def run_matrix(
    planned: Sequence[PlannedRun],
    *,
    target: Target,
    runs_root: Path,
    max_concurrency: int,
    auth_mode: AuthMode = "api",
    auth_modes: Mapping[AgentProvider, AuthMode] | None = None,
    price_overrides: dict[str, Rates] | None = None,
    skill: Skill | None = None,
    skill_name: str | None = None,
    sandbox_base: Path | None = None,
    keep_sandbox: bool = False,
    stderr: Callable[[str], None] | None = None,
    on_start: Callable[[PlannedRun], None] | None = None,
    on_done: Callable[[RunOutcome], None] | None = None,
    env_passthrough: Sequence[str] | None = None,
) -> list[RunOutcome]:
    """Run planned runs concurrently, bounded by ``max_concurrency``.

    One failing run never takes down the pass — :func:`acumen.runner.run_once` records
    agent errors as failed runs rather than raising.

    Parameters
    ----------
    planned
        The runs to execute; already filtered for resume by :func:`pending`.
    target
        The prepared target.
    runs_root
        The ``runs/`` root.
    max_concurrency
        Ceiling on simultaneous agents.
    auth_mode
        Which credential every run authenticates with; benchmark passes always use ``"api"``
        so the recorded ``cost_usd`` reflects real metered spend.
    skill
        The skill every run in this matrix installs, or ``None`` for the baseline. One
        matrix is one arm, so this is a property of the pass rather than of a run.
    skill_name
        Name of the skill under test (``config.skill_name``), used to recognise it in each
        transcript. Pass it in the baseline arm too — nothing is installed there, but a
        baseline run that loads it anyway is the contamination worth catching.
    sandbox_base
        Parent directory for sandboxes.
    keep_sandbox
        Leave sandboxes on disk for inspection.
    stderr
        Callback for each run's CLI subprocess stderr. Pass one shared
        :class:`~acumen.runner.StderrFilter` so the per-spawn startup warnings are
        printed once for the whole pass instead of once per run.
    on_start
        Optional callable invoked with each :class:`PlannedRun` as it is admitted through
        the concurrency gate and begins, for progress output.
    on_done
        Optional callable invoked with each :class:`~acumen.runner.RunOutcome` as it
        lands, for progress output.
    env_passthrough
        Extra environment variable names each run carries into its sandbox on top of the
        built-in allowlist (the operator's ``config.env_passthrough``).

    Returns
    -------
    The outcomes, in completion order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    outcomes: list[RunOutcome] = []

    async def one(item: PlannedRun) -> RunOutcome:
        async with semaphore:
            if on_start is not None:
                on_start(item)
            outcome = await run_once(
                key=item.key,
                task=item.task,
                target=target,
                run_dir=run_dir(runs_root, item.key),
                model=item.model,
                max_turns=item.max_turns,
                max_usd=item.max_usd,
                auth_mode=(auth_modes or {}).get(provider_for_model(item.model), auth_mode),
                skill=skill,
                skill_name=skill_name,
                sandbox_base=sandbox_base,
                keep_sandbox=keep_sandbox,
                stderr=stderr,
                env_passthrough=env_passthrough,
                price_overrides=price_overrides,
            )
            if on_done is not None:
                on_done(outcome)
            return outcome

    for coro in asyncio.as_completed([one(item) for item in planned]):
        outcomes.append(await coro)
    return outcomes


def summarize(outcomes: Sequence[RunOutcome]) -> dict[str, int]:
    """Count outcomes by reason, for a one-line pass summary."""
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
    return counts
