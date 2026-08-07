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
from acumen.prices import PriceTable
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


class BenchmarkInvalidError(RuntimeError):
    """Raised when a harness failure, not a model, decided the outcome of a pass.

    Two conditions qualify, both infrastructure rather than evidence: the provider
    credential ran out of usage or credit, and the sandbox refused a host the target needed.
    """

    def __init__(self, outcomes: RunOutcome | Sequence[RunOutcome]):
        values = (outcomes,) if isinstance(outcomes, RunOutcome) else tuple(outcomes)
        if not values:
            raise ValueError("BenchmarkInvalidError needs at least one invalid outcome")
        self.outcome = values[0]
        self.outcomes = values
        details: list[str] = []
        for outcome in values:
            payload = outcome.payload
            cell = (
                f"{outcome.key.arm}/{outcome.key.split}/{outcome.key.model}/{outcome.key.task_id}/rep_{outcome.key.rep}"
            )
            if outcome.reason == "sandbox_blocked":
                detail = payload.get("error") or "the sandbox proxy refused a host"
                details.append(
                    f"the agent sandbox refused an outbound host during {cell}. Every remaining "
                    "cell would be refused the same host, so the pass was cancelled. Name the "
                    "hosts the target needs in config 'allowed_domains', or leave it empty to "
                    f"lift the restriction entirely. Sandbox error: {detail}"
                )
                continue
            provider = "Claude" if payload.get("agent") == "claude" else "Codex"
            mode = payload.get("auth_mode", "selected")
            detail = payload.get("error") or "the provider reported exhausted usage or credit"
            details.append(
                f"{provider} {mode} authentication ran out of usage or credit during {cell}. "
                f"Remaining {provider} cells were cancelled; other providers continued. "
                f"Provider error: {detail}"
            )
        super().__init__("benchmark invalid: " + " | ".join(details))


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
    prices: PriceTable | None = None,
    skill: Skill | None = None,
    skill_name: str | None = None,
    sandbox_base: Path | None = None,
    keep_sandbox: bool = False,
    stderr: Callable[[str], None] | None = None,
    on_start: Callable[[PlannedRun], None] | None = None,
    on_done: Callable[[RunOutcome], None] | None = None,
    env_passthrough: Sequence[str] | None = None,
    allowed_domains: Sequence[str] = (),
) -> list[RunOutcome]:
    """Run planned runs concurrently, bounded by ``max_concurrency``.

    Ordinary failing runs do not take down the pass. Two harness failures do, at different
    scopes, both preserving the diagnostic result before raising
    :class:`BenchmarkInvalidError`.

    Provider quota/credit exhaustion is provider-scoped: it cancels only that provider's
    remaining work and lets the other finish its queued cells, since the empty credential is
    the one thing they do not share.

    A host the sandbox refuses is pass-scoped. Every remaining cell would be refused the same
    host whatever model it runs, so continuing only fills ``runs/`` with results that measure
    the allowlist rather than the models.

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
        Which credential every run authenticates with. Under ``"session"``, the recorded
        Claude's SDK value is API-equivalent rather than necessarily metered spend;
        Codex uses token inference against ``prices``.
    prices
        The rates every run in this pass is priced by, and which each records alongside its
        cost. One table for the whole matrix, resolved once before any spend, so a pass is
        never priced by two different sets of numbers. Defaults to the built-in table.
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
    exhausted: dict[AgentProvider, RunOutcome] = {}
    blocked: list[RunOutcome] = []
    tasks: list[asyncio.Task[RunOutcome | None]] = []
    task_providers: dict[asyncio.Task[RunOutcome | None], AgentProvider] = {}

    async def one(item: PlannedRun) -> RunOutcome | None:
        provider = provider_for_model(item.model)
        async with semaphore:
            # A sibling cell may have exhausted this provider while this one waited for a
            # concurrency slot. Do not submit fresh work to a credential known to be empty.
            if provider in exhausted:
                return None
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
                auth_mode=(auth_modes or {}).get(provider, auth_mode),
                skill=skill,
                skill_name=skill_name,
                sandbox_base=sandbox_base,
                keep_sandbox=keep_sandbox,
                stderr=stderr,
                env_passthrough=env_passthrough,
                allowed_domains=allowed_domains,
                prices=prices,
            )
            if on_done is not None:
                on_done(outcome)
            if outcome.reason == "sandbox_blocked":
                # Not provider-scoped: the sandbox refuses the same host for every cell, so
                # letting the pass continue only fills runs/ with invalid results.
                blocked.append(outcome)
                for peer in tasks:
                    if peer is not asyncio.current_task() and not peer.done():
                        peer.cancel()
                return outcome
            if outcome.reason == "provider_exhausted" and provider not in exhausted:
                exhausted[provider] = outcome
                current = asyncio.current_task()
                # Stop both queued and in-flight siblings for this provider. Tasks for the
                # other provider retain their places in the shared concurrency pool.
                for peer in tasks:
                    if peer is not current and task_providers.get(peer) == provider and not peer.done():
                        peer.cancel()
            return outcome

    for item in planned:
        task = asyncio.create_task(one(item))
        tasks.append(task)
        task_providers[task] = provider_for_model(item.model)
    try:
        remaining = set(tasks)
        while remaining:
            done, remaining = await asyncio.wait(remaining, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    continue
                outcome = task.result()  # propagate unexpected per-cell harness exceptions
                if outcome is not None:
                    outcomes.append(outcome)
    except BaseException:
        # External cancellation or an unexpected harness exception still stops everything;
        # provider exhaustion itself is handled above at provider scope.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if blocked:
        raise BenchmarkInvalidError(tuple(blocked))
    if exhausted:
        raise BenchmarkInvalidError(tuple(exhausted.values()))
    return outcomes


def summarize(outcomes: Sequence[RunOutcome]) -> dict[str, int]:
    """Count outcomes by reason, for a one-line pass summary."""
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
    return counts
