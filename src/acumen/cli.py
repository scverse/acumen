"""Command-line entry point — a thin shell over the importable API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from acumen.agents import AgentError, AgentProvider, check_agent_cli, provider_for_model
from acumen.bench import BenchmarkInvalidError, PlannedRun, build_matrix, pending, run_matrix, summarize
from acumen.check import (
    DEFAULT_JOBS,
    DEFAULT_TIMEOUT,
    SCRIPTS_DIRNAME,
    CheckError,
    CheckResult,
    CheckSummary,
    check_tasks,
    import_probe,
    orphan_scripts,
    select_tasks,
    summarize_checks,
)
from acumen.config import Config, ConfigError, load_config
from acumen.draft import DraftError, draft_skill
from acumen.env import DEFAULT_CACHE_ROOT, AuthMode, EnvError, prepare_target, resolve_auth_mode
from acumen.grade import INVALID_REASONS
from acumen.improve import ImproveError, improve_skill
from acumen.logs import LiveLog
from acumen.paths import SPLITS, arm_name
from acumen.pricefeed import (
    PRICE_SOURCES,
    PRICE_TIER,
    PriceFeedError,
    diff_rates,
    fetch_table,
    refresh,
    to_yaml_block,
    try_fetch_table,
)
from acumen.prices import PriceTable, Rates
from acumen.report import ReportError, build_report
from acumen.review import ReviewError, ReviewResult, ReviewStatus, ReviewVerdict, review_tasks
from acumen.runner import RunOutcome, StderrFilter
from acumen.scaffold import InitError, is_scaffold_tasks, scaffold
from acumen.ship import ShipError, ship_skill
from acumen.skills import Skill, SkillError, available_versions, latest_version, load_skill
from acumen.taskgen import TaskGenError, generate_tasks
from acumen.tasks import Task, TaskError, load_tasks


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    arm = parser.add_mutually_exclusive_group()
    arm.add_argument("--no-skill", action="store_true", help="run the baseline arm alone")
    arm.add_argument("--skill", metavar="VERSION", help="run one skill version alone, e.g. v1")
    parser.add_argument("--split", choices=SPLITS, action="append", help="restrict to a split (repeatable)")
    parser.add_argument("--task", metavar="ID", action="append", help="restrict to a task id (repeatable)")
    parser.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    parser.add_argument("--replicates", type=int, help="override config n_replicates")
    parser.add_argument("--no-resume", action="store_true", help="re-run runs that already completed")
    parser.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    parser.add_argument("--keep-sandboxes", action="store_true", help="leave run sandboxes on disk")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    parser.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the matrix and exit without running agents, over the same arms the real pass would cover",
    )
    _add_auth_arg(parser)


def _add_log_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared live-log flags to a meta-agent subcommand."""
    parser.add_argument("--stream", action="store_true", help="mirror the agent's conversation to the terminal live")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), dest="log_dir", help="directory for the run log (default: logs/)"
    )


def _add_feedback_arg(parser: argparse.ArgumentParser, *, extra: str = "") -> None:
    """Add the optional ``--feedback`` flag to an authoring subcommand.

    The text is injected into the agent's prompt as a subordinated guidance block; it never
    overrides the isolation or anti-overfit rules. ``extra`` appends a per-command note to the
    help text.
    """
    help_text = "extra guidance for the agent, injected into its prompt as subordinate guidance"
    parser.add_argument("--feedback", help=(help_text + extra) or None)


def _add_auth_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--auth`` flag to a command that spawns agents.

    Every agentic command defaults to the provider subscription ("session") when a login is
    present and falls back to the API key otherwise — ``bench`` included, since it prices runs
    from their token counts rather than from a billed figure only the API reports.
    """
    parser.add_argument(
        "--auth",
        choices=("auto", "session", "api"),
        default="auto",
        help="which credential to bill: 'session' (Claude/Codex subscription), 'api' (provider API), "
        "or 'auto' (default: session if you're logged in, else the API)",
    )


def _print_auth(mode: AuthMode, provider: AgentProvider = "claude") -> None:
    """Report which credential the run will bill, so the choice is never silent."""
    product = "Claude" if provider == "claude" else "Codex"
    label = f"{product} subscription (session)" if mode == "session" else f"{product} API key"
    print(f"auth: {label}", flush=True)


def _warn_codex_accounting(provider: AgentProvider) -> None:
    """Say what a Codex cap can and cannot do before the spend, not after.

    ``codex exec`` has no cap of its own, so acumen enforces both against the event stream.
    That works exactly as advertised for turns, which stream. Usage does not: Codex reports it
    once, when the turn ends, so ``max_usd`` can only be recognized after the money is spent.
    It still records the run as a budget failure — the same outcome Claude would give it — but
    the only cap that actually *bounds* a Codex run is ``max_turns``.
    """
    if provider == "codex":
        print(
            "note: Codex reports usage only when a turn ends, so max_usd marks an over-budget "
            "run as a failure but cannot stop the spend — bound Codex runs with max_turns",
            file=sys.stderr,
        )


def _warn_unpriced(models: set[str], prices: PriceTable) -> None:
    """Name models with no rates before the spend, not after.

    An unpriced model still records its tokens; only ``cost_usd`` is left unset. Saying so
    up front is what stops a missing rate from reading as a free model in the report.
    """
    unpriced = sorted(model for model in models if prices.lookup(model) is None)
    if unpriced:
        print(
            f"note: no token rates for {', '.join(unpriced)}. These runs record tokens but no "
            "cost; add a 'prices:' entry in config.yaml to price them.",
            file=sys.stderr,
        )


def _bench_prices(cfg: Config) -> PriceTable:
    """Resolve the rates this pass will be priced by, before it spends anything.

    Raises :class:`PriceFeedError` upward: cost is a headline metric of the report and is
    frozen into every result, so a pass that cannot establish rates should not run.
    """
    for name, url in PRICE_SOURCES.items():
        print(f"prices: fetching {name} ({url})", flush=True)
    table = fetch_table(cfg.prices)
    print(f"prices: {len(table.fetched)} model(s) resolved as of {table.fetched_as_of} ({PRICE_TIER})")
    return table


def _agent_prices(cfg: Config, *, model: str | None = None) -> PriceTable:
    """Rates for one interactive agent command, degrading to unpriced if the pages are down.

    Unlike a benchmark, these commands report cost as progress rather than storing it as
    evidence, so a pricing outage should not stop the work. It is still said out loud, and
    the Codex consequence is named: ``max_usd`` is enforced from these rates, so an
    unpriced model has no budget cap at all.
    """
    table, failure = try_fetch_table(cfg.prices)
    if failure is None:
        return table
    print(f"warning: could not fetch pricing ({failure}); this run reports no cost", file=sys.stderr)
    if model is not None and provider_for_model(model) == "codex" and table.lookup(model) is None:
        print(
            f"warning: max_usd cannot be enforced for {model} without rates. Bound this run "
            "with max_turns, or pin rates in config.yaml under 'prices:'.",
            file=sys.stderr,
        )
    return table


def _print_log_result(log: LiveLog) -> None:
    """Print where the rendered HTML log landed, once a run has finalized."""
    if log.html_rendered:
        print(f"log → {log.html_path}")
    else:
        print("note: HTML log not rendered — the jsonl log is complete", file=sys.stderr)


def _fmt_secs(seconds: float) -> str:
    """Compact wall-clock duration, e.g. ``9s`` / ``2m41s`` / ``1h04m``."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _key_label(key) -> str:
    return f"{key.arm}/{key.split}/{key.model}/{key.task_id}/rep_{key.rep}"


class _Progress:
    """Progress reporter for a concurrent bench pass.

    Prints a line when each run starts and finishes, each stamped with the wall-clock
    elapsed since the pass began, the number in flight, and a running pass tally — the
    context a long, interleaved pass needs to be readable as it scrolls by.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.started = 0
        self.done = 0
        self.passed = 0
        self.running = 0
        self._t0 = time.monotonic()

    @property
    def elapsed(self) -> float:
        """Seconds since the pass began."""
        return time.monotonic() - self._t0

    def _stamp(self) -> str:
        return f"+{_fmt_secs(self.elapsed):>6}"

    def on_start(self, item) -> None:
        self.started += 1
        self.running += 1
        print(
            f"[{self._stamp()}] ▶ start {_key_label(item.key)}"
            f"  (running {self.running}, {self.started}/{self.total} started)",
            flush=True,
        )

    def on_done(self, outcome: RunOutcome) -> None:
        self.done += 1
        self.running -= 1
        if outcome.success:
            self.passed += 1
        p = outcome.payload
        invalid = outcome.reason in INVALID_REASONS
        mark = "⚠ INVALID" if invalid else ("✓ pass" if outcome.success else "✗ FAIL")
        toks = int(p.get("input_tokens", 0)) + int(p.get("output_tokens", 0))
        dur = _fmt_secs(float(p.get("duration_s", 0.0)))
        cost_available = p.get("cost_available", True) and p.get("cost_usd") is not None
        cost_label = f"${float(p['cost_usd']):.2f}" if cost_available else "cost n/a"
        stats = f"{_fmt_tokens(toks)}tok {cost_label} {dur}"
        print(
            f"[{self._stamp()}] {mark} {_key_label(outcome.key)}"
            f"  ({outcome.reason})  {stats}"
            f"  [{self.done}/{self.total} done, {self.passed} passed]",
            flush=True,
        )
        if outcome.reason == "provider_exhausted":
            print(f"error: provider usage/credit exhausted: {p.get('error') or 'no provider detail'}", file=sys.stderr)
        elif outcome.reason == "sandbox_blocked":
            print(
                "error: the agent sandbox refused an outbound host; egress is meant to be "
                f"unrestricted, so this is a harness bug: {p.get('error') or 'no sandbox detail'}",
                file=sys.stderr,
            )


def _fmt_tokens(value: int) -> str:
    """Compact token count, e.g. ``118k`` / ``1.2M``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return str(value)


def _fmt_cost(value: float | None) -> str:
    """Format a known cost without presenting unavailable pricing as free."""
    return f"${value:.2f}" if value is not None else "cost n/a"


@dataclass(frozen=True)
class _Arm:
    """One arm of a pass: its version, its loaded skill, and its matrix."""

    version: str | None  # None => the noskill baseline
    skill: Skill | None
    planned: list[PlannedRun]
    todo: list[PlannedRun]

    @property
    def name(self) -> str:
        return arm_name(self.version)


def _resolve_arms(cfg: Config, tasks: Sequence[Task], args: argparse.Namespace) -> list[_Arm]:
    """Build the arms this invocation covers, loading every skill before anything is spent.

    Naming an arm (``--skill vN`` / ``--no-skill``) covers that one alone. Naming none covers
    the whole project: the baseline plus every version in ``skills/``, which is the comparison
    the report wants anyway. A version that will not load raises here, before target prep, so a
    broken skill costs nothing rather than surfacing part-way through a paid pass.
    """
    if args.skill is not None:
        versions: list[str | None] = [args.skill]
    elif args.no_skill:
        versions = [None]
    else:
        versions = [None, *available_versions(args.skills)]

    arms = []
    for version in versions:
        skill = None if version is None else load_skill(args.skills, version, expect_name=cfg.skill_name)
        planned = build_matrix(cfg, tasks, skill=version, splits=args.split or SPLITS, task_ids=args.task)
        todo = pending(planned, args.runs, resume=not args.no_resume)
        arms.append(_Arm(version=version, skill=skill, planned=planned, todo=todo))
    return arms


def _print_plan(arms: Sequence[_Arm]) -> None:
    """Print what each arm holds, and the total when a pass spans several."""
    width = max(len(f"arm {arm.name}:") for arm in arms) if len(arms) > 1 else 0
    indent = "    " if len(arms) > 1 else ""
    for arm in arms:
        label = f"arm {arm.name}:".ljust(width)
        complete = len(arm.planned) - len(arm.todo)
        print(f"{label} {len(arm.planned)} runs planned, {complete} already complete, {len(arm.todo)} to run")
        if arm.skill is not None:
            print(f"{indent}skill {arm.skill.version}: {arm.skill.name} ({arm.skill.hash[:19]}…)")
    if len(arms) > 1:
        planned = sum(len(arm.planned) for arm in arms)
        todo = sum(len(arm.todo) for arm in arms)
        print(f"total: {planned} runs planned, {todo} to run across {len(arms)} arms")


def _print_runs(todo: Sequence[PlannedRun]) -> None:
    for item in todo:
        k = item.key
        print(f"  {k.arm}/{k.split}/{k.model}/{k.task_id}/rep_{k.rep}")


def _print_run_summary(outcomes: Sequence[RunOutcome], elapsed: float, *, label: str = "") -> None:
    """Print the pass tally: how many passed, how long, what it cost, why runs failed."""
    passed = sum(1 for o in outcomes if o.success)
    counts = summarize(outcomes)
    breakdown = ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))
    priced = [
        float(o.payload["cost_usd"])
        for o in outcomes
        if o.payload.get("cost_available", True) and o.payload.get("cost_usd") is not None
    ]
    total_cost = sum(priced)
    unpriced = len(outcomes) - len(priced)
    cost_summary = f"${total_cost:.2f}" if priced else "cost n/a"
    if unpriced and priced:
        cost_summary += f" + {unpriced} unpriced run(s)"
    elif unpriced:
        cost_summary += f" ({unpriced} unpriced run(s))"
    prefix = f"{label}: " if label else ""
    print(f"\n{prefix}{passed}/{len(outcomes)} passed in {_fmt_secs(elapsed)}  ({cost_summary}, {breakdown})")


def _print_skill_loading(outcomes: Sequence[RunOutcome], arm: _Arm, skill_name: str) -> None:
    """Say whether the skill reached the agent — the comparison means nothing otherwise."""
    loaded = sum(1 for o in outcomes if o.payload.get("skill_loaded"))
    if arm.skill is not None:
        print(f"skill loaded in {loaded}/{len(outcomes)} runs")
        if loaded == 0:
            print(
                f"warning: {arm.name} never loaded the skill — that arm is not measuring the skill",
                file=sys.stderr,
            )
    elif loaded:
        print(f"warning: {skill_name} loaded in {loaded} baseline runs", file=sys.stderr)


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.replicates:
        cfg = replace(cfg, n_replicates=args.replicates)

    arms = _resolve_arms(cfg, tasks, args)
    _print_plan(arms)
    if args.dry_run:
        for arm in arms:
            _print_runs(arm.todo)
        return 0

    todo = [item for arm in arms for item in arm.todo]
    if not todo:
        return 0

    # One resolved mode per provider in the matrix, so a mixed pass bills each side correctly.
    providers = {provider_for_model(item.model) for item in todo}
    auth_modes = {provider: resolve_auth_mode(args.auth, provider=provider) for provider in providers}
    for provider in sorted(providers):
        check_agent_cli(provider)
        _print_auth(auth_modes[provider], provider)
        _warn_codex_accounting(provider)
    if "session" in auth_modes.values():
        print(
            "note: cost_usd for session-billed runs is what they would have cost at API "
            "rates, not metered spend; each run records its auth_mode",
            file=sys.stderr,
        )
    # Before the target is built and before any agent runs: an unreachable pricing page
    # must cost nothing, and a pass must never be priced by a table it cannot date.
    try:
        prices = _bench_prices(cfg)
    except PriceFeedError as err:
        print(f"error: {err}", file=sys.stderr)
        print(
            "Each run's cost is frozen into its result, so a pass that cannot establish "
            "rates would store figures nothing backs. Retry when the pages are reachable, "
            "or pin rates with a 'prices:' block in config.yaml.",
            file=sys.stderr,
        )
        return 2
    _warn_unpriced({item.model for item in todo}, prices)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)

    # Arms run one after another: every run in a matrix shares one skill, and a sequential
    # pass keeps each arm's tally readable while the progress counter spans the whole thing.
    running = [arm for arm in arms if arm.todo]
    print(f"running {len(todo)} runs, up to {cfg.max_concurrency} at a time:", flush=True)
    progress = _Progress(len(todo))
    collected: list[RunOutcome] = []
    try:
        for arm in running:
            if len(running) > 1:
                print(f"\n=== arm {arm.name}: {len(arm.todo)} runs ===", flush=True)
            started = time.monotonic()
            outcomes = asyncio.run(
                run_matrix(
                    arm.todo,
                    target=target,
                    runs_root=args.runs,
                    max_concurrency=cfg.max_concurrency,
                    auth_modes=auth_modes,
                    skill=arm.skill,
                    skill_name=cfg.skill_name,
                    keep_sandbox=args.keep_sandboxes,
                    stderr=StderrFilter(),
                    on_start=progress.on_start,
                    on_done=progress.on_done,
                    env_passthrough=cfg.env_passthrough,
                    prices=prices,
                )
            )
            collected.extend(outcomes)
            _print_run_summary(outcomes, time.monotonic() - started, label=arm.name if len(running) > 1 else "")
            _print_skill_loading(outcomes, arm, cfg.skill_name)
    except BenchmarkInvalidError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        print(
            "Fix or replenish that credential, then rerun the same command; invalid and "
            "cancelled cells remain pending.",
            file=sys.stderr,
        )
        return 2

    if len(running) > 1:
        _print_run_summary(collected, progress.elapsed, label=f"all {len(running)} arms")
    print(f"runs written to {args.runs.resolve()}")
    return 0


def _cmd_draft(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, draft_model=args.model)

    existing = available_versions(args.skills)
    if existing and not args.force:
        print(
            f"skills already exist ({', '.join(existing)}) — drafting would add "
            f"another version. Pass --force to draft anyway, or use `acumen improve` "
            f"to build on {existing[-1]}.",
            file=sys.stderr,
        )
        return 2

    provider = provider_for_model(cfg.draft_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(f"drafting with {cfg.draft_model} (this reads the package source) ...", flush=True)

    log = LiveLog.open(args.log_dir, "draft", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            draft_skill(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.draft_model),
                target=target,
                skills_root=args.skills,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                feedback=args.feedback,
                log=log,
            )
        )
    skill = result.skill
    files = sorted(p.relative_to(skill.directory).as_posix() for p in skill.directory.rglob("*") if p.is_file())
    print(f"\nwrote {skill.directory}")
    print(f"  name:        {skill.name}")
    print(f"  description: {skill.description}")
    print(f"  hash:        {skill.hash}")
    print(f"  files:       {', '.join(files)}")
    print(f"  cost:        {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    _print_log_result(log)
    print(f"\nnext: acumen bench --skill {skill.version}")
    return 0


def _cmd_improve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.model:
        cfg = replace(cfg, improve_model=args.model)

    versions = available_versions(args.skills)
    if not versions:
        print(
            f"no skill versions under {args.skills} — run `acumen draft` first, then bench it",
            file=sys.stderr,
        )
        return 2
    parent = args.from_version or latest_version(args.skills)
    # Immutability guard: the improved version is always the next unused directory,
    # so an existing version is never in the write path. Say the parent plainly up front.
    skill = load_skill(args.skills, parent, expect_name=cfg.skill_name)
    print(f"improving {skill.version} ({skill.name}, {skill.hash[:19]}…) with {cfg.improve_model}")

    provider = provider_for_model(cfg.improve_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    log = LiveLog.open(args.log_dir, "improve", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            improve_skill(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.improve_model),
                target=target,
                skills_root=args.skills,
                runs_root=args.runs,
                tasks=tasks,
                auth_mode=auth_mode,
                parent_version=parent,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                feedback=args.feedback,
                log=log,
            )
        )
    new = result.skill
    files = sorted(p.relative_to(new.directory).as_posix() for p in new.directory.rglob("*") if p.is_file())
    print(f"\nwrote {new.directory}  (parent {result.parent})")
    print(f"  name:        {new.name}")
    print(f"  description: {new.description}")
    print(f"  hash:        {new.hash}")
    print(f"  files:       {', '.join(files)}")
    print(f"  evidence:    {result.n_train_runs} train runs ({result.n_train_failures} failing)")
    print(f"  cost:        {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    if new.hash == skill.hash:
        print(
            "warning: the new version is byte-identical to its parent — the improver changed nothing",
            file=sys.stderr,
        )
    _print_log_result(log)
    print(f"\nnext: acumen bench --skill {new.version} && acumen report")
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, tasks_model=args.model)

    out = args.out
    # `acumen init` writes a placeholder here, and generating over it is the documented next
    # step — so the untouched placeholder is not something to protect. Anything the user has
    # actually edited still needs --force, and that is checked before the costly target prep.
    overwrite = args.force or is_scaffold_tasks(out)
    if out.exists() and not overwrite:
        print(
            f"{out} already exists — pass --force to overwrite it",
            file=sys.stderr,
        )
        return 2
    if out.exists() and not args.force:
        print(f"replacing the untouched placeholder at {out}")

    provider = provider_for_model(cfg.tasks_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(
        f"generating tasks with {cfg.tasks_model} (this reads the source and runs package code) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "tasks", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            generate_tasks(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.tasks_model),
                target=target,
                out_path=out,
                scripts_root=args.scripts,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                force=overwrite,
                feedback=args.feedback,
                log=log,
            )
        )
    print(f"\nwrote {result.out_path.resolve()}")
    print(f"  tasks:   {len(result.tasks)} ({', '.join(t.id for t in result.tasks)})")
    no_script = [task.id for task in result.tasks if not task.needs_script]
    scripts = f"{len(result.scripts)} in {args.scripts}"
    if no_script:
        scripts += f" ({len(no_script)} task(s) need none: {', '.join(no_script)})"
    print(f"  scripts: {scripts}")
    print(f"  cost:    {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    _print_log_result(log)
    if result.missing_scripts:
        gaps = ", ".join(f"{task_id}/{split}" for task_id, split in result.missing_scripts)
        print(
            f"warning: the agent wrote no reproducer for {gaps}. `acumen check` reports these "
            "as missing, so their answers cannot be confirmed.",
            file=sys.stderr,
        )
    for name in result.unexpected_scripts:
        print(
            f"warning: the agent left {name} in its scripts directory, which matches no task "
            "and split, so it was not kept",
            file=sys.stderr,
        )
    print("\nnext: review the tasks, then `acumen check`, `acumen draft` and `acumen bench`")
    return 0


#: How each status prints in the check table. Only ``skipped`` is renamed: "n/a" reads as
#: "nothing to check here", where "skipped" would read as "we did not get to it".
_CHECK_LABELS = {"skipped": "n/a"}

#: Width of the widest status label, so streamed rows line up before all of them are known.
_STATUS_WIDTH = 12


def _clip(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` characters, marking that it was cut."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _check_detail(result: CheckResult, timeout: float) -> str:
    """The rightmost column: what this status means for this split, in one line."""
    if result.status == "ok":
        return _clip(result.answer or "", 60)
    if result.status == "wrong_answer":
        return f"got {_clip(result.answer or '', 28)} / want {_clip(result.expected, 28)}"
    if result.status == "format_error":
        return f"right content, formatting differs: {_clip(result.answer or '', 40)}"
    if result.status == "no_answer":
        return "ran clean but wrote no answer.md"
    if result.status == "error":
        code = f"exit {result.returncode}" if result.returncode is not None else "failed to start"
        return f"{code}: {_clip(result.error_tail or '', 60)}"
    if result.status == "timeout":
        return f"killed after {_fmt_secs(timeout)}"
    if result.status == "missing":
        return f"no reproducer at {result.script}"
    return "needs_script: false"


#: How a review verdict prints. An unreviewed split shows a dash rather than a word: it is the
#: absence of a judgement, not a judgement.
_REVIEW_LABELS: dict[str, str] = {"unreviewed": "-"}

#: Width of the widest review label, so the column lines up before every verdict is in.
_REVIEW_WIDTH = 8


def _check_header(*, task_width: int, review: bool) -> str:
    """The table's header row, with the review column only when there is one."""
    head = f"  {'task':<{task_width}}  {'split':<5}  {'status':<{_STATUS_WIDTH}}  {'time':>6}  "
    if review:
        head += f"{'review':<{_REVIEW_WIDTH}}  "
    return head + "detail"


def _check_row(result: CheckResult, *, task_width: int, timeout: float, review: ReviewStatus | None = None) -> str:
    """Format one result as a table row, optionally carrying its review verdict."""
    status = _CHECK_LABELS.get(result.status, result.status)
    elapsed = _fmt_secs(result.seconds) if result.script is not None and result.status != "missing" else "-"
    row = f"  {result.task_id:<{task_width}}  {result.split:<5}  {status:<{_STATUS_WIDTH}}  {elapsed:>6}  "
    if review is not None:
        row += f"{_REVIEW_LABELS.get(review, review):<{_REVIEW_WIDTH}}  "
    return row + _check_detail(result, timeout)


def _print_flagged(flagged: Sequence[ReviewVerdict], *, task_width: int) -> None:
    """Print what the review flagged: one line naming the contradiction, one naming the fix."""
    noun = "split" if len(flagged) == 1 else "splits"
    print(f"\n{len(flagged)} {noun} the review flagged")
    label_width = max(task_width + 6, 12)
    for verdict in flagged:
        label = f"{verdict.task_id}/{verdict.split}"
        print(f"  {label:<{label_width}}  {verdict.issue or 'flagged with no reason given'}")
        if verdict.fix:
            print(f"  {'':<{label_width}}  fix: {verdict.fix}")


def _print_check_summary(summary: CheckSummary, review: ReviewResult | None = None) -> None:
    """Print the statistics: how much of the task set has reproducible, coherent ground truth."""

    def line(label: str, part: int, whole: int, pct: float | None) -> str:
        share = "" if pct is None else f"  ({pct:g}%)"
        return f"  {label:<22} {part:>3}/{whole:<3}{share}"

    print("\nsummary")
    print(f"  {'splits checked':<22} {summary.n_splits:>3}")
    if summary.n_non_code_splits:
        print(
            f"  {'needing a reproducer':<22} {summary.n_code_splits:>3}      ({summary.n_non_code_splits} marked needs_script: false)"
        )
    print(line("with a reproducer", summary.n_with_script, summary.n_code_splits, summary.pct_with_script))
    print(line("reproduced", summary.n_ok, summary.n_code_splits, summary.pct_reproduced))
    for status in ("wrong_answer", "format_error", "no_answer", "error", "timeout", "missing"):
        count = summary.by_status.get(status, 0)
        if count:
            print(f"    {status:<20} {count:>3}")
    print(
        line(
            "tasks fully reproduced",
            summary.n_tasks_reproduced,
            summary.n_code_tasks,
            summary.pct_tasks_reproduced,
        )
        + "  (both splits)"
    )
    if review is not None:
        judged = sum(1 for verdict in review.verdicts.values() if verdict.status != "unreviewed")
        n_flagged = len(review.flagged)
        pct = None if summary.n_splits == 0 else round(100.0 * judged / summary.n_splits, 1)
        print(line("reviewed", judged, summary.n_splits, pct))
        if n_flagged:
            print(f"    {'mismatch':<20} {n_flagged:>3}")


def _cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, check_model=args.model)
    tasks = select_tasks(load_tasks(args.tasks), args.task)
    splits = args.split or list(SPLITS)
    review_on = not args.no_review

    # Everything the review needs is resolved before the target is built and before a single
    # script runs: a missing agent CLI or an unusable credential must not surface after half an
    # hour of analyses, when it was knowable up front.
    auth_mode: AuthMode = "api"
    prices: PriceTable | None = None
    if review_on:
        provider = provider_for_model(cfg.check_model)
        check_agent_cli(provider)
        auth_mode = resolve_auth_mode(args.auth, provider=provider)
        _print_auth(auth_mode, provider)
        _warn_codex_accounting(provider)
        prices = _agent_prices(cfg, model=cfg.check_model)

    orphans = orphan_scripts(tasks, args.scripts) if not args.task else []
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    ok, detail = import_probe(target.python, target.pkg_name)
    if not ok:
        print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", file=sys.stderr)
        print(f"error: {target.pkg_name} does not import in the target venv:\n{detail}", file=sys.stderr)
        print(
            "Every task would fail the same way, so nothing was run. Fix the target's "
            "dependency selection in config.yaml (extras, dependency_groups, pip_packages) "
            "and rerun, or rebuild the venv with --refresh-target.",
            file=sys.stderr,
        )
        return 2
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (imports as {detail})")
    print(f"reproducers:  {args.scripts}")

    n_rows = len(tasks) * len(splits)
    task_width = max((len(task.id) for task in tasks), default=4)
    print(f"\nchecking {n_rows} task splits, up to {args.jobs} at a time:")
    print(_check_header(task_width=task_width, review=False))

    # Rows print as each reproducer finishes, since a real analysis takes minutes and a silent
    # terminal for half an hour is not progress.
    def on_done(result: CheckResult) -> None:
        print(_check_row(result, task_width=task_width, timeout=args.timeout), flush=True)

    work_root = Path(tempfile.mkdtemp(prefix="acumen-check-")) if args.keep else None
    results = check_tasks(
        tasks,
        scripts_root=args.scripts,
        python=target.python,
        splits=splits,
        timeout=args.timeout,
        jobs=args.jobs,
        work_root=work_root,
        on_done=on_done,
    )
    if work_root is not None:
        print(f"\nworking directories kept at {work_root}")

    summary = summarize_checks(results, tasks)
    review: ReviewResult | None = None
    review_failed: str | None = None
    if review_on:
        # The second phase. Reproducing the answer says the code and the answer agree; it says
        # nothing about whether the PROMPT asks for what they produce, which is the other way a
        # task silently costs a whole pass.
        print(f"\nreviewing {len(results)} task splits with {cfg.check_model} ...", flush=True)
        log = LiveLog.open(args.log_dir, "check", stream=args.stream)
        print(f"log → {log.jsonl_path}", flush=True)
        try:
            with log:
                review = asyncio.run(
                    review_tasks(
                        cfg=cfg,
                        target=target,
                        tasks=tasks,
                        results=results,
                        auth_mode=auth_mode,
                        prices=prices,
                        max_turns=args.max_turns,
                        max_usd=args.max_usd,
                        log=log,
                    )
                )
        except ReviewError as err:
            # The deterministic results still stand on their own, so they are reported below
            # rather than thrown away. The exit code says the check did not complete.
            review_failed = str(err)
        else:
            print(f"review: {_fmt_cost(review.cost_usd)} over {review.turns} turns")
            _print_log_result(log)

    # The full table in task order, now that every verdict is in. It replaces restating the
    # failures: with several reproducers running at once the streamed rows are in completion
    # order, and the review column did not exist yet when they were printed.
    if review is not None:
        print()
        print(_check_header(task_width=task_width, review=True))
        for result in results:
            verdict = review.status_for(result.task_id, result.split)
            print(_check_row(result, task_width=task_width, timeout=args.timeout, review=verdict))
        if review.flagged:
            _print_flagged(review.flagged, task_width=task_width)
    else:
        failed = [result for result in results if not result.ok]
        # Restated in task order only when something actually reproduced: when nothing passed the
        # streamed table already *is* that list, and repeating it says nothing. Rows for tasks
        # that need no reproducer do not count as passes here.
        if failed and summary.n_ok:
            print(f"\n{len(failed)} of {summary.n_code_splits} splits did not reproduce:")
            for result in failed:
                print(_check_row(result, task_width=task_width, timeout=args.timeout))
    _print_check_summary(summary, review)
    for warning in review.warnings if review is not None else ():
        sys.stdout.flush()
        print(f"warning: {warning}", file=sys.stderr)
    if orphans:
        # Flushed first: piped stdout is block-buffered, and a warning that lands above the
        # table it refers to reads as being about something else.
        sys.stdout.flush()
        for path in orphans:
            print(f"warning: {path} matches no task and split, so nothing runs it", file=sys.stderr)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "target": {
                        "repo": cfg.repo,
                        "ref": cfg.ref,
                        "commit": target.commit,
                        "pkg_version": target.fingerprint,
                    },
                    "scripts_root": str(args.scripts),
                    "results": [
                        {
                            **result.to_dict(),
                            "review": (
                                None if review is None else review.verdicts[(result.task_id, result.split)].to_dict()
                            ),
                        }
                        for result in results
                    ],
                    "summary": summary.to_dict(),
                    "review": None
                    if review is None
                    else {
                        "model": cfg.check_model,
                        "cost_usd": review.cost_usd,
                        "turns": review.turns,
                        "flagged": len(review.flagged),
                        "warnings": list(review.warnings),
                    },
                    "orphans": [str(path) for path in orphans],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.json}")

    if review_failed is not None:
        sys.stdout.flush()
        print(f"\nerror: {review_failed}", file=sys.stderr)
        print(
            "The deterministic results above still stand, but the coherence review did not run, "
            "so this is not a clean check.",
            file=sys.stderr,
        )
        return 2
    if summary.ok and review is not None and review.flagged:
        sys.stdout.flush()
        print(
            "\nevery answer reproduced, but the review flagged the splits above: a prompt that "
            "asks for something other than what its script and answer produce fails every agent "
            "that reads it correctly",
            file=sys.stderr,
        )
        return 1
    if summary.ok:
        print("\nevery task's ground truth reproduced: the answers are safe to benchmark against")
        return 0
    sys.stdout.flush()
    if summary.n_with_script == 0:
        # A hand-written tasks.yaml has no reproducers yet, and "16 missing" is not advice.
        print(
            f"\nno task has a reproducer yet. Write one per split as {args.scripts}/<id>-<split>.py, "
            "each redoing the analysis and writing its answer to answer.md in the working "
            "directory it is started in, or run `acumen tasks` to regenerate the task set with "
            "its reproducers. Mark a task that needs no code with `needs_script: false`.",
            file=sys.stderr,
        )
        return 1
    print(
        "\nfix the reproducers and answers above before benchmarking: a task whose ground "
        "truth cannot be reproduced costs a full pass and measures nothing",
        file=sys.stderr,
    )
    return 1


def _cmd_ship(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, ship_model=args.model)

    # Validate the version exists before the (costly) target prep.
    load_skill(args.skills, args.version, expect_name=cfg.skill_name)

    where = (
        "a local path — the change is written to the working tree"
        if cfg.is_local
        else ("a GitHub URL — the change is delivered as a pull request")
    )
    print(f"shipping {args.version} of {cfg.skill_name} into {cfg.repo} ({where})")
    provider = provider_for_model(cfg.ship_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(
        f"running the ship agent with {cfg.ship_model} (real env: it builds, installs, and "
        f"{'opens a PR' if not cfg.is_local else 'edits the working tree'}) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "ship", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            ship_skill(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.ship_model),
                target=target,
                skills_root=args.skills,
                version=args.version,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                force=args.force,
                log=log,
            )
        )
    print(f"\nshipped {result.skill.version} of {result.skill.name}")
    print(f"  mode:  {'pull request' if result.mode == 'github' else 'working tree (local)'}")
    print(f"  cost:  {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    _print_log_result(log)
    if result.summary:
        print("\nagent summary:")
        print(result.summary)
    return 0


def _parse_palette(values: list[str] | None) -> dict[str, str]:
    """Parse ``--palette MODEL=COLOUR`` arguments into a mapping.

    The flag repeats, and one value may carry several comma-separated pairs — neither a
    model id nor a colour spec contains a comma, so the split is unambiguous.
    """
    palette = {}
    for value in values or []:
        for pair in value.split(","):
            if not pair.strip():
                continue
            model, sep, color = pair.partition("=")
            if not sep or not model.strip() or not color.strip():
                raise ReportError(f"--palette expects MODEL=COLOUR, got {pair.strip()!r}")
            palette[model.strip()] = color.strip()
    return palette


def _cmd_report(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks) if args.tasks.exists() else None
    if tasks is None:
        print(f"note: {args.tasks} not found — per-task prompts will be omitted", file=sys.stderr)
    skills_root = args.skills if args.skills.is_dir() else None
    if skills_root is None:
        print(f"note: {args.skills} not found — skill rationale/diff will be omitted", file=sys.stderr)
    report = build_report(args.runs, args.out, tasks, skills_root=skills_root, palette=_parse_palette(args.palette))
    df = report.results
    arms = ", ".join(sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)))
    print(f"aggregated {report.n_runs} runs across arms: {arms}")
    for arm in sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)):
        group = df[df["arm_label"] == arm]
        print(f"  {arm}: {int(group['success'].sum())}/{len(group)} passed")
    print(f"wrote {args.out.resolve()}")
    print(f"wrote {args.out.resolve().with_suffix('.csv')}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    written = scaffold(args.directory, force=args.force)
    for path in written:
        print(f"wrote {path}")
    print("\nnext: edit config.yaml (repo) and tasks.yaml, then `acumen draft`")
    return 0


def _cmd_prices(args: argparse.Namespace) -> int:
    """Show the rates cost is computed from, or check pinned rates against the pages.

    Both modes read the providers' pages, since that is the only place rates come from.
    ``--refresh`` differs in what it prints: the models whose published price disagrees
    with a ``prices:`` pin in ``config.yaml``, which are the only rates that can drift
    unnoticed once everything else is read live.
    """
    overrides: dict[str, Rates] = {}
    models: set[str] = set()
    if args.config.is_file():
        cfg = load_config(args.config)
        overrides, models = cfg.prices, {m.strip().lower() for m in cfg.models}

    for name, url in PRICE_SOURCES.items():
        print(f"fetching {name}: {url}", flush=True)
    try:
        fetched = refresh(today=date.today())
    except PriceFeedError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if not args.refresh:
        table = PriceTable(overrides=overrides, fetched=fetched, fetched_as_of=date.today().isoformat())
        shown = sorted(set(fetched) | set(overrides)) if args.all else sorted(models | set(overrides))
        print(f"\nrates (USD per million tokens) as of {table.fetched_as_of} — {PRICE_TIER}")
        for model in shown:
            found = table.lookup(model)
            if found is None:
                print(f"  {model:28} unpriced (not published; add it under 'prices:' to price it)")
                continue
            print(
                f"  {model:28} in ${found.rates.input:<7} cached ${found.rates.cached_input:<7} "
                f"write ${found.rates.cache_write:<7} out ${found.rates.output:<7} ({found.source})"
            )
        if not args.all:
            print(f"\n{len(fetched)} model(s) published in total; show them all with --all")
        return 0

    # Only pinned rates can disagree with a page: everything else is read live each run.
    if not overrides:
        print("\nno 'prices:' pins in config.yaml, so nothing can drift — rates are read live each run")
        return 0
    changes = diff_rates(overrides, {m: r for m, r in fetched.items() if m in overrides})

    if not changes:
        print(f"\nup to date — all {len(overrides)} pinned rate(s) match what the providers publish")
        return 0

    print(f"\n{len(changes)} pinned rate(s) differ from the published price ({PRICE_TIER}):")
    for change in changes:
        print(change.describe())

    block = to_yaml_block({change.model: change.after for change in changes})
    if args.out is not None:
        args.out.write_text(block)
        print(f"\nwrote {args.out} — merge its 'prices:' block into config.yaml to adopt these")
    else:
        print("\nadopt by updating config.yaml (or re-run with --out PATH):\n")
        print(block, end="")
    # Never applied automatically: a pin is a deliberate statement, and a mis-parsed tier
    # or context band would otherwise overwrite it with a plausible wrong number.
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``acumen`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="acumen", description="Build, benchmark, and optimize agentic skills for Python packages."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser(
        "bench",
        help="run a benchmark pass over every arm, or one named arm",
        description="Bench every arm the project has (the baseline plus each version in skills/), "
        "or a single arm with --no-skill / --skill vN. Completed runs are skipped, so rerunning "
        "only costs the cells that are missing.",
    )
    _add_bench_args(bench)
    bench.set_defaults(func=_cmd_bench)

    draft = sub.add_parser("draft", help="draft a skill from the target package's source")
    draft.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    draft.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    draft.add_argument("--model", help="override config draft_model")
    draft.add_argument("--max-turns", type=int, help="cap turns for the drafting agent (default: unbounded)")
    draft.add_argument("--max-usd", type=float, help="cap spend for the drafting agent (default: unbounded)")
    draft.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    draft.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    draft.add_argument("--force", action="store_true", help="draft another version even if some already exist")
    _add_auth_arg(draft)
    _add_feedback_arg(draft, extra=" (e.g. package context, what the skill should emphasise)")
    _add_log_args(draft)
    draft.set_defaults(func=_cmd_draft)

    improve = sub.add_parser("improve", help="improve the current skill into a new version from its train results")
    improve.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    improve.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    improve.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    improve.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    improve.add_argument("--from", dest="from_version", metavar="VERSION", help="version to improve (default: latest)")
    improve.add_argument("--model", help="override config improve_model")
    improve.add_argument("--max-turns", type=int, help="cap turns for the improving agent (default: unbounded)")
    improve.add_argument("--max-usd", type=float, help="cap spend for the improving agent (default: unbounded)")
    improve.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    improve.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    _add_auth_arg(improve)
    _add_feedback_arg(
        improve,
        extra=" (e.g. what to fix or emphasise; do NOT paste test-split answers — that defeats the held-out split)",
    )
    _add_log_args(improve)
    improve.set_defaults(func=_cmd_improve)

    tasks_cmd = sub.add_parser("tasks", help="autonomously generate a tasks.yaml from the target package")
    tasks_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    tasks_cmd.add_argument("--out", type=Path, default=Path("tasks.yaml"), help="tasks.yaml to write")
    tasks_cmd.add_argument(
        "--scripts",
        type=Path,
        default=Path(SCRIPTS_DIRNAME),
        help="directory to keep the reproducer script for each split in, for `acumen check`",
    )
    tasks_cmd.add_argument("--model", help="override config tasks_model")
    tasks_cmd.add_argument("--max-turns", type=int, help="cap turns for the generation agent (default: unbounded)")
    tasks_cmd.add_argument("--max-usd", type=float, help="cap spend for the generation agent (default: unbounded)")
    tasks_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    tasks_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    tasks_cmd.add_argument("--force", action="store_true", help="overwrite an existing tasks file")
    _add_auth_arg(tasks_cmd)
    _add_feedback_arg(tasks_cmd, extra=" (e.g. which functionality to skip or focus on)")
    _add_log_args(tasks_cmd)
    tasks_cmd.set_defaults(func=_cmd_tasks)

    check = sub.add_parser(
        "check",
        help="verify each task's answer by running its reproducer, then review prompt/answer/script agreement",
        description=f"Two phases. First, run {SCRIPTS_DIRNAME}/<task>-<split>.py for every task in "
        "the target venv and compare the answer.md it writes against the answer recorded in "
        "tasks.yaml. Then an agent reads each task's prompt, recorded answer and reproducer "
        "together and says whether they describe the same thing, since a prompt that asks for "
        "something else than its script computes fails every agent that reads it correctly. Both "
        "phases catch a broken task before a benchmark pass pays for it. A task that needs no code "
        "sets 'needs_script: false'; --no-review skips the agent and costs nothing.",
    )
    check.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    check.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    check.add_argument(
        "--scripts", type=Path, default=Path(SCRIPTS_DIRNAME), help="directory holding the reproducer scripts"
    )
    check.add_argument("--task", metavar="ID", action="append", help="restrict to a task id (repeatable)")
    check.add_argument("--split", choices=SPLITS, action="append", help="restrict to a split (repeatable)")
    check.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds one script may run (default: {DEFAULT_TIMEOUT:g})",
    )
    check.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS, help=f"scripts to run at once (default: {DEFAULT_JOBS})"
    )
    check.add_argument("--keep", action="store_true", help="leave each script's working directory on disk")
    check.add_argument("--json", type=Path, default=None, help="also write the results and summary here as JSON")
    check.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    check.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    check.add_argument(
        "--no-review",
        action="store_true",
        help="skip the coherence review: run the reproducers only, spawning no agent and spending nothing",
    )
    check.add_argument("--model", help="override config check_model (the review agent)")
    check.add_argument("--max-turns", type=int, help="cap turns for the review agent (default: unbounded)")
    check.add_argument("--max-usd", type=float, help="cap spend for the review agent (default: unbounded)")
    _add_auth_arg(check)
    _add_log_args(check)
    check.set_defaults(func=_cmd_check)

    ship = sub.add_parser("ship", help="make a benchmarked skill installable into the target package")
    ship.add_argument(
        "--skill", dest="version", metavar="VERSION", required=True, help="skill version to ship, e.g. v2"
    )
    ship.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    ship.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    ship.add_argument("--model", help="override config ship_model")
    ship.add_argument("--max-turns", type=int, help="cap turns for the ship agent (default: unbounded)")
    ship.add_argument("--max-usd", type=float, help="cap spend for the ship agent (default: unbounded)")
    ship.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    ship.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    ship.add_argument("--force", action="store_true", help="ship even if the package already has an installer")
    _add_auth_arg(ship)
    _add_log_args(ship)
    ship.set_defaults(func=_cmd_ship)

    report = sub.add_parser("report", help="aggregate the run tree into a self-contained report.html")
    report.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    report.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml (for task text)")
    report.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree (rationale/diff)")
    report.add_argument("--out", type=Path, default=Path("report.html"), help="output HTML path (overwritten)")
    report.add_argument(
        "--palette",
        action="append",
        metavar="MODEL=COLOUR",
        help="recolour a model's bars, e.g. --palette claude-opus-5=#3b7ea1 (repeatable, or comma-separated)",
    )
    report.set_defaults(func=_cmd_report)

    prices = sub.add_parser("prices", help="show the token rates cost is computed from, or re-check them")
    prices.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml (for overrides)")
    prices.add_argument("--refresh", action="store_true", help="fetch the providers' pricing pages and diff them")
    prices.add_argument("--all", action="store_true", help="with --refresh, report every model, not just yours")
    prices.add_argument("--out", type=Path, default=None, help="with --refresh, write the 'prices:' block here")
    prices.set_defaults(func=_cmd_prices)

    init = sub.add_parser("init", help="scaffold a starter config.yaml and tasks.yaml")
    init.add_argument("--dir", type=Path, default=Path("."), dest="directory", help="directory to scaffold into")
    init.add_argument("--force", action="store_true", help="overwrite existing config.yaml / tasks.yaml")
    init.set_defaults(func=_cmd_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Returns
    -------
    A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        CheckError,
        ConfigError,
        TaskError,
        EnvError,
        AgentError,
        PriceFeedError,
        SkillError,
        DraftError,
        ImproveError,
        TaskGenError,
        ReviewError,
        ShipError,
        ReportError,
        InitError,
    ) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted — completed runs are preserved; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
