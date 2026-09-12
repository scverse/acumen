"""Command-line entry point — a thin shell over the importable API."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
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
from acumen.env import DEFAULT_CACHE_ROOT, AuthMode, EnvError, prepare_target, resolve_auth_mode
from acumen.epoch import EpochPlan, resolve_epoch
from acumen.grade import INVALID_REASONS
from acumen.improve import ImproveError, improve_skill
from acumen.logs import LiveLog
from acumen.paths import SPLITS, Split, arm_name
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
from acumen.skills import Skill, SkillError, available_versions, latest_version, load_skill, skill_dir
from acumen.taskgen import TaskGenError, generate_tasks
from acumen.tasks import Task, TaskError, load_tasks
from acumen.training import (
    EpochRow,
    best_version,
    build_training_rows,
    epochs_since_best,
    patience_exhausted,
    write_training_csv,
)
from acumen.wiki import WikiError, collect_arm_runs, update_wiki


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
    Turns stream, so that cap is exact. Usage arrives once per model response, read from the
    running total Codex records for the turn, so ``max_usd`` stops the run partway through it
    rather than the instant the cap is crossed: one response can still carry it past the limit.
    """
    if provider == "codex":
        print(
            "note: Codex reports usage once per model response, so max_usd stops the run at "
            "the first report past the cap and a single response can still overshoot it",
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


def _fmt_rate(value: float | None) -> str:
    """Format a 0..1 success rate as a percentage, or ``n/a`` when unknown."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.0%}"


def _epoch_bar(
    done: int,
    total: int,
    row: EpochRow | None,
    *,
    best: str | None,
    patience: int,
    since_best: int,
    fixed: bool,
) -> str:
    """A tqdm-style one-liner summarising an epoch: bar, version, train/valid success, patience."""
    width = 14
    filled = round(width * done / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    version = row.version if row is not None else "?"
    train = _fmt_rate(row.train_success) if row is not None else "n/a"
    valid = _fmt_rate(row.valid_success) if row is not None else "n/a"
    parts = [f"Epoch {done}/{total} |{bar}| {version}", f"train {train}", f"valid {valid}"]
    if row is not None and not (isinstance(row.valid_cost, float) and math.isnan(row.valid_cost)):
        parts.append(f"${row.valid_cost:.2f}/run")
    if not fixed and best is not None:
        parts.append(f"(best {best}, patience {min(since_best, patience)}/{patience})")
    return "  ".join(parts)


# Progress rendering for `epoch`/`fit`. Three modes, resolved once per command:
#   "bars"    — tqdm-style single line rewritten in place with \r (a live terminal)
#   "plain"   — throttled fresh lines, no \r (output redirected to a file / CI)
#   "verbose" — today's per-run scrolling logs, via _Progress (opt-in with --verbose)
_BAR_WIDTH = 14
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _progress_mode(args: argparse.Namespace) -> str:
    """Pick the progress style: explicit --verbose wins, else live bars on a TTY, else plain."""
    if getattr(args, "verbose", False):
        return "verbose"
    if sys.stdout.isatty() and not getattr(args, "stream", False):
        return "bars"
    return "plain"


def _epoch_header(mode: str, n: int, total: int) -> None:
    """Print the marker that starts an epoch — a heavy banner in verbose, a compact line else."""
    if mode == "verbose":
        print(f"\n{'═' * 78}\nEPOCH {n}/{total}\n{'═' * 78}", flush=True)
    else:
        print(f"\nEpoch {n}/{total}", flush=True)


def _progress_bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    filled = width if not total else max(0, min(width, round(width * done / total)))
    return "█" * filled + "░" * (width - filled)


class _PhaseBar:
    r"""A tqdm-style bar for one bench or wiki phase of an epoch.

    Exposes the ``on_start``/``on_done(RunOutcome)`` callbacks ``run_matrix`` expects (bench) and
    ``on_wiki_task(TaskWikiResult)`` for the wiki pass. In ``bars`` mode it rewrites one line with
    ``\r``; in ``plain`` mode it prints throttled fresh lines so redirected logs stay readable.
    (``verbose`` keeps :class:`_Progress` and never builds this, so only those two modes reach here.)
    """

    _THROTTLE_S = 5.0

    def __init__(self, label: str, total: int, mode: str, *, track_success: bool = True) -> None:
        self.label = label
        self.total = total
        self.mode = mode
        self.track_success = track_success
        self.done = 0
        self.passed = 0
        self.cost = 0.0
        self._t0 = time.monotonic()
        self._last_print = 0.0
        if mode == "bars":
            print("\r" + self._line(), end="", flush=True)

    @property
    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def set_total(self, total: int) -> None:
        """Set the denominator once it is known (the wiki pass reports it via ``on_plan``)."""
        self.total = total
        if self.mode == "bars":
            print("\r" + self._line(), end="", flush=True)

    def on_start(self, item) -> None:
        """Present so it can be handed to ``run_matrix``; the bar only renders on completion."""

    def on_done(self, outcome: RunOutcome) -> None:
        payload = outcome.payload
        priced = payload.get("cost_available", True) and payload.get("cost_usd") is not None
        self._surface_error(outcome)
        self._tick(success=outcome.success, cost=float(payload["cost_usd"]) if priced else None)

    def on_wiki_task(self, result) -> None:
        self._tick(success=None, cost=result.cost_usd)

    def _tick(self, *, success: bool | None, cost: float | None) -> None:
        self.done += 1
        if success:
            self.passed += 1
        if cost is not None:
            self.cost += cost
        if self.mode == "bars":
            print("\r" + self._line(), end="", flush=True)
        elif self.done < self.total and self._elapsed - self._last_print >= self._THROTTLE_S:
            self._last_print = self._elapsed
            print("  " + self._line(), flush=True)

    def _line(self) -> str:
        pct = round(100 * self.done / self.total) if self.total else 100
        parts = [f"{self.label:>14} {pct:>3}%|{_progress_bar(self.done, self.total)}| {self.done}/{self.total}"]
        parts.append(f"[{_fmt_secs(self._elapsed)}]")
        if self.track_success:
            parts.append(f"success {_fmt_rate(self.passed / self.done if self.done else None)} (mean)")
        parts.append(f"{_fmt_cost(self.cost)} total")
        return "  ".join(parts)

    def _surface_error(self, outcome: RunOutcome) -> None:
        """Let genuine harness failures through even in bars mode; ordinary test fails just lower the rate."""
        detail = outcome.payload.get("error")
        if outcome.reason == "provider_exhausted":
            msg = f"provider usage/credit exhausted: {detail or 'no provider detail'}"
        elif outcome.reason == "sandbox_blocked":
            msg = f"agent sandbox refused an outbound host (harness bug): {detail or 'no sandbox detail'}"
        else:
            return
        print(f"{chr(10) if self.mode == 'bars' else ''}error: {msg}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        """Leave the completed bar on screen (bars) or print the final line (plain)."""
        print(("\r" if self.mode == "bars" else "  ") + self._line(), flush=True)


class _Spinner:
    """An indeterminate elapsed-time spinner for the single-agent improve step (no sub-progress).

    In ``bars`` mode a daemon thread rewrites the line while ``improve_skill`` blocks the main
    thread; other modes just print a start line. Use as a context manager around the blocking call.
    """

    def __init__(self, label: str, mode: str) -> None:
        self.label = label
        self.mode = mode
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Spinner:
        if self.mode == "bars":
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print(f"  {self.label} ...", flush=True)
        return self

    def _spin(self) -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if self._stop.is_set():
                break
            print(f"\r  {self.label} {frame} {_fmt_secs(time.monotonic() - self._t0)}", end="", flush=True)
            self._stop.wait(0.1)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            print("\r\033[K", end="", flush=True)  # clear the spinner line for the done line


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


def _print_skill_loading(outcomes: Sequence[RunOutcome], arm: _Arm, skill_name: str, *, quiet: bool = False) -> None:
    """Say whether the skill reached the agent — the comparison means nothing otherwise.

    ``quiet`` drops the routine "loaded in N/M" line (the bars keep phase output compact) but
    still raises the warnings, which signal a broken comparison and must never be swallowed.
    """
    loaded = sum(1 for o in outcomes if o.payload.get("skill_loaded"))
    if arm.skill is not None:
        if not quiet:
            print(f"skill loaded in {loaded}/{len(outcomes)} runs")
        if loaded == 0:
            print(
                f"warning: {arm.name} never loaded the skill — that arm is not measuring the skill",
                file=sys.stderr,
            )
    elif loaded:
        print(f"warning: {skill_name} loaded in {loaded} baseline runs", file=sys.stderr)


def _resolve_bench_auth(models: set[str], auth: str) -> dict[AgentProvider, AuthMode]:
    """Resolve one auth mode per provider present, checking each CLI and printing the choice."""
    providers = {provider_for_model(model) for model in models}
    auth_modes = {provider: resolve_auth_mode(auth, provider=provider) for provider in providers}
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
    return auth_modes


def _execute_arms(
    arms: Sequence[_Arm],
    *,
    cfg: Config,
    target,
    runs_root: Path,
    auth_modes: dict[AgentProvider, AuthMode],
    prices: PriceTable,
    keep_sandboxes: bool,
    progress: _Progress | _PhaseBar | None = None,
    quiet: bool = False,
) -> list[RunOutcome]:
    """Run each arm's pending runs sequentially, sharing one progress counter across them.

    Arms run one after another: every run in a matrix shares one skill, and a sequential pass
    keeps each arm's tally readable while the progress counter spans the whole thing. ``quiet``
    (epoch/fit bar mode) suppresses the per-arm banner and tally so the phase bar is the only
    output; harness warnings still surface. Raises :class:`BenchmarkInvalidError` if a harness
    failure decides the pass.
    """
    running = [arm for arm in arms if arm.todo]
    todo = [item for arm in running for item in arm.todo]
    if not todo:
        return []
    progress = progress or _Progress(len(todo))
    collected: list[RunOutcome] = []
    for arm in running:
        if len(running) > 1 and not quiet:
            print(f"\n=== arm {arm.name}: {len(arm.todo)} runs ===", flush=True)
        started = time.monotonic()
        outcomes = asyncio.run(
            run_matrix(
                arm.todo,
                target=target,
                runs_root=runs_root,
                max_concurrency=cfg.max_concurrency,
                auth_modes=auth_modes,
                skill=arm.skill,
                skill_name=cfg.skill_name,
                keep_sandbox=keep_sandboxes,
                stderr=StderrFilter(),
                on_start=progress.on_start,
                on_done=progress.on_done,
                env_passthrough=cfg.env_passthrough,
                prices=prices,
            )
        )
        collected.extend(outcomes)
        if not quiet:
            _print_run_summary(outcomes, time.monotonic() - started, label=arm.name if len(running) > 1 else "")
        _print_skill_loading(outcomes, arm, cfg.skill_name, quiet=quiet)
    return collected


def _build_arms(
    specs: Sequence[tuple[str | None, Sequence[Split]]],
    *,
    cfg: Config,
    tasks: Sequence[Task],
    runs_root: Path,
    skills_root: Path,
    resume: bool = True,
    task_ids: Sequence[str] | None = None,
) -> list[_Arm]:
    """Build arms for an explicit list of ``(version, splits)`` specs (for the epoch orchestrator)."""
    arms = []
    for version, splits in specs:
        skill = None if version is None else load_skill(skills_root, version, expect_name=cfg.skill_name)
        planned = build_matrix(cfg, tasks, skill=version, splits=splits, task_ids=task_ids)
        todo = pending(planned, runs_root, resume=resume)
        arms.append(_Arm(version=version, skill=skill, planned=planned, todo=todo))
    return arms


def _invalid_bench_note() -> None:
    print(
        "Fix or replenish that credential, then rerun the same command; invalid and cancelled cells remain pending.",
        file=sys.stderr,
    )


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
    auth_modes = _resolve_bench_auth({item.model for item in todo}, args.auth)
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

    running = [arm for arm in arms if arm.todo]
    print(f"running {len(todo)} runs, up to {cfg.max_concurrency} at a time:", flush=True)
    progress = _Progress(len(todo))
    try:
        collected = _execute_arms(
            arms,
            cfg=cfg,
            target=target,
            runs_root=args.runs,
            auth_modes=auth_modes,
            prices=prices,
            keep_sandboxes=args.keep_sandboxes,
            progress=progress,
        )
    except BenchmarkInvalidError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        _invalid_bench_note()
        return 2

    if len(running) > 1:
        _print_run_summary(collected, progress.elapsed, label=f"all {len(running)} arms")
    print(f"runs written to {args.runs.resolve()}")
    return 0


def _cmd_improve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

    # Parent is an explicit --from, else the latest version, else None (create the first skill
    # from the noskill wiki). improve_skill resolves and validates the parent itself.
    parent = args.from_version or latest_version(args.skills)
    parent_skill = None if parent is None else load_skill(args.skills, parent, expect_name=cfg.skill_name)
    # Fail before the costly target prep if there is nothing to learn from: the improver needs the
    # parent arm's train runs (and the wiki built from them).
    parent_arm = arm_name(parent)
    if not collect_arm_runs(args.runs, parent_arm, tasks, split="train"):
        hint = (
            "acumen bench --no-skill --split train"
            if parent is None
            else f"acumen bench --skill {parent} --split train"
        )
        print(
            f"no train-split runs found for {parent_arm} under {args.runs / parent_arm / 'train'} — "
            f"run `{hint}` and `acumen wiki` first",
            file=sys.stderr,
        )
        return 2
    if parent_skill is None:
        print(f"no skill versions under {args.skills} — creating the first skill from the wiki with {cfg.meta_model}")
    else:
        print(
            f"improving {parent_skill.version} ({parent_skill.name}, {parent_skill.hash[:19]}…) with {cfg.meta_model}"
        )

    provider = provider_for_model(cfg.meta_model)
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
                prices=_agent_prices(cfg, model=cfg.meta_model),
                target=target,
                skills_root=args.skills,
                runs_root=args.runs,
                wiki_root=args.wiki,
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
    parent_label = result.parent or "noskill"
    print(f"\nwrote {new.directory}  (parent {parent_label})")
    print(f"  name:        {new.name}")
    print(f"  description: {new.description}")
    print(f"  hash:        {new.hash}")
    print(f"  files:       {', '.join(files)}")
    print(f"  evidence:    {result.n_train_runs} train runs ({result.n_train_failures} failing)")
    print(f"  cost:        {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    if parent_skill is not None and new.hash == parent_skill.hash:
        print(
            "warning: the new version is byte-identical to its parent — the improver changed nothing",
            file=sys.stderr,
        )
    _print_log_result(log)
    print(f"\nnext: acumen bench --skill {new.version} && acumen report")
    return 0


def _wiki_version(args: argparse.Namespace) -> str:
    """Resolve the version label a wiki update records, from ``--skill``/``--no-skill``/default."""
    if args.skill is not None:
        return args.skill if args.skill.startswith("v") else f"v{args.skill}"
    if args.no_skill:
        return arm_name(None)
    return latest_version(args.skills) or arm_name(None)


def _print_wiki_task(result) -> None:
    """Print one task's wiki-update outcome as it lands, with any brevity warnings."""
    print(f"  wiki [{result.version}] {result.task_id}: {_fmt_cost(result.cost_usd)} over {result.turns} turns")
    for warning in result.warnings:
        print(f"    warning: {warning}", file=sys.stderr)


def _run_wiki(
    *,
    cfg: Config,
    tasks: Sequence[Task],
    target,
    version: str,
    runs_root: Path,
    wiki_root: Path,
    skills_root: Path,
    prices: PriceTable,
    auth_mode: AuthMode,
    log_dir: Path,
    stream: bool,
    mode: str = "verbose",
):
    """Update the wiki for one arm and report progress; returns the task results.

    ``mode`` (``verbose``/``bars``/``plain``) picks the reporting style — a per-task tally in
    verbose, a single progress bar otherwise. The bar's denominator is the count of tasks the
    pass will actually run, reported once via ``update_wiki``'s ``on_plan`` callback.
    """
    bar = None if mode == "verbose" else _PhaseBar("wiki", 0, mode, track_success=False)
    if mode == "verbose":
        print(f"updating wiki for [{version}] with {cfg.meta_model} (one agent per task) ...", flush=True)
    results = asyncio.run(
        update_wiki(
            cfg=cfg,
            target=target,
            runs_root=runs_root,
            wiki_root=wiki_root,
            skills_root=skills_root,
            tasks=tasks,
            version=version,
            prices=prices,
            auth_mode=auth_mode,
            max_concurrency=cfg.max_concurrency,
            log_dir=log_dir,
            stream=stream,
            on_plan=(bar.set_total if bar is not None else None),
            on_task_done=(_print_wiki_task if bar is None else bar.on_wiki_task),
        )
    )
    if bar is not None:
        bar.finish()
    elif not results:
        print(f"  wiki already had [{version}] for every task — nothing to do")
    else:
        total = sum(r.cost_usd or 0.0 for r in results)
        print(f"wiki: updated {len(results)} task(s) for [{version}]  ({_fmt_cost(total)})")
    return results


def _cmd_wiki(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

    version = _wiki_version(args)
    if version != arm_name(None):
        # Validate a named skill version exists before the costly target prep.
        load_skill(args.skills, version, expect_name=cfg.skill_name)

    provider = provider_for_model(cfg.meta_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    _run_wiki(
        cfg=cfg,
        tasks=tasks,
        target=target,
        version=version,
        runs_root=args.runs,
        wiki_root=args.wiki,
        skills_root=args.skills,
        prices=_agent_prices(cfg, model=cfg.meta_model),
        auth_mode=auth_mode,
        log_dir=args.log_dir,
        stream=args.stream,
    )
    print(f"\nwiki written to {args.wiki.resolve()}")
    return 0


def _prepare_pass(cfg: Config, args: argparse.Namespace):
    """Resolve auth, freeze prices, and build the target once for a pass that spends on benches.

    Shared by ``epoch`` and ``fit`` so a multi-epoch run prepares the target and rates once rather
    than per epoch. Raises :class:`PriceFeedError` if rates cannot be established (benchmark cost is
    frozen into every result, so a pass must not run without them).
    """
    auth_modes = _resolve_bench_auth(set(cfg.models) | {cfg.meta_model}, args.auth)
    meta_auth = auth_modes[provider_for_model(cfg.meta_model)]
    prices = _bench_prices(cfg)
    _warn_unpriced(set(cfg.models) | {cfg.meta_model}, prices)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)
    return auth_modes, meta_auth, prices, target


def _run_one_epoch(
    args: argparse.Namespace,
    *,
    cfg: Config,
    tasks: list[Task],
    target,
    prices: PriceTable,
    auth_modes: dict[AgentProvider, AuthMode],
    meta_auth: AuthMode,
    mode: str = "verbose",
) -> EpochPlan:
    """Run one training epoch on an already-prepared pass, returning the resolved plan.

    The four steps: bench the parent arm on train (the first epoch also benches the noskill
    baseline on valid), distil that arm into the wiki, create or improve the skill (skipped when
    the version already exists — a resumed epoch), then bench the new version on valid.

    Auth, prices, and the target are passed in, not re-derived, so ``fit`` runs many epochs against
    one prepared target. Raises :class:`BenchmarkInvalidError` if a harness failure decides a bench;
    the caller reports it.
    """
    runs_root = args.runs

    def valid_complete(version: str) -> bool:
        planned = build_matrix(cfg, tasks, skill=version, splits=["valid"])
        return not pending(planned, runs_root, resume=True)

    plan = resolve_epoch(args.skills, valid_complete=valid_complete)
    parent_label = plan.parent_version or arm_name(None)
    tail = " (resuming)" if plan.resumed else ""
    print(f"epoch: learning from [{parent_label}] → producing {plan.new_version}{tail}")

    def bench_phase(specs, *, label: str, verbose_banner: str) -> None:
        """Run one bench step: a numbered banner + plan + tally in verbose, a phase bar otherwise."""
        if mode == "verbose":
            print(verbose_banner, flush=True)
        bench_arms = _build_arms(specs, cfg=cfg, tasks=tasks, runs_root=runs_root, skills_root=args.skills)
        if mode == "verbose":
            _print_plan(bench_arms)
        bar = None if mode == "verbose" else _PhaseBar(label, sum(len(a.todo) for a in bench_arms), mode)
        _execute_arms(
            bench_arms,
            cfg=cfg,
            target=target,
            runs_root=runs_root,
            auth_modes=auth_modes,
            prices=prices,
            keep_sandboxes=False,
            progress=bar,
            quiet=mode != "verbose",
        )
        if bar is not None:
            bar.finish()

    # Step 1 — bench the parent arm on the training signal. The first epoch also benches the
    # noskill baseline on valid, so the report has an in-epoch baseline to compare against.
    if plan.first:
        train_specs: list[tuple[str | None, Sequence[Split]]] = [(None, ["train", "valid"])]
    else:
        train_specs = [(plan.parent_version, ["train"])]
    bench_phase(
        train_specs,
        label="bench baseline" if plan.first else "bench train",
        verbose_banner=f"\n[1/4] benchmarking [{parent_label}] on the training signal ...",
    )

    # Step 2 — distil that arm into the wiki (idempotent: recorded arms are skipped).
    if mode == "verbose":
        print(f"\n[2/4] updating the wiki for [{parent_label}] ...", flush=True)
    _run_wiki(
        cfg=cfg,
        tasks=tasks,
        target=target,
        version=parent_label,
        runs_root=runs_root,
        wiki_root=args.wiki,
        skills_root=args.skills,
        prices=prices,
        auth_mode=meta_auth,
        log_dir=args.log_dir,
        stream=args.stream,
        mode=mode,
    )

    # Step 3 — create or improve the skill (skipped when the version already exists: a resumed
    # epoch that crashed after improve).
    if skill_dir(args.skills, plan.new_version).exists():
        resumed_msg = f"{plan.new_version} already exists — skipping improve (resumed epoch)"
        print(f"\n[3/4] {resumed_msg}" if mode == "verbose" else f"  {resumed_msg}", flush=True)
    else:
        verb = "creating" if plan.first else "improving"
        log = LiveLog.open(args.log_dir, "improve", stream=args.stream)
        if mode == "verbose":
            print(f"\n[3/4] {verb} the skill → {plan.new_version} with {cfg.meta_model} ...", flush=True)
            print(f"log → {log.jsonl_path}", flush=True)
        spinner = nullcontext() if mode == "verbose" else _Spinner(f"inferring skill → {plan.new_version}", mode)
        improve_started = time.monotonic()
        with log, spinner:
            result = asyncio.run(
                improve_skill(
                    cfg=cfg,
                    prices=prices,
                    target=target,
                    skills_root=args.skills,
                    runs_root=runs_root,
                    wiki_root=args.wiki,
                    tasks=tasks,
                    auth_mode=meta_auth,
                    parent_version=plan.parent_version,
                    feedback=args.feedback,
                    log=log,
                )
            )
        improve_elapsed = _fmt_secs(time.monotonic() - improve_started)
        new = result.skill
        if mode == "verbose":
            print(f"wrote {new.directory}  (parent {result.parent or 'noskill'})")
            print(f"  description: {new.description}")
            print(f"  cost:        {_fmt_cost(result.cost_usd)} over {result.turns} turns in {improve_elapsed}")
            _print_log_result(log)
        else:
            print(
                f"  inferring skill → {new.version} done  "
                f"({improve_elapsed}, {_fmt_cost(result.cost_usd)}, {result.turns} turns)",
                flush=True,
            )

    # Step 4 — bench the new version on the held-out valid signal (completes it if unfinished).
    bench_phase(
        [(plan.new_version, ["valid"])],
        label="bench valid",
        verbose_banner=f"\n[4/4] benchmarking {plan.new_version} on the held-out valid signal ...",
    )
    return plan


def _cmd_epoch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.replicates:
        cfg = replace(cfg, n_replicates=args.replicates)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

    try:
        auth_modes, meta_auth, prices, target = _prepare_pass(cfg, args)
    except PriceFeedError as err:
        print(f"error: {err}", file=sys.stderr)
        print("Retry when the pricing pages are reachable, or pin rates in config.yaml.", file=sys.stderr)
        return 2

    try:
        plan = _run_one_epoch(
            args,
            cfg=cfg,
            tasks=tasks,
            target=target,
            prices=prices,
            auth_modes=auth_modes,
            meta_auth=meta_auth,
            mode=_progress_mode(args),
        )
    except BenchmarkInvalidError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        _invalid_bench_note()
        return 2

    print(f"\nepoch complete: {plan.new_version} produced and benched on valid.")
    print("next: `acumen report` to see it, or `acumen epoch` again for another round")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.replicates:
        cfg = replace(cfg, n_replicates=args.replicates)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

    fixed = args.epochs is not None
    limit = args.epochs if fixed else args.max_epochs
    if limit < 1:
        print("error: nothing to run — --epochs/--max-epochs must be >= 1", file=sys.stderr)
        return 2
    if fixed:
        print(f"fit: running exactly {limit} epoch(s) (early stopping disabled)")
    else:
        print(f"fit: up to {limit} epoch(s), early stopping with patience {args.patience}")

    try:
        auth_modes, meta_auth, prices, target = _prepare_pass(cfg, args)
    except PriceFeedError as err:
        print(f"error: {err}", file=sys.stderr)
        print("Retry when the pricing pages are reachable, or pin rates in config.yaml.", file=sys.stderr)
        return 2

    mode = _progress_mode(args)
    for done in range(limit):
        _epoch_header(mode, done + 1, limit)
        try:
            plan = _run_one_epoch(
                args,
                cfg=cfg,
                tasks=tasks,
                target=target,
                prices=prices,
                auth_modes=auth_modes,
                meta_auth=meta_auth,
                mode=mode,
            )
        except BenchmarkInvalidError as err:
            print(f"\nerror: {err}", file=sys.stderr)
            _invalid_bench_note()
            return 2

        # Rebuild the whole training curve from runs/ each epoch, so the CSV is always consistent
        # with what exists and a resumed fit produces the same table.
        rows = build_training_rows(args.runs, cfg)
        write_training_csv(rows, args.out)
        current = next((row for row in rows if row.version == plan.new_version), None)
        valids = [row.valid_success for row in rows]
        best = best_version(rows)
        since = epochs_since_best(valids)
        print(_epoch_bar(done + 1, limit, current, best=best, patience=args.patience, since_best=since, fixed=fixed))

        if not fixed and patience_exhausted(valids, args.patience):
            print(f"\nearly stop: validation mean success did not improve in {args.patience} epoch(s).")
            break

    rows = build_training_rows(args.runs, cfg)
    best = best_version(rows)
    if best is not None:
        best_row = next(row for row in rows if row.version == best)
        print(f"\nfit complete: best version {best} (valid {_fmt_rate(best_row.valid_success)}).")
    else:
        print("\nfit complete.")
    print(f"training curve → {args.out.resolve()}")
    print("next: `acumen report` for the full breakdown")
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

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

    provider = provider_for_model(cfg.meta_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(
        f"generating tasks with {cfg.meta_model} (this reads the source and runs package code) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "tasks", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            generate_tasks(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.meta_model),
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
    print("\nnext: review the tasks, then `acumen check`, then `acumen epoch`")
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
        cfg = replace(cfg, meta_model=args.model)
    tasks = select_tasks(load_tasks(args.tasks), args.task)
    splits = args.split or list(SPLITS)
    review_on = not args.no_review

    # Everything the review needs is resolved before the target is built and before a single
    # script runs: a missing agent CLI or an unusable credential must not surface after half an
    # hour of analyses, when it was knowable up front.
    auth_mode: AuthMode = "api"
    prices: PriceTable | None = None
    if review_on:
        provider = provider_for_model(cfg.meta_model)
        check_agent_cli(provider)
        auth_mode = resolve_auth_mode(args.auth, provider=provider)
        _print_auth(auth_mode, provider)
        _warn_codex_accounting(provider)
        prices = _agent_prices(cfg, model=cfg.meta_model)

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
        print(f"\nreviewing {len(results)} task splits with {cfg.meta_model} ...", flush=True)
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
                        "model": cfg.meta_model,
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
        cfg = replace(cfg, meta_model=args.model)

    # Validate the version exists before the (costly) target prep.
    load_skill(args.skills, args.version, expect_name=cfg.skill_name)

    where = (
        "a local path — the change is written to the working tree"
        if cfg.is_local
        else ("a GitHub URL — the change is delivered as a pull request")
    )
    print(f"shipping {args.version} of {cfg.skill_name} into {cfg.repo} ({where})")
    provider = provider_for_model(cfg.meta_model)
    check_agent_cli(provider)
    auth_mode = resolve_auth_mode(args.auth, provider=provider)
    _print_auth(auth_mode, provider)
    _warn_codex_accounting(provider)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(
        f"running the ship agent with {cfg.meta_model} (real env: it builds, installs, and "
        f"{'opens a PR' if not cfg.is_local else 'edits the working tree'}) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "ship", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            ship_skill(
                cfg=cfg,
                prices=_agent_prices(cfg, model=cfg.meta_model),
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
    print("\nnext: edit config.yaml (repo) and tasks.yaml, then `acumen epoch`")
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

    improve = sub.add_parser(
        "improve",
        help="create or improve the skill from the knowledge wiki",
        description="Read the knowledge wiki (per-task observations/hypotheses distilled from train "
        "runs) and the filtered package source, then create the first skill (when none exist) or "
        "improve the latest into the next version. The held-out valid split is never reachable.",
    )
    improve.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    improve.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    improve.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    improve.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    improve.add_argument("--wiki", type=Path, default=Path("wiki"), help="root of the knowledge wiki")
    improve.add_argument("--from", dest="from_version", metavar="VERSION", help="version to improve (default: latest)")
    improve.add_argument("--model", help="override config meta_model")
    improve.add_argument("--max-turns", type=int, help="cap turns for the improving agent (default: unbounded)")
    improve.add_argument("--max-usd", type=float, help="cap spend for the improving agent (default: unbounded)")
    improve.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    improve.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    _add_auth_arg(improve)
    _add_feedback_arg(
        improve,
        extra=" (e.g. what to fix or emphasise; do NOT paste valid-split answers — that defeats the held-out split)",
    )
    _add_log_args(improve)
    improve.set_defaults(func=_cmd_improve)

    wiki = sub.add_parser(
        "wiki",
        help="distil an arm's train runs into the knowledge wiki (one agent per task)",
        description="For each task, read that arm's train-split runs across models and replicates "
        "and append a terse [version][model] block to wiki/<task>/observations.md and hypothesis.md. "
        "Cumulative and idempotent: an arm already recorded for a task is skipped.",
    )
    wiki.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    wiki.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    wiki.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    wiki.add_argument("--wiki", type=Path, default=Path("wiki"), help="root of the knowledge wiki")
    wiki.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    wiki_arm = wiki.add_mutually_exclusive_group()
    wiki_arm.add_argument("--no-skill", action="store_true", help="record the baseline (noskill) arm")
    wiki_arm.add_argument("--skill", metavar="VERSION", help="record one skill version, e.g. v1 (default: latest)")
    wiki.add_argument("--model", help="override config meta_model")
    wiki.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    wiki.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    wiki.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    _add_auth_arg(wiki)
    _add_log_args(wiki)
    wiki.set_defaults(func=_cmd_wiki)

    epoch = sub.add_parser(
        "epoch",
        help="run one training epoch: bench train, update wiki, improve, bench valid",
        description="One training epoch end to end: bench the current arm on the training signal, "
        "distil it into the wiki, create/improve the skill, then bench the new version on the "
        "held-out valid signal. Fully resumable — re-run to continue a crashed epoch or start the "
        "next one.",
    )
    epoch.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    epoch.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    epoch.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    epoch.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    epoch.add_argument("--wiki", type=Path, default=Path("wiki"), help="root of the knowledge wiki")
    epoch.add_argument("--model", help="override config meta_model")
    epoch.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    epoch.add_argument("--replicates", type=int, help="override config n_replicates")
    epoch.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    epoch.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    epoch.add_argument("--verbose", action="store_true", help="use the full scrolling logs instead of progress bars")
    _add_auth_arg(epoch)
    _add_feedback_arg(epoch, extra=" (passed to the improver; do NOT paste valid-split answers)")
    _add_log_args(epoch)
    epoch.set_defaults(func=_cmd_epoch)

    fit = sub.add_parser(
        "fit",
        help="run many training epochs with early stopping (like training a model)",
        description="Run `acumen epoch` back to back until validation stops improving (patience) or "
        "a hard cap is hit, writing a per-epoch training curve to training.csv and a progress line "
        "each epoch. Fully resumable — re-run to continue.",
    )
    fit.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    fit.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    fit.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    fit.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    fit.add_argument("--wiki", type=Path, default=Path("wiki"), help="root of the knowledge wiki")
    fit.add_argument("--out", type=Path, default=Path("training.csv"), help="training-curve CSV to write")
    fit.add_argument(
        "--patience", type=int, default=2, help="stop after this many epochs with no validation gain (default: 2)"
    )
    fit.add_argument("--max-epochs", type=int, default=10, help="hard cap on epochs (default: 10)")
    fit.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="run exactly this many epochs (disables early stopping and ignores --max-epochs)",
    )
    fit.add_argument("--model", help="override config meta_model")
    fit.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    fit.add_argument("--replicates", type=int, help="override config n_replicates")
    fit.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    fit.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    fit.add_argument("--verbose", action="store_true", help="use the full scrolling logs instead of progress bars")
    _add_auth_arg(fit)
    _add_feedback_arg(fit, extra=" (passed to the improver; do NOT paste valid-split answers)")
    _add_log_args(fit)
    fit.set_defaults(func=_cmd_fit)

    tasks_cmd = sub.add_parser("tasks", help="autonomously generate a tasks.yaml from the target package")
    tasks_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    tasks_cmd.add_argument("--out", type=Path, default=Path("tasks.yaml"), help="tasks.yaml to write")
    tasks_cmd.add_argument(
        "--scripts",
        type=Path,
        default=Path(SCRIPTS_DIRNAME),
        help="directory to keep the reproducer script for each split in, for `acumen check`",
    )
    tasks_cmd.add_argument("--model", help="override config meta_model")
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
    check.add_argument("--model", help="override config meta_model (the review agent)")
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
    ship.add_argument("--model", help="override config meta_model")
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
        ImproveError,
        WikiError,
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
