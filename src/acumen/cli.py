"""Command-line entry point — a thin shell over the importable API."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

from acumen.bench import build_matrix, pending, run_matrix, summarize
from acumen.config import ConfigError, load_config
from acumen.draft import DraftError, draft_skill
from acumen.env import DEFAULT_CACHE_ROOT, AuthMode, EnvError, prepare_target, resolve_auth_mode
from acumen.improve import ImproveError, improve_skill
from acumen.logs import LiveLog
from acumen.paths import SPLITS
from acumen.report import ReportError, build_report
from acumen.runner import RunOutcome, StderrFilter
from acumen.scaffold import InitError, scaffold
from acumen.ship import ShipError, ship_skill
from acumen.skills import SkillError, available_versions, latest_version, load_skill
from acumen.taskgen import TaskGenError, generate_tasks
from acumen.tasks import TaskError, load_tasks


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    arm = parser.add_mutually_exclusive_group()
    arm.add_argument("--no-skill", action="store_true", help="run the baseline arm (the default)")
    arm.add_argument("--skill", metavar="VERSION", help="run with a skill version, e.g. v1")
    parser.add_argument("--split", choices=SPLITS, action="append", help="restrict to a split (repeatable)")
    parser.add_argument("--task", metavar="ID", action="append", help="restrict to a task id (repeatable)")
    parser.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    parser.add_argument("--replicates", type=int, help="override config n_replicates")
    parser.add_argument("--no-resume", action="store_true", help="re-run runs that already completed")
    parser.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    parser.add_argument("--keep-sandboxes", action="store_true", help="leave run sandboxes on disk")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    parser.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit without running agents")


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
    """Add the ``--auth`` flag to a single-agent subcommand.

    These commands spawn one agent, so they default to the Claude subscription ("session")
    when a subscription login is present and fall back to the API key otherwise. ``bench`` has
    no such flag — it always bills the API so its recorded ``cost_usd`` is real.
    """
    parser.add_argument(
        "--auth",
        choices=("auto", "session", "api"),
        default="auto",
        help="which credential to bill: 'session' (Claude subscription), 'api' (Anthropic API), "
        "or 'auto' (default: session if you're logged in, else the API)",
    )


def _print_auth(mode: AuthMode) -> None:
    """Report which credential the run will bill, so the choice is never silent."""
    label = "Claude subscription (session)" if mode == "session" else "API key"
    print(f"auth: {label}", flush=True)


def _print_log_result(log: LiveLog) -> None:
    """Print where the rendered HTML log landed, once a run has finalized."""
    if log.html_rendered:
        print(f"log → {log.html_path}")
    else:
        print("note: HTML log not rendered (claude-code-log missing?) — the jsonl log is complete", file=sys.stderr)


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
        mark = "✓ pass" if outcome.success else "✗ FAIL"
        p = outcome.payload
        toks = int(p.get("input_tokens", 0)) + int(p.get("output_tokens", 0))
        cost = float(p.get("cost_usd", 0.0))
        dur = _fmt_secs(float(p.get("duration_s", 0.0)))
        stats = f"{_fmt_tokens(toks)}tok ${cost:.2f} {dur}"
        print(
            f"[{self._stamp()}] {mark} {_key_label(outcome.key)}"
            f"  ({outcome.reason})  {stats}"
            f"  [{self.done}/{self.total} done, {self.passed} passed]",
            flush=True,
        )


def _fmt_tokens(value: int) -> str:
    """Compact token count, e.g. ``118k`` / ``1.2M``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return str(value)


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.replicates:
        cfg = replace(cfg, n_replicates=args.replicates)

    version = args.skill  # None => the noskill baseline
    skill = None
    if version is not None:
        skill = load_skill(args.skills, version, expect_name=cfg.skill_name)

    planned = build_matrix(cfg, tasks, skill=version, splits=args.split or SPLITS, task_ids=args.task)
    todo = pending(planned, args.runs, resume=not args.no_resume)

    arm = "noskill" if version is None else f"skill_{version}"
    print(f"arm {arm}: {len(planned)} runs planned, {len(planned) - len(todo)} already complete, {len(todo)} to run")
    if skill is not None:
        print(f"skill {skill.version}: {skill.name} ({skill.hash[:19]}…)")
    if args.dry_run:
        for item in todo:
            k = item.key
            print(f"  {k.arm}/{k.split}/{k.model}/{k.task_id}/rep_{k.rep}")
        return 0
    if not todo:
        return 0

    # bench records real per-run cost, so it must bill the API — never the subscription.
    auth_mode = resolve_auth_mode("api", allow_session=False)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)

    print(f"running {len(todo)} runs, up to {cfg.max_concurrency} at a time:", flush=True)
    progress = _Progress(len(todo))
    outcomes = asyncio.run(
        run_matrix(
            todo,
            target=target,
            runs_root=args.runs,
            max_concurrency=cfg.max_concurrency,
            auth_mode=auth_mode,
            skill=skill,
            keep_sandbox=args.keep_sandboxes,
            stderr=StderrFilter(),
            on_start=progress.on_start,
            on_done=progress.on_done,
            env_passthrough=cfg.env_passthrough,
        )
    )

    passed = sum(1 for o in outcomes if o.success)
    counts = summarize(outcomes)
    breakdown = ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))
    total_cost = sum(float(o.payload.get("cost_usd", 0.0)) for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} passed in {_fmt_secs(progress.elapsed)}  (${total_cost:.2f}, {breakdown})")

    # The comparison is only meaningful if the skill actually reached the agent, so say
    # so rather than leaving it to be discovered later in the transcripts.
    loaded = sum(1 for o in outcomes if o.payload.get("skill_loaded"))
    if skill is not None:
        print(f"skill loaded in {loaded}/{len(outcomes)} runs")
        if loaded == 0:
            print(
                "warning: the skill never loaded — this arm is not measuring the skill",
                file=sys.stderr,
            )
    elif loaded:
        print(f"warning: the Skill tool fired in {loaded} baseline runs", file=sys.stderr)
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

    auth_mode = resolve_auth_mode(args.auth, allow_session=True)
    _print_auth(auth_mode)
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
    print(f"  cost:        ${result.cost_usd:.2f} over {result.turns} turns")
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

    auth_mode = resolve_auth_mode(args.auth, allow_session=True)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    log = LiveLog.open(args.log_dir, "improve", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            improve_skill(
                cfg=cfg,
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
    print(f"  cost:        ${result.cost_usd:.2f} over {result.turns} turns")
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
    # Fail before the (costly) target prep and agent run if we'd have to clobber.
    if out.exists() and not args.force:
        print(
            f"{out} already exists — pass --force to overwrite it (e.g. the placeholder from `acumen init`)",
            file=sys.stderr,
        )
        return 2

    auth_mode = resolve_auth_mode(args.auth, allow_session=True)
    _print_auth(auth_mode)
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
                target=target,
                out_path=out,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                force=args.force,
                feedback=args.feedback,
                log=log,
            )
        )
    print(f"\nwrote {result.out_path.resolve()}")
    print(f"  tasks: {len(result.tasks)} ({', '.join(t.id for t in result.tasks)})")
    print(f"  cost:  ${result.cost_usd:.2f} over {result.turns} turns")
    _print_log_result(log)
    print("\nnext: review the tasks, then `acumen draft` and `acumen bench`")
    return 0


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
    auth_mode = resolve_auth_mode(args.auth, allow_session=True)
    _print_auth(auth_mode)
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
    print(f"  cost:  ${result.cost_usd:.2f} over {result.turns} turns")
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


def build_parser() -> argparse.ArgumentParser:
    """Build the ``acumen`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="acumen", description="Build, benchmark, and optimize Claude skills for Python packages."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="run a benchmark pass")
    _add_bench_args(bench)
    bench.set_defaults(func=_cmd_bench)

    draft = sub.add_parser("draft", help="draft a skill from the target package's source")
    draft.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    draft.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    draft.add_argument("--model", help="override config draft_model")
    draft.add_argument("--max-turns", type=int, help="override config max_turns for the drafting agent")
    draft.add_argument("--max-usd", type=float, help="override config max_usd for the drafting agent")
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
    improve.add_argument("--max-turns", type=int, help="override config max_turns for the improving agent")
    improve.add_argument("--max-usd", type=float, help="override config max_usd for the improving agent")
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
        ConfigError,
        TaskError,
        EnvError,
        SkillError,
        DraftError,
        ImproveError,
        TaskGenError,
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
