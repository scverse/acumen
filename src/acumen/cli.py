"""Command-line entry point — a thin shell over the importable API."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from acumen.bench import build_matrix, pending, run_matrix, summarize
from acumen.config import ConfigError, load_config
from acumen.env import DEFAULT_CACHE_ROOT, EnvError, prepare_target
from acumen.paths import SPLITS
from acumen.runner import RunOutcome
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
    parser.add_argument("--no-resume", action="store_true", help="re-run runs that already completed")
    parser.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    parser.add_argument("--keep-sandboxes", action="store_true", help="leave run sandboxes on disk")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit without running agents")


def _progress(total: int) -> Callable[[RunOutcome], None]:
    state = {"done": 0}

    def report(outcome: RunOutcome) -> None:
        state["done"] += 1
        mark = "PASS" if outcome.success else "FAIL"
        key = outcome.key
        print(
            f"[{state['done']}/{total}] {mark} {key.arm}/{key.split}/{key.model}/{key.task_id}/rep_{key.rep}"
            f" ({outcome.reason})",
            flush=True,
        )

    return report


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)

    skill = args.skill  # None => the noskill baseline
    planned = build_matrix(cfg, tasks, skill=skill, splits=args.split or SPLITS, task_ids=args.task)
    todo = pending(planned, args.runs, resume=not args.no_resume)

    arm = "noskill" if skill is None else f"skill_{skill}"
    print(f"arm {arm}: {len(planned)} runs planned, {len(planned) - len(todo)} already complete, {len(todo)} to run")
    if args.dry_run:
        for item in todo:
            k = item.key
            print(f"  {k.arm}/{k.split}/{k.model}/{k.task_id}/rep_{k.rep}")
        return 0
    if not todo:
        return 0

    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)

    outcomes = asyncio.run(
        run_matrix(
            todo,
            target=target,
            runs_root=args.runs,
            max_concurrency=cfg.max_concurrency,
            keep_sandbox=args.keep_sandboxes,
            on_done=_progress(len(todo)),
        )
    )

    passed = sum(1 for o in outcomes if o.success)
    counts = summarize(outcomes)
    breakdown = ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))
    print(f"\n{passed}/{len(outcomes)} passed  ({breakdown})")
    print(f"runs written to {args.runs.resolve()}")
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
    except (ConfigError, TaskError, EnvError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted — completed runs are preserved; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
