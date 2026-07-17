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
from acumen.draft import DraftError, draft_skill
from acumen.env import DEFAULT_CACHE_ROOT, EnvError, prepare_target
from acumen.paths import SPLITS
from acumen.report import ReportError, build_report
from acumen.runner import RunOutcome
from acumen.skills import SkillError, available_versions, load_skill
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

    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)

    outcomes = asyncio.run(
        run_matrix(
            todo,
            target=target,
            runs_root=args.runs,
            max_concurrency=cfg.max_concurrency,
            skill=skill,
            keep_sandbox=args.keep_sandboxes,
            on_done=_progress(len(todo)),
        )
    )

    passed = sum(1 for o in outcomes if o.success)
    counts = summarize(outcomes)
    breakdown = ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))
    print(f"\n{passed}/{len(outcomes)} passed  ({breakdown})")

    # The comparison is only meaningful if the skill actually reached the agent, so say
    # so rather than leaving it to be discovered later in the transcripts (§7.4).
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

    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(f"drafting with {cfg.draft_model} (this reads the package source) ...", flush=True)

    result = asyncio.run(
        draft_skill(
            cfg=cfg,
            target=target,
            skills_root=args.skills,
            max_turns=args.max_turns,
            max_usd=args.max_usd,
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
    print(f"\nnext: acumen bench --skill {skill.version}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks) if args.tasks.exists() else None
    if tasks is None:
        print(f"note: {args.tasks} not found — per-task prompts will be omitted", file=sys.stderr)
    report = build_report(args.runs, args.out, tasks)
    df = report.results
    arms = ", ".join(sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)))
    print(f"aggregated {report.n_runs} runs across arms: {arms}")
    for arm in sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)):
        group = df[df["arm_label"] == arm]
        print(f"  {arm}: {int(group['success'].sum())}/{len(group)} passed")
    print(f"wrote {args.out.resolve()}")
    print(f"wrote {args.out.resolve().with_suffix('.csv')}")
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
    draft.set_defaults(func=_cmd_draft)

    report = sub.add_parser("report", help="aggregate the run tree into a self-contained report.html")
    report.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    report.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml (for task text)")
    report.add_argument("--out", type=Path, default=Path("report.html"), help="output HTML path (overwritten)")
    report.set_defaults(func=_cmd_report)
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
    except (ConfigError, TaskError, EnvError, SkillError, DraftError, ReportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted — completed runs are preserved; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
