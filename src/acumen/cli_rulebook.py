"""Command-line entry points for the rulebook-evolution experiment.

``acumen evolve`` and its support commands live here rather than in :mod:`acumen.cli`, which owns
the epoch/wiki loop. They share its helpers (auth flags, log flags) and are registered by
:func:`add_parsers`, so the two loops evolve independently.

- ``draft`` writes a skill from the package source, optionally under a rulebook's instructions.
- ``lockbox`` carves a write-once hold-out from a task set.
- ``coverage`` / ``warm`` measure what the task set verifies and pre-download its datasets.
- ``evolve`` is the generational rulebook search (see :mod:`acumen.evolve`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from acumen.agents import check_agent_cli, provider_for_model
from acumen.config import Config, load_config
from acumen.coverage import (
    SCRIPT_DIRS,
    inventory_in_venv,
    load_scripts,
    load_task_scripts,
    measure_coverage,
    skill_mentions,
)
from acumen.difficulty import HeadroomSelection
from acumen.draft import draft_skill
from acumen.env import DEFAULT_CACHE_ROOT, Target, prepare_target, resolve_auth_mode
from acumen.evolve import Generation, IslandResult, run_archipelago, run_evolve
from acumen.evolve_analyze import main as _verdict_main
from acumen.evolve_status import main as _status_main
from acumen.folds import FoldError, split_lockbox, write_lockbox
from acumen.logs import LiveLog
from acumen.mining import (
    _KNOWN_ALIASES,
    SEARCH_INTERVAL_S,
    MineResult,
    MiningError,
    _repo_id,
    default_queries,
    mine_github,
    mine_urls,
    submodule_repos,
    write_candidates,
)
from acumen.runner import RunOutcome
from acumen.skills import available_versions, load_skill
from acumen.taskgen import dump_tasks
from acumen.tasks import load_tasks
from acumen.taskshard import ShardOutcome, generate_tasks_sharded
from acumen.transient import TransientLimitError
from acumen.warm import WarmOutcome, collect_dataset_calls, warm_datasets

#: Exit code for "the platform refused work; nothing bad was recorded — rerun the identical command".
#: Distinct from 2 (a configuration error) so an unattended runner can tell wait-and-retry from fix-it.
EXIT_PAUSED = 3


def _auth(args: argparse.Namespace, cfg: Config) -> tuple[str, object]:
    """Resolve the one credential every agent in these commands bills, and name it."""
    from acumen.cli import _effective_auth, _print_auth

    provider = provider_for_model(cfg.meta_model)
    other = {provider_for_model(m) for m in cfg.models} - {provider}
    if other:
        raise SystemExit(
            f"error: meta_model {cfg.meta_model!r} and models {cfg.models} span providers "
            f"({provider}, {', '.join(sorted(other))}); the rulebook loop bills one credential — "
            "use models from a single provider"
        )
    check_agent_cli(provider)
    mode = resolve_auth_mode(_effective_auth(args.auth, cfg.meta_auth), provider=provider)
    _print_auth(mode, provider)
    return provider, mode


def _priced(cfg: Config) -> Config:
    """``cfg`` with rates filled in for every model these commands run, so cost is recorded.

    The agents inside ``draft`` / ``evolve`` build their rate table from ``cfg.prices``, and an empty
    table prices nothing — costs come back unavailable, which would zero the spend figures a run
    reports and projects. Rates are fetched the way ``improve`` and ``tasks`` fetch them, degrade to
    unpriced with a warning if the pricing pages are down, and an operator's own ``prices:`` block
    always wins.
    """
    from acumen.cli import _agent_prices

    table = _agent_prices(cfg, model=cfg.meta_model)
    filled = dict(cfg.prices)
    for model in dict.fromkeys([cfg.meta_model, *cfg.models]):
        rates = table.rates(model)
        if rates is not None:
            filled.setdefault(model, rates)
    return replace(cfg, prices=filled)


def _prepare(args: argparse.Namespace, cfg: Config) -> Target:
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    return target


# ── draft ──────────────────────────────────────────────────────────────────────────────────


def _cmd_draft(args: argparse.Namespace) -> int:
    from acumen.cli import _print_log_result

    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, meta_model=args.model)

    cfg = _priced(cfg)
    existing = available_versions(args.skills)
    if existing and not args.force:
        print(
            f"skills already exist ({', '.join(existing)}) — drafting would add another version. "
            f"Pass --force to draft anyway, or use `acumen improve` to build on {existing[-1]}.",
            file=sys.stderr,
        )
        return 2

    # Drafting the same rulebook into several skills is how a rulebook gets a mean and a spread
    # instead of a single sample: same text, own --skills root per draft.
    rulebook = args.rulebook.read_text() if args.rulebook else None

    _, auth_mode = _auth(args, cfg)
    target = _prepare(args, cfg)
    source = "the package source" if rulebook is None else f"{args.rulebook} + the package source"
    print(f"drafting with {cfg.meta_model} (this reads {source}) ...", flush=True)

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
                rulebook=rulebook,
                rationale="initial draft" if rulebook is None else f"drafted from {args.rulebook}",
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


# ── sharded task generation ────────────────────────────────────────────────────────────────


def cmd_tasks_sharded(args: argparse.Namespace, cfg: Config, target: Target, out: Path, auth_mode, prices) -> int:
    """Run sharded task generation (per notebook, or per mined candidate), streaming progress."""
    shards_dir = args.shards_dir or out.parent / f"{out.stem}.shards"
    unit = "mined candidate" if args.candidates else "notebook"
    print(
        f"generating tasks per {unit} with {cfg.meta_model}, up to {args.max_concurrency or cfg.max_concurrency} "
        f"at once (each reads its {unit} and runs package code) ...",
        flush=True,
    )
    print(f"shards → {shards_dir}   logs → {args.log_dir}", flush=True)

    def on_done(outcome: ShardOutcome) -> None:
        if outcome.status == "cached":
            mark, detail = "·", f"cached, {outcome.n_tasks} tasks"
        elif outcome.status == "generated":
            mark, detail = "✓", f"{outcome.n_tasks} tasks, ${outcome.cost_usd:.2f}"
        else:
            mark, detail = "✗", f"FAILED — {outcome.error}"
        print(f"  {mark} {outcome.slug}: {detail}", flush=True)

    result = asyncio.run(
        generate_tasks_sharded(
            cfg=cfg,
            target=target,
            out_path=out,
            shards_dir=shards_dir,
            auth_mode=auth_mode,
            prices=prices,
            max_turns=args.max_turns,
            max_usd=args.max_usd,
            max_concurrency=args.max_concurrency,
            force=args.force,
            feedback=args.feedback,
            log_dir=args.log_dir,
            stream=args.stream,
            notebook_filter=args.notebook,
            candidates_dir=args.candidates,
            on_shard_done=on_done,
        )
    )
    print(f"\nwrote {result.out_path.resolve()}")
    print(f"  tasks:  {len(result.tasks)} from {result.n_ok}/{len(result.outcomes)} shards")
    if result.n_failed:
        failed = ", ".join(o.slug for o in result.outcomes if o.status == "failed")
        print(f"  failed: {result.n_failed} shards ({failed}) — rerun to retry them", file=sys.stderr)
    print(f"  cost:   ${result.cost_usd:.2f}")
    if result.paused:
        skipped = sum(1 for o in result.outcomes if (o.error or "").startswith("skipped:"))
        print(
            f"\npaused: the platform refused a shard ({result.paused[:160]}); {skipped} shard(s) were skipped and "
            "the merged file holds what landed — rerun the same command to resume them once the limit resets",
            file=sys.stderr,
        )
        return EXIT_PAUSED
    print("\nnext: review the tasks, then `acumen check`")
    return 0


# ── mine ───────────────────────────────────────────────────────────────────────────────────


def _mine(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    target = _prepare(args, cfg)
    # The gate counts only the package's real public API as a "symbol", the same inventory
    # coverage uses — so a mined file is kept for touching real analyses, not for mentioning names.
    inventory = inventory_in_venv(target.python, target.pkg_name).names
    results: list[MineResult] = []
    if not args.no_github:
        queries = args.query or default_queries(target.pkg_name, args.alias)
        # The target itself, its tutorial submodules (already sharded as notebooks), and anything
        # the operator names: not analyses, or already covered.
        own = _repo_id(cfg.repo)
        exclude = [*([own] if own else []), *submodule_repos(target.src_dir), *(args.exclude_repo or [])]
        print(
            f"GitHub code search: {len(queries)} queries x (.ipynb, .py), up to {args.limit} hits each, "
            f"paced at {SEARCH_INTERVAL_S:.0f}s (10 searches/min limit) ...",
            flush=True,
        )
        results.append(
            mine_github(
                target.pkg_name,
                queries,
                exclude_repos=exclude,
                limit=args.limit,
                inventory=inventory,
                min_symbols=args.min_symbols,
                on_progress=lambda msg: print(f"  {msg}", flush=True),
            )
        )
    if args.url:
        print(f"web pages: {len(args.url)} ...", flush=True)
        results.append(mine_urls(target.pkg_name, args.url, inventory=inventory, min_symbols=args.min_symbols))

    written, rejected = write_candidates(results, args.out)
    kept = sum(len(r.kept) for r in results)
    print(f"\nmined {kept} candidate(s) ({len(written)} new) → {args.out.resolve()}")
    for cand in (c for r in results for c in r.kept):
        print(f"  + {cand.slug}  [{cand.kind}] {len(cand.symbols)} symbols  {cand.origin}")
    if rejected:
        print(f"\nrejected {len(rejected)}:")
        for reason, n in MineResult(rejected=rejected).reasons().items():
            print(f"  {n:4d}  {reason}")
    print(f"\nnext: `acumen tasks --candidates {args.out} --out tasks_mined.yaml` to turn them into tasks")
    return 0


def _cmd_mine(args: argparse.Namespace) -> int:
    try:
        return _mine(args)
    except MiningError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


# ── status / verdict ───────────────────────────────────────────────────────────────────────


def _cmd_status(args: argparse.Namespace) -> int:
    return _status_main([str(args.runs)])


def _cmd_verdict(args: argparse.Namespace) -> int:
    return _verdict_main(list(args.arms))


# ── warm / coverage ────────────────────────────────────────────────────────────────────────


def _warm_cache(cfg: Config, target: Target, tasks_path: Path) -> list[WarmOutcome]:
    """Pre-download every dataset the tasks' ground-truth scripts load into the shared cache.

    Runs before a benchmark matrix so concurrent runs never race the same first download. Only
    meaningful when ``config.dataset_cache_dirs`` is set — without the sandbox symlinks the shared
    cache is never seen — so callers gate on that. Reads the same ``scripts/`` dir as coverage.
    """
    scripts = load_task_scripts(tasks_path)
    calls = collect_dataset_calls(scripts, target.pkg_name)
    if not calls:
        print(
            f"warm: no dataset loader calls found beside {tasks_path} ({', '.join(SCRIPT_DIRS)}) — nothing to pre-download"
        )
        return []
    print(f"warming {len(calls)} dataset(s) into {target.datasets_dir} (sequential, once per target):", flush=True)

    def on_done(outcome: WarmOutcome) -> None:
        mark = "✓" if outcome.ok else "✗"
        print(f"  {mark} {outcome.call.source}" + (f"  ({outcome.error})" if not outcome.ok else ""), flush=True)

    return warm_datasets(target.python, target.pkg_name, calls, target.datasets_dir, on_done=on_done)


def _cmd_warm(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    target = _prepare(args, cfg)
    if not cfg.dataset_cache_dirs:
        print(
            "warning: config has no 'dataset_cache_dirs' — sandboxes will not see the shared cache, so "
            "warming it has no effect on benchmark runs. Set e.g. dataset_cache_dirs: [data, cache].",
            file=sys.stderr,
        )
    outcomes = _warm_cache(cfg, target, args.tasks)
    return 1 if any(not o.ok for o in outcomes) else 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    target = _prepare(args, cfg)

    inventory = inventory_in_venv(target.python, target.pkg_name)

    scripts_dir = args.scripts or " + ".join(str(args.tasks.parent / d) for d in SCRIPT_DIRS)
    scripts = load_scripts(args.scripts) if args.scripts else load_task_scripts(args.tasks)
    tasks = load_tasks(args.tasks)
    coverage = measure_coverage(inventory, scripts)

    if args.queue:
        # Machine-friendly: just the uncovered symbols, one per line, for feeding into a targeted
        # generation run's --feedback.
        for name in coverage.uncovered:
            print(name)
        return 0

    covered_n = len(coverage.covered)
    total_n = len(inventory.names)
    print(f"\nAPI coverage of {inventory.package} {inventory.version}: {covered_n}/{total_n} ({coverage.rate:.0%})")
    print(f"  ground-truth scripts read from {scripts_dir} ({len(scripts)} script(s), {len(tasks)} task(s))")
    missing = [t.id for t in tasks if t.id not in scripts]
    if missing:
        print(f"  {len(missing)} task(s) have no confirmation script (contribute no coverage): {', '.join(missing)}")
    if coverage.uncovered:
        print(f"\nuncovered — the generation queue ({len(coverage.uncovered)} symbols):")
        for name in coverage.uncovered:
            print(f"  {name}")
        print("\nfeed these to a targeted run: `acumen coverage --queue | ...` then `acumen tasks --feedback`")
    else:
        print("\nevery inventory symbol is exercised by at least one task — no generation queue")
    if args.skill:
        _print_skill_coverage(inventory, coverage.covered, args.skills, args.skill, target.pkg_name)
    return 0


def _print_skill_coverage(inventory, covered: frozenset[str], skills_root: Path, version: str, pkg: str) -> None:
    """Benchmark coverage beside skill coverage: what is verified, what is merely taught, what is neither.

    The loop can only optimize what the benchmark scores, so any symbol the skill teaches but no task
    exercises is guidance nothing tests — and the leanness pressure has no reason to keep it. This is
    the map of that gap.
    """
    skill = load_skill(skills_root, version)
    mentioned = skill_mentions(inventory, skill.directory, aliases=[_KNOWN_ALIASES.get(pkg, pkg)])
    names = inventory.names
    verified = covered & mentioned
    taught_only = sorted(mentioned - covered)
    tested_only = sorted(covered - mentioned)
    neither = sorted(names - covered - mentioned)
    print(f"\nskill {version} ({skill.size / 1024:.1f} KB) vs the benchmark, over {len(names)} symbols:")
    print(f"  taught AND verified:      {len(verified)}")
    print(f"  taught, never verified:   {len(taught_only)}  {', '.join(taught_only)[:200]}")
    print(f"  verified, not taught:     {len(tested_only)}  {', '.join(tested_only)[:200]}")
    print(f"  neither:                  {len(neither)}")


# ── lockbox ───────────────────────────────────────────────────────────────────────────────


def _cmd_lockbox(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    working_path = args.working or args.tasks.with_name(f"{args.tasks.stem}.working.yaml")
    if working_path.exists() and not args.force:
        print(f"{working_path} already exists — pass --force to overwrite it", file=sys.stderr)
        return 2
    working_ids, lock_ids = split_lockbox([t.id for t in tasks], args.fraction, args.seed)
    by_id = {t.id: t for t in tasks}
    try:
        box = write_lockbox(
            args.out, [by_id[i] for i in lock_ids], seed=args.seed, fraction=args.fraction, dump=dump_tasks
        )
    except FoldError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    working_path.write_text(dump_tasks([by_id[i] for i in working_ids]))
    print(f"lockbox: {len(lock_ids)} task(s) held back → {box.directory.resolve()}  ({box.digest[:19]}…)")
    print(f"working: {len(working_ids)} task(s) → {working_path.resolve()}")
    print("\nnext: `acumen evolve --tasks <working> --lockbox <lockbox>`; the lockbox is scored once, at the end")
    return 0


# ── evolve ────────────────────────────────────────────────────────────────────────────────


def _evolve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    cfg = _priced(cfg)
    _, auth_mode = _auth(args, cfg)
    target = _prepare(args, cfg)
    if cfg.dataset_cache_dirs and not args.no_warm:
        _warm_cache(cfg, target, args.tasks)
    print(
        f"EVOLVE: {args.generations} generation(s); screen {args.screen_size} task(s), rotated every "
        f"{args.epoch_len} generation(s); accept at >= +{args.accept_delta} screen passes; full-bench "
        f"confirmation every {args.confirm_every} accept(s); {args.drafts} lockbox draft(s)/version"
        + ("; NO LOCKBOX" if args.no_lockbox else f"; lockbox {args.lockbox}"),
        flush=True,
    )

    def on_done(outcome: RunOutcome) -> None:
        mark = "✓" if outcome.success else "✗"
        k = outcome.key
        print(f"  {mark} {k.split}/{k.task_id}/rep_{k.rep} ({outcome.reason})", flush=True)

    def on_select(selection: HeadroomSelection) -> None:
        print(
            f"headroom selection: kept {len(selection.selected)} task(s); "
            f"{len(selection.solved)} baseline-solved and {len(selection.unscreened)} unscreened left out",
            flush=True,
        )

    def on_generation(gen: Generation) -> None:
        ch, c = gen.champion_screen, gen.candidate_screen
        verdict = "ACCEPT" if gen.accepted else "reject"
        if gen.confirm_ran:
            verdict += " -> " + ("CONFIRMED on full bench" if gen.confirm_promoted else "REVERTED on full bench")
        print(f"\n=== generation {gen.index}: {gen.parent} -> {gen.candidate}  [{verdict}] ===")
        print(f"  directive: {gen.directive}")
        print(
            f"  screen ({len(gen.subset)} tasks): champion {ch.passed}/{ch.total} vs candidate "
            f"{c.passed}/{c.total};  confirmed champion: {gen.confirmed}",
            flush=True,
        )

    def print_drafts(baseline, drafts) -> None:
        shown = {ds.version: ds for ds in (baseline, drafts)}
        for ds in shown.values():
            per = "  ".join(
                f"d{i} {s.passed}/{s.total} ({size / 1024:.1f}KB)"
                for i, (s, size) in enumerate(zip(ds.scores, ds.sizes, strict=True), start=1)
            )
            print(f"  lockbox drafts of {ds.version}: {per}   mean {ds.mean_rate:.0%}, spread {ds.spread:.0%}")

    if args.islands:

        def on_island(isl: IslandResult) -> None:
            print(
                f"\n--- island {isl.index} done: champion {isl.run.champion} "
                f"({isl.run.accepted}/{len(isl.run.generations)} generation(s) accepted, "
                f"{len(isl.task_ids)} task(s)) ---",
                flush=True,
            )

        run = asyncio.run(
            run_archipelago(
                cfg=cfg,
                target=target,
                skills_root=args.skills,
                rulebooks_root=args.rulebooks,
                runs_root=args.runs,
                tasks=tasks,
                k=args.islands,
                generations=args.generations,
                screen_size=args.screen_size,
                screen_drafts=args.screen_drafts,
                epoch_len=args.epoch_len,
                accept_delta=args.accept_delta,
                confirm_every=args.confirm_every,
                n_drafts=args.drafts,
                seed=args.seed,
                lockbox_dir=None if args.no_lockbox else args.lockbox,
                allow_no_lockbox=args.no_lockbox,
                auth_mode=auth_mode,
                feedback=args.feedback,
                log_dir=args.log_dir,
                stream=args.stream,
                headroom_only=args.headroom,
                max_wallclock_s=args.max_hours * 3600 if args.max_hours else None,
                on_select=on_select,
                on_island=on_island,
                on_generation=on_generation,
                on_bench_done=on_done,
            )
        )
        print(f"\n=== archipelago finished: {args.islands} island(s), merged into {run.merged.version} ===")
        for isl in run.islands:
            print(
                f"  island {isl.index}: champion {isl.run.champion} "
                f"({isl.run.accepted}/{len(isl.run.generations)} accepted, {len(isl.task_ids)} task(s))"
            )
        print(f"  meta-rules (the merge's rationale):\n{run.merged.rationale}")
        st, mt = run.seed_test, run.merged_test
        print(
            f"  full working test (cross-island validation): seed {st.passed}/{st.total} ({st.rate:.0%}) -> "
            f"merged {mt.passed}/{mt.total} ({mt.rate:.0%})"
        )
        if run.lockbox_drafts is not None and run.lockbox_baseline_drafts is not None:
            print_drafts(run.lockbox_baseline_drafts, run.lockbox_drafts)
            print(f"  LOCKBOX mean Δ (seed -> merged over {args.drafts} draft(s)): {run.lockbox_mean_delta:+.0%}")
        else:
            print("  lockbox not opened — no generalisation claim from this run", file=sys.stderr)
        return 0

    run = asyncio.run(
        run_evolve(
            cfg=cfg,
            target=target,
            skills_root=args.skills,
            rulebooks_root=args.rulebooks,
            runs_root=args.runs,
            tasks=tasks,
            generations=args.generations,
            screen_size=args.screen_size,
            screen_drafts=args.screen_drafts,
            epoch_len=args.epoch_len,
            accept_delta=args.accept_delta,
            confirm_every=args.confirm_every,
            n_drafts=args.drafts,
            seed=args.seed,
            lockbox_dir=None if args.no_lockbox else args.lockbox,
            allow_no_lockbox=args.no_lockbox,
            auth_mode=auth_mode,
            feedback=args.feedback,
            log_dir=args.log_dir,
            stream=args.stream,
            headroom_only=args.headroom,
            max_wallclock_s=args.max_hours * 3600 if args.max_hours else None,
            on_select=on_select,
            on_generation=on_generation,
            on_bench_done=on_done,
        )
    )

    print(f"\n=== evolution finished: {run.stopped_because} ===")
    print(f"  champion: rulebook {run.champion}  ({run.accepted}/{len(run.generations)} generation(s) accepted)")
    print(f"  journal: {run.journal_path}")
    if run.lockbox_drafts is not None and run.lockbox_baseline_drafts is not None:
        print_drafts(run.lockbox_baseline_drafts, run.lockbox_drafts)
        print(f"  LOCKBOX mean Δ (seed -> champion over {args.drafts} draft(s)): {run.lockbox_mean_delta:+.0%}")
    else:
        print("  lockbox not opened — no generalisation claim from this run", file=sys.stderr)
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    try:
        return _evolve(args)
    except TransientLimitError as err:
        print(f"\npaused: {err}", file=sys.stderr)
        print("nothing bad was recorded — wait, then rerun the identical command to resume", file=sys.stderr)
        return EXIT_PAUSED


def add_tasks_args(tasks_cmd: argparse.ArgumentParser) -> None:
    """Add the sharded-generation flags to ``acumen tasks``."""
    tasks_cmd.add_argument(
        "--per-notebook",
        action="store_true",
        help="shard generation: one agent per tutorial notebook, resumable, merged at the end",
    )
    tasks_cmd.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="where per-shard files live (default: <out>.shards beside --out); delete a shard file to redo it",
    )
    tasks_cmd.add_argument(
        "--notebook",
        metavar="SUBSTR",
        action="append",
        help="with sharded generation: only shards whose path contains SUBSTR (repeatable)",
    )
    tasks_cmd.add_argument(
        "--candidates",
        type=Path,
        metavar="DIR",
        help="shard over the mined candidate scripts in DIR instead of notebooks",
    )
    tasks_cmd.add_argument(
        "--max-concurrency", type=int, help="with sharded generation: override config max_concurrency"
    )


def add_parsers(sub: argparse._SubParsersAction) -> None:
    """Register the rulebook-experiment subcommands on the main parser."""
    from acumen.cli import _add_auth_arg, _add_feedback_arg, _add_log_args

    draft = sub.add_parser("draft", help="draft a skill from the target package's source")
    draft.add_argument(
        "--rulebook",
        type=Path,
        help="draft under the instructions in this rulebook file (from a rulebooks root, or "
        "`acumen evolve`); repeat into separate --skills roots to score a rulebook as a "
        "mean over drafts instead of a single sample",
    )
    draft.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    draft.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    draft.add_argument("--model", help="override config meta_model")
    draft.add_argument("--max-turns", type=int, help="cap turns for the drafting agent (default: unbounded)")
    draft.add_argument("--max-usd", type=float, help="cap spend for the drafting agent (default: unbounded)")
    draft.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    draft.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    draft.add_argument("--force", action="store_true", help="draft another version even if some already exist")
    _add_auth_arg(draft, default=None)
    _add_feedback_arg(draft, extra=" (e.g. package context, what the skill should emphasise)")
    _add_log_args(draft)
    draft.set_defaults(func=_cmd_draft)

    mine = sub.add_parser(
        "mine",
        help="harvest real analyses that use the target (GitHub code search, given web pages) into candidate scripts",
    )
    mine.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    mine.add_argument("--out", type=Path, default=Path("mined"), help="directory of candidate scripts (additive)")
    mine.add_argument(
        "--query", action="append", help="a GitHub code-search query (repeatable; default: import + namespaces)"
    )
    mine.add_argument("--alias", help="the package's conventional import alias for default queries (e.g. sq)")
    mine.add_argument("--limit", type=int, default=100, help="max hits per query per file type (default 100)")
    mine.add_argument("--url", action="append", help="a page/notebook/script URL to lift code from (repeatable)")
    mine.add_argument("--no-github", action="store_true", help="skip GitHub; only the given --url pages")
    mine.add_argument(
        "--exclude-repo",
        metavar="OWNER/NAME",
        action="append",
        help="skip hits from this repository (repeatable; the target and its submodules are always skipped)",
    )
    mine.add_argument(
        "--min-symbols", type=int, default=1, help="public API symbols a candidate must reference (default 1)"
    )
    mine.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    mine.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    mine.set_defaults(func=_cmd_mine)

    status = sub.add_parser(
        "status",
        help="one-screen health check of a running or finished experiment (exit 1 = a PROBLEM line)",
    )
    status.add_argument("runs", type=Path, nargs="?", default=Path("runs"), help="run tree (default: runs)")
    status.set_defaults(func=_cmd_status)

    verdict = sub.add_parser(
        "verdict",
        help="turn run trees into a verdict: mean +- spread over drafts, paired sign tests, hard/easy split",
        description="The first LABEL=RUNDIR is the baseline everything is compared against. Drafts of one "
        "rulebook share a label before a dot (v7.d1, v7.d2, ...) and are pooled.",
    )
    verdict.add_argument("arms", nargs="+", metavar="LABEL=RUNDIR", help="an arm and its run directory")
    verdict.set_defaults(func=_cmd_verdict)

    coverage_cmd = sub.add_parser(
        "coverage",
        help="report which of the target's public API symbols the task set exercises (the rest are the queue)",
    )
    coverage_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    coverage_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    coverage_cmd.add_argument(
        "--scripts", type=Path, help="dir of ground-truth scripts (default: 'scripts/' and 'tasks/' beside tasks.yaml)"
    )
    coverage_cmd.add_argument("--queue", action="store_true", help="print only the uncovered symbols, one per line")
    coverage_cmd.add_argument(
        "--skill", metavar="VERSION", help="also report what this skill version teaches vs what the benchmark verifies"
    )
    coverage_cmd.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    coverage_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    coverage_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    coverage_cmd.set_defaults(func=_cmd_coverage)

    warm_cmd = sub.add_parser(
        "warm",
        help="pre-download the datasets the tasks' ground-truth scripts load into the shared per-target cache",
    )
    warm_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    warm_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    warm_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    warm_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    warm_cmd.set_defaults(func=_cmd_warm)

    lockbox_cmd = sub.add_parser(
        "lockbox",
        help="hold back a fraction of tasks, written once, never optimized on, scored once at the very end",
    )
    lockbox_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="the full task set")
    lockbox_cmd.add_argument("--out", type=Path, default=Path("lockbox"), help="lockbox directory to create")
    lockbox_cmd.add_argument(
        "--working", type=Path, help="where to write the remaining (working) tasks (default: <tasks>.working.yaml)"
    )
    lockbox_cmd.add_argument("--fraction", type=float, default=0.2, help="fraction of tasks to hold back (0.2)")
    lockbox_cmd.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    lockbox_cmd.add_argument("--force", action="store_true", help="overwrite an existing working file")
    lockbox_cmd.set_defaults(func=_cmd_lockbox)

    evolve = sub.add_parser(
        "evolve",
        help="generational rulebook evolution: improve-from-best under rotating exploration "
        "directives, screened on rotating task subsets, ratcheted by full-bench confirmation",
    )
    evolve.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    evolve.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    evolve.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    evolve.add_argument("--rulebooks", type=Path, default=Path("rulebooks"), help="root of the rulebook tree")
    evolve.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    evolve.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    evolve.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    evolve.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    evolve.add_argument("--no-warm", action="store_true", help="skip pre-downloading datasets into the shared cache")
    evolve.add_argument(
        "--headroom",
        action="store_true",
        help="evolve only on tasks the no-skill baseline does not already pass on the test split "
        "(per config model; needs prior `bench --no-skill` runs — unscreened tasks are left out)",
    )
    evolve.add_argument("--generations", type=int, default=20, help="generation budget (default 20; per island)")
    evolve.add_argument(
        "--islands",
        type=int,
        metavar="K",
        help="evolve K independent islands on disjoint task partitions, then cross-pollinate the "
        "champions into one merged rulebook (meta-rules = edits that replicated across islands)",
    )
    evolve.add_argument(
        "--screen-size", type=int, default=12, help="tasks per screen subset (cheap, noisy tier; default 12)"
    )
    evolve.add_argument(
        "--screen-drafts",
        type=int,
        default=1,
        help="score each version on the screen over N independent drafts (default 1). Drafting is "
        "the noisiest step — the same rulebook drafted twice differs by ~8pp — and that noise is a "
        "per-skill offset more screen tasks cannot average away. The accept bar scales with N, so "
        "--accept-delta keeps meaning passes per draft",
    )
    evolve.add_argument(
        "--epoch-len", type=int, default=5, help="generations before the screen subset rotates (default 5)"
    )
    evolve.add_argument(
        "--accept-delta",
        type=int,
        default=2,
        help="screen passes a candidate must beat the champion by (the noise floor; default 2)",
    )
    evolve.add_argument(
        "--confirm-every",
        type=int,
        default=5,
        help="screen accepts between full-bench confirmations of the champion (the ratchet; default 5)",
    )
    evolve.add_argument(
        "--drafts",
        type=int,
        default=3,
        help="independent skill drafts per version for the final lockbox verdict (default 3)",
    )
    evolve.add_argument("--seed", type=int, default=0, help="screen-subset rotation seed (default 0)")
    evolve.add_argument("--max-hours", type=float, help="wall-clock cap, checked between generations")
    evolve.add_argument("--lockbox", type=Path, default=Path("lockbox"), help="lockbox dir from `acumen lockbox`")
    evolve.add_argument(
        "--no-lockbox", action="store_true", help="run without a lockbox (no generalisation claim possible)"
    )
    _add_auth_arg(evolve, default=None)
    _add_feedback_arg(evolve)
    _add_log_args(evolve)
    evolve.set_defaults(func=_cmd_evolve)
