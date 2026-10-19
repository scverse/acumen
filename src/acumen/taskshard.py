"""Sharded task generation: one agent per notebook, or per mined real-world analysis.

:func:`acumen.taskgen.generate_tasks` has one agent enumerate and cover a whole package in a single
context, which serializes the package into one agent's stamina and loses everything on a single
error. This module fans the same generation out instead: the harness enumerates the shards (a
target's tutorial notebooks, or a directory of mined candidate scripts — see :mod:`acumen.mining`),
runs one agent per shard behind a semaphore, and merges the shard files into one ``tasks.yaml``.

Everything is resumable by file presence. A shard whose ``<slug>.yaml`` already parses is reused; a
shard that fails is recorded and skipped rather than taking the pass down; a platform limit pauses
the remaining shards rather than failing them one by one against the same wall. To redo one shard,
delete its file.

The generation agent is the same isolated one :func:`~acumen.taskgen.generate_tasks` runs — same
filtered source copy, same skill-bias guard, same prompt rules — only the scope section of the
prompt differs (:func:`acumen.prompts.taskgen_shard_prompt`, :func:`acumen.prompts.taskgen_mined_prompt`).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.check import SCRIPTS_DIRNAME, script_path
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import SPLITS, slugify
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import taskgen_mined_prompt, taskgen_shard_prompt
from acumen.scrub import build_filtered_source, make_skill_guard
from acumen.taskgen import TASKS_FILE, TaskGenError, _validate_generated, dump_tasks, harvest_scripts
from acumen.tasks import Task, TaskError, load_tasks, parse_tasks
from acumen.transient import is_transient

#: Directories under a checkout that hold ``.ipynb`` files that are not tutorials.
_NOTEBOOK_SKIP_DIRS = frozenset({".ipynb_checkpoints"})


def notebook_shards(source: Path) -> list[Path]:
    """Return every tutorial notebook under ``source``, as paths relative to it, sorted.

    Enumeration happens here in the harness — not inside an agent — so the fan-out, the per-shard
    resume and the merge are all driven from a list the code controls rather than one an agent
    rediscovers each run. Jupyter checkpoint copies are skipped: they are stale duplicates.
    """
    return [
        nb.relative_to(source) for nb in sorted(source.rglob("*.ipynb")) if not (set(nb.parts) & _NOTEBOOK_SKIP_DIRS)
    ]


def _shard_slug(notebook: Path) -> str:
    """A filesystem-safe, collision-resistant slug for a notebook shard.

    Built from the notebook's path *relative to the source root*, not just its basename, so two
    notebooks named the same in different galleries get distinct shards, shard files and task-id
    prefixes. It is the shard file's stem and the task-id namespace, so it must round-trip.
    """
    return slugify("-".join(notebook.with_suffix("").parts))


def _namespace_task(task: Task, slug: str) -> Task:
    """Prefix a task's id with its shard slug, so merged ids are unique and traceable.

    Each shard's agent picks ids blind to the other shards, so two shards can independently pick the
    same id. The merge re-validates the result regardless.
    """
    return replace(task, id=f"{slug}__{task.id}")


@dataclass(frozen=True)
class ShardOutcome:
    """The outcome of one shard within a sharded generation pass."""

    #: Notebook path relative to the source root (or the candidate's file name).
    notebook: Path
    slug: str
    shard_path: Path
    #: ``"generated"`` (an agent ran and wrote it now), ``"cached"`` (a valid shard file was already
    #: present and reused), or ``"failed"`` (the agent errored or wrote nothing valid).
    status: str
    n_tasks: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    error: str | None = None
    log_jsonl: Path | None = None
    log_html: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether this shard contributed tasks (freshly generated or reused from cache)."""
        return self.status in ("generated", "cached")


@dataclass(frozen=True)
class ShardedResult:
    """The outcome of a sharded generation pass: the merged task set plus per-shard detail."""

    tasks: list[Task]
    out_path: Path
    shards_dir: Path
    outcomes: list[ShardOutcome]
    cost_usd: float
    #: Set when the platform refused a shard (limit, overload, network) and the rest were skipped:
    #: the merged file holds what landed, and a rerun resumes the skipped shards.
    paused: str | None = None

    @property
    def n_ok(self) -> int:
        """How many shards contributed tasks."""
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def n_failed(self) -> int:
        """How many shards failed."""
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")


def merge_shards(shards_dir: Path, out_path: Path) -> list[Task]:
    """Merge every ``<slug>.yaml`` shard in ``shards_dir`` into one validated ``tasks.yaml``.

    Each shard's task ids are prefixed with the shard slug (the file's stem) so the union has unique,
    traceable ids, then the whole set is round-tripped through the strict loader — a merged file the
    rest of the pipeline would reject can never be written. A shard's reproducer scripts
    (``<slug>/<id>-<split>.py``) are copied under the same namespaced id, so ``acumen check`` and
    ``acumen coverage`` find them. Cheap and idempotent: re-runnable after any subset of shards
    regenerates, which is what makes the whole pass resumable.

    Raises
    ------
    TaskGenError
        If no shard files hold any task.
    """
    merged: list[Task] = []
    scripts: dict[Path, str] = {}
    scripts_root = out_path.parent / SCRIPTS_DIRNAME
    for shard_file in sorted(shards_dir.glob("*.yaml")):
        slug = shard_file.stem
        for task in load_tasks(shard_file):
            namespaced = _namespace_task(task, slug)
            merged.append(namespaced)
            for split in SPLITS:
                source = script_path(shards_dir / slug, task.id, split)
                if source.is_file():
                    scripts[script_path(scripts_root, namespaced.id, split)] = source.read_text()
    if not merged:
        raise TaskGenError(f"no tasks found across shards in {shards_dir} — nothing to merge")
    # Round-trip through dump + strict parse to enforce cross-shard uniqueness before writing.
    tasks = parse_tasks(yaml.safe_load(dump_tasks(merged)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_tasks(tasks))
    for path, text in scripts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tasks


def _cached_shard(notebook: Path, slug: str, shard_path: Path) -> ShardOutcome | None:
    """Return a ``"cached"`` outcome if ``shard_path`` already holds a valid shard, else ``None``.

    Resume is by file presence. A file that is present but *unparseable* is treated as absent, so a
    truncated write from an interrupted run is regenerated rather than trusted.
    """
    if not shard_path.is_file():
        return None
    try:
        tasks = load_tasks(shard_path)
    except TaskError:
        return None
    return ShardOutcome(notebook=notebook, slug=slug, shard_path=shard_path, status="cached", n_tasks=len(tasks))


async def _run_shard_agent(
    *,
    work_root: Path,
    source_copy: Path,
    target: Target,
    make_prompt: Callable[[Path, Path], str],
    cfg: Config,
    model: str,
    max_turns: int | None,
    max_usd: float | None,
    auth_mode: AuthMode,
    prices: PriceTable,
    log: LiveLog | None,
    seed_files: Mapping[str, str] | None = None,
) -> tuple[list[Task], AgentResult, float]:
    """Run one generation agent in ``work_root``; return its validated tasks, result and cost.

    The isolation-critical core, matching :func:`acumen.taskgen.generate_tasks`: a throwaway ``HOME``
    and config dir under ``work_root``, the scrubbed env, the skill-bias guard against
    ``target.src_dir``, and read access to the shared *filtered* ``source_copy`` rather than the real
    checkout. The caller owns ``work_root`` and its teardown. ``make_prompt(staged, scripts_dir)``
    builds the prompt; ``seed_files`` (``{name: text}``) are written into the work dir first — how a
    mined candidate script reaches its agent without living in the package source copy.

    Raises
    ------
    TaskGenError
        If the agent failed, produced no result, errored, or wrote no valid ``tasks.yaml``.
    """
    work = work_root / "work"
    home = work_root / "home"
    provider = provider_for_model(model)
    config_dir = home / (".claude" if provider == "claude" else ".codex")
    staged_scripts = work / SCRIPTS_DIRNAME
    for path in (work, home, config_dir, home / "tmp", staged_scripts):
        path.mkdir(parents=True, exist_ok=True)
    staged = work / TASKS_FILE
    for name, text in (seed_files or {}).items():
        (work / name).write_text(text)

    env = label_env(
        build_agent_env(
            config_dir=config_dir,
            home=home,
            extra_path=[target.bin_dir],
            auth_mode=auth_mode,
            extra_allow=cfg.env_passthrough,
            provider=provider,
        ),
        work_root,
    )
    options = AgentOptions(
        cwd=work,
        env=env,
        model=model,
        max_turns=max_turns,
        max_usd=max_usd,
        price_usd=pricer(model, prices),
        read_dirs=(source_copy, target.venv_dir),
        write_dirs=(work,),
        discover_skills=False,
        claude_hooks=(
            {"PreToolUse": [make_skill_guard(target.src_dir, exempt=(work,))]} if provider == "claude" else None
        ),
        deny_paths=(target.src_dir.resolve(),),
    )

    result: AgentResult | None = None
    agent_error: Exception | None = None
    try:
        result = await run_agent(
            make_prompt(staged, staged_scripts),
            options=options,
            on_event=log.append if log is not None else None,
        )
    except Exception as err:  # noqa: BLE001 - a failed generation is an error to report, re-raised below
        agent_error = err
    finally:
        if log is not None:
            log.finalize(config_dir=config_dir, work_dir=work, result=result)

    if agent_error is not None:
        raise TaskGenError(
            f"the task-generation agent failed: {type(agent_error).__name__}: {agent_error}"
        ) from agent_error
    if result is None:
        raise TaskGenError("the task-generation agent produced no result message")
    if result.is_error:
        raise TaskGenError(f"the task-generation agent errored: {result.subtype} {result.errors or ''}".strip())

    tasks = _validate_generated(staged)
    cost = (
        resolve_cost(
            result.total_cost_usd,
            price_usage(result.usage, model=model, provider=result.provider, prices=prices),
        ).cost_usd
        or 0.0
    )
    return tasks, result, cost


async def generate_tasks_sharded(
    *,
    cfg: Config,
    target: Target,
    out_path: Path,
    shards_dir: Path,
    auth_mode: AuthMode = "session",
    prices: PriceTable | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    max_concurrency: int | None = None,
    force: bool = False,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    notebook_filter: Sequence[str] | None = None,
    candidates_dir: Path | None = None,
    on_shard_start: Callable[[ShardOutcome], None] | None = None,
    on_shard_done: Callable[[ShardOutcome], None] | None = None,
) -> ShardedResult:
    """Generate ``tasks.yaml`` by fanning out one generation agent per notebook (or mined script).

    With ``candidates_dir`` the shards are the mined candidate scripts in that directory (one per
    ``*.py``, see :mod:`acumen.mining`) instead of the package's notebooks: each is seeded into its
    agent's working directory and prompted with :func:`acumen.prompts.taskgen_mined_prompt`. Same
    isolation, same per-shard cache and merge.

    All shards share ONE read-only filtered source copy (built once); each still gets its own
    throwaway work/home and its own labelled processes, reaped independently.

    Parameters
    ----------
    cfg
        The pass config; supplies ``meta_model``, ``max_concurrency`` and ``env_passthrough``.
    target
        The prepared target: the source checkout to shard and the venv to run pipelines against.
    out_path
        The merged ``tasks.yaml``. Refuses to overwrite an existing file unless ``force``.
    shards_dir
        Directory of per-shard ``<slug>.yaml`` files. To regenerate a shard, delete its file
        (``force`` governs only ``out_path``, never the shard cache).
    notebook_filter
        Keep only shards whose notebook path (or candidate slug) contains any of these substrings.
    candidates_dir
        Shard over the mined ``*.py`` candidates here instead of the package's notebooks.
    on_shard_start, on_shard_done
        Optional progress callbacks, invoked as each shard is admitted and as it lands.

    Returns
    -------
    The sharded result: the merged tasks, the per-shard outcomes, and the total cost.

    Raises
    ------
    TaskGenError
        If ``out_path`` exists without ``force``, there is nothing to shard, or no shard produced
        any task.
    """
    if out_path.exists() and not force:
        raise TaskGenError(f"{out_path} already exists — pass force=True to overwrite it")

    selected_model = model or cfg.meta_model
    table = prices if prices is not None else PriceTable(overrides=cfg.prices)
    holder = Path(tempfile.mkdtemp(prefix="acumen-shards-"))
    try:
        source_copy = build_filtered_source(target.src_dir, holder / "source")
        # A shard is (slug, path-within-source-copy | None, candidate file | None).
        if candidates_dir is not None:
            candidates = sorted(candidates_dir.glob("*.py"))
            if not candidates:
                raise TaskGenError(f"no mined candidates (*.py) under {candidates_dir}")
            shards: list[tuple[str, Path | None, Path | None]] = [(c.stem, None, c) for c in candidates]
        else:
            notebooks = notebook_shards(source_copy)
            if not notebooks:
                raise TaskGenError(
                    f"no notebooks (*.ipynb) found under {target.src_dir} — sharded generation needs "
                    "tutorials to shard on. Check submodules are present, or use generate_tasks to "
                    "infer analyses from source instead."
                )
            shards = [(_shard_slug(nb), nb, None) for nb in notebooks]
        if notebook_filter:
            shards = [s for s in shards if any(sub in (s[1].as_posix() if s[1] else s[0]) for sub in notebook_filter)]
            if not shards:
                raise TaskGenError(f"no shards matched --notebook {list(notebook_filter)}")
        shards_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrency or cfg.max_concurrency)
        outcomes: list[ShardOutcome] = []
        # Set by the first shard the platform refuses: every shard still queued is then skipped
        # rather than failed one by one against the same wall.
        paused: list[str] = []

        async def run_shard(shard: tuple[str, Path | None, Path | None]) -> ShardOutcome:
            slug, nb_path, candidate = shard
            notebook = nb_path if nb_path is not None else Path(candidate.name)  # type: ignore[union-attr]
            shard_path = shards_dir / f"{slug}.yaml"
            cached = _cached_shard(notebook, slug, shard_path)
            if cached is not None:
                if on_shard_start is not None:
                    on_shard_start(cached)
                if on_shard_done is not None:
                    on_shard_done(cached)
                return cached

            async with semaphore:
                pending = ShardOutcome(notebook=notebook, slug=slug, shard_path=shard_path, status="failed")
                if paused:
                    outcome = replace(pending, error=f"skipped: paused after {paused[0][:120]}")
                    if on_shard_done is not None:
                        on_shard_done(outcome)
                    return outcome
                if on_shard_start is not None:
                    on_shard_start(pending)
                work_root = holder / "shards" / slug
                work_root.mkdir(parents=True, exist_ok=True)
                log = LiveLog.open(log_dir, f"tasks-{slug}", stream=stream) if log_dir is not None else None
                if candidate is not None:
                    # The mined script is seeded into the agent's work dir (not the source copy —
                    # it is not package source) and the prompt points at it there.
                    seed_name = f"analysis-{slug}.py"
                    seed_files: dict[str, str] | None = {seed_name: candidate.read_text()}
                    analysis_path = work_root / "work" / seed_name

                    def make_prompt(staged: Path, scripts: Path, analysis: Path = analysis_path) -> str:
                        return taskgen_mined_prompt(
                            package=target.pkg_name,
                            src=source_copy,
                            python=target.python,
                            out=staged,
                            scripts_dir=scripts,
                            analysis=analysis,
                            feedback=feedback,
                        )

                else:
                    seed_files = None

                    def make_prompt(staged: Path, scripts: Path, nb: Path = notebook) -> str:
                        return taskgen_shard_prompt(
                            package=target.pkg_name,
                            src=source_copy,
                            python=target.python,
                            out=staged,
                            scripts_dir=scripts,
                            notebook=nb.as_posix(),
                            feedback=feedback,
                        )

                try:
                    tasks, result, cost = await _run_shard_agent(
                        work_root=work_root,
                        source_copy=source_copy,
                        target=target,
                        make_prompt=make_prompt,
                        cfg=cfg,
                        model=selected_model,
                        max_turns=max_turns,
                        max_usd=max_usd,
                        auth_mode=auth_mode,
                        prices=table,
                        log=log,
                        seed_files=seed_files,
                    )
                    # Scripts first, the shard yaml last: the yaml's presence is what marks the shard
                    # done, so an interrupted write is regenerated rather than half-trusted.
                    shard_scripts = shards_dir / slug
                    shard_scripts.mkdir(parents=True, exist_ok=True)
                    staged_dir = work_root / "work" / SCRIPTS_DIRNAME
                    harvest_scripts(tasks, staged_dir, shard_scripts)
                    shard_path.write_text(dump_tasks(tasks))
                    outcome = ShardOutcome(
                        notebook=notebook,
                        slug=slug,
                        shard_path=shard_path,
                        status="generated",
                        n_tasks=len(tasks),
                        cost_usd=cost,
                        turns=result.num_turns,
                        log_jsonl=log.jsonl_path if log is not None else None,
                        log_html=log.html_path if log is not None and log.html_rendered else None,
                    )
                except Exception as err:  # noqa: BLE001 - one shard failing must not take the pass down
                    if is_transient(str(err)) and not paused:
                        paused.append(str(err))
                    outcome = ShardOutcome(
                        notebook=notebook,
                        slug=slug,
                        shard_path=shard_path,
                        status="failed",
                        error=f"{type(err).__name__}: {err}",
                        log_jsonl=log.jsonl_path if log is not None else None,
                        log_html=log.html_path if log is not None and log.html_rendered else None,
                    )
                finally:
                    if log is not None:
                        log.close()
                    reap(work_root)
                    shutil.rmtree(work_root, ignore_errors=True)
                if on_shard_done is not None:
                    on_shard_done(outcome)
                return outcome

        for coro in asyncio.as_completed([run_shard(s) for s in shards]):
            outcomes.append(await coro)

        outcomes.sort(key=lambda o: o.slug)
        tasks = merge_shards(shards_dir, out_path)
        return ShardedResult(
            tasks=tasks,
            out_path=out_path,
            shards_dir=shards_dir,
            outcomes=outcomes,
            cost_usd=sum(outcome.cost_usd for outcome in outcomes),
            paused=paused[0] if paused else None,
        )
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
