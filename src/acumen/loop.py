"""The rulebook layer: draft a skill from a rulebook, score it, improve the rulebook.

The rulebook (:mod:`acumen.rulebooks`) is the SKILL-generating instructions; skills are drafted from
it and are intermediates. One improvement step reads a drafted skill's **train-split** failure
evidence and rewrites the rulebook — the instructions, never the instance — then a fresh skill is
drafted from the new version and benchmarked. :mod:`acumen.evolve` drives these pieces generationally;
this module holds the pieces themselves:

* :func:`improve_rulebook` — the improve agent, behind the held-out guard of :mod:`acumen.evidence`.
* :func:`_ensure_skill` / :func:`_ensure_skill_variants` / :func:`_ensure_rulebook` — draft on demand,
  resumable by file presence, so a rerun never respawns an agent for work already on disk.
* :func:`score` / :class:`DraftScores` — pass and skill-load counts read from disk, and a rulebook
  scored as a mean over N independent drafts, because the same text drafted twice differs by ~8pp.
* :func:`_lockbox_eval` — opens the lockbox once and benches a version over N drafts.

Every boundary is structural: evidence is collected only for the tasks an agent may learn from, and a
``PreToolUse`` guard (:func:`acumen.evidence.make_test_guard`) denies everything else — test splits,
held-out tasks, the lockbox, the run trees. A platform limit is a pause
(:class:`acumen.transient.TransientLimitError`), never a failed candidate.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen import rulebooks as rb
from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.bench import BenchmarkInvalidError, PlannedRun, build_matrix, pending, run_matrix
from acumen.config import Config
from acumen.draft import DraftError, draft_skill
from acumen.env import AuthMode, Target, build_agent_env
from acumen.evidence import _write_material, collect_train_runs, make_test_guard
from acumen.improve import _read_rationale
from acumen.logs import LiveLog
from acumen.paths import RESULT_FILE, Split, arm_name, is_complete, run_dir
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import rulebook_improve_prompt
from acumen.runner import RunOutcome
from acumen.skills import Skill, available_versions, load_skill
from acumen.tasks import Task
from acumen.transient import TransientLimitError, is_transient

#: Subdirectory of a runs root that holds the lockbox arms, and of a skills/runs root that holds
#: the extra drafts of each rulebook version.
LOCKBOX_RUNS_DIRNAME = "lockbox"
DRAFTS_DIRNAME = "drafts"


class LoopError(RuntimeError):
    """Raised when a loop iteration cannot proceed."""


@dataclass(frozen=True)
class RulebookResult:
    """A newly written rulebook version and what it cost to produce."""

    version: str
    parent: str
    path: Path
    rationale: str
    changed: bool
    cost_usd: float
    turns: int
    n_train_runs: int
    n_train_failures: int
    log_jsonl: Path | None = None
    log_html: Path | None = None


@dataclass(frozen=True)
class Score:
    """Two independent success metrics over a benchmarked slice: passing, and loading the skill.

    ``passed`` and ``loaded`` are orthogonal (a baseline run can pass without any skill; a skill run
    can load the skill and still fail), so they are tracked separately — a skill that never loads is
    a distinct, and prior, failure to one that loads and gets the wrong answer. The first real
    measured loop run turned entirely on this distinction: every arm failed *because the skill never
    loaded*, invisible on pass rate alone.
    """

    passed: int
    total: int
    #: How many of these runs actually loaded the skill under test (0 for the noskill arm).
    loaded: int = 0

    @property
    def rate(self) -> float:
        """Pass rate in ``[0, 1]``; ``0.0`` when nothing was scored."""
        return self.passed / self.total if self.total else 0.0

    @property
    def load_rate(self) -> float:
        """Skill-load rate in ``[0, 1]``; ``0.0`` when nothing was scored."""
        return self.loaded / self.total if self.total else 0.0


# ── Scoring ────────────────────────────────────────────────────────────────────────────


def score(runs_root: Path, planned: Sequence[PlannedRun]) -> Score:
    """Read ``result.json`` for each planned run and tally both success metrics: passes and loads.

    Reads from disk rather than trusting a ``run_matrix`` return, so a resumed pass (where the
    matrix ran nothing because everything was already complete) still scores correctly.
    """
    passed = total = loaded = 0
    for item in planned:
        directory = run_dir(runs_root, item.key)
        if not is_complete(directory):
            continue
        try:
            data = json.loads((directory / RESULT_FILE).read_text())
        except (OSError, ValueError):
            continue
        total += 1
        passed += bool(data.get("success"))
        loaded += bool(data.get("skill_loaded"))
    return Score(passed=passed, total=total, loaded=loaded)


# ── Benchmarking a skill (subscription auth, pass rate not dollars) ─────────────────────


async def _bench(
    *,
    cfg: Config,
    target: Target,
    skill: Skill,
    tasks: Sequence[Task],
    runs_root: Path,
    splits: Sequence[Split],
    auth_mode: AuthMode,
    task_ids: Sequence[str] | None,
    max_concurrency: int,
    on_start: Callable[[PlannedRun], None] | None,
    on_done: Callable[[RunOutcome], None] | None,
) -> list[PlannedRun]:
    """Bench one skill on the given splits, resuming completed runs. Returns the full planned set.

    The loop scores from disk (:func:`score`) over the returned planned set, so this ignores the
    outcomes ``run_matrix`` hands back.
    """
    planned = build_matrix(cfg, tasks, skill=skill.version, splits=splits, task_ids=task_ids)
    todo = pending(planned, runs_root, skill_hash=skill.hash)
    if todo:
        try:
            await run_matrix(
                todo,
                target=target,
                runs_root=runs_root,
                max_concurrency=max_concurrency,
                auth_mode=auth_mode,
                skill=skill,
                skill_name=cfg.skill_name,
                on_start=on_start,
                on_done=on_done,
                env_passthrough=cfg.env_passthrough,
                dataset_cache_dirs=cfg.dataset_cache_dirs,
            )
        except BenchmarkInvalidError as err:
            # A provider limit, an exhausted credential or a dropped network is the platform's
            # problem: those cells stay pending and a rerun resumes them. A refused sandbox host is
            # a harness bug that no wait fixes, so it is left to propagate.
            if all(o.reason in {"provider_exhausted", "connection_error"} for o in err.outcomes):
                raise TransientLimitError(str(err)) from err
            raise
    return planned


# ── The outer-improve agent (edits the rulebook, not the skill) ─────────────────────────


async def improve_rulebook(
    *,
    cfg: Config,
    target: Target,
    rulebooks_root: Path,
    runs_root: Path,
    benched_skill: Skill,
    tasks: Sequence[Task],
    auth_mode: AuthMode = "session",
    parent_version: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    feedback: str | None = None,
    log: LiveLog | None = None,
    held_out_ids: Sequence[str] = (),
    deny_dirs: Sequence[Path] = (),
    prices: PriceTable | None = None,
) -> RulebookResult:
    """Improve the current rulebook into the next version from the drafted skill's train evidence.

    The outer-loop analogue of :func:`acumen.improve.improve_skill`: same train-split evidence, same
    held-out guard, but the artifact under edit is the rulebook (draft instructions), not the skill.
    ``benched_skill`` is the skill drafted from the parent rulebook and just benchmarked — its arm is
    where the train evidence lives.

    The evidence is structurally limited to ``tasks``: only their train runs are collected and
    written into the agent's work dir. For a CV fold, ``tasks`` is the optimize set and
    ``held_out_ids`` the rest — the guard then also denies the held-out tasks' runs in *every*
    split (their train runs name the task and its answer), and ``deny_dirs`` (the lockbox, the CV
    trees) wholesale. Evidence and guard together are the fold boundary.

    Raises
    ------
    LoopError
        If there is no parent rulebook, no train evidence, or the agent produced an invalid rulebook.
    """
    parent = parent_version or rb.latest_version(rulebooks_root)
    if parent is None:
        raise LoopError(f"no rulebook to improve under {rulebooks_root} — seed one first")
    parent_text = rb.load_rulebook(rulebooks_root, parent).text
    new_version = rb.next_version(rulebooks_root)
    if rb.rulebook_dir(rulebooks_root, new_version).exists():
        raise LoopError(f"rulebook {new_version} already exists — versions are immutable")

    arm = arm_name(benched_skill.version)
    train_runs = collect_train_runs(runs_root, arm, list(tasks))
    if not train_runs:
        raise LoopError(
            f"no train-split runs for {benched_skill.version} under {runs_root / arm / 'train'} — "
            "the skill must be benched on the train split before the rulebook can be improved"
        )
    n_failures = sum(1 for r in train_runs if not r.success)

    holder = Path(tempfile.mkdtemp(prefix="acumen-rulebook-"))
    try:
        work = holder / "work"
        staged_rulebook = work / rb.RULEBOOK_FILE
        train_dir = work / "train"
        rationale_path = work / "rationale.md"
        home = holder / "home"
        selected_model = model or cfg.meta_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        staged_rulebook.write_text(parent_text)
        # Reuse the skill improver's evidence writer verbatim — the material is identical.
        _write_material(train_dir, train_runs)

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
        prompt = rulebook_improve_prompt(
            package=target.pkg_name,
            rulebook_path=staged_rulebook,
            train_dir=train_dir,
            rationale_path=rationale_path,
            parent_version=parent,
            new_version=new_version,
            feedback=feedback,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            max_turns=max_turns,
            max_usd=max_usd,
            price_usd=pricer(selected_model, table),
            # The agent reads only the staged train evidence; it edits only its work directory.
            write_dirs=(work,),
            discover_skills=False,
            # Same held-out guard as the skill improver — plus, for a CV fold, the held-out tasks
            # and the lockbox/CV trees. See make_test_guard. Built only for Claude.
            claude_hooks=(
                {"PreToolUse": [make_test_guard(runs_root, held_out_ids=held_out_ids, deny_dirs=deny_dirs)]}
                if provider == "claude"
                else None
            ),
            # Codex has no hook: deny the run tree and every denied directory wholesale.
            deny_paths=(runs_root.resolve(), *(d.resolve() for d in deny_dirs)),
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(prompt, options=options, on_event=log.append if log is not None else None)
        except Exception as err:  # noqa: BLE001 - a failed improve is an error to report, re-raised below
            agent_error = err
        finally:
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            if is_transient(str(agent_error)):
                raise TransientLimitError(
                    f"the platform refused the rulebook-improve agent ({str(agent_error)[:160]}); nothing was "
                    "written — rerun to resume once the limit resets"
                ) from agent_error
            raise LoopError(f"the rulebook-improve agent failed: {type(agent_error).__name__}: {agent_error}") from (
                agent_error
            )
        if result is None:
            raise LoopError("the rulebook-improve agent produced no result message")
        if result.is_error:
            detail = f"{result.subtype} {' '.join(map(str, result.errors or []))}".strip()
            if is_transient(detail):
                raise TransientLimitError(
                    f"the platform refused the rulebook-improve agent ({detail[:160]}); nothing was written — "
                    "rerun to resume once the limit resets"
                )
            raise LoopError(f"the rulebook-improve agent errored: {detail}")

        new_text = staged_rulebook.read_text()
        rationale = _read_rationale(rationale_path, result, parent, new_version)
        try:
            # write_rulebook validates placeholders — a broken template fails here, not at draft
            # time — and records parent/rationale/hash in meta.json, the provenance chain.
            written = rb.write_rulebook(
                rulebooks_root, new_version, new_text, parent=parent, rationale=rationale, feedback=feedback
            )
        except rb.RulebookError as err:
            raise LoopError(f"the rulebook-improve agent produced an invalid rulebook: {err}") from err

        return RulebookResult(
            version=new_version,
            parent=parent,
            path=written.path,
            rationale=rationale,
            changed=new_text != parent_text,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(result.usage, model=selected_model, provider=result.provider, prices=table),
            ).cost_usd,
            turns=result.num_turns,
            n_train_runs=len(train_runs),
            n_train_failures=n_failures,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


# ── One crude iteration ─────────────────────────────────────────────────────────────────


async def _ensure_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebook_text: str,
    expect_version: str,
    auth_mode: AuthMode,
    rationale: str,
    log_dir: Path | None,
    stream: bool,
    log_name: str | None = None,
) -> tuple[Skill, float]:
    """Draft a skill from ``rulebook_text`` if ``expect_version`` isn't present, else load it.

    Resume-friendly: a re-run reuses an already-drafted skill instead of erroring on the immutable
    version. Asserts the drafted version matches ``expect_version`` so the rulebook<->skill lockstep
    the loop relies on can't silently drift. ``log_name`` disambiguates the live log when several
    roots draft the same ``expect_version`` (extra drafts are all ``v1`` in their own trees).
    """
    if expect_version in available_versions(skills_root):
        return load_skill(skills_root, expect_version, expect_name=cfg.skill_name), 0.0
    name = log_name or f"loop-draft-{expect_version}"
    log = LiveLog.open(log_dir, name, stream=stream) if log_dir is not None else None
    try:
        result = await draft_skill(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            auth_mode=auth_mode,
            rationale=rationale,
            rulebook=rulebook_text,
            log=log,
        )
    except DraftError as err:
        # The platform refusing the drafting agent is a pause, not a failed draft: nothing was
        # written, and the caller's rerun resumes here.
        if is_transient(str(err)):
            raise TransientLimitError(
                f"the platform refused the drafting agent ({str(err)[:160]}); rerun to resume once the limit resets"
            ) from err
        raise
    finally:
        if log is not None:
            log.close()
    if result.skill.version != expect_version:
        raise LoopError(
            f"drafted skill {result.skill.version} but expected {expect_version} — the "
            "rulebook/skill version lockstep is broken; run the loop in a clean workspace"
        )
    return result.skill, result.cost_usd


async def _ensure_skill_variants(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    version: str,
    n: int,
    auth_mode: AuthMode,
    log_dir: Path | None,
    stream: bool,
) -> tuple[list[Skill], float]:
    """Ensure ``n`` independently drafted skills of rulebook ``version`` exist; return them, primary first.

    Draft 1 is the primary ``skills/<version>`` (drafted anyway by the loop); drafts 2..n live at
    :func:`draft_variant_root` — each its own skills-root holding a single ``v1``, so the ``vN``
    lockstep of the main tree is untouched. Resume-friendly like everything else: a present draft is
    loaded, not re-drafted, so re-running with a larger ``n`` only adds the missing drafts.
    """
    text = rb.load_rulebook(rulebooks_root, version).text
    total_cost = 0.0
    skills: list[Skill] = []
    primary, cost = await _ensure_skill(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebook_text=text,
        expect_version=version,
        auth_mode=auth_mode,
        rationale=f"drafted from rulebook {version}",
        log_dir=log_dir,
        stream=stream,
    )
    skills.append(primary)
    total_cost += cost
    for i in range(2, n + 1):
        extra, cost = await _ensure_skill(
            cfg=cfg,
            target=target,
            skills_root=draft_variant_root(skills_root, version, i),
            rulebook_text=text,
            expect_version="v1",
            auth_mode=auth_mode,
            rationale=f"extra draft d{i} of rulebook {version} — draft-variance sample",
            log_dir=log_dir,
            stream=stream,
            log_name=f"loop-draft-{version}-d{i}",
        )
        skills.append(extra)
        total_cost += cost
    return skills, total_cost


async def _ensure_rulebook(
    *,
    cfg: Config,
    target: Target,
    rulebooks_root: Path,
    runs_root: Path,
    benched_skill: Skill,
    tasks: Sequence[Task],
    baseline_version: str,
    baseline_text: str,
    auth_mode: AuthMode,
    feedback: str | None,
    log_dir: Path | None,
    stream: bool,
    deny_dirs: Sequence[Path] = (),
    log_name: str = "loop-rulebook-v2",
    expect_version: str | None = None,
) -> RulebookResult:
    """Improve the rulebook, or — on a resumed run — reconstruct the already-improved version.

    Keeps the loop resumable: if the improved version already exists on disk, the (expensive)
    improve agent is not re-run; its recorded rationale is read back so the report is still complete.
    ``expect_version`` pins which version counts as "the improved one" — a multi-iteration loop
    resuming iteration 1 of a three-version chain must reconstruct ``v2``, not the latest ``v3``.
    Without it, the latest version past the baseline is taken (the single-iteration behaviour).
    """
    latest = expect_version if expect_version in rb.available_versions(rulebooks_root) else None
    if expect_version is None:
        latest = rb.latest_version(rulebooks_root)
        if latest == baseline_version:
            latest = None
    elif latest is None and rb.next_version(rulebooks_root) != expect_version:
        raise LoopError(
            f"expected to write rulebook {expect_version} but the chain's next version is "
            f"{rb.next_version(rulebooks_root)} — the rulebook chain is not linear; run in a clean workspace"
        )
    if latest is not None:
        resumed = rb.load_rulebook(rulebooks_root, latest)
        meta = rb.rulebook_meta(rulebooks_root, latest)
        rationale = meta.rationale if meta is not None and meta.rationale else "(resumed; rationale not recorded)"
        return RulebookResult(
            version=latest,
            parent=meta.parent if meta is not None and meta.parent else baseline_version,
            path=resumed.path,
            rationale=rationale,
            changed=resumed.text != baseline_text,
            cost_usd=0.0,
            turns=0,
            n_train_runs=0,
            n_train_failures=0,
        )
    log = LiveLog.open(log_dir, log_name, stream=stream) if log_dir is not None else None
    try:
        return await improve_rulebook(
            cfg=cfg,
            target=target,
            rulebooks_root=rulebooks_root,
            runs_root=runs_root,
            benched_skill=benched_skill,
            tasks=tasks,
            auth_mode=auth_mode,
            parent_version=baseline_version,
            feedback=feedback,
            log=log,
            deny_dirs=deny_dirs,
        )
    finally:
        if log is not None:
            log.close()


def draft_variant_root(root: Path, version: str, i: int) -> Path:
    """Where draft ``i`` (>= 2) of ``version`` lives under a skills or runs root.

    Draft 1 is the primary draft at ``root/<version>`` (the rulebook<->skill lockstep the loop
    relies on); extras are each their own single-version tree, so nothing about the ``vN`` naming
    contract changes: ``root/drafts/<version>/d<i>`` holds a ``v1`` skill (or that skill's runs).
    """
    return root / DRAFTS_DIRNAME / version / f"d{i}"


@dataclass(frozen=True)
class DraftScores:
    """One rulebook version scored over N independently drafted skills on the same tasks.

    Index 0 is the primary draft (``skills/<version>``); the rest are the extra drafts under
    ``skills/drafts/<version>/``. The mean is the version's score; the spread is the measured
    draft-to-draft noise a single-draft number hides.
    """

    version: str
    scores: list[Score]
    #: Each draft's content size in bytes (the leanness axis varies across drafts too).
    sizes: list[int]

    @property
    def rates(self) -> list[float]:
        """Per-draft pass rates, primary first."""
        return [s.rate for s in self.scores]

    @property
    def mean_rate(self) -> float:
        """Mean pass rate over drafts — the version's score under N-draft scoring."""
        return sum(self.rates) / len(self.scores) if self.scores else 0.0

    @property
    def spread(self) -> float:
        """Max minus min draft pass rate: the draft-to-draft noise, as a rate."""
        return (max(self.rates) - min(self.rates)) if self.scores else 0.0

    @property
    def mean_load_rate(self) -> float:
        """Mean skill-load rate over drafts."""
        return sum(s.load_rate for s in self.scores) / len(self.scores) if self.scores else 0.0


async def _lockbox_eval(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    version: str,
    n_drafts: int,
    tasks: Sequence[Task],
    auth_mode: AuthMode,
    max_concurrency: int,
    log_dir: Path | None,
    stream: bool,
    on_bench_start: Callable[[PlannedRun], None] | None,
    on_bench_done: Callable[[RunOutcome], None] | None,
) -> DraftScores:
    """Score every draft of one rulebook version on the lockbox tasks' test split.

    The primary draft's runs land in ``runs/lockbox/`` as before; each extra draft benches into its
    own tree (``runs/drafts/<version>/d<i>/lockbox/``), so no arm collides. Resumable by file
    presence, which is what "scored once" means operationally: each draft is benched on the lockbox
    at most once, however many times the loop is re-run — and re-running with a larger ``n_drafts``
    only adds the missing drafts.
    """
    skills, _ = await _ensure_skill_variants(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebooks_root=rulebooks_root,
        version=version,
        n=n_drafts,
        auth_mode=auth_mode,
        log_dir=log_dir,
        stream=stream,
    )
    scores: list[Score] = []
    for i, skill in enumerate(skills, start=1):
        root = runs_root if i == 1 else draft_variant_root(runs_root, version, i)
        planned = await _bench(
            cfg=cfg,
            target=target,
            skill=skill,
            tasks=tasks,
            runs_root=root / LOCKBOX_RUNS_DIRNAME,
            splits=["test"],
            auth_mode=auth_mode,
            task_ids=None,
            max_concurrency=max_concurrency,
            on_start=on_bench_start,
            on_done=on_bench_done,
        )
        scores.append(score(root / LOCKBOX_RUNS_DIRNAME, planned))
    return DraftScores(version=version, scores=scores, sizes=[s.size for s in skills])
