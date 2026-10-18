"""Generational rulebook evolution: improve-from-best, cheap screens, a confirmed ratchet.

Each generation edits the champion *rulebook* (:mod:`acumen.rulebooks`) — the instructions a skill is
drafted from — instead of a skill. Scoring one rulebook means drafting it into a skill, and drafting
is noisy (the same text drafted twice differs by several points), so decisions are made on a mean over
independent drafts and every boundary is structural. This module trades per-decision certainty for
decision volume — the shape of an autoresearch run with hundreds of generations. The pieces it
composes (the improve agent, resumable drafting, scoring over drafts, the lockbox evaluation) live in
:mod:`acumen.loop`; see ``docs/how-evolve-works.md`` for the whole picture.

* **Improve from the best, always.** Every generation's improve agent starts from the current
  champion and its failure evidence, never from a regressed child. Rejected candidates keep their
  version directories (provenance) but nothing descends from them.
* **Wild exploration.** Each generation carries a rotating *directive* (:data:`DIRECTIVES`) injected
  as maintainer feedback — bold restructures, aggressive deletion, inversion of an assumption —
  so the search tries large moves, not only one-more-bullet edits.
* **Two-tier selection.** A generation is scored on a small rotating **screen** subset (cheap,
  noisy, allowed to be wrong); a screen win only makes the candidate the *screen champion*. The
  **confirmed champion** — the version improvement builds from permanently — only changes after a
  full held-out bench (every ``confirm_every`` accepts), and a failed confirmation reverts the
  screen champion. Screen decisions may thrash; the ratchet may not.
* **Archive everything.** Every generation appends one JSON line (directive, versions, subset,
  scores, decision) to ``runs/evolve.jsonl`` — hundreds of recorded accept/reject decisions are the
  dataset a cross-pollination step mines for edits that replicate across independent islands.
* The **lockbox** stays write-once and is opened once at the very end (champion and seed, each over
  ``n_drafts`` independent drafts) — screens and confirmations never touch it.

Everything resumes by file presence and *determinism*: the directive, screen subset, and candidate
version of generation ``g`` are pure functions of ``(g, seed)``, so a resumed run replays every
decision from on-disk scores without spawning an agent for finished work.
"""

from __future__ import annotations

import difflib
import json
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen import rulebooks as rb
from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.bench import PlannedRun
from acumen.config import Config
from acumen.difficulty import HeadroomSelection, screen, select_headroom
from acumen.env import AuthMode, Target, build_agent_env
from acumen.evidence import make_test_guard
from acumen.folds import FoldError, Lockbox, check_disjoint, load_lockbox_tasks, make_folds, read_lockbox
from acumen.improve import _read_rationale
from acumen.logs import LiveLog
from acumen.loop import (
    DRAFTS_DIRNAME,
    LOCKBOX_RUNS_DIRNAME,
    DraftScores,
    LoopError,
    RulebookResult,
    Score,
    _bench,
    _ensure_rulebook,
    _ensure_skill,
    _ensure_skill_variants,
    _lockbox_eval,
    draft_variant_root,
    score,
)
from acumen.paths import SPLITS
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import pollinate_prompt
from acumen.runner import RunOutcome
from acumen.skills import Skill
from acumen.tasks import Task
from acumen.transient import TransientLimitError, is_transient

#: The journal of decisions, one JSON object per finished generation, under the runs root.
JOURNAL_FILE = "evolve.jsonl"

#: Exploration directives, rotated deterministically over generations (``g-1 mod len``). Each is
#: injected into the improve agent's prompt as maintainer feedback, so the meta.json of every
#: version records which directive shaped it. The mix is deliberate: focused repair, bold
#: restructuring, deletion pressure, concretization, loading/routing work, assumption-inversion,
#: generalization, and a free slot — a search that only ever adds bullets converges on bloat.
DIRECTIVES: tuple[str, ...] = (
    "Make one focused edit addressing the largest cluster of failures in the evidence.",
    "Be bold: restructure the rulebook wholesale if a different organization would generate better "
    "skills — reorder, merge, or split sections freely. Large rewrites are allowed.",
    "Delete aggressively: find the guidance you believe pays for itself least and remove it. "
    "Leanness is a virtue; instruction no failure ever needed is suspect.",
    "Concretize: for the failure classes in the evidence, make the rulebook demand exact, runnable "
    "recipes (function sequence, parameter values) instead of prose advice.",
    "Attack loading and routing: rewrite how the rulebook instructs the skill's description and "
    "SKILL.md-to-references routing, so the right guidance is in front of the agent at the right time.",
    "Question one assumption the rulebook takes for granted and invert or drop it; say in the "
    "rationale which assumption and why the evidence licenses doubting it.",
    "Generalize: find two or more rules that are special cases of one deeper principle and replace "
    "them with that principle.",
    "Free choice: make the change you most believe in, however unconventional.",
)


def directive_for(generation: int) -> str:
    """The exploration directive of 1-based ``generation`` — deterministic, so resume replays it."""
    if generation < 1:
        raise LoopError(f"generation must be >= 1, got {generation}")
    return DIRECTIVES[(generation - 1) % len(DIRECTIVES)]


def screen_subset(ids: Sequence[str], size: int, epoch: int, seed: int) -> list[str]:
    """The screen tasks of ``epoch`` — a seeded sample, stable within an epoch, rotated across them.

    Rotation is the guard against a search that optimizes the screen instead of the package: over
    epochs the subsets cover the whole pool, and a champion is re-screened on every new subset it
    survives into. Deterministic in ``(seed, epoch)`` so a resumed run rebuilds the same subsets.
    """
    pool = sorted(set(ids))
    if size < 1:
        raise LoopError(f"screen size must be >= 1, got {size}")
    if len(pool) <= size:
        return pool
    return sorted(random.Random(f"{seed}:{epoch}").sample(pool, size))


@dataclass(frozen=True)
class Generation:
    """One recorded generation: what was tried, on which screen, and what was decided."""

    index: int
    directive: str
    #: The screen champion the candidate was improved from.
    parent: str
    #: The rulebook/skill version this generation wrote (``v{index+1}``), kept even if rejected.
    candidate: str
    subset: tuple[str, ...]
    champion_screen: Score
    candidate_screen: Score
    accepted: bool
    #: Whether a full-bench confirmation ran after this generation, and what it decided.
    confirm_ran: bool
    #: With ``confirm_ran``: ``True`` promoted the challenger, ``False`` reverted to the incumbent.
    confirm_promoted: bool | None
    #: The confirmed champion after this generation — the version improvement ratchets from.
    confirmed: str
    cost_usd: float

    def to_json(self) -> dict:
        """The journal line — flat, so the archive is greppable and mineable."""
        return {
            "generation": self.index,
            "directive": self.directive,
            "parent": self.parent,
            "candidate": self.candidate,
            "subset": list(self.subset),
            "champion_screen": [self.champion_screen.passed, self.champion_screen.total],
            "candidate_screen": [self.candidate_screen.passed, self.candidate_screen.total],
            "accepted": self.accepted,
            "confirm_ran": self.confirm_ran,
            "confirm_promoted": self.confirm_promoted,
            "confirmed": self.confirmed,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass(frozen=True)
class EvolveRun:
    """The outcome of one evolution run: the decisions, the champion, and (optionally) the verdict."""

    generations: list[Generation]
    #: The confirmed champion at the end — pending screen accepts are confirmed (or reverted) first.
    champion: str
    stopped_because: str
    journal_path: Path
    #: Lockbox verdicts over ``n_drafts`` drafts per version; ``None`` when the lockbox was not
    #: opened (islands never open it — only the final merged champion does).
    lockbox_drafts: DraftScores | None = None
    lockbox_baseline_drafts: DraftScores | None = None
    selection: HeadroomSelection | None = None
    cost_usd: float = 0.0

    @property
    def accepted(self) -> int:
        """How many generations won their screen."""
        return sum(1 for g in self.generations if g.accepted)

    @property
    def lockbox_mean_delta(self) -> float | None:
        """Lockbox mean-rate change from the seed to the champion, over all drafts."""
        if self.lockbox_drafts is None or self.lockbox_baseline_drafts is None:
            return None
        return self.lockbox_drafts.mean_rate - self.lockbox_baseline_drafts.mean_rate


def _read_journal(path: Path) -> set[int]:
    """The generation indices already journaled — appends are idempotent across resumes."""
    if not path.is_file():
        return set()
    done: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            done.add(int(json.loads(line)["generation"]))
        except (ValueError, KeyError, TypeError):
            continue
    return done


async def run_evolve(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    generations: int = 20,
    screen_size: int = 12,
    screen_drafts: int = 1,
    epoch_len: int = 5,
    accept_delta: int = 2,
    confirm_every: int = 5,
    n_drafts: int = 3,
    seed: int = 0,
    lockbox_dir: Path | None = None,
    allow_no_lockbox: bool = False,
    evaluate_lockbox: bool = True,
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    headroom_only: bool = False,
    max_wallclock_s: float | None = None,
    extra_deny_dirs: Sequence[Path] = (),
    clock: Callable[[], float] = time.monotonic,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_generation: Callable[[Generation], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> EvolveRun:
    """Run ``generations`` improve-from-best steps with screen selection and a confirmed ratchet.

    Per generation ``g`` (all deterministic in ``(g, seed)``, so resume replays from disk):

    1. Bench the screen champion on this epoch's subset, train+test (train = the improve evidence,
       test = the score the candidate must beat; both mostly cached within an epoch).
    2. Improve the rulebook **from the screen champion** under :func:`directive_for`'s directive
       (composed with the caller's ``feedback``), into version ``v{g+1}``; draft one skill from it.
    3. Bench the candidate on the subset's test split; **accept** (candidate becomes screen
       champion) iff it beats the champion by ``accept_delta`` passes on the identical subset.
    4. Every ``confirm_every`` accepts, bench the screen champion on the FULL task set's test split
       and compare with the confirmed champion's full bench: strictly more passes promotes it;
       anything else reverts the screen champion to the confirmed one. Pending accepts are
       confirmed the same way once the generation budget or wall-clock stops the run.

    ``accept_delta`` exists because the screen is noisy by construction (12-ish tasks): it should be
    set from measured draft variance, not hope — a screen win smaller than the noise
    floor must not even challenge. ``screen_drafts`` attacks that noise at its source: every score
    above is summed over that many independent drafts of the version, and the accept bar scales
    with it, so ``accept_delta`` keeps meaning "passes per draft" while the noise around it shrinks.
    Draft noise does not shrink with more screen tasks — it is a per-skill offset — so this is the
    only lever that removes it rather than damping it.

    With ``evaluate_lockbox`` (and a lockbox): the seed and the final champion are scored on the
    lockbox over ``n_drafts`` independent drafts each — the run's one honest number. Pass
    ``evaluate_lockbox=False`` for island runs whose champion feeds cross-pollination instead; the
    lockbox is still read and checked disjoint, and its trees denied to every improve agent.

    Raises
    ------
    LoopError
        On a missing/overlapping lockbox (unless waived), degenerate parameters, or a broken
        version chain (the workspace was tampered with between resumes).
    """
    if generations < 1:
        raise LoopError(f"generations must be >= 1, got {generations}")
    if accept_delta < 1:
        raise LoopError(f"accept_delta must be >= 1 (a tie must not churn the champion), got {accept_delta}")
    if confirm_every < 1:
        raise LoopError(f"confirm_every must be >= 1, got {confirm_every}")
    if epoch_len < 1:
        raise LoopError(f"epoch_len must be >= 1, got {epoch_len}")
    if n_drafts < 1:
        raise LoopError(f"n_drafts must be >= 1, got {n_drafts}")
    if screen_drafts < 1:
        raise LoopError(f"screen_drafts must be >= 1, got {screen_drafts}")

    started = clock()
    concurrency = max_concurrency or cfg.max_concurrency
    total_cost = 0.0

    lockbox: Lockbox | None = None
    if lockbox_dir is not None:
        try:
            lockbox = read_lockbox(lockbox_dir)
            check_disjoint(tasks, lockbox)
        except FoldError as err:
            raise LoopError(str(err)) from err
    elif not allow_no_lockbox:
        raise LoopError(
            "no lockbox given — an evolution run makes hundreds of selections, all optimistic; the "
            "lockbox is the only number that is not. Run `acumen lockbox` and pass --lockbox, or "
            "pass --no-lockbox to proceed without a generalisation claim"
        )

    selection: HeadroomSelection | None = None
    if headroom_only:
        diffs = screen(runs_root, tasks, by_model=True)
        selection = select_headroom(diffs, tasks, split="test", models=cfg.models)
        if on_select is not None:
            on_select(selection)
        if not selection.selected:
            raise LoopError("no task has headroom to evolve on — run `acumen bench --no-skill` on more tasks first")
        tasks = selection.selected

    by_id = {t.id: t for t in tasks}
    pool_ids = sorted(by_id)
    deny_dirs: list[Path] = [runs_root / LOCKBOX_RUNS_DIRNAME, runs_root / DRAFTS_DIRNAME, *extra_deny_dirs]
    if lockbox is not None:
        deny_dirs.append(lockbox.directory)

    # Seed: rulebook v1 and its skill. The champion chain starts here.
    seed_version = rb.seed_default(rulebooks_root)
    if seed_version != "v1":
        raise LoopError(f"expected the rulebook chain to start at v1, found {seed_version}")
    champion = confirmed = "v1"
    accepts_since_confirm = 0
    journal_path = runs_root / JOURNAL_FILE
    journaled = _read_journal(journal_path)
    history: list[Generation] = []
    reason = f"reached the generation budget ({generations})"

    async def ensure(version: str) -> tuple[Skill, float]:
        return await _ensure_skill(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            rulebook_text=rb.load_rulebook(rulebooks_root, version).text,
            expect_version=version,
            auth_mode=auth_mode,
            rationale=f"drafted from rulebook {version}",
            log_dir=log_dir,
            stream=stream,
        )

    async def bench_on(version: str, task_ids: Sequence[str] | None, splits: Sequence[str]) -> Score:
        """Score ``version`` on a slice, **summed over ``screen_drafts`` independent drafts**.

        Drafting is the noisiest step in the pipeline: the same rulebook text drafted twice differs
        by ~8pp of pass rate (measured), which is the size of the effects selection is trying to
        see. That noise is a per-skill offset, so a bigger screen does not shrink it — only more
        drafts do. Summing rather than averaging keeps the arithmetic integral; the accept bar is
        scaled by the same factor, so ``accept_delta`` keeps meaning "passes per draft".

        Draft 1 is the primary ``skills/<version>`` benched into ``runs_root`` — so the improve
        agent's evidence and every existing run tree are exactly where they were at
        ``screen_drafts=1``. Drafts 2..N get their own skills and runs roots, so no arm collides
        and a rerun with a larger N only adds the missing drafts.
        """
        skills, cost = await _ensure_skill_variants(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            rulebooks_root=rulebooks_root,
            version=version,
            n=screen_drafts,
            auth_mode=auth_mode,
            log_dir=log_dir,
            stream=stream,
        )
        nonlocal total_cost
        total_cost += cost
        totals = Score(passed=0, total=0, loaded=0)
        for i, skill in enumerate(skills, start=1):
            dest = runs_root if i == 1 else draft_variant_root(runs_root, version, i)
            planned = await _bench(
                cfg=cfg,
                target=target,
                skill=skill,
                tasks=tasks,
                runs_root=dest,
                splits=list(splits),
                auth_mode=auth_mode,
                task_ids=list(task_ids) if task_ids is not None else None,
                max_concurrency=concurrency,
                on_start=on_bench_start,
                on_done=on_bench_done,
            )
            part = score(dest, [p for p in planned if p.key.split == "test"])
            totals = Score(
                passed=totals.passed + part.passed,
                total=totals.total + part.total,
                loaded=totals.loaded + part.loaded,
            )
        return totals

    async def confirm(challenger: str, incumbent: str) -> bool:
        """Full-bench the challenger and incumbent; True promotes the challenger."""
        challenger_full = await bench_on(challenger, None, ["test"])
        incumbent_full = await bench_on(incumbent, None, ["test"])
        return challenger_full.passed > incumbent_full.passed

    for g in range(1, generations + 1):
        if max_wallclock_s is not None and clock() - started >= max_wallclock_s:
            reason = f"wall-clock cap of {max_wallclock_s:.0f}s reached before generation {g}"
            break
        gen_cost_before = total_cost
        directive = directive_for(g)
        epoch = (g - 1) // epoch_len
        subset = screen_subset(pool_ids, screen_size, epoch, seed)
        subset_tasks = [by_id[i] for i in subset]
        candidate_version = f"v{g + 1}"
        parent = champion

        # 1. The champion on this epoch's screen: train (evidence) + test (the bar).
        champion_screen = await bench_on(parent, subset, SPLITS)
        champion_skill, _ = await ensure(parent)

        # 2. Improve FROM THE CHAMPION under this generation's directive; draft the candidate.
        composed = directive if not feedback else f"{directive}\n\nMaintainer feedback: {feedback}"
        rulebook = await _ensure_rulebook(
            cfg=cfg,
            target=target,
            rulebooks_root=rulebooks_root,
            runs_root=runs_root,
            benched_skill=champion_skill,
            tasks=subset_tasks,
            baseline_version=parent,
            baseline_text=rb.load_rulebook(rulebooks_root, parent).text,
            auth_mode=auth_mode,
            feedback=composed,
            log_dir=log_dir,
            stream=stream,
            deny_dirs=deny_dirs,
            log_name=f"evolve-rulebook-{candidate_version}",
            expect_version=candidate_version,
        )
        total_cost += rulebook.cost_usd

        # 3. Screen the candidate on the identical subset; accept only past the noise floor.
        candidate_screen = await bench_on(candidate_version, subset, ["test"])
        accepted = candidate_screen.passed - champion_screen.passed >= accept_delta * screen_drafts
        if accepted:
            champion = candidate_version
            accepts_since_confirm += 1

        # 4. The ratchet: every confirm_every accepts, a full bench decides — or reverts.
        confirm_ran = accepted and accepts_since_confirm >= confirm_every
        confirm_promoted: bool | None = None
        if confirm_ran:
            confirm_promoted = await confirm(champion, confirmed)
            if confirm_promoted:
                confirmed = champion
            else:
                champion = confirmed
            accepts_since_confirm = 0

        record = Generation(
            index=g,
            directive=directive,
            parent=parent,
            candidate=candidate_version,
            subset=tuple(subset),
            champion_screen=champion_screen,
            candidate_screen=candidate_screen,
            accepted=accepted,
            confirm_ran=confirm_ran,
            confirm_promoted=confirm_promoted,
            confirmed=confirmed,
            cost_usd=total_cost - gen_cost_before,
        )
        history.append(record)
        if g not in journaled:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with journal_path.open("a") as fh:
                fh.write(json.dumps(record.to_json()) + "\n")
            journaled.add(g)
        if on_generation is not None:
            on_generation(record)

    # Pending screen accepts are not left dangling: the run's champion is always confirmed.
    if champion != confirmed:
        if await confirm(champion, confirmed):
            confirmed = champion

    lockbox_drafts = baseline_drafts = None
    if evaluate_lockbox and lockbox is not None:
        lock_tasks = load_lockbox_tasks(lockbox)
        common = {
            "cfg": cfg,
            "target": target,
            "skills_root": skills_root,
            "rulebooks_root": rulebooks_root,
            "runs_root": runs_root,
            "n_drafts": n_drafts,
            "tasks": lock_tasks,
            "auth_mode": auth_mode,
            "max_concurrency": concurrency,
            "log_dir": log_dir,
            "stream": stream,
            "on_bench_start": on_bench_start,
            "on_bench_done": on_bench_done,
        }
        baseline_drafts = await _lockbox_eval(version="v1", **common)
        lockbox_drafts = baseline_drafts if confirmed == "v1" else await _lockbox_eval(version=confirmed, **common)

    return EvolveRun(
        generations=history,
        champion=confirmed,
        stopped_because=reason,
        journal_path=journal_path,
        lockbox_drafts=lockbox_drafts,
        lockbox_baseline_drafts=baseline_drafts,
        selection=selection,
        cost_usd=total_cost,
    )


# ── Islands and cross-pollination ───────────────────────────────────────────────────────

#: Subdirectory (under skills/, rulebooks/, and runs/) holding each island's independent trees.
ISLANDS_DIRNAME = "islands"


def island_root(root: Path, index: int) -> Path:
    """Where island ``index`` lives under a skills, rulebooks, or runs root."""
    return root / ISLANDS_DIRNAME / f"island-{index}"


@dataclass(frozen=True)
class IslandResult:
    """One island's finished evolution: its task pool, its champion, and where its trees live."""

    index: int
    task_ids: tuple[str, ...]
    run: EvolveRun
    rulebooks_root: Path
    skills_root: Path
    runs_root: Path


@dataclass(frozen=True)
class ArchipelagoRun:
    """k independent evolutions, one cross-pollinated merge, and the merge's honest scores."""

    islands: list[IslandResult]
    #: The merged rulebook — its rationale is the meta-rule list with supporting islands.
    merged: RulebookResult
    #: The seed's and the merged version's skills on the FULL working test split (cross-island
    #: validation: every island's surviving edits, scored on every other island's tasks too).
    seed_test: Score
    merged_test: Score
    lockbox_drafts: DraftScores | None = None
    lockbox_baseline_drafts: DraftScores | None = None
    selection: HeadroomSelection | None = None
    cost_usd: float = 0.0

    @property
    def lockbox_mean_delta(self) -> float | None:
        """Lockbox mean-rate change from the seed to the merged champion, over all drafts."""
        if self.lockbox_drafts is None or self.lockbox_baseline_drafts is None:
            return None
        return self.lockbox_drafts.mean_rate - self.lockbox_baseline_drafts.mean_rate


def _champion_chain(rulebooks_root: Path, champion: str) -> list[tuple[str, str]]:
    """The ``(parent, version)`` edges from the seed to the champion, oldest first.

    Exactly the edits that survived screens and confirmations — rejected candidates keep their
    version directories but are on nobody's parent chain.
    """
    chain: list[tuple[str, str]] = []
    cur = champion
    seen = {cur}
    while cur != "v1":
        meta = rb.rulebook_meta(rulebooks_root, cur)
        parent = meta.parent if meta is not None and meta.parent else "v1"
        if parent not in rb.available_versions(rulebooks_root) or parent in seen:
            break  # the chain left this root (an island seed's parent is a main-tree label)
        chain.append((parent, cur))
        seen.add(parent)
        cur = parent
    return list(reversed(chain))


def _write_island_materials(dest: Path, islands: Sequence[IslandResult]) -> None:
    """Lay out what the pollination agent reads: per island, champion + journal + surviving diffs."""
    for island in islands:
        d = dest / f"island-{island.index}"
        (d / "diffs").mkdir(parents=True, exist_ok=True)
        (d / "champion.md").write_text(rb.load_rulebook(island.rulebooks_root, island.run.champion).text)
        journal = island.run.journal_path
        (d / "journal.jsonl").write_text(journal.read_text() if journal.is_file() else "")
        for parent, version in _champion_chain(island.rulebooks_root, island.run.champion):
            diff = "".join(
                difflib.unified_diff(
                    rb.load_rulebook(island.rulebooks_root, parent).text.splitlines(keepends=True),
                    rb.load_rulebook(island.rulebooks_root, version).text.splitlines(keepends=True),
                    fromfile=f"island-{island.index} {parent}",
                    tofile=f"island-{island.index} {version}",
                )
            )
            (d / "diffs" / f"{parent}-to-{version}.diff").write_text(diff)


async def pollinate(
    *,
    cfg: Config,
    target: Target,
    rulebooks_root: Path,
    islands: Sequence[IslandResult],
    auth_mode: AuthMode = "session",
    parent_version: str = "v1",
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    feedback: str | None = None,
    log: LiveLog | None = None,
    deny_dirs: Sequence[Path] = (),
    prices: PriceTable | None = None,
) -> RulebookResult:
    """Merge the islands' champions into the next main-chain rulebook version via a meta-agent.

    The agent gets, per island: the champion rulebook, the full decision journal, and one unified
    diff per surviving edit (:func:`_champion_chain`). Its brief (:func:`acumen.prompts.pollinate_prompt`)
    is to keep only guidance that REPLICATED across islands — the structural reading of "always
    seems to improve" — and its rationale is the meta-rule list with supporting islands, recorded
    in the merged version's ``meta.json``. The same test guard as every improve agent applies, with
    ``deny_dirs`` covering every run tree and the lockbox.

    Raises
    ------
    LoopError
        If the merged version already exists (resume at the caller decides that), or the agent
        produced an invalid rulebook.
    """
    parent_text = rb.load_rulebook(rulebooks_root, parent_version).text
    new_version = rb.next_version(rulebooks_root)
    if rb.rulebook_dir(rulebooks_root, new_version).exists():
        raise LoopError(f"rulebook {new_version} already exists — versions are immutable")

    holder = Path(tempfile.mkdtemp(prefix="acumen-pollinate-"))
    try:
        work = holder / "work"
        staged = work / rb.RULEBOOK_FILE
        islands_dir = work / "islands"
        rationale_path = work / "rationale.md"
        home = holder / "home"
        selected_model = model or cfg.meta_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (work, islands_dir, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        staged.write_text(parent_text)
        _write_island_materials(islands_dir, islands)

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
        prompt = pollinate_prompt(
            package=target.pkg_name,
            k=len(islands),
            rulebook_path=staged,
            islands_dir=islands_dir,
            rationale_path=rationale_path,
            parent_version=parent_version,
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
            write_dirs=(work,),
            discover_skills=False,
            claude_hooks=(
                {"PreToolUse": [make_test_guard(rulebooks_root.parent, held_out_ids=(), deny_dirs=deny_dirs)]}
                if provider == "claude"
                else None
            ),
            deny_paths=tuple(d.resolve() for d in deny_dirs),
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(prompt, options=options, on_event=log.append if log is not None else None)
        except Exception as err:  # noqa: BLE001 - a failed pollination is an error to report, re-raised below
            agent_error = err
        finally:
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            if is_transient(str(agent_error)):
                raise TransientLimitError(
                    f"the platform refused the pollination agent ({str(agent_error)[:160]}); nothing was "
                    "written — rerun to resume once the limit resets"
                ) from agent_error
            raise LoopError(f"the pollination agent failed: {type(agent_error).__name__}: {agent_error}") from (
                agent_error
            )
        if result is None:
            raise LoopError("the pollination agent produced no result message")
        if result.is_error:
            detail = f"{result.subtype} {' '.join(map(str, result.errors or []))}".strip()
            if is_transient(detail):
                raise TransientLimitError(
                    f"the platform refused the pollination agent ({detail[:160]}); nothing was written — "
                    "rerun to resume once the limit resets"
                )
            raise LoopError(f"the pollination agent errored: {detail}")

        new_text = staged.read_text()
        rationale = _read_rationale(rationale_path, result, parent_version, new_version)
        try:
            written = rb.write_rulebook(
                rulebooks_root, new_version, new_text, parent=parent_version, rationale=rationale, feedback=feedback
            )
        except rb.RulebookError as err:
            raise LoopError(f"the pollination agent produced an invalid rulebook: {err}") from err

        return RulebookResult(
            version=new_version,
            parent=parent_version,
            path=written.path,
            rationale=rationale,
            changed=new_text != parent_text,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(result.usage, model=selected_model, provider=result.provider, prices=table),
            ).cost_usd,
            turns=result.num_turns,
            n_train_runs=0,
            n_train_failures=0,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


async def run_archipelago(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    k: int = 3,
    generations: int = 10,
    screen_size: int = 12,
    screen_drafts: int = 1,
    epoch_len: int = 5,
    accept_delta: int = 2,
    confirm_every: int = 5,
    n_drafts: int = 3,
    seed: int = 0,
    lockbox_dir: Path | None = None,
    allow_no_lockbox: bool = False,
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    headroom_only: bool = False,
    max_wallclock_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_island: Callable[[IslandResult], None] | None = None,
    on_generation: Callable[[Generation], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> ArchipelagoRun:
    """Evolve ``k`` islands on disjoint task partitions, cross-pollinate, validate, open the lockbox.

    Each island runs :func:`run_evolve` for ``generations`` generations on its own partition
    (:func:`acumen.folds.make_folds` held-out sets — disjoint, covering), in its own rulebook/skill/
    run trees, with every other island's runs and the whole main run tree denied to its agents —
    the islands' learnings are independent by construction, which is what makes replication across
    them evidence. Islands never open the lockbox.

    Then :func:`pollinate` merges the champions into the next main-chain version (skipped on resume
    when it already exists), the merged skill and the seed skill are benched on the FULL working
    test split — an edit that won on island A is thereby validated on islands B and C's tasks —
    and finally both are scored on the lockbox over ``n_drafts`` drafts each.

    Sequential on purpose: islands share one subscription session window; parallel islands would
    just trade pauses.
    """
    if k < 2:
        raise LoopError(f"an archipelago needs k >= 2 islands (replication is the point), got {k}")
    started = clock()
    total_cost = 0.0

    lockbox: Lockbox | None = None
    if lockbox_dir is not None:
        try:
            lockbox = read_lockbox(lockbox_dir)
            check_disjoint(tasks, lockbox)
        except FoldError as err:
            raise LoopError(str(err)) from err
    elif not allow_no_lockbox:
        raise LoopError("no lockbox given — pass --lockbox (or --no-lockbox to waive the final held-out verdict)")

    selection: HeadroomSelection | None = None
    if headroom_only:
        diffs = screen(runs_root, tasks, by_model=True)
        selection = select_headroom(diffs, tasks, split="test", models=cfg.models)
        if on_select is not None:
            on_select(selection)
        if not selection.selected:
            raise LoopError("no task has headroom to evolve on — run `acumen bench --no-skill` on more tasks first")
        tasks = selection.selected

    by_id = {t.id: t for t in tasks}
    try:
        folds = make_folds(sorted(by_id), k, seed)
    except FoldError as err:
        raise LoopError(str(err)) from err

    seed_version = rb.seed_default(rulebooks_root)
    if seed_version != "v1":
        raise LoopError(f"expected the main rulebook chain to start at v1, found {seed_version}")
    seed_text = rb.load_rulebook(rulebooks_root, "v1").text

    islands: list[IslandResult] = []
    for fold in folds:
        i = fold.index
        pool = tuple(fold.held_out)  # held-out sets are disjoint and cover the working set
        rb_root = island_root(rulebooks_root, i)
        sk_root = island_root(skills_root, i)
        rn_root = island_root(runs_root, i)
        if "v1" not in rb.available_versions(rb_root):
            rb.write_rulebook(
                rb_root,
                "v1",
                seed_text,
                parent=f"seed:{seed_version}",
                rationale=f"island {i} seed — a copy of the main rulebook {seed_version}, "
                f"evolving independently on {len(pool)} task(s)",
            )
        remaining = None
        if max_wallclock_s is not None:
            remaining = max(0.0, max_wallclock_s - (clock() - started))
        run = await run_evolve(
            cfg=cfg,
            target=target,
            skills_root=sk_root,
            rulebooks_root=rb_root,
            runs_root=rn_root,
            tasks=[by_id[t] for t in pool],
            generations=generations,
            screen_size=screen_size,
            screen_drafts=screen_drafts,
            epoch_len=epoch_len,
            accept_delta=accept_delta,
            confirm_every=confirm_every,
            n_drafts=1,
            seed=seed,
            lockbox_dir=lockbox_dir,
            allow_no_lockbox=allow_no_lockbox,
            evaluate_lockbox=False,
            auth_mode=auth_mode,
            max_concurrency=max_concurrency,
            feedback=feedback,
            log_dir=(log_dir / f"island-{i}") if log_dir is not None else None,
            stream=stream,
            headroom_only=False,
            max_wallclock_s=remaining,
            extra_deny_dirs=[runs_root],  # denies the main tree AND every sibling island's runs
            clock=clock,
            on_generation=on_generation,
            on_bench_start=on_bench_start,
            on_bench_done=on_bench_done,
        )
        total_cost += run.cost_usd
        result = IslandResult(
            index=i, task_ids=pool, run=run, rulebooks_root=rb_root, skills_root=sk_root, runs_root=rn_root
        )
        islands.append(result)
        if on_island is not None:
            on_island(result)

    # Cross-pollination into the main chain (resumed from disk if the merged version exists).
    merged_version = "v2"
    deny: list[Path] = [runs_root]
    if lockbox is not None:
        deny.append(lockbox.directory)
    if merged_version in rb.available_versions(rulebooks_root):
        resumed = rb.load_rulebook(rulebooks_root, merged_version)
        meta = rb.rulebook_meta(rulebooks_root, merged_version)
        merged = RulebookResult(
            version=merged_version,
            parent=meta.parent if meta is not None and meta.parent else "v1",
            path=resumed.path,
            rationale=(meta.rationale if meta is not None and meta.rationale else "(resumed; rationale not recorded)"),
            changed=resumed.text != seed_text,
            cost_usd=0.0,
            turns=0,
            n_train_runs=0,
            n_train_failures=0,
        )
    else:
        log = LiveLog.open(log_dir, "pollinate-v2", stream=stream) if log_dir is not None else None
        try:
            merged = await pollinate(
                cfg=cfg,
                target=target,
                rulebooks_root=rulebooks_root,
                islands=islands,
                auth_mode=auth_mode,
                parent_version="v1",
                feedback=feedback,
                log=log,
                deny_dirs=deny,
            )
        finally:
            if log is not None:
                log.close()
        total_cost += merged.cost_usd

    # Cross-island validation: seed and merged skills on the FULL working test split.
    async def full_test(version: str) -> Score:
        skill, cost = await _ensure_skill(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            rulebook_text=rb.load_rulebook(rulebooks_root, version).text,
            expect_version=version,
            auth_mode=auth_mode,
            rationale=f"drafted from rulebook {version}",
            log_dir=log_dir,
            stream=stream,
        )
        nonlocal total_cost
        total_cost += cost
        planned = await _bench(
            cfg=cfg,
            target=target,
            skill=skill,
            tasks=tasks,
            runs_root=runs_root,
            splits=["test"],
            auth_mode=auth_mode,
            task_ids=None,
            max_concurrency=max_concurrency or cfg.max_concurrency,
            on_start=on_bench_start,
            on_done=on_bench_done,
        )
        return score(runs_root, planned)

    seed_test = await full_test("v1")
    merged_test = await full_test(merged_version)

    lockbox_drafts = baseline_drafts = None
    if lockbox is not None:
        common = {
            "cfg": cfg,
            "target": target,
            "skills_root": skills_root,
            "rulebooks_root": rulebooks_root,
            "runs_root": runs_root,
            "n_drafts": n_drafts,
            "tasks": load_lockbox_tasks(lockbox),
            "auth_mode": auth_mode,
            "max_concurrency": max_concurrency or cfg.max_concurrency,
            "log_dir": log_dir,
            "stream": stream,
            "on_bench_start": on_bench_start,
            "on_bench_done": on_bench_done,
        }
        baseline_drafts = await _lockbox_eval(version="v1", **common)
        lockbox_drafts = await _lockbox_eval(version=merged_version, **common)

    return ArchipelagoRun(
        islands=islands,
        merged=merged,
        seed_test=seed_test,
        merged_test=merged_test,
        lockbox_drafts=lockbox_drafts,
        lockbox_baseline_drafts=baseline_drafts,
        selection=selection,
        cost_usd=total_cost,
    )
