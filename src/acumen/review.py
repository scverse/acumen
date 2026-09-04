"""The task-review agent: does each task's prompt, answer, and reproducer agree?

:mod:`acumen.check` answers one question deterministically — does the recorded answer still come
out of running the code? A task can pass that and still be broken, because the *prompt* can
describe something else. A real example: a prompt asking for the three most deactivated pathways
"sorted by score (ascending)" whose script and recorded answer are both in descending order. The
script ran, the answer reproduced, the deterministic check said ``ok`` — and every agent that read
the prompt correctly produced the reverse order and was graded wrong. That pass measured the
task's phrasing, not the model.

So one agent reads the whole task set — each split's prompt, its recorded answer, its reproducer,
and what happened when acumen ran it — and returns ``ok`` or ``mismatch`` per split, with one
short line naming the contradiction and one naming the fix. It never edits ``tasks.yaml``: the
repair is the maintainer's call, and a reviewer that rewrote prompts could quietly reshape the
benchmark.

The agent reads a **staged packet** (:func:`write_packet`), not the project: copies of the
prompts, answers and scripts, so it has no path back into ``tasks.yaml`` or the real ``tasks/``
tree and cannot write to either. It does get the target's source and venv, since deciding whether
a prompt describes what a script computes sometimes needs the package's own semantics.

Its output is parsed defensively (:func:`parse_reviews`): a sloppy or partial verdict file
degrades to ``unreviewed`` rows plus warnings rather than discarding a good deterministic run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.check import CheckResult
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import Split
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import review_prompt
from acumen.tasks import Task

#: The verdict file the agent writes into its work dir, and we harvest.
REVIEW_FILE = "review.json"

#: The staged packet directory inside the agent's work dir.
PACKET_DIRNAME = "review"

#: Subdirectory of the packet holding copies of the reproducers.
PACKET_SCRIPTS = "scripts"

#: The digest the agent reads first.
PACKET_DIGEST = "TASKS.md"

#: Longest ``issue``/``fix`` kept. The prompt asks for one short clause; this is what makes the
#: table stay readable when the agent writes a paragraph anyway.
MAX_NOTE_CHARS = 140

ReviewStatus = Literal[
    "ok",  # prompt, answer and script agree
    "mismatch",  # they contradict each other; issue/fix say how
    "unreviewed",  # the review was skipped, or the agent returned no verdict for this split
]

#: The verdicts the agent is allowed to return. ``unreviewed`` is ours to assign, never its.
_AGENT_VERDICTS = frozenset({"ok", "mismatch"})


class ReviewError(RuntimeError):
    """Raised when the review could not be carried out at all."""


@dataclass(frozen=True)
class ReviewVerdict:
    """One split's review outcome."""

    task_id: str
    split: Split
    status: ReviewStatus
    #: What contradicts what, in one clause. ``None`` for ``ok`` and ``unreviewed``.
    issue: str | None = None
    #: What to change, in one clause. Never a rewritten prompt.
    fix: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this split was not flagged. An unreviewed split is not a failure."""
        return self.status != "mismatch"

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-serialisable data, for ``acumen check --json``."""
        return {"verdict": self.status, "issue": self.issue, "fix": self.fix}


@dataclass(frozen=True)
class ReviewResult:
    """What the review run produced, and what it cost."""

    verdicts: dict[tuple[str, str], ReviewVerdict]
    cost_usd: float | None
    turns: int
    #: Rows the agent left out, entries naming no reviewed split, and notes it wrote too long.
    warnings: tuple[str, ...] = ()
    log_jsonl: Path | None = None
    log_html: Path | None = None

    @property
    def flagged(self) -> list[ReviewVerdict]:
        """Every split the review called a mismatch, in the order the verdicts were built."""
        return [verdict for verdict in self.verdicts.values() if verdict.status == "mismatch"]

    def status_for(self, task_id: str, split: Split) -> ReviewStatus:
        """The verdict for one split, or ``unreviewed`` if the review produced none."""
        verdict = self.verdicts.get((task_id, split))
        return verdict.status if verdict is not None else "unreviewed"


# ── The packet the agent reads ─────────────────────────────────────────────────────────


def _one_line(text: str) -> str:
    """Collapse whitespace and truncate, so one row of the table stays one row."""
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_NOTE_CHARS else flat[: MAX_NOTE_CHARS - 1].rstrip() + "…"


def _outcome_line(result: CheckResult) -> str:
    """State what happened when acumen ran this split's reproducer, in the agent's terms."""
    if result.status == "ok":
        return "The script ran and reproduced the recorded answer exactly."
    if result.status == "wrong_answer":
        return (
            f"The script ran cleanly but produced `{result.answer}`, NOT the recorded answer. "
            "One of the three artifacts is stale — say which."
        )
    if result.status == "format_error":
        return (
            f"The script ran and produced `{result.answer}`: the same content as the recorded "
            "answer but formatted differently."
        )
    if result.status == "no_answer":
        return "The script ran but wrote no answer, so it cannot be compared with the recorded one."
    if result.status == "error":
        return "The script failed to run, so only the prompt and the recorded answer can be compared."
    if result.status == "timeout":
        return "The script was still running when it was killed, so its answer is unknown."
    if result.status == "missing":
        return "There is no reproducer for this split. Compare the prompt and the recorded answer alone."
    return "This task is marked as needing no code. Compare the prompt and the recorded answer alone."


def write_packet(packet_dir: Path, tasks: list[Task], results: list[CheckResult]) -> list[CheckResult]:
    """Lay out the review material: a digest plus copies of the reproducers.

    Modelled on :func:`acumen.improve._write_material`. Scripts are **copied**, and the digest
    names them by their packet-relative filename only, so nothing in what the agent reads points
    back at the project's ``tasks/`` directory or at ``tasks.yaml``.

    Parameters
    ----------
    packet_dir
        Where to write the packet; created if missing.
    tasks
        The tasks under review, for their prompts and answers.
    results
        The deterministic results, which supply each split's outcome and the script to copy.

    Returns
    -------
    The results actually written into the packet, in digest order — the set the agent is expected
    to return a verdict for.
    """
    scripts_dir = packet_dir / PACKET_SCRIPTS
    scripts_dir.mkdir(parents=True, exist_ok=True)
    by_task = {task.id: task for task in tasks}

    lines = [
        "# Task set under review",
        "",
        "One section per task split below. For each, decide whether the PROMPT, the RECORDED",
        "ANSWER, and the SCRIPT describe the same thing.",
        "",
        "**The `train` and `test` splits of a task differ on purpose.** They are two instances of",
        "one analysis with two different answers, deliberately asking about different groups,",
        "conditions, datasets, or directions. A difference between the two is the design of the",
        "benchmark, not a defect — judge each split on its own terms.",
        "",
    ]
    written: list[CheckResult] = []
    for result in results:
        task = by_task.get(result.task_id)
        if task is None:  # pragma: no cover - the caller passes results built from these tasks
            continue
        written.append(result)
        script_note = "no script for this split"
        if result.script is not None and result.script.is_file():
            name = result.script.name
            shutil.copyfile(result.script, scripts_dir / name)
            script_note = f"`{PACKET_SCRIPTS}/{name}`"
        lines += [
            f"## {result.task_id} / {result.split}",
            "",
            "### Prompt",
            "",
            task.split(result.split).prompt.strip(),
            "",
            "### Recorded answer",
            "",
            "```",
            result.expected,
            "```",
            "",
            "### What happened when acumen ran the script",
            "",
            _outcome_line(result),
            "",
            f"Script: {script_note}",
            "",
        ]
    (packet_dir / PACKET_DIGEST).write_text("\n".join(lines) + "\n")
    return written


# ── The agent's verdicts ───────────────────────────────────────────────────────────────


def parse_reviews(raw: Any, reviewed: list[CheckResult]) -> tuple[list[ReviewVerdict], list[str]]:
    """Turn the agent's ``review.json`` into verdicts, defensively.

    Nothing here raises on a bad entry. A review is a judgement layered on top of a deterministic
    result that already stands on its own, so a malformed or partial verdict file must degrade to
    ``unreviewed`` rows and warnings rather than discard the run.

    Parameters
    ----------
    raw
        The parsed JSON document.
    reviewed
        The splits the agent was asked about; anything it names outside this set is dropped, and
        anything in it the agent omitted comes back ``unreviewed``.

    Returns
    -------
    ``(verdicts, warnings)`` — one verdict per entry in ``reviewed``, in that order, and the
    human-readable complaints about what the agent got wrong.
    """
    warnings: list[str] = []
    wanted = [(result.task_id, result.split) for result in reviewed]
    found: dict[tuple[str, str], ReviewVerdict] = {}

    entries = raw.get("reviews") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return (
            [ReviewVerdict(task_id=task_id, split=split, status="unreviewed") for task_id, split in wanted],
            [f"the review agent wrote no 'reviews' list to {REVIEW_FILE}, so no split was reviewed"],
        )

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            warnings.append(f"review entry {index} is not a mapping, ignored")
            continue
        task_id, split = entry.get("task"), entry.get("split")
        key = (task_id, split)
        if not isinstance(task_id, str) or not isinstance(split, str) or key not in wanted:
            warnings.append(f"review entry {index} names {task_id!r}/{split!r}, which was not under review, ignored")
            continue
        if key in found:
            warnings.append(f"{task_id}/{split} was reviewed twice; keeping the first verdict")
            continue
        verdict = entry.get("verdict")
        if verdict not in _AGENT_VERDICTS:
            warnings.append(f"{task_id}/{split} has verdict {verdict!r}, which is not 'ok' or 'mismatch', ignored")
            continue
        issue = entry.get("issue")
        fix = entry.get("fix")
        issue = _one_line(issue) if isinstance(issue, str) and issue.strip() else None
        fix = _one_line(fix) if isinstance(fix, str) and fix.strip() else None
        if verdict == "mismatch" and issue is None:
            # Kept, because the flag is still a signal, but a verdict with no reason cannot be
            # acted on and that is the review agent's failure, not the task's.
            warnings.append(f"{task_id}/{split} was flagged with no reason given")
        if verdict == "ok":
            issue = fix = None  # nothing to say about a task that holds together
        found[key] = ReviewVerdict(task_id=task_id, split=split, status=verdict, issue=issue, fix=fix)  # type: ignore[arg-type]

    missing = [key for key in wanted if key not in found]
    if missing:
        named = ", ".join(f"{task_id}/{split}" for task_id, split in missing)
        warnings.append(f"the review agent returned no verdict for {named}")
    verdicts = [
        found.get(key) or ReviewVerdict(task_id=key[0], split=key[1], status="unreviewed")  # type: ignore[arg-type]
        for key in wanted
    ]
    return verdicts, warnings


def _load_verdicts(staged: Path, reviewed: list[CheckResult]) -> tuple[list[ReviewVerdict], list[str]]:
    """Read and parse the agent's verdict file, or fail if it wrote none."""
    if not staged.is_file():
        raise ReviewError(
            f"the review agent did not write {REVIEW_FILE} — no split was reviewed. Inspect the "
            "run log, or rerun with --no-review to take the deterministic check alone."
        )
    try:
        raw = json.loads(staged.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ReviewError(f"the review agent's {REVIEW_FILE} is not readable JSON: {err}") from err
    return parse_reviews(raw, reviewed)


async def review_tasks(
    *,
    cfg: Config,
    target: Target,
    tasks: list[Task],
    results: list[CheckResult],
    auth_mode: AuthMode = "session",
    prices: PriceTable | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    log: LiveLog | None = None,
) -> ReviewResult:
    """Review a whole task set for agreement between prompt, answer, and reproducer.

    One agent covers every split, so it costs a single run however many tasks there are.

    Parameters
    ----------
    cfg
        The pass config; supplies ``meta_model``.
    target
        The prepared target. The reviewer reads its source and venv, since judging whether a
        prompt describes what a script computes can need the package's own semantics.
    tasks
        The tasks under review, for their prompts and answers.
    results
        The deterministic results from :func:`acumen.check.check_tasks`, which tell the reviewer
        what running each script actually produced.
    auth_mode
        Which credential the review agent authenticates with (see
        :func:`acumen.env.build_agent_env`).
    model
        Override for the review model; defaults to ``cfg.meta_model``.
    max_turns, max_usd
        Caps for the review agent. Unbounded by default, like every other meta-agent.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The verdicts, keyed by ``(task_id, split)``, plus cost and any complaints about the agent's
    output.

    Raises
    ------
    ReviewError
        If the agent failed, errored, or wrote no verdict file. The caller still holds the
        deterministic results, which stand on their own.
    """
    if not results:
        raise ReviewError("there is nothing to review")

    holder = Path(tempfile.mkdtemp(prefix="acumen-review-"))
    try:
        work = holder / "work"
        home = holder / "home"
        selected_model = model or cfg.meta_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        packet = work / PACKET_DIRNAME
        for path in (work, home, config_dir, home / "tmp", packet):
            path.mkdir(parents=True, exist_ok=True)
        staged = work / REVIEW_FILE

        # Copies, so nothing the agent reads points back at tasks.yaml or the project's tasks/.
        reviewed = write_packet(packet, tasks, results)

        # Marks the agent's processes so the teardown below can find what it leaves running.
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

        prompt = review_prompt(
            package=target.pkg_name,
            version=target.pkg_version,
            src=target.src_dir,
            python=target.python,
            packet_dir=packet,
            out=staged,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            # No default cap: only bound the agent if the caller asked, as with every meta-agent.
            max_turns=max_turns,
            max_usd=max_usd,
            # Codex reports no billed figure, so a budget cap needs the run's own rate table.
            price_usd=pricer(selected_model, table),
            # Judging a prompt against a script can need the package's semantics: which direction
            # a statistic runs, what a function returns. Same access the drafter gets.
            read_dirs=(target.src_dir, target.venv_dir),
            write_dirs=(work,),
            # A skill shipped by the target must not colour the judgement of the tasks that
            # measure it.
            discover_skills=False,
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(prompt, options=options, on_event=log.append if log is not None else None)
        except Exception as err:  # noqa: BLE001 - a failed review is an error to report, re-raised below
            agent_error = err
        finally:
            # Render the HTML log while the throwaway config dir still holds the native
            # transcript — in a finally so an aborted run stays inspectable.
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            raise ReviewError(f"the review agent failed: {type(agent_error).__name__}: {agent_error}") from agent_error
        if result is None:
            raise ReviewError("the review agent produced no result message")
        if result.is_error:
            raise ReviewError(f"the review agent errored: {result.subtype} {result.errors or ''}".strip())

        verdicts, warnings = _load_verdicts(staged, reviewed)
        return ReviewResult(
            verdicts={(verdict.task_id, verdict.split): verdict for verdict in verdicts},
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(result.usage, model=selected_model, provider=result.provider, prices=table),
            ).cost_usd,
            turns=result.num_turns,
            warnings=tuple(warnings),
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        # Kill anything the agent left running before removing the directory it runs in.
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
