"""Verify that each task's ground-truth answer is reproducible by running code.

An answer in ``tasks.yaml`` is only worth benchmarking against if it is actually correct. A
typo in a hand-written answer, an answer the generation agent mis-transcribed, or a pipeline
that no longer runs because the target does not install cleanly all produce the same silent
outcome: every benchmark cell for that task fails, costs real money, and enters the report as
evidence about the model rather than as a broken task.

So each task carries a **reproducer**: ``<scripts_root>/{task_id}-{split}.py``, a script that
recomputes the answer in the target venv and writes it to ``answer.md`` in its own working
directory — the same contract a benchmark run has, so the same :func:`acumen.grade.grade_run`
judges both. ``acumen check`` runs them and reports which reproduce, which fail, and which are
absent. A task that genuinely needs no code sets ``needs_script: false`` and is reported as
``skipped`` rather than as a gap.

Reproducers are the operator's own scripts, so they run with the operator's environment (real
``HOME``, so a target's dataset cache is shared across scripts) rather than the scrubbed,
throwaway environment an agent gets. Each one still gets a fresh empty working directory, so
its ``answer.md`` cannot be another script's and nothing it writes lands in the project.

The scripts hold the ground truth for the **held-out valid split**, so nothing may ever read
them into an agent. That holds today because ``bench``, ``draft`` and ``improve`` confine their
agents to explicit read roots (:mod:`acumen.guard`) that never include the project directory.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acumen.grade import grade_run
from acumen.paths import SPLITS, Split
from acumen.tasks import Task

#: Default directory holding the reproducers, as a sibling of ``tasks.yaml``.
SCRIPTS_DIRNAME = "tasks"

#: Default seconds one reproducer may run for. Generous on purpose: a real analysis downloads
#: its own data and priors the first time it runs.
DEFAULT_TIMEOUT = 1800.0

#: Default reproducers run at once. These are data-science pipelines, so the ceiling is memory
#: rather than CPU; raise it with ``--jobs`` when the target's scripts are light.
DEFAULT_JOBS = 4

CheckStatus = Literal[
    "ok",  # answer.md reproduced the expected answer exactly
    "format_error",  # right content, formatting the exact match would reject
    "wrong_answer",  # ran cleanly and produced a different answer: the recorded answer is suspect
    "no_answer",  # ran cleanly but wrote no answer.md
    "error",  # non-zero exit
    "timeout",  # exceeded the timeout and was killed
    "missing",  # needs_script is true, but there is no script
    "skipped",  # needs_script is false, so no script is expected
]

#: Statuses that mean the ground truth was not confirmed. ``skipped`` is not among them: a task
#: that needs no code has nothing to confirm.
FAILED: frozenset[CheckStatus] = frozenset({"format_error", "wrong_answer", "no_answer", "error", "timeout", "missing"})

#: Reasons :func:`acumen.grade.grade_run` can return, mapped onto our statuses.
_GRADE_STATUS: dict[str, CheckStatus] = {
    "ok": "ok",
    "wrong_answer": "wrong_answer",
    "format_error": "format_error",
    "no_answer_file": "no_answer",
}

#: How much of a failed script's error output to keep for its report line.
_TAIL_CHARS = 400

#: Seconds the one-off import probe may take.
_PROBE_TIMEOUT = 120


class CheckError(RuntimeError):
    """Raised when the check cannot be carried out at all."""


def script_name(task_id: str, split: Split) -> str:
    """Return the reproducer filename for one task and split, e.g. ``bulk-train.py``."""
    if split not in SPLITS:
        raise CheckError(f"split must be one of {SPLITS}, got {split!r}")
    return f"{task_id}-{split}.py"


def script_path(scripts_root: Path, task_id: str, split: Split) -> Path:
    """Return where the reproducer for one task and split lives.

    The name is always *built* from the id and split, never parsed out of a filename: a task id
    may itself contain ``-``, so splitting on it would be ambiguous.
    """
    return scripts_root / script_name(task_id, split)


def select_tasks(tasks: Sequence[Task], task_ids: Sequence[str] | None) -> list[Task]:
    """Restrict ``tasks`` to the named ids, in file order.

    An id that matches nothing is an error rather than an empty selection, mirroring
    :func:`acumen.bench.build_matrix`: a typo in ``--task`` should not read as "everything
    checked out fine".
    """
    if not task_ids:
        return list(tasks)
    wanted = set(task_ids)
    known = {task.id for task in tasks}
    unknown = wanted - known
    if unknown:
        raise CheckError(f"unknown task ids: {sorted(unknown)} (known: {sorted(known)})")
    return [task for task in tasks if task.id in wanted]


def orphan_scripts(tasks: Sequence[Task], scripts_root: Path) -> list[Path]:
    """Return the ``.py`` files in ``scripts_root`` that match no task and split.

    A renamed task id leaves its old reproducer behind, where it looks like coverage and is
    never run. Reporting them is how that stays visible.
    """
    if not scripts_root.is_dir():
        return []
    expected = {script_name(task.id, split) for task in tasks for split in SPLITS}
    return sorted(path for path in scripts_root.glob("*.py") if path.name not in expected)


@dataclass(frozen=True)
class ScriptRun:
    """What running one reproducer produced, before it is attributed to a task."""

    status: CheckStatus
    #: What the script wrote to ``answer.md``, or ``None`` if it wrote nothing.
    answer: str | None
    seconds: float
    returncode: int | None = None
    #: The tail of the script's error output, kept only when something went wrong.
    error_tail: str | None = None


@dataclass(frozen=True)
class CheckResult:
    """The outcome of checking one task's split."""

    task_id: str
    split: Split
    status: CheckStatus
    #: The reproducer that ran, or ``None`` when the task needs none.
    script: Path | None
    answer: str | None
    expected: str
    seconds: float
    returncode: int | None = None
    error_tail: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this split's ground truth was confirmed, or needed no confirming."""
        return self.status not in FAILED

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-serialisable data, for ``acumen check --json``."""
        return {
            "task_id": self.task_id,
            "split": self.split,
            "status": self.status,
            "script": None if self.script is None else str(self.script),
            "answer": self.answer,
            "expected": self.expected,
            "seconds": round(self.seconds, 2),
            "returncode": self.returncode,
            "error_tail": self.error_tail,
        }


@dataclass(frozen=True)
class CheckSummary:
    """Aggregate statistics over a set of :class:`CheckResult`."""

    #: Every (task, split) pair checked.
    n_splits: int
    #: Those belonging to a task with ``needs_script`` set, i.e. those a reproducer must cover.
    n_code_splits: int
    #: Code splits that had a reproducer to run.
    n_with_script: int
    #: Code splits whose answer reproduced exactly.
    n_ok: int
    by_status: dict[CheckStatus, int]
    #: Tasks with ``needs_script`` set, and how many reproduced on every checked split.
    n_code_tasks: int
    n_tasks_reproduced: int

    @property
    def n_non_code_splits(self) -> int:
        """Splits belonging to a task that needs no reproducer."""
        return self.n_splits - self.n_code_splits

    @property
    def pct_with_script(self) -> float | None:
        """Percentage of code splits that have a reproducer, or ``None`` if there are none."""
        return _pct(self.n_with_script, self.n_code_splits)

    @property
    def pct_reproduced(self) -> float | None:
        """Percentage of code splits whose answer reproduced."""
        return _pct(self.n_ok, self.n_code_splits)

    @property
    def pct_tasks_reproduced(self) -> float | None:
        """Percentage of code tasks that reproduced on every checked split."""
        return _pct(self.n_tasks_reproduced, self.n_code_tasks)

    @property
    def ok(self) -> bool:
        """Whether every code split reproduced, which is what the exit code follows."""
        return self.n_ok == self.n_code_splits

    def to_dict(self) -> dict[str, object]:
        """Render as JSON-serialisable data, for ``acumen check --json``."""
        return {
            "splits": self.n_splits,
            "code_splits": self.n_code_splits,
            "non_code_splits": self.n_non_code_splits,
            "with_script": self.n_with_script,
            "reproduced": self.n_ok,
            "pct_with_script": self.pct_with_script,
            "pct_reproduced": self.pct_reproduced,
            "code_tasks": self.n_code_tasks,
            "tasks_reproduced": self.n_tasks_reproduced,
            "pct_tasks_reproduced": self.pct_tasks_reproduced,
            "by_status": dict(self.by_status),
            "ok": self.ok,
        }


def _pct(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(100.0 * part / whole, 1)


def _tail(text: str | None) -> str | None:
    """Return the end of a script's output, or ``None`` if it produced none."""
    stripped = (text or "").strip()
    return stripped[-_TAIL_CHARS:].strip() or None


def script_env(bin_dir: Path) -> dict[str, str]:
    """Build the environment a reproducer runs in.

    Deliberately the operator's own environment with the target venv prepended to ``PATH``,
    not the scrubbed agent environment from :func:`acumen.env.build_agent_env`. These scripts
    are the project's own ground truth rather than something under test, and keeping the real
    ``HOME`` means the target's dataset and prior-knowledge caches are shared across every
    script instead of being re-downloaded once per run.

    Parameters
    ----------
    bin_dir
        The target venv's ``bin`` directory.

    Returns
    -------
    The environment to pass to the subprocess.
    """
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")]).rstrip(os.pathsep)
    env["VIRTUAL_ENV"] = str(bin_dir.parent)
    # A reproducer that plots must not try to open a window on a headless machine.
    env.setdefault("MPLBACKEND", "Agg")
    return env


#: Probe body run in the target venv. ``prepare_target`` already proved the *distribution* is
#: installed (it reads its metadata version), which is not the same as the package importing:
#: a missing compiled extension or a broken transitive dependency only shows up on import. The
#: import name can differ from the distribution name (``scikit-learn`` imports as ``sklearn``),
#: so the module names come from the distribution's own metadata, with the normalised
#: distribution name as a fallback for a package that ships no ``top_level.txt``.
_PROBE_CODE = """\
import importlib
import importlib.metadata as md
import sys

dist_name = sys.argv[1]
candidates = []
try:
    dist = md.distribution(dist_name)
    candidates += (dist.read_text("top_level.txt") or "").split()
except Exception:
    dist = None
candidates.append(dist_name.replace("-", "_"))

errors = []
for name in dict.fromkeys(candidates):
    if name.startswith("_"):
        continue
    try:
        module = importlib.import_module(name)
    except Exception as err:
        errors.append(f"import {name}: {type(err).__name__}: {err}")
        continue
    version = getattr(module, "__version__", None)
    if not version:
        try:
            version = md.version(dist_name)
        except Exception:
            version = "unknown version"
    print(f"{name} {version}")
    raise SystemExit(0)

print("\\n".join(errors) or f"no importable module found for {dist_name}", file=sys.stderr)
raise SystemExit(1)
"""


def import_probe(python: Path, package: str) -> tuple[bool, str]:
    """Check that the target package imports in its venv.

    The most common way for every task to fail at once is that the package did not install
    cleanly, and running every reproducer to learn that is a waste. This answers it in one
    subprocess before anything else runs.

    Parameters
    ----------
    python
        The target venv interpreter.
    package
        The target's distribution name (``Target.pkg_name``); the import name is derived from
        its metadata.

    Returns
    -------
    ``(ok, detail)``: on success ``"<module> <version>"``, otherwise the error output.
    """
    try:
        proc = subprocess.run(
            # Not resolved, for the same reason as in :func:`run_reproducer`: following a venv's
            # ``bin/python`` symlink would probe the base interpreter instead.
            [str(python.absolute()), "-c", _PROBE_CODE, package],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            env=script_env(python.parent),
        )
    except OSError as err:
        return False, f"cannot run {python}: {err}"
    except subprocess.TimeoutExpired:
        return False, f"importing {package} timed out after {_PROBE_TIMEOUT}s"
    if proc.returncode != 0:
        return False, _tail(proc.stderr or proc.stdout) or f"exit {proc.returncode}"
    return True, proc.stdout.strip()


def _terminate(proc: subprocess.Popen) -> None:
    """Kill a timed-out reproducer and everything it started.

    ``start_new_session=True`` made the script a session leader, so its pid is also its process
    group id: killing the group reaches the children it spawned, which killing the script alone
    would leave running against a working directory about to be removed.
    """
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows has no process groups to signal
            proc.kill()
    except (OSError, ProcessLookupError):  # already exited between the timeout and the signal
        pass


def run_reproducer(
    script: Path,
    *,
    python: Path,
    expected: str,
    work_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> ScriptRun:
    """Run one reproducer in ``work_dir`` and grade the ``answer.md`` it wrote.

    Parameters
    ----------
    script
        The reproducer to run.
    python
        The target venv interpreter, so the script sees exactly the packages a benchmark run
        would.
    expected
        The answer recorded in ``tasks.yaml``.
    work_dir
        An empty directory to run in; the script's ``answer.md`` is read from here.
    timeout
        Seconds before the script and its children are killed.

    Returns
    -------
    What the run produced, graded.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Absolute, because the script runs with cwd set to the work dir: a relative path such as
    # ``tasks/bulk-train.py`` would otherwise be looked up inside that empty directory.
    #
    # Absolute but NOT resolved for the interpreter: a venv's ``bin/python`` is a symlink to the
    # base interpreter, and Python finds its ``pyvenv.cfg`` (and therefore the venv's
    # site-packages) from the path it was invoked by. Following the symlink would run the base
    # interpreter instead, where the target package is not installed at all.
    command = [str(python.absolute()), str(script.resolve())]
    try:
        # The operator's own reproducer, run in the target venv at their request.
        proc = subprocess.Popen(
            command,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=script_env(python.parent),
            # Its own session, so _terminate can reach whatever the script spawned.
            start_new_session=True,
        )
    except OSError as err:
        return ScriptRun(
            status="error", answer=None, seconds=time.monotonic() - started, error_tail=f"cannot run {script}: {err}"
        )

    timed_out = False
    with proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(proc)
            stdout, stderr = proc.communicate()
    seconds = time.monotonic() - started

    grade = grade_run(work_dir, expected)
    if timed_out:
        return ScriptRun(status="timeout", answer=grade.answer, seconds=seconds, error_tail=_tail(stderr))
    if proc.returncode != 0:
        # A broken pipeline is a failure even if answer.md happens to hold the right string:
        # whatever wrote it did not finish, so the answer is not evidence of anything.
        return ScriptRun(
            status="error",
            answer=grade.answer,
            seconds=seconds,
            returncode=proc.returncode,
            error_tail=_tail(stderr or stdout),
        )
    status = _GRADE_STATUS[grade.reason]
    return ScriptRun(
        status=status,
        answer=grade.answer,
        seconds=seconds,
        returncode=proc.returncode,
        error_tail=None if status == "ok" else _tail(stderr),
    )


def check_task_split(
    task: Task,
    split: Split,
    *,
    scripts_root: Path,
    python: Path,
    work_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> CheckResult:
    """Check one split of one task: locate its reproducer, run it, and grade what it wrote."""
    expected = task.split(split).answer.strip()
    if not task.needs_script:
        return CheckResult(
            task_id=task.id, split=split, status="skipped", script=None, answer=None, expected=expected, seconds=0.0
        )
    script = script_path(scripts_root, task.id, split)
    if not script.is_file():
        return CheckResult(
            task_id=task.id, split=split, status="missing", script=script, answer=None, expected=expected, seconds=0.0
        )
    run = run_reproducer(script, python=python, expected=expected, work_dir=work_dir, timeout=timeout)
    return CheckResult(
        task_id=task.id,
        split=split,
        status=run.status,
        script=script,
        answer=run.answer,
        expected=expected,
        seconds=run.seconds,
        returncode=run.returncode,
        error_tail=run.error_tail,
    )


def check_tasks(
    tasks: Sequence[Task],
    *,
    scripts_root: Path,
    python: Path,
    splits: Sequence[Split] = SPLITS,
    timeout: float = DEFAULT_TIMEOUT,
    jobs: int = DEFAULT_JOBS,
    work_root: Path | None = None,
    on_done: Callable[[CheckResult], None] | None = None,
) -> list[CheckResult]:
    """Check every task and split, running the reproducers that exist.

    Parameters
    ----------
    tasks
        The tasks to check, already filtered by the caller.
    scripts_root
        The directory holding the reproducers.
    python
        The target venv interpreter.
    splits
        Which splits to check.
    timeout
        Seconds any one reproducer may run for.
    jobs
        How many reproducers to run at once.
    work_root
        Directory to create the per-script working directories in. When given, the caller owns
        it and it is left on disk, which is how ``--keep`` preserves what a failing script
        wrote. When ``None``, a temporary one is used and removed afterwards.
    on_done
        Called with each result as it completes, so a caller can stream progress. Results are
        returned in task order regardless of completion order.

    Returns
    -------
    One result per (task, split), in task order then split order.
    """
    if jobs < 1:
        raise CheckError(f"jobs must be >= 1, got {jobs}")
    for split in splits:
        if split not in SPLITS:
            raise CheckError(f"split must be one of {SPLITS}, got {split!r}")

    holder = work_root if work_root is not None else Path(tempfile.mkdtemp(prefix="acumen-check-"))
    holder.mkdir(parents=True, exist_ok=True)
    items = [(task, split) for task in tasks for split in splits]

    def run(item: tuple[Task, Split]) -> CheckResult:
        task, split = item
        result = check_task_split(
            task,
            split,
            scripts_root=scripts_root,
            python=python,
            work_dir=holder / f"{task.id}-{split}",
            timeout=timeout,
        )
        if on_done is not None:
            on_done(result)
        return result

    try:
        if jobs == 1:
            return [run(item) for item in items]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(run, items))
    finally:
        if work_root is None:
            shutil.rmtree(holder, ignore_errors=True)


def summarize_checks(results: Iterable[CheckResult], tasks: Sequence[Task]) -> CheckSummary:
    """Aggregate results into the statistics ``acumen check`` prints.

    Parameters
    ----------
    results
        The per-split results.
    tasks
        The tasks they came from, read for ``needs_script``.

    Returns
    -------
    The summary. Percentages are ``None`` rather than ``0`` when their denominator is empty, so
    "no code tasks at all" never reads as "nothing reproduced".
    """
    collected = list(results)
    needs = {task.id: task.needs_script for task in tasks}
    by_status: dict[CheckStatus, int] = {}
    per_task: dict[str, list[CheckResult]] = defaultdict(list)
    code: list[CheckResult] = []
    for result in collected:
        by_status[result.status] = by_status.get(result.status, 0) + 1
        if needs.get(result.task_id, True):
            code.append(result)
            per_task[result.task_id].append(result)

    return CheckSummary(
        n_splits=len(collected),
        n_code_splits=len(code),
        n_with_script=sum(1 for result in code if result.status != "missing"),
        n_ok=sum(1 for result in code if result.status == "ok"),
        by_status=by_status,
        n_code_tasks=len(per_task),
        n_tasks_reproduced=sum(1 for group in per_task.values() if all(item.status == "ok" for item in group)),
    )
