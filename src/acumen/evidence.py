"""Train-split evidence for the rulebook loop, and the guard that keeps held-out material out of reach.

The rulebook improver reads how a skill *performed* on the tasks it was given — their TRAIN split
only — and never a held-out task's runs, the test split, or a lockbox. Upstream's wiki-based
``acumen improve`` has its own guard for the ``valid`` split; this module is the broader one the
cross-validated and lockbox protocols need, and the plain evidence collection behind it.

The boundary is enforced twice: structurally (the agent's readable material is a curated copy, and
the real ``runs/`` tree is not on its working directory) and by a ``PreToolUse`` hook that denies any
call naming a held-out path. See :func:`find_test_access`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acumen.improve import ImproveError
from acumen.paths import (
    ANSWER_FILE,
    RESULT_FILE,
    SCRIPT_FILE,
    TRANSCRIPT_HTML,
    TRANSCRIPT_JSONL,
    parse_run_dir,
)
from acumen.tasks import Task

if TYPE_CHECKING:
    from claude_agent_sdk import HookMatcher


#: The subtree component that marks a held-out split. A path under ``runs/<arm>/test/…`` is
#: what the improver must never reach.
_TEST_SPLIT = "test"

#: tool_input keys that carry a filesystem path across the default Claude Code toolset.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens. Coarse on
#: purpose — the structural isolation is the real guarantee; this only has to catch an
#: absolute path named directly in a command.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))


@dataclass(frozen=True)
class TrainRun:
    """One train-split benchmark run of the parent skill, as evidence for the improver.

    ``success`` and ``skill_loaded`` are independent outcomes with independent fixes: a run
    that never loaded the skill says nothing about the skill's body, only about its
    ``description``. Both reach the improver so it can tell the two apart.
    """

    task_id: str
    model: str
    rep: int
    prompt: str
    expected: str
    answer: str | None
    reason: str
    success: bool
    directory: Path
    #: Whether the agent actually invoked the skill. ``None`` when the transcript could not
    #: be read — undetermined, which is not the same as "did not load".
    skill_loaded: bool | None = None


# ── Train evidence ─────────────────────────────────────────────────────────────────────


def collect_train_runs(runs_root: Path, arm: str, tasks: list[Task]) -> list[TrainRun]:
    """Collect the parent arm's train-split runs into evidence for the improver.

    Walks ``runs_root/<arm>/train`` for ``result.json`` files and pairs each with its task's
    train prompt and ground-truth answer. Only train runs are ever collected — the test
    subtree is never even visited.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    arm
        The arm whose train runs are the improvement signal, e.g. ``"skill_v1"``.
    tasks
        The loaded tasks, to recover each run's prompt and expected answer.

    Returns
    -------
    The train runs, failures first then by task/rep — so a reader sees what went wrong up
    front.
    """
    train_root = runs_root / arm / "train"
    by_id = {task.id: task for task in tasks}
    runs: list[TrainRun] = []
    if not train_root.is_dir():
        return runs
    for result_path in sorted(train_root.rglob(RESULT_FILE)):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError) as err:
            raise ImproveError(f"cannot read {result_path}: {err}") from err
        key = parse_run_dir(runs_root, result_path.parent)
        task = by_id.get(key.task_id)
        if task is None:
            # A run for a task no longer in tasks.yaml is stale evidence; skip it rather
            # than guess a prompt for it.
            continue
        runs.append(
            TrainRun(
                task_id=key.task_id,
                model=str(data.get("model", key.model)),
                rep=key.rep,
                prompt=task.train.prompt.strip(),
                expected=task.train.answer.strip(),
                answer=data.get("answer"),
                reason=str(data.get("reason", "")),
                success=bool(data.get("success", False)),
                directory=result_path.parent,
                skill_loaded=_loaded(data.get("skill_loaded")),
            )
        )
    runs.sort(key=lambda r: (r.success, r.task_id, r.model, r.rep))
    return runs


def _loaded(value: Any) -> bool | None:
    """Coerce a ``result.json`` ``skill_loaded`` field, preserving "undetermined" as ``None``."""
    return None if value is None else bool(value)


def load_rates(runs: list[TrainRun]) -> dict[str, tuple[int, int, int]]:
    """Count how often the skill actually loaded, per model.

    Reported per model rather than pooled because the load rate varies far more across models
    than across skill versions — a pooled number invites the improver to chase a model whose
    behaviour no wording can reach.

    Parameters
    ----------
    runs
        The train runs to tally.

    Returns
    -------
    ``{model: (loaded, total, undetermined)}``, where ``undetermined`` counts runs whose
    transcript could not be read and which are therefore excluded from ``loaded``.
    """
    rates: dict[str, tuple[int, int, int]] = {}
    for run in runs:
        loaded, total, unknown = rates.get(run.model, (0, 0, 0))
        rates[run.model] = (
            loaded + (run.skill_loaded is True),
            total + 1,
            unknown + (run.skill_loaded is None),
        )
    return dict(sorted(rates.items()))


def _load_mark(loaded: bool | None) -> str:
    """The phrase naming a run's load outcome, used identically in SUMMARY.md and run.md."""
    if loaded is None:
        return "skill load UNDETERMINED"
    return "skill LOADED" if loaded else "skill NOT LOADED"


def _load_section(runs: list[TrainRun]) -> list[str]:
    """The SUMMARY.md section reporting per-model load rates."""
    rates = load_rates(runs)
    lines = [
        "## Did the skill load at all?",
        "",
        "A run where the skill never loaded is not evidence about the skill's body — the agent",
        "never read it. Only the `description` in the frontmatter decides whether a skill loads.",
        "",
        "| model | loaded | runs | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_loaded = total_runs = total_unknown = 0
    for model, (loaded, total, unknown) in rates.items():
        total_loaded += loaded
        total_runs += total
        total_unknown += unknown
        lines.append(f"| `{model}` | {loaded} | {total} | {loaded / total:.0%} |")
    if total_runs:
        lines.extend(["", f"**Overall: {total_loaded}/{total_runs} ({total_loaded / total_runs:.0%}).**"])
    if total_unknown:
        noun = "run" if total_unknown == 1 else "runs"
        lines.append(
            f"({total_unknown} {noun} had no readable transcript, so loading is undetermined and is "
            "counted as not loaded above.)"
        )
    lines.append("")
    return lines


def _write_material(train_dir: Path, runs: list[TrainRun]) -> None:
    """Lay out the curated train evidence the improver reads.

    A ``SUMMARY.md`` digest (per-model load rates, then runs with failures first) plus one
    directory per run holding the run's ``script.py``, transcript, and a ``run.md`` stating
    prompt/expected/actual/outcome and whether the skill loaded. Copied, not linked, so the
    agent never has a path back into the real runs tree.
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for r in runs if not r.success)
    noun = "run" if len(runs) == 1 else "runs"
    lines = [
        "# Train-split evidence for the current skill",
        "",
        "Each run below benchmarked the current skill on one train task. Two things were",
        "recorded per run, and they fail for different reasons and take different fixes:",
        "whether the agent got the answer RIGHT, and whether the agent LOADED the skill at",
        "all. Read the load rates first, then the failing runs. Open a run's `run.md`,",
        "`script.py`, and `transcript.html` for the detail.",
        "",
        f"**{len(runs)} {noun}, {n_fail} failing.**",
        "",
        *_load_section(runs),
        "## Runs",
        "",
    ]
    for run in runs:
        slug = f"{run.task_id}__{run.model}__rep_{run.rep}"
        mark = "PASS" if run.success else "FAIL"
        lines.append(
            f"- `{slug}/` — **{mark}** ({run.reason}) — {_load_mark(run.skill_loaded)} — "
            f"task `{run.task_id}` on `{run.model}` — expected `{run.expected}`, got `{run.answer!r}`"
        )
        run_out = train_dir / slug
        run_out.mkdir(parents=True, exist_ok=True)
        (run_out / "run.md").write_text(
            f"# {run.task_id} — {mark} ({run.reason})\n\n"
            f"- Model: `{run.model}`\n"
            f"- Skill: {_load_mark(run.skill_loaded)}\n\n"
            f"## Task prompt\n\n{run.prompt}\n\n"
            f"## Expected answer\n\n{run.expected}\n\n"
            f"## Answer the agent gave\n\n{run.answer!r}\n\n"
            f"## Files\n\n- `script.py` — the script the agent wrote\n"
            f"- `transcript.html` — the full run transcript\n"
        )
        for name in (SCRIPT_FILE, ANSWER_FILE, TRANSCRIPT_HTML, TRANSCRIPT_JSONL):
            src = run.directory / name
            if src.is_file():
                shutil.copyfile(src, run_out / name)
    (train_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ── Test-access guard ──────────────────────────────────────────────────────────────────


def _blocks(candidate: str, runs_root: Path) -> bool:
    """Return whether ``candidate`` resolves under a ``runs/*/test/`` subtree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        rel = resolved.relative_to(runs_root)
    except ValueError:
        return False
    parts = rel.parts
    # runs_root/<arm>/test/... — the split is the second component.
    return len(parts) >= 2 and parts[1] == _TEST_SPLIT


def _blocks_holdout(candidate: str, runs_root: Path, held_out_ids: frozenset[str], deny_dirs: Sequence[Path]) -> bool:
    """Whether ``candidate`` reaches a held-out *task's* runs (any split) or a denied directory.

    The CV and lockbox boundaries: a fold's improve agent may read the optimize tasks' train
    runs and nothing about the held-out tasks — not even their train runs, which name the task
    and its answer — and never the lockbox directory or any other CV fold's tree.
    """
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    for denied in deny_dirs:
        if resolved == denied or denied in resolved.parents:
            return True
    if held_out_ids:
        try:
            rel = resolved.relative_to(runs_root)
        except ValueError:
            return False
        parts = rel.parts
        # runs_root/<arm>/<split>/<model>/<task_id>/... — the task is the fourth component.
        return len(parts) >= 4 and parts[3] in held_out_ids
    return False


def find_test_access(
    tool_name: str,
    tool_input: dict[str, Any],
    runs_root: Path,
    *,
    held_out_ids: frozenset[str] = frozenset(),
    deny_dirs: Sequence[Path] = (),
) -> str | None:
    """Return the first path in a tool call that reaches held-out material, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without
    standing up an agent. Checks the path-bearing tool_input keys, and — for shell tools —
    the whitespace/metacharacter-split tokens of the command, since a Bash call can name a
    path no structured field would.

    Three things are held out, and all three are denied here: the **test split** of every task
    (always), the **runs of held-out tasks** in any split (a CV fold's held-out set, when
    ``held_out_ids`` is given), and any **denied directory** (the lockbox, other folds' trees).

    Parameters
    ----------
    tool_name
        The tool being invoked; unused today but kept so the guard can special-case tools.
    tool_input
        The tool's arguments.
    runs_root
        The project ``runs/`` root, resolved by the caller.
    held_out_ids
        Task ids whose runs are off-limits in every split.
    deny_dirs
        Resolved directories that are off-limits wholesale.

    Returns
    -------
    The offending path string, or ``None`` if the call touches nothing held out.
    """

    def hit(value: str) -> bool:
        return _blocks(value, runs_root) or _blocks_holdout(value, runs_root, held_out_ids, deny_dirs)

    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and hit(value):
            return value
    command = tool_input.get("command")
    if isinstance(command, str):
        for token in command.translate(_SHELL_SPLIT).split():
            token = token.rstrip(",;")
            if ("runs" in token or token.startswith("/") or token.startswith("~") or deny_dirs) and hit(token):
                return token
    return None


def make_test_guard(
    runs_root: Path, *, held_out_ids: Sequence[str] = (), deny_dirs: Sequence[Path] = ()
) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies any access to held-out material.

    ``matcher=None`` fires the hook for every tool. The hook resolves paths against the real
    project ``runs/`` root, so it holds regardless of the agent's ``cwd``. With ``held_out_ids``
    and ``deny_dirs`` it is the CV-fold / lockbox guard (see :func:`find_test_access`); without
    them it is the plain test-split guard every improver has always run under.
    """
    # Imported here so a Codex-only install never needs the Claude SDK.
    from claude_agent_sdk import HookMatcher

    root = runs_root.resolve()
    ids = frozenset(held_out_ids)
    denied = tuple(d.resolve() for d in deny_dirs)

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_test_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
            held_out_ids=ids,
            deny_dirs=denied,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen blocks the improver from reading held-out material ({hit}). "
                    "Only the train-split evidence of the tasks you were given is available to you."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])
