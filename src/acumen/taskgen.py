"""The task-generation agent: mine the target package for real analyses and write ``tasks.yaml``.

``acumen tasks`` autonomously benchmarks the target's *functionalities* — the analyses a user
would actually run — and writes a benchmark-ready ``tasks.yaml``. Like the drafter it
reads the package **source** (it has to understand the API to design a real pipeline) and, like
no benchmark agent, it also **runs code in the venv**: every ground-truth answer is obtained by
executing the pipeline and reading the real output, never by copying doc output.

There is no test-split guard here (unlike the improver) — no runs exist yet, so there is nothing
to leak. Isolation is otherwise the same as the other meta-agents: scrubbed env, throwaway
``HOME`` and ``CLAUDE_CONFIG_DIR``.

**Existing skills must not bias task generation.** A target repo may already ship skills or
agent-instruction files (``SKILL.md``, ``.claude/skills/``, ``CLAUDE.md``, ``AGENTS.md``,
``.cursor/``, Copilot instructions). If the generator read them, it would mine the tasks the
author already anticipated and phrase them the way the skill does — defeating the point of an
independent benchmark. So, as with the test-split guard, this is enforced two ways: the agent
reads a **filtered copy** of the source with those artifacts stripped out
(:func:`build_filtered_source`), and a ``PreToolUse`` hook denies any tool call that resolves to
one of them — or to the original unfiltered tree — wherever the agent points it
(:func:`find_skill_access`). ``setting_sources=[]`` additionally means no skill is ever
discovered or loaded into the generator itself.

The script the agent writes to confirm an answer is **scratch**: it lives in a throwaway work
dir that is deleted when this returns, so nothing about the ground-truth pipeline is persisted —
only the task (prompt + answer) lands in ``tasks.yaml``. The output is validated through
:func:`acumen.tasks.parse_tasks` before it is written, so ``acumen tasks`` can never emit a
``tasks.yaml`` the rest of the pipeline would reject.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.prompts import taskgen_prompt
from acumen.tasks import Task, TaskError, load_tasks

#: The filename the generation agent writes and we harvest from its work dir.
TASKS_FILE = "tasks.yaml"

#: Basenames anywhere in the tree that are agent-facing skill/guidance artifacts. Reading any
#: of these would let existing guidance bias which tasks the generator mines, so they are both
#: stripped from the source copy the agent reads and denied by the guard hook.
_ARTIFACT_BASENAMES = frozenset({"SKILL.md", "CLAUDE.md", "AGENTS.md", "copilot-instructions.md"})

#: Directory names that hold agent guidance; dropped from the copy and denied wherever they
#: appear on a resolved path (``.claude/skills``, ``.cursor/rules``, …).
_ARTIFACT_DIRS = frozenset({".claude", ".cursor"})

#: tool_input keys carrying a filesystem path — the coarse set the improver's guard also uses.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))


class TaskGenError(RuntimeError):
    """Raised when tasks could not be generated."""


@dataclass(frozen=True)
class TaskGenResult:
    """The outcome of a generation run: the tasks written and what it cost."""

    tasks: list[Task]
    out_path: Path
    cost_usd: float
    turns: int
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Skill-bias isolation ───────────────────────────────────────────────────────────────


def _copy_ignore(root: Path):
    """A ``shutil.copytree`` ignore callback that drops skill/guidance artifacts (and ``.git``).

    Skill directories and agent-instruction files are removed so the generator physically
    cannot read them. A top-level ``skills/`` is treated as agent skills (the packaging
    convention) and dropped at the root only, so a legitimately-named source directory deeper
    in the tree is left alone.
    """
    root = root.resolve()

    def ignore(dir_path: str, names: list[str]) -> set[str]:
        here = Path(dir_path).resolve()
        drop: set[str] = set()
        for name in names:
            if name == ".git" or name in _ARTIFACT_DIRS or name in _ARTIFACT_BASENAMES:
                drop.add(name)
            elif here == root and name == "skills":
                drop.add(name)
        return drop

    return ignore


def build_filtered_source(src: Path, dest: Path) -> Path:
    """Copy ``src`` to ``dest`` with skills and agent-guidance stripped out.

    This is the structural half of the skill-bias isolation: the generator's ``add_dirs`` points
    at the returned copy, not the real checkout, so existing skills are simply absent from what
    it can read.

    Parameters
    ----------
    src
        The real target source checkout.
    dest
        Where to write the filtered copy; must not already exist.

    Returns
    -------
    ``dest``.
    """
    shutil.copytree(src, dest, ignore=_copy_ignore(src), symlinks=True)
    return dest


def _artifact_hit(candidate: str, original_src: Path) -> str | None:
    """Return ``candidate`` if it resolves to a skill/guidance artifact or the original tree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved.name in _ARTIFACT_BASENAMES:
        return candidate
    if set(resolved.parts) & _ARTIFACT_DIRS:
        return candidate
    # The unfiltered source tree is off-limits — the agent must read the filtered copy, so any
    # path back into the original checkout (which still holds the stripped artifacts) is denied.
    try:
        resolved.relative_to(original_src)
    except ValueError:
        return None
    return candidate


def find_skill_access(tool_name: str, tool_input: dict[str, Any], original_src: Path) -> str | None:
    """Return the first path in a tool call that reaches a skill/guidance artifact, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without standing up
    an agent (mirrors :func:`acumen.improve.find_test_access`). Checks the path-bearing
    tool_input keys and — for shell tools — the metacharacter-split command tokens, since a Bash
    call can name a path no structured field would.

    Parameters
    ----------
    tool_name
        The tool being invoked; unused today but kept so the guard can special-case tools.
    tool_input
        The tool's arguments.
    original_src
        The real (unfiltered) source checkout, resolved by the caller.

    Returns
    -------
    The offending path string, or ``None`` if the call touches no artifact.
    """
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            hit = _artifact_hit(value, original_src)
            if hit is not None:
                return hit
    command = tool_input.get("command")
    if isinstance(command, str):
        for raw in command.translate(_SHELL_SPLIT).split():
            token = raw.rstrip(",;")
            if not token:
                continue
            hit = _artifact_hit(token, original_src)
            if hit is not None:
                return hit
    return None


def make_skill_guard(original_src: Path) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies the generator any existing skill/guidance.

    ``matcher=None`` fires the hook for every tool. Paths are resolved against the real source
    checkout, so the guard holds regardless of the agent's ``cwd``.
    """
    root = original_src.resolve()

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_skill_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen hides existing skills and agent-instruction files from the task "
                    f"generator so they cannot bias task selection ({hit}). Design tasks from the "
                    "package's API and its user-facing docs only, using the provided source copy."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])


# ── Task serialisation ─────────────────────────────────────────────────────────────────


def _task_to_dict(task: Task) -> dict[str, object]:
    """Serialise a :class:`Task` back to the ``tasks.yaml`` mapping shape."""
    entry: dict[str, object] = {
        "id": task.id,
        "train": {"prompt": task.train.prompt, "answer": task.train.answer},
        "test": {"prompt": task.test.prompt, "answer": task.test.answer},
    }
    if task.max_turns is not None:
        entry["max_turns"] = task.max_turns
    if task.max_usd is not None:
        entry["max_usd"] = task.max_usd
    if task.model is not None:
        entry["model"] = task.model
    return entry


class _TaskDumper(yaml.SafeDumper):
    """A SafeDumper that renders multi-line strings as literal blocks, for readable prompts."""


def _represent_str(dumper: yaml.Dumper, value: str) -> yaml.ScalarNode:
    # Multi-line prompts are far easier to review as `|` literal blocks than folded/quoted.
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_TaskDumper.add_representer(str, _represent_str)


def dump_tasks(tasks: list[Task]) -> str:
    """Render tasks as ``tasks.yaml`` text, preserving key order and multi-line prompts.

    The inverse of :func:`acumen.tasks.load_tasks`. Round-tripping through this is how the
    combined (existing + generated) task set is validated before anything is written to disk.
    Multi-line prompts are emitted as ``|`` literal blocks so a reviewer can read them.
    """
    doc = {"tasks": [_task_to_dict(task) for task in tasks]}
    return yaml.dump(doc, Dumper=_TaskDumper, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100)


def _validate_generated(staged: Path) -> list[Task]:
    """Load and validate the agent's ``tasks.yaml``, mapping failures to a TaskGenError."""
    if not staged.is_file():
        raise TaskGenError(
            f"the task-generation agent did not write {TASKS_FILE} — nothing to save. "
            "Inspect the prompt or raise max_turns."
        )
    try:
        return load_tasks(staged)
    except TaskError as err:
        raise TaskGenError(f"the generated {TASKS_FILE} is not valid: {err}") from err


async def generate_tasks(
    *,
    cfg: Config,
    target: Target,
    out_path: Path,
    auth_mode: AuthMode = "session",
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    force: bool = False,
    feedback: str | None = None,
    log: LiveLog | None = None,
) -> TaskGenResult:
    """Generate a ``tasks.yaml`` for the target package by mining and executing its analyses.

    Parameters
    ----------
    cfg
        The pass config; supplies ``tasks_model``.
    target
        The prepared target, supplying the source checkout and the interpreter to run
        pipelines against for ground truth.
    auth_mode
        Which credential the generation agent authenticates with — ``"session"`` (the Claude
        subscription) or ``"api"`` (see :func:`acumen.env.build_agent_env`).
    out_path
        Where the tasks are written. Refuses to overwrite an existing file unless ``force``
        is set. There is deliberately no "append" mode: the agent generates blind to the
        existing file (it never reads it — see the module docstring on isolation), so it cannot
        avoid re-covering functionality already present, and appending would silently grow the
        set with semantic duplicates. To combine generated tasks with a curated file, write to a
        separate ``out_path`` and merge by hand, where the overlap can actually be judged.
    model
        Override for the generation model; defaults to ``cfg.tasks_model``.
    max_turns, max_usd
        Caps for the generation agent. **Unbounded by default**: generating tasks means
        running package code iteratively, so no default budget is imposed — pass explicit caps
        to bound it.
    force
        Overwrite an existing ``out_path`` (e.g. the placeholder from ``acumen init``).
    feedback
        Optional maintainer guidance, injected into the generation prompt as a subordinated
        block — e.g. which functionality to skip. Nothing but ``tasks.yaml`` is persisted (task
        generation writes no meta), so the feedback is not recorded on disk.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The generation result: the task set written and what it cost.
    """
    if out_path.exists() and not force:
        raise TaskGenError(f"{out_path} already exists — pass force=True to overwrite it")

    holder = Path(tempfile.mkdtemp(prefix="acumen-tasks-"))
    try:
        work = holder / "work"
        home = holder / "home"
        config_dir = home / ".claude"
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        staged = work / TASKS_FILE

        # The agent reads a copy of the source with skills/agent-guidance stripped, never the
        # real checkout — so existing skills cannot bias the tasks it generates.
        source_copy = build_filtered_source(target.src_dir, holder / "source")

        env = build_agent_env(
            config_dir=config_dir,
            home=home,
            extra_path=[target.bin_dir],
            auth_mode=auth_mode,
            extra_allow=cfg.env_passthrough,
        )

        prompt = taskgen_prompt(
            package=target.pkg_name,
            src=source_copy,
            python=target.python,
            out=staged,
            feedback=feedback,
        )
        options = ClaudeAgentOptions(
            cwd=str(work),
            env=env,
            model=model or cfg.tasks_model,
            # No default budget cap: only bound the agent if the caller asked.
            max_turns=max_turns,
            max_budget_usd=max_usd,
            # The generator reads the target source, like the drafter; benchmark agents
            # never do. It points at the *filtered* copy, not the real checkout.
            add_dirs=[str(source_copy)],
            # No skill discovery at all — the generator must not load a skill that would bias it.
            setting_sources=[],
            permission_mode="bypassPermissions",
            system_prompt={"type": "preset", "preset": "claude_code"},
            # Belt-and-braces over the filtered copy: deny any call that reaches an existing
            # skill/guidance artifact or the original unfiltered source, wherever pointed.
            hooks={"PreToolUse": [make_skill_guard(target.src_dir)]},
        )

        result: ResultMessage | None = None
        agent_error: Exception | None = None
        try:
            async for message in query(prompt=prompt, options=options):
                if log is not None:
                    log.append(message)
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # noqa: BLE001 - a failed generation is an error to report, re-raised below
            agent_error = err
        finally:
            # Render the HTML log while the throwaway config dir still holds the native
            # transcript — in a finally so an aborted run (the SDK raises on a cap breach,
            # after yielding the result) is still inspectable.
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

        generated = _validate_generated(staged)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dump_tasks(generated))
        return TaskGenResult(
            tasks=generated,
            out_path=out_path,
            cost_usd=result.total_cost_usd or 0.0,
            turns=result.num_turns,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        shutil.rmtree(holder, ignore_errors=True)
