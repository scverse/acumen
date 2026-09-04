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
agent-instruction files (``SKILL.md``, ``.agents/skills/``, ``.claude/skills/``,
``CLAUDE.md``, ``AGENTS.md``,
``.cursor/``, Copilot instructions). If the generator read them, it would mine the tasks the
author already anticipated and phrase them the way the skill does — defeating the point of an
independent benchmark. So, as with the test-split guard, this is enforced two ways: the agent
reads a **filtered copy** of the source with those artifacts stripped out
(:func:`acumen.scrub.build_filtered_source`), and a ``PreToolUse`` hook denies any tool call that
resolves to one of them — or to the original unfiltered tree — wherever the agent points it
(:func:`acumen.scrub.find_skill_access`). ``setting_sources=[]`` additionally means no skill is ever
discovered or loaded into the generator itself.

The script the agent runs to confirm an answer is **kept**, as ``tasks/{id}-{split}.py`` next to
the tasks file: an answer nothing can recompute is an answer nobody can check, and a wrong one
costs a whole benchmark pass to discover. :mod:`acumen.check` reruns them on demand. Harvesting
happens only after the agent's ``tasks.yaml`` validates through :func:`acumen.tasks.parse_tasks`,
so a rejected generation leaves no scripts behind and ``acumen tasks`` can never emit a
``tasks.yaml`` the rest of the pipeline would reject.

These scripts hold the ground truth for the **held-out test split**, so no agent may ever read
them. That holds because ``bench``, ``draft`` and ``improve`` confine their agents to explicit
read roots (:mod:`acumen.guard`) that never include the project directory — a property to
preserve when changing any of them.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.check import SCRIPTS_DIRNAME, script_name, script_path
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import SPLITS, Split
from acumen.prices import PriceTable, price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import taskgen_prompt
from acumen.scrub import build_filtered_source, make_skill_guard
from acumen.tasks import Task, TaskError, load_tasks

#: The filename the generation agent writes and we harvest from its work dir.
TASKS_FILE = "tasks.yaml"


class TaskGenError(RuntimeError):
    """Raised when tasks could not be generated."""


@dataclass(frozen=True)
class TaskGenResult:
    """The outcome of a generation run: the tasks written and what it cost."""

    tasks: list[Task]
    out_path: Path
    cost_usd: float | None
    turns: int
    #: Reproducer scripts harvested into the scripts root, in task order.
    scripts: tuple[Path, ...] = ()
    #: Splits that expected a reproducer and got none, as ``(task_id, split)`` pairs. The tasks
    #: are still written: a missing script is a gap ``acumen check`` reports, not a failure.
    missing_scripts: tuple[tuple[str, Split], ...] = ()
    #: Files the agent left in its scripts directory that match no task and split.
    unexpected_scripts: tuple[str, ...] = ()
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Task serialisation ─────────────────────────────────────────────────────────────────


def _task_to_dict(task: Task) -> dict[str, object]:
    """Serialise a :class:`Task` back to the ``tasks.yaml`` mapping shape."""
    entry: dict[str, object] = {"id": task.id}
    # Beside the id rather than below the prompts: it says how to read the whole task, and after
    # two multi-line prompt blocks a reader would never see it. Only the non-default is written,
    # since `needs_script: true` on every task is noise and its absence already means that.
    if not task.needs_script:
        entry["needs_script"] = False
    entry["train"] = {"prompt": task.train.prompt, "answer": task.train.answer}
    entry["test"] = {"prompt": task.test.prompt, "answer": task.test.answer}
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


@dataclass(frozen=True)
class Harvest:
    """What :func:`harvest_scripts` moved, and what it could not."""

    scripts: tuple[Path, ...]
    missing: tuple[tuple[str, Split], ...]
    unexpected: tuple[str, ...]


def harvest_scripts(tasks: list[Task], staged_dir: Path, scripts_root: Path) -> Harvest:
    """Copy the agent's reproducer scripts out of its work dir into the project.

    Only names that match a real ``(task id, split)`` are taken, and only for tasks that declare
    they need one. Anything else the agent left there is reported rather than copied: a file
    named after no task is never run by :mod:`acumen.check`, so silently keeping it would look
    like coverage that does not exist.

    Parameters
    ----------
    tasks
        The validated tasks, read for their ids and ``needs_script``.
    staged_dir
        The agent's scripts directory inside its throwaway work dir.
    scripts_root
        Where the reproducers are kept, created if missing.

    Returns
    -------
    The scripts copied, the splits that expected one and had none, and the unmatched files.
    """
    copied: list[Path] = []
    missing: list[tuple[str, Split]] = []
    wanted: set[str] = set()
    for task in tasks:
        if not task.needs_script:
            continue
        for split in SPLITS:
            name = script_name(task.id, split)
            wanted.add(name)
            source = staged_dir / name
            if not source.is_file():
                missing.append((task.id, split))
                continue
            dest = script_path(scripts_root, task.id, split)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            copied.append(dest)
    staged = {path.name for path in staged_dir.glob("*.py")} if staged_dir.is_dir() else set()
    return Harvest(scripts=tuple(copied), missing=tuple(missing), unexpected=tuple(sorted(staged - wanted)))


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
    scripts_root: Path | None = None,
    auth_mode: AuthMode = "session",
    prices: PriceTable | None = None,
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
        The pass config; supplies ``meta_model``.
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
    scripts_root
        Where the reproducer scripts are kept. Defaults to a ``tasks/`` directory beside
        ``out_path``. Scripts are harvested only after the generated tasks validate, and only
        for the ``(task, split)`` pairs the tasks themselves declare.
    model
        Override for the generation model; defaults to ``cfg.meta_model``.
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

    scripts_root = scripts_root if scripts_root is not None else out_path.parent / SCRIPTS_DIRNAME

    holder = Path(tempfile.mkdtemp(prefix="acumen-tasks-"))
    try:
        work = holder / "work"
        home = holder / "home"
        selected_model = model or cfg.meta_model
        table = prices if prices is not None else PriceTable(overrides=cfg.prices)
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        # The scripts directory is created up front so the agent writes into a path that exists,
        # rather than having to work out that it must make one first.
        staged_scripts = work / SCRIPTS_DIRNAME
        for path in (work, home, config_dir, home / "tmp", staged_scripts):
            path.mkdir(parents=True, exist_ok=True)
        staged = work / TASKS_FILE

        # The agent reads a copy of the source with skills/agent-guidance stripped, never the
        # real checkout — so existing skills cannot bias the tasks it generates.
        source_copy = build_filtered_source(target.src_dir, holder / "source")

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

        prompt = taskgen_prompt(
            package=target.pkg_name,
            src=source_copy,
            python=target.python,
            out=staged,
            scripts_dir=staged_scripts,
            feedback=feedback,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            # No default budget cap: only bound the agent if the caller asked.
            max_turns=max_turns,
            max_usd=max_usd,
            # Codex reports no billed figure, so a budget cap needs the run's own rate table.
            price_usd=pricer(selected_model, table),
            # The generator reads the target source, like the drafter; benchmark agents
            # never do. It points at the *filtered* copy, not the real checkout.
            read_dirs=(source_copy, target.venv_dir),
            write_dirs=(work,),
            # No skill discovery at all — the generator must not load a skill that would bias it.
            discover_skills=False,
            # Belt-and-braces over the filtered copy: deny any call that reaches an existing
            # skill/guidance artifact or the original unfiltered source, wherever pointed. Built
            # only for Claude — the hook is an SDK object, and Codex gets ``deny_paths`` below.
            claude_hooks={"PreToolUse": [make_skill_guard(target.src_dir)]} if provider == "claude" else None,
            # Codex reads the filtered copy and is denied the original checkout.
            deny_paths=(target.src_dir.resolve(),),
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(
                prompt,
                options=options,
                on_event=log.append if log is not None else None,
            )
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
        # After the tasks are validated and written, so a rejected generation leaves the
        # project's scripts directory exactly as it was.
        harvest = harvest_scripts(generated, staged_scripts, scripts_root)
        return TaskGenResult(
            tasks=generated,
            out_path=out_path,
            scripts=harvest.scripts,
            missing_scripts=harvest.missing,
            unexpected_scripts=harvest.unexpected,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(
                    result.usage,
                    model=selected_model,
                    provider=result.provider,
                    prices=table,
                ),
            ).cost_usd,
            turns=result.num_turns,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        # Kill anything the agent left running before removing the directory it runs in.
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
