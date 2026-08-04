"""The drafting agent: write ``skills/v1/`` from the target package's source.

Unlike a benchmark agent, the drafter reads the target's source — it is documenting
the package, so it needs to see it. It is otherwise held to the same isolation: scrubbed
env, throwaway config dir, no user settings or memories.

The agent writes into a staging directory, not into ``skills/`` directly. Only a skill
that loads and validates is promoted to a version, so a failed or half-finished draft
never leaves a broken ``skills/vN/`` behind — versions are immutable, which means
they must be right the moment they appear.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from acumen.agents import AgentOptions, AgentResult, provider_for_model, run_agent
from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.prices import price_usage, pricer, resolve_cost
from acumen.procs import label_env, reap
from acumen.prompts import draft_prompt
from acumen.skills import (
    SKILL_FILE,
    Skill,
    SkillError,
    load_skill,
    next_version,
    skill_dir,
    write_meta,
)


class DraftError(RuntimeError):
    """Raised when a skill could not be drafted."""


@dataclass(frozen=True)
class DraftResult:
    """A drafted skill version and what it cost to produce."""

    skill: Skill
    cost_usd: float | None
    turns: int
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


def _validate_staged(staging: Path, skill_name: str) -> None:
    """Fail loudly if the agent's output isn't a usable skill, before it becomes a version."""
    if not (staging / SKILL_FILE).is_file():
        raise DraftError(
            f"the drafting agent did not write {SKILL_FILE} — nothing to promote. Inspect the run log or the prompt."
        )
    # load_skill enforces the frontmatter contract (name matches, description present).
    try:
        load_skill(staging.parent, staging.name, expect_name=skill_name)
    except SkillError as err:
        raise DraftError(f"the drafted skill is not valid: {err}") from err


async def draft_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    auth_mode: AuthMode = "session",
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    rationale: str = "initial draft",
    feedback: str | None = None,
    log: LiveLog | None = None,
) -> DraftResult:
    """Draft a new skill version from the target package's source.

    Parameters
    ----------
    cfg
        The pass config; supplies ``skill_name`` and ``draft_model``.
    target
        The prepared target, supplying the source checkout and interpreter.
    skills_root
        The ``skills/`` root. The new version is the next unused one.
    auth_mode
        Which credential the drafting agent authenticates with — ``"session"`` (the Claude
        subscription) or ``"api"`` (see :func:`acumen.env.build_agent_env`).
    model
        Override for the drafting model; defaults to the config's.
    max_turns, max_usd
        Turn and budget caps for the drafting agent. **Unbounded by default** — the config's
        ``max_turns``/``max_usd`` cap benchmark agents only; pass explicit caps to bound the
        drafter.
    rationale
        Recorded in ``meta.json`` as why this version exists.
    feedback
        Optional maintainer guidance, injected into the draft prompt as a subordinated block
        and recorded in ``meta.json`` as provenance.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The drafted skill, loaded and validated.
    """
    version = next_version(skills_root)
    dest = skill_dir(skills_root, version)
    if dest.exists():
        raise DraftError(f"{dest} already exists — skill versions are immutable and never overwritten")

    holder = Path(tempfile.mkdtemp(prefix="acumen-draft-"))
    try:
        # The staging dir is named for the version so load_skill can validate it in place.
        work = holder / "work"
        staging = work / version
        home = holder / "home"
        selected_model = model or cfg.draft_model
        provider = provider_for_model(selected_model)
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (staging, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

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

        prompt = draft_prompt(
            package=target.pkg_name,
            version=target.pkg_version,
            src=target.src_dir,
            python=target.python,
            out=staging,
            skill_name=cfg.skill_name,
            feedback=feedback,
        )
        options = AgentOptions(
            cwd=work,
            env=env,
            model=selected_model,
            # No default turn or budget cap: only bound the agent if the caller asked. The
            # config's ``max_turns``/``max_usd`` cap benchmark agents only.
            max_turns=max_turns,
            max_usd=max_usd,
            # Codex reports no billed figure, so a budget cap needs the run's own rate table.
            price_usd=pricer(selected_model, cfg.prices),
            # The drafter reads the target source; benchmark agents never do.
            read_dirs=(target.src_dir, target.venv_dir),
            write_dirs=(work,),
            discover_skills=True,
        )

        result: AgentResult | None = None
        agent_error: Exception | None = None
        try:
            result = await run_agent(
                prompt,
                options=options,
                on_event=log.append if log is not None else None,
            )
        except Exception as err:  # noqa: BLE001 - a failed draft is an error to report, re-raised below
            agent_error = err
        finally:
            # Render the HTML log while the throwaway config dir still holds the native
            # transcript — in a finally so an aborted run (the SDK raises on a cap breach,
            # after yielding the result) is still inspectable.
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            raise DraftError(f"the drafting agent failed: {type(agent_error).__name__}: {agent_error}") from agent_error
        if result is None:
            raise DraftError("the drafting agent produced no result message")
        if result.is_error:
            raise DraftError(f"the drafting agent errored: {result.subtype} {result.errors or ''}".strip())

        _validate_staged(staging, cfg.skill_name)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, dest)
        write_meta(dest, parent=None, rationale=rationale, feedback=feedback)
        skill = load_skill(skills_root, version, expect_name=cfg.skill_name)
        return DraftResult(
            skill=skill,
            cost_usd=resolve_cost(
                result.total_cost_usd,
                price_usage(result.usage, model=selected_model, provider=result.provider, overrides=cfg.prices),
            ).cost_usd,
            turns=result.num_turns,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        # Kill anything the agent left running before removing the directory it runs in.
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
