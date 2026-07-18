"""The shipping agent: make a benchmarked skill installable *into* the target package.

Once a maintainer has benchmarked a skill and knows which version is best, ``acumen ship
--skill vN`` packages that version into the target so the package gains a ``<dist>-install-skills``
console script — one command drops the skill into ``~/.claude/skills/<name>/`` for the package's
users (§6, M7). For a GitHub URL the change arrives as a pull request the agent opens itself; for
a local path it is written straight into the working tree.

The whole point is **generalization**: the agent reasons about the repo — distribution vs import
name, src-vs-flat layout, build backend — rather than pattern-matching decoupler's shape. So, like
draft/improve/taskgen, this is an autonomous agent, not a template stamper. The one thing acumen
stages for it is a clean copy of the skill content (``SKILL.md`` + ``references/``, with acumen's
``meta.json`` bookkeeping stripped) to be copied verbatim into the wheel.

**Deliberately NOT isolated (§7 note).** Unlike every other agent, the shipper runs in the user's
real environment — real network, real git/``gh`` credentials, real ``uv`` — because it builds,
installs, pushes, and opens a PR. There is nothing to keep honest here: its output is a PR the
maintainer reviews (or a working-tree diff), so isolation would only stop it doing its job. The
correctness gate is therefore not env scrubbing but the **build-verify** the agent performs (§10.11):
build the wheel in a fresh venv, install it, run the entry point, confirm ``SKILL.md`` ships from
the installed location — the only thing that catches a mis-wired backend shipping a wheel with no
skill data. That gate lives in the agent's own turns; acumen trusts the agent's report.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from acumen.config import Config
from acumen.env import Target, claude_cli_dir
from acumen.prompts import ship_prompt
from acumen.skills import Skill, content_files, load_skill

#: A package "already ships an installer" if any console script is named like this.
_INSTALLER_SUFFIX = "-install-skills"


class ShipError(RuntimeError):
    """Raised when a skill could not be shipped."""


@dataclass(frozen=True)
class ShipResult:
    """The outcome of a shipping run: what was delivered and what it cost."""

    skill: Skill
    #: ``"github"`` (a PR was opened) or ``"local"`` (the working tree was edited).
    mode: str
    #: The agent's final message — for a GitHub target it carries the PR URL.
    summary: str
    cost_usd: float
    turns: int


def _script_tables(pyproject: dict) -> list[dict]:
    """Return every table that can declare console scripts across build backends."""
    tables: list[dict] = []
    scripts = pyproject.get("project", {}).get("scripts")
    if isinstance(scripts, dict):
        tables.append(scripts)
    # poetry keeps its scripts under [tool.poetry.scripts]; check it too so a poetry package
    # that already ships an installer is not clobbered.
    poetry = pyproject.get("tool", {}).get("poetry", {}).get("scripts")
    if isinstance(poetry, dict):
        tables.append(poetry)
    return tables


def installer_exists(src_dir: Path) -> bool:
    """Whether the target already ships a skills installer.

    True if any declared console script is named ``<something>-install-skills`` (the contract
    this command creates), or a ``_skills/install.py`` already exists under the source tree.
    Used as a pre-flight so ``acumen ship`` refuses to clobber an existing installer without
    ``--force``, before the (costly, real-env) agent runs.

    Parameters
    ----------
    src_dir
        The target package checkout.

    Returns
    -------
    Whether an installer is already present.
    """
    pyproject = src_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        for table in _script_tables(data):
            if any(str(name).endswith(_INSTALLER_SUFFIX) for name in table):
                return True
    return any(p.name == "install.py" and p.parent.name == "_skills" for p in src_dir.rglob("_skills/install.py"))


def _stage_skill_payload(skill: Skill, dest: Path) -> Path:
    """Copy the skill's content files (no ``meta.json``) into ``dest`` for verbatim shipping.

    :func:`acumen.skills.content_files` already excludes acumen's ``meta.json`` bookkeeping, so
    what lands in ``dest`` is exactly what should ship in the wheel — ``SKILL.md`` plus any
    ``references/`` tree, structure preserved.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for path in content_files(skill.directory):
        rel = path.relative_to(skill.directory)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return dest


def _ship_env(target: Target) -> dict[str, str]:
    """The environment for the (unisolated) ship agent: the real env, target venv on PATH.

    Unlike the benchmark and meta-agents this is NOT scrubbed (§7 note) — the agent needs real
    git/``gh`` credentials, network, and ``uv``. The target venv's ``bin`` is prepended so
    ``python`` resolves to the interpreter with the package installed, matching the other agents;
    everything else the user has (PATH, HOME, credentials, tokens) is carried through untouched.
    """
    env = dict(os.environ)
    parts = [str(target.bin_dir)]
    cli_dir = claude_cli_dir()
    if cli_dir is not None:
        parts.append(str(cli_dir))
    existing = env.get("PATH", "")
    if existing:
        parts.append(existing)
    env["PATH"] = os.pathsep.join(parts)
    return env


async def ship_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    version: str,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    force: bool = False,
) -> ShipResult:
    """Make skill ``version`` installable into the target package.

    Parameters
    ----------
    cfg
        The pass config; supplies ``skill_name`` and ``ship_model``, and ``is_local`` decides
        whether delivery is a PR (GitHub URL) or a working-tree edit (local path).
    target
        The prepared target, supplying the checkout the agent modifies (its ``src_dir``) and the
        venv interpreter. For a URL this is acumen's clone; for a local path it is the given dir.
    skills_root
        The ``skills/`` root; ``version`` is loaded and validated from it.
    version
        The skill version to ship, e.g. ``v2``. Required — there is no implicit "latest"/"best".
    model
        Override for the ship model; defaults to ``cfg.ship_model``.
    max_turns, max_usd
        Caps for the agent. **Unbounded by default** (§5): shipping builds and installs the
        package iteratively and runs git/``gh``, so no default budget is imposed.
    force
        Proceed even if the target already ships a skills installer (otherwise refused).

    Returns
    -------
    The ship result: the version shipped, the delivery mode, the agent's summary, and cost.
    """
    skill = load_skill(skills_root, version, expect_name=cfg.skill_name)

    # Pre-flight, before the costly real-env agent: refuse to clobber an existing installer.
    if installer_exists(target.src_dir) and not force:
        raise ShipError(
            f"{target.pkg_name} already ships a skills installer — pass force=True to ship anyway. "
            "Shipping over an existing installer risks overwriting a maintainer's own setup."
        )

    mode = "local" if cfg.is_local else "github"

    holder = Path(tempfile.mkdtemp(prefix="acumen-ship-"))
    try:
        skill_src = _stage_skill_payload(skill, holder / "skill")

        prompt = ship_prompt(
            package=target.pkg_name,
            skill_name=cfg.skill_name,
            version=skill.version,
            checkout=target.src_dir,
            skill_src=skill_src,
            mode=mode,
        )
        options = ClaudeAgentOptions(
            # The agent edits the checkout in place, so its cwd IS the checkout.
            cwd=str(target.src_dir),
            # Real environment — NOT scrubbed (§7 note): git/gh creds + network + uv.
            env=_ship_env(target),
            model=model or cfg.ship_model,
            # No default budget cap (§2/§5): only bound the agent if the caller asked.
            max_turns=max_turns,
            max_budget_usd=max_usd,
            # It reads the staged skill payload to copy into the wheel.
            add_dirs=[str(skill_src)],
            # No skill discovery: the shipper packages a skill, it does not consume one.
            setting_sources=[],
            permission_mode="bypassPermissions",
            system_prompt={"type": "preset", "preset": "claude_code"},
        )

        result: ResultMessage | None = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # a failed ship is an error to report, not a traceback to dump
            raise ShipError(f"the shipping agent failed: {type(err).__name__}: {err}") from err

        if result is None:
            raise ShipError("the shipping agent produced no result message")
        if result.is_error:
            raise ShipError(f"the shipping agent errored: {result.subtype} {result.errors or ''}".strip())

        return ShipResult(
            skill=skill,
            mode=mode,
            summary=result.result or "",
            cost_usd=result.total_cost_usd or 0.0,
            turns=result.num_turns,
        )
    finally:
        shutil.rmtree(holder, ignore_errors=True)
