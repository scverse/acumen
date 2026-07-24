"""Per-run sandbox: a fresh empty directory with the target venv on PATH.

The agent sees an installed package and nothing else — no repo source, no user
settings, no memories. Each run gets its own sandbox, throwaway ``HOME`` and throwaway
``CLAUDE_CONFIG_DIR``, so runs cannot see each other either.

A skill arm differs from the baseline in exactly one way: ``skills/v{n}/`` is
copied into ``<sandbox>/.claude/skills/<name>/``, where the agent's own project settings
discovery finds it. Same prompt, same tools, same caps, same env otherwise.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from acumen.env import AuthMode, Target, build_agent_env
from acumen.skills import Skill, content_files

#: Where the ``claude`` CLI looks for project skills, relative to the agent's cwd.
SKILLS_SUBDIR = Path(".claude") / "skills"


@dataclass(frozen=True)
class Sandbox:
    """A prepared, isolated working directory for exactly one agent run."""

    root: Path
    home: Path
    config_dir: Path
    env: dict[str, str]
    authenticated: bool
    skill: Skill | None = None

    @property
    def transcript_root(self) -> Path:
        """Where the ``claude`` CLI writes transcripts for this run."""
        return self.config_dir / "projects"

    @property
    def skills_dir(self) -> Path:
        """The project skills directory the agent discovers, ``<root>/.claude/skills``."""
        return self.root / SKILLS_SUBDIR

    @property
    def skill_hash(self) -> str | None:
        """The installed skill's content hash, or ``None`` for the baseline arm."""
        return self.skill.hash if self.skill else None


def install_skill(root: Path, skill: Skill) -> Path:
    """Copy a skill's content into a sandbox as a discoverable project skill.

    ``meta.json`` is deliberately left behind: it is acumen's provenance record, and the
    agent consuming the skill has no business reading the rationale for its own version.

    Parameters
    ----------
    root
        The sandbox root — the agent's ``cwd``.
    skill
        The loaded skill to install.

    Returns
    -------
    The installed skill directory, ``<root>/.claude/skills/<name>``.
    """
    dest = root / SKILLS_SUBDIR / skill.name
    dest.mkdir(parents=True, exist_ok=True)
    for src in content_files(skill.directory):
        target = dest / src.relative_to(skill.directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    return dest


@contextmanager
def sandbox(
    target: Target,
    *,
    auth_mode: AuthMode,
    base: Path | None = None,
    keep: bool = False,
    skill: Skill | None = None,
) -> Iterator[Sandbox]:
    """Create a fresh sandbox for one run and clean it up afterwards.

    Parameters
    ----------
    target
        The prepared target; its venv ``bin`` goes on the sandbox PATH.
    auth_mode
        Which credential the run authenticates with (see :func:`acumen.env.build_agent_env`).
        Benchmark runs always pass ``"api"`` so the recorded ``cost_usd`` is real.
    base
        Parent directory for the sandbox. Defaults to the system temp dir.
    keep
        Leave the sandbox on disk after the run — useful when debugging a failure.
    skill
        Skill to install for this run, or ``None`` for the baseline arm.

    Yields
    ------
    The sandbox, whose ``root`` is the agent's ``cwd``.
    """
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(prefix="acumen-run-", dir=base))
    try:
        root = holder / "work"
        home = holder / "home"
        config_dir = home / ".claude"
        for path in (root, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        if skill is not None:
            install_skill(root, skill)
        env = build_agent_env(config_dir=config_dir, home=home, extra_path=[target.bin_dir], auth_mode=auth_mode)
        yield Sandbox(
            root=root,
            home=home,
            config_dir=config_dir,
            env=env,
            # Only session mode seeds the OAuth login into the throwaway config dir.
            authenticated=auth_mode == "session",
            skill=skill,
        )
    finally:
        if not keep:
            shutil.rmtree(holder, ignore_errors=True)
