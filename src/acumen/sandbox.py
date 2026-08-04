"""Per-run sandbox: a fresh empty directory with the target venv on PATH.

The agent sees an installed package and nothing else — no repo source, no user
settings, no memories. Each run gets its own sandbox, throwaway ``HOME`` and throwaway
provider config directory, so runs cannot see each other either.

A skill arm differs from the baseline in exactly one way: ``skills/v{n}/`` is
copied into the selected provider's project skill directory, where the agent's own
discovery finds it. Same prompt, same tools, same caps, same env otherwise.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from acumen.agents import AgentProvider
from acumen.env import AuthMode, Target, build_agent_env
from acumen.procs import label_env, reap
from acumen.skills import Skill, content_files

SKILLS_SUBDIRS = {
    "claude": Path(".claude") / "skills",
    "codex": Path(".agents") / "skills",
}


@dataclass(frozen=True)
class Sandbox:
    """A prepared, isolated working directory for exactly one agent run."""

    root: Path
    home: Path
    config_dir: Path
    env: dict[str, str]
    authenticated: bool
    provider: AgentProvider = "claude"
    skill: Skill | None = None

    @property
    def transcript_root(self) -> Path:
        """Where Claude writes transcripts for this run (unused by Codex)."""
        return self.config_dir / "projects"

    @property
    def skills_dir(self) -> Path:
        """The project skills directory the selected agent discovers."""
        return self.root / SKILLS_SUBDIRS[self.provider]

    @property
    def skill_hash(self) -> str | None:
        """The installed skill's content hash, or ``None`` for the baseline arm."""
        return self.skill.hash if self.skill else None


def install_skill(root: Path, skill: Skill, *, provider: AgentProvider = "claude") -> Path:
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
    The selected provider's installed project-skill directory.
    """
    dest = root / SKILLS_SUBDIRS[provider] / skill.name
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
    env_passthrough: Sequence[str] | None = None,
    provider: AgentProvider = "claude",
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
    env_passthrough
        Extra environment variable names to carry into the sandbox on top of the built-in
        allowlist (the operator's ``config.env_passthrough``).

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
        config_dir = home / (".claude" if provider == "claude" else ".codex")
        for path in (root, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        if skill is not None:
            install_skill(root, skill, provider=provider)
        # The marker is what lets the teardown below find whatever this agent leaves running.
        env = label_env(
            build_agent_env(
                config_dir=config_dir,
                home=home,
                extra_path=[target.bin_dir],
                auth_mode=auth_mode,
                extra_allow=env_passthrough,
                provider=provider,
            ),
            holder,
        )
        yield Sandbox(
            root=root,
            home=home,
            config_dir=config_dir,
            env=env,
            # Only session mode seeds the OAuth login into the throwaway config dir.
            authenticated=auth_mode == "session",
            provider=provider,
            skill=skill,
        )
    finally:
        # The agent's own descendants outlive it — a backgrounded job, or anything still
        # running when the CLI was terminated on a cap breach or a Ctrl-C. Kill them before
        # the directory goes, so nothing is left writing into a path that no longer exists.
        # ``keep`` preserves the files for inspection, never the processes.
        reap(holder)
        if not keep:
            shutil.rmtree(holder, ignore_errors=True)
