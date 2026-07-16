"""Per-run sandbox: a fresh empty directory with the target venv on PATH.

The agent sees an installed package and nothing else — no repo source (§7.3), no user
settings, no memories. Each run gets its own sandbox, throwaway ``HOME`` and throwaway
``CLAUDE_CONFIG_DIR``, so runs cannot see each other either.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from acumen.env import Target, scrubbed_env, seed_credentials


@dataclass(frozen=True)
class Sandbox:
    """A prepared, isolated working directory for exactly one agent run."""

    root: Path
    home: Path
    config_dir: Path
    env: dict[str, str]
    authenticated: bool

    @property
    def transcript_root(self) -> Path:
        """Where the ``claude`` CLI writes transcripts for this run."""
        return self.config_dir / "projects"


@contextmanager
def sandbox(target: Target, *, base: Path | None = None, keep: bool = False) -> Iterator[Sandbox]:
    """Create a fresh sandbox for one run and clean it up afterwards.

    Parameters
    ----------
    target
        The prepared target; its venv ``bin`` goes on the sandbox PATH.
    base
        Parent directory for the sandbox. Defaults to the system temp dir.
    keep
        Leave the sandbox on disk after the run — useful when debugging a failure.

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
        authenticated = seed_credentials(config_dir)
        env = scrubbed_env(config_dir=config_dir, home=home, extra_path=[target.bin_dir])
        yield Sandbox(root=root, home=home, config_dir=config_dir, env=env, authenticated=authenticated)
    finally:
        if not keep:
            shutil.rmtree(holder, ignore_errors=True)
