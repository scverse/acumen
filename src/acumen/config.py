"""Config schema and loader for ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .paths import PathError, slugify


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is malformed."""


@dataclass(frozen=True)
class Config:
    """Everything a pass needs that isn't a task: target, models, budgets, concurrency."""

    repo: str
    skill_name: str
    ref: str = "main"
    extras: list[str] = field(default_factory=list)
    python: str = "3.12"
    #: Names of environment variables to carry from the operator's shell into isolated
    #: agents (bench sandboxes and the draft/improve/tasks meta-agents), on top of the
    #: built-in auth/proxy allowlist. The agent env is otherwise a clean allowlist — every
    #: other inherited variable is blanked — so a target that needs a runtime variable
    #: (``OMP_NUM_THREADS``, ``R_HOME``, a service key) must name it here. Values are passed
    #: through verbatim; only list what the package genuinely needs.
    env_passthrough: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: ["claude-opus-5"])
    n_replicates: int = 3
    max_concurrency: int = 4
    max_turns: int = 40
    #: Budget cap (USD) for benchmark agents only; draft/improve/ship/tasks are unbounded by default.
    max_usd: float = 3.0
    draft_model: str = "claude-opus-5"
    improve_model: str = "claude-opus-5"
    tasks_model: str = "claude-opus-5"
    ship_model: str = "claude-opus-5"

    @property
    def is_local(self) -> bool:
        """Whether ``repo`` points at a local path rather than a remote URL."""
        return not _looks_remote(self.repo)


_REMOTE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")

_KNOWN = {
    "repo",
    "ref",
    "extras",
    "python",
    "env_passthrough",
    "models",
    "n_replicates",
    "max_concurrency",
    "max_turns",
    "max_usd",
    "draft_model",
    "improve_model",
    "tasks_model",
    "ship_model",
    "skill_name",
}


def _looks_remote(repo: str) -> bool:
    return repo.startswith(_REMOTE_PREFIXES)


def derive_skill_name(repo: str) -> str:
    """Derive a default skill name from a ``repo`` URL or local path.

    Takes the last path component (``.../scanpy`` or ``git@…:owner/scanpy.git`` ->
    ``scanpy``), drops a trailing ``.git``, and slugifies it to a lowercase, filesystem-safe
    token. This is the fallback when ``config.yaml`` omits ``skill_name``, so the name is one
    fewer field a user can get wrong.
    """
    stem = Path(repo.rstrip("/")).name.removesuffix(".git")
    try:
        return slugify(stem).lower()
    except PathError as err:
        raise ConfigError(f"cannot derive a skill name from repo {repo!r}; set 'skill_name' explicitly") from err


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string, got {value!r}")
    return value


def _optional_str(raw: dict[str, Any], key: str, default: str) -> str:
    if key not in raw:
        return default
    return _require_str(raw, key)


def _positive_int(raw: dict[str, Any], key: str, default: int) -> int:
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"'{key}' must be a positive integer, got {value!r}")
    return value


def _positive_float(raw: dict[str, Any], key: str, default: float) -> float:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive number, got {value!r}")
    return float(value)


def _str_list(raw: dict[str, Any], key: str, default: list[str]) -> list[str]:
    if key not in raw:
        return list(default)
    value = raw[key]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"'{key}' must be a list of non-empty strings, got {value!r}")
    return list(value)


def parse_config(raw: Any) -> Config:
    """Validate an already-parsed ``config.yaml`` document.

    Parameters
    ----------
    raw
        The mapping loaded from YAML.

    Returns
    -------
    The validated config.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"config.yaml must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _KNOWN
    if unknown:
        raise ConfigError(f"unknown keys: {sorted(unknown)} (known keys: {sorted(_KNOWN)})")
    if "repo" not in raw:
        raise ConfigError("missing required key 'repo'")

    repo = _require_str(raw, "repo")
    skill_name = _require_str(raw, "skill_name") if "skill_name" in raw else derive_skill_name(repo)

    models = _str_list(raw, "models", ["claude-opus-5"])
    if not models:
        raise ConfigError("'models' must list at least one model")
    if len(set(models)) != len(models):
        raise ConfigError(f"'models' contains duplicates: {models}")

    default_model = models[0]
    return Config(
        repo=repo,
        skill_name=skill_name,
        ref=_optional_str(raw, "ref", "main"),
        extras=_str_list(raw, "extras", []),
        python=_optional_str(raw, "python", "3.12"),
        env_passthrough=_str_list(raw, "env_passthrough", []),
        models=models,
        n_replicates=_positive_int(raw, "n_replicates", 3),
        max_concurrency=_positive_int(raw, "max_concurrency", 4),
        max_turns=_positive_int(raw, "max_turns", 40),
        max_usd=_positive_float(raw, "max_usd", 3.0),
        draft_model=_optional_str(raw, "draft_model", default_model),
        improve_model=_optional_str(raw, "improve_model", default_model),
        tasks_model=_optional_str(raw, "tasks_model", default_model),
        ship_model=_optional_str(raw, "ship_model", default_model),
    )


def load_config(path: Path) -> Config:
    """Load and validate ``config.yaml`` from disk.

    A local ``repo`` path is resolved relative to the config file, so a config is
    portable with the directory that contains it.
    """
    try:
        text = path.read_text()
    except OSError as err:
        raise ConfigError(f"cannot read config file {path}: {err}") from err
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path} is not valid YAML: {err}") from err
    try:
        cfg = parse_config(raw)
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err
    if cfg.is_local:
        resolved = (path.parent / cfg.repo).expanduser()
        if not resolved.exists():
            raise ConfigError(f"{path}: repo {cfg.repo!r} is not a URL and does not exist as a local path ({resolved})")
        cfg = replace(cfg, repo=str(resolved.resolve()))
    return cfg
