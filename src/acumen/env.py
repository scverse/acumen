"""Target intake and process environment.

Two jobs:

* :func:`prepare_target` — clone (or adopt a local path), build one venv with the target
  package installed, and record the resolved commit and package version. Cached by
  (repo, ref) so a pass doesn't re-clone.
* :func:`scrubbed_env` — the filtered environment benchmark agents run under: auth and
  PATH only, a throwaway ``HOME`` and ``CLAUDE_CONFIG_DIR``, and nothing that could leak
  the user's own settings or memories into a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from acumen.config import Config
from acumen.paths import slugify

READY_MARKER = ".acumen-ready"

#: Environment variables carried into agent runs. Everything else is dropped.
#: Auth and provider routing, because a run cannot authenticate without them; proxy and
#: TLS settings, because web access is part of the benchmark.
ENV_ALLOWLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)

_BASE_PATH = ("/usr/local/bin", "/usr/bin", "/bin")


class EnvError(RuntimeError):
    """Raised when the target cannot be prepared."""


@dataclass(frozen=True)
class Target:
    """A prepared benchmark target: source checkout plus the venv it's installed into."""

    source: str
    ref: str
    src_dir: Path
    venv_dir: Path
    commit: str
    pkg_name: str
    pkg_version: str

    @property
    def bin_dir(self) -> Path:
        """The venv's ``bin`` directory — what goes on an agent's PATH."""
        return self.venv_dir / ("Scripts" if os.name == "nt" else "bin")

    @property
    def python(self) -> Path:
        """The venv interpreter, with the target package importable."""
        return self.bin_dir / ("python.exe" if os.name == "nt" else "python")

    @property
    def fingerprint(self) -> str:
        """The ``pkg_version`` string recorded in ``result.json``, e.g. ``numpy 2.1.0``."""
        return f"{self.pkg_name} {self.pkg_version}"


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise EnvError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def cache_key(repo: str, ref: str) -> str:
    """Return the cache directory name for a (repo, ref) pair."""
    digest = hashlib.sha256(f"{repo}\n{ref}".encode()).hexdigest()[:12]
    stem = slugify(Path(repo.rstrip("/")).name.removesuffix(".git") or "target")
    return f"{stem}-{digest}"


def _clone(repo: str, ref: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", "--quiet", repo, str(dest)])
    _run(["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", ref], cwd=dest)


def _resolve_commit(src_dir: Path) -> str:
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=src_dir)
    except EnvError:
        return "local"  # a local path that isn't a git checkout is still a valid target


def _package_name(src_dir: Path) -> str:
    pyproject = src_dir / "pyproject.toml"
    if not pyproject.is_file():
        raise EnvError(f"{src_dir} has no pyproject.toml — acumen needs an installable package")
    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise EnvError(f"cannot parse {pyproject}: {err}") from err
    name = data.get("project", {}).get("name")
    if not name:
        raise EnvError(f"{pyproject} does not declare [project].name")
    return str(name)


def _installed_version(python: Path, pkg_name: str) -> str:
    code = f"import importlib.metadata as m; print(m.version({pkg_name!r}))"
    try:
        return _run([str(python), "-c", code])
    except EnvError as err:
        raise EnvError(f"{pkg_name} is not importable in the target venv after install: {err}") from err


def prepare_target(cfg: Config, cache_root: Path, *, refresh: bool = False) -> Target:
    """Clone or adopt the target package and install it into a cached venv.

    Parameters
    ----------
    cfg
        The pass config; supplies ``repo``, ``ref``, ``extras`` and ``python``.
    cache_root
        Directory to hold checkouts and venvs, keyed by (repo, ref).
    refresh
        Rebuild even if a ready-marked cache entry exists.

    Returns
    -------
    The prepared target, with the resolved commit and installed package version.
    """
    if shutil.which("uv") is None:
        raise EnvError("uv is not on PATH — acumen uses it to build the target venv")
    entry = cache_root / cache_key(cfg.repo, cfg.ref)
    venv_dir = entry / "venv"
    marker = entry / READY_MARKER

    if cfg.is_local:
        src_dir = Path(cfg.repo).expanduser().resolve()
        if not src_dir.is_dir():
            raise EnvError(f"local repo path does not exist: {src_dir}")
    else:
        src_dir = entry / "src"

    if not refresh and marker.is_file():
        try:
            cached = json.loads(marker.read_text())
            target = Target(
                source=cfg.repo,
                ref=cfg.ref,
                src_dir=Path(cached["src_dir"]),
                venv_dir=venv_dir,
                commit=cached["commit"],
                pkg_name=cached["pkg_name"],
                pkg_version=cached["pkg_version"],
            )
        except (OSError, KeyError, json.JSONDecodeError):
            target = None  # a corrupt marker just means we rebuild
        else:
            # A local target's working tree can move under us; a clone at a pinned ref cannot.
            if target.python.is_file() and (not cfg.is_local or _resolve_commit(src_dir) == target.commit):
                return target

    if not cfg.is_local:
        _clone(cfg.repo, cfg.ref, src_dir)

    entry.mkdir(parents=True, exist_ok=True)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    _run(["uv", "venv", "--python", cfg.python, str(venv_dir)])

    pkg_name = _package_name(src_dir)
    spec = str(src_dir)
    if cfg.extras:
        spec = f"{src_dir}[{','.join(cfg.extras)}]"
    python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    _run(["uv", "pip", "install", "--python", str(python), spec])

    target = Target(
        source=cfg.repo,
        ref=cfg.ref,
        src_dir=src_dir,
        venv_dir=venv_dir,
        commit=_resolve_commit(src_dir),
        pkg_name=pkg_name,
        pkg_version=_installed_version(python, pkg_name),
    )
    marker.write_text(
        json.dumps(
            {
                "src_dir": str(target.src_dir),
                "commit": target.commit,
                "pkg_name": target.pkg_name,
                "pkg_version": target.pkg_version,
            },
            indent=2,
        )
    )
    return target


def claude_cli_dir() -> Path | None:
    """Return the directory holding the ``claude`` CLI, which the SDK shells out to."""
    found = shutil.which("claude")
    return Path(found).parent if found else None


def seed_credentials(config_dir: Path) -> bool:
    """Copy the user's Claude credentials into a throwaway config dir.

    A scrubbed ``HOME`` hides ``~/.claude/.credentials.json``, which is how
    OAuth-authenticated users are logged in — without this an isolated run cannot
    authenticate at all. Only the credentials file is copied: settings, memories and
    project history stay behind, which is the isolation the benchmark actually needs.

    Returns
    -------
    Whether a credentials file was found and copied.
    """
    real = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / ".credentials.json"
    if not real.is_file():
        return False
    config_dir.mkdir(parents=True, exist_ok=True)
    dest = config_dir / ".credentials.json"
    shutil.copyfile(real, dest)
    dest.chmod(0o600)
    return True


def scrubbed_env(*, config_dir: Path, home: Path, extra_path: list[Path] | None = None) -> dict[str, str]:
    """Build the environment an isolated agent runs under.

    Everything outside :data:`ENV_ALLOWLIST` is dropped. ``HOME`` and
    ``CLAUDE_CONFIG_DIR`` point at throwaway directories so no user settings or
    ``CLAUDE.md`` memories are discoverable.

    Parameters
    ----------
    config_dir
        Throwaway ``CLAUDE_CONFIG_DIR``; also where the transcript will land.
    home
        Throwaway ``HOME``.
    extra_path
        Directories to prepend to ``PATH`` — the target venv's ``bin`` goes here, so
        ``python`` in the sandbox is the interpreter with the package installed.

    Returns
    -------
    The environment mapping to hand to the SDK.
    """
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}

    path_parts = [str(p) for p in (extra_path or [])]
    cli_dir = claude_cli_dir()
    if cli_dir is not None:
        path_parts.append(str(cli_dir))
    node_dir = shutil.which("node")
    if node_dir is not None:
        path_parts.append(str(Path(node_dir).parent))
    path_parts.extend(_BASE_PATH)
    seen: list[str] = []
    for part in path_parts:
        if part not in seen:
            seen.append(part)

    env["PATH"] = os.pathsep.join(seen)
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    # Skill discovery needs setting_sources=["project"], but project discovery
    # also walks UP from cwd and auto-loads every CLAUDE.md it passes. Verified: an agent
    # recited a canary planted in its sandbox's parent directory having made zero tool
    # calls. Sandboxes live under a temp dir whose ancestors we do not control, so a stray
    # CLAUDE.md anywhere above them would silently enter every run's context and break the
    # memory isolation. This disables memory discovery outright; skills still load (verified).
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["TMPDIR"] = str(home / "tmp")
    env["LANG"] = os.environ.get("LANG", "C.UTF-8")
    # Keep pip/uv from reaching into the real user's caches and configs.
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    return env


def _default_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "acumen"


DEFAULT_CACHE_ROOT = _default_cache_root()


def sdk_version() -> str:
    """Return the installed ``claude-agent-sdk`` version, for the run fingerprint."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("claude-agent-sdk")
    except PackageNotFoundError:  # pragma: no cover - the SDK is a hard dependency
        return "unknown"


def python_version() -> str:
    """Return the interpreter version running acumen itself."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
