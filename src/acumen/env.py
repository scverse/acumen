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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acumen.agents import AgentProvider
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
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
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


def codex_cli_dir() -> Path | None:
    """Return the directory holding the ``codex`` CLI."""
    found = shutil.which("codex")
    return Path(found).parent if found else None


def agent_cli_dir(provider: AgentProvider) -> Path | None:
    """Return the directory holding the selected provider's CLI."""
    return claude_cli_dir() if provider == "claude" else codex_cli_dir()


#: Allowlisted variables that, present and non-empty, let a run authenticate on their own:
#: a direct Anthropic key/token, or a flag routing to a cloud provider whose own (AWS/GCP)
#: credentials then apply. A subset of :data:`ENV_ALLOWLIST` — the entries that carry or
#: enable authentication, as opposed to proxy/TLS routing.
AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

#: A resolved authentication mode for an agent run. ``"session"`` uses the provider's stored
#: account/subscription login; ``"api"`` uses a metered API credential. Acumen derives
#: ``cost_usd`` from tokens in either mode, and records the mode so its meaning stays explicit.
AuthMode = Literal["session", "api"]

#: Allowlisted variables that carry *metered* (API/cloud) authentication, as distinct from a
#: subscription OAuth login. A subset of :data:`AUTH_ENV_VARS` that deliberately excludes
#: ``CLAUDE_CODE_OAUTH_TOKEN`` — that token bills the subscription, not the API. A run in
#: ``"api"`` mode keeps these and drops the OAuth token; a ``"session"`` run does the reverse.
API_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

#: The subscription (OAuth) token variable — a portable ``claude setup-token`` credential.
#: Kept in ``"session"`` mode and dropped in ``"api"`` mode so exactly one auth path is live.
SESSION_AUTH_ENV_VAR = "CLAUDE_CODE_OAUTH_TOKEN"

CODEX_API_AUTH_ENV_VARS = ("CODEX_API_KEY", "OPENAI_API_KEY")


def _credentials_path() -> Path:
    """The user's Claude OAuth credentials file — what :func:`seed_credentials` copies."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    return Path(base) / ".credentials.json"


def _codex_home() -> Path:
    """The user's real Codex home, which contains ``auth.json``."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _codex_credentials_path() -> Path:
    return _codex_home() / "auth.json"


def _codex_persisted_auth_mode() -> AuthMode | None:
    """Classify the credential stored by ``codex login``, without exposing its value.

    Both account and API-key logins write ``auth.json``. The file's existence therefore
    cannot distinguish subscription usage from metered API usage: recent Codex versions
    identify API-key login with ``auth_mode: apikey`` and account login with
    ``auth_mode: chatgpt``. The payload shapes are retained as compatibility fallbacks.
    Unknown or unreadable files are not guessed at.
    """
    try:
        data = json.loads(_codex_credentials_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    mode = data.get("auth_mode")
    if mode == "apikey" or data.get("OPENAI_API_KEY"):
        return "api"
    if mode == "chatgpt" or isinstance(data.get("tokens"), dict):
        return "session"
    return None


def _has_oauth_credentials() -> bool:
    """Whether the user's credentials file holds a Claude subscription (OAuth) login.

    ``claude`` login writes ``.credentials.json`` as ``{"claudeAiOauth": {...}}`` — the
    presence of that object is what distinguishes a subscription login from a bare API-key
    setup, and it is the signal that "session usage" is available. A missing, unreadable, or
    API-only credentials file is not a subscription.
    """
    try:
        data = json.loads(_credentials_path().read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data.get("claudeAiOauth"), dict)


def session_auth_available(provider: AgentProvider = "claude") -> bool:
    """Whether an agent run could bill the Claude subscription ("session usage").

    For Claude, true when a portable ``CLAUDE_CODE_OAUTH_TOKEN`` is set or a subscription
    OAuth login is on disk. For Codex, true only for an account login in ``auth.json``; a
    persisted API-key login is deliberately not a session. An empty variable does not count.
    """
    if provider == "codex":
        return _codex_persisted_auth_mode() == "session"
    if os.environ.get(SESSION_AUTH_ENV_VAR):
        return True
    return _has_oauth_credentials()


def api_auth_available(provider: AgentProvider = "claude") -> bool:
    """Whether an agent run could bill the Anthropic API (or a cloud provider) per token.

    True when any metered auth variable is set (:data:`API_AUTH_ENV_VARS`): a direct
    Anthropic key/token, or a Bedrock/Vertex routing flag whose own cloud credentials then
    apply. A Codex API key persisted by ``codex login --with-api-key`` also counts. The
    subscription OAuth token is deliberately excluded — it bills the plan, not the API.
    """
    variables = API_AUTH_ENV_VARS if provider == "claude" else CODEX_API_AUTH_ENV_VARS
    available = any(os.environ.get(var) for var in variables)
    return available or (provider == "codex" and _codex_persisted_auth_mode() == "api")


def resolve_auth_mode(
    requested: str,
    *,
    provider: AgentProvider = "claude",
) -> AuthMode:
    """Resolve a requested auth choice to the concrete mode a run will use, or fail loudly.

    A preflight guard for every agentic command: it both validates that the chosen credential
    is actually reachable and reports which mode the run will bill, so the choice is never
    silent.

    ``bench`` used to be barred from ``"session"`` on the grounds that only metered API billing
    yields a real per-run ``cost_usd``. That stopped being true when cost became a function of
    the token counts both billing modes report (:mod:`acumen.prices`): a subscription run prices
    exactly as accurately as a metered one. What the figure *means* does change — under
    ``"session"`` it is what the run would have cost at API rates, not money billed — so the
    mode is recorded in each ``result.json`` rather than the choice being taken away.

    Parameters
    ----------
    requested
        The user's choice: ``"auto"`` (prefer the subscription, else the API), ``"session"``
        (force the subscription), or ``"api"`` (force the API).

    Returns
    -------
    ``"session"`` or ``"api"``.
    """
    if requested == "session":
        if not session_auth_available(provider):
            if provider == "codex":
                raise EnvError(
                    "no Codex account login found for --auth session. Log in with `codex login` "
                    "so ~/.codex/auth.json exists."
                )
            raise EnvError(
                "no Claude subscription login found for --auth session. Log in with `claude` so "
                "~/.claude/.credentials.json exists, or set CLAUDE_CODE_OAUTH_TOKEN (from "
                "`claude setup-token`)."
            )
        return "session"
    if requested == "api":
        if not api_auth_available(provider):
            if provider == "codex":
                raise EnvError("no API credential found for --auth api. Set CODEX_API_KEY (or OPENAI_API_KEY).")
            raise EnvError(
                "no API credential found for --auth api. Set ANTHROPIC_API_KEY (or "
                "ANTHROPIC_AUTH_TOKEN), or enable a provider with CLAUDE_CODE_USE_BEDROCK / "
                "CLAUDE_CODE_USE_VERTEX."
            )
        return "api"

    # "auto": prefer the subscription (it's what the user is paying a flat rate for), fall
    # back to the metered API, and only then give up.
    if session_auth_available(provider):
        return "session"
    if api_auth_available(provider):
        return "api"
    if provider == "codex":
        raise EnvError(
            "no Codex credentials found — log in with `codex login` (subscription), "
            "or set CODEX_API_KEY / OPENAI_API_KEY (API)."
        )
    raise EnvError(
        "no Claude credentials found — an agent run cannot authenticate. Log in with `claude` "
        "so ~/.claude/.credentials.json exists (subscription), or set ANTHROPIC_API_KEY (or "
        "ANTHROPIC_AUTH_TOKEN / CLAUDE_CODE_OAUTH_TOKEN), or enable a provider with "
        "CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX."
    )


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
    real = _credentials_path()
    if not real.is_file():
        return False
    config_dir.mkdir(parents=True, exist_ok=True)
    dest = config_dir / ".credentials.json"
    shutil.copyfile(real, dest)
    dest.chmod(0o600)
    return True


def seed_codex_credentials(config_dir: Path) -> bool:
    """Copy only Codex's login token into an isolated ``CODEX_HOME``."""
    real = _codex_credentials_path()
    if not real.is_file():
        return False
    config_dir.mkdir(parents=True, exist_ok=True)
    dest = config_dir / "auth.json"
    shutil.copyfile(real, dest)
    dest.chmod(0o600)
    return True


def _apply_auth_mode(
    env: dict[str, str],
    auth_mode: AuthMode | None,
    provider: AgentProvider = "claude",
) -> None:
    """Neutralize the credential variables the chosen provider and mode must not authenticate with.

    Two rules, in order. First, the *other* provider's credentials are blanked whatever the
    mode: they can never authenticate this run, so allowlisting them would only carry the
    operator's second API key into a web-enabled agent. Second, within the selected provider
    exactly one auth path is left live: ``"session"`` neutralizes the metered API/cloud
    variables and keeps the subscription login; ``"api"`` neutralizes the OAuth token and keeps
    the metered ones. ``None`` leaves the selected provider's own credentials untouched (the
    historical behavior, for callers that don't select a mode).

    Codex has no session *variable* — its subscription login is the ``auth.json`` that
    :func:`seed_codex_credentials` copies into the throwaway ``CODEX_HOME`` — so ``"session"``
    there just blanks the API keys.

    The variables are set to ``""``, not deleted. The SDK builds the agent subprocess env as
    ``{**os.environ, **options.env}`` — it merges our mapping *over* the inherited environment —
    so a credential we merely omit falls back through from ``os.environ`` unchanged and the run
    silently authenticates with the wrong one. An explicit empty value overrides the inherited
    one, which the CLI reads as unset.
    """
    if provider == "claude":
        drop = [*CODEX_API_AUTH_ENV_VARS]
        if auth_mode == "session":
            drop += API_AUTH_ENV_VARS
        elif auth_mode == "api":
            drop.append(SESSION_AUTH_ENV_VAR)
    else:
        drop = [*API_AUTH_ENV_VARS, SESSION_AUTH_ENV_VAR]
        if auth_mode == "session":
            drop += CODEX_API_AUTH_ENV_VARS
    for var in drop:
        env[var] = ""
    if provider == "codex" and auth_mode == "api":
        # ``codex exec`` consumes CODEX_API_KEY. Accept the conventional OpenAI variable too
        # and translate it here, without mutating the operator's shell.
        env["CODEX_API_KEY"] = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def scrubbed_env(
    *,
    config_dir: Path,
    home: Path,
    extra_path: list[Path] | None = None,
    auth_mode: AuthMode | None = None,
    extra_allow: Sequence[str] | None = None,
    provider: AgentProvider = "claude",
) -> dict[str, str]:
    """Build the environment an isolated agent runs under.

    The result is a clean allowlist: only :data:`ENV_ALLOWLIST` (plus ``extra_allow``) and
    a handful of throwaway overrides survive; **every other variable inherited from the
    operator's shell is blanked**. ``HOME`` and ``CLAUDE_CONFIG_DIR`` point at throwaway
    directories so no user settings or ``CLAUDE.md`` memories are discoverable.

    The blanking is not cosmetic. The SDK builds the agent subprocess env as
    ``{**os.environ, **options.env}`` — it merges this mapping *over* the inherited
    environment (see :func:`_apply_auth_mode`), so a variable we merely *omit* falls back
    through from ``os.environ`` unchanged. To actually keep an ambient secret out of the
    (web-enabled) agent we must return it set to ``""``, which overrides the inherited value
    and the CLI reads as unset. So we enumerate everything in ``os.environ`` that isn't
    allowlisted or overridden and empty it explicitly.

    Parameters
    ----------
    config_dir
        Throwaway ``CLAUDE_CONFIG_DIR``; also where the transcript will land.
    home
        Throwaway ``HOME``.
    extra_path
        Directories to prepend to ``PATH`` — the target venv's ``bin`` goes here, so
        ``python`` in the sandbox is the interpreter with the package installed.
    auth_mode
        Which credential the run should authenticate with. ``"session"`` keeps only the
        subscription OAuth token, ``"api"`` keeps only the metered API/cloud credentials, and
        ``None`` leaves both in place (the historical behavior). Note that ``"session"`` needs
        :func:`seed_credentials` to have placed the OAuth login in ``config_dir`` — pair the
        two via :func:`build_agent_env`.
    extra_allow
        Additional variable names to carry through from ``os.environ`` on top of
        :data:`ENV_ALLOWLIST` — the operator's declared ``env_passthrough``, for a target
        that needs a runtime variable the built-in allowlist doesn't cover.

    Returns
    -------
    The environment mapping to hand to the SDK.
    """
    allow = (*ENV_ALLOWLIST, *(extra_allow or ()))
    env = {key: os.environ[key] for key in allow if key in os.environ}
    _apply_auth_mode(env, auth_mode, provider)

    path_parts = [str(p) for p in (extra_path or [])]
    cli_dir = agent_cli_dir(provider)
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
    env["CODEX_HOME"] = str(config_dir)
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

    # Blank every inherited variable we did not deliberately keep or override. Omission is a
    # no-op under the SDK's env merge, so the only way to stop the operator's ambient secrets
    # (cloud creds, service tokens, anything exported in the shell) from reaching a web-enabled
    # agent is to override each one with an empty value. Anything a target legitimately needs is
    # declared via extra_allow (config env_passthrough) and was kept above.
    for key in os.environ:
        if key not in env:
            env[key] = ""
    return env


def build_agent_env(
    *,
    config_dir: Path,
    home: Path,
    extra_path: list[Path] | None = None,
    auth_mode: AuthMode,
    extra_allow: Sequence[str] | None = None,
    provider: AgentProvider = "claude",
) -> dict[str, str]:
    """Prepare an isolated agent's environment for a resolved auth mode.

    Pairs the two steps that must agree. Claude seeds its subscription OAuth login only in
    ``"session"`` mode. Codex stores both account and API-key logins in ``auth.json``, so the
    file is seeded only when its classified mode matches ``auth_mode``. Environment credentials
    for the other mode are stripped either way, keeping billing deterministic.

    Every isolated agent (bench sandboxes and the draft/improve/tasks meta-agents) builds its
    env through here, so the seed-and-scrub pairing lives in one place. ``extra_allow`` (the
    operator's ``env_passthrough``) names variables to carry through on top of the built-in
    allowlist, for a target that needs one at runtime.
    """
    if provider == "claude" and auth_mode == "session":
        seed_credentials(config_dir)
    elif provider == "codex" and _codex_persisted_auth_mode() == auth_mode:
        # Codex stores either account tokens or an API key in auth.json. Copy only the
        # credential matching the resolved mode into the isolated CODEX_HOME.
        seed_codex_credentials(config_dir)
    return scrubbed_env(
        config_dir=config_dir,
        home=home,
        extra_path=extra_path,
        auth_mode=auth_mode,
        extra_allow=extra_allow,
        provider=provider,
    )


def _default_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "acumen"


DEFAULT_CACHE_ROOT = _default_cache_root()


def sdk_version() -> str:
    """Return the installed ``claude-agent-sdk`` version, for the run fingerprint.

    ``"unknown"`` on a Codex-only install, where the SDK is deliberately absent — the
    fingerprint records what drove the run, and nothing Claude drove can be recorded here.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("claude-agent-sdk")
    except PackageNotFoundError:
        return "unknown"


def agent_version(provider: AgentProvider) -> str:
    """Return the installed provider runtime version for run fingerprints."""
    if provider == "claude":
        return sdk_version()
    try:
        output = _run(["codex", "--version"])
    except EnvError:
        return "unknown"
    return output.removeprefix("codex-cli ").strip()


def python_version() -> str:
    """Return the interpreter version running acumen itself."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
