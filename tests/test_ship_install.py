"""Agent-free validation of the ``SHIP_INSTALL_TEMPLATE`` install script.

The shipping agent copies this template verbatim into the target package as
``_skills/install.py``. Here we render it, drop it into a throwaway importable package with a
bundled ``data/`` skill, and drive its ``main()`` the way the wired console script would — for
every ``--agent`` framework, for ``--dest``, and for the no-target error path. No agent, no
network; this is the smoke test that catches a broken template before a paid ship run.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from acumen.prompts import SHIP_INSTALL_TEMPLATE

SKILL_NAME = "widget"
_PKG = "acumen_ship_probe"


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Render the template into an importable package and return its module.

    ``HOME`` and the per-framework env vars are pointed at a scratch tree so installs never
    touch the real user config.
    """
    root = tmp_path / "pkgroot"
    pkg = root / _PKG
    (pkg / "data" / "references").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "install.py").write_text(SHIP_INSTALL_TEMPLATE.replace("__SKILL_NAME__", SKILL_NAME))
    (pkg / "data" / "SKILL.md").write_text("---\nname: widget\n---\nbody\n")
    (pkg / "data" / "references" / "api.md").write_text("api notes\n")

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)

    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(f"{_PKG}.install", None)
    sys.modules.pop(_PKG, None)
    module = importlib.import_module(f"{_PKG}.install")
    yield module
    sys.modules.pop(f"{_PKG}.install", None)
    sys.modules.pop(_PKG, None)


def _installed_ok(dest: Path) -> bool:
    return (dest / "SKILL.md").read_text() == "---\nname: widget\n---\nbody\n" and (
        dest / "references" / "api.md"
    ).is_file()


@pytest.mark.parametrize("agent", ["claude", "codex", "agents", "claude-science"])
def test_each_agent_lands_the_skill(install: ModuleType, tmp_path: Path, agent: str) -> None:
    if agent == "claude-science":
        science = tmp_path / "home" / ".claude-science"
        science.mkdir(parents=True)
        (science / "active-org.json").write_text(json.dumps({"org_uuid": "org-1234"}))

    assert install.main(["--agent", agent]) == 0
    dest = install.resolve_dest(agent, None)
    assert _installed_ok(dest)


def test_dest_overrides_agent(install: ModuleType, tmp_path: Path) -> None:
    dest = tmp_path / "explicit"
    assert install.main(["--dest", str(dest)]) == 0
    assert _installed_ok(dest)


def test_no_target_errors_cleanly(install: ModuleType, capsys: pytest.CaptureFixture) -> None:
    assert install.main([]) == 1
    err = capsys.readouterr().err
    assert "--agent" in err and "--dest" in err


def test_print_path_reports_the_bundled_source(install: ModuleType, capsys: pytest.CaptureFixture) -> None:
    assert install.main(["--print-path"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert (printed / "SKILL.md").is_file()


def test_reinstall_is_idempotent_then_force_replaces_a_difference(
    install: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dest = tmp_path / "dest"
    assert install.main(["--dest", str(dest)]) == 0
    capsys.readouterr()

    # Identical content already there — a plain reinstall is a no-op success.
    assert install.main(["--dest", str(dest)]) == 0
    assert "up to date" in capsys.readouterr().out

    # A drifted copy refuses without --force, then --force restores the bundle.
    (dest / "SKILL.md").write_text("tampered\n")
    assert install.main(["--dest", str(dest)]) == 1
    assert "--force" in capsys.readouterr().err
    assert install.main(["--dest", str(dest), "--force"]) == 0
    assert _installed_ok(dest)


def test_check_reports_match_and_drift(install: ModuleType, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    assert install.main(["--dest", str(dest), "--check"]) == 1  # not installed yet
    assert install.main(["--dest", str(dest)]) == 0
    assert install.main(["--dest", str(dest), "--check"]) == 0  # matches
    (dest / "SKILL.md").write_text("tampered\n")
    assert install.main(["--dest", str(dest), "--check"]) == 1  # drifted


def test_claude_science_rejects_a_bad_org_uuid(
    install: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    science = tmp_path / "home" / ".claude-science"
    science.mkdir(parents=True)
    (science / "active-org.json").write_text(json.dumps({"org_uuid": "../escape"}))
    assert install.main(["--agent", "claude-science"]) == 1
    assert "org_uuid" in capsys.readouterr().err


def test_env_var_overrides_the_framework_root(
    install: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex_home = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    assert install.main(["--agent", "codex"]) == 0
    assert _installed_ok(codex_home / "skills" / SKILL_NAME)
