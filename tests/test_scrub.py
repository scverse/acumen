"""The first-party-skill scrub: what it takes, what it must leave, and that it repeats.

A target package can ship agent guidance inside itself — decoupler installs
``site-packages/decoupler/_skills/data/SKILL.md`` plus a thirteen-file ``references/`` tree —
and a benchmark agent finds it by grepping the venv it was handed. Measured across four passes,
18.2% of baseline runs read it, so the baseline was not skill-free. These tests pin the removal
against the real shapes (an installed venv, a source checkout) and, just as importantly, pin
what it must NOT remove: a package whose own code lives in a ``skills/`` module still works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acumen.config import parse_config
from acumen.env import READY_MARKER, cache_key, prepare_target
from acumen.scrub import build_filtered_source, find_guidance, scrub_venv

SITE = "lib/python3.12/site-packages"


def write(path: Path, text: str = "x\n") -> Path:
    """Create ``path`` and its parents, with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    """A venv shaped like the real thing: a shipped skill, its installer, and ordinary code."""
    root = tmp_path / "venv"
    site = root / SITE
    write(site / "target/__init__.py", "__version__ = '1.0'\n")
    write(site / "target/_skills/__init__.py", "")
    write(site / "target/_skills/install.py", "def main(): ...\n")
    write(site / "target/_skills/data/SKILL.md", "---\nname: target\n---\n")
    write(site / "target/_skills/data/references/methods.md")
    write(site / "target/_skills/data/references/priors.md")
    # uv writes this console script for the installer entry point, and ``bin`` is on the PATH.
    write(root / "bin/target-install-skills", "from target._skills.install import main\nmain()\n")
    write(root / "bin/target-report", "from target.report import main\nmain()\n")
    return root


def test_scrub_venv_takes_the_shipped_skill_and_its_installer(venv: Path) -> None:
    """The skill data, the subpackage holding it, and the entry point that points at it."""
    removed = scrub_venv(venv)

    assert removed == [venv / SITE / "target/_skills", venv / "bin/target-install-skills"]
    assert not list(venv.rglob("SKILL.md"))
    # Nothing an agent could grep is left behind, under any name.
    assert not [path for path in venv.rglob("*") if "skill" in path.name.lower()]
    # ...and the package itself is untouched, so it still imports and behaves as installed.
    assert (venv / SITE / "target/__init__.py").read_text() == "__version__ = '1.0'\n"
    assert (venv / "bin/target-report").is_file()


def test_scrub_venv_is_idempotent(venv: Path) -> None:
    """It runs on every ``prepare_target``, including cache hits, so repeats must be free."""
    assert scrub_venv(venv)
    assert scrub_venv(venv) == []


def test_scrub_venv_keeps_a_skills_module_that_is_really_code(tmp_path: Path) -> None:
    """``skills`` is a plausible module name; only a ``SKILL.md`` beneath it makes it guidance."""
    root = tmp_path / "venv"
    write(root / SITE / "engine/skills/planner.py", "class Planner: ...\n")
    write(root / SITE / "engine/_skills/registry.py", "REGISTRY = {}\n")

    assert scrub_venv(root) == []
    assert (root / SITE / "engine/skills/planner.py").is_file()
    assert (root / SITE / "engine/_skills/registry.py").is_file()


@pytest.mark.parametrize("name", [".claude", ".agents", ".codex", ".cursor", ".claude-plugin"])
def test_scrub_venv_takes_guidance_directories_wherever_they_sit(tmp_path: Path, name: str) -> None:
    """These hold nothing but agent guidance, so they go on their name alone."""
    root = tmp_path / "venv"
    write(root / SITE / f"target/{name}/skills/target/SKILL.md")

    assert scrub_venv(root) == [root / SITE / "target" / name]
    assert not list(root.rglob("*.md"))


def test_scrub_venv_takes_a_skill_bundle_under_any_directory_name(tmp_path: Path) -> None:
    """A ``SKILL.md`` and the ``references/`` beside it are one bundle; the entry point is not enough."""
    root = tmp_path / "venv"
    write(root / SITE / "target/guidance/SKILL.md")
    write(root / SITE / "target/guidance/references/deep.md")
    write(root / SITE / "target/guidance/loader.py", "LOADER = 1\n")

    removed = scrub_venv(root)

    assert set(removed) == {
        root / SITE / "target/guidance/SKILL.md",
        root / SITE / "target/guidance/references",
    }
    # Only the guidance goes: the directory it sat in is code and stays.
    assert (root / SITE / "target/guidance/loader.py").is_file()


def test_find_guidance_reports_outermost_paths_only(venv: Path) -> None:
    """The result is removable as it stands, so nothing inside a hit is listed separately."""
    hits = find_guidance(venv)

    assert hits == [venv / SITE / "target/_skills"]
    assert not any(hit.name == "SKILL.md" for hit in hits)


def test_find_guidance_reports_without_removing(venv: Path) -> None:
    """Finding is separate from removing, so a caller can list a target without changing it."""
    assert find_guidance(venv)
    assert (venv / SITE / "target/_skills/data/SKILL.md").is_file()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A source checkout carrying every shape of guidance a repo can ship."""
    root = tmp_path / "src"
    write(root / "pyproject.toml", '[project]\nname = "target"\n')
    write(root / "src/target/__init__.py")
    write(root / "src/target/_skills/data/SKILL.md")
    write(root / "src/target/_skills/data/references/methods.md")
    write(root / "src/target/skills/planner.py", "class Planner: ...\n")
    write(root / "docs/tutorial.md", "# Tutorial\n")
    write(root / "CLAUDE.md")
    write(root / "AGENTS.md")
    write(root / ".claude/skills/target/SKILL.md")
    write(root / ".agents/skills/target/SKILL.md")
    write(root / ".claude-plugin/plugin.json", "{}\n")
    write(root / ".github/copilot-instructions.md")
    write(root / "skills/target/SKILL.md")
    write(root / ".git/objects/ab/cdef", "blob\n")
    return root


def test_filtered_source_strips_guidance_and_keeps_the_package(checkout: Path, tmp_path: Path) -> None:
    """What ``draft`` and ``tasks`` read: the package and its docs, none of its agent guidance."""
    copy = build_filtered_source(checkout, tmp_path / "copy")

    assert (copy / "pyproject.toml").is_file()
    assert (copy / "docs/tutorial.md").is_file()
    assert (copy / "src/target/__init__.py").is_file()
    # A ``skills`` module holding code is part of the package, so it survives the filter too.
    assert (copy / "src/target/skills/planner.py").is_file()
    assert not list(copy.rglob("SKILL.md"))
    assert not list(copy.rglob("*.md.orig"))
    for gone in (
        "src/target/_skills",
        "CLAUDE.md",
        "AGENTS.md",
        ".claude",
        ".agents",
        ".claude-plugin",
        ".github/copilot-instructions.md",
        "skills",
        # The object store holds every stripped file, so it cannot come along either.
        ".git",
    ):
        assert not (copy / gone).exists(), gone


def test_filtered_source_leaves_the_original_checkout_alone(checkout: Path, tmp_path: Path) -> None:
    """``ship`` commits from the real checkout, and a local target's checkout is the user's tree."""
    before = sorted(path.relative_to(checkout).as_posix() for path in checkout.rglob("*"))

    build_filtered_source(checkout, tmp_path / "copy")

    assert sorted(path.relative_to(checkout).as_posix() for path in checkout.rglob("*")) == before


def test_prepare_target_scrubs_a_cached_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A venv cached before the scrub existed is still marked ready, so the cache hit cleans it."""
    checkout = tmp_path / "target"
    write(checkout / "pyproject.toml", '[project]\nname = "target"\nversion = "0.1"\n')
    cfg = parse_config({"repo": str(checkout), "skill_name": "target"})

    cache = tmp_path / "cache"
    entry = cache / cache_key(cfg.repo, cfg.ref)
    write(entry / "venv/bin/python", "#!/bin/sh\n")
    write(entry / f"venv/{SITE}/target/_skills/data/SKILL.md")
    (entry / READY_MARKER).write_text(
        json.dumps({"src_dir": str(checkout), "commit": "local", "pkg_name": "target", "pkg_version": "0.1"})
    )
    # `uv` is only needed to *build* a venv; this target is a cache hit and never builds one.
    monkeypatch.setattr("acumen.env.shutil.which", lambda _name: "/usr/bin/uv")

    target = prepare_target(cfg, cache)

    assert target.pkg_version == "0.1"
    assert not list((entry / "venv").rglob("SKILL.md"))
    assert (entry / "venv/bin/python").is_file()
