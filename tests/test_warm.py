"""Tests for the dataset-cache warm-up and the sandbox symlinks it serves.

``warm_datasets`` is exercised for real against a synthetic ``fakepkg`` on ``PYTHONPATH`` using
this interpreter, so the subprocess path, the cwd-relative download, and per-call failure
isolation are all covered without any network.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from acumen.config import ConfigError, parse_config
from acumen.env import Target
from acumen.sandbox import link_dataset_cache, sandbox
from acumen.warm import DatasetCall, collect_dataset_calls, find_dataset_calls, warm_datasets

# ── Static extraction ───────────────────────────────────────────────────────────────────


def test_find_dataset_calls_resolves_aliases_and_literals() -> None:
    script = textwrap.dedent("""
        import squidpy as sq
        from squidpy.datasets import merfish as mf
        a = sq.datasets.visium("V1_Adult_Mouse_Brain", include_hires_tiff=True)
        b = mf()
        c = sq.datasets.visium("V1_Adult_Mouse_Brain", include_hires_tiff=True)  # duplicate
        sq.gr.spatial_neighbors(a)  # not a dataset loader
    """)
    calls = find_dataset_calls(script, "squidpy")
    assert {c.source for c in calls} == {
        "squidpy.datasets.visium('V1_Adult_Mouse_Brain', include_hires_tiff=True)",
        "squidpy.datasets.merfish()",
    }
    assert {c.qualname for c in calls} == {"squidpy.datasets.visium", "squidpy.datasets.merfish"}


def test_find_dataset_calls_skips_computed_arguments() -> None:
    script = "import squidpy as sq\nname = pick()\nsq.datasets.visium(name)\nsq.datasets.visium(**opts)\n"
    assert find_dataset_calls(script, "squidpy") == []


def test_collect_dataset_calls_unions_and_tolerates_bad_script() -> None:
    scripts = {
        "t1": "import squidpy as sq\nsq.datasets.merfish()\n",
        "t2": "import squidpy as sq\nsq.datasets.merfish()\nsq.datasets.seqfish()\n",
        "bad": "def (:\n",
    }
    calls = collect_dataset_calls(scripts, "squidpy")
    assert [c.source for c in calls] == ["squidpy.datasets.merfish()", "squidpy.datasets.seqfish()"]


# ── Execution against a fake package ────────────────────────────────────────────────────


def _fake_pkg(root: Path) -> None:
    """``fakepkg.datasets`` whose loaders write into a cwd-relative ``data/`` like squidpy does."""
    pkg = root / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import datasets\n")
    (pkg / "datasets.py").write_text(
        textwrap.dedent("""
        from pathlib import Path
        def good(name="x"):
            d = Path("data"); d.mkdir(exist_ok=True)
            (d / f"{name}.h5ad").write_text("blob")
        def broken():
            raise RuntimeError("download failed")
        """)
    )


def test_warm_datasets_downloads_into_shared_root_and_isolates_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pkg(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    shared = tmp_path / "datasets"
    calls = [
        DatasetCall("fakepkg.datasets.good", "fakepkg.datasets.good('merfish')"),
        DatasetCall("fakepkg.datasets.broken", "fakepkg.datasets.broken()"),
        DatasetCall("fakepkg.datasets.good", "fakepkg.datasets.good(name='seqfish')"),
    ]
    seen: list[str] = []
    outcomes = warm_datasets(
        Path(sys.executable), "fakepkg", calls, shared, on_done=lambda o: seen.append(o.call.source)
    )

    assert [o.ok for o in outcomes] == [True, False, True]  # the failure did not stop the rest
    assert "download failed" in outcomes[1].error
    assert seen == [c.source for c in calls]  # sequential, in order
    # The cwd-relative `data/` landed in the shared root — exactly where sandboxes symlink to.
    assert (shared / "data" / "merfish.h5ad").read_text() == "blob"
    assert (shared / "data" / "seqfish.h5ad").exists()


# ── Sandbox symlinks ────────────────────────────────────────────────────────────────────


def test_link_dataset_cache_creates_symlinks_to_shared_dirs(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    shared = tmp_path / "datasets"
    links = link_dataset_cache(root, shared, ["data", "cache"])

    assert [p.name for p in links] == ["data", "cache"]
    for name in ("data", "cache"):
        link = root / name
        assert link.is_symlink() and link.resolve() == (shared / name).resolve()
        assert (shared / name).is_dir()  # shared side created eagerly
    # A file written through the sandbox path is visible in the shared cache.
    (root / "data" / "x.h5ad").write_text("blob")
    assert (shared / "data" / "x.h5ad").read_text() == "blob"


def test_sandbox_links_dataset_cache_dirs(tmp_path: Path) -> None:
    asyncio.run(_sandbox_links_dataset_cache_dirs(tmp_path))


async def _sandbox_links_dataset_cache_dirs(tmp_path: Path) -> None:
    venv = tmp_path / "entry" / "venv"
    venv.mkdir(parents=True)
    target = Target(
        source="x", ref="main", src_dir=tmp_path / "src", venv_dir=venv, commit="c", pkg_name="p", pkg_version="1"
    )
    async with sandbox(target, auth_mode="api", base=tmp_path / "sb", dataset_cache_dirs=["data"]) as box:
        assert (box.root / "data").is_symlink()
        assert (box.root / "data").resolve() == (target.datasets_dir / "data").resolve()
    # Nothing linked when not configured.
    async with sandbox(target, auth_mode="api", base=tmp_path / "sb2") as box:
        assert not (box.root / "data").exists()


# ── Config ──────────────────────────────────────────────────────────────────────────────


def test_config_dataset_cache_dirs() -> None:
    cfg = parse_config({"repo": "https://github.com/scverse/squidpy"})
    assert cfg.dataset_cache_dirs == []
    cfg = parse_config({"repo": "https://github.com/scverse/squidpy", "dataset_cache_dirs": ["data", "cache"]})
    assert cfg.dataset_cache_dirs == ["data", "cache"]
    with pytest.raises(ConfigError, match="single directory name"):
        parse_config({"repo": "https://github.com/scverse/squidpy", "dataset_cache_dirs": ["../escape"]})
    with pytest.raises(ConfigError, match="single directory name"):
        parse_config({"repo": "https://github.com/scverse/squidpy", "dataset_cache_dirs": ["a/b"]})
