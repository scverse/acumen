"""Tests for API coverage — introspection of the call surface and AST reference scanning.

``build_inventory`` is exercised against a synthetic package written to a tmp dir and imported,
so the test does not depend on any real target being installed. ``scan_references`` and the
``Coverage`` view are pure and tested directly.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from acumen.coverage import (
    CLASS,
    FUNCTION,
    Coverage,
    Inventory,
    Symbol,
    build_inventory,
    load_scripts,
    measure_coverage,
    scan_references,
)


def _write_pkg(root: Path) -> None:
    """A synthetic ``fakepkg`` with a public subpackage, a private one, and a re-export."""
    pkg = root / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import gr, _internal\n__version__ = '9.9.9'\n")
    # public subpackage with a function and a class defined here
    (pkg / "gr.py").write_text(
        textwrap.dedent("""
        from collections import OrderedDict  # a re-export that must NOT count
        def spatial_neighbors(x):
            return x
        class Graph:
            pass
        _hidden = OrderedDict  # underscore attr, ignored
        """)
    )
    # private subpackage — its symbols are not part of the public surface
    (pkg / "_internal.py").write_text("def secret():\n    return 1\n")


def test_build_inventory_public_surface(tmp_path: Path) -> None:
    _write_pkg(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        inv = build_inventory("fakepkg")
    finally:
        sys.path.remove(str(tmp_path))
        for mod in list(sys.modules):
            if mod == "fakepkg" or mod.startswith("fakepkg."):
                del sys.modules[mod]

    assert inv.package == "fakepkg"
    assert inv.version == "9.9.9"
    names = inv.names
    assert "fakepkg.gr.spatial_neighbors" in names
    assert "fakepkg.gr.Graph" in names
    # re-exported dependency symbol is excluded (defined outside the package)
    assert "fakepkg.gr.OrderedDict" not in names
    # private module and underscore attrs are excluded
    assert not any(".secret" in n or "_hidden" in n for n in names)
    kinds = {s.qualname: s.kind for s in inv.symbols}
    assert kinds["fakepkg.gr.spatial_neighbors"] == FUNCTION
    assert kinds["fakepkg.gr.Graph"] == CLASS


def test_inventory_round_trip(tmp_path: Path) -> None:
    inv = Inventory(
        package="p",
        version="1.0",
        symbols=(Symbol("p.gr.f", "p.gr._impl", FUNCTION), Symbol("p.C", "p", CLASS)),
    )
    path = tmp_path / "inventory.json"
    inv.write(path)
    assert Inventory.read(path) == inv


def test_scan_references_import_as() -> None:
    script = "import squidpy as sq\nsq.gr.spatial_neighbors(adata)\nsq.pl.spatial_scatter(adata)\n"
    refs = scan_references(script, "squidpy")
    assert refs == {"squidpy.gr.spatial_neighbors", "squidpy.pl.spatial_scatter"}


def test_scan_references_from_import() -> None:
    script = "from squidpy.gr import spatial_neighbors as sn\nsn(adata)\n"
    assert scan_references(script, "squidpy") == {"squidpy.gr.spatial_neighbors"}


def test_scan_references_submodule_import() -> None:
    script = "import squidpy.gr\nsquidpy.gr.nhood_enrichment(adata)\n"
    assert scan_references(script, "squidpy") == {"squidpy.gr.nhood_enrichment"}


def test_scan_references_ignores_foreign_and_unresolvable() -> None:
    script = textwrap.dedent("""
        import numpy as np
        import squidpy as sq
        np.mean(x)               # foreign package, ignored
        sq                       # bare module ref, not a call name, ignored
        obj.method().chained     # unresolvable chain, ignored
    """)
    assert scan_references(script, "squidpy") == set()


def _inv(*names: str) -> Inventory:
    return Inventory(package="squidpy", version="1", symbols=tuple(Symbol(n, "squidpy", FUNCTION) for n in names))


def test_measure_coverage_and_queue() -> None:
    inv = _inv("squidpy.gr.spatial_neighbors", "squidpy.gr.nhood_enrichment", "squidpy.pl.spatial_scatter")
    scripts = {
        "t1": "import squidpy as sq\nsq.gr.spatial_neighbors(a)\n",
        "t2": "import squidpy as sq\nsq.pl.spatial_scatter(a)\nsq.gr.not_in_inventory(a)\n",
    }
    cov = measure_coverage(inv, scripts)
    assert cov.covered == {"squidpy.gr.spatial_neighbors", "squidpy.pl.spatial_scatter"}
    # nhood_enrichment is the only inventory symbol nobody touched -> the queue
    assert cov.uncovered == ("squidpy.gr.nhood_enrichment",)
    assert cov.rate == 2 / 3


def test_measure_coverage_tolerates_bad_script() -> None:
    inv = _inv("squidpy.gr.spatial_neighbors")
    cov = measure_coverage(inv, {"broken": "def (:\n"})
    assert cov.per_task["broken"] == frozenset()
    assert cov.covered == frozenset()


def test_load_scripts(tmp_path: Path) -> None:
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "task_a.py").write_text("x = 1\n")
    (d / "task_b.py").write_text("y = 2\n")
    assert load_scripts(tmp_path / "nonexistent") == {}
    assert load_scripts(d) == {"task_a": "x = 1\n", "task_b": "y = 2\n"}


def test_coverage_empty_inventory() -> None:
    cov = Coverage(inventory=Inventory("p", "1", ()), per_task={})
    assert cov.rate == 0.0
    assert cov.uncovered == ()


def test_mention_references_reads_prose_and_aliases() -> None:
    from acumen.coverage import mention_references

    text = (
        "Build the graph with `sq.gr.spatial_neighbors(adata)` then run sq.gr.nhood_enrichment.\n"
        "```python\nsquidpy.pl.spatial_scatter(adata)\n```\nUnrelated: np.mean, mysq.gr.fake, sq.gr\n"
    )
    refs = mention_references(text, "squidpy", aliases=["sq"])
    assert refs == {
        "squidpy.gr.spatial_neighbors",
        "squidpy.gr.nhood_enrichment",
        "squidpy.pl.spatial_scatter",
        "squidpy.gr",
    }
    # Without the alias, the `sq.` spellings are invisible — the caller must say what the package is called.
    assert mention_references(text, "squidpy") == {"squidpy.pl.spatial_scatter"}


def test_skill_mentions_counts_only_inventory_symbols(tmp_path: Path) -> None:
    from acumen.coverage import skill_mentions

    inv = _inv("squidpy.gr.spatial_neighbors", "squidpy.gr.nhood_enrichment", "squidpy.im.process")
    skill = tmp_path / "v1"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: squidpy\ndescription: x\n---\nUse sq.gr.spatial_neighbors first.\n")
    (skill / "references" / "images.md").write_text("`sq.im.process` smooths; sq.gr.made_up does not exist.\n")
    (skill / "meta.json").write_text('{"note": "sq.gr.nhood_enrichment"}')  # bookkeeping, not content

    assert skill_mentions(inv, skill, aliases=["sq"]) == {"squidpy.gr.spatial_neighbors", "squidpy.im.process"}


def test_skill_mentions_credits_unique_bare_backticked_names(tmp_path: Path) -> None:
    from acumen.coverage import skill_mentions

    inv = _inv("squidpy.gr.sepal", "squidpy.gr.ligrec", "squidpy.pl.ligrec", "squidpy.im.process")
    skill = tmp_path / "v1"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: s\ndescription: x\n---\nRun `sepal(adata, ...)` for SVGs; `ligrec` pairs; then `process` and process.\n"
    )
    # `sepal` is unique -> credited. `ligrec` exists in gr AND pl -> ambiguous, not credited.
    # `process` is unique here but only backticked mentions count, and it is credited as im.process.
    assert skill_mentions(inv, skill, aliases=["sq"]) == {"squidpy.gr.sepal", "squidpy.im.process"}
