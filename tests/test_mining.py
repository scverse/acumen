"""Tests for task mining — the structural quality gate, both harvest legs (gh mocked), and output.

Nothing here touches the network: ``gh`` is replaced by a runner returning canned JSON/raw text,
and the web leg is fed a fake fetcher. What is under test is the *gate* — which files count as an
analysis — the notebook flattening, dedup, provenance, and the seam into sharded task generation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from acumen import taskshard
from acumen.config import Config
from acumen.env import Target
from acumen.mining import (
    HEADER_PREFIX,
    INDEX_FILE,
    Candidate,
    MineResult,
    _repo_id,
    assess,
    default_queries,
    extract_code_from_page,
    gh_search,
    mine_github,
    mine_urls,
    notebook_to_source,
    submodule_repos,
    toplevel_package_calls,
    write_candidates,
)
from acumen.tasks import Task, TaskSplit
from acumen.taskshard import generate_tasks_sharded

INVENTORY = frozenset({"squidpy.gr.spatial_neighbors", "squidpy.gr.nhood_enrichment", "squidpy.datasets.merfish"})

ANALYSIS = "import squidpy as sq\nadata = sq.datasets.merfish()\nsq.gr.spatial_neighbors(adata)\n"
LIBRARY = "import squidpy as sq\n\ndef run(adata):\n    sq.gr.spatial_neighbors(adata)\n    return adata\n"


# ── Gate ────────────────────────────────────────────────────────────────────────────────


def test_toplevel_calls_distinguish_analysis_from_library() -> None:
    assert toplevel_package_calls(ANALYSIS, "squidpy") == {"squidpy.datasets.merfish", "squidpy.gr.spatial_neighbors"}
    assert toplevel_package_calls(LIBRARY, "squidpy") == set()
    # Calls under top-level control flow still count as analysis (a notebook's `for` loop).
    looped = "import squidpy as sq\nfor a in x:\n    sq.gr.nhood_enrichment(a, cluster_key='c')\n"
    assert toplevel_package_calls(looped, "squidpy") == {"squidpy.gr.nhood_enrichment"}


def test_assess_keeps_analyses_and_rejects_the_rest() -> None:
    assert assess(ANALYSIS, "squidpy", INVENTORY) == ("squidpy.datasets.merfish", "squidpy.gr.spatial_neighbors")
    assert assess(LIBRARY, "squidpy", INVENTORY) == "no top-level package call (library code, not an analysis)"
    assert assess("import numpy as np\nnp.mean([1])\n", "squidpy", INVENTORY) == (
        "references no public API symbol of the package"
    )
    # A symbol outside the real inventory does not count, even if it looks like the package.
    assert "no public API symbol" in assess("import squidpy as sq\nsq.gr.made_up(1)\n", "squidpy", INVENTORY)
    assert assess("def (:\n", "squidpy", INVENTORY) == "does not parse as Python"
    assert "too large" in assess("x = 1\n" * 100_000, "squidpy", INVENTORY)


def test_notebook_to_source_flattens_code_cells_and_comments_magics() -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title"]},
            {"cell_type": "code", "source": ["%matplotlib inline\n", "import squidpy as sq\n"]},
            {"cell_type": "code", "source": "!pip install squidpy\nsq.gr.spatial_neighbors(adata)"},
        ]
    }
    src = notebook_to_source(json.dumps(nb))
    assert src.count("# %%") == 2
    assert "# %matplotlib inline" in src and "# !pip install" in src
    assert toplevel_package_calls(src, "squidpy") == {"squidpy.gr.spatial_neighbors"}


# ── GitHub leg (gh mocked) ──────────────────────────────────────────────────────────────

_SHA = "a" * 40


def _hit(repo: str, path: str) -> dict:
    return {"path": path, "repository": {"nameWithOwner": repo}, "url": f"https://github.com/{repo}/blob/{_SHA}/{path}"}


def _fake_gh(files: dict[tuple[str, str], str], search_hits: list[dict]):
    """A ``gh`` stand-in: search returns ``search_hits`` for every query, api returns file text."""
    calls: list[list[str]] = []

    def run(args):
        calls.append(list(args))
        if args[0] == "search":
            return json.dumps(search_hits)
        if args[0] == "api":
            # repos/{owner}/{name}/contents/{path}?ref=sha
            spec = args[1].split("/contents/", 1)
            repo = spec[0].removeprefix("repos/")
            path = spec[1].split("?", 1)[0]
            return files[(repo, path)]
        raise AssertionError(args)

    return run, calls


def test_gh_search_parses_hits_and_pins_the_commit() -> None:
    run, calls = _fake_gh({}, [_hit("lab/proj", "analysis.py")])
    hits = gh_search("import squidpy", limit=5, language="python", run=run)
    assert [(h.repo, h.path, h.sha) for h in hits] == [("lab/proj", "analysis.py", _SHA)]
    assert "--language" in calls[0] and "--limit" in calls[0]


def test_mine_github_gates_dedups_and_excludes_the_target_repo() -> None:
    nb = json.dumps({"cells": [{"cell_type": "code", "source": ANALYSIS}]})
    files = {
        ("lab/proj", "analysis.py"): ANALYSIS,
        ("lab/proj", "notebooks/fig3.ipynb"): nb,
        ("other/copy", "vendored/analysis.py"): ANALYSIS,  # same content -> duplicate
        ("tools/wrap", "src/wrap/core.py"): LIBRARY,  # library, not an analysis
        ("scverse/squidpy", "docs/x.py"): ANALYSIS,  # the target itself
        ("lab/proj", "tests/test_it.py"): ANALYSIS,  # tests dir
    }
    hits = [_hit(repo, path) for (repo, path) in files]
    run, calls = _fake_gh(files, hits)
    slept: list[float] = []

    result = mine_github(
        "squidpy",
        ["import squidpy"],
        exclude_repos=["scverse/squidpy"],
        limit=50,
        inventory=INVENTORY,
        run=run,
        sleep=slept.append,
    )

    assert [(c.repo, c.path, c.kind) for c in result.kept] == [
        ("lab/proj", "analysis.py", "py"),
        ("lab/proj", "notebooks/fig3.ipynb", "ipynb"),
    ]
    assert result.kept[0].sha == _SHA and result.kept[0].origin.endswith("/analysis.py")
    assert result.kept[0].symbols == ("squidpy.datasets.merfish", "squidpy.gr.spatial_neighbors")
    reasons = result.reasons()
    assert reasons["duplicate content"] == 1
    assert reasons["no top-level package call (library code, not an analysis)"] == 1
    assert reasons["the target package's own repository (or a submodule of it)"] == 1
    assert reasons["under a tests/vendored/build directory"] == 1
    # One query x two file types = two searches, paced by one sleep between them.
    assert sum(1 for c in calls if c[0] == "search") == 2 and len(slept) == 1
    # The notebook was flattened before assessment and kept as a candidate.
    assert result.kept[1].text.startswith("# %%")


def test_submodule_repos_reads_gitmodules(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text(
        '[submodule "docs/notebooks"]\n\tpath = docs/notebooks\n\turl = https://github.com/scverse/squidpy-tutorials.git\n'
        '[submodule "x"]\n\tpath = x\n\turl = git@github.com:scverse/other.git\n'
    )
    assert submodule_repos(tmp_path) == ["scverse/squidpy-tutorials", "scverse/other"]
    assert submodule_repos(tmp_path / "nowhere") == []


def test_default_queries_use_the_conventional_alias() -> None:
    qs = default_queries("squidpy")
    assert qs[0] == "import squidpy" and "sq.gr." in qs and "squidpy.datasets" in qs
    assert "zz.gr." in default_queries("squidpy", alias="zz")
    assert _repo_id("https://github.com/scverse/squidpy") == "scverse/squidpy"
    assert _repo_id("git@github.com:scverse/squidpy.git") == "scverse/squidpy"
    assert _repo_id("/local/path") is None


# ── Web leg ─────────────────────────────────────────────────────────────────────────────


def test_extract_code_from_page_by_kind() -> None:
    html = (
        "<h1>Tutorial</h1><pre>import squidpy as sq\nsq.gr.spatial_neighbors(adata)\n</pre><pre>$ pip install x</pre>"
    )
    code = extract_code_from_page(html, url="https://docs.example.org/tutorial.html")
    assert "sq.gr.spatial_neighbors" in code and "pip install" not in code
    md = "text\n```python\nimport squidpy as sq\nsq.gr.nhood_enrichment(a, cluster_key='c')\n```\n"
    assert "nhood_enrichment" in extract_code_from_page(md, url="https://x/README.md")
    assert extract_code_from_page(ANALYSIS, url="https://x/a.py") == ANALYSIS
    doctest = "<pre>>>> import squidpy as sq\n>>> sq.gr.spatial_neighbors(adata)\n</pre>"
    assert ">>>" not in extract_code_from_page(doctest, url="https://x/page")


def test_mine_urls_uses_the_same_gate() -> None:
    pages = {
        "https://a/tut.html": f"<pre>{ANALYSIS}</pre>",
        "https://b/lib.html": f"<pre>{LIBRARY}</pre>",
        "https://c/empty.html": "<p>no code</p>",
    }
    result = mine_urls("squidpy", list(pages), inventory=INVENTORY, fetch=pages.__getitem__)
    assert [c.origin for c in result.kept] == ["https://a/tut.html"]
    assert result.kept[0].source == "web" and result.kept[0].kind == "page"
    assert {r.origin: r.reason for r in result.rejected} == {
        "https://b/lib.html": "no top-level package call (library code, not an analysis)",
        "https://c/empty.html": "no Python code blocks found",
    }


# ── Output + the task-gen seam ──────────────────────────────────────────────────────────


def _cand(slug: str, origin: str, text: str = ANALYSIS) -> Candidate:
    return Candidate(slug=slug, source="github", origin=origin, kind="py", text=text, symbols=("squidpy.gr.x",))


def test_write_candidates_is_additive_with_provenance(tmp_path: Path) -> None:
    out = tmp_path / "mined"
    first = MineResult(kept=[_cand("lab-proj-analysis", "https://x/1")])
    written, _ = write_candidates([first], out)
    assert [p.name for p in written] == ["lab-proj-analysis.py"]
    assert (out / "lab-proj-analysis.py").read_text().startswith(f"{HEADER_PREFIX}https://x/1\n")

    # Re-run with the same candidate plus a slug collision from a different origin.
    second = MineResult(kept=[_cand("lab-proj-analysis", "https://x/1"), _cand("lab-proj-analysis", "https://y/2")])
    written, _ = write_candidates([second], out)
    assert [p.name for p in written] == ["lab-proj-analysis-2.py"]  # the original is untouched
    index = json.loads((out / INDEX_FILE).read_text())
    assert [e["file"] for e in index["candidates"]] == ["lab-proj-analysis.py", "lab-proj-analysis-2.py"]
    assert all("text" not in e for e in index["candidates"])


def test_sharded_generation_over_candidates_seeds_the_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='pkg'\n")
    mined = tmp_path / "mined"
    mined.mkdir()
    (mined / "lab-proj-analysis.py").write_text(f"{HEADER_PREFIX}https://x/1\n{ANALYSIS}")
    seen: dict[str, tuple[str, dict | None]] = {}

    async def fake_agent(*, work_root: Path, make_prompt, seed_files=None, **_):
        seen[work_root.name] = (
            make_prompt(work_root / "work" / "tasks.yaml", work_root / "work" / "scripts"),
            seed_files,
        )
        task = Task(id="t", train=TaskSplit("g", "A"), test=TaskSplit("g", "B"))
        return [task], type("R", (), {"num_turns": 1})(), 0.1

    monkeypatch.setattr(taskshard, "_run_shard_agent", fake_agent)
    target = Target(
        source=str(src),
        ref="main",
        src_dir=src,
        venv_dir=tmp_path / "venv",
        commit="c",
        pkg_name="pkg",
        pkg_version="1",
    )
    result = asyncio.run(
        generate_tasks_sharded(
            cfg=Config(repo=str(src), skill_name="pkg"),
            target=target,
            out_path=tmp_path / "tasks.yaml",
            shards_dir=tmp_path / "shards",
            candidates_dir=mined,
        )
    )

    prompt, seeds = seen["lab-proj-analysis"]
    assert seeds == {"analysis-lab-proj-analysis.py": (mined / "lab-proj-analysis.py").read_text()}
    assert "analysis-lab-proj-analysis.py" in prompt and "OWN DATA IS NOT AVAILABLE" in prompt
    assert [t.id for t in result.tasks] == ["lab-proj-analysis__t"]  # namespaced like a notebook shard
