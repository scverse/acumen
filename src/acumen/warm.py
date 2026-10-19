"""Warm the target's shared dataset cache before a benchmark pass.

Every benchmark sandbox symlinks its cwd-relative dataset directories (``config.dataset_cache_dirs``)
to one persistent directory per target (:attr:`acumen.env.Target.datasets_dir`), so a dataset is
downloaded once and served from there afterwards. That alone leaves one problem: a *concurrent*
pass, whose first several runs all need the same not-yet-downloaded dataset, starts several
downloads into the same place at once — a race the package's own cache logic is not built for.

So we warm first. This module finds the dataset-loader calls the benchmark will make — by static
analysis of the ground-truth confirmation scripts task generation persisted (the same scripts
:mod:`acumen.coverage` reads) — and executes each distinct call **once, sequentially**, in the
target venv, with the shared cache as the working directory. By the time the matrix runs, every
dataset it needs is already on disk and no run downloads anything.

Why static extraction and not "re-run the scripts": a confirmation script runs the whole analysis
(minutes), but only its ``sq.datasets.X(...)`` call matters here. Pulling that call out of the AST
and re-executing just it costs one download and nothing else. Only calls whose arguments are all
literals are extractable — a loader called with a computed argument is skipped and reported, never
guessed.

The warm subprocess is acumen's own code running the package, not an agent, so it runs under the
operator's normal environment — the env scrub exists to keep secrets away from a web-enabled
agent, which this is not.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen.coverage import attr_chain, collect_aliases, resolve_parts

#: The subpackage of the target under which dataset loaders live. squidpy/scanpy convention;
#: a target with a different layout would need this to become configurable.
DATASETS_MODULE = "datasets"


@dataclass(frozen=True)
class DatasetCall:
    """One dataset-loader call lifted from a script, ready to re-execute on its own."""

    #: Fully qualified loader, e.g. ``squidpy.datasets.visium``.
    qualname: str
    #: The call as re-executable source, e.g. ``squidpy.datasets.visium('V1_Adult_Mouse_Brain')``.
    source: str


def _literal(node: ast.AST) -> object:
    """``ast.literal_eval`` a node, raising ``ValueError`` if it is not a literal."""
    return ast.literal_eval(node)


def find_dataset_calls(script: str, pkg_name: str, *, module: str = DATASETS_MODULE) -> list[DatasetCall]:
    """Lift every ``<pkg>.<module>.<loader>(...)`` call with all-literal arguments from a script.

    Resolves the script's import aliases (``import squidpy as sq``, ``from squidpy.datasets import
    visium``) so the call is returned under its fully qualified name regardless of how the script
    spelled it. Duplicates within one script collapse to one call. A call with a non-literal
    argument is dropped — it cannot be re-executed faithfully in isolation.

    Raises
    ------
    SyntaxError
        If ``script`` does not parse; callers decide whether to skip that script.
    """
    tree = ast.parse(script)
    aliases = collect_aliases(tree, pkg_name)
    prefix = f"{pkg_name}.{module}."
    found: dict[str, DatasetCall] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = attr_chain(node.func)
        if not parts:
            continue
        qualname = resolve_parts(parts, aliases)
        if qualname is None or not qualname.startswith(prefix):
            continue
        try:
            args = [repr(_literal(a)) for a in node.args]
            kwargs = [f"{kw.arg}={_literal(kw.value)!r}" for kw in node.keywords if kw.arg is not None]
        except ValueError:
            continue  # computed argument — not reproducible on its own
        if any(kw.arg is None for kw in node.keywords):
            continue  # **kwargs splat — same reason
        source = f"{qualname}({', '.join([*args, *kwargs])})"
        found.setdefault(source, DatasetCall(qualname=qualname, source=source))
    return list(found.values())


def collect_dataset_calls(scripts: Mapping[str, str], pkg_name: str) -> list[DatasetCall]:
    """Union the dataset calls across many scripts (``{task_id: source}``), deduplicated, sorted.

    An unparseable script contributes nothing rather than aborting the warm-up: a bad script is a
    task-gen problem, and one should not stop every other dataset from being cached.
    """
    found: dict[str, DatasetCall] = {}
    for source in scripts.values():
        try:
            calls = find_dataset_calls(source, pkg_name)
        except SyntaxError:
            continue
        for call in calls:
            found.setdefault(call.source, call)
    return sorted(found.values(), key=lambda c: c.source)


@dataclass(frozen=True)
class WarmOutcome:
    """The result of executing one dataset call during warm-up."""

    call: DatasetCall
    ok: bool
    #: The tail of stderr on failure, for the operator; empty on success.
    error: str = ""


def warm_datasets(
    python: Path,
    pkg_name: str,
    calls: Sequence[DatasetCall],
    shared_root: Path,
    *,
    on_done: Callable[[WarmOutcome], None] | None = None,
) -> list[WarmOutcome]:
    """Execute each dataset call once, sequentially, with ``shared_root`` as the working directory.

    Sequential on purpose — it is the whole point (see the module docstring): one download per
    dataset, no concurrent writers to the same cache path. ``shared_root`` is the target's
    :attr:`~acumen.env.Target.datasets_dir`; because the loaders write to a cwd-relative directory,
    running with that cwd puts the files exactly where the sandboxes' symlinks point. A failing call
    is recorded and the rest continue — a dataset that will not download is a problem for the runs
    that need it, not for the others.

    Parameters
    ----------
    python
        The target venv interpreter (the package must be importable there).
    pkg_name
        The target package, imported before the call.
    calls
        Distinct calls to execute (see :func:`collect_dataset_calls`).
    shared_root
        The shared dataset cache; created if missing.
    on_done
        Optional progress callback per call.
    """
    shared_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[WarmOutcome] = []
    for call in calls:
        code = f"import {pkg_name}\n{call.source}\n"
        proc = subprocess.run([str(python), "-c", code], cwd=shared_root, capture_output=True, text=True)
        if proc.returncode == 0:
            outcome = WarmOutcome(call=call, ok=True)
        else:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            outcome = WarmOutcome(call=call, ok=False, error=f"exit {proc.returncode}: {tail}")
        outcomes.append(outcome)
        if on_done is not None:
            on_done(outcome)
    return outcomes
