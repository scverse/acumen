"""API coverage: which of the target package's public symbols the task set exercises.

A task set is only a benchmark of the *package* to the extent it touches the package's API.
Two questions this module answers, so generation can be driven toward the gaps rather than
piling more tasks onto the same handful of functions:

* **What could be covered?** :func:`build_inventory` introspects the *installed* package and
  enumerates its public call surface — the dotted names a user actually writes
  (``squidpy.gr.spatial_neighbors``), not the private module a symbol is defined in. This is
  the denominator.
* **What is covered?** :func:`measure_coverage` reads the ground-truth confirmation scripts the
  task-generation agent persisted (one per task, ``scripts/<task_id>.py``) and, by **static AST
  analysis**, records which inventory symbols each script references. The union is the numerator.

Why static analysis and not a runtime tracer: the ground-truth scripts are run *by the agent*
inside a throwaway sandbox, so a ``sys.setprofile`` hook would have to survive that teardown and
would count every exploratory run, not the confirmed pipeline; and re-executing the scripts
ourselves to trace them re-pays the (often minutes-long) pipeline cost purely to learn which names
were called. squidpy analyses are written as direct attribute calls (``sq.gr.spatial_neighbors(...)``),
so an AST scan recovers the referenced symbols essentially perfectly and for free. It is
deliberately *conservative*: a name it cannot resolve statically simply does not count as covered,
which errs toward re-queueing a symbol rather than declaring a gap closed it hasn't.

``inventory − covered`` is the generation queue: the symbols no task exercises yet, which the next
task-generation pass should target.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Kinds of public callable we count as part of the API surface.
FUNCTION = "function"
CLASS = "class"


class CoverageError(RuntimeError):
    """Raised when the package cannot be introspected for its API surface."""


@dataclass(frozen=True)
class Symbol:
    """One public, in-package callable, named as a user calls it (``squidpy.gr.spatial_neighbors``)."""

    qualname: str
    module: str
    kind: str

    def as_dict(self) -> dict[str, str]:
        """A JSON-serializable view of this symbol."""
        return {"qualname": self.qualname, "module": self.module, "kind": self.kind}


@dataclass(frozen=True)
class Inventory:
    """The target package's public call surface — the denominator for coverage."""

    package: str
    version: str
    symbols: tuple[Symbol, ...]

    @property
    def names(self) -> frozenset[str]:
        """The dotted call names, as a set for intersection with references."""
        return frozenset(s.qualname for s in self.symbols)

    def as_dict(self) -> dict[str, object]:
        """A JSON-serializable view of the inventory."""
        return {
            "package": self.package,
            "version": self.version,
            "symbols": [s.as_dict() for s in self.symbols],
        }

    def write(self, path: Path) -> None:
        """Persist to JSON so the (import-heavy) introspection is done once per package version."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Inventory:
        """Rebuild an inventory from its :meth:`as_dict` form."""
        return cls(
            package=str(data["package"]),
            version=str(data["version"]),
            symbols=tuple(Symbol(s["qualname"], s["module"], s["kind"]) for s in data["symbols"]),  # type: ignore[index]
        )

    @classmethod
    def read(cls, path: Path) -> Inventory:
        """Load an inventory previously persisted by :meth:`write`."""
        return cls.from_dict(json.loads(path.read_text()))


def _iter_public_modules(pkg: object, pkg_name: str) -> Iterable[str]:
    """Yield the dotted names of ``pkg`` and every public in-package submodule reachable from it.

    Walks the package tree with :func:`pkgutil.walk_packages`, skipping any path component that
    starts with ``_`` — private modules are not part of the call surface a user writes. Import
    failures on individual submodules (an optional-dependency subpackage, say) are swallowed: a
    module that cannot be imported cannot contribute callable symbols anyway.
    """
    yield pkg_name
    paths = getattr(pkg, "__path__", None)
    if paths is None:
        return  # a single-module package, not a package tree
    for info in pkgutil.walk_packages(paths, prefix=f"{pkg_name}."):
        rel = info.name[len(pkg_name) + 1 :]
        if any(part.startswith("_") for part in rel.split(".")):
            continue
        yield info.name


def build_inventory(pkg_name: str) -> Inventory:
    """Introspect the installed package and enumerate its public call surface.

    A symbol is counted when it is a public (non-underscore) attribute of a public in-package
    module, is a function or class, and is *defined in* the package (``__module__`` starts with
    ``pkg_name``) rather than re-exported from a dependency — an ``AnnData`` re-exported at
    ``squidpy.datasets`` should not inflate squidpy's own surface. The recorded ``qualname`` is the
    public attribute path (``squidpy.gr.spatial_neighbors``), which is the form task scripts write
    and therefore the form :func:`measure_coverage` matches against — not the private definition
    module.

    Returns
    -------
    The inventory, symbols sorted by qualified name.

    Raises
    ------
    CoverageError
        If the top-level package cannot be imported.
    """
    try:
        top = importlib.import_module(pkg_name)
    except Exception as err:  # any import failure is the same operator-facing problem
        raise CoverageError(f"cannot import target package {pkg_name!r} to introspect its API: {err}") from err
    version = getattr(top, "__version__", "") or _installed_version(pkg_name)

    found: dict[str, Symbol] = {}
    for mod_name in _iter_public_modules(top, pkg_name):
        try:
            module = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - a submodule that won't import contributes nothing
            continue
        for attr, obj in vars(module).items():
            if attr.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            origin = getattr(obj, "__module__", "") or ""
            if origin != pkg_name and not origin.startswith(f"{pkg_name}."):
                continue  # re-exported from a dependency; not this package's own surface
            qualname = f"{mod_name}.{attr}"
            kind = CLASS if inspect.isclass(obj) else FUNCTION
            found.setdefault(qualname, Symbol(qualname=qualname, module=origin, kind=kind))
    return Inventory(package=pkg_name, version=version, symbols=tuple(sorted(found.values(), key=lambda s: s.qualname)))


def _installed_version(pkg_name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return "unknown"


def inventory_in_venv(python: Path, pkg_name: str) -> Inventory:
    """Build the inventory by introspecting the package in the **target venv**, not this process.

    :func:`build_inventory` has to import the target package, which is installed in the target's
    venv — never in the acumen process. So we shell out to that interpreter and run the
    introspection there, printing the inventory as JSON we read back. The bootstrap loads this very
    file *by path* (``importlib.util``) rather than ``import acumen.coverage``: the ``acumen``
    package's ``__init__`` pulls in the agent SDK, which the target venv does not have, so importing
    the package there would fail before reaching this stdlib-only module.

    Raises
    ------
    CoverageError
        If the subprocess fails or prints nothing parseable.
    """
    import subprocess

    # Redirect stdout to stderr while importing the package and introspecting, so a package that
    # prints a banner on import cannot corrupt the JSON — only the final dump reaches real stdout.
    bootstrap = (
        "import importlib.util,sys,json,contextlib\n"
        f"spec=importlib.util.spec_from_file_location('_acumen_cov',{str(Path(__file__).resolve())!r})\n"
        "m=importlib.util.module_from_spec(spec); sys.modules['_acumen_cov']=m; spec.loader.exec_module(m)\n"
        "with contextlib.redirect_stdout(sys.stderr):\n"
        f"    inv=m.build_inventory({pkg_name!r})\n"
        "sys.stdout.write(json.dumps(inv.as_dict()))\n"
    )
    proc = subprocess.run([str(python), "-c", bootstrap], capture_output=True, text=True)
    if proc.returncode != 0:
        raise CoverageError(
            f"introspecting {pkg_name!r} in the target venv failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    try:
        return Inventory.from_dict(json.loads(proc.stdout))
    except (ValueError, KeyError) as err:
        raise CoverageError(f"could not parse inventory JSON from the target venv: {err}") from err


def attr_chain(node: ast.AST) -> list[str] | None:
    """Flatten an attribute/name access into its dotted parts, or ``None`` if it isn't one.

    ``sq.gr.spatial_neighbors`` -> ``["sq", "gr", "spatial_neighbors"]``. A subscript or call in
    the middle of the chain (``a.b()[0].c``) makes it unresolvable as a static dotted name, so we
    give up and return ``None`` rather than guess.
    """
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


def scan_references(script: str, pkg_name: str) -> set[str]:
    """Return the package-qualified dotted names a script statically references.

    Resolves the two import forms a script uses against ``pkg_name`` and reports every attribute
    access that lands inside the package as its full dotted call name:

    * ``import squidpy as sq`` / ``import squidpy.gr`` — an alias (or the real name) for a module,
      so ``sq.gr.spatial_neighbors`` resolves to ``squidpy.gr.spatial_neighbors``.
    * ``from squidpy.gr import spatial_neighbors as sn`` — a local name bound to a qualified
      symbol, so a bare ``sn(...)`` resolves to ``squidpy.gr.spatial_neighbors``.

    A name it cannot resolve to ``pkg_name`` is not returned — the scan is intentionally
    conservative (see the module docstring). Returns the *referenced* names; intersect with an
    :class:`Inventory` to keep only real API symbols.

    Raises
    ------
    SyntaxError
        If ``script`` is not parseable Python — the caller decides whether that invalidates a task.
    """
    tree = ast.parse(script)
    aliases = collect_aliases(tree, pkg_name)

    # Only the *maximal* attribute chain is a reference: in `sq.gr.spatial_neighbors`, the inner
    # `sq.gr` is a module we pass through, not a symbol. Skip any node that is itself the `.value`
    # of an enclosing Attribute.
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            inner.add(id(node.value))

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute | ast.Name):
            continue
        if id(node) in inner:
            continue
        parts = attr_chain(node)
        if not parts:
            continue
        resolved = resolve_parts(parts, aliases)
        if resolved is not None:
            referenced.add(resolved)
    return referenced


@dataclass(frozen=True)
class Aliases:
    """How a script's local names map onto a package: the import bindings the resolver needs."""

    #: local module alias -> real dotted module (``sq`` -> ``squidpy``, ``g`` -> ``squidpy.gr``)
    modules: Mapping[str, str]
    #: local symbol name -> real dotted qualname (``sn`` -> ``squidpy.gr.spatial_neighbors``)
    symbols: Mapping[str, str]


def collect_aliases(tree: ast.AST, pkg_name: str) -> Aliases:
    """Gather the import bindings of ``pkg_name`` in a parsed script.

    Handles the two forms a script uses: ``import squidpy as sq`` / ``import squidpy.gr [as g]``
    bind a *module* alias; ``from squidpy.gr import spatial_neighbors [as sn]`` binds a *symbol*
    alias. ``import squidpy.gr`` with no ``as`` binds only the top name (``squidpy``) — Python
    semantics, and the resolver relies on it. Star imports are ignored (unresolvable statically).
    """
    modules: dict[str, str] = {}
    symbols: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != pkg_name and not alias.name.startswith(f"{pkg_name}."):
                    continue
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    top = alias.name.split(".", 1)[0]
                    modules[top] = top
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (mod == pkg_name or mod.startswith(f"{pkg_name}.")):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    symbols[alias.asname or alias.name] = f"{mod}.{alias.name}"
    return Aliases(modules=modules, symbols=symbols)


def resolve_parts(parts: Sequence[str], aliases: Aliases) -> str | None:
    """Resolve a dotted access (``["sq", "gr", "f"]``) to a package-qualified name, or ``None``.

    A module alias head needs at least one more part (a bare ``sq`` names the package, not a
    symbol); a symbol alias must be a bare name (``sn``, not ``sn.something``). Anything else is
    not a reference into the package.
    """
    head = parts[0]
    if head in aliases.modules:
        return ".".join([aliases.modules[head], *parts[1:]]) if len(parts) > 1 else None
    if head in aliases.symbols and len(parts) == 1:
        return aliases.symbols[head]
    return None


@dataclass(frozen=True)
class Coverage:
    """Which inventory symbols the task set exercises, and which it does not."""

    inventory: Inventory
    per_task: Mapping[str, frozenset[str]]

    @property
    def covered(self) -> frozenset[str]:
        """Inventory symbols referenced by at least one task's ground-truth script."""
        names = self.inventory.names
        hit: set[str] = set()
        for refs in self.per_task.values():
            hit |= refs & names
        return frozenset(hit)

    @property
    def uncovered(self) -> tuple[str, ...]:
        """The generation queue: inventory symbols no task references yet, sorted."""
        return tuple(sorted(self.inventory.names - self.covered))

    @property
    def rate(self) -> float:
        """Fraction of the inventory the task set covers, in ``[0, 1]``."""
        total = len(self.inventory.names)
        return len(self.covered) / total if total else 0.0


def measure_coverage(inventory: Inventory, scripts: Mapping[str, str]) -> Coverage:
    """Measure how much of ``inventory`` the ground-truth ``scripts`` exercise.

    Parameters
    ----------
    inventory
        The package's public call surface (:func:`build_inventory`).
    scripts
        ``{task_id: script_source}`` — the persisted ground-truth confirmation scripts. An
        unparseable script contributes no references (its :class:`SyntaxError` is swallowed here so
        one bad script cannot blind the whole coverage reading; task-gen validation is where a bad
        script is rejected).

    Returns
    -------
    A :class:`Coverage` view over the inventory and per-task references.
    """
    per_task: dict[str, frozenset[str]] = {}
    for task_id, source in scripts.items():
        try:
            per_task[task_id] = frozenset(scan_references(source, inventory.package))
        except SyntaxError:
            per_task[task_id] = frozenset()
    return Coverage(inventory=inventory, per_task=per_task)


def mention_references(text: str, pkg_name: str, *, aliases: Sequence[str] = ()) -> set[str]:
    """Package-qualified dotted names *mentioned* anywhere in ``text`` — prose, backticks, code.

    The counterpart of :func:`scan_references` for documents that are not Python: a ``SKILL.md``
    says ``sq.gr.spatial_neighbors`` in a sentence or a fenced block and never imports anything, so
    an AST cannot see it. A regex over ``<pkg>.<a>.<b>`` and ``<alias>.<a>.<b>`` spellings can, and
    for a skill that is the honest question — *does it teach this symbol at all?* Conservative in
    the same direction as the AST scan: a name the text does not spell out is not mentioned.

    Parameters
    ----------
    aliases
        Import aliases to accept as the package (``["sq"]`` for squidpy) — skills use the
        conventional alias without ever importing it.
    """
    heads = "|".join(re.escape(h) for h in (pkg_name, *aliases))
    pattern = re.compile(rf"(?<![\w.])(?:{heads})((?:\.[A-Za-z_]\w*)+)")
    found: set[str] = set()
    for match in pattern.finditer(text):
        found.add(f"{pkg_name}{match.group(1)}")
    return found


def skill_mentions(inventory: Inventory, skill_dir: Path, *, aliases: Sequence[str] = ()) -> frozenset[str]:
    """Inventory symbols a skill's content files mention — what the skill *teaches*.

    Set beside :attr:`Coverage.covered` this gives the two numbers that matter for a skill: what the
    benchmark verifies, and what the skill claims. Their difference is the guidance nothing tests.
    """
    from acumen.skills import content_files

    mentioned: set[str] = set()
    names = inventory.names
    # A bare backticked name (`spatial_neighbors`) counts when exactly one inventory symbol has
    # that leaf — skills routinely write "call `nhood_enrichment`" without the dotted path. An
    # ambiguous leaf (`process`, `visium` — several symbols, or a common word) does not count:
    # crediting it would credit a guess.
    leaves: dict[str, list[str]] = {}
    for name in names:
        leaves.setdefault(name.rsplit(".", 1)[1], []).append(name)
    unique_leaf = {leaf: owners[0] for leaf, owners in leaves.items() if len(owners) == 1}
    for path in content_files(skill_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # A mention of `sq.gr.spatial_neighbors(adata)` also spells every prefix; only whole
        # inventory names count, so `sq.gr` alone is not a symbol.
        mentioned |= mention_references(text, inventory.package, aliases=aliases) & names
        for token in _BACKTICK_RE.findall(text):
            if token in unique_leaf:
                mentioned.add(unique_leaf[token])
    return frozenset(mentioned)


#: A backticked bare identifier, optionally called: `` `sepal` `` or `` `sepal(...)` ``.
_BACKTICK_RE = re.compile(r"`([A-Za-z_]\w*)(?:\([^`]*\))?`")


#: Where ground-truth scripts live beside a ``tasks.yaml``. ``scripts/<id>.py`` is the layout this
#: experiment's recorded corpus uses; ``tasks/<id>-<split>.py`` is what ``acumen tasks`` writes now
#: (one reproducer per split). Both are read, so a corpus built either way is measured.
SCRIPT_DIRS = ("scripts", "tasks")
_SPLIT_SUFFIXES = ("-train", "-valid", "-test")


def load_task_scripts(tasks_path: Path) -> dict[str, str]:
    """Load every ground-truth script beside ``tasks_path`` into ``{task_id: source}``.

    Reads each directory in :data:`SCRIPT_DIRS` that exists. A per-split script
    (``<id>-train.py``) is filed under ``<id>`` and its source joined onto any other split's, so a
    task counts as covered by everything its reproducers call, however the corpus is laid out.
    """
    merged: dict[str, str] = {}
    for dirname in SCRIPT_DIRS:
        for stem, text in load_scripts(tasks_path.parent / dirname).items():
            task_id = next((stem[: -len(sfx)] for sfx in _SPLIT_SUFFIXES if stem.endswith(sfx)), stem)
            merged[task_id] = f"{merged[task_id]}\n{text}" if task_id in merged else text
    return merged


def load_scripts(scripts_dir: Path) -> dict[str, str]:
    """Load ``scripts/<task_id>.py`` from a directory into ``{task_id: source}``.

    The task id is the file stem, so this pairs with how task generation persists the confirmation
    scripts. A missing directory yields an empty mapping (nothing covered), not an error.
    """
    if not scripts_dir.is_dir():
        return {}
    return {p.stem: p.read_text() for p in sorted(scripts_dir.glob("*.py"))}
