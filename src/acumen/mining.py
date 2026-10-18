"""Task mining: harvest real analyses that use the target package into candidate scripts.

The benchmark's task pool should be what practitioners *actually do* with the package — not only
what its own tutorials demonstrate. This module finds such analyses in the wild and turns each into
a **candidate**: one Python file (a script, or a notebook flattened to its code cells) with a
provenance header, written under a ``mined/`` directory. Task generation then shards over those
candidates exactly as it shards over tutorial notebooks (``acumen tasks --candidates mined/``), with
a prompt that tells the agent to re-ground each analysis on a bundled dataset, since the mined
script's own data is not available.

Two sources:

* **GitHub code search** (``gh search code``), the bulk of the supply: queries for the package's
  import and its analysis namespaces, over ``.py`` files and ``.ipynb`` notebooks. Each hit is
  fetched at the exact commit the search returned (``gh api …/contents?ref=<sha>``), so a candidate
  is reproducible and its origin URL is pinned.
* **Given web pages** (``--url``): a tutorial page, a rendered notebook, a raw ``.py``/``.ipynb``/
  ``.md`` — code blocks are lifted out of the HTML or the fences. There is no keyless web *search*
  API worth depending on, so the web leg takes URLs the operator already knows rather than
  pretending to discover them.

The quality gate is **structural, not a prompt**. A file is kept only if it (1) parses, (2)
references at least one symbol of the package's real public API (the same inventory ``acumen
coverage`` uses), and (3) makes at least one package call at *module level* — a library that wraps
the package calls it only inside ``def``s and is not an analysis, however many symbols it mentions.
The target's own repository, test directories, and private-looking files are excluded outright, and
exact-duplicate content (the same tutorial vendored into several repos) collapses to one candidate.

Licensing note: only the mined text is used, locally, to *derive* a task's goal and answer; the
candidate file itself is not shipped anywhere, and ``index.json`` records every origin.

GitHub's code-search endpoint allows ten queries a minute, so the harvest is paced; the contents
endpoint is generous (5000/h) and is not.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from acumen.coverage import attr_chain, collect_aliases, resolve_parts, scan_references
from acumen.paths import slugify

#: Seconds between code-search queries: GitHub allows 10/min on that endpoint.
SEARCH_INTERVAL_S = 6.5

#: Candidates bigger than this are not "inline-light" analyses and would strain a shard agent.
MAX_CANDIDATE_CHARS = 200_000

#: Path components that mark a file as tests, vendored, or generated — never an analysis.
_SKIP_PARTS = frozenset({"test", "tests", "testing", "site-packages", "__pycache__", "node_modules", "build", "dist"})

#: Conventional import aliases, so the default queries match how people actually write the calls.
_KNOWN_ALIASES = {"squidpy": "sq", "scanpy": "sc", "anndata": "ad", "spatialdata": "sd", "muon": "mu"}

_BLOB_SHA_RE = re.compile(r"/blob/([0-9a-f]{40})/")
_FENCE_RE = re.compile(r"```(?:python|py|ipython3?)?\s*\n(.*?)```", re.DOTALL)

#: The provenance header every candidate file starts with; ``index.json`` holds the same in full.
HEADER_PREFIX = "# mined from: "

INDEX_FILE = "index.json"


class MiningError(RuntimeError):
    """Raised when a source cannot be queried at all (as opposed to one hit being rejected)."""


@dataclass(frozen=True)
class Candidate:
    """One real-world analysis, as a runnable-looking Python source with its origin pinned."""

    slug: str
    source: str  # "github" | "web"
    origin: str  # blob URL pinned to a commit, or the page URL
    kind: str  # "py" | "ipynb" | "page"
    text: str
    symbols: tuple[str, ...]
    repo: str | None = None
    path: str | None = None
    sha: str | None = None

    def meta(self) -> dict[str, object]:
        """The index entry: everything but the text."""
        data = asdict(self)
        del data["text"]
        data["chars"] = len(self.text)
        return data


@dataclass(frozen=True)
class Rejection:
    """Why a hit was not kept — surfaced so the gate's behaviour is inspectable, not silent."""

    origin: str
    reason: str


@dataclass
class MineResult:
    """The harvest: what was kept, what was rejected and why, and the queries that ran."""

    kept: list[Candidate] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def reasons(self) -> dict[str, int]:
        """Rejection reasons histogram, most common first."""
        counts: dict[str, int] = {}
        for r in self.rejected:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ── Source text handling ────────────────────────────────────────────────────────────────


def notebook_to_source(text: str) -> str:
    """Flatten a ``.ipynb`` to its code cells, in order, as one Python source.

    Cells are separated by ``# %%`` markers so cell boundaries survive. IPython-only lines
    (``%magic``, ``!shell``) are commented out rather than dropped: they are not Python, but they
    document what the author ran (installs, timing) and must not break parsing.
    """
    try:
        nb = json.loads(text)
    except ValueError as err:
        raise MiningError(f"not a notebook: {err}") from err
    cells = nb.get("cells") or []
    parts: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        body = "".join(src) if isinstance(src, list) else str(src)
        lines = [(f"# {line}" if line.lstrip().startswith(("%", "!")) else line) for line in body.splitlines()]
        parts.append("# %%\n" + "\n".join(lines))
    return "\n\n".join(parts) + ("\n" if parts else "")


def toplevel_package_calls(source: str, pkg_name: str) -> set[str]:
    """Package-qualified names called at *module level* — the mark of an analysis, not a library.

    Walks only the module body and the bodies of top-level compound statements (``if``, ``for``,
    ``with``, ``try``), never descending into ``def``/``class``: a wrapper library calls the package
    exclusively inside functions and should come back empty here.

    Raises
    ------
    SyntaxError
        If ``source`` does not parse.
    """
    tree = ast.parse(source)
    aliases = collect_aliases(tree, pkg_name)
    found: set[str] = set()

    def visit(stmts: Iterable[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                    continue  # ast.walk still descends, but nested defs are rare at top level
                if isinstance(node, ast.Call):
                    parts = attr_chain(node.func)
                    resolved = resolve_parts(parts, aliases) if parts else None
                    if resolved is not None:
                        found.add(resolved)

    visit(tree.body)
    return found


def assess(
    text: str, pkg_name: str, inventory: frozenset[str] | None, *, min_symbols: int = 1
) -> tuple[str, ...] | str:
    """Apply the structural gate to one candidate's text.

    Returns the package symbols it references (sorted) when it passes, or a rejection reason.
    ``inventory`` restricts "symbol" to the package's real public API when given (so a script that
    only touches private helpers or misspelled names does not count as coverage).
    """
    if len(text) > MAX_CANDIDATE_CHARS:
        return f"too large (> {MAX_CANDIDATE_CHARS} chars)"
    try:
        refs = scan_references(text, pkg_name)
        calls = toplevel_package_calls(text, pkg_name)
    except SyntaxError:
        return "does not parse as Python"
    symbols = refs & inventory if inventory is not None else refs
    if len(symbols) < min_symbols:
        return "references no public API symbol of the package"
    if not calls:
        return "no top-level package call (library code, not an analysis)"
    return tuple(sorted(symbols))


def _content_key(text: str) -> str:
    return hashlib.sha256("".join(text.split()).encode()).hexdigest()


# ── GitHub ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Hit:
    """One code-search result."""

    repo: str
    path: str
    url: str

    @property
    def sha(self) -> str | None:
        """The commit the search result is pinned to, parsed from its blob URL."""
        match = _BLOB_SHA_RE.search(self.url)
        return match.group(1) if match else None


def _run_gh(args: Sequence[str]) -> str:
    """Run a ``gh`` command and return stdout; ``MiningError`` on failure."""
    try:
        proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    except FileNotFoundError as err:
        raise MiningError(
            "mining searches GitHub through the `gh` CLI, which is not on PATH — install it "
            "(https://cli.github.com) and run `gh auth login`, or pass --no-github with --url pages"
        ) from err
    if proc.returncode != 0:
        raise MiningError(f"gh {' '.join(args[:2])} failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
    return proc.stdout


Runner = Callable[[Sequence[str]], str]


def gh_search(
    query: str, *, limit: int, extension: str | None = None, language: str | None = None, run: Runner = _run_gh
) -> list[Hit]:
    """One ``gh search code`` query, parsed. ``gh`` paginates up to ``limit`` itself."""
    args = ["search", "code", query, "--json", "path,repository,url", "--limit", str(limit)]
    if extension:
        args += ["--extension", extension]
    if language:
        args += ["--language", language]
    try:
        rows = json.loads(run(args) or "[]")
    except ValueError as err:
        raise MiningError(f"gh search returned non-JSON: {err}") from err
    return [Hit(repo=r["repository"]["nameWithOwner"], path=r["path"], url=r["url"]) for r in rows]


def gh_raw(repo: str, path: str, sha: str | None, *, run: Runner = _run_gh) -> str:
    """Fetch a file's raw content at a pinned commit via the contents API (uses ``gh`` auth)."""
    ref = f"?ref={sha}" if sha else ""
    return run(
        ["api", f"repos/{repo}/contents/{urllib.parse.quote(path)}{ref}", "-H", "Accept: application/vnd.github.raw"]
    )


def default_queries(pkg_name: str, alias: str | None = None) -> list[str]:
    """The query set that finds real usage: the import, and each analysis namespace by alias.

    Namespaces are scverse convention (``gr``/``im``/``pl``/``tl``); for a package laid out
    differently pass explicit ``--query`` terms instead.
    """
    alias = alias or _KNOWN_ALIASES.get(pkg_name, pkg_name)
    return [f"import {pkg_name}", *(f"{alias}.{ns}." for ns in ("gr", "im", "pl", "tl")), f"{pkg_name}.datasets"]


def _repo_id(repo_url: str) -> str | None:
    """``owner/name`` from a GitHub URL, or ``None`` for a non-GitHub/local target."""
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?/?$", repo_url)
    return match.group(1) if match else None


def submodule_repos(src_dir: Path) -> list[str]:
    """``owner/name`` of every GitHub submodule the target checkout declares.

    A package's tutorials routinely live in a submodule (squidpy → ``scverse/squidpy-tutorials``);
    those notebooks are already sharded directly, so mining them again would only duplicate the
    tutorial corpus under a different origin. Read from ``.gitmodules`` so the exclusion follows
    the checkout rather than a hand-kept list.
    """
    path = src_dir / ".gitmodules"
    if not path.is_file():
        return []
    repos: list[str] = []
    for line in path.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip() == "url":
            repo = _repo_id(value.strip())
            if repo:
                repos.append(repo)
    return repos


def _skip_path(path: str) -> str | None:
    parts = Path(path).parts
    if any(p.lower() in _SKIP_PARTS for p in parts[:-1]):
        return "under a tests/vendored/build directory"
    if parts and parts[-1].startswith("_"):
        return "private-looking filename"
    return None


def mine_github(
    pkg_name: str,
    queries: Sequence[str],
    *,
    exclude_repos: Sequence[str] = (),
    limit: int,
    inventory: frozenset[str] | None,
    min_symbols: int = 1,
    run: Runner = _run_gh,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] | None = None,
) -> MineResult:
    """Harvest candidates from GitHub code search: every query, over ``.py`` and ``.ipynb``.

    Hits are deduplicated by (repo, path) before fetching, and candidates by content after.
    ``exclude_repos`` (``owner/name``, case-insensitive) are skipped outright — the package's own
    repository and its tutorial submodules, whose notebooks are already sharded directly and whose
    source is not an analysis. Forks of it slip through only if they hold genuinely different
    files, which is fine.
    """
    excluded = {r.lower() for r in exclude_repos}
    result = MineResult()
    seen_hits: set[tuple[str, str]] = set()
    hits: list[tuple[Hit, str]] = []
    first = True
    for query in queries:
        for extension, language in (("ipynb", None), (None, "python")):
            if not first:
                sleep(SEARCH_INTERVAL_S)
            first = False
            label = f"{query!r} ({extension or language})"
            result.queries.append(label)
            if on_progress:
                on_progress(f"searching {label} …")
            try:
                found = gh_search(query, limit=limit, extension=extension, language=language, run=run)
            except MiningError as err:
                result.rejected.append(Rejection(origin=label, reason=f"search failed: {err}"))
                continue
            for hit in found:
                key = (hit.repo, hit.path)
                if key in seen_hits:
                    continue
                seen_hits.add(key)
                hits.append((hit, "ipynb" if hit.path.endswith(".ipynb") else "py"))

    seen_content: set[str] = set()
    for hit, kind in hits:
        if hit.repo.lower() in excluded:
            result.rejected.append(Rejection(hit.url, "the target package's own repository (or a submodule of it)"))
            continue
        why = _skip_path(hit.path)
        if why:
            result.rejected.append(Rejection(hit.url, why))
            continue
        try:
            raw = gh_raw(hit.repo, hit.path, hit.sha, run=run)
            text = notebook_to_source(raw) if kind == "ipynb" else raw
        except MiningError as err:
            result.rejected.append(Rejection(hit.url, f"fetch failed: {err}"))
            continue
        verdict = assess(text, pkg_name, inventory, min_symbols=min_symbols)
        if isinstance(verdict, str):
            result.rejected.append(Rejection(hit.url, verdict))
            continue
        key = _content_key(text)
        if key in seen_content:
            result.rejected.append(Rejection(hit.url, "duplicate content"))
            continue
        seen_content.add(key)
        stem = Path(hit.path).with_suffix("").as_posix()
        result.kept.append(
            Candidate(
                slug=slugify(f"{hit.repo}-{stem}".replace("/", "-"))[:100],
                source="github",
                origin=hit.url,
                kind=kind,
                text=text,
                symbols=verdict,
                repo=hit.repo,
                path=hit.path,
                sha=hit.sha,
            )
        )
    return result


# ── Web pages ───────────────────────────────────────────────────────────────────────────


class _PreExtractor(HTMLParser):
    """Collect the text of every ``<pre>`` block — how rendered notebooks and docs show code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            self._depth += 1
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._depth:
            self._depth -= 1
            self.blocks.append("".join(self._buf))

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buf.append(data)


def extract_code_from_page(text: str, *, url: str = "") -> str:
    """Lift Python out of a fetched document, by what the URL says it is.

    ``.ipynb`` → its code cells; ``.py`` → itself; ``.md`` → fenced python blocks; anything else is
    treated as HTML and every ``<pre>`` block is taken. Blocks that do not parse on their own are
    dropped (prose, shell transcripts, output cells), and a leading ``>>>`` prompt is stripped.
    """
    lower = url.lower().split("?", 1)[0]
    if lower.endswith(".ipynb"):
        return notebook_to_source(text)
    if lower.endswith(".py"):
        return text
    if lower.endswith((".md", ".markdown", ".rst")):
        blocks = _FENCE_RE.findall(text)
    else:
        parser = _PreExtractor()
        parser.feed(text)
        blocks = parser.blocks
    kept: list[str] = []
    for block in blocks:
        code = "\n".join(line[4:] if line.startswith(">>> ") else line for line in block.splitlines())
        code = "\n".join(line for line in code.splitlines() if not line.startswith("... "))
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        if code.strip():
            kept.append(code.rstrip() + "\n")
    return "\n# %%\n".join(kept)


def fetch_url(url: str) -> str:
    """GET a URL as text (identifying as acumen), for the web leg."""
    req = urllib.request.Request(url, headers={"User-Agent": "acumen-mine/0.1 (+https://github.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as err:
        raise MiningError(f"cannot fetch {url}: {err}") from err


def mine_urls(
    pkg_name: str,
    urls: Sequence[str],
    *,
    inventory: frozenset[str] | None,
    min_symbols: int = 1,
    fetch: Callable[[str], str] = fetch_url,
) -> MineResult:
    """Harvest candidates from operator-given pages; one candidate per page at most."""
    result = MineResult()
    seen_content: set[str] = set()
    for url in urls:
        result.queries.append(url)
        try:
            text = extract_code_from_page(fetch(url), url=url)
        except MiningError as err:
            result.rejected.append(Rejection(url, str(err)))
            continue
        if not text.strip():
            result.rejected.append(Rejection(url, "no Python code blocks found"))
            continue
        verdict = assess(text, pkg_name, inventory, min_symbols=min_symbols)
        if isinstance(verdict, str):
            result.rejected.append(Rejection(url, verdict))
            continue
        key = _content_key(text)
        if key in seen_content:
            result.rejected.append(Rejection(url, "duplicate content"))
            continue
        seen_content.add(key)
        parsed = urllib.parse.urlparse(url)
        result.kept.append(
            Candidate(
                slug=slugify(f"{parsed.netloc}-{parsed.path}".replace("/", "-"))[:100],
                source="web",
                origin=url,
                kind="page",
                text=text,
                symbols=verdict,
            )
        )
    return result


# ── Output ──────────────────────────────────────────────────────────────────────────────


def write_candidates(results: Sequence[MineResult], out_dir: Path) -> tuple[list[Path], list[Rejection]]:
    """Write ``<slug>.py`` per kept candidate (with a provenance header) and ``index.json``.

    Idempotent and additive: a slug already on disk is left as it is (so a re-run after more
    searching extends the pool instead of rewriting it), and the index is rebuilt from every
    candidate present. Two different origins that slugify identically get a numeric suffix.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    entries: list[dict[str, object]] = []
    rejected: list[Rejection] = []
    taken = {p.stem for p in out_dir.glob("*.py")}
    for result in results:
        rejected.extend(result.rejected)
        for cand in result.kept:
            slug = cand.slug
            n = 2
            while slug in taken:
                existing = out_dir / f"{slug}.py"
                if existing.is_file() and existing.read_text().startswith(f"{HEADER_PREFIX}{cand.origin}\n"):
                    break  # the same origin, already written
                slug = f"{cand.slug}-{n}"
                n += 1
            path = out_dir / f"{slug}.py"
            if not path.exists():
                path.write_text(f"{HEADER_PREFIX}{cand.origin}\n# symbols: {', '.join(cand.symbols)}\n\n{cand.text}")
                written.append(path)
            taken.add(slug)
            entries.append({**cand.meta(), "slug": slug, "file": path.name})
    index = {"candidates": entries, "rejected": [asdict(r) for r in rejected]}
    (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")
    return written, rejected
