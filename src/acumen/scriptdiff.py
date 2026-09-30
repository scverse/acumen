"""Deterministic pass/fail script discriminators for the wiki.

Given the ``script.py`` files of passing and failing benchmark runs of ONE task+arm, extract — with
the standard library ``ast`` only — the calls and keyword arguments that *separate* the two groups:
what passers do that failers do not, what failers do that passers do not, and (falling out of the
same comparison) where the two groups call the same function with different argument values.

This is the deterministic evidence the wiki agent otherwise has to reconstruct by eye from
transcripts under a tight brevity budget — which is how a decisive parameter such as
``dc.pp.filter_by_prop(min_prop=0.1)`` gets missed. The block this module produces is short, ranked
(strongest separators first) and hard-capped; calls common to both groups are omitted entirely, so
its size tracks the number of *discriminating* facts, not the length of the scripts.

The signal is **correlational, not causal**: two runs can differ on a call that has nothing to do
with why one passed. The rendered block says so, and the wiki agent is told to confirm against the
transcripts and source before writing its hypothesis.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

#: Default ceiling on facts listed per direction (passer-leaning, failer-leaning).
MAX_FACTS = 15

#: Default hard ceiling on the rendered block, in characters. The block is a wiki file that the
#: improver re-reads every epoch, so it must stay small regardless of how many facts separate.
MAX_CHARS = 1500

#: An atom must lean at least this strongly toward one group (|pass_frac - fail_frac|) to be worth
#: reporting; below it the difference is noise from model-to-model variation.
MIN_SEPARATION = 0.5

#: Longest a single atom string may be before it is truncated, so one giant kwarg value cannot
#: blow the budget on its own.
_MAX_ATOM_CHARS = 72


def _dotted_name(node: ast.expr) -> str | None:
    """Return the dotted name of a call target (``dc.pp.filter_by_expr``), or ``None``.

    Handles ``Name`` and chained ``Attribute`` nodes; anything else (a call result, a subscript)
    has no stable name to key on and is skipped.
    """
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _value_repr(node: ast.expr) -> str:
    """Render a keyword-argument value compactly and deterministically.

    A literal is rendered via :func:`ast.literal_eval` + :func:`repr` so ``0.1`` and ``1e-1`` agree;
    anything else falls back to :func:`ast.unparse` so a dynamic value (``"~" + "+".join(...)``)
    still reads as itself. The result is truncated so one long value cannot dominate the block.
    """
    try:
        text = repr(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        try:
            text = ast.unparse(node)
        except Exception:  # noqa: BLE001 - unparse should not fail, but never let it break analysis
            text = "<expr>"
    text = " ".join(text.split())
    if len(text) > _MAX_ATOM_CHARS:
        text = text[: _MAX_ATOM_CHARS - 1] + "…"
    return text


def extract_atoms(source: str) -> set[str] | None:
    """Return the set of call/kwarg atoms in one script, or ``None`` if it does not parse.

    Each :class:`ast.Call` contributes its dotted function name (``dc.op.collectri``) and one atom
    per keyword argument (``dc.pp.filter_by_prop(min_prop=0.1)``). Presence within a run is all that
    matters, so the result is a set: a call made twice counts once. Positional arguments are not
    keyed on — their values are rarely nameable and add noise.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    atoms: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name is None:
            continue
        atoms.add(name)
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat — nothing to name
                continue
            atoms.add(f"{name}({kw.arg}={_value_repr(kw.value)})")
    return atoms


def _read_atoms(scripts: Sequence[Path]) -> tuple[list[set[str]], int]:
    """Read and parse each script, returning per-run atom sets and the unparseable count.

    A missing or unparseable script is skipped (not an empty set), so it neither invents a
    discriminator nor dilutes one; the count is surfaced in the rendered block.
    """
    per_run: list[set[str]] = []
    skipped = 0
    for path in scripts:
        try:
            source = path.read_text()
        except OSError:
            skipped += 1
            continue
        atoms = extract_atoms(source)
        if atoms is None:
            skipped += 1
            continue
        per_run.append(atoms)
    return per_run, skipped


def _counts(per_run: Sequence[set[str]]) -> dict[str, int]:
    """Count how many runs each atom appears in."""
    counts: dict[str, int] = {}
    for atoms in per_run:
        for atom in atoms:
            counts[atom] = counts.get(atom, 0) + 1
    return counts


def analyze_scripts(
    passers: Sequence[Path],
    failers: Sequence[Path],
    *,
    version: str,
    task_id: str,
    max_facts: int = MAX_FACTS,
    max_chars: int = MAX_CHARS,
) -> str | None:
    """Render the discriminator block for one task+arm, or ``None`` when there is no signal.

    Returns ``None`` unless there is at least one parseable passing script and one parseable failing
    script — with only one side, nothing *discriminates*. Otherwise returns a markdown block headed
    ``[version]`` listing the atoms that lean toward passers and those that lean toward failers, each
    with its ``pass``/``fail`` run counts, strongest separation first, capped at ``max_facts`` per
    direction and ``max_chars`` overall.

    Parameters
    ----------
    passers, failers
        The ``script.py`` paths of the passing and failing runs.
    version
        The arm label for the block heading: ``"noskill"``/``"v1"``/``"v2"``…
    task_id
        The task, for the heading.
    max_facts
        Ceiling on facts listed per direction.
    max_chars
        Hard ceiling on the returned block.
    """
    pass_runs, pass_skipped = _read_atoms(passers)
    fail_runs, fail_skipped = _read_atoms(failers)
    if not pass_runs or not fail_runs:
        return None

    n_pass, n_fail = len(pass_runs), len(fail_runs)
    pass_counts = _counts(pass_runs)
    fail_counts = _counts(fail_runs)

    scored: list[tuple[float, str, int, int]] = []
    for atom in set(pass_counts) | set(fail_counts):
        pc = pass_counts.get(atom, 0)
        fc = fail_counts.get(atom, 0)
        separation = pc / n_pass - fc / n_fail
        if abs(separation) >= MIN_SEPARATION:
            scored.append((separation, atom, pc, fc))

    # Passer-leaning: strongest positive separation first, then most passers. Failer-leaning mirrors.
    passer_facts = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], -s[2]))[:max_facts]
    failer_facts = sorted((s for s in scored if s[0] < 0), key=lambda s: (s[0], -s[3]))[:max_facts]
    if not passer_facts and not failer_facts:
        return None

    lines = [
        f"## [{version}] — {task_id} ({n_pass} pass / {n_fail} fail train runs with scripts)",
        "",
        "_Mechanical AST diff of `script.py` calls & kwargs. CORRELATIONAL, not causal — confirm "
        "against the transcripts before concluding._",
        "",
    ]

    def render(title: str, facts: list[tuple[float, str, int, int]]) -> None:
        if not facts:
            return
        lines.append(f"**{title}:**")
        for _sep, atom, pc, fc in facts:
            lines.append(f"- `{atom}` — {pc}/{n_pass} pass, {fc}/{n_fail} fail")
        lines.append("")

    render("In passers, rare/absent in failers", passer_facts)
    render("In failers, rare/absent in passers", failer_facts)

    skipped = pass_skipped + fail_skipped
    if skipped:
        noun = "script" if skipped == 1 else "scripts"
        lines.append(f"_{skipped} {noun} could not be read or parsed and were skipped._")

    block = "\n".join(lines).rstrip() + "\n"
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + "\n…(truncated)\n"
    return block
