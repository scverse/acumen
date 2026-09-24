"""A side-by-side (split) HTML diff renderer, shared across acumen's HTML outputs.

The report diffs each skill version against its parent; the transcript diffs a file edit's before
against its after. Both want the same thing — old on the left, new on the right, changed words
lit up — so the renderer lives here and neither page owns it. It is pure stdlib
(``difflib``/``re``/``html``) so the transcript can use it without importing the report (and its
matplotlib/pandas weight). The styling lives in :data:`acumen.theme.DIFF_CSS`; this module emits
only class names.
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass

#: Unchanged lines kept on either side of a change, as in ``diff -u``. Longer untouched runs
#: collapse to a single marker row, so a change reads as only what actually moved.
_DIFF_CONTEXT = 3

#: Below this line-similarity the two sides of a replaced row are treated as unrelated text,
#: so the row is tinted whole instead of being picked apart into confetti-sized highlights.
_DIFF_INLINE_RATIO = 0.5


@dataclass(frozen=True)
class _DiffRow:
    """One row of a split diff: the same logical line on each side, either side possibly absent.

    ``kind`` is the change type from :class:`difflib.SequenceMatcher` (``equal``, ``replace``,
    ``delete``, ``insert``), plus ``gap`` for the marker standing in for an elided run of
    unchanged lines. A side is ``None`` where that version has no line there at all — a pure
    insertion has no left-hand text — and renders as an inert filler cell.
    """

    kind: str
    left_no: int | None
    left: str | None
    right_no: int | None
    right: str | None


def _equal_rows(before: list[str], after: list[str], i: int, j: int, count: int) -> list[_DiffRow]:
    """``count`` rows of unchanged text, starting at line ``i`` on the left and ``j`` on the right."""
    return [_DiffRow("equal", i + k + 1, before[i + k], j + k + 1, after[j + k]) for k in range(count)]


def _split_diff_rows(before: list[str], after: list[str]) -> list[_DiffRow]:
    """Align two versions of a file into side-by-side rows, old on the left, new on the right.

    Changed runs pair off line by line, so a rewritten paragraph sits opposite its replacement
    rather than being stacked below it; where one side runs out, the other continues against
    filler. Unchanged stretches beyond :data:`_DIFF_CONTEXT` lines from any change collapse to
    a gap row.
    """
    rows: list[_DiffRow] = []
    opcodes = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            head = _DIFF_CONTEXT if index > 0 else 0
            tail = _DIFF_CONTEXT if index < len(opcodes) - 1 else 0
            if i2 - i1 > head + tail:
                rows += _equal_rows(before, after, i1, j1, head)
                rows.append(_DiffRow("gap", None, None, None, None))
                rows += _equal_rows(before, after, i2 - tail, j2 - tail, tail)
            else:
                rows += _equal_rows(before, after, i1, j1, i2 - i1)
            continue
        left, right = before[i1:i2], after[j1:j2]
        for k in range(max(len(left), len(right))):
            has_left, has_right = k < len(left), k < len(right)
            rows.append(
                _DiffRow(
                    tag,
                    i1 + k + 1 if has_left else None,
                    left[k] if has_left else None,
                    j1 + k + 1 if has_right else None,
                    right[k] if has_right else None,
                )
            )
    return rows


def _inline_pair(left: str, right: str) -> tuple[str, str]:
    """Both sides of a replaced line, escaped, with the words that differ wrapped in ``<mark>``.

    The comparison runs over words rather than characters, so a changed word lights up whole
    instead of down to the letters it happens to share with its replacement. Lines too
    dissimilar to be a rewrite of one another are left unmarked — highlighting nearly every
    word says less than the row tint already does.
    """
    a, b = re.findall(r"\w+|\W", left), re.findall(r"\w+|\W", right)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    if matcher.ratio() < _DIFF_INLINE_RATIO:
        return html.escape(left), html.escape(right)
    marked = ["", ""]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for side, tokens, start, end in ((0, a, i1, i2), (1, b, j1, j2)):
            chunk = html.escape("".join(tokens[start:end]))
            if chunk:
                marked[side] += chunk if tag == "equal" else f"<mark>{chunk}</mark>"
    return marked[0], marked[1]


def _diff_cells(number: int | None, text: str | None, marked: str | None, cls: str) -> str:
    """The line-number and content cell for one side of a row, or filler where that side is empty."""
    if text is None:
        return '<td class="ln"></td><td class="diff-none"></td>'
    body = marked if marked is not None else html.escape(text)
    return f'<td class="ln">{number}</td><td class="{cls}">{body or "&nbsp;"}</td>'


def split_diff_table(rel: str, before: list[str], after: list[str], labels: tuple[str, str]) -> str:
    """One file's split diff as a table: line numbers and text for each version, side by side."""
    body: list[str] = []
    for row in _split_diff_rows(before, after):
        if row.kind == "gap":
            body.append('<tr class="diff-gap"><td colspan="4">&hellip;</td></tr>')
            continue
        left_mark = right_mark = None
        if row.kind == "replace" and row.left is not None and row.right is not None:
            left_mark, right_mark = _inline_pair(row.left, row.right)
        left_cls = "diff-ctx" if row.kind == "equal" else "diff-del"
        right_cls = "diff-ctx" if row.kind == "equal" else "diff-add"
        body.append(
            "<tr>"
            + _diff_cells(row.left_no, row.left, left_mark, left_cls)
            + _diff_cells(row.right_no, row.right, right_mark, right_cls)
            + "</tr>"
        )
    head = (
        f'<tr class="diff-head"><td class="ln"></td><td>{html.escape(labels[0])}</td>'
        f'<td class="ln"></td><td>{html.escape(labels[1])}</td></tr>'
    )
    return (
        f'<div class="diff"><div class="diff-file">{html.escape(rel)}</div>'
        f'<table class="diff-table"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


#: Backwards-compatible alias — ``report.py`` historically named this ``_split_diff_table``.
_split_diff_table = split_diff_table
