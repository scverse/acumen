"""A small, safe markdown-to-HTML renderer shared across acumen's HTML outputs.

Both the transcript renderer (:mod:`acumen.trajectory`) and the report
(:mod:`acumen.report`) turn agent- or maintainer-authored markdown into HTML, and
neither wants an external renderer dependency. The renderer here is deliberately a
common-case subset of CommonMark — paragraphs, headings, bullet/ordered lists, fenced
code, pipe tables and inline emphasis — with everything HTML-escaped first, so no input
can inject markup and anything unrecognized falls through as plain text.
"""

from __future__ import annotations

import re
from html import escape

#: How much of one fenced code block the page keeps. The full text lives in the source
#: beside the rendered HTML (e.g. the transcript JSONL).
OUTPUT_CAP = 20_000


def clip(text: str) -> str:
    """Truncate ``text`` to :data:`OUTPUT_CAP`, noting how much was dropped."""
    if len(text) <= OUTPUT_CAP:
        return text
    return text[:OUTPUT_CAP] + f"\n… {len(text) - OUTPUT_CAP} more characters"


def _md_inline(text: str) -> str:
    """Render inline markdown (code, bold, italic, links) on one line, escaping HTML first."""
    codes: list[str] = []

    def stash(match: re.Match[str]) -> str:
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{escape(codes[int(m.group(1))])}</code>", text)


def _table_cells(line: str) -> list[str]:
    """Split one pipe-table row into cells, dropping the optional outer pipes."""
    stripped = line.strip()
    stripped = stripped.removeprefix("|").removesuffix("|")
    return stripped.split("|")


def _is_table_delimiter(line: str) -> bool:
    """Whether ``line`` is a GFM table delimiter row (``| --- | :--: |``)."""
    cells = _table_cells(line)
    return "|" in line and bool(cells) and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells)


def _table_align(spec: str) -> str:
    spec = spec.strip()
    left, right = spec.startswith(":"), spec.endswith(":")
    if left and right:
        return "center"
    if right:
        return "right"
    return "left" if left else ""


def _render_table(header: list[str], aligns: list[str], rows: list[list[str]]) -> str:
    def cell(tag: str, text: str, index: int) -> str:
        align = aligns[index] if index < len(aligns) else ""
        style = f' style="text-align:{align}"' if align else ""
        return f"<{tag}{style}>{_md_inline(text.strip())}</{tag}>"

    head = "<tr>" + "".join(cell("th", value, i) for i, value in enumerate(header)) + "</tr>"
    body = "".join("<tr>" + "".join(cell("td", value, i) for i, value in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def render_markdown(text: str) -> str:
    """A small, safe markdown-to-HTML renderer — headings, lists, code, tables, inline.

    Deliberately a common-case subset (not full CommonMark): the input is paragraphs, bullet
    lists, fenced code and inline emphasis. Everything is HTML-escaped, so no input can inject
    markup, and anything unrecognized falls through as plain text.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []
    list_tag = ""

    def flush_para() -> None:
        if para:
            out.append("<p>" + "<br>".join(_md_inline(line) for line in para) + "</p>")
            para.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if items:
            out.append(f"<{list_tag}>" + "".join(f"<li>{_md_inline(it)}</li>" for it in items) + f"</{list_tag}>")
            items.clear()
            list_tag = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            flush_list()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # skip the closing fence
            out.append(f"<pre><code>{escape(clip(chr(10).join(code)))}</code></pre>")
            continue
        if "|" in stripped and i + 1 < len(lines) and _is_table_delimiter(lines[i + 1]):
            flush_para()
            flush_list()
            header = _table_cells(line)
            aligns = [_table_align(spec) for spec in _table_cells(lines[i + 1])]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(_table_cells(lines[i]))
                i += 1
            out.append(_render_table(header, aligns, rows))
            continue
        if not stripped:
            flush_para()
            flush_list()
        elif heading := re.match(r"(#{1,6})\s+(.*)", stripped):
            flush_para()
            flush_list()
            level = min(len(heading.group(1)) + 2, 6)  # start at h3 so content never out-shouts the page
            out.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
        elif bullet := re.match(r"[-*+]\s+(.*)", stripped):
            flush_para()
            if list_tag and list_tag != "ul":
                flush_list()
            list_tag = "ul"
            items.append(bullet.group(1))
        elif ordered := re.match(r"\d+[.)]\s+(.*)", stripped):
            flush_para()
            if list_tag and list_tag != "ol":
                flush_list()
            list_tag = "ol"
            items.append(ordered.group(1))
        else:
            flush_list()
            para.append(stripped)
        i += 1
    flush_para()
    flush_list()
    return "".join(out)
