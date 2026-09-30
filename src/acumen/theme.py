"""The shared warm-neutral palette and the CSS fragments built from it.

acumen renders two kinds of HTML — the aggregate report (:mod:`acumen.report`) and the per-run
transcript (:mod:`acumen.trajectory`) — and both should read as one product: the same ink on the
same page, and an identical red/green diff. The palette tokens and the CSS built from them live
here so neither renderer owns them and the transcript never has to import the report (which would
pull matplotlib into every per-run render).

The tokens are the maintainer's fixed scheme: the page and the plot area are white, so a figure
sits on the page with no visible frame and the ink carries the structure.
"""

from __future__ import annotations

INK = "#1c1813"  # axes, text, ticks
PAGE = "#ffffff"  # page + figure background
PLOT_BG = "#ffffff"  # the plot area itself; also the hairline between adjacent bars
SURFACE = "#f7f3ec"  # the one tinted surface — the inline diffs, so code reads as a block
BAR = "#565149"  # a neutral tone, used where model hue does not apply
ACCENT = "#b2ac9e"  # the light warm tone, for table tints and notes

#: The ``:root`` custom properties both pages expose, so a rule can reach a token as
#: ``var(--surface)`` rather than baking a hex in. ``color-scheme: light`` because the diff
#: colours below are tuned for a light page.
PALETTE_ROOT_CSS = f":root {{ color-scheme: light; --ink: {INK}; --page: {PAGE}; --surface: {SURFACE}; --bar: {BAR}; }}"

#: The split diff: old version left, new version right, one bordered box per changed file. Shared
#: verbatim so the report's skill diffs and the transcript's file edits look identical.
DIFF_CSS = f"""\
.diff {{ background: var(--surface); border: 1px solid {INK}22; border-radius: 4px;
        overflow-x: auto; margin: 0.4rem 0 1.4rem; }}
.diff-file {{ color: {INK}; font-weight: 700; padding: 0.4rem 0.8rem;
        border-bottom: 1px solid {INK}22; font-size: 0.82rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
table.diff-table {{ display: table; table-layout: fixed; width: 100%; min-width: 34rem; margin: 0;
        border-collapse: collapse; overflow-x: visible; font-size: 0.82rem; line-height: 1.35;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
table.diff-table td {{ border: 0; padding: 0 0.5rem; text-align: left; vertical-align: top;
        white-space: pre-wrap; overflow-wrap: anywhere; background: none; }}
table.diff-table td.ln {{ width: 2.8rem; text-align: right; user-select: none;
        color: {INK}77; background: {INK}0a; }}
table.diff-table tr.diff-head td {{ color: {BAR}; font-weight: 600; background: {INK}0a;
        border-bottom: 1px solid {INK}22; }}
table.diff-table tr.diff-gap td {{ color: {INK}77; text-align: center; background: {INK}0a;
        border-top: 1px solid {INK}22; border-bottom: 1px solid {INK}22; }}
table.diff-table td.diff-add {{ background: #4c7a3322; color: #2f5d1c; }}
table.diff-table td.diff-del {{ background: #a4432b22; color: #8a2f1b; }}
table.diff-table td.diff-none {{ background: {INK}0a; }}
table.diff-table td.diff-ctx {{ color: {INK}bb; }}
table.diff-table td.diff-add mark {{ background: #4c7a3355; color: inherit; }}
table.diff-table td.diff-del mark {{ background: #a4432b55; color: inherit; }}"""
