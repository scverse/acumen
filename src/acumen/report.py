"""Aggregate ``result.json`` files into a self-contained HTML report.

The report reads **only** ``result.json`` — never the transcripts. Each result is
the unit of record, so aggregation is just a walk of the run tree, a load per leaf, and a
few group-bys. Figures are matplotlib rendered to PNG and inlined as base64 data URIs, so
``report.html`` opens standalone with no sidecar files and no new dependencies.

The report is regenerated (overwritten) at each skill version — it always reflects every
run currently on disk across every arm.

The visualisations show **test-split** performance — the held-out measure of whether a
skill helps. The full per-run table below them still lists both splits, so train runs
remain inspectable.
"""

from __future__ import annotations

import base64
import difflib
import html
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we never open a window, we render straight to PNG bytes

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.patches import Patch

from acumen.paths import NOSKILL_ARM, RESULT_FILE, TRANSCRIPT_HTML, skill_from_arm
from acumen.skills import SkillError, read_meta, skill_content, skill_dir
from acumen.tasks import Task

# ── Palette ──────────────────────────────────────────────────────────────────────────
# A warm-neutral scheme fixed by the maintainer. The page and the plot area are both white,
# so a figure sits on the page with no visible frame around it and the ink carries the
# structure. Bars are coloured by model (see the sequential ramp below); train/test is a
# texture, not a colour.
INK = "#1c1813"  # axes, text, ticks
PAGE = "#ffffff"  # page + figure background
PLOT_BG = "#ffffff"  # the plot area itself; also the hairline between adjacent bars
SURFACE = "#f7f3ec"  # the one tinted surface — the inline skill diffs, so code reads as a block
BAR = "#565149"  # a neutral tone, used where model hue does not apply
ACCENT = "#b2ac9e"  # the light warm tone, for table tints and notes

# The model is the hue: a single-hue ramp keyed to model "potency". The most capable model
# (Fable) is the DARKEST / most saturated — the Claude orange — and each weaker tier is a
# lighter tint of that same hue. The mapping is fixed by *tier* (parsed from the model id),
# so a given tier always reads the same colour regardless of which — or how many — models a
# pass happens to include. This is only the *default*: a caller can recolour any model by id
# (see :func:`resolve_palette`), while the page and plot surfaces above stay fixed.
_MODEL_ORDER = ("fable", "opus", "sonnet", "haiku")
_MODEL_COLOR = {
    "fable": "#d86f53",  # Claude orange — most potent, darkest/most saturated
    "opus": "#e18c74",  # a lighter tint
    "sonnet": "#e8a794",  # lighter still
    "haiku": "#eec0b0",  # lightest — least potent
}
_OTHER_MODEL_COLOR = "#8a8378"  # neutral fallback for an unrecognised model id

# Train vs test is a *texture*, not a colour — so it never competes with the model hue.
_SPLIT_ORDER = ("train", "test")
_SPLIT_HATCH = {"train": "////", "test": ""}

#: rcParams applied around every figure via :func:`matplotlib.pyplot.rc_context`, so the
#: report's styling never leaks into a caller's global matplotlib state.
_RC = {
    "figure.facecolor": PAGE,
    "savefig.facecolor": PAGE,
    "axes.facecolor": PLOT_BG,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "font.size": 10,
    "legend.fontsize": 9,
}

#: The split every figure reports on. Train runs feed the improver; the report measures
#: held-out (test) performance.
_REPORTED_SPLIT = "test"


#: Brand assets bundled with the package (see ``src/acumen/assets``). Inlined as ``data:``
#: URIs so ``report.html`` stays self-contained — the favicon and the sidebar banner
#: travel inside the HTML, with no sidecar image files.
_ASSETS = Path(__file__).parent / "assets"


def _asset_data_uri(name: str) -> str:
    """Base64 ``data:`` URI for a bundled SVG asset, for inline ``src``/``href`` use."""
    encoded = base64.b64encode((_ASSETS / name).read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


class ReportError(ValueError):
    """Raised when there is nothing to report on, or the run tree is unreadable."""


def arm_label(arm: str) -> str:
    """Human label for an arm — ``"noskill"`` or a skill version like ``"v1"``.

    This is the machine-ish form used in the data table and CSV; the figures use the
    prettier :func:`_skill_label`.
    """
    return NOSKILL_ARM if arm == NOSKILL_ARM else (skill_from_arm(arm) or arm)


def _skill_label(arm: str) -> str:
    """Figure label for an arm — ``"No skill"``, ``"Skill v1"``, ``"Skill v2"``, …"""
    if arm == NOSKILL_ARM:
        return "No skill"
    version = skill_from_arm(arm)
    return f"Skill {version}" if version else arm


def _arm_sort_key(arm: str) -> int:
    """Order arms as noskill, v1, v2, … — the order a reader expects to compare in."""
    if arm == NOSKILL_ARM:
        return -1
    version = skill_from_arm(arm)
    return int(version[1:]) if version else 0


def _fmt_tokens(value: float) -> str:
    """Compact human token count, e.g. ``118k`` / ``1.2M``."""
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def _fmt_seconds(value: float) -> str:
    """Compact human duration, e.g. ``45s`` / ``2.5m`` / ``1.1h``."""
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.1f}m"
    return f"{value:.0f}s"


def load_results(runs_root: Path) -> pd.DataFrame:
    """Load every ``result.json`` under ``runs_root`` into a DataFrame.

    Parameters
    ----------
    runs_root
        The ``runs/`` root directory.

    Returns
    -------
    One row per completed run. In addition to the ``result.json`` fields, each row carries
    ``total_tokens`` and — for the runs table — ``result_path`` and ``transcript_path``
    (the sibling ``transcript.html``, whether or not it exists).

    Raises
    ------
    ReportError
        If ``runs_root`` is missing or holds no readable results.
    """
    if not runs_root.is_dir():
        raise ReportError(f"no runs directory: {runs_root}")

    rows: list[dict] = []
    for result_path in sorted(runs_root.rglob(RESULT_FILE)):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise ReportError(f"cannot read {result_path}: {err}") from err
        data["result_path"] = result_path.resolve()
        data["transcript_path"] = (result_path.parent / TRANSCRIPT_HTML).resolve()
        rows.append(data)

    if not rows:
        raise ReportError(f"no {RESULT_FILE} files under {runs_root} — run `acumen bench` first")

    df = pd.DataFrame(rows)
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
    df["arm_label"] = df["arm"].map(arm_label)
    df["_arm_order"] = df["arm"].map(_arm_sort_key)
    df = df.sort_values(["_arm_order", "split", "task_id", "model", "rep"]).reset_index(drop=True)
    return df


def _arms_in_order(df: pd.DataFrame) -> list[str]:
    """Arms present in ``df``, noskill first then ascending skill versions."""
    return sorted(df["arm"].unique(), key=_arm_sort_key)


def _rate_and_error(successes: pd.Series) -> tuple[float, float]:
    """Success rate and its binomial standard error for a group of boolean runs."""
    n = len(successes)
    if n == 0:
        return math.nan, 0.0
    p = float(successes.mean())
    return p, math.sqrt(p * (1.0 - p) / n)


def arm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm success rate and resource totals for the runs in ``df``.

    The caller filters ``df`` first — to the test split, and optionally to one task. Each
    row is one arm, in report order.

    Returns
    -------
    Columns ``arm``, ``arm_label``, ``rate``, ``stderr``, ``tokens``, ``cost``, ``time``,
    ``n``.
    """
    rows = []
    for arm in _arms_in_order(df):
        group = df[df["arm"] == arm]
        rate, stderr = _rate_and_error(group["success"])
        rows.append(
            {
                "arm": arm,
                "arm_label": arm_label(arm),
                "rate": rate,
                "stderr": stderr,
                "tokens": float(group["total_tokens"].sum()),
                "cost": float(group["cost_usd"].sum()),
                "time": float(group["duration_s"].sum()),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────────────────


#: The four metric columns: (title, key, x-axis formatter, bar-label fn). The key selects
#: the value; ``"rate"`` is special-cased for binomial error bars.
_COLUMNS = [
    ("Success rate", "rate", mticker.PercentFormatter(1.0), lambda v: f"{v:.0%}"),
    ("Total tokens", "tokens", mticker.FuncFormatter(lambda v, _p: _fmt_tokens(v)), _fmt_tokens),
    ("Total cost", "cost", mticker.FuncFormatter(lambda v, _p: f"${v:.2f}"), lambda v: f"${v:.2f}"),
    ("Total time", "time", mticker.FuncFormatter(lambda v, _p: _fmt_seconds(v)), _fmt_seconds),
]

#: Metric key → the column that is summed for it (``rate`` is handled separately).
_SUM_COLUMN = {"tokens": "total_tokens", "cost": "cost_usd", "time": "duration_s"}


def _model_tier(model: str) -> str:
    """Map a model id (e.g. ``claude-opus-5``) onto its family, or ``"other"``."""
    lower = model.lower()
    for tier in _MODEL_ORDER:
        if tier in lower:
            return tier
    return "other"


def _model_color(model: str) -> str:
    """The sequential ramp colour for a model, by tier."""
    return _MODEL_COLOR.get(_model_tier(model), _OTHER_MODEL_COLOR)


#: A trailing snapshot date on a model id, e.g. the ``-20251001`` on
#: ``claude-haiku-4-5-20251001``. Some ids pin a dated snapshot and some don't
#: (``claude-opus-5``); we strip it for display so the legend reads consistently.
_MODEL_DATE_RE = re.compile(r"-\d{6,8}$")


def _model_label(model: str) -> str:
    """Display label for a model id, with any trailing snapshot date removed.

    ``claude-haiku-4-5-20251001`` -> ``claude-haiku-4-5``; ``claude-opus-5`` is unchanged.
    The raw id is kept everywhere it is data (the runs table, the CSV) — this only tidies
    what the figures show.
    """
    return _MODEL_DATE_RE.sub("", model)


def resolve_palette(models: Sequence[str], overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Bar colour per model id — the tier defaults, with ``overrides`` merged over them.

    Parameters
    ----------
    models
        Every model id the report covers. The result has an entry for each of them, so a
        figure never has to fall back mid-draw.
    overrides
        Colours keyed by model id, e.g. ``{"claude-opus-5": "#3b7ea1"}``. A key may also be
        the display form with the snapshot date stripped (``claude-haiku-4-5`` for
        ``claude-haiku-4-5-20251001``), since that is what the legend shows. Any matplotlib
        colour spec is accepted — hex, a named colour, an ``rgb``/``rgba`` tuple.

    Raises
    ------
    ReportError
        If a colour is unparseable, or a key matches no model in ``models``. Both are
        caught up front so a typo shows as a clear message rather than a miscoloured
        figure or a matplotlib error part-way through rendering.
    """
    colors = {model: _model_color(model) for model in models}
    if not overrides:
        return colors

    unparseable = sorted(f"{key}={value!r}" for key, value in overrides.items() if not is_color_like(value))
    if unparseable:
        raise ReportError(f"palette: not a colour: {', '.join(unparseable)}")

    by_label: dict[str, list[str]] = {}
    for model in models:
        by_label.setdefault(_model_label(model), []).append(model)
    unknown = []
    for key, value in overrides.items():
        if key in colors:
            colors[key] = value
        elif key in by_label:
            for model in by_label[key]:
                colors[model] = value
        else:
            unknown.append(key)
    if unknown:
        known = ", ".join(sorted(models))
        raise ReportError(f"palette: no such model: {', '.join(sorted(unknown))} (models in these runs: {known})")
    return colors


def _models_in_order(df: pd.DataFrame) -> list[str]:
    """Models present, most-potent first (fable, opus, sonnet, haiku), then any others."""

    def key(model: str) -> tuple[int, str]:
        tier = _model_tier(model)
        return (_MODEL_ORDER.index(tier) if tier in _MODEL_ORDER else len(_MODEL_ORDER), model)

    return sorted(df["model"].unique(), key=key)


def _cell_value(df: pd.DataFrame, arm: str, model: str, split: str, key: str) -> tuple[float, float, bool]:
    """One (arm, model, split) cell's metric value, its error bar, and whether it has runs.

    For ``rate`` it is the success rate and its binomial standard error. For the resource
    totals it is the sum and the standard deviation of that sum — ``sqrt(n) * std`` over
    the per-run values, i.e. how much the total would vary run to run (0 with <2 runs).

    The third element distinguishes a genuine zero (e.g. a 0% success rate) from a cell with
    no runs at all: a model not run in this arm reserves its slot but draws no bar.
    """
    group = df[(df["arm"] == arm) & (df["model"] == model) & (df["split"] == split)]
    if group.empty:
        return 0.0, 0.0, False
    if key == "rate":
        rate, error = _rate_and_error(group["success"])
        return rate, error, True
    values = group[_SUM_COLUMN[key]]
    total = float(values.sum())
    error = float(values.std(ddof=1)) * math.sqrt(len(values)) if len(values) > 1 else 0.0
    return total, error, True


def _draw_cell(
    ax: plt.Axes,
    df: pd.DataFrame,
    arm: str,
    models: list[str],
    splits: list[str],
    key: str,
    *,
    value_label,
    colors: Mapping[str, str],
) -> None:
    """Draw one grid cell — one fixed slot per model (colour), grouped by split (texture).

    Every model in ``models`` gets a slot at the same y-position in every cell, so bars line
    up across skills. A model with no runs in this arm reserves its slot but draws nothing —
    no bar, no error bar, no label — so a stretched neighbour can't be mistaken for it.
    """
    n_models = len(models)
    n_splits = len(splits)
    group_h = 0.8
    bar_h = group_h / n_splits
    nan = float("nan")
    for si, split in enumerate(splits):
        offset = (si - (n_splits - 1) / 2) * bar_h
        cells = [_cell_value(df, arm, model, split, key) for model in models]
        # A NaN width draws no bar and no error bar; the y-slot is still occupied.
        bars = ax.barh(
            [m + offset for m in range(n_models)],
            [value if present else nan for value, _err, present in cells],
            bar_h * 0.86,
            color=[colors[model] for model in models],
            hatch=_SPLIT_HATCH[split] if n_splits > 1 else "",
            edgecolor=PLOT_BG,
            linewidth=0.6,
            xerr=[err if present else nan for _value, err, present in cells],
            capsize=2,
            error_kw={"ecolor": INK, "elinewidth": 1},
        )
        labels = [value_label(value) if present else "" for value, _err, present in cells]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=6, color=INK)
    ax.set_yticks([])
    # One slot per model, tight even padding — half a slot each side, no wasted band.
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.invert_yaxis()  # first model (most potent) on top
    ax.grid(axis="x", color=INK, alpha=0.14, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def metrics_figure(df: pd.DataFrame, *, split_hue: bool, colors: Mapping[str, str] | None = None) -> plt.Figure:
    """The metrics grid: one subplot row per skill, one column per metric.

    Columns (success rate with binomial error bars, total tokens, total cost, total time)
    share an x-axis so skills are comparable within a metric. Inside each cell, bars are
    coloured by model; when ``split_hue`` is set, train and test are shown as two textures
    (test solid, train hatched) rather than two colours, leaving the model hue intact.

    Parameters
    ----------
    df
        Results to plot — the full frame, or one task's slice.
    split_hue
        ``True`` for the per-task view (train + test as texture); ``False`` for the
        overview (the reported test split only).
    colors
        Bar colour per model id, as returned by :func:`resolve_palette`. It may cover more
        models than ``df`` holds — one map resolved over the whole report is reused for
        every per-task figure, so a model keeps its colour across them. Models missing
        from it fall back to their tier default; ``None`` uses the defaults throughout.
    """
    arms = _arms_in_order(df)
    models = _models_in_order(df)
    resolved = colors or {}
    colors = {model: resolved.get(model, _model_color(model)) for model in models}
    splits = list(_SPLIT_ORDER) if split_hue else [_REPORTED_SPLIT]
    # Every model gets a fixed slot in every arm's cell, so bars align across skills and a
    # model that wasn't run leaves a reserved gap rather than letting its neighbour stretch.
    n_rows = max(1, len(arms))
    max_bars = max(1, len(models) * len(splits))
    with plt.rc_context(_RC):
        # Height scales with the number of bars per cell; the small constant covers the
        # per-row breathing room, not a fixed block, so few-model plots stay compact.
        row_h = 0.26 * max_bars + 0.22
        fig, axes = plt.subplots(n_rows, 4, sharex="col", squeeze=False, figsize=(8.4, n_rows * row_h + 0.6))
        for c, (title, key, formatter, value_label) in enumerate(_COLUMNS):
            for r, arm in enumerate(arms):
                ax = axes[r][c]
                _draw_cell(ax, df, arm, models, splits, key, value_label=value_label, colors=colors)
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
                if r == 0:
                    ax.set_title(title, fontsize=10)
                if c == 0:
                    ax.set_ylabel(_skill_label(arm), rotation=0, ha="right", va="center", fontweight="bold")
            # sharex="col" ties the whole column together, so one xlim covers every row.
            if key == "rate":
                axes[-1][c].set_xlim(0, 1.28)
            else:
                reach = max(
                    (
                        v + e
                        for arm in arms
                        for m in models
                        for s in splits
                        for v, e, present in [_cell_value(df, arm, m, s, key)]
                        if present
                    ),
                    default=1.0,
                )
                axes[-1][c].set_xlim(0, reach * 1.35 if reach else 1)

        model_labels = [_model_label(m) for m in models]
        model_handles = [
            Patch(facecolor=colors[m], edgecolor=PLOT_BG, label=lab)
            for m, lab in zip(models, model_labels, strict=True)
        ]
        fig.legend(
            model_handles, model_labels, title="model", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 0.98)
        )
        if split_hue:
            split_handles = [Patch(facecolor="#d8d2c6", edgecolor=INK, hatch=_SPLIT_HATCH[s], label=s) for s in splits]
            fig.legend(split_handles, splits, title="split", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 0.5))
        fig.supylabel("Skills", fontweight="bold")
        fig.tight_layout()
    return fig


def figure_data_uri(fig: plt.Figure) -> str:
    """Render a figure to a base64 PNG ``data:`` URI, then close it.

    The figure background is saved **transparent** (``facecolor="none"``) so the margins
    between subplots pick up the page colour in the HTML but export blank — the axes
    interiors keep their own opaque fill. ``transparent=True`` is deliberately *not* used:
    it would also blank the axes patches.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True)
class Report:
    """A rendered report and the data behind it."""

    html: str
    results: pd.DataFrame

    @property
    def n_runs(self) -> int:
        """How many runs the report covers."""
        return len(self.results)


def _integrity_notes(df: pd.DataFrame) -> list[str]:
    """Flag arms whose ``skill_loaded`` disagrees with what the arm name claims.

    A skill arm that never loaded the skill under test is not measuring it; a noskill run
    that did load it is not a clean baseline. Either makes the comparison a lie, so it is
    surfaced in the report rather than left in the transcripts. Other skills the agent may
    reach — the ones bundled with the CLI — do not count either way.
    """
    notes = []
    if "skill_loaded" not in df.columns:
        return notes
    for arm in _arms_in_order(df):
        group = df[df["arm"] == arm]
        loaded = int(group["skill_loaded"].fillna(False).sum())
        if arm == NOSKILL_ARM:
            if loaded:
                notes.append(f"baseline arm '{arm}' loaded the skill under test in {loaded}/{len(group)} runs")
        elif loaded < len(group):
            notes.append(f"skill arm '{arm_label(arm)}' loaded the skill in only {loaded}/{len(group)} runs")
    return notes


def _loaded_cell(row: pd.Series) -> str:
    """Per-run 'skill loaded' cell: did this run load the skill under test?

    ``—`` for a baseline run (no skill to load), ``yes``/``no`` for a skill run, and ``?``
    when the transcript was unavailable so it could not be determined. Any other skill the
    agent loaded — the CLI ships its own — is not this skill and does not read ``yes``. A
    ``no`` is highlighted: that is a skill run not measuring its skill, and the transcript
    link in the same row is where to see why it did not load.
    """
    if row["arm"] == NOSKILL_ARM:
        return "&mdash;"
    raw = row.get("skill_loaded")
    if raw is None or (not isinstance(raw, bool) and pd.isna(raw)):
        return "?"
    return "yes" if bool(raw) else '<span class="skill-miss">no</span>'


def _runs_table_html(df: pd.DataFrame, out_dir: Path) -> str:
    """Per-run table with resource use and a link to each run's ``transcript.html``.

    ``answer`` and ``expected`` are deliberately **not** shown here — they can be long free
    text that would blow out the table width — but they remain columns in the sidecar CSV
    for anyone inspecting individual runs.
    """
    headers = [
        "arm",
        "split",
        "task",
        "model",
        "rep",
        "skill loaded",
        "reason",
        "turns",
        "in tok",
        "out tok",
        "total tok",
        "cost $",
        "time",
        "transcript",
    ]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for _, row in df.iterrows():
        transcript: Path = row["transcript_path"]
        if transcript.exists():
            try:
                href = transcript.relative_to(out_dir).as_posix()
            except ValueError:
                href = transcript.resolve().as_uri()
            link = f'<a href="{html.escape(href)}">view</a>'
        else:
            link = "&mdash;"
        status = "pass" if row["success"] else "fail"
        cells = [
            html.escape(str(row["arm_label"])),
            html.escape(str(row["split"])),
            html.escape(str(row["task_id"])),
            html.escape(str(row["model"])),
            str(int(row["rep"])),
            _loaded_cell(row),
            html.escape(str(row["reason"])),
            str(int(row["turns"])),
            f"{int(row['input_tokens']):,}",
            f"{int(row['output_tokens']):,}",
            f"{int(row['total_tokens']):,}",
            f"{float(row['cost_usd']):.3f}",
            _fmt_seconds(float(row["duration_s"])),
            link,
        ]
        body.append(f'<tr class="{status}">' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


_STYLE = f"""
:root {{ color-scheme: light; --ink: {INK}; --page: {PAGE}; --surface: {SURFACE}; --bar: {BAR}; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; line-height: 1.5;
       background: var(--page); color: var(--ink); display: flex; align-items: flex-start; }}
nav.toc {{ position: sticky; top: 0; flex: 0 0 210px; height: 100vh; overflow-y: auto;
       padding: 1.5rem 1rem; border-right: 1px solid {INK}22; font-size: 0.9rem; }}
nav.toc .toc-banner {{ display: block; width: 100%; max-width: 170px; height: auto; margin: 0 0 0.9rem; }}
nav.toc .toc-title {{ font-weight: 700; margin-bottom: 0.6rem; }}
nav.toc ul {{ list-style: none; margin: 0; padding: 0; }}
nav.toc li {{ margin: 0.15rem 0; }}
nav.toc ul ul {{ padding-left: 0.8rem; }}
nav.toc a {{ display: block; padding: 0.15rem 0.4rem; border-radius: 3px; text-decoration: none;
       color: var(--ink); border-left: 2px solid transparent; }}
nav.toc a:hover {{ background: {INK}0d; }}
nav.toc a.active {{ background: {BAR}22; border-left-color: var(--bar); font-weight: 600; }}
main {{ flex: 1 1 auto; min-width: 0; max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; }}
h1 {{ margin-bottom: 0.2rem; }}
h2 {{ border-bottom: 2px solid {BAR}33; padding-bottom: 0.2rem; scroll-margin-top: 1rem; }}
h3 {{ margin: 1.4rem 0 0.2rem; scroll-margin-top: 1rem; }}
a {{ color: var(--bar); }}
.meta {{ color: {BAR}; font-size: 0.9rem; margin-bottom: 1.5rem; }}
.task-desc {{ color: {BAR}; font-size: 0.9rem; margin: 0 0 0.4rem; }}
figure {{ margin: 0.6rem 0 1.6rem; text-align: center; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 1rem 0; display: block;
        overflow-x: auto; }}
th, td {{ border: 1px solid {INK}22; padding: 0.3rem 0.55rem; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ background: {BAR}1a; }}
tbody tr:nth-child(even) td {{ background: {INK}08; }}
tr.fail td {{ background: {ACCENT}66; }}
.skill-miss {{ color: #a4432b; font-weight: 700; }}
.note {{ background: {ACCENT}55; border-left: 3px solid var(--bar); padding: 0.5rem 0.8rem;
        margin: 0.5rem 0; border-radius: 3px; }}
.rationale {{ margin: 0.3rem 0 0.8rem; white-space: pre-wrap; }}
pre.diff {{ background: var(--surface); border: 1px solid {INK}22; border-radius: 4px;
        padding: 0.6rem 0.8rem; overflow-x: auto; font-size: 0.82rem; line-height: 1.35;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0.4rem 0 1.4rem; }}
pre.diff span {{ display: block; white-space: pre; }}
.diff-add {{ background: #4c7a3322; color: #2f5d1c; }}
.diff-del {{ background: #a4432b22; color: #8a2f1b; }}
.diff-hunk {{ color: var(--bar); font-weight: 600; }}
.diff-file {{ color: {INK}; font-weight: 700; }}
.diff-ctx {{ color: {INK}bb; }}
section {{ margin-top: 2.5rem; scroll-margin-top: 1rem; }}
@media (max-width: 720px) {{
  body {{ display: block; }}
  nav.toc {{ position: static; height: auto; flex-basis: auto; border-right: none;
       border-bottom: 1px solid {INK}22; }}
  nav.toc ul {{ columns: 2; }}
  main {{ margin: 1rem auto; }}
}}
"""

#: Highlights the nav link for whichever section is on screen. Progressive enhancement —
#: the anchor links jump correctly with JS disabled; this only adds the active marker.
_SCROLLSPY_JS = """
const links = new Map();
document.querySelectorAll('nav.toc a').forEach(a => links.set(a.getAttribute('href').slice(1), a));
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(a => a.classList.remove('active'));
      const a = links.get(e.target.id);
      if (a) a.classList.add('active');
    }
  });
}, { rootMargin: '0px 0px -75% 0px' });
document.querySelectorAll('[id]').forEach(el => { if (links.has(el.id)) observer.observe(el); });
"""


def _task_desc_html(task: Task) -> str:
    """The task's own prompts (train and test) — the actual task, not the runtime prompt.

    This is the text from ``tasks.yaml``, distinct from the full benchmark prompt the
    agent sees (which prepends the harness preamble in :mod:`acumen.prompts`).
    """
    parts = []
    for split in _SPLIT_ORDER:
        prompt = " ".join(task.split(split).prompt.split())
        parts.append(f'<p class="task-desc"><strong>{split}:</strong> {html.escape(prompt)}</p>')
    return "".join(parts)


def _diff_line_class(line: str) -> str:
    """CSS class for one line of a unified diff, by its leading marker."""
    if line.startswith("@@"):
        return "diff-hunk"
    if line.startswith(("+++", "---")):
        return "diff-file"
    if line.startswith("+"):
        return "diff-add"
    if line.startswith("-"):
        return "diff-del"
    return "diff-ctx"


def _skill_diff_html(parent: dict[str, str] | None, child: dict[str, str]) -> str:
    """Unified diff of a skill version's content against its parent, as coloured HTML.

    ``parent`` is ``None`` for the first version (nothing to diff against). Files present in
    only one side diff against an empty file, so added and removed references show up too.
    """
    if parent is None:
        return '<p class="task-desc">Initial version — no parent to diff against.</p>'
    lines: list[str] = []
    for rel in sorted(set(parent) | set(child)):
        before = parent.get(rel, "").splitlines(keepends=True)
        after = child.get(rel, "").splitlines(keepends=True)
        if before == after:
            continue
        diff = difflib.unified_diff(before, after, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)
        for raw in diff:
            text = raw.rstrip("\n")
            lines.append(f'<span class="{_diff_line_class(text)}">{html.escape(text)}</span>')
    if not lines:
        return '<p class="task-desc">No content change from the parent version.</p>'
    return '<pre class="diff">' + "\n".join(lines) + "</pre>"


def _skills_section_html(df: pd.DataFrame, skills_root: Path | None) -> tuple[str, str]:
    """Render the per-version skill provenance section: rationale + diff against parent.

    Returns ``(section_html, toc_html)`` — both empty when there is no skill directory to
    read, so a noskill-only report simply omits the section.
    """
    if skills_root is None or not skills_root.is_dir():
        return "", ""
    versions = [v for a in _arms_in_order(df) if (v := skill_from_arm(a)) is not None]
    if not versions:
        return "", ""

    blocks: list[str] = []
    toc: list[str] = []
    for version in versions:
        directory = skill_dir(skills_root, version)
        if not directory.is_dir():
            continue
        try:
            content = skill_content(directory)
        except SkillError:
            continue
        meta = read_meta(directory)
        parent_version = meta.parent if meta else None
        rationale = (meta.rationale if meta else "") or "(no rationale recorded)"
        hash_short = (meta.hash if meta else "")[:19]
        feedback = meta.feedback if meta else None

        parent_content: dict[str, str] | None = None
        if parent_version:
            parent_dir = skill_dir(skills_root, parent_version)
            if parent_dir.is_dir():
                try:
                    parent_content = skill_content(parent_dir)
                except SkillError:
                    parent_content = None

        anchor = f"skill-{version}"
        provenance = f"parent: {html.escape(parent_version)}" if parent_version else "initial draft"
        blocks.append(
            f'<h3 id="{html.escape(anchor)}">Skill {html.escape(version)}</h3>'
            f'<p class="task-desc">{provenance}'
            f"{f' &middot; {html.escape(hash_short)}…' if hash_short else ''}</p>"
            f'<p class="rationale">{html.escape(rationale)}</p>'
            f"{f'<p class="rationale"><em>Maintainer feedback:</em> {html.escape(feedback)}</p>' if feedback else ''}"
            f"{_skill_diff_html(parent_content, content)}"
        )
        toc.append(f'<li><a href="#{html.escape(anchor)}">Skill {html.escape(version)}</a></li>')

    if not blocks:
        return "", ""
    section = (
        '<section id="skills">\n<h2>Skill versions</h2>\n'
        '<p class="task-desc">Why each version was written, and what changed from its parent.</p>\n'
        + "".join(blocks)
        + "\n</section>"
    )
    toc_html = f'<li><a href="#skills">Skill versions</a><ul>{"".join(toc)}</ul></li>'
    return section, toc_html


def render_report(
    df: pd.DataFrame,
    out_dir: Path,
    tasks: Sequence[Task] | None = None,
    data_href: str | None = None,
    skills_root: Path | None = None,
    palette: Mapping[str, str] | None = None,
) -> str:
    """Render the full report HTML for the results in ``df``.

    Parameters
    ----------
    df
        Results from :func:`load_results`.
    out_dir
        The directory ``report.html`` will live in — transcript links are made relative
        to it so the report stays portable alongside the run tree.
    tasks
        The loaded tasks, if available — used to print each task's own prompt under its
        per-task heading. When ``None``, the heading shows the task id alone.
    data_href
        Relative link to the aggregated data file (the CSV backing the runs table), shown
        above it so a reader can grab the numbers. When ``None``, no link is shown.
    skills_root
        The ``skills/`` root, if available — used to render the per-version rationale and a
        diff of each skill against its parent. When ``None`` (or missing), the section is
        omitted, so a noskill-only report still renders.
    palette
        Bar colours keyed by model id, overriding the tier defaults — see
        :func:`resolve_palette`. Resolved once here, against every model in ``df``, so a
        bad key fails before any figure is drawn and each model keeps one colour across
        the overview and the per-task figures.
    """
    colors = resolve_palette(_models_in_order(df), palette)
    overview_uri = figure_data_uri(metrics_figure(df, split_hue=False, colors=colors))
    task_by_id = {t.id: t for t in tasks or []}
    skills_section, skills_toc = _skills_section_html(df, skills_root)

    task_blocks = []
    toc_tasks = []
    for task_id in sorted(df["task_id"].unique()):
        subset = df[df["task_id"] == task_id]
        uri = figure_data_uri(metrics_figure(subset, split_hue=True, colors=colors))
        task = task_by_id.get(task_id)
        desc = _task_desc_html(task) if task is not None else ""
        anchor = f"task-{task_id}"
        task_blocks.append(
            f'<h3 id="{html.escape(anchor)}">{html.escape(task_id)}</h3>'
            f"{desc}"
            f'<figure><img alt="Metrics for {html.escape(task_id)}" src="{uri}"></figure>'
        )
        toc_tasks.append(f'<li><a href="#{html.escape(anchor)}">{html.escape(task_id)}</a></li>')

    toc = f"""<nav class="toc">
<img class="toc-banner" src="{_asset_data_uri("banner.svg")}" alt="acumen">
<div class="toc-title">acumen report</div>
<ul>
<li><a href="#overview">Overview</a></li>
<li><a href="#per-task">Per-task breakdown</a><ul>{"".join(toc_tasks)}</ul></li>
{skills_toc}
<li><a href="#runs">Runs</a></li>
</ul>
</nav>"""

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    arms = ", ".join(arm_label(a) for a in _arms_in_order(df))
    tasks = ", ".join(sorted(df["task_id"].unique()))
    notes = "".join(f'<div class="note">⚠ {html.escape(n)}</div>' for n in _integrity_notes(df))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/svg+xml" href="{_asset_data_uri("logo.svg")}">
<title>acumen report</title>
<style>{_STYLE}</style>
</head>
<body>
{toc}
<main>
<h1>acumen benchmark report</h1>
<div class="meta">Generated {generated} &middot; {len(df)} runs &middot; test split shown
 &middot; arms: {html.escape(arms)} &middot; tasks: {html.escape(tasks)}</div>
{notes}
<section id="overview">
<h2>Overview</h2>
<p class="task-desc">Values for test runs.</p>
<figure><img alt="Success rate, tokens, cost and time per skill" src="{overview_uri}"></figure>
</section>
<section id="per-task">
<h2>Per-task breakdown</h2>
{"".join(task_blocks)}
</section>
{skills_section}
<section id="runs">
<h2>Runs</h2>
{f'<p class="task-desc">Full data: <a href="{html.escape(data_href)}">{html.escape(data_href)}</a></p>' if data_href else ""}
{_runs_table_html(df, out_dir)}
</section>
</main>
<script>{_SCROLLSPY_JS}</script>
</body>
</html>
"""


def build_report(
    runs_root: Path,
    out_path: Path,
    tasks: Sequence[Task] | None = None,
    skills_root: Path | None = None,
    palette: Mapping[str, str] | None = None,
) -> Report:
    """Aggregate the run tree and render ``report.html``.

    The report is overwritten in place, so it always reflects every run on disk.

    Parameters
    ----------
    runs_root
        The ``runs/`` root to aggregate.
    out_path
        Where to write ``report.html``.
    tasks
        The loaded tasks, if available — used to print each task's prompt under its
        per-task heading.
    skills_root
        The ``skills/`` root, if available — used to render each skill version's rationale
        and its diff against the parent version. When ``None``, the section is omitted.
    palette
        Bar colours keyed by model id, e.g. ``{"claude-opus-5": "#3b7ea1"}``, overriding
        the tier defaults — see :func:`resolve_palette`.

    Returns
    -------
    The rendered :class:`Report` (also written to ``out_path``).
    """
    df = load_results(runs_root)
    resolve_palette(_models_in_order(df), palette)  # reject a bad palette before writing anything
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # A sidecar CSV of the aggregated results, for anyone who wants to reanalyse or plot
    # the numbers themselves. Drop the internal helper columns (Path objects and sort
    # keys) so it stays a clean, portable table.
    data_path = out_path.with_suffix(".csv")
    internal = [c for c in ("result_path", "transcript_path", "_arm_order") if c in df.columns]
    df.drop(columns=internal).to_csv(data_path, index=False)

    html_text = render_report(
        df, out_path.parent, tasks, data_href=data_path.name, skills_root=skills_root, palette=palette
    )
    out_path.write_text(html_text)
    return Report(html=html_text, results=df)
