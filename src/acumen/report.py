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
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from acumen.paths import NOSKILL_ARM, RESULT_FILE, TRANSCRIPT_HTML, skill_from_arm
from acumen.skills import SkillError, read_meta, skill_content, skill_dir
from acumen.tasks import Task

# ── Palette ──────────────────────────────────────────────────────────────────────────
# A warm-neutral scheme fixed by the maintainer. The page and the plot area are both white,
# so a figure sits on the page with no visible frame around it and the ink carries the
# structure. Bars are coloured by model (see the sequential ramp below), with a grey bar
# pooling every model; train/test is a texture, not a colour.
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

# Alongside the per-model bars, each cell carries one pooled bar over every model at once —
# the overall position of the models being evaluated. It is deliberately outside the model
# ramp: a neutral grey, so it reads as a summary of the hues rather than another hue.
_ALL_MODELS = "\x00all-models"  # sentinel model id; not a value any real id can take
_ALL_MODELS_LABEL = "all models"
_ALL_MODELS_COLOR = "#9b968d"

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


def _arm_marker(arm: str) -> str | tuple[int, int, int]:
    """Scatter marker for an arm — an ✕ for the baseline, then a widening polygon per version.

    The trade-off figure spends hue on the *model*, the same as every other figure here, so
    the arm needs a second channel: shape. Arms are ordinal, and most shape sets are not, so
    the versions walk a polygon ladder — v1 a triangle, v2 a square, v3 a pentagon — where the
    side count rises with the version and the progression reads in order. matplotlib takes the
    ``(numsides, style, angle)`` form, so the ladder is arithmetic rather than a lookup table
    that eventually runs out of shapes.

    The baseline sits deliberately off the ladder: it is not a version of anything, and an ✕
    says so at a glance. Past roughly v6 the polygons start converging on a circle; there the
    legend and the labelled pooled marks carry the identity instead.

    Each polygon is drawn point-up, so v1 reads as a triangle and v2 as a diamond rather than
    a square — the side count is what carries the order, not the orientation.
    """
    if arm == NOSKILL_ARM:
        return "X"
    version = skill_from_arm(arm)
    return (int(version[1:]) + 2, 0, 0) if version else "o"


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


def _loaded_flags(df: pd.DataFrame) -> pd.Series:
    """Per-run "loaded the skill under test" booleans, one per row of ``df``.

    A run whose transcript could not be read leaves ``skill_loaded`` unset — undetermined,
    which is not evidence the skill loaded, so it reads ``False`` here and counts against
    the load rate. Results predating the field are treated the same way.
    """
    if "skill_loaded" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["skill_loaded"].fillna(False).astype(bool)


def arm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm success rate and resource totals for the runs in ``df``.

    The caller filters ``df`` first — to the test split, and optionally to one task. Each
    row is one arm, in report order.

    The resource columns here are **totals** — the budget view, "what did this arm cost to
    run" — which is deliberately not what the figures show: those are per-run means (see
    :func:`metrics_figure`). Divide by ``n`` to get from one to the other.

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


#: The metric columns: (title, key, x-axis formatter, bar-label fn). The key selects the
#: value; the proportion keys below are special-cased for binomial error bars.
#:
#: Every column is a **per-run mean**, so all five are on one footing: a bar answers "what
#: does one run of this arm look like?", and none of them moves just because an arm happens
#: to have been run more times. Cost carries three decimals — the same precision as the runs
#: table, which is now the same unit — because a per-run mean can sit well under a cent.
_COLUMNS = [
    ("Success rate", "rate", mticker.PercentFormatter(1.0), lambda v: f"{v:.0%}"),
    ("Skill loaded", "loaded", mticker.PercentFormatter(1.0), lambda v: f"{v:.0%}"),
    ("Tokens / run", "tokens", mticker.FuncFormatter(lambda v, _p: _fmt_tokens(v)), _fmt_tokens),
    ("Cost / run", "cost", mticker.FuncFormatter(lambda v, _p: f"${v:.3f}"), lambda v: f"${v:.3f}"),
    ("Time / run", "time", mticker.FuncFormatter(lambda v, _p: _fmt_seconds(v)), _fmt_seconds),
]

#: Metric key → the boolean column it averages. A proportion is the mean of a 0/1 column, so
#: it is a per-run mean like the rest; it differs only in being drawn on a fixed 0–1 axis.
_PROPORTION_COLUMN = {"rate": "success", "loaded": "skill_loaded"}

#: Metric key → the resource column it averages (the proportions are handled separately).
_MEAN_COLUMN = {"tokens": "total_tokens", "cost": "cost_usd", "time": "duration_s"}


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


def _bar_rows(models: Sequence[str]) -> list[str]:
    """The bar slots in a cell: one per model, plus the pooled row when there are several.

    With a single model the pooled bar would simply restate it, so it is left out.
    """
    return [*models, _ALL_MODELS] if len(models) > 1 else list(models)


def _row_label(model: str) -> str:
    """Legend label for a bar slot — a model id, or the pooled row's name."""
    return _ALL_MODELS_LABEL if model == _ALL_MODELS else _model_label(model)


def _cell_value(df: pd.DataFrame, arm: str, model: str, split: str, key: str) -> tuple[float, float, bool]:
    """One (arm, model, split) cell's metric value, its error bar, and whether it has runs.

    Every metric is a **mean over runs**, with the standard error of that mean as the error
    bar — so all five columns are read the same way and none of them is inflated by an arm
    simply having more runs behind it. For the proportions (``rate``, ``loaded``) that mean
    is the fraction of runs and the error is the binomial standard error; for the resources
    it is the per-run average and ``std / sqrt(n)`` (0 with <2 runs, which have no spread to
    report). A proportion is just the mean of a 0/1 column, so the two cases agree.

    ``model`` may be the :data:`_ALL_MODELS` sentinel, which pools every model in the arm.
    Pooling is over *runs*, not over the per-model bars: an under-sampled model must not
    weigh as much as a well-sampled one. So the grey bar sits among the model bars rather
    than above them — it is where an average run lands, whichever model drew it.

    The third element distinguishes a genuine zero (e.g. a 0% success rate) from a cell with
    no runs at all: a model not run in this arm reserves its slot but draws no bar.
    """
    rows = (df["arm"] == arm) & (df["split"] == split)
    group = df[rows if model == _ALL_MODELS else rows & (df["model"] == model)]
    if group.empty:
        return 0.0, 0.0, False
    if key in _PROPORTION_COLUMN:
        flags = _loaded_flags(group) if key == "loaded" else group[_PROPORTION_COLUMN[key]]
        rate, error = _rate_and_error(flags)
        return rate, error, True
    values = group[_MEAN_COLUMN[key]]
    mean = float(values.mean())
    error = float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, error, True


def _draw_cell(
    ax: plt.Axes,
    df: pd.DataFrame,
    arm: str,
    rows: list[str],
    splits: list[str],
    key: str,
    *,
    value_label,
    colors: Mapping[str, str],
) -> None:
    """Draw one grid cell — one fixed slot per bar row (colour), grouped by split (texture).

    ``rows`` is the models plus, last, the pooled :data:`_ALL_MODELS` row. Every row gets a
    slot at the same y-position in every cell, so bars line up across skills. A model with no
    runs in this arm reserves its slot but draws nothing — no bar, no error bar, no label —
    so a stretched neighbour can't be mistaken for it.
    """
    n_models = len(rows)
    n_splits = len(splits)
    group_h = 0.8
    bar_h = group_h / n_splits
    nan = float("nan")
    for si, split in enumerate(splits):
        offset = (si - (n_splits - 1) / 2) * bar_h
        cells = [_cell_value(df, arm, model, split, key) for model in rows]
        # A NaN width draws no bar and no error bar; the y-slot is still occupied.
        bars = ax.barh(
            [m + offset for m in range(n_models)],
            [value if present else nan for value, _err, present in cells],
            bar_h * 0.86,
            color=[colors[model] for model in rows],
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

    Every bar is a mean over runs — success rate, skill-load rate, then tokens, cost and
    time per run — each with the standard error of that mean. Reporting the resources per
    run rather than as totals keeps them comparable with the two rates and with each other:
    an arm covering more tasks or more reps no longer looks more expensive for it. Columns
    share an x-axis so skills are comparable within a metric. The load rate is the share of
    runs that loaded the skill under test — the baseline arm is drawn too, where a bar above
    zero means the baseline is not clean.

    Inside each cell, bars are coloured by model, with a final grey bar pooling every model
    at once — where an average run lands overall. When ``split_hue`` is set, train and test
    are shown as two textures (test solid, train hatched) rather than two colours, leaving
    the model hue intact.

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
    rows = _bar_rows(models)
    resolved = colors or {}
    colors = {model: resolved.get(model, _model_color(model)) for model in models}
    colors[_ALL_MODELS] = _ALL_MODELS_COLOR
    splits = list(_SPLIT_ORDER) if split_hue else [_REPORTED_SPLIT]
    # Every row gets a fixed slot in every arm's cell, so bars align across skills and a
    # model that wasn't run leaves a reserved gap rather than letting its neighbour stretch.
    n_rows = max(1, len(arms))
    max_bars = max(1, len(rows) * len(splits))
    with plt.rc_context(_RC):
        # Height scales with the number of bars per cell; the small constant covers the
        # per-row breathing room, not a fixed block, so few-model plots stay compact.
        row_h = 0.26 * max_bars + 0.22
        fig, axes = plt.subplots(
            n_rows, len(_COLUMNS), sharex="col", squeeze=False, figsize=(10.0, n_rows * row_h + 0.6)
        )
        for c, (title, key, formatter, value_label) in enumerate(_COLUMNS):
            for r, arm in enumerate(arms):
                ax = axes[r][c]
                _draw_cell(ax, df, arm, rows, splits, key, value_label=value_label, colors=colors)
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
                if r == 0:
                    ax.set_title(title, fontsize=10)
                if c == 0:
                    ax.set_ylabel(_skill_label(arm), rotation=0, ha="right", va="center", fontweight="bold")
            # sharex="col" ties the whole column together, so one xlim covers every row.
            if key in _PROPORTION_COLUMN:
                axes[-1][c].set_xlim(0, 1.28)
            else:
                reach = max(
                    (
                        v + e
                        for arm in arms
                        for m in rows
                        for s in splits
                        for v, e, present in [_cell_value(df, arm, m, s, key)]
                        if present
                    ),
                    default=1.0,
                )
                axes[-1][c].set_xlim(0, reach * 1.35 if reach else 1)

        model_labels = [_row_label(m) for m in rows]
        model_handles = [
            Patch(facecolor=colors[m], edgecolor=PLOT_BG, label=lab) for m, lab in zip(rows, model_labels, strict=True)
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


def _pareto_front(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The non-dominated ``(cost, rate)`` points, cheapest first.

    A point is dominated when another costs no more *and* succeeds no less — nothing would make
    you pick it. What survives is the frontier: the best rate available at each price.

    Sorting by cost ascending and rate descending lets one pass do it. Walking that order, a
    point is on the frontier exactly when it beats the best rate seen so far, since everything
    already passed costs no more. The tie-break on rate matters: at equal cost the better rate
    comes first and knocks out its twin.
    """
    front: list[tuple[float, float]] = []
    best = -math.inf
    for cost, rate in sorted(points, key=lambda p: (p[0], -p[1])):
        if rate > best:
            front.append((cost, rate))
            best = rate
    return front


def _pareto_steps(front: Sequence[tuple[float, float]], y_min: float) -> tuple[list[float], list[float]]:
    """The frontier as a staircase: x and y for a path tracing the edge of what was achieved.

    Between two frontier points the best rate available is the cheaper one's, so the path holds
    that rate until the price of the next one is reached and then steps up — a curve through the
    points would claim results between them that nobody measured. The path drops to the floor at
    the cheapest point, closing the region off: to its left, nothing was achieved at any rate.
    """
    xs: list[float] = []
    ys: list[float] = []
    held = y_min  # the rate carried forward from the previous step, or the floor at the first
    for cost, rate in front:
        xs += [cost, cost]
        ys += [held, rate]
        held = rate
    return xs, ys


def tradeoff_figure(df: pd.DataFrame, *, colors: Mapping[str, str] | None = None) -> plt.Figure:
    """Cost per run against success rate — where each skill version buys what, and at what price.

    The metrics grid reports every measure on its own axis, which answers "how much?" but never
    "was it worth it?". This plots the two headline measures against each other, so the reader
    can see whether a version bought accuracy, saved money, or both. Up and to the left is better.

    Each model gets its own mark per arm, in the model's colour; each arm also gets a larger grey
    mark pooling every model, carrying standard errors on both axes — the same statistic the
    grid's grey bar reports. Only the pooled marks are labelled; labelling every point would
    collide. Hue stays the *model*, as in every other figure here, so the arm rides a second
    channel: marker shape (see :func:`_arm_marker`). Identity therefore never rests on colour alone.

    A staircase traces the **Pareto frontier** over every mark shown — the combinations nothing
    else beats on both counts at once. Anything below and to the right of it is dominated:
    something on the line costs less *and* succeeds more, so there is no reason to choose it.
    The frontier is drawn over the pooled marks as well as the per-model ones, so it really is
    the outer edge of the whole plot and no mark can float above it.

    Cost is anchored at zero, but the rate axis starts just below the lowest mark rather than at
    zero. Anchoring it too would strand every point in the top third of an otherwise empty panel,
    close enough together to hide the differences the figure exists to show. That is safe here in
    a way it would not be on the grid's bars: position carries the value, not the length of
    anything, the axis is labelled in percent throughout, and the pooled marks show their error
    bars — so a small gap cannot read as a large one.

    Parameters
    ----------
    df
        Results from :func:`load_results`. Only the reported (test) split is plotted.
    colors
        Mark colour per model id, as returned by :func:`resolve_palette`; models missing from it
        fall back to their tier default, and ``None`` uses the defaults throughout.
    """
    arms = _arms_in_order(df)
    models = _models_in_order(df)
    resolved = colors or {}
    colors = {model: resolved.get(model, _model_color(model)) for model in models}
    colors[_ALL_MODELS] = _ALL_MODELS_COLOR

    # (arm, model, cost, cost_err, rate, rate_err) for every cell with runs behind it. Both
    # coordinates come from _cell_value, so these marks cannot drift from the grid's bars.
    points = []
    for arm in arms:
        for model in [*models, _ALL_MODELS]:
            cost, cost_err, present = _cell_value(df, arm, model, _REPORTED_SPLIT, "cost")
            if not present:
                continue
            rate, rate_err, _ = _cell_value(df, arm, model, _REPORTED_SPLIT, "rate")
            points.append((arm, model, cost, cost_err, rate, rate_err))

    x_max = max((cost + err for _a, _m, cost, err, _r, _re in points), default=0.0) * 1.15 or 1.0
    # The pooled marks carry error bars and the per-model ones do not, so only the pooled ones
    # reach below their own value. Round down to a tenth so the ticks land on whole percents.
    floors = [rate - err if model == _ALL_MODELS else rate for _a, model, _c, _ce, rate, err in points]
    y_min = max(0.0, math.floor((min(floors, default=0.0) - 0.02) * 10) / 10)
    y_max = 1.04  # a little air over a perfect score, without a tick past 100%
    front = _pareto_front([(cost, rate) for _a, _m, cost, _e, rate, _re in points])

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        steps_x, steps_y = _pareto_steps(front, y_min)
        ax.plot(steps_x, steps_y, color=INK, alpha=0.38, linewidth=1.1, zorder=2)
        if front:
            # Named at the foot of its first riser — the cheap-and-poor corner, which by
            # construction has nothing plotted in it.
            ax.annotate(
                "Pareto frontier",
                (front[0][0], y_min),
                textcoords="offset points",
                xytext=(5, 4),
                ha="left",
                va="bottom",
                fontsize=7,
                color=INK,
                alpha=0.6,
            )

        for arm, model, cost, _err, rate, _rate_err in points:
            if model == _ALL_MODELS:
                continue
            ax.plot(
                cost,
                rate,
                marker=_arm_marker(arm),
                color=colors[model],
                markersize=7,
                # An ink edge, not the page colour: the lighter tiers of the model ramp are pale
                # enough that a white outline on a white page dissolves the mark's own shape,
                # which is the channel carrying the arm.
                markeredgecolor=INK,
                markeredgewidth=0.7,
                linestyle="none",
                zorder=3,
            )
        # The pooled marks go on last so they sit above the per-model cloud they summarise.
        for arm, model, cost, cost_err, rate, rate_err in points:
            if model != _ALL_MODELS:
                continue
            ax.errorbar(
                cost,
                rate,
                xerr=cost_err,
                yerr=rate_err,
                marker=_arm_marker(arm),
                color=_ALL_MODELS_COLOR,
                markersize=10,
                markeredgecolor=INK,
                markeredgewidth=0.9,
                linestyle="none",
                capsize=3,
                ecolor=INK,
                elinewidth=1,
                zorder=4,
            )
            ax.annotate(
                _skill_label(arm),
                (cost, rate),
                textcoords="offset points",
                xytext=(10, 6),
                fontsize=8,
                fontweight="bold",
                color=INK,
                zorder=5,
            )

        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"${v:.3f}"))
        ax.set_xlabel("Cost / run")
        ax.set_ylabel("Success rate")
        ax.grid(color=INK, alpha=0.14, linewidth=0.8)
        ax.set_axisbelow(True)
        # The ticks have no length, so by default the labels sit tight against the spines and
        # the pair nearest the origin collide: the first cost label is centred on the corner,
        # which puts half of it under the rate label sitting at the same height. Dropping the
        # cost labels clears the rate label by height instead, which holds however wide either
        # label runs — unlike nudging them sideways, which depends on the numbers' width.
        ax.tick_params(length=0, pad=4)
        ax.tick_params(axis="x", pad=9)

        rows = _bar_rows(models)
        model_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                color=colors[m],
                markersize=7,
                markeredgecolor=INK,
                label=_row_label(m),
            )
            for m in rows
        ]
        fig.legend(
            model_handles,
            [_row_label(m) for m in rows],
            title="model",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
            title_fontsize=9,
            labelspacing=0.35,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        # Shape is the channel on show here, so these proxies are drawn as outlines only: any
        # fill would read as a model colour, and there is no model to name in a key about shape.
        arm_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=_arm_marker(a),
                markerfacecolor="none",
                markersize=7,
                markeredgecolor=INK,
                markeredgewidth=1.1,
                label=_skill_label(a),
            )
            for a in arms
        ]
        fig.legend(
            arm_handles,
            [_skill_label(a) for a in arms],
            title="skill",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 0.52),
            fontsize=8,
            title_fontsize=9,
            labelspacing=0.35,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.tight_layout()
    return fig


#: Pixel width every figure is rendered to, whatever its width in inches. Roughly twice the
#: widest the text column ever gets, so the marks stay sharp on high-density screens and after
#: the browser scales the image down to the column.
_FIGURE_WIDTH_PX = 1800


def figure_data_uri(fig: plt.Figure) -> str:
    """Render a figure to a base64 PNG ``data:`` URI, then close it.

    The dpi is derived from the figure's own width rather than fixed, so every figure lands at
    :data:`_FIGURE_WIDTH_PX` whatever its size in inches. The page scales images to fit one text
    column, so a fixed dpi renders the narrow figures at fewer pixels than the wide ones and
    leaves the browser to upscale them — which is visible as soft, pixelated marks next to a
    crisp one on the same page.

    The figure background is saved **transparent** (``facecolor="none"``) so the margins
    between subplots pick up the page colour in the HTML but export blank — the axes
    interiors keep their own opaque fill. ``transparent=True`` is deliberately *not* used:
    it would also blank the axes patches.
    """
    buffer = io.BytesIO()
    dpi = _FIGURE_WIDTH_PX / fig.get_figwidth()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ── Significance ─────────────────────────────────────────────────────────────────────

#: Resamples behind every p-value and frontier share. Large enough that the resolution floor
#: (1/resamples) sits far below any threshold anyone reads off the table.
_BOOTSTRAP_RESAMPLES = 20_000

#: Fixed, so regenerating a report never moves its p-values. A conclusion that changed because
#: the report was rebuilt would be worse than no conclusion.
_BOOTSTRAP_SEED = 0

#: What a resample draws. Runs on the same task are correlated — some tasks are simply harder —
#: so the task is the independent unit, not the run. Treating all runs as independent would be
#: pseudoreplication and would understate every standard error. Models are deliberately *not*
#: part of this: they are a fixed set that was chosen, not a sample to generalise over.
_CLUSTER_COLUMN = "task_id"

#: Below this many tasks the resamples repeat too often to say anything — with one task every
#: resample is identical and p collapses to 0 or 1. The section says so rather than printing
#: numbers it cannot support.
_MIN_CLUSTERS = 3


def _holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, returned in the order given.

    Controls the probability of *any* false claim across the family — the right target when the
    family feeds a single decision (ship this version or not), rather than the false-discovery
    rate, which earns its keep over hundreds of tests. Holm is also valid under arbitrary
    dependence, which matters here: every comparison reuses the baseline arm and the same
    bootstrap draws, so the tests are far from independent.

    The running maximum enforces monotonicity — a smaller raw p can never adjust to a larger
    value than one above it in the ordering.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = running
    return adjusted


def _cluster_totals(df: pd.DataFrame, arms: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summed cost, successes and run counts per (task, arm) over the reported split.

    These are the sufficient statistics for the bootstrap: both metrics are ratios of sums, so a
    resample never has to touch an individual run again.
    """
    test = df[df["split"] == _REPORTED_SPLIT]
    clusters = sorted(test[_CLUSTER_COLUMN].unique())
    shape = (len(clusters), len(arms))
    cost, successes, runs = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for ci, cluster in enumerate(clusters):
        for ai, arm in enumerate(arms):
            rows = test[(test[_CLUSTER_COLUMN] == cluster) & (test["arm"] == arm)]
            cost[ci, ai] = rows["cost_usd"].sum()
            successes[ci, ai] = rows["success"].sum()
            runs[ci, ai] = len(rows)
    return cost, successes, runs


def _bootstrap_arms(
    cost: np.ndarray, successes: np.ndarray, runs: np.ndarray, *, resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole tasks and recompute every arm's rate and cost per run.

    Drawing ``n`` tasks with replacement *is* a multinomial over the tasks, so the cluster counts
    come from one call and the whole bootstrap collapses to two matrix products — every arm and
    every resample at once. Each resample reuses the same drawn tasks for every arm, which is
    what makes the comparison paired: task difficulty cancels instead of adding noise.
    """
    n_clusters = len(cost)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=resamples).astype(float)
    drawn = counts @ runs
    # An arm absent from every drawn task divides by zero; the NaN is dropped downstream rather
    # than being counted as evidence either way.
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ successes) / drawn, (counts @ cost) / drawn


def _frontier_probability(rate: np.ndarray, cost_per_run: np.ndarray) -> np.ndarray:
    """Share of resamples in which each arm is non-dominated.

    The same rule as :func:`_pareto_front` — cheaper-or-equal and better-or-equal, strictly so on
    one axis — broadcast over every pair of arms and every resample at once, so the column and
    the plot's staircase cannot disagree. Read it as robustness, not as a test: it says how often
    an arm survives a fresh draw of tasks, and has no null hypothesis to correct for.
    """
    cost, other_cost = cost_per_run[:, :, None], cost_per_run[:, None, :]
    own, other = rate[:, :, None], rate[:, None, :]
    dominated = (other_cost <= cost) & (other >= own) & ((other_cost < cost) | (other > own))
    return (~dominated.any(axis=2)).mean(axis=0)


def _dominance_p(
    rate: np.ndarray, cost_per_run: np.ndarray, challenger: int, reference: int
) -> tuple[float, float, float]:
    """One-sided p per axis and their max: does ``challenger`` beat ``reference`` on *both*?

    An intersection–union test. Dominance is a conjunction — cheaper *and* more accurate — so the
    evidence for it is only as strong as its weakest half, and the max is the p-value. That the
    two axes need no multiplicity correction is a property of the conjunction, not an oversight:
    getting lucky on one axis buys nothing when the other must clear as well.

    This is what stops a version buying its way in on price. An arm that is dramatically cheaper
    but no more accurate scores overwhelming evidence on cost, none on rate, and fails.
    """
    d_rate = rate[:, challenger] - rate[:, reference]
    d_cost = cost_per_run[:, challenger] - cost_per_run[:, reference]
    usable = np.isfinite(d_rate) & np.isfinite(d_cost)
    d_rate, d_cost = d_rate[usable], d_cost[usable]
    if not len(d_rate):
        return 1.0, 1.0, 1.0
    floor = 1 / len(d_rate)  # the bootstrap cannot resolve below one resample
    p_rate = max(float((d_rate <= 0).mean()), floor)
    p_cost = max(float((d_cost >= 0).mean()), floor)
    return p_rate, p_cost, max(p_rate, p_cost)


@dataclass(frozen=True)
class SkillTests:
    """Whether the arms differ by more than noise, and how robustly.

    Attributes
    ----------
    arms
        One row per arm: its success rate, cost per run, and the share of resamples in which it
        sits on the Pareto frontier.
    comparisons
        One row per claim "this version dominates the baseline", with the per-axis deltas, the
        intersection–union p, and the Holm-adjusted p across them.
    baseline
        The arm everything is measured against — the noskill arm when there is one.
    n_clusters
        Tasks behind the resampling — the real sample size for every p-value here.
    """

    arms: pd.DataFrame
    comparisons: pd.DataFrame
    baseline: str
    n_clusters: int

    @property
    def usable(self) -> bool:
        """Whether there were enough tasks, and enough arms, to test anything."""
        return self.n_clusters >= _MIN_CLUSTERS and not self.comparisons.empty


def skill_tests(df: pd.DataFrame, *, resamples: int = _BOOTSTRAP_RESAMPLES, seed: int = _BOOTSTRAP_SEED) -> SkillTests:
    """Test which skill versions beat which, on cost and success rate jointly.

    Deliberately reports no single combined score. Folding rate and cost into one number picks an
    exchange rate between them on the reader's behalf, and that choice decides the answer: a
    version that is much cheaper while being *worse* at the task can come out ahead. The
    dominance test in :func:`_dominance_p` requires both axes to improve, so it cannot be bought
    off that way, and :func:`_frontier_probability` ranks the arms without needing a threshold.

    Only the versus-baseline claims are tested — the question the benchmark exists to answer, and
    the family the Holm adjustment covers. Testing every version against every other as well
    would spend the correction budget on claims nobody makes (whether the *baseline* dominates a
    skill) and can bury the real result: on a four-arm report it turns a Holm p of 0.03 into 0.15.

    Parameters
    ----------
    df
        Results from :func:`load_results`. Only the reported (test) split is used.
    resamples, seed
        Bootstrap size and its seed. The default seed is fixed so a rebuilt report reproduces
        its own p-values exactly.
    """
    arms = _arms_in_order(df)
    cost, successes, runs = _cluster_totals(df, arms)
    n_clusters = len(cost)
    baseline = arms[0] if arms else NOSKILL_ARM  # noskill sorts first when it is present
    if n_clusters < _MIN_CLUSTERS or len(arms) < 2:
        empty = pd.DataFrame(columns=["challenger", "reference", "d_rate", "d_cost", "p", "p_adjusted"])
        return SkillTests(
            arms=pd.DataFrame(columns=["arm", "rate", "cost", "frontier"]),
            comparisons=empty,
            baseline=baseline,
            n_clusters=n_clusters,
        )

    rate, cost_per_run = _bootstrap_arms(cost, successes, runs, resamples=resamples, seed=seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_rate = successes.sum(axis=0) / runs.sum(axis=0)
        observed_cost = cost.sum(axis=0) / runs.sum(axis=0)
    arm_rows = pd.DataFrame(
        {
            "arm": arms,
            "rate": observed_rate,
            "cost": observed_cost,
            "frontier": _frontier_probability(rate, cost_per_run),
        }
    )

    rows = []
    for challenger in range(1, len(arms)):
        p_rate, p_cost, p = _dominance_p(rate, cost_per_run, challenger, 0)
        rows.append(
            {
                "challenger": arms[challenger],
                "reference": baseline,
                "d_rate": observed_rate[challenger] - observed_rate[0],
                "d_cost": observed_cost[challenger] - observed_cost[0],
                "p_rate": p_rate,
                "p_cost": p_cost,
                "p": p,
            }
        )
    comparisons = pd.DataFrame(rows)
    comparisons["p_adjusted"] = _holm(comparisons["p"].tolist())
    return SkillTests(arms=arm_rows, comparisons=comparisons, baseline=baseline, n_clusters=n_clusters)


def _fmt_delta_rate(value: float) -> str:
    """A change in success rate, in percentage *points* — 77.8% to 73.3% is -4.4pp, not -4.4%."""
    return f"{value * 100:+.1f}pp"


def _fmt_delta_cost(value: float) -> str:
    """A change in cost per run, with the sign outside the currency: ``-$0.083``."""
    return f"{'+' if value >= 0 else '-'}${abs(value):.3f}"


def _fmt_p(p: float) -> str:
    """A p-value at the precision the bootstrap can actually resolve."""
    if not math.isfinite(p):
        return "&mdash;"
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


#: The table's columns: (header, field, higher-is-better, formatter). The direction is per column
#: because "best" does not point one way here — the strongest arm has the *highest* success rate
#: and the *lowest* cost, and a reader scanning the bold cells for the winner should not have to
#: keep track of which column runs which way. The first few describe an arm on its own; the rest
#: compare it with the baseline.
_TEST_COLUMNS = (
    ("Success rate", "rate", True, lambda v: f"{v:.1%}"),
    ("Cost / run", "cost", False, lambda v: f"${v:.3f}"),
    ("On frontier", "frontier", True, lambda v: f"{v:.1%}"),
    ("&Delta; rate", "d_rate", True, _fmt_delta_rate),
    ("&Delta; cost", "d_cost", False, _fmt_delta_cost),
    ("p", "p", False, _fmt_p),
    ("Holm p", "p_adjusted", False, _fmt_p),
)

#: How many of the above describe the arm itself; the remainder are the comparison, and sit under
#: a shared header so it is clear they all refer to the baseline rather than to each other.
_ARM_COLUMNS = 3


def _best_cells(values: Sequence[float | None], *, highest: bool) -> set[int]:
    """Row indices holding the best value in one column, by that column's own direction.

    A lone value has beaten nothing, so a column with fewer than two populated cells gets no
    emphasis at all. Ties are all marked rather than arbitrarily crowning whichever came first.
    """
    present = [v for v in values if v is not None and math.isfinite(v)]
    if len(present) < 2:
        return set()
    target = max(present) if highest else min(present)
    return {i for i, v in enumerate(values) if v is not None and v == target}


def _tests_table_html(tests: SkillTests) -> str:
    """The significance table: one row per arm, with the best value in each column set bold."""
    if not tests.usable:
        return (
            f'<div class="note">Not enough to test: a paired bootstrap needs at least {_MIN_CLUSTERS} '
            f"tasks and two arms to compare (this report has {tests.n_clusters}). The figures above "
            f"still describe what was measured.</div>"
        )

    by_arm = {row.challenger: row for row in tests.comparisons.itertuples()}
    compared = [field for _title, field, _highest, _fmt in _TEST_COLUMNS[_ARM_COLUMNS:]]
    records = []
    for row in tests.arms.itertuples():
        match = by_arm.get(row.arm)
        # The baseline is not compared with itself: those cells stay empty, and take no part in
        # deciding which value in the column is best.
        records.append(
            {f: getattr(row, f) for f in ("rate", "cost", "frontier")}
            | {f: (None if match is None else getattr(match, f)) for f in compared}
        )
    best = [
        _best_cells([record[field] for record in records], highest=highest)
        for _title, field, highest, _fmt in _TEST_COLUMNS
    ]

    body = []
    for index, (arm, record) in enumerate(zip(tests.arms["arm"], records, strict=True)):
        cells = [f"<td>{_skill_label(arm)}</td>"]
        for column, (_title, field, _highest, fmt) in enumerate(_TEST_COLUMNS):
            value = record[field]
            text = "&mdash;" if value is None else fmt(value)
            cells.append(f"<td><strong>{text}</strong></td>" if index in best[column] else f"<td>{text}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    own = "".join(f'<th rowspan="2">{title}</th>' for title, *_ in _TEST_COLUMNS[:_ARM_COLUMNS])
    versus = "".join(f"<th>{title}</th>" for title, *_ in _TEST_COLUMNS[_ARM_COLUMNS:])
    span = len(_TEST_COLUMNS) - _ARM_COLUMNS
    return f"""<table>
<thead>
<tr><th rowspan="2">Skill</th>{own}<th colspan="{span}">Compared with {_skill_label(tests.baseline)}</th></tr>
<tr>{versus}</tr>
</thead>
<tbody>{"".join(body)}</tbody>
</table>"""


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

    A skill arm that mostly failed to load the skill under test is not measuring it; a
    noskill run that did load it is not a clean baseline. Either makes the comparison a lie,
    so it is surfaced in the report rather than left in the transcripts. Other skills the
    agent may reach — the ones bundled with the CLI — do not count either way.

    Occasional misses are normal and the per-run table already marks them, so a skill arm is
    only flagged once fewer than half its runs loaded the skill.
    """
    notes = []
    if "skill_loaded" not in df.columns:
        return notes
    for arm in _arms_in_order(df):
        group = df[df["arm"] == arm]
        loaded = int(_loaded_flags(group).sum())
        if arm == NOSKILL_ARM:
            if loaded:
                notes.append(f"baseline arm '{arm}' loaded the skill under test in {loaded}/{len(group)} runs")
        elif loaded * 2 < len(group):
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
    tradeoff_uri = figure_data_uri(tradeoff_figure(df, colors=colors))
    tests = skill_tests(df)
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
<li><a href="#overview">Overview</a><ul><li><a href="#tradeoff">Cost vs. success</a></li>
<li><a href="#dominance">Is the difference real?</a></li></ul></li>
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
<p class="task-desc">Per-run means over test runs; error bars are standard errors.</p>
<figure><img alt="Success rate, tokens, cost and time per skill" src="{overview_uri}"></figure>
<h3 id="tradeoff">Cost vs. success</h3>
<p class="task-desc">One mark per model and skill version, where colour is the model and shape is
 the skill version. The grey mark pools every model for that version, with standard errors on
 both axes. The staircase is the Pareto frontier: anything below and to the right of it is
 dominated, because something on the line costs less <em>and</em> succeeds more.</p>
<figure><img alt="Cost per run against success rate, by model and skill version" src="{tradeoff_uri}"></figure>
<h3 id="dominance">Is the difference real?</h3>
<p class="task-desc">A version <em>dominates</em> the baseline when it is both cheaper and more
 accurate by more than chance. The two axes are tested separately and the weaker of the two
 p-values is reported, so a version cannot pass on price while being worse at the task.
 Improving both numbers is not enough on its own: a gain the resampling cannot tell apart from
 noise leaves the Holm p high, which says the evidence is thin rather than that the version is
 bad. Best value in each column is <strong>bold</strong>, highest for the rates and lowest for
 cost and for the p-values. <em>On frontier</em> is how often an arm holds the Pareto frontier
 across resamples. Resampling is over the {tests.n_clusters} test tasks, paired across arms, so the
 unit is the task rather than the run.</p>
{_tests_table_html(tests)}
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
    data = df.drop(columns=internal)
    if "skill_loaded" in data.columns:
        # The HTML table keeps '?' for a run whose transcript could not be read, but the CSV
        # is for counting: every row is True or False, so summing the column means what it
        # looks like — the same reading the skill-loaded figure takes.
        data["skill_loaded"] = _loaded_flags(data)
    data.to_csv(data_path, index=False)

    html_text = render_report(
        df, out_path.parent, tasks, data_href=data_path.name, skills_root=skills_root, palette=palette
    )
    out_path.write_text(html_text)
    return Report(html=html_text, results=df)
