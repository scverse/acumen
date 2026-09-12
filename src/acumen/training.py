"""The training curve: assemble per-epoch metrics from the runs tree, and the early-stopping rule.

``acumen fit`` runs many epochs; this module turns the accumulated ``runs/`` tree into one row per
epoch — one per produced skill version — carrying mean success and mean cost, overall and per
model, for the train arm the epoch learned from and the valid arm it produced. It also holds the
pure early-stopping rule (patience over validation mean success).

Everything here is rebuilt from disk each epoch rather than accumulated in memory, so a resumed
``fit`` yields exactly the same table. Each version ``vN`` is epoch N: the *valid* arm is
``skill_vN`` (what the epoch produced), and the *train* arm is ``skill_v(N-1)`` — or ``noskill``
for ``v1`` — the arm that epoch learned from.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from acumen.config import Config
from acumen.report import load_results

#: Matches a skill arm directory name (``skill_v1``, ``skill_v2``, …).
_SKILL_ARM = re.compile(r"^skill_v(\d+)$")


@dataclass(frozen=True)
class EpochRow:
    """One epoch's metrics: the arm it learned from (train) and the version it produced (valid)."""

    epoch: int
    version: str  # produced version, e.g. "v1"
    parent: str  # arm learned from: "noskill" or "v<N-1>"
    n_train: int
    n_valid: int
    #: Mean success (0..1) across all models; ``nan`` when the arm/split has no runs.
    train_success: float
    valid_success: float
    #: Mean cost per run (USD); ``nan`` when nothing priced or no runs.
    train_cost: float
    valid_cost: float
    #: Per-model means, keyed by model id; a model with no runs maps to ``nan``.
    train_success_by_model: dict[str, float]
    valid_success_by_model: dict[str, float]
    train_cost_by_model: dict[str, float]
    valid_cost_by_model: dict[str, float]


def _mean(series: pd.Series) -> float:
    """Mean of a numeric series, ``nan`` when empty."""
    return float(series.mean()) if len(series) else math.nan


def _mean_cost(frame: pd.DataFrame) -> float:
    """Mean cost per run, over the runs that have a price; ``nan`` when none do."""
    priced = frame["cost_usd"].dropna()
    return float(priced.mean()) if len(priced) else math.nan


def build_training_rows(runs_root: Path, cfg: Config) -> list[EpochRow]:
    """Build the per-epoch training curve from the runs tree.

    One row per produced version present in ``runs/`` (a ``skill_vN`` arm with runs), in version
    order. Per-model columns cover ``cfg.models`` so the table's shape is stable across epochs even
    before a model has any runs for a given arm.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    cfg
        The pass config; ``cfg.models`` fixes the per-model column set and order.

    Returns
    -------
    The rows, epoch 1 first. Empty if no produced version has runs yet.
    """
    df = load_results(runs_root)
    models = list(cfg.models)

    arms: list[tuple[int, str]] = []
    for arm in df["arm"].unique():
        match = _SKILL_ARM.match(str(arm))
        if match:
            arms.append((int(match.group(1)), str(arm)))
    arms.sort()

    def by_model(frame: pd.DataFrame, column: str) -> dict[str, float]:
        return {model: _mean(frame[frame["model"] == model][column]) for model in models}

    def cost_by_model(frame: pd.DataFrame) -> dict[str, float]:
        return {model: _mean_cost(frame[frame["model"] == model]) for model in models}

    rows: list[EpochRow] = []
    for index, (number, arm) in enumerate(arms):
        parent_arm = "noskill" if index == 0 else arms[index - 1][1]
        parent_label = "noskill" if index == 0 else f"v{arms[index - 1][0]}"
        train = df[(df["arm"] == parent_arm) & (df["split"] == "train")]
        valid = df[(df["arm"] == arm) & (df["split"] == "valid")]
        rows.append(
            EpochRow(
                epoch=index + 1,
                version=f"v{number}",
                parent=parent_label,
                n_train=len(train),
                n_valid=len(valid),
                train_success=_mean(train["success"]),
                valid_success=_mean(valid["success"]),
                train_cost=_mean_cost(train),
                valid_cost=_mean_cost(valid),
                train_success_by_model=by_model(train, "success"),
                valid_success_by_model=by_model(valid, "success"),
                train_cost_by_model=cost_by_model(train),
                valid_cost_by_model=cost_by_model(valid),
            )
        )
    return rows


def _cell(value: float) -> str:
    """Format a float for CSV: a plain number, or empty for ``nan`` (never a bogus 0)."""
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.6g}"


def write_training_csv(rows: list[EpochRow], path: Path) -> None:
    """Write the training curve to ``path`` as wide CSV, one row per epoch.

    Columns: ``epoch, version, parent, train_success, valid_success, train_cost, valid_cost,
    n_train, n_valid`` then, per model, ``train_success__<model>``, ``valid_success__<model>``,
    ``train_cost__<model>``, ``valid_cost__<model>``. Empty cells (not ``0``) mark a missing value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    models: list[str] = list(rows[0].train_success_by_model) if rows else []

    header = [
        "epoch",
        "version",
        "parent",
        "train_success",
        "valid_success",
        "train_cost",
        "valid_cost",
        "n_train",
        "n_valid",
    ]
    for model in models:
        header += [f"train_success__{model}", f"valid_success__{model}", f"train_cost__{model}", f"valid_cost__{model}"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            record = [
                row.epoch,
                row.version,
                row.parent,
                _cell(row.train_success),
                _cell(row.valid_success),
                _cell(row.train_cost),
                _cell(row.valid_cost),
                row.n_train,
                row.n_valid,
            ]
            for model in models:
                record += [
                    _cell(row.train_success_by_model.get(model, math.nan)),
                    _cell(row.valid_success_by_model.get(model, math.nan)),
                    _cell(row.train_cost_by_model.get(model, math.nan)),
                    _cell(row.valid_cost_by_model.get(model, math.nan)),
                ]
            writer.writerow(record)


# ── Early stopping ───────────────────────────────────────────────────────────────────────


def epochs_since_best(scores: list[float]) -> int:
    """How many epochs have passed since the best validation score, counting from its first hit.

    The best is the *first* version achieving the maximum (strict ``>`` — a later tie does not
    reset it), matching the rule that a new version must strictly beat the best-so-far to count as
    improvement. ``nan`` scores are ignored when finding the best but still advance the count.
    """
    best_index: int | None = None
    best_value: float | None = None
    for index, score in enumerate(scores):
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue
        if best_value is None or score > best_value:
            best_value, best_index = score, index
    if best_index is None:
        return 0
    return (len(scores) - 1) - best_index


def patience_exhausted(scores: list[float], patience: int) -> bool:
    """Whether validation has failed to improve for ``patience`` consecutive epochs."""
    return epochs_since_best(scores) >= patience


def best_version(rows: list[EpochRow]) -> str | None:
    """The version with the highest validation mean success (first on a tie), or ``None``."""
    best: str | None = None
    best_value: float | None = None
    for row in rows:
        score = row.valid_success
        if isinstance(score, float) and math.isnan(score):
            continue
        if best_value is None or score > best_value:
            best_value, best = score, row.version
    return best
