"""Tests for baseline difficulty screening — pure aggregation over the run tree — and headroom selection."""

from __future__ import annotations

import json
from pathlib import Path

from acumen.difficulty import FLAKY, HARD, SOLVED, Difficulty, screen, select_headroom
from acumen.paths import RESULT_FILE, RunKey, run_dir
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"
STRONG = "claude-sonnet-5"


def _task(task_id: str) -> Task:
    return Task(id=task_id, train=TaskSplit("p", "A"), test=TaskSplit("p", "B"))


def _write(runs: Path, task_id: str, split: str, rep: int, success: bool, model: str = MODEL) -> None:
    key = RunKey(arm="noskill", split=split, model=model, task_id=task_id, rep=rep)
    d = run_dir(runs, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / RESULT_FILE).write_text(json.dumps({"success": success, "model": model}))


def test_difficulty_strata() -> None:
    assert Difficulty("t", "test", 0, 3).stratum == HARD
    assert Difficulty("t", "test", 3, 3).stratum == SOLVED
    assert Difficulty("t", "test", 1, 3).stratum == FLAKY
    assert Difficulty("t", "test", 0, 3).has_headroom
    assert Difficulty("t", "test", 1, 3).has_headroom
    assert not Difficulty("t", "test", 3, 3).has_headroom
    assert Difficulty("t", "test", 1, 2).pass_rate == 0.5


def test_screen_aggregates_baseline_runs(tmp_path: Path) -> None:
    tasks = [_task("easy"), _task("hard"), _task("flaky"), _task("stale_ignored")]
    # easy: baseline passes both reps; hard: fails; flaky: 1 of 2.
    _write(tmp_path, "easy", "test", 1, True)
    _write(tmp_path, "easy", "test", 2, True)
    _write(tmp_path, "hard", "test", 1, False)
    _write(tmp_path, "flaky", "test", 1, True)
    _write(tmp_path, "flaky", "test", 2, False)
    # a run whose task is not in the list is stale and ignored
    _write(tmp_path, "gone", "test", 1, False)

    diffs = {d.task_id: d for d in screen(tmp_path, tasks, splits=["test"])}

    assert set(diffs) == {"easy", "hard", "flaky"}  # 'gone' filtered out
    assert diffs["easy"].stratum == SOLVED and not diffs["easy"].has_headroom
    assert diffs["hard"].stratum == HARD and diffs["hard"].has_headroom
    assert diffs["flaky"].passed == 1 and diffs["flaky"].total == 2 and diffs["flaky"].has_headroom
    assert all(d.model is None for d in diffs.values())  # pooled readings carry no model


def test_screen_empty_when_no_runs(tmp_path: Path) -> None:
    assert screen(tmp_path, [_task("t")], splits=["test"]) == []


def test_screen_by_model_keeps_reference_models_apart(tmp_path: Path) -> None:
    """A task hard for one model and trivial for another is not merely 'flaky' — it is both."""
    tasks = [_task("t")]
    _write(tmp_path, "t", "test", 1, False, model=MODEL)
    _write(tmp_path, "t", "test", 1, True, model=STRONG)

    pooled = screen(tmp_path, tasks, splits=["test"])
    assert [d.stratum for d in pooled] == [FLAKY]  # pooling hides the model dependence

    by_model = {d.model: d for d in screen(tmp_path, tasks, splits=["test"], by_model=True)}
    assert by_model[MODEL].stratum == HARD and by_model[MODEL].has_headroom
    assert by_model[STRONG].stratum == SOLVED and not by_model[STRONG].has_headroom


def test_select_headroom_keeps_only_tasks_the_baseline_fails_for_a_configured_model(tmp_path: Path) -> None:
    tasks = [_task("solved"), _task("hard"), _task("never"), _task("hard_for_strong_only")]
    _write(tmp_path, "solved", "test", 1, True)
    _write(tmp_path, "hard", "test", 1, False)
    _write(tmp_path, "hard", "train", 1, True)  # train headroom is irrelevant to the held-out score
    _write(tmp_path, "hard_for_strong_only", "test", 1, True, model=MODEL)
    _write(tmp_path, "hard_for_strong_only", "test", 1, False, model=STRONG)
    diffs = screen(tmp_path, tasks, by_model=True)

    sel = select_headroom(diffs, tasks, split="test", models=[MODEL])
    assert [t.id for t in sel.selected] == ["hard"]
    assert sel.solved == ["solved", "hard_for_strong_only"]  # STRONG's reading is not asked about
    assert sel.unscreened == ["never"]

    # Judged against both models, headroom for ANY of them is enough — the pooled score can move.
    both = select_headroom(diffs, tasks, split="test", models=[MODEL, STRONG])
    assert [t.id for t in both.selected] == ["hard", "hard_for_strong_only"]

    # With no model restriction, every reading counts; order follows the caller's task list.
    any_model = select_headroom(diffs, tasks, split="test")
    assert [t.id for t in any_model.selected] == ["hard", "hard_for_strong_only"]
