"""Unit tests for the pure parts the CLI sits on: grader, schemas, paths, skills.

Deliberately thin — one test per behaviour that would silently corrupt a benchmark if it
broke, not an exhaustive sweep of each validator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acumen.bench import build_matrix, pending
from acumen.config import ConfigError, derive_skill_name, load_config, parse_config
from acumen.grade import grade_answer, grade_run
from acumen.paths import RunKey, arm_name, is_complete, parse_run_dir, run_dir
from acumen.skills import SkillError, load_skill, skill_hash
from acumen.tasks import TaskError, load_tasks, parse_tasks

# --- grading ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected", "success", "reason"),
    [
        ("SPI1", "SPI1", True, "ok"),
        ("  SPI1\n", "SPI1", True, "ok"),  # graded after strip()
        ("STAT1", "SPI1", False, "wrong_answer"),
        ("**SPI1**", "SPI1", False, "format_error"),  # right content, forbidden formatting
        (None, "SPI1", False, "no_answer_file"),
    ],
)
def test_grade_answer(answer: str | None, expected: str, success: bool, reason: str) -> None:
    grade = grade_answer(answer, expected)
    assert (grade.success, grade.reason) == (success, reason)


def test_grade_run_reads_answer_md(tmp_path: Path) -> None:
    (tmp_path / "answer.md").write_text("SPI1\n")
    assert grade_run(tmp_path, "SPI1").success
    assert grade_run(tmp_path / "empty", "SPI1").reason == "no_answer_file"


# --- run paths -------------------------------------------------------------------------


def test_run_dir_round_trips(tmp_path: Path) -> None:
    key = RunKey(arm="skill_v2", split="test", model="claude-opus-4-8", task_id="tf_activity", rep=3)
    directory = run_dir(tmp_path, key)

    assert directory == tmp_path / "skill_v2/test/claude-opus-4-8/tf_activity/rep_3"
    assert parse_run_dir(tmp_path, directory) == key
    assert key.skill == "v2"


def test_arm_name() -> None:
    assert arm_name(None) == "noskill"
    assert arm_name("v1") == "skill_v1"


def test_is_complete_needs_a_result_file(tmp_path: Path) -> None:
    assert not is_complete(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    assert is_complete(tmp_path)


# --- config / tasks --------------------------------------------------------------------


def test_config_defaults_and_derived_skill_name() -> None:
    cfg = parse_config({"repo": "https://github.com/scverse/scanpy"})

    assert cfg.skill_name == "scanpy"
    assert cfg.ref == "main"
    assert cfg.n_replicates == 3
    assert not cfg.is_local
    assert derive_skill_name("git@github.com:scverse/scanpy.git") == "scanpy"


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config({"repo": "https://example.com/pkg", "modles": ["x"]})


def test_load_config_resolves_a_local_repo(project: Path) -> None:
    cfg = load_config(project / "config.yaml")

    assert cfg.is_local
    assert Path(cfg.repo) == (project / "target").resolve()


def test_load_tasks(project: Path) -> None:
    (task,) = load_tasks(project / "tasks.yaml")

    assert task.id == "example_task"
    assert task.split("train").answer == "TRAIN_ANSWER"
    assert task.split("test").prompt.startswith("Do the same")


def test_tasks_reject_duplicate_ids() -> None:
    entry = {"id": "dup", "train": {"prompt": "p", "answer": "a"}, "test": {"prompt": "p", "answer": "a"}}
    with pytest.raises(TaskError, match="duplicate task id"):
        parse_tasks({"tasks": [entry, dict(entry)]})


# --- matrix ----------------------------------------------------------------------------


def test_build_matrix_and_resume(project: Path, model: str, make_result) -> None:
    cfg = load_config(project / "config.yaml")
    tasks = load_tasks(project / "tasks.yaml")

    planned = build_matrix(cfg, tasks, skill="v1")
    assert len(planned) == 2  # 1 task x 2 splits x 1 model x 1 replicate
    assert {p.key.split for p in planned} == {"train", "test"}
    assert all(p.key.arm == "skill_v1" for p in planned)

    runs = project / "runs"
    make_result(runs, RunKey(arm="skill_v1", split="train", model=model, task_id="example_task", rep=1))
    assert [p.key.split for p in pending(planned, runs)] == ["test"]
    assert len(pending(planned, runs, resume=False)) == 2


# --- skills ----------------------------------------------------------------------------


def test_load_skill(skills_root: Path) -> None:
    skill = load_skill(skills_root, "v1", expect_name="target")

    assert (skill.version, skill.name, skill.number) == ("v1", "target", 1)
    assert skill.hash.startswith("sha256:")


def test_load_skill_rejects_a_name_mismatch(skills_root: Path) -> None:
    with pytest.raises(SkillError, match="config.skill_name"):
        load_skill(skills_root, "v1", expect_name="something_else")


def test_skill_hash_ignores_meta_json(skills_root: Path) -> None:
    """``meta.json`` carries the hash, so hashing it would be circular."""
    directory = skills_root / "v1"
    before = skill_hash(directory)

    (directory / "meta.json").write_text('{"rationale": "rewritten"}')
    assert skill_hash(directory) == before

    (directory / "SKILL.md").write_text("---\nname: target\ndescription: d\n---\nnew body\n")
    assert skill_hash(directory) != before
