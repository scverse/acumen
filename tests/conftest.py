"""Shared fixtures for the acumen test suite.

The suite is deliberately small and offline: it covers the CLI's main commands and the
pure helpers underneath them, and never spawns an agent. Anything that would call a
Claude or Codex agent (``draft``, ``improve``, ``tasks``, ``ship``) is only exercised up to
the pre-flight checks that run *before* the agent — those are the parts a unit test can
honestly assert.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acumen.paths import RESULT_FILE, RunKey, run_dir

CONFIG_YAML = """\
repo: target
ref: main
models: [claude-haiku-4-5-20251001]
n_replicates: 1
max_concurrency: 2
skill_name: target
"""

TASKS_YAML = """\
tasks:
  - id: example_task
    train:
      prompt: Do the training analysis and report the single symbol.
      answer: TRAIN_ANSWER
    test:
      prompt: Do the same analysis on the held-out input.
      answer: TEST_ANSWER
"""

SKILL_MD = """\
---
name: target
description: How to use the target package for its main analyses.
---

# target

Call `target.run()` first.
"""

MODEL = "claude-haiku-4-5-20251001"


@pytest.fixture
def target_pkg(tmp_path: Path) -> Path:
    """A minimal local package, so a config's local ``repo:`` path resolves."""
    pkg = tmp_path / "target"
    pkg.mkdir()
    (pkg / "pyproject.toml").write_text('[project]\nname = "target"\nversion = "0.1"\n')
    return pkg


@pytest.fixture
def project(tmp_path: Path, target_pkg: Path) -> Path:
    """A scratch acumen project: ``config.yaml`` + ``tasks.yaml`` for a local target."""
    (tmp_path / "config.yaml").write_text(CONFIG_YAML)
    (tmp_path / "tasks.yaml").write_text(TASKS_YAML)
    return tmp_path


@pytest.fixture
def skills_root(project: Path) -> Path:
    """A skill tree holding one valid version, ``v1``."""
    directory = project / "skills" / "v1"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(SKILL_MD)
    (directory / "meta.json").write_text(json.dumps({"version": 1, "parent": None, "rationale": "initial draft"}))
    return project / "skills"


def write_result(runs_root: Path, key: RunKey, **overrides) -> Path:
    """Write a plausible ``result.json`` for one run, marking that run complete.

    Only the fields the report reads are populated — enough to exercise aggregation and
    the resume check without standing up a real agent run.
    """
    directory = run_dir(runs_root, key)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": key.task_id,
        "split": key.split,
        "skill": key.skill,
        "arm": key.arm,
        "model": key.model,
        "rep": key.rep,
        "success": True,
        "reason": "ok",
        "turns": 7,
        "cost_usd": 0.12,
        "input_tokens": 4000,
        "output_tokens": 300,
        "duration_s": 42.0,
        "answer": "TRAIN_ANSWER",
        "expected": "TRAIN_ANSWER",
        "pkg_version": "target 0.1",
        "commit": "abc1234",
        "skill_hash": None,
        "skill_name": None,
        "skill_loaded": key.arm != "noskill",
        "sdk_version": "0.2.120",
    }
    payload.update(overrides)
    (directory / RESULT_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    return directory


@pytest.fixture
def model() -> str:
    """The model the fixture config benchmarks with — it appears in run paths."""
    return MODEL


@pytest.fixture
def make_result():
    """Expose :func:`write_result` to tests as a fixture."""
    return write_result


@pytest.fixture
def runs_root(project: Path) -> Path:
    """A run tree with one complete baseline run per split."""
    root = project / "runs"
    for split in ("train", "test"):
        write_result(root, RunKey(arm="noskill", split=split, model=MODEL, task_id="example_task", rep=1))
    return root
