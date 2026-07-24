"""Scaffold starter ``config.yaml`` and ``tasks.yaml`` files for ``acumen init``.

The templates are package-agnostic placeholders, kept heavily commented so a new user can
read what each field does and fill in the blanks rather than starting from an empty file.
The placeholders (``OWNER/REPO``, the ``REPLACE_ME`` answers) are meant to be edited before
the first run.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_FILENAME = "config.yaml"
TASKS_FILENAME = "tasks.yaml"

#: An annotated config with placeholder target/skill fields for the user to fill in.
CONFIG_TEMPLATE = """\
# acumen config — the target package and how to benchmark against it.
# Every field below except `repo` has a default; delete a line to take it.

repo: https://github.com/OWNER/REPO   # GitHub URL, or a local path to an installable package
ref: main                             # branch, tag, or commit (ignored for local paths)
extras: []                            # optional pip extras, e.g. [dev, test]
python: "3.12"                        # interpreter for the target's throwaway venv

models: [claude-opus-4-8, claude-sonnet-5, claude-haiku-4-5-20251001]   # benchmark models; a pass is models x tasks x reps x splits
n_replicates: 3                       # runs per (model, task, split) cell
max_concurrency: 4                    # simultaneous agents

max_turns: 40                         # per-agent turn cap; a breach is recorded as a failure
max_usd: 3.0                          # budget cap (USD) for benchmark agents; draft/improve/ship/tasks are unbounded

draft_model: claude-opus-4-8          # model that writes the first skill (reads the source)
improve_model: claude-opus-4-8        # model that improves a skill from its train results
tasks_model: claude-opus-4-8          # model for `acumen tasks` (generates this file); runs code
ship_model: claude-opus-4-8           # model for `acumen ship` (wires a multi-framework install-skills script)

# skill_name: your_skill              # name for the built skill; defaults to the repo's name
"""

#: A tasks file with one placeholder task, showing the required shape. Each task needs a
#: stable `id` and a train/test pair; the answer is graded by exact string match after
#: `strip()`.
TASKS_TEMPLATE = """\
# acumen tasks — what the agent is asked to do, and the answer it is graded against.
#
# Each task has a stable `id` (used in run paths — renaming it orphans old runs) and a
# train/test pair. The improver only ever sees train results; test is the held-out measure
# of whether a skill actually helps. Answers are compared by EXACT string match after
# strip(), so keep them to a single unambiguous token.
#
# Do NOT name the target package in a prompt. The agent is already told, before every task,
# that your package (from `config.yaml`) is installed and is the one to use — so "using
# <package>, ..." is redundant. Describe only the analysis: the goal, the input, and exactly
# what to report. (`acumen tasks` follows the same rule when it generates this file.)
#
# A good task is one your target package solves and the baseline gets wrong — that is where
# a skill has room to help. Replace the placeholders below with your own, or run
# `acumen tasks` to generate this file automatically from the target package.
#
# Optional per-task overrides: `max_turns`, `max_usd`, `model`.

tasks:
  - id: example_task
    train:
      prompt: >-
        Describe, concretely, the analysis the agent should carry out on the training input
        — the data, the goal, and exactly what to report. Do not name the package (it is
        provided). End by asking for only the final answer, so it grades as a single token.
      answer: REPLACE_ME_TRAIN
    test:
      prompt: >-
        The same kind of analysis on a different, held-out input. This split is what measures
        whether the skill generalizes; the improver never sees its results.
      answer: REPLACE_ME_TEST
"""


class InitError(ValueError):
    """Raised when scaffolding cannot proceed — e.g. files exist and ``force`` is unset."""


def scaffold(directory: Path, *, force: bool = False) -> list[Path]:
    """Write starter ``config.yaml`` and ``tasks.yaml`` into ``directory``.

    Parameters
    ----------
    directory
        Where to scaffold. Created if it does not exist.
    force
        Overwrite existing files. Without it, an existing ``config.yaml`` or ``tasks.yaml``
        is left untouched and an error is raised, so a real project is never clobbered.

    Returns
    -------
    The paths written, in the order they were written.

    Raises
    ------
    InitError
        If a target file exists and ``force`` is not set.
    """
    directory.mkdir(parents=True, exist_ok=True)
    targets = {directory / CONFIG_FILENAME: CONFIG_TEMPLATE, directory / TASKS_FILENAME: TASKS_TEMPLATE}

    if not force:
        existing = [p for p in targets if p.exists()]
        if existing:
            names = ", ".join(p.name for p in existing)
            verb = "exists" if len(existing) == 1 else "exist"
            raise InitError(f"{names} already {verb} in {directory} — pass --force to overwrite")

    written = []
    for path, template in targets.items():
        path.write_text(template)
        written.append(path)
    return written
