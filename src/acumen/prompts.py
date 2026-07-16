"""Hardcoded prompts.

The harness preamble carries the entire grading scheme: grading is exact string match on
``answer.md`` (§2), so any stray word, header or code fence turns a correct run into a
recorded failure. It is therefore blunt, repeated, and shows worked good/bad examples —
see §10.1, which flags format noise as the highest risk in the project.
"""

from __future__ import annotations

from pathlib import Path

HARNESS_PREAMBLE = """\
You are completing one task in an automated benchmark. Your work is graded by a script,
not read by a human.

# Where you work

- Your working directory is `{sandbox}`. Work only there.
- Do NOT read or write any path outside `{sandbox}`.
- The target package (`{package}`) is already installed. Run Python with `{python}`,
  which is also `python` on your PATH. Do not create virtualenvs and do not install or
  upgrade packages.
- You have web access. Use it if it helps.

# What you must leave behind

Exactly two files in `{sandbox}`:

1. `script.py` — a runnable Python script that reproduces how you got your answer.
2. `answer.md` — your final answer, and NOTHING else.

# The format of answer.md — read this twice

`answer.md` is compared to the expected answer by EXACT STRING MATCH, after stripping
leading and trailing whitespace. The comparison is case-sensitive. There is no human and
no judge: a single extra word, character, or line makes a correct answer score as WRONG.

`answer.md` must contain ONLY the answer itself:

- NO explanation, reasoning, or commentary.
- NO markdown headers (`#`), bold (`**`), bullets, or code fences (```).
- NO quotes around the answer.
- NO label such as "Answer:" or "The answer is".
- NO trailing period, and no trailing notes.

Worked example. If the task were "Which transcription factor is most active?" and the
correct answer were SPI1, then the ENTIRE contents of `answer.md` must be:

SPI1

Every one of these is scored as WRONG, even though each contains the right answer:

- `The most active TF is SPI1.`
- `**SPI1**`
- `# Answer\\nSPI1`
- `"SPI1"`
- ```` ```\\nSPI1\\n``` ````
- `SPI1 (adjusted p = 0.001)`

If the task asks for a number, write only the number, with exactly the precision the
task asks for. If it asks for several items, follow the task's stated separator and
order exactly.

# Task

{task}

# Reminder

When you are done, `{sandbox}/answer.md` must contain your answer and nothing else — no
prose, no formatting, no code fences. `{sandbox}/script.py` must reproduce it.
"""


def benchmark_prompt(task_prompt: str, *, sandbox: Path, python: Path, package: str) -> str:
    """Build the full prompt for one benchmark run.

    Parameters
    ----------
    task_prompt
        The task's own prompt, from ``tasks.yaml``.
    sandbox
        The agent's working directory.
    python
        The interpreter with the target package installed.
    package
        The target package name, named so the agent doesn't hunt for it.

    Returns
    -------
    The harness preamble with the task embedded.
    """
    return HARNESS_PREAMBLE.format(sandbox=sandbox, python=python, package=package, task=task_prompt.strip())
