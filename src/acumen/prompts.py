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


DRAFT_PROMPT = """\
You are writing a Claude Skill for the Python package `{package}` (version {version}).

A skill is documentation written for an agent, not for a human. Its only purpose is to
make an agent that has never used `{package}` succeed at real tasks with it on the first
try. It is not a tutorial, not a README, and not a sales pitch.

# What you can read

- The package's source is at `{src}`. Read it — the source, the docstrings, the examples,
  the docs directory, the tests. This is the ground truth about how the package behaves.
- You have web access if the published docs help.
- `{package}` is also installed; run `{python}` to check anything you are unsure about.
  Verify claims before you write them down.

# What you must write

Your working directory is `{out}`. Write:

1. `{out}/SKILL.md` — required. It must begin with YAML frontmatter, exactly:

---
name: {skill_name}
description: <one sentence: what this skill covers and when to use it>
---

   The `name` must be exactly `{skill_name}`.

   The `description` is load-bearing and must be HONEST. It is the only part of the skill
   an agent sees before deciding whether to open it, and it is the only thing that gets
   the skill loaded at the right moment. State what the skill covers and when it applies.
   Do not oversell it, and do not claim coverage the body does not deliver.

2. `{out}/references/*.md` — optional. Use these for detail that only some tasks need.

# How to write it

- **Write what is not guessable.** An agent already knows Python and can read a
  traceback. Spend your words on what it would get WRONG by guessing: non-obvious
  defaults, required preprocessing, the function that looks right but isn't, where
  results are written, argument shapes and orientation, footguns the API invites.
- **Generalize.** Write guidance that holds across tasks. Do not enumerate cases.
- **Progressive disclosure.** `SKILL.md` should be short and route to `references/` for
  depth. An agent pays for every token of it on every task, including the tasks where it
  is irrelevant. If `SKILL.md` is long, you are taxing every run.
- **Be concrete.** A correct short code example beats a paragraph of prose. Show the real
  call, with the arguments that matter.
- **Prefer removing text over adding it.** Anything that merely restates the obvious is
  worse than nothing: it costs tokens and buries the parts that matter.
- **No hedging.** Say what to do.

# Verify before you finish

Do not write claims you have not checked. If you assert a default value, a return type,
or where an output lands, confirm it in the source or by running `{python}`. A skill that
confidently states something false is worse than no skill at all — it will send an agent
in the wrong direction with full confidence.

When you are done, `{out}/SKILL.md` must exist and start with the frontmatter above.
"""


def draft_prompt(*, package: str, version: str, src: Path, python: Path, out: Path, skill_name: str) -> str:
    """Build the prompt for the drafting agent.

    Unlike a benchmark agent, the drafter gets read access to the target's source (§6) —
    it is writing documentation about the package, so it needs to see it.

    Parameters
    ----------
    package
        The target package name.
    version
        The installed version, so the skill describes what is actually installed.
    src
        The package checkout, readable by this agent only.
    python
        The interpreter with the package installed, for verifying claims.
    out
        The staging directory the agent writes the skill into.
    skill_name
        The name the frontmatter must declare — ``config.skill_name``.

    Returns
    -------
    The draft prompt.
    """
    return DRAFT_PROMPT.format(package=package, version=version, src=src, python=python, out=out, skill_name=skill_name)


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
