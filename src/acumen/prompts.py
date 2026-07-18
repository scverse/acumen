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


IMPROVE_PROMPT = """\
You are improving a Claude Skill for the Python package `{package}` (version {version}).

A skill is documentation written for an agent, not a human. Its only purpose is to make an
agent that has never used `{package}` succeed at real tasks with it on the first try.

You are producing version {new_version} — an improvement of version {parent_version}.

# What you can read

- `{skill_dir}` — the current skill ({parent_version}). This directory has been pre-filled
  with a copy of it. EDIT THESE FILES IN PLACE; what they contain when you finish becomes
  version {new_version}.
- `{train_dir}` — evidence from benchmarking the current skill on the TRAIN split. Read
  `{train_dir}/SUMMARY.md` first. For each run you will find the task prompt, the expected
  answer, the answer the agent actually gave, whether it passed, the `script.py` it wrote,
  and its full transcript. This is your only signal about what the skill gets right and
  what it gets wrong.
- `{package}` is installed; run `{python}` (also `python` on your PATH) to verify any
  claim about the API before you write it down. Do not install or upgrade packages.
- You have web access if the published docs help.

# What you are NOT allowed to see

You are optimising against a TRAIN split. A separate, held-out TEST split is what measures
whether your changes actually generalise rather than memorising these particular tasks. You
must not see the test split, and any tool call that reaches test results will be BLOCKED.
Do not attempt it — reaching test data would invalidate the whole benchmark.

# What you must write

1. Edit the skill in place under `{skill_dir}`. When you finish, `{skill_dir}/SKILL.md`
   must still begin with YAML frontmatter whose `name` is exactly `{skill_name}` and whose
   `description` is an honest one-sentence statement of what the skill covers and when to
   use it.

2. `{rationale_path}` — one short paragraph stating WHAT you changed and WHY, grounded in
   the train evidence. Write it here, OUTSIDE the skill directory. Do not put the rationale
   inside `{skill_dir}`.

# How to improve it

- **Fix what the evidence shows is broken.** Read the failing runs first. A failure is
  either the skill steering the agent wrong, or the skill saying nothing where it should
  have. Address the cause you can see in the transcript, not one you imagine.
- **Generalise — never overfit.** NEVER name a specific dataset, parameter value, column,
  expected answer, or task from the train runs in the skill. The skill must help on tasks
  you have not seen. Guidance that enumerates these particular cases is cheating and will
  fail the test split.
- **Prefer removing text over adding it.** A shorter skill that an agent reads and follows
  beats a longer one it skims. Every token is paid on every task, including the ones where
  the skill is irrelevant. If a passage did not change any outcome, cut it.
- **Verify before you write.** Do not assert a default, a return type, or where an output
  lands without confirming it in the installed package or the docs.
- **No hedging.** Say what to do.

When you are done, `{skill_dir}/SKILL.md` exists and starts with the frontmatter above, and
`{rationale_path}` contains your rationale.
"""


TASKGEN_PROMPT = """\
You are writing a benchmark of real analysis tasks for a Python package (`{package}`). Each
task states a GOAL a user has, in plain language, plus the single answer a correct analysis
produces. The benchmark measures whether an AI agent can reach that goal with the package on
its own — so a task gives the objective and almost nothing else.

# What you can read

- The package's source is at `{src}`. Read it — the source, docstrings, examples, and above
  all its docs/tutorials/vignettes. This is the ground truth about what the package does.
- The package is installed; run `{python}` (also `python` on your PATH) to execute code. Do
  not create virtualenvs and do not install or upgrade packages — work with what is here.
- You have web access if the published docs or tutorials help.

# Ignore any existing skills or agent instructions — deliberately hidden

The package may ship skills or agent-instruction files written for it (`SKILL.md`,
`.claude/skills/`, `CLAUDE.md`, `AGENTS.md`, `.cursor/`, Copilot instructions). These have
been stripped from the source above, and any attempt to reach them — or the original
unfiltered checkout — is BLOCKED. This is on purpose: reading pre-written guidance would bias
which analyses you pick and how you phrase them, and this benchmark must be independent of it.

# Cover every tutorial — enumerate them FIRST

Before writing anything, find EVERY tutorial / vignette / worked example the package publishes:
its documentation gallery, an `examples/`, `tutorials/`, or `docs/` directory, notebooks,
README walkthroughs. List them all. You will write AT LEAST ONE task per tutorial — do not stop
after a handful, and do not cover only the easy ones. A published tutorial is a real analysis
someone thought worth doing; that is exactly the unit of work this benchmark should measure.
Only if the package has genuinely no tutorials should you infer analyses from its source.

# How to write a task — like a lazy human, not a manual

Each task's prompt is ONE short paragraph of ordinary English: the GOAL a user wants, and
nothing about how to reach it. Write it the way a busy analyst types a request into a terminal
— a sentence or two, a clear objective, minimal detail. The agent is supposed to work out the
"how" itself; that is what is being tested.

HARD rules for the prompt text:
- ONE paragraph. NO numbered steps, no procedure — state the goal, not a recipe.
- NO code: no function or method names as code, no call signatures, no argument names, no
  names of result containers or output fields.
- Do NOT name the package, and never mention a version. acumen adds which package to use when
  it runs the task — naming it here is redundant and repetitive.
- Do NOT describe the data — not its shape, columns, dtypes, or how it is stored. If the
  analysis uses a bundled dataset, just NAME it ("using the covid5k data") and stop there.
- Do NOT name the method/algorithm or give parameters — let the agent choose the approach. ONE
  exception: if a tutorial is fundamentally about a single named method, you may name that
  method in plain words; even then name only what is essential, and give a parameter only if
  that parameter is the whole point of the task.
- No worked example of the answer, and no hint toward it.

The one thing you MAY state precisely is the OUTPUT. End the paragraph by saying exactly what to
report and in what form, so the answer is unambiguous and gradeable — e.g. "give the gene
symbol", "report how many are left", "report the value rounded to two decimals". Keep the answer
small: a single name, category, count, or number.

Illustration (style only — invent tasks that fit the actual package):
- BAD (reads like a skill): "Load the toy data with `dc.ds.toy()`. Run `dc.mt.ulm(adata, net,
  tmin=3)`; scores land in `adata.obsm['score_ulm']`. Take rows where group == 'A', average per
  source, and report the top column."
- GOOD (a lazy goal): "Using the pbmc3k data, find which transcription factor is most active in
  the monocytes. Give only the factor's symbol."

# Train and test variants

Give each task a train and a test variant of the SAME goal, differing only in the input or the
target it asks about (a different cell type, group, condition, or dataset). Two instances of one
analysis with two different correct answers — so a skill cannot pass by memorising one answer.

# Ground truth by execution

Get each answer by actually DOING the analysis in the venv with `{python}` and reading the real
result — never from a tutorial's printed output or the docs. Your scratch scripts stay in this
working directory and are discarded; only the tasks written to `{out}` are kept, so the
benchmarked agent must rederive everything from the goal alone. Before recording an answer,
confirm the goal has exactly ONE defensible answer: if a competent analyst could read the goal
two ways and get two results, tighten only the OUTPUT sentence (what to report, or its
precision) until one answer stands — never by adding back instructions.

# What you must write

Your working directory is `{out_dir}`. Write the tasks to `{out}` as YAML with exactly this
shape (the loader is strict — unknown keys are rejected):

tasks:
  - id: <short, unique, filesystem-safe: letters, digits, '.', '_', '-'>
    train:
      prompt: |
        <one-paragraph goal for the train variant>
      answer: "<the exact answer string the real train run produced>"
    test:
      prompt: |
        <one-paragraph goal for the test variant>
      answer: "<the exact answer string the real test run produced>"

`id` must be unique across all tasks. Both `train` and `test` are required, each with a
non-empty `prompt` and a non-empty `answer`. Do not add other keys unless you deliberately
want a per-task override (`max_turns`, `max_usd`, or `model` are the only ones allowed).

# Before you finish

- Every `answer` is the real output of a script you ran in the venv — not a guess, not lifted
  from docs.
- Every prompt is ONE paragraph: a goal in plain English, with no steps, no code, no package
  name, no version, no data description — only the goal and a precise statement of the output.
- You wrote at least one task per tutorial, and covered all of them.
- `{out}` exists and parses as the YAML above with at least one task.
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


def improve_prompt(
    *,
    package: str,
    version: str,
    python: Path,
    skill_dir: Path,
    train_dir: Path,
    rationale_path: Path,
    skill_name: str,
    parent_version: str,
    new_version: str,
) -> str:
    """Build the prompt for the improving agent.

    Unlike the drafter, the improver never sees the package source (§6) — it works from the
    current skill and the *train-split* evidence of how that skill performed. The test split
    is unreachable, enforced structurally (§7.1), not by this prompt.

    Parameters
    ----------
    package
        The target package name.
    version
        The installed package version, so any verification runs against what is installed.
    python
        The interpreter with the package installed, for checking claims.
    skill_dir
        The staging directory, pre-filled with a copy of the parent skill, that the agent
        edits in place to produce the new version.
    train_dir
        The directory of train-split evidence the agent reads (``SUMMARY.md`` + per-run
        material).
    rationale_path
        Where the agent writes its rationale — outside ``skill_dir`` so it never becomes
        skill content.
    skill_name
        The name the frontmatter must keep — ``config.skill_name``.
    parent_version, new_version
        The version being improved and the version being produced, e.g. ``v1`` -> ``v2``.

    Returns
    -------
    The improve prompt.
    """
    return IMPROVE_PROMPT.format(
        package=package,
        version=version,
        python=python,
        skill_dir=skill_dir,
        train_dir=train_dir,
        rationale_path=rationale_path,
        skill_name=skill_name,
        parent_version=parent_version,
        new_version=new_version,
    )


def taskgen_prompt(*, package: str, src: Path, python: Path, out: Path) -> str:
    """Build the prompt for the task-generation agent (M6).

    Like the drafter, the generator gets read access to the target's source (§6) — it must
    understand the API to design real analyses — plus the installed venv, since it obtains
    every ground-truth answer by *executing* the pipeline, never by reading doc output.

    The package name is passed only to orient the agent; the tasks it writes must NOT name the
    package or any version (acumen injects the package via the benchmark harness), so each task
    stays a generic, version-agnostic goal.

    Parameters
    ----------
    package
        The target package name, for the agent's orientation only.
    src
        The (filtered) package checkout, readable by this agent only.
    python
        The interpreter with the package installed, used to run pipelines for ground truth.
    out
        The ``tasks.yaml`` file the agent writes into its working directory.

    Returns
    -------
    The task-generation prompt.
    """
    return TASKGEN_PROMPT.format(package=package, src=src, python=python, out=out, out_dir=out.parent)


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
