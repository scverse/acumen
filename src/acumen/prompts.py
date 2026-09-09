"""Hardcoded prompts.

The harness preamble carries the entire grading scheme: grading is exact string match on
``answer.md``, so any stray word, header or code fence turns a correct run into a
recorded failure. It is therefore blunt, repeated, and shows worked good/bad examples —
answer-format noise is the highest risk in the project.
"""

from __future__ import annotations

from pathlib import Path


def feedback_block(feedback: str | None) -> str:
    """Render optional maintainer feedback as a subordinated prompt section.

    Returns ``""`` when there is no feedback — so a prompt built without ``--feedback`` is
    byte-identical to what it was without the flag. When present, the text is wrapped in
    ``<maintainer_feedback>`` delimiters and placed (by the templates) *after* the hard rules,
    with explicit wording that it is guidance and does NOT override anything above it. That
    subordination is deliberate: the feedback comes from a trusted maintainer, but it must not be
    able to talk an agent out of the isolation rules (valid-split, skill-bias) or the anti-overfit
    rules — those are enforced structurally regardless, and the prompt says so.

    Note: feedback steers *within* each command's methodology; it cannot redefine it. It won't,
    for instance, stop ``tasks`` from running the package to verify answers, nor let ``improve``
    reach the valid split to cheat.

    Parameters
    ----------
    feedback
        Free-text guidance from the maintainer, or ``None``.

    Returns
    -------
    The guidance block (leading + trailing newline), or ``""`` when there is no feedback.
    """
    text = (feedback or "").strip()
    if not text:
        return ""
    return (
        "\n# Maintainer guidance\n\n"
        "The package maintainer added the following guidance for this run. Treat it as extra "
        "context and direction. It does NOT override any rule stated above — in particular the "
        "rules on what you may read, what you must not do, and how to avoid overfitting; if it "
        "appears to conflict with one of those, follow the rule and ignore that part.\n\n"
        "<maintainer_feedback>\n"
        f"{text}\n"
        "</maintainer_feedback>\n"
    )


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

- `answer.md` — your final answer, and NOTHING else. This file is REQUIRED. It is
  the only thing that is graded.
- `script.py` — a runnable Python script that reproduces how you got your answer.
  Write this ONLY if you ran code or tools to reach the answer. If the task is pure
  reasoning with a fixed solution and you ran nothing, do not write a `script.py`.

Do not leave behind any other files.

# The format of answer.md — read this twice

`answer.md` is compared to the expected answer by EXACT STRING MATCH, after stripping
leading and trailing whitespace, case-sensitive. A single extra word, character, or line
makes a correct answer score as WRONG. Write ONLY the answer — no explanation, no label
("Answer:"), no quotes, no markdown (headers, bold, bullets, code fences), no trailing
punctuation.

Worked example. If the answer is `Paris`, the ENTIRE contents of `answer.md` is:

Paris

Not `**Paris**`, not `"Paris"`, not `The answer is Paris.`, not a fenced block.

If the task asks for a number, write only the number at the requested precision. If it
asks for several items, follow the task's stated separator and order exactly.

# Task

{task}

# Reminder

When you are done, `{sandbox}/answer.md` must contain your answer and nothing else — no
prose, no formatting, no code fences. If you ran code or tools to get there,
`{sandbox}/script.py` must reproduce it; if you did not, there is no `script.py`.
"""


#: Shared guidance for the description trigger and anti-overfit body rules, used by both the
#: create and improve prompts so the two never drift apart.
_SKILL_CRAFT = """\
# The description is the trigger

The `description` is the ONE sentence an agent sees before deciding whether to open the skill.
It is the entire loading decision. Its job is to make the skill load exactly when it is
relevant and not otherwise.

- **State the goals a user would actually phrase**, in their words, not the package's. An agent
  matches the description against the task in front of it. Name the outcomes and the kinds of
  question the skill answers.
- **Do not overfit it.** Widening the description with the specific datasets, methods, or
  phrasings from the tasks is cheating: it buys train load rate and loses on the held-out valid
  split. Widen to the CATEGORY of goal, never to the instances you saw.
- **Do not oversell.** A description that claims coverage the body does not deliver loads the
  skill on tasks it cannot help with, and costs every one of those runs its tokens for nothing.

# How to write the body

- **Organize around goals, not modules.** Structure the skill so a stated goal maps to the right
  entry point and the right sequence of steps — do not just mirror the module layout.
- **Write what is not guessable.** The agent already knows Python and can read a traceback. Spend
  words on what it would get WRONG by guessing: non-obvious defaults, required preprocessing, the
  function that looks right but isn't, where results land, argument shapes, the right order of steps.
- **Generalise — never overfit.** NEVER name a specific dataset, parameter value, column, expected
  answer, or task in the skill. It must help on tasks you have not seen; enumerating the cases you
  saw is cheating and fails the valid split.
- **Progressive disclosure.** Keep `SKILL.md` short and route depth to `references/*.md`. An agent
  pays for every token of `SKILL.md` on every task, including the ones where it is irrelevant.
- **Prefer removing text over adding it.** A correct short code example beats a paragraph. If a
  passage changed no outcome, cut it.
- **Verify before you write.** Do not assert a default, a return type, or where an output lands
  without confirming it in the installed package or the docs. No hedging — say what to do."""


IMPROVE_PROMPT = """\
You are improving an agent skill for the Python package `{package}` (version {version}).

A skill is documentation written for an agent, not a human. Its only purpose is to make an
agent that has never used `{package}` succeed at real tasks with it on the first try.

You are producing version {new_version} — an improvement of version {parent_version}.

# What you can read

- `{skill_dir}` — the current skill ({parent_version}). This directory has been pre-filled with a
  copy of it. EDIT THESE FILES IN PLACE; what they contain when you finish becomes {new_version}.
- `{wiki_dir}` — the KNOWLEDGE WIKI: one directory per task, each with `observations.md` (what
  agents did across models and replicates, how often the skill loaded, and whether loading led to
  success) and `hypothesis.md` (why it did or did not work). Entries are tagged `[version][model]`
  and accumulate across versions — `noskill` is the baseline, then `v1`, `v2`, … READ THIS FIRST.
  It is the distilled signal of what the skill gets right and wrong; the trend across versions
  tells you what past changes did.
- `{transcripts_dir}` — the raw TRAIN-split transcripts behind the wiki, for the parent skill.
  Drill in here only when the wiki is not specific enough to act on.
- `{src}` — the package source (filtered: any skill or agent-guidance files a package ships have
  been stripped and are BLOCKED — write from the API, not from someone else's skill). `{package}`
  is installed; run `{python}` to verify a claim before writing it. Do not install packages.
- You have web access if the published docs help.

# What you are NOT allowed to see

You are optimising against the TRAIN split. A separate, held-out VALID split measures whether your
changes generalise rather than memorise these tasks. Any tool call that reaches valid results is
BLOCKED. Do not attempt it — reaching valid data would invalidate the whole benchmark.

# Two different failures, two different fixes

The wiki records whether the agent LOADED the skill and whether it SUCCEEDED. Separate them before
changing anything — they look identical in a list of failures and have opposite fixes.

1. **The skill never loaded.** The agent never read the body, so nothing in the body caused this
   and nothing you write in the body can fix it. The only lever is the `description`.
2. **The skill loaded and the run still failed.** Now the body is on trial. Fix it, using the
   cause the wiki (or a transcript) actually shows — not one you imagine.
3. **The skill loaded and the run passed.** Evidence the body works. Do not rewrite it for style.

A skill that is never loaded scores exactly like no skill at all. If the wiki shows load rates low
across the board, that is the biggest available win — and the only lever is the `description`. If
one model loads reliably and another almost never does, that gap is mostly the model's behaviour;
do not contort the sentence chasing it.

{craft}

# What you must write

1. Edit the skill in place under `{skill_dir}`. When you finish, `{skill_dir}/SKILL.md` must still
   begin with YAML frontmatter whose `name` is exactly `{skill_name}`.
2. `{rationale_path}` — one short paragraph stating WHAT you changed and WHY, grounded in the wiki.
   Write it OUTSIDE the skill directory; do not put it inside `{skill_dir}`.
{feedback}
When you are done, `{skill_dir}/SKILL.md` exists and starts with the frontmatter above, and
`{rationale_path}` contains your rationale.
"""


CREATE_PROMPT = """\
You are writing the FIRST agent skill (version {new_version}) for the Python package `{package}`
(version {version}).

A skill is documentation written for an agent, not a human. Its only purpose is to make an agent
that has never used `{package}` succeed at real tasks with it on the first try. It is not a
tutorial, not a README, not a sales pitch. The agent arrives with a goal in plain English; the
skill routes it to the right entry point and sequence of steps.

# What you can read

- `{wiki_dir}` — the KNOWLEDGE WIKI: one directory per task, each with `observations.md` (what
  agents did WITHOUT any skill — the `noskill` baseline — across models and replicates, and how
  often they got it right) and `hypothesis.md` (why they succeeded or failed). READ THIS FIRST: it
  tells you exactly where an unaided agent goes wrong, which is precisely what your skill must fix.
- `{transcripts_dir}` — the raw TRAIN-split transcripts behind the wiki. Drill in when the wiki is
  not specific enough.
- `{src}` — the package source (filtered: any skill or agent-guidance files a package ships have
  been stripped and are BLOCKED — write from the API, the source, and the user-facing docs, not
  from someone else's skill). `{package}` is installed; run `{python}` to verify claims. Do not
  install packages.
- You have web access if the published docs help.

# What you are NOT allowed to see

The wiki and transcripts are from the TRAIN split only. A held-out VALID split measures whether
your skill generalises. Any tool call that reaches valid results is BLOCKED — do not attempt it.

# What you must write

Your staging directory is `{skill_dir}`. Write:

1. `{skill_dir}/SKILL.md` — required. It must begin with YAML frontmatter, exactly:

---
name: {skill_name}
description: <one sentence: what this skill covers and when to use it>
---

   The `name` must be exactly `{skill_name}`. The `description` is load-bearing and must be HONEST —
   it is the only thing an agent sees before deciding whether to open the skill.

2. `{skill_dir}/references/*.md` — optional, for detail only some tasks need.

3. `{rationale_path}` — one short paragraph on what the skill covers and why, grounded in the
   baseline wiki. Write it OUTSIDE `{skill_dir}`.

{craft}
{feedback}
When you are done, `{skill_dir}/SKILL.md` exists and starts with the frontmatter above, and
`{rationale_path}` contains your rationale.
"""


WIKI_PROMPT = """\
You are a benchmark analyst keeping a running knowledge wiki about how well an agent skill for the
Python package `{package}` helps on ONE task. Your job this round: read what happened when the
`[{version}]` arm was benchmarked on this task across several models and replicates, and record it
BRIEFLY into two files.

`[{version}]` is the skill version under review. `noskill` means no skill was installed (the
baseline). {skill_note}

# What you are given

- `{evidence_dir}/INDEX.md` — every run of `[{version}]` on task `{task_id}`, one line each: pass or
  fail, whether the skill LOADED, the model, the expected answer, and the answer given.
- `{evidence_dir}/<model>__rep_<n>/` — that run's `transcript.html` (what the agent actually did),
  its `answer.md`, and its `script.py`. Open a few to understand the pattern; you do NOT need to
  read every one.
- The package source is at `{src}` and it is installed — run `{python}` if a fact helps you explain
  an outcome. Optional; do not go down a rabbit hole.
{skill_body_note}
# What you must write — APPEND, do not rewrite

Two files already exist and may contain entries from earlier arms. LEAVE those untouched and ADD
your new entries at the end. `noskill` is the first arm; later arms are `v1`, `v2`, …

1. `{observations_path}` — add ONE line per model, in EXACTLY this format:

   - [{version}][<model>]: <what the agents did, the % of runs the skill loaded, and — when it
     loaded — whether it worked>

   Say the general idea, not a play-by-play. For `noskill`, "loaded" does not apply — just say what
   the agents did and how often they got it right.

2. `{hypothesis_path}` — add ONE line per model, in EXACTLY this format:

   - [{version}][<model>]: <why it did or did not work — one or two short sentences>

# BE BRIEF — this is the whole point

This wiki is read in full by the skill improver every round and grows forever. A verbose wiki is
WORSE than a short one. Hard rules:

- One to two lines per entry. No transcript quotes, no step-by-step, no per-run breakdown, no
  restating the task. Compress across replicates into the pattern.
- The hypothesis is one or two short sentences. No essays, no hedging.

GOOD observation:
  - [{version}][claude-opus-5]: Loaded 3/3; agents used the right entry point and passed every run.
GOOD hypothesis:
  - [{version}][claude-opus-5]: The skill named the correct function and output location, which is
    the step agents otherwise guess wrong.
BAD (too verbose): a paragraph recounting each replicate's tool calls and reasoning.

When you are done, `{observations_path}` and `{hypothesis_path}` each contain your new
`[{version}]` line(s) appended after whatever was already there — and nothing else changed.
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
`.agents/skills/`, `.claude/skills/`, `.codex/`, `CLAUDE.md`, `AGENTS.md`, `.cursor/`,
Copilot instructions). These have
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

# Train and valid variants

Give each task a train and a valid variant of the SAME goal, differing only in the input or the
target it asks about (a different cell type, group, condition, or dataset). Two instances of one
analysis with two different correct answers — so a skill cannot pass by memorising one answer.

# Ground truth by execution

Get each answer by actually DOING the analysis in the venv with `{python}` and reading the real
result — never from a tutorial's printed output or the docs. Before recording an answer, confirm
the goal has exactly ONE defensible answer: if a competent analyst could read the goal two ways
and get two results, tighten only the OUTPUT sentence (what to report, or its precision) until
one answer stands — never by adding back instructions.

# Keep the script that produced each answer — it is a deliverable

The script you run to obtain an answer is NOT scratch. Save one per split to
`{scripts_dir}/<id>-<split>.py`, using the SAME `id` you gave the task in `{out}` — so the task
`bulk` needs `{scripts_dir}/bulk-train.py` and `{scripts_dir}/bulk-valid.py`. These are what
`acumen check` reruns later to confirm the answer still holds, so each one must:

- Be SELF-CONTAINED and runnable from ANY empty working directory: `<python> <script>` with no
  arguments, no setup, no input files. Do not read or write anything outside the working
  directory it is started in, do not depend on files you leave in this directory, and do not
  depend on another script having run first.
- Write the answer, and NOTHING else, to `answer.md` in its working directory — exactly the
  string you record as that split's `answer`, with no label, heading, code fence, or markup.
  Progress messages and diagnostics go to stdout/stderr instead, where they are ignored.
- Install nothing. Use only what is already in the venv.
- Be deterministic. If a step involves randomness, set the seed inside the script so the answer
  is the same on every rerun.

Confirm each script by running it in an empty directory and reading the `answer.md` it produced;
the `answer` you record is that file's exact content. A task that needs no code at all to answer
(a licence, a supported species, a fact stated in the docs) gets `needs_script: false` in the
task and no script.

# What you must write

Your working directory is `{out_dir}`. Write the tasks to `{out}` as YAML with exactly this
shape (the loader is strict — unknown keys are rejected):

tasks:
  - id: <short, unique, filesystem-safe: letters, digits, '.', '_', '-'>
    train:
      prompt: |
        <one-paragraph goal for the train variant>
      answer: "<the exact answer string the real train run produced>"
    valid:
      prompt: |
        <one-paragraph goal for the valid variant>
      answer: "<the exact answer string the real valid run produced>"

`id` must be unique across all tasks. Both `train` and `valid` are required, each with a
non-empty `prompt` and a non-empty `answer`. Add `needs_script: false` at the task level (a
sibling of `id`) only for a task that needs no code to answer; it defaults to true and is then
omitted. Do not add other keys unless you deliberately want a per-task override (`max_turns`,
`max_usd`, or `model` are the only ones allowed).
{feedback}
# Before you finish

- Every `answer` is the exact content of the `answer.md` written by the script you ran in the
  venv — not a guess, not lifted from docs.
- Every task with `needs_script` unset has BOTH `{scripts_dir}/<id>-train.py` and
  `{scripts_dir}/<id>-valid.py`, each verified by running it in an empty directory.
- Every prompt is ONE paragraph: a goal in plain English, with no steps, no code, no package
  name, no version, no data description — only the goal and a precise statement of the output.
- You wrote at least one task per tutorial, and covered all of them.
- `{out}` exists and parses as the YAML above with at least one task.
"""


REVIEW_PROMPT = """\
You are reviewing a benchmark of analysis tasks for a Python package (`{package}` {version}) for
INTERNAL CONSISTENCY. Each task states a goal in plain language, records the single answer a
correct analysis produces, and (usually) ships a reproducer script that recomputes that answer.

Your one question, for each task split: **do the prompt, the recorded answer, and the script
describe the same thing?** When they do not, every agent that reads the prompt correctly is
graded wrong, and a whole benchmark pass measures the task's phrasing instead of the model. That
is what you are here to catch, and nothing else.

# What you are given

- `{packet_dir}/TASKS.md` lists every task split: its prompt verbatim, the answer recorded for
  it, what happened when acumen ran its reproducer, and the script's filename.
- `{packet_dir}/scripts/` holds those reproducers. Read the ones you are judging.
- The package's source is at `{src}` and the package is installed; run `{python}` if you need it.

# Do not redo the analysis

acumen has ALREADY run every reproducer and told you the outcome in `TASKS.md`. Do not rerun a
pipeline and do not recompute an answer — that work is done and repeating it is what makes this
review expensive. You may read the source, or run a couple of lines, to confirm what a function
returns or which sign convention it uses. That is the limit.

# What counts as a mismatch

- **The prompt's stated order or direction disagrees with the script or the answer.** The prompt
  says ascending where the script sorts descending; it asks for the most X where the script takes
  the least; the recorded answer is in the opposite order from the one the prompt requests. This
  is the most common defect and the easiest to read past, so check it on every split that states
  an ordering.
- **The prompt asks about something else than the script does**: a different dataset, group, cell
  type, condition, statistic, or a different number of items.
- **The answer's FORM is not what the prompt asks for**: the wrong separator, more or fewer items
  than requested, a different rounding or precision, a name where a number was asked for.
- **The prompt cannot be answered as written**: it names a dataset, field, or capability the
  package does not have, so no correct analysis reaches the recorded answer.

# What is NOT a mismatch

- **Train and valid differ on purpose.** They are two instances of one analysis with two different
  answers, deliberately asking about different groups, conditions, datasets, or directions. A
  difference between the two splits is the design, not a defect.
- **A terse prompt naming no function, parameter, or output field.** Working out HOW is exactly
  what the benchmark tests; a prompt that only states the goal is correct by design.
- **A script that is longer, slower, or less elegant than it needs to be.** You are not reviewing
  code quality.
- **A reproducer that merely failed to run** (a crash, a missing script, a timeout). That is
  already reported. Judge only whether the artifacts you CAN read contradict each other; if a
  split has no script, review its prompt against its recorded answer alone.

# Be brief — that is the deliverable

For a mismatch, `issue` names the contradiction in one short clause and `fix` names what to
change in one short clause. Do NOT rewrite the prompt, do not quote it back, do not explain the
analysis, and do not suggest improvements to a task that holds together. Anything past one line
is truncated, so put the contradiction first. Say nothing at all for a verdict of `ok`.

# What you must write

Write `{out}` as JSON with exactly this shape, with ONE entry for every task split listed in
`TASKS.md` and no other keys:

{{"reviews": [
  {{"task": "<task id>", "split": "train", "verdict": "ok"}},
  {{"task": "<task id>", "split": "valid", "verdict": "mismatch",
   "issue": "prompt says ascending; script and answer are descending",
   "fix": "say descending in the prompt, or reverse the answer"}}
]}}

`verdict` is `"ok"` or `"mismatch"` — there is no third value. `issue` and `fix` are required on
every `mismatch` and omitted on every `ok`. A verdict of `mismatch` with no reason cannot be
acted on, and a task you are unsure about is `ok`: say `mismatch` only where you can name the
contradiction.

# Before you finish

- `{out}` exists, parses as the JSON above, and has one entry per split in `TASKS.md`.
- Every `mismatch` names a contradiction you actually read in the artifacts, not one you suspect.
- No `issue` or `fix` is longer than one short clause.
"""


#: Canonical ``install.py`` the shipping agent adapts. The single placeholder is
#: ``__SKILL_NAME__`` (rendered by :func:`ship_prompt` via ``str.replace`` so the template's
#: own braces need no escaping). It uses ``__package__`` so the import package name never has
#: to be edited into it, and ``importlib.resources`` so it reads the skill data from wherever
#: the wheel installed it — which is exactly what the build-verify gate confirms.
SHIP_INSTALL_TEMPLATE = '''\
"""Install the bundled ``__SKILL_NAME__`` skill into an agentic framework's skills directory.

Console script (wired as ``<dist>-install-skills`` in ``pyproject.toml``): copy the skill that
ships inside this package into a chosen framework's skills directory, so an agent can load it.
The same ``SKILL.md`` + ``references/`` bundle is a cross-framework standard, so it installs
verbatim — no per-framework conversion.

Frameworks (``--agent``):

- ``claude``          -> ``~/.claude/skills`` (honours ``CLAUDE_CONFIG_DIR``)
- ``codex``           -> ``~/.codex/skills`` (honours ``CODEX_HOME``)
- ``agents``          -> ``~/.agents/skills``
- ``claude-science``  -> the active org's skills dir, resolved from
  ``~/.claude-science/active-org.json``

``--dest`` overrides all of them. There is **no default framework**: pass ``--agent`` or
``--dest``. The skill files live in the ``data/`` directory next to this module and are read via
``importlib.resources``, so this works from an installed wheel, not just an editable checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

#: The skill name; it installs to ``<framework-skills-dir>/<SKILL_NAME>/``.
SKILL_NAME = "__SKILL_NAME__"

#: Framework -> (env var that overrides the config root, default config root). The skills
#: directory is ``<root>/skills``. ``claude-science`` is resolved separately.
_AGENT_ROOTS = {
    "codex": ("CODEX_HOME", "~/.codex"),
    "claude": ("CLAUDE_CONFIG_DIR", "~/.claude"),
    "agents": (None, "~/.agents"),
}

#: The frameworks ``--agent`` accepts.
AGENTS = (*sorted(_AGENT_ROOTS), "claude-science")


def source_dir() -> Path:
    """Return the package-owned skill directory (the bundle that gets copied)."""
    source = Path(str(resources.files(__package__).joinpath("data")))
    if not (source / "SKILL.md").is_file():
        raise RuntimeError(f"packaged skill data is missing: {source}")
    return source


def _claude_science_skills_dir() -> Path:
    """Resolve the active Claude Science org's skills directory."""
    root = Path("~/.claude-science").expanduser()
    active_org_path = root / "active-org.json"
    try:
        active_org = json.loads(active_org_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "cannot resolve the Claude Science active organization from "
            f"{active_org_path}; pass --dest instead"
        ) from error
    org_uuid = active_org.get("org_uuid") if isinstance(active_org, dict) else None
    if (
        not isinstance(org_uuid, str)
        or not org_uuid
        or Path(org_uuid).name != org_uuid
        or org_uuid in {".", ".."}
    ):
        raise ValueError(f"invalid Claude Science org_uuid in {active_org_path}; pass --dest instead")
    return root / "orgs" / org_uuid / "skills"


def _skills_dir(agent: str) -> Path:
    """Return the skills directory (parent of the install dir) for a framework."""
    if agent == "claude-science":
        return _claude_science_skills_dir()
    variable, fallback = _AGENT_ROOTS[agent]
    root = os.environ.get(variable) if variable is not None else None
    return Path(root or fallback).expanduser() / "skills"


def resolve_dest(agent: str | None, dest: Path | None) -> Path:
    """Resolve the install destination from ``--agent`` / ``--dest``.

    ``--dest`` wins. With neither, raise ``ValueError`` — there is no default framework.
    """
    if dest is not None:
        return dest.expanduser()
    if agent is None:
        raise ValueError("pass --agent {" + ",".join(AGENTS) + "} or --dest to choose where to install")
    return _skills_dir(agent) / SKILL_NAME


def _snapshot(root: Path) -> dict[str, bytes]:
    """Map each file under ``root`` to its bytes, for exact tree comparison."""
    return {
        str(item.relative_to(root)): item.read_bytes()
        for item in root.rglob("*")
        if item.is_file()
    }


def _matches(source: Path, target: Path) -> bool:
    """Report whether ``target`` is a byte-for-byte copy of ``source``."""
    return target.is_dir() and _snapshot(source) == _snapshot(target)


def main(argv: list[str] | None = None) -> int:
    """Install the bundled skill; entry point for the ``<dist>-install-skills`` script."""
    parser = argparse.ArgumentParser(
        description=f"Install the {SKILL_NAME} skill bundled with this package into an agent's skills directory.",
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default=None,
        help="framework to install into (no default — pass this or --dest)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="exact skills directory to install into (overrides --agent)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing installation that differs from the bundled skill",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="report whether the installed skill matches the bundled one; do not install",
    )
    action.add_argument(
        "--print-path",
        action="store_true",
        help="print the bundled skill's location inside the package and exit",
    )
    args = parser.parse_args(argv)

    try:
        source = source_dir()
        if args.print_path:
            print(source)
            return 0
        if args.check:
            target = resolve_dest(args.agent, args.dest)
            if not target.exists():
                print(f"{SKILL_NAME} skill is not installed at {target}", file=sys.stderr)
                return 1
            if _matches(source, target):
                print(f"{SKILL_NAME} skill at {target} matches the bundled copy")
                return 0
            print(f"{SKILL_NAME} skill at {target} differs from the bundled copy", file=sys.stderr)
            return 1

        dest = resolve_dest(args.agent, args.dest)
        if dest.exists():
            if _matches(source, dest):
                print(f"{SKILL_NAME} skill already up to date at {dest}")
                return 0
            if not args.force:
                print(f"{dest} already exists and differs; pass --force to overwrite", file=sys.stderr)
                return 1
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"installed the {SKILL_NAME} skill to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


SHIP_PROMPT = """\
You are making a benchmarked Skill installable *into* the Python package it documents.
When you are done, the package will expose a console script `<dist>-install-skills` that drops
the skill into the skills directory of whichever agentic framework the user names —
`--agent {{claude,codex,agents,claude-science}}`, or an explicit `--dest` — so the package's own
users get the agent guidance with one command, wherever they run their agent. The same skill
bundle installs verbatim into every framework; there is no per-framework conversion.

You work in the package's checkout, and this is the REAL environment — real network, real
git/`gh` credentials, real `uv`. You may run any command you need.

# The package checkout

- Your working directory is `{checkout}`. This is the package you are modifying.
- Its declared distribution name is `{package}` (from `pyproject.toml`), but DO NOT assume the
  import package name or the layout — detect them (see below).

# The skill to ship

- The skill to package is at `{skill_src}` (version {version}). It contains `SKILL.md` and
  maybe a `references/` tree. Copy its ENTIRE contents VERBATIM into the package — do not author,
  edit, summarise, or reformat any skill file. `acumen ship` only packages what was already
  benchmarked.

# Detect — never assume

Read `{checkout}/pyproject.toml` and work out, from the file itself:

1. The **distribution name** (`[project].name`, or the build-backend's equivalent). The console
   script MUST be named `<dist-name>-install-skills`.
2. The **import package name** and its **directory on disk**. It may be `src/<pkg>/` (src-layout)
   or `<pkg>/` (flat-layout) or something else — find the real directory that holds the package's
   `__init__.py`. The entry point targets the import package, `<pkg>._skills.install:main`.
3. The **build backend** (`[build-system].build-backend`). You will wire packaging differently
   for hatchling vs setuptools vs flit/pdm/poetry (see below).

# What to create

Inside the import package directory, create a `_skills/` subpackage:

- `_skills/__init__.py` — may be empty.
- `_skills/install.py` — use the canonical template shown at the end of this prompt VERBATIM (it
  already has the right skill name baked in and reads its data via `importlib.resources`, so it
  needs no per-package editing). Do not rewrite it.
- `_skills/data/` — a VERBATIM copy of everything under `{skill_src}` (so
  `_skills/data/SKILL.md`, `_skills/data/references/...`, etc.).

Then wire two things in `pyproject.toml`:

- The entry point, under `[project.scripts]` (or the backend's script table):
  `<dist-name>-install-skills = "<import-pkg>._skills.install:main"`.
- **Packaging so the non-`.py` skill files actually ship in the wheel.** This is the step that
  silently fails and the whole point of the build-verify gate. `SKILL.md` and the `references/*.md`
  are DATA files, not modules — a naive build drops them. Wire whatever the detected backend needs:
  - **hatchling** ships everything under the package directory automatically; usually nothing extra
    is needed, but confirm `_skills/data` is included (a restrictive `[tool.hatch.build.targets.wheel]`
    `include`/`packages` may need `_skills/data` added, or `force-include`).
  - **setuptools** needs the data declared: `include-package-data = true` plus a `MANIFEST.in`
    (`recursive-include <pkg>/_skills/data *`), or an explicit `[tool.setuptools.package-data]`
    entry for the `_skills.data` files. Also make sure `_skills` (and `_skills.data` if it needs a
    package) are found by `find`/`packages`.
  - **flit / pdm / poetry** each have their own include mechanism — use it so `_skills/data/**`
    is bundled.

# Verify by building — this is the correctness gate, do NOT skip it

A wheel that installs but ships NO skill data is the failure mode this whole task exists to
prevent, and it fails silently. Before you deliver anything, prove the data ships:

1. Build a wheel from `{checkout}` (e.g. `uv build --wheel`).
2. Install THAT wheel into a FRESH throwaway venv (`uv venv`, then `uv pip install <the-wheel>`) —
   a fresh venv, NOT an editable install, because an editable install sees the source tree
   regardless of packaging and would hide the bug.
3. From that venv, run `<dist-name>-install-skills --agent codex --dest <a-scratch-dir>` and
   confirm `<a-scratch-dir>/SKILL.md` exists and is byte-identical to `{skill_src}/SKILL.md`.
   (There is no default framework, so pass an explicit `--dest` — or `--agent` — here.) Also run
   `<dist-name>-install-skills --print-path` and confirm it prints a path inside the installed
   package that contains the shipped `SKILL.md`.
4. If `SKILL.md` did not land, your packaging is wrong — FIX IT and rebuild. Do not proceed to
   delivery until a fresh-venv install ships the skill.

# Scope limits

- Bundle EXACTLY ONE skill — version {version}, copied verbatim. Nothing else.
- Do NOT write any test file. (Deliberate scope decision.)
- Update the README / docs to mention the `<dist-name>-install-skills` command ONLY if there is a
  natural spot for it (e.g. an existing install or usage section). Keep it to a sentence, noting
  the `--agent {{claude,codex,agents,claude-science}}` (or `--dest`) choice. If there is no natural
  spot, skip it — do not invent a section.

# Deliver

{delivery}

# The canonical install.py (use verbatim as `_skills/install.py`)

```python
{install_template}
```

When you are done, report what you changed: the import package dir you found, the build backend,
the console-script name, the packaging change you made, the result of the fresh-venv build-verify,
and {delivery_report}.
"""


SHIP_DELIVERY_GITHUB = """\
This target is a GitHub repository and `{checkout}` is a git checkout of it with `origin` set to
the target. Deliver the change as a pull request, running git/`gh` yourself:

- Create a new branch (e.g. `acumen/ship-skill-{version}`).
- Commit your changes with a clear message.
- Push the branch to `origin`. This assumes you (the maintainer) have write access. If the push
  is REJECTED for lack of access, STOP and report that clearly — do not try to fork or find
  another remote.
- Open a pull request with `gh pr create`, titled for the skill installer, its body summarising
  what shipped and the build-verify result.

Do not merge the PR — the maintainer reviews it."""


SHIP_DELIVERY_LOCAL = """\
This target is a LOCAL path, `{checkout}`. Write the change directly into the working tree —
create the files and edit `pyproject.toml` in place. Do NOT create a branch, commit, or open a
PR; leave the changes in the working tree for the user to review with `git diff` (or as plain
edits if it is not a git repo)."""


def improve_prompt(
    *,
    package: str,
    version: str,
    src: Path,
    python: Path,
    skill_dir: Path,
    wiki_dir: Path,
    transcripts_dir: Path,
    rationale_path: Path,
    skill_name: str,
    new_version: str,
    parent_version: str | None = None,
    feedback: str | None = None,
) -> str:
    """Build the prompt for the improving agent, in create or improve mode.

    The improver works from the knowledge wiki (distilled per-task notes tagged
    ``[version][model]``) plus the filtered package source. When ``parent_version`` is ``None`` it
    is the FIRST skill and this is *create* mode — the wiki holds only the ``noskill`` baseline and
    the staging dir is empty. Otherwise it is *improve* mode — the staging dir is pre-filled with
    the parent skill to edit in place.

    Both modes read only TRAIN-split evidence; the held-out valid split is unreachable, enforced
    structurally and by a guard hook, not by this prompt. The source is the *filtered* copy, so a
    skill the package itself ships cannot bias the optimization.

    Parameters
    ----------
    package, version
        The target package name and installed version.
    src
        The filtered package checkout (bundled skills/agent-guidance stripped).
    python
        The interpreter with the package installed, for verifying claims.
    skill_dir
        The staging directory. In improve mode it is pre-filled with the parent skill; in create
        mode it is empty and the agent writes ``SKILL.md`` from scratch.
    wiki_dir
        The staged knowledge wiki the agent reads first.
    transcripts_dir
        Staged train-split transcripts, for drill-down beyond the wiki.
    rationale_path
        Where the agent writes its rationale — outside ``skill_dir``.
    skill_name
        The name the frontmatter must carry — ``config.skill_name``.
    new_version
        The version being produced, e.g. ``v1`` or ``v2``.
    parent_version
        The version being improved, or ``None`` for the first (create) skill.
    feedback
        Optional maintainer guidance, subordinated below the hard rules. ``None`` leaves the
        prompt byte-identical to a run without the flag.

    Returns
    -------
    The improve or create prompt.
    """
    common = {
        "package": package,
        "version": version,
        "src": src,
        "python": python,
        "skill_dir": skill_dir,
        "wiki_dir": wiki_dir,
        "transcripts_dir": transcripts_dir,
        "rationale_path": rationale_path,
        "skill_name": skill_name,
        "new_version": new_version,
        "craft": _SKILL_CRAFT,
        "feedback": feedback_block(feedback),
    }
    if parent_version is None:
        return CREATE_PROMPT.format(**common)
    return IMPROVE_PROMPT.format(parent_version=parent_version, **common)


def wiki_prompt(
    *,
    package: str,
    version: str,
    src: Path,
    python: Path,
    task_id: str,
    evidence_dir: Path,
    observations_path: Path,
    hypothesis_path: Path,
    skill_dir: Path | None = None,
) -> str:
    """Build the prompt for one wiki agent (one task, one arm).

    The agent reads staged copies of that arm's train-split runs and appends a terse
    ``[version][model]`` block to the task's ``observations.md`` and ``hypothesis.md``. Brevity is
    enforced by the prompt: the wiki is read whole by the improver every epoch and grows forever.

    Parameters
    ----------
    package
        The target package name, for orientation.
    version
        The arm label recorded in each entry: ``"noskill"`` or ``"v1"``/``"v2"``…
    src
        The filtered package checkout, for grounding an explanation. Never the raw checkout.
    python
        The interpreter with the package installed.
    task_id
        The task being summarised.
    evidence_dir
        The staged run evidence (``INDEX.md`` + per-run directories).
    observations_path, hypothesis_path
        The two files to append to, pre-seeded with any earlier arms' entries.
    skill_dir
        The arm's skill content directory, or ``None`` for ``noskill``.

    Returns
    -------
    The wiki prompt.
    """
    if skill_dir is None:
        skill_note = "This round is `noskill`, so there is no skill body — just record how the agents did unaided."
        skill_body_note = ""
    else:
        skill_note = (
            f"The skill body under review is at `{skill_dir}`; read it so your hypothesis can cite what it said."
        )
        skill_body_note = f"- `{skill_dir}` — the `[{version}]` skill body the runs above were given.\n"
    return WIKI_PROMPT.format(
        package=package,
        version=version,
        src=src,
        python=python,
        task_id=task_id,
        evidence_dir=evidence_dir,
        observations_path=observations_path,
        hypothesis_path=hypothesis_path,
        skill_note=skill_note,
        skill_body_note=skill_body_note,
    )


def taskgen_prompt(
    *, package: str, src: Path, python: Path, out: Path, scripts_dir: Path, feedback: str | None = None
) -> str:
    """Build the prompt for the task-generation agent.

    Like the drafter, the generator gets read access to the target's source — it must
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
    scripts_dir
        Where the agent saves the reproducer script for each split. Unlike everything else in
        its working directory these are harvested and kept, because ``acumen check`` reruns them
        to confirm the answers still hold.
    feedback
        Optional maintainer guidance, subordinated below the hard rules — e.g. functionality to
        skip. ``None`` leaves the prompt byte-identical to a run without the flag.

    Returns
    -------
    The task-generation prompt.
    """
    return TASKGEN_PROMPT.format(
        package=package,
        src=src,
        python=python,
        out=out,
        out_dir=out.parent,
        scripts_dir=scripts_dir,
        feedback=feedback_block(feedback),
    )


def review_prompt(*, package: str, version: str, src: Path, python: Path, packet_dir: Path, out: Path) -> str:
    """Build the prompt for the task-review agent.

    The reviewer judges whether each task's prompt, recorded answer, and reproducer script agree
    with each other. It reads a staged packet acumen writes (never the project's own files) plus
    the target's source and venv, since deciding whether a prompt describes what a script computes
    can need the package's own semantics — which direction a statistic runs, what a function
    returns.

    It is deliberately told not to rerun the pipelines: acumen has already run every reproducer
    and hands the outcome over in the packet, so recomputing answers is pure cost.

    Parameters
    ----------
    package, version
        The target's name and installed version, for orientation.
    src
        The package checkout, readable by this agent.
    python
        The interpreter with the package installed, for confirming an API's behaviour.
    packet_dir
        The staged review packet: ``TASKS.md`` plus copies of the reproducers.
    out
        The JSON verdict file the agent writes.

    Returns
    -------
    The review prompt.
    """
    return REVIEW_PROMPT.format(
        package=package,
        version=version,
        src=src,
        python=python,
        packet_dir=packet_dir,
        out=out,
    )


def ship_prompt(
    *,
    package: str,
    skill_name: str,
    version: str,
    checkout: Path,
    skill_src: Path,
    mode: str,
) -> str:
    """Build the prompt for the shipping agent.

    Unlike every other agent, the shipper runs UNISOLATED — real network, git/``gh``
    credentials, and ``uv`` — because it builds, installs, pushes, and opens a PR.
    It reasons about the package (distribution vs import name, src-vs-flat layout, build
    backend) rather than assuming decoupler's shape, so this is an autonomous agent.

    Parameters
    ----------
    package
        The target's distribution name (``[project].name``), for orientation. The agent still
        detects the import package name and layout itself.
    skill_name
        The skill's frontmatter name — baked into the install script and the
        ``<framework-skills-dir>/<name>/`` install path.
    version
        The skill version being shipped, e.g. ``v2`` — named in the branch/commit/PR.
    checkout
        The package checkout the agent modifies (its ``cwd``): the local path for a local
        target, or acumen's clone for a GitHub URL.
    skill_src
        A staged copy of the skill's content files (``SKILL.md`` + ``references/``, without
        ``meta.json``), to be copied verbatim into ``_skills/data/``.
    mode
        ``"github"`` (deliver as a PR) or ``"local"`` (edit the working tree directly).

    Returns
    -------
    The ship prompt.
    """
    install_template = SHIP_INSTALL_TEMPLATE.replace("__SKILL_NAME__", skill_name)
    if mode == "github":
        delivery = SHIP_DELIVERY_GITHUB.format(checkout=checkout, version=version)
        delivery_report = "the branch you pushed and the URL of the pull request you opened"
    else:
        delivery = SHIP_DELIVERY_LOCAL.format(checkout=checkout)
        delivery_report = "confirmation that the working tree now carries the change"
    return SHIP_PROMPT.format(
        package=package,
        version=version,
        checkout=checkout,
        skill_src=skill_src,
        delivery=delivery,
        delivery_report=delivery_report,
        install_template=install_template,
    )


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
