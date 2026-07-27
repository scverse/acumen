---
name: acumen
description: Use for any question or task involving the Python package `acumen` (its CLI or its API) — setting up a benchmark project against a target package, writing or generating benchmark tasks, drafting/improving/hand-editing an agent Skill, running and interpreting skill-vs-baseline benchmark passes, shipping a skill into the target package, diagnosing a run, and choosing what to do next in that loop — open it before answering or running anything, because acumen's defaults, guardrails and correct next step are not guessable from the command names.
---

# acumen

One project directory targets one Python package. Everything is driven by the `acumen`
CLI (a thin shell over an importable API — see `references/python-api.md`). Commands are
run from the project dir and default to `config.yaml`, `tasks.yaml`, `skills/`, `runs/`,
`logs/`.

```bash
acumen init                # scaffold config.yaml + tasks.yaml (placeholders the user must fill)
acumen tasks               # optional: mine the package for tasks, run them for ground truth
acumen bench --no-skill    # baseline arm  (this is also the default arm)
acumen draft               # agent reads the package source -> skills/v1/
acumen bench --skill v1
acumen improve             # agent reads v1's TRAIN runs -> skills/v2/
acumen bench --skill v2
acumen report              # report.html + report.csv over every arm on disk
acumen ship --skill v2     # wire a `<dist>-install-skills` script into the target package
```

That is the shape of the loop, not a script to execute unattended, and not a fixed order:
several stages have more than one correct next move, and some stages call for talking to
the user rather than running the next command at all.

## The loop is collaborative — propose, discuss, then run

`bench` bills the Anthropic API for every cell and the other commands drive long
autonomous agents, so the user decides each step and when to stop. At every stage: state
what you would do next and why, and let the user choose. Never chain commands
unattended, and never decide on your own that a skill is good enough.

**Before running anything, get from the user what acumen cannot infer** — the target, the
models, the budgets, whether tasks and results look right, whether to ship. Guessing those
is the most common way to waste a pass. The step after a command is often a question, not
a command.

| Just happened | What to propose next |
|---|---|
| `acumen init` | Both files are placeholders (`repo: OWNER/REPO`, `REPLACE_ME` answers). Ask the user for what only they can supply — the target repo/path, ref, models, budgets — and fill in `config.yaml` with their answers. Do not guess a target or run the next command. |
| `config.yaml` filled | Offer both ways to get tasks: `acumen tasks` to generate them, or writing `tasks.yaml` by hand. Review generated tasks with the user before benching. |
| config + tasks ready | `acumen bench --no-skill` and `acumen draft` are **both** correct next steps — they are independent, and you need the baseline arm and a skill arm before any comparison means anything. Offer both. |
| a skill version exists | `acumen bench --skill vN`. |
| any `bench` finished | `acumen report` — the per-arm, per-split numbers come from the report; do not judge a version by eyeballing runs or by one arm's pass count. |
| `report` written | Discuss the results with the user and propose next steps: `improve` + re-bench if it has not beaten the baseline or the previous version, `ship` once a version has proven out, or stop. The user decides. |
| user has a skill dir from elsewhere | Copy it into `skills/v1/` (next unused version) inside the acumen project — versions are only ever read from `skills/`; there is no import command and no external path flag. |

## Route by goal

| Goal | Command | Depth |
|---|---|---|
| Start a project for package X | `acumen init`, then fill `config.yaml` with the user | `references/setup.md` |
| Get benchmark tasks without writing them | `acumen tasks [--force] [--feedback "…"]` | `references/setup.md` |
| Write tasks by hand | edit `tasks.yaml` | `references/setup.md` |
| Get a first skill | `acumen draft [--feedback "…"]` | `references/authoring.md` |
| Measure whether the skill helps | `acumen bench --no-skill` and `acumen bench --skill vN` | `references/benchmark.md` |
| Make the skill better | `acumen improve [--from vN]` then bench the new version | `references/authoring.md` |
| See results / decide when to stop | `acumen report` | `references/benchmark.md` |
| Hand-edit a skill version | copy `skills/vN` → `skills/v(N+1)`, edit, bench it | `references/authoring.md` |
| Give package users the skill | `acumen ship --skill vN` | `references/ship.md` |
| Script any of this in Python | `from acumen import …` | `references/python-api.md` |
| A run failed / the skill did nothing | inspect `runs/…/result.json`, `logs/*.jsonl` | `references/benchmark.md` |

## Preconditions

- Python ≥ 3.12, and **`uv` on PATH** — acumen builds the target's venv with it.
- The target (`repo`: a GitHub URL or a local path) must be pip-installable and declare
  `[project].name` in `pyproject.toml`. A local `repo` path is resolved **relative to
  `config.yaml`**.
- Credentials: `bench` **always bills the Anthropic API** and refuses to run without
  `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN` / Bedrock / Vertex) — it records real
  per-run `cost_usd`, and it has no `--auth` flag. `tasks`/`draft`/`improve`/`ship`
  default to the Claude subscription if you are logged in (`--auth {auto,session,api}`).
- The target is cloned + installed into a venv cached under `~/.cache/acumen`, keyed by
  (repo, ref). Use `--refresh-target` after changing the target's own source.

## What guessing gets wrong

1. **Grading is exact string match on `answer.md` after `strip()`, case-sensitive.**
   Nothing is normalized. `**TOKEN**` against `TOKEN` fails (recorded as `format_error`,
   not `wrong_answer`). Task answers must be one short unambiguous token.
2. **A pass is models × tasks × splits × replicates.** The scaffolded config lists 3
   models and `n_replicates: 3`, so *one* task costs 18 agent runs *per arm*. Check with
   `acumen bench --dry-run` (it plans and exits, spending nothing) and agree the size with
   the user before spending. Trim with `models:`, `n_replicates: 1`, `--task ID`, `--split`.
3. **`max_turns`/`max_usd` in `config.yaml` cap benchmark agents only.** `draft`,
   `improve`, `tasks`, and `ship` are **unbounded** unless you pass `--max-turns`/`--max-usd`.
4. **Skill versions are immutable.** `draft` refuses (exit 2) if any `skills/vN` exists;
   `improve` always writes the next unused directory. Never edit a benched version in
   place — its hash is recorded in every `result.json`.
5. **`SKILL.md` frontmatter `name` must equal `config.skill_name`**, which defaults to the
   repo's last path component, slugified and lowercased (`.../My_Pkg` → `my_pkg`). A
   mismatch makes `bench`/`improve`/`ship` fail on load. `description` must be non-empty.
6. **`improve` needs benched train evidence for its parent.** Run
   `acumen bench --skill vN` before `acumen improve`, or it errors with nothing to read.
7. **Never leak the test split.** `improve` is structurally and hook-blocked from
   `runs/*/test/`; don't defeat that by pasting test answers into `--feedback`. A widening
   train/test gap in the report is the overfitting signal you are watching for.
8. **Task prompts must not name the target package** — the harness already tells the agent
   which package to use and that it is installed.
9. **Resume is automatic**: a run is "complete" when its `result.json` exists and is
   non-empty, and completed runs are skipped. `--no-resume` re-runs them. Renaming a task
   `id` orphans its old runs (the id is a path component).
10. **Confirm the skill actually loaded.** `bench` prints `skill loaded in N/M runs` from
    per-run `skill_loaded` evidence, and warns when a skill arm never fired the Skill tool
    (that arm measures nothing) or when the baseline did.
