# Benchmarking and reading results

## `acumen bench`

```bash
acumen bench [--no-skill | --skill v1] [--split train|test]... [--task ID]...
             [--replicates N] [--max-concurrency N] [--dry-run] [--no-resume]
             [--keep-sandboxes] [--refresh-target]
             [--config config.yaml] [--tasks tasks.yaml] [--runs runs] [--skills skills]
             [--cache ~/.cache/acumen]
```

- Neither `--no-skill` nor `--skill` → the **baseline** arm (they are mutually exclusive).
- `--split` and `--task` are repeatable; omitted means all.
- `--dry-run` prints the planned matrix and exits **before** target prep and before any
  agent runs — free, and the right way to check the size of a pass.
- `--replicates` / `--max-concurrency` override the config for this pass only.
- Always bills the API (`ANTHROPIC_API_KEY` etc.); there is no `--auth` here, because the
  recorded `cost_usd` must be real metered spend.
- Ctrl-C is safe: completed runs are preserved and the next invocation resumes.

**Arm parity is the whole point.** Both arms get an identical prompt, tools, caps, and
environment; the only difference is that a skill arm copies the skill's content files into
`<sandbox>/.claude/skills/<skill_name>/` where project settings discovery finds it.
`meta.json` is never copied. Each run gets a fresh empty sandbox with the target venv on
PATH, a throwaway `HOME` and `CLAUDE_CONFIG_DIR` — no repo source, no user settings, no
`CLAUDE.md` memories, no visibility of other runs.

## The run tree

```
runs/{arm}/{split}/{model}/{task_id}/rep_{n}/
```

`arm` is `noskill` or `skill_v1`, `skill_v2`, …; `model` is slugified. Each completed leaf
holds five files:

| File | What |
|---|---|
| `answer.md` | What the agent wrote — the only thing graded |
| `script.py` | The agent's reproduction script (absent if it ran no code) |
| `transcript.jsonl` | SDK-native transcript |
| `transcript.html` | Rendered transcript (via `claude-code-log`) |
| `result.json` | The unit of record — written **last** |

`result.json` presence + non-zero size is what marks a run complete, which is what makes
resume safe. Useful fields: `success`, `reason`, `answer`, `expected`, `model`, `turns`,
`cost_usd`, `input_tokens`, `output_tokens`, `duration_s`, `skill_hash`, `skill_name`,
`skill_loaded`, `pkg_version`, `commit`, `session_id`, `subtype`, `error`.

## Reasons

| `reason` | Meaning |
|---|---|
| `ok` | Exact match — the only success |
| `wrong_answer` | Content differs |
| `format_error` | Content would match but formatting broke the exact match (bold, quotes, "Answer:", code fence, trailing period) |
| `no_answer_file` | The agent never wrote `answer.md` |
| `budget` / `max_turns` | Cap breached — a **failure regardless** of what `answer.md` contains |
| `error` | The agent crashed or produced no result |

A run of `format_error`s means the harness's answer-format instruction is losing to the
task prompt; a run of `no_answer_file` usually means the task is too big for `max_turns`.

## Diagnosing a disappointing arm

1. `skill loaded in 0/N runs` on a skill arm → the arm measured nothing. The skill's
   `description` is what decides whether it loads; make it name the goals the task prompts
   actually phrase.
2. The Skill tool firing in *baseline* runs is also flagged — that baseline is not clean.
3. Open `runs/.../transcript.html` for a failing run, or re-run one cell with
   `--task ID --split train --replicates 1 --keep-sandboxes` and inspect the sandbox.
4. Compare `answer` vs `expected` in `result.json` before blaming the skill — a systematic
   near-miss is a task-authoring problem, not a skill problem.

## `acumen report`

```bash
acumen report [--runs runs] [--tasks tasks.yaml] [--skills skills] [--out report.html]
              [--palette claude-opus-5=#3b7ea1]...
```

- Writes `report.html` **and a sidecar `report.csv`** next to it, overwriting both; the
  report always reflects every run currently on disk across every arm.
- Reads only `result.json` files — never transcripts. Fails if `runs/` holds none.
- **The figures show the TEST split** (the held-out measure). The per-run table below them
  lists both splits.
- Sections: Overview (success rate, tokens, cost, time per arm, bars coloured by model),
  Per-task breakdown, Runs table (links to each `transcript.html`), and — when `--skills`
  resolves — Skill versions with each version's rationale and diff against its parent.
- `--tasks` / `--skills` are optional; missing ones only drop their sections (it says so).
- `--palette MODEL=COLOUR` is repeatable and accepts comma-separated pairs. A key that
  matches no model in the data, or an unparseable colour, is rejected up front.

Run `acumen report` after every bench — it is how a version is judged, and the only view
that compares arms and splits side by side. Then take the results to the user: say what
improved, what did not, and what you would do next (`improve` and re-bench, more tasks,
`ship`, or stop). Train improving while test does not is overfitting — that gap is exactly
what the split exists to expose — but the call on when to stop and what to ship is the
user's.
