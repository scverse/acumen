# Project setup: `config.yaml` and `tasks.yaml`

`acumen init [--dir DIR] [--force]` writes both files as annotated placeholders. It refuses
to clobber either unless `--force`. Both loaders are **strict: unknown keys are rejected**.

Nothing else can run until `config.yaml` is filled in: the scaffold ships
`repo: https://github.com/OWNER/REPO` and a `REPLACE_ME` example task. Ask the user for the
values only they know (target repo/path, ref, extras, which models, budgets) instead of
inventing them.

## config.yaml

Only `repo` is required; delete a line to take its default.

| Key | Default | Notes |
|---|---|---|
| `repo` | — | GitHub URL (`https://`, `git@`, `ssh://`, `git://`) or a local path. A local path is resolved relative to `config.yaml` and must exist. |
| `ref` | `main` | Branch/tag/commit; ignored for local paths. |
| `extras` | `[]` | pip extras installed with the target, e.g. `[test]`. |
| `python` | `"3.12"` | Interpreter for the target's venv (quote it — `3.10` unquoted is a float). |
| `env_passthrough` | `[]` | Extra env var names agents may keep. The agent env is a clean allowlist (auth/proxy/TLS only); **everything else from your shell is blanked**. A target needing `OMP_NUM_THREADS`, `R_HOME`, a service key, etc. must name it here. |
| `models` | `[claude-opus-5]` | Benchmark models. Duplicates rejected. `models[0]` is the default for the four `*_model` keys below. |
| `n_replicates` | `3` | Runs per (model, task, split) cell. |
| `max_concurrency` | `4` | Simultaneous benchmark agents. |
| `max_turns` | `40` | **Benchmark agents only.** |
| `max_usd` | `3.0` | **Benchmark agents only.** |
| `draft_model` / `improve_model` / `tasks_model` / `ship_model` | `models[0]` | Meta-agent models; overridable per command with `--model`. |
| `skill_name` | repo basename, slugified + lowercased | Must equal the `name:` in the skill's frontmatter. |

The scaffold's `models:` line lists three models — with `n_replicates: 3` that is 18 runs
per task per arm. Cut it down before the first real pass.

## tasks.yaml

```yaml
tasks:
  - id: some_analysis            # unique, filesystem-safe: [A-Za-z0-9._-]; used as a path component
    train:
      prompt: >-
        One paragraph: the goal, the input, and exactly what to report.
      answer: ONE_TOKEN
    test:
      prompt: >-
        The same analysis on a different input / target.
      answer: ANOTHER_TOKEN
    # optional per-task overrides — the only other allowed keys:
    # max_turns: 60
    # max_usd: 5.0
    # model: claude-sonnet-5
```

Both `train` and `test` are required, each with non-empty `prompt` and `answer`. Both
splits always run in a pass; only train results ever reach `improve`.

### Writing a task that measures anything

- **Answers are graded by exact string match after `strip()`, case-sensitive.** Keep the
  answer to one token: a name, a category, a count, a number at a stated precision. End the
  prompt by stating exactly what to report and in what form.
- **Do not name the target package**, and do not mention a version — the harness preamble
  already tells the agent the package is installed and which interpreter to use.
- **State the goal, not the recipe.** No numbered steps, no function/argument names, no
  description of the data's shape or columns. What is being measured is whether the agent
  can find the "how" itself; a prompt that spells out the calls measures nothing.
- Train and test must be **the same analysis on different inputs**, with different correct
  answers — otherwise a skill can pass by memorizing one answer.
- A good task is one the target package solves and the **no-skill baseline gets wrong** —
  a task the baseline already passes leaves no room to show a gain. The `--no-skill` arm is
  what tells you; it can be run before or after the first skill exists.
- Renaming an `id` orphans every existing run under it.

## `acumen tasks` — generate tasks.yaml

```bash
acumen tasks [--out tasks.yaml] [--force] [--feedback "skip the plotting API"] \
             [--model M] [--max-turns N] [--max-usd X] [--stream] [--log-dir logs]
```

One autonomous agent that reads the package source **and executes it in the target venv**:
every ground-truth answer comes from a script it actually ran, not from docs. It enumerates
the package's tutorials/vignettes and writes at least one task per tutorial, so expect a
sizeable file. Unbounded in turns and cost by default.

- Refuses to overwrite an existing `--out` without `--force` (the check runs *before* the
  costly target prep).
- **There is no append mode.** The generator never reads your existing tasks file, so
  appending would silently duplicate. To combine, generate to a separate path and merge by
  hand.
- Existing skills and agent-instruction files (`SKILL.md`, `.claude/`, `CLAUDE.md`,
  `AGENTS.md`, `.cursor/`, root `skills/`) are stripped from the source copy it reads and
  blocked by a hook, so pre-written guidance cannot bias which tasks it picks.
- The output is validated through the strict tasks loader before it is written — it can
  never emit a file the rest of the pipeline would reject.
- **Review what it produces.** The answers are only as good as the scripts it ran.
