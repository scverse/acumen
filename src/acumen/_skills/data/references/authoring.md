# Authoring skill versions: draft, improve, hand-edit

## The skill directory contract

```
skills/v1/
  SKILL.md          # required, YAML frontmatter with `name` and `description`
  references/*.md   # optional
  meta.json         # acumen bookkeeping — NOT part of the skill
```

- `name` must equal `config.skill_name` exactly; `description` must be non-empty. Both are
  enforced on load by `bench`, `improve`, and `ship`.
- `meta.json` (`version`, `parent`, `rationale`, `hash`, optional `feedback`) is written by
  acumen. It is excluded from the content hash and is never copied to a consuming agent.
- The content hash covers every other file's relative path and bytes, and is recorded in
  every `result.json` — so editing a version after benching it silently invalidates the
  comparison. **Versions are immutable; always make a new one.**

## `acumen draft`

```bash
acumen draft [--force] [--feedback "…"] [--model M] [--max-turns N] [--max-usd X]
             [--stream] [--log-dir logs] [--auth auto|session|api]
```

One agent that reads the target's **source** (docstrings, examples, docs) and writes
`skills/v1/`. It works in a staging dir; only a skill that loads and validates is promoted,
so a failed draft leaves no broken version behind. Exits 2 if any version already exists —
pass `--force` to add another, or use `improve` to build on the latest.

## `acumen improve`

```bash
acumen improve [--from vN] [--feedback "…"] [--model M] [--max-turns N] [--max-usd X]
               [--stream] [--log-dir logs] [--auth auto|session|api]
```

Reads **how the parent version performed on the train split** — not the package source —
and edits a copy of it into the next version.

- Defaults to the latest version; `--from vN` picks another. Always writes the next unused
  directory.
- Requires `runs/skill_vN/train/**/result.json` to exist: bench the parent first.
- The agent gets a curated copy of train runs (`SUMMARY.md` with failures first, plus each
  run's prompt, expected, actual, `script.py`, and transcript). The real `runs/` tree is not
  in scope, and a `PreToolUse` hook denies any path resolving under `runs/*/test/`.
- The CLI warns if the new version is byte-identical to its parent (the improver changed
  nothing) — that is a signal to give it sharper evidence or `--feedback`.

## `--feedback` on `tasks` / `draft` / `improve`

Free text injected into the agent's prompt as *subordinate* guidance — it never overrides
the isolation or anti-overfit rules. Use it for what the agent cannot infer: package
context, what to emphasise, which functionality to skip. For `draft`/`improve` it is
recorded in `meta.json` and shown in the report. **Never paste test-split answers into
`improve --feedback`** — that defeats the split.

## Writing or editing a version by hand

Perfectly supported; the agents are a convenience, not a requirement. A skill authored
outside the project is benched the same way: copy the directory in as the next unused
`skills/vN` — `bench` only ever loads versions from the project's `skills/` root.

```bash
cp -r skills/v2 skills/v3    # then edit skills/v3/, and delete its inherited meta.json
acumen bench --skill v3
```

A hand-made version with no `meta.json` benches fine; it just has no rationale/diff in the
report. What makes a version score well:

- The `description` decides whether the skill loads at all — name the goals a user would
  actually phrase, honestly. An unloaded skill is a wasted arm.
- Keep `SKILL.md` short and push depth into `references/`; every token is paid on every
  task, including ones where the skill is irrelevant.
- Spend words on what an agent would get wrong by guessing: non-obvious defaults, required
  preprocessing, the function that looks right but isn't, argument shapes/orientation,
  where output lands, the right order of steps.
- **Never name a dataset, parameter value, column, or expected answer from the train
  tasks.** That is overfitting, and the test split will catch it.
- Verify every claim against the installed package before writing it down.

## Watching a long agent run

`tasks`/`draft`/`improve`/`ship` each write `logs/acumen-<command>-<YYYYMMDD-HHMMSS>.jsonl`,
one compact event per SDK message, **flushed as it goes** — read that file to follow
progress instead of streaming into your context. Tool results are recorded by status and
size, not inlined. `--stream` mirrors the conversation to the terminal; `--log-dir` moves
the logs. A rendered `.html` transcript lands beside the jsonl at the end (skipped, with a
note, if `claude-code-log` is missing).
