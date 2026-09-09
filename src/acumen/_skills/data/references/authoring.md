# Authoring skill versions: epoch, wiki, improve, hand-edit

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

## `acumen epoch` — the loop

```bash
acumen epoch [--feedback "…"] [--model M] [--max-concurrency N] [--replicates N]
             [--stream] [--log-dir logs] [--auth auto|session|api]
```

One training epoch, end to end and fully resumable:

1. **Bench the current arm on `train`** (the first epoch also benches the `noskill` baseline
   on `valid`, for the report).
2. **Update the wiki** for that arm — one agent per task (see `acumen wiki` below).
3. **Create or improve the skill** into the next version, from the wiki (see `acumen improve`).
4. **Bench the new version on the held-out `valid` split.**

Re-run it: a crashed epoch resumes at the step it stopped (state is read from disk — a version
present but not valid-benched means "finish that epoch"; a fully benched latest means "start the
next"). A completed epoch's next invocation learns from the version it just produced.

## `acumen wiki` — the knowledge base

```bash
acumen wiki [--no-skill | --skill vN] [--feedback n/a] [--model M] [--max-concurrency N]
            [--stream] [--log-dir logs] [--auth auto|session|api]
```

For each task, one agent reads that arm's **train** runs across models and replicates and
**appends** a terse block to `wiki/<task>/observations.md` and `wiki/<task>/hypothesis.md`, tagged
`[version][model]` (`noskill` first, then `v1`, `v2`, …). Cumulative and idempotent: an arm
already recorded for a task (tracked in `wiki/<task>/.arms`) is skipped. Defaults to the latest
arm; `--no-skill` records the baseline, `--skill vN` a specific version. Entries are kept short on
purpose — the whole wiki is read by the improver every epoch.

## `acumen improve`

```bash
acumen improve [--from vN] [--feedback "…"] [--model M] [--max-turns N] [--max-usd X]
               [--stream] [--log-dir logs] [--auth auto|session|api]
```

Reads the **wiki** (the distilled per-task notes) plus the **filtered** package source, and
either creates the first skill (when no version exists yet, from the `noskill` wiki) or edits a
copy of the parent into the next version.

- Defaults to the latest version; `--from vN` picks another. Always writes the next unused
  directory. With no versions it creates `v1`.
- Requires the parent arm's `runs/<arm>/train/**/result.json` to exist (so run an epoch, or bench
  the arm and `acumen wiki`, first).
- The agent reads a staged copy of the wiki + the parent arm's train transcripts (for drill-down).
  The real `runs/` tree is denied, and a `PreToolUse` hook denies any path under `runs/*/valid/`.
  The source is the *filtered* copy — a skill the package itself ships is stripped and blocked, so
  the optimization never re-serves it.
- The CLI warns if the new version is byte-identical to its parent (the improver changed nothing).

## `--feedback` on `tasks` / `improve` / `epoch`

Free text injected into the agent's prompt as *subordinate* guidance — it never overrides
the isolation or anti-overfit rules. Use it for what the agent cannot infer: package
context, what to emphasise, which functionality to skip. For `improve` it is recorded in
`meta.json` and shown in the report. **Never paste valid-split answers into `--feedback`** —
that defeats the split.

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
  tasks.** That is overfitting, and the valid split will catch it.
- Verify every claim against the installed package before writing it down.

## Watching a long agent run

`tasks`/`wiki`/`improve`/`ship` each write `logs/acumen-<command>-<YYYYMMDD-HHMMSS>.jsonl`
(the wiki writes one per task, `acumen-wiki-<task>-…`),
one compact event per agent message, **flushed as it goes** — read that file to follow
progress instead of streaming into your context. Tool results are recorded by status and
size, not inlined. `--stream` mirrors the conversation to the terminal; `--log-dir` moves
the logs. At the end, the run is mapped into acumen's harness-neutral trajectory model and a
rendered `.html` transcript plus a portable `.trajectory.json` land beside the jsonl — the same
model and renderer for every provider, so the reports look alike.
