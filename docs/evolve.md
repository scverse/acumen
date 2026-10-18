# Evolving a rulebook: protocol and operating rules

`acumen evolve` searches over a **rulebook** — the instructions a skill is drafted from — instead of over
one skill. This page is the protocol for getting a powered answer to one question on any target
package:

> Does an evolved rulebook beat (a) the bare model and (b) the seed rulebook, on a hold-out of at
> least 100 tasks that nothing was selected on?

It is written for a person or an unattended agent. A clean negative is a result; report it.

Two facts shape everything below.

1. **The unit of measurement is the draft, not the task.** The same rulebook text drafted into a skill
   twice scores differently (on the experiment this protocol came from, sd ≈ 8 percentage points). That
   noise is a per-skill offset, so running *more tasks* does not shrink it — only *more drafts* do.
2. **A skill can make things worse.** Report regressions on tasks the bare model already passes
   separately from gains; an overall rate hides them.

## What you need

- A target package, and a `config.yaml` for it (`acumen init` scaffolds one).
- A task set with ground-truth answers (`tasks.yaml`, plus the reproducer scripts that produced them).
- A seed rulebook under `rulebooks/v1/rulebook.md`. `acumen.rulebooks.seed_default` writes the built-in
  draft instructions; a previously evolved champion works too.
- A credential: a logged-in Claude or Codex subscription (`--auth session`, which `--auth auto` picks
  when you are logged in) or an API key (`--auth api`). Both work. The difference is operational: a
  subscription has usage windows, so a long run will pause (exit code 3, below) when one is spent and
  you rerun the identical command after it resets; an API key with a high rate limit runs straight
  through, and `max_concurrency` becomes the only limit. Raise it as far as your limit allows.
- The [`gh` CLI](https://cli.github.com), logged in, if you mine candidates (`acumen mine`).
- Disk for the target's checkout, its venv and its datasets.

Every command runs from one working directory, with the same `config.yaml`.

## Commands

| Command | Purpose |
| --- | --- |
| `acumen mine --out DIR` | Harvest real analyses that use the target (GitHub code search through the `gh` CLI, plus any `--url` pages) into candidate scripts. Candidates are kept only if they call the package's public API. |
| `acumen tasks --candidates DIR --out F --shards-dir D` | One agent per mined candidate script (or `--per-notebook` for tutorials); resumable per shard; merged into one task file. |
| `acumen lockbox --tasks F --out DIR --working W --fraction 0.5` | Split a task set into a write-once, digest-verified hold-out and a working pool. |
| `acumen coverage --config C --tasks F` | Which public API symbols the tasks exercise, and (`--skill vN`) what a skill teaches versus what the benchmark verifies. No tokens. |
| `acumen warm --config C --tasks F` | Pre-download the datasets the reproducers load into a shared per-target cache, so concurrent runs never race the first download. |
| `acumen bench --no-skill / --skill vN --split test` | Bench one arm. Completed runs are skipped, so rerunning only costs missing cells. |
| `acumen draft --rulebook R --skills DIR` | Draft a skill from a rulebook. Drafting one rulebook into several `--skills` roots gives it a mean and a spread. |
| `acumen evolve ...` | The generational search; see below. |

`acumen status` and `acumen verdict` turn a run tree into a health check and a verdict.

## Stage 1 — grow the benchmark

A 36-task hold-out cannot resolve a 5-task effect; at 100 paired tasks a +5pp effect is detectable.
Mine candidate analyses for your target, turn them into tasks, and carve a fresh hold-out (mining needs
the [`gh` CLI](https://cli.github.com), logged in with `gh auth login`):

```bash
acumen mine --config config.yaml --out mined        # skip if you already have a mined/ directory
acumen tasks --config config.yaml --candidates mined --out tasks_new.yaml \
  --shards-dir tasks_new.shards --log-dir logs_new
acumen lockbox --tasks tasks_new.yaml --out lockbox2 --working tasks_new.working.yaml --fraction 0.5
```

Expect a large fraction of candidates to be discarded as ungradeable; that is correct. Merge the
working half into your pool and assert both hold-outs stay disjoint from it. Stop when the hold-out is
big enough, not when the candidates run out. The hold-out is write-once: **do not benchmark against it
until the verdict.** Every look spends its independence, so a hold-out that has been opened repeatedly
is a validation set, not a clean hold-out.

## Stage 2 — re-establish the floor, over drafts

Use one run tree for everything from here: `evolve --headroom` reads the no-skill floor from its own
`--runs` root, and a floor benched elsewhere is invisible to it.

```bash
acumen warm  --config config.yaml --tasks tasks_pool.yaml
acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs --no-skill --split test
for d in 1 2 3; do
  acumen draft --config config.yaml --rulebook rulebooks/v1/rulebook.md --skills champ_d$d
  acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs_champ_d$d \
    --skill v1 --skills champ_d$d --split test
done
acumen verdict noskill=runs/noskill champ.d1=runs_champ_d1/skill_v1 \
  champ.d2=runs_champ_d2/skill_v1 champ.d3=runs_champ_d3/skill_v1
```

The seed, scored as a mean over three drafts, is the number the evolved rulebook has to beat. The
per-draft spread sets `--accept-delta` for Stage 3: roughly half the spread in passes per screen, and
never below 2. **Write the spread down — it is a result in itself.** If the three drafts spread by more
than 15 percentage points, stop and ask a human before evolving on this target.

## Stage 3 — evolve, and the verdict

`evolve` runs for hours to days. First a throwaway sanity run in its own trees, with the hold-out
sealed, and check the projected spend:

```bash
cp -R rulebooks rulebooks_sanity
acumen evolve --config config.yaml --tasks tasks_pool.yaml \
  --rulebooks rulebooks_sanity --skills skills_sanity --runs runs_sanity \
  --generations 2 --screen-size 12 --screen-drafts 3 --accept-delta 2 \
  --no-lockbox --log-dir logs_sanity
```

`runs_sanity/evolve.jsonl` should have two lines whose screen `total` is 36 (12 tasks × 3 drafts).
Delete every `*_sanity` tree afterwards. Then the real run, which evolves, merges the islands,
validates the merge on the full working pool, and only then opens the hold-out — once — scoring the
seed and the merged champion over three drafts each:

```bash
acumen evolve --config config.yaml --tasks tasks_pool.yaml \
  --rulebooks rulebooks --skills skills --runs runs \
  --islands 3 --generations 40 --headroom \
  --screen-size 24 --screen-drafts 3 --accept-delta <from stage 2> --confirm-every 3 \
  --lockbox lockbox2 --drafts 3 --log-dir logs_evolve
```

- `--screen-drafts N` drafts each candidate N times and judges the sum, so a generation is accepted for
  signal rather than draft luck. `--accept-delta` still means passes *per draft*.
- `--islands K` evolves K rulebooks independently on disjoint task partitions, then cross-pollinates:
  only edits that replicated across at least two islands survive the merge. Replication across islands
  is the evidence standard.
- `--headroom` restricts evolution to tasks the bare model fails.
- `--confirm-every N` tightens the ratchet: a screen win becomes champion only after a full-benchmark
  confirmation, and a failed confirmation reverts.
- Resume is free and agent-free: rerun the **identical** command after any interruption.

Each island keeps one line per generation in `runs/islands/island-<i>/evolve.jsonl` — directive, screen
subset, scores and decision. Those files are the dataset behind any claim about *what kind of edits*
help, so keep them. When it finishes it prints `merged into vN`. Bench the bare model on the same
hold-out, then run the verdict:

```bash
acumen bench --config config.yaml --tasks lockbox2/tasks.yaml --runs runs/lockbox \
  --no-skill --split test
acumen verdict noskill=runs/lockbox/noskill \
  seed.d1=runs/lockbox/skill_v1 seed.d2=runs/drafts/v1/d2/lockbox/skill_v1 \
  seed.d3=runs/drafts/v1/d3/lockbox/skill_v1 \
  merged.d1=runs/lockbox/skill_vN merged.d2=runs/drafts/vN/d2/lockbox/skill_v1 \
  merged.d3=runs/drafts/vN/d3/lockbox/skill_v1
```

Report what it prints, including when it says *NOT distinguishable from noise*.

## Operating it unattended

### Hard rules

1. **Never look at the second hold-out early.** No `bench` on it and no `--lockbox` on any run except
   the final evolve run.
2. **Never edit a hold-out.** Hold-outs are write-once and digest-verified.
3. **Never change a flag on a resumed run.** Rerunning the identical command continues it; a changed
   command applies new decision rules on top of generations decided under the old ones. Start fresh
   `--runs/--skills/--rulebooks` trees instead.
4. **Never point two runs at one `--runs` tree**, and never run two agent-spawning `acumen` commands at
   once — they share your rate limit.
5. **Never touch answers or ground-truth scripts** to make a number move, and never paste hold-out task
   text into `--feedback`.
6. **Never delete a `result.json`** unless `acumen status` reports it as a platform-killed run
   recorded as a failure. Say that you did.

### Launch detached, poll

```bash
nohup sh -c 'acumen evolve ...; echo "EXIT=$?"' > evolve.out 2>&1 &
echo $! > evolve.pid
```

When it is no longer alive, `tail -1 evolve.out` shows `EXIT=<code>`:

| Exit | Meaning | Do |
| --- | --- | --- |
| 0 | Finished | Next stage. |
| 3 | Paused: rate limit, overload, exhausted credit or network drop. Nothing bad was recorded. | Wait 10 minutes, rerun the **identical** command. |
| 2 | Configuration or usage error | Read the message, fix, rerun. Do not retry blindly. |
| other | Crash | Read the last 50 lines of the output and the newest `logs_*/*.jsonl`. Stop and report. |

Every 20–30 minutes: `kill -0 $(cat evolve.pid) && echo alive`, `acumen status runs`
(progress per island, spend, skill-load rate; exit 1 means a `PROBLEM` line), and
`tail -3 logs_evolve/*.jsonl`. The status script checks:

- **Platform-killed runs recorded as failures.** Runs killed by a limit or network drop are recorded
  `valid: false` and stay pending; one recorded as an ordinary failure drags an arm down silently.
- **Accept rate above 50%** after six generations: `--accept-delta` is inside the noise. Stop and ask;
  raising it means restarting in fresh trees.
- **Zero accepts in ten generations:** the bar may be out of reach. Stop and ask.
- **Skill load rate below 80%:** a draft may not trigger. Report which one.

Budget from measurement, not from a rule of thumb: after the sanity run, read `acumen status`'s
spend line and extrapolate to the real run's generations, islands, screen size and drafts before
launching it.

### Stop and ask a human when

- any stage ends with exit code 2 or a crash;
- Stage 1 yields fewer than 100 tasks in the second hold-out;
- Stage 2's drafts of the champion spread by more than 15 percentage points;
- `acumen status` reports a problem you cannot resolve under the rules above;
- you want to change the protocol in any way — a different split, model, task filter or flag.

### Report after each stage

The exact commands and their exit codes; `status` and `analyze` output verbatim; anything that broke,
what you did about it, and any `result.json` you deleted. At the end, send back the per-island
`evolve.jsonl` archives, `rulebooks/` (the merged champion's `meta.json` rationale is the meta-rule
list, and each island's chain is under `rulebooks/islands/`), the Stage 2 and verdict `analyze` output,
and the benchmark itself (hold-out, task pool).

## Gotchas

- **A dead run is not a failed task.** Session limits, rate limits, overload and network drops are
  recorded invalid and the pass stops with exit code 3. Run the status script before believing an arm
  that is inexplicably worse than its siblings.
- **Never run two agent fleets at once on one credential.**
- **Agents are only as good as the ground truth.** Some task answers are genuinely ambiguous (a pair of
  names whose order is arbitrary). If a rulebook edit's rationale is "make the agent guess the ordering
  convention", that is the benchmark leaking, not a finding.
- **Skill size is a cost, not a win.** Track size per draft; `acumen report` shows it.
