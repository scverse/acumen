# How rulebook evolution works

`acumen fit` improves one skill, epoch by epoch. `acumen evolve` asks a different question: *does the
procedure that improves skills generalise?* This page explains the machinery. For the step-by-step
procedure and the rules for running it unattended, see [Evolving a rulebook](evolve.md).

## Why a second loop

Two things make a single-skill result hard to trust.

1. **Drafting is noisy.** A skill is written by an agent. The same instructions drafted twice produce
   skills that differ by several percentage points on the same tasks, and that noise is a per-skill
   offset: benchmarking on more tasks does not average it away, only drafting more skills does.
2. **The thing you select on leaks.** If the same tasks pick the winner and report its score, the score
   is optimistic.

So `evolve` (a) optimises the *instructions* the skill is drafted from, and measures every candidate as
a mean over several independent drafts, and (b) keeps a write-once hold-out that nothing is ever
selected on.

## The pieces

**Rulebook.** The template text the drafting agent's prompt is built from: `rulebooks/vN/rulebook.md`.
The built-in draft prompt is the seed (`v1`), and it keeps the placeholders (`{package}`, `{src}`,
`{out}`, `{skill_name}`, …) a draft needs. Each version is content-hashed, immutable, and has a
`meta.json` with its parent, rationale and hash, so a score is attributed to *this text*, and the chain
of edits can be walked from any rulebook back to the seed. The rulebook is what evolves; **skills are
intermediates** drafted fresh from whichever rulebook is on trial.

**Draft.** `acumen draft --rulebook R` runs the drafting agent on the target's source under rulebook
`R` and writes a skill. Drafting the same rulebook into several `--skills` roots gives it a mean and a
spread.

**Screen.** A small subset of the working tasks (`--screen-size`), re-sampled every few generations
(`--epoch-len`) so the search cannot overfit one subset. Cheap, and allowed to be noisy.

**Champion.** Two of them. The *screen champion* is whatever last won a screen. The *confirmed
champion* is what improvement actually builds from, and it only changes after a full-benchmark
confirmation.

**Lockbox.** A hold-out of tasks written once by `acumen lockbox`, digest-verified, and never touched by
any screen, confirmation or improve agent. It is opened once, at the very end.

**Headroom.** A skill can only be shown to help on tasks the bare model does not already solve.
`--headroom` reads the no-skill floor from `runs/` and restricts the search to tasks the baseline fails
(or passes only sometimes).

## One generation

Every generation does the same thing, and every step is a pure function of `(generation, seed)`, which
is what makes a run resumable.

1. **Pick the directive.** Directives rotate: make one focused edit; restructure boldly; delete
   aggressively; concretise into runnable recipes; fix loading and routing; question an assumption;
   generalise; or free choice. A search that only ever adds bullets converges on bloat, so some
   directives push for large moves and for deletion. The directive is recorded in the new version's
   `meta.json`.
2. **Improve from the champion, always.** The improve agent reads the champion's *train-split* failure
   evidence on the tasks it may learn from, and rewrites the rulebook under the directive. A rejected
   candidate keeps its directory (provenance) but nothing ever descends from it.
3. **Draft and screen.** The candidate rulebook is drafted into a skill `--screen-drafts` times. The
   champion and the candidate are benchmarked on the *same* screen subset.
4. **Accept or reject.** The candidate becomes the screen champion if it beats the champion by at least
   `--accept-delta` passes per draft, i.e. `candidate_passes − champion_passes ≥ accept_delta ×
   screen_drafts`. The bar exists because a screen is noisy by construction; a tie must never churn
   the champion.
5. **The ratchet.** Every `--confirm-every` accepts, the screen champion is benchmarked on the *full*
   working pool. If it beats the confirmed champion it is promoted; if not, the search reverts. Screen
   decisions may thrash; the ratchet may not.
6. **Journal.** One JSON line is appended to `runs/evolve.jsonl`: the directive, parent and candidate
   versions, the screen subset, both screen scores, the decision, whether a confirmation ran, the
   confirmed champion, and the cost. Hundreds of these are the dataset behind any claim about *which
   kinds of edits help*.

Nothing the improve agent can reach includes the lockbox, the test split of any task, or the runs of
tasks held out from it; this is enforced twice, by what is copied into its working directory and by a
`PreToolUse` guard that denies any call naming a held-out path.

## Islands

One search can find a lucky edit. `--islands K` runs `K` independent searches, each from the same seed
rulebook, on **disjoint partitions of the working tasks**. They never see each other's tasks or runs.

When they finish, a *pollination* agent reads each island's champion rulebook, its decision journal and
one unified diff per surviving edit, and writes the next rulebook keeping only guidance that
**replicated across at least two islands**. Its rationale is the list of meta-rules with the islands
that support each, recorded in the merged version's `meta.json`. Replication across independent task
partitions is the evidence standard.

The merged rulebook is then validated on the full working pool against the seed, and only then is the
lockbox opened — once.

## The verdict

The lockbox is scored for the seed and for the merged champion, each over `--drafts` independent drafts,
alongside the bare model on the same tasks. `acumen verdict` turns those run trees into:

- **mean ± spread over drafts**, because the draft is the unit of measurement, not the task;
- **a paired sign test** against the baseline over the tasks where the two disagree;
- **a hard/easy split**, because the tasks the bare model already passes are where a skill *regresses*,
  and an overall rate hides that.

A negative result on a powered hold-out is a result. `acumen verdict` says *NOT distinguishable from
noise* when that is what it finds.

## Resuming and determinism

Everything resumes by file presence. A rerun of the identical command replays every finished decision
from the scores on disk without spawning an agent for finished work. That is why a flag must never be
changed on a resumed run: the new decision rules would apply on top of generations decided under the old
ones. Start fresh `--runs/--skills/--rulebooks` trees instead.

A run killed by the platform (a usage limit, an overloaded API, a dropped network) is never recorded as
a failed task. The run is stored as invalid (`valid: false`), stays pending, and the command stops with
**exit code 3**; rerun the identical command to resume it. `acumen status` counts these and flags the
one dangerous case: a platform failure recorded as an ordinary failure, which would silently drag an arm
down.

## What it writes

```text
config.yaml  tasks.yaml                 # your inputs
lockbox/  lockbox2/                     # write-once hold-outs (tasks.yaml + a digest manifest)
rulebooks/
  v1/ rulebook.md  meta.json            # the seed, then every version the search wrote
  diffs/                                # unified diffs between a version and its parent
  islands/island-<i>/ ...               # each island's own chain, with --islands
skills/ v1/ SKILL.md  references/       # skills drafted from the rulebooks (intermediates)
  drafts/<version>/d<n>/ ...            # extra drafts of one version, for its mean and spread
runs/
  noskill/ skill_v1/ ...                # <arm>/<split>/<model>/<task>/rep_<n>/result.json
  evolve.jsonl                          # the decision journal
  islands/island-<i>/evolve.jsonl       # one journal per island
  lockbox/                              # the one-time hold-out evaluation
  drafts/<version>/d<n>/ ...            # benchmark runs of the extra drafts
logs_*/                                 # a live JSONL (and HTML) log per agent
```

## How it relates to `fit`

| | `acumen fit` | `acumen evolve` |
| --- | --- | --- |
| What changes | one skill, improved in place from a knowledge wiki | the rulebook the skills are drafted from |
| A candidate is scored as | one skill | a mean over `--screen-drafts` drafts |
| Selection | validation split, early stopping | rotating screens + a confirmed ratchet |
| Hold-out | the `test` split of each task | a separate lockbox of whole tasks |
| Use it to | get a good skill for your package | measure whether the improvement procedure generalises |

The two loops share the benchmark, the sandbox, the report and the shipping step, and they are
independent: a rulebook evolved here can seed a skill that you then ship with `acumen ship`.

## Commands

| Command | What it does |
| --- | --- |
| `acumen mine` | Harvest real analyses that use your package (GitHub code search through `gh`, plus `--url` pages) into candidate scripts. |
| `acumen tasks --candidates DIR` | Turn mined candidates into tasks, one agent each, resumable per shard. `--per-notebook` shards on the tutorial notebooks instead. |
| `acumen lockbox` | Carve a write-once hold-out out of a task set. |
| `acumen coverage` | Which public API symbols the tasks exercise; with `--skill`, what a skill teaches versus what is verified. Spends no tokens. |
| `acumen warm` | Pre-download the datasets the tasks' reproducers load into a shared per-target cache (`dataset_cache_dirs`). |
| `acumen draft` | Draft a skill from the package source, optionally under a rulebook. |
| `acumen evolve` | The generational search described above. |
| `acumen status` | Health check of a running or finished experiment. Exit 1 means a `PROBLEM` line. |
| `acumen verdict` | The comparison behind the answer. |

The key `evolve` flags: `--generations`, `--screen-size`, `--screen-drafts`, `--epoch-len`,
`--accept-delta`, `--confirm-every`, `--headroom`, `--islands`, `--drafts`, `--lockbox` / `--no-lockbox`,
`--seed`, `--max-hours`. Run `acumen evolve --help` for the full list.

## Using it on your own package

```bash
acumen init                                 # config.yaml and a starter tasks.yaml
acumen mine --config config.yaml --out mined
acumen tasks --config config.yaml --candidates mined --out tasks_all.yaml --shards-dir tasks.shards
acumen lockbox --tasks tasks_all.yaml --out lockbox --working tasks.yaml --fraction 0.3
acumen warm --config config.yaml --tasks tasks.yaml
acumen bench --config config.yaml --tasks tasks.yaml --runs runs --no-skill --split test
acumen evolve --config config.yaml --tasks tasks.yaml --lockbox lockbox \
  --islands 3 --generations 20 --headroom --screen-drafts 3
acumen status runs
```

With no `rulebooks/` directory, `evolve` seeds `v1` from the built-in draft prompt. Authenticate with
your Claude or Codex login (`--auth session`) or an API key (`--auth api`); a subscription has usage
windows, so a long run will pause (exit 3) and resume when you rerun the identical command.
