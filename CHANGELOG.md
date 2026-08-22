# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [0.0.1dev]

### Added

- Review each task for internal consistency in `acumen check`, in a second phase after the
  reproducers run: one agent reads every split's prompt, recorded answer and reproducer
  together and adds an `ok`/`mismatch` column, with one line naming the contradiction and one
  naming the fix. Reproducing an answer only proves the code and the answer agree; a prompt
  asking for something else — ordering, statistic, group, count, or answer format — fails every
  agent that reads it correctly, and costs a whole pass to discover. The reviewer reads a staged
  copy of the task set with no path back to `tasks.yaml`, never edits a task, and picks its model
  from the new `check_model` config key. `--no-review` runs the reproducers alone and spends
  nothing.
- Add `acumen check`, which verifies every task's ground truth by rerunning the script that
  produced it. Each task keeps a reproducer at `tasks/<id>-<split>.py` that redoes the
  analysis in the target venv and writes its answer to `answer.md`, graded by the same
  comparison a benchmark run gets. The command reports a row per task and split, the summary
  statistics (share of the task set with a reproducer, share that reproduces, tasks that
  reproduce on both splits), and exits non-zero if anything did not, so a wrong answer or a
  broken pipeline is caught before a pass pays to discover it. It probes that the package
  imports at all before running anything, since that failure would otherwise be reported once
  per task.
- Keep the scripts `acumen tasks` runs to obtain each answer instead of discarding them: the
  generator now writes one reproducer per split into `--scripts` (default `tasks/`), and
  reports any split it left without one. An answer nothing can recompute is an answer nobody
  can check.
- Add an optional task-level `needs_script` key to `tasks.yaml`, defaulting to true. A task
  that needs no code to answer sets it false and is reported as not applicable by
  `acumen check` rather than as a missing reproducer.

- Run Claude and Codex models side by side in benchmark matrices and use either
  provider for drafting, improving, task generation, and shipping.
- Compute `cost_usd` from each run's token breakdown rather than the provider's own
  figure, so both providers are priced by one arithmetic path and cached input is billed
  at its own rate. The rates used are frozen into `result.json`. Where a backend reports
  dollars of its own, that figure is recorded beside it as `provider_cost_usd` (the report's
  sidecar CSV calls it `recorded_cost_usd`) together with its distance from the inferred
  value, but nothing plotted, printed or summed reads it: a Claude SDK total covers nested
  subagents its own usage block does not, so a console tally on that basis would disagree
  with the report it summarises. A model no layer prices stays unpriced even when the
  provider reported dollars, and the report warns at the top, naming the models that need a
  `prices:` entry.
- Add `acumen prices` to show the rate table and `acumen prices --refresh` to diff it
  against the providers' published pricing, plus a `prices:` config key to override it.
- Ship no rate table at all: rates are read from the providers' pricing pages on every
  command that prices something, since a table compiled into a release is wrong from
  whatever date prices next move, and each run's cost is frozen when written rather than
  corrected later. `prices:` in `config.yaml` still overrides, and still wins.
- Fail `acumen bench` when the pricing pages cannot be read, before anything is spent:
  cost is a headline metric of the report, so a pass that cannot establish rates should
  not run. `draft`, `improve`, `tasks`, and `ship` degrade to unpriced instead, warning
  that Codex's `max_usd` cannot be enforced without rates.
- Record `price_source` and `price_rates_as_of` next to `price_rates` in every
  `result.json`, so runs benchmarked months apart remain individually attributable and a
  single report can mix them without restating either.
- Flag arms in a report that were priced on different dates: the cost difference between
  them includes any change in provider pricing, not only the skill's effect.
- Render an HTML transcript for Codex runs too, from the `codex exec` event stream.
- Run the whole comparison from one `acumen bench`: with no arm selected it now covers
  every arm the project has — the baseline plus each version in `skills/` — benching them
  one after another against a single prepared target, with per-arm counts and tallies and a
  combined total. `--dry-run` plans the same set for free, and `--no-skill` / `--skill vN`
  still restrict the pass to one arm. A version in `skills/` that fails to load stops the
  pass at planning rather than being dropped from the comparison.

### Changed

- Order the report's cost-vs-success figure with arrows instead of marker shapes. Skill
  versions are a sequence, so each model's own marks are now joined baseline to v1 to v2 and
  on, in that model's colour and never crossing to another model, with the pooled grey marks
  carrying their own chain. Shape is left to say only whether a mark is a skill or the
  baseline, which frees the version labels off the panel entirely: nothing is named in place,
  and the key holds two marks and an arrow however many versions ran. Every hop is drawn and
  every hop is straight, leaving one mark's rim and landing on the next; two versions that
  landed on the same result simply hide their arrow under the overlap, which is the reading.
  The pooled mark is now the size of every other one, set apart by its colour and its error
  bars alone, since drawn larger it read as a bigger measurement rather than a summary.
- Run the Pareto staircase out to the right edge of the cost-vs-success panel, and dash it. It
  stopped at the dearest frontier mark, which left the stretch beyond it looking like open
  ground when paying more than the best mark cannot buy less than it did; dashing separates a
  line no run lies along from the arrows, which are drawn between marks that do.

- Make both backends optional, so a Claude-only and a Codex-only install are each complete:
  the Claude Agent SDK moves to the `claude` extra (`pip install acumen[claude]`, or
  `acumen[all]`) and Codex needs only its CLI on `PATH`. Selecting a model whose backend is
  not installed fails preflight with the command that installs it.
- Enforce `max_turns` and `max_usd` for Codex, which has no cap of its own, from its event
  stream. Turns are counted in completed model actions rather than `codex exec` invocations —
  one invocation is a single Codex turn, so the old count was always 1 — and the run is
  stopped at the cap. `max_usd` can only mark the outcome: Codex reports usage when a turn
  ends, so an over-budget run is recorded as a `budget` failure after the spend.

- Allow `acumen bench --auth {auto,session,api}`, defaulting to the provider's subscription
  like every other command. The old API-only rule existed because a per-run `cost_usd` needed
  metered billing; cost is now derived from token counts, which a subscription run reports
  just as fully. `result.json` records the run's `auth_mode`, since under `session` the figure
  is what the run would have cost at API rates rather than metered spend.
- Treat exhausted provider subscription usage and API credit as benchmark-invalid
  infrastructure failures: print the provider error, cancel only that provider's remaining
  cells while other providers finish, exit non-zero, and refuse to report or improve from the
  invalid evidence. Resuming automatically retries the invalid and cancelled cells after the
  credential is replenished.

### Fixed

- Remove the target package's own agent guidance from the venv before any agent runs, so the
  baseline arm is really skill-free. A package can ship a first-party skill inside itself
  (`site-packages/<pkg>/_skills/data/SKILL.md` plus a `references/` tree, the shape `acumen ship`
  produces), and although nothing registers it and no prompt mentions it, an agent that greps the
  venv it was handed finds it: measured across four passes and 4608 runs, 18.2% of baseline runs
  read it and 12.2% of skill-arm runs read both it and the skill under test, concentrated in
  exactly the tasks that discriminate between arms and varying fiftyfold by model. So the
  comparison the whole benchmark rests on was partly against a skill nobody chose. `prepare_target`
  now scrubs the finished venv of skill directories, `SKILL.md`/`CLAUDE.md`/`AGENTS.md`/Copilot
  instructions, `.claude`/`.agents`/`.codex`/`.cursor`/`.claude-plugin` trees, and any console
  script left pointing into what it removed; only agent-facing data goes, never code, so the
  package imports and behaves exactly as installed. A `skills/` directory holding code is kept —
  a directory only counts as guidance when a `SKILL.md` sits somewhere beneath it. The scrub is
  idempotent and also runs on a cache hit, so a venv built by an earlier version is cleaned in
  place rather than needing `--refresh-target`. The source checkout is deliberately untouched:
  `ship` commits from it, and for a local target it is the user's own working tree. `draft` instead
  joins `tasks` in reading a filtered copy of the checkout, since a skill drafted from the
  maintainer's own skill is not the independent artifact the report presents. The host user's skill
  catalog is left alone — it is a realistic environment and identical across arms.

- Close a Claude run's session when the run is over, and record the result that run produced. A
  terminal result was not the end of a session: a Bash command the agent left running keeps the
  CLI alive, and when it finishes the CLI queues the notification as a *new* prompt and re-enters
  the model. The old loop kept assigning whatever result arrived last, so one measured run of
  1117 seconds over 41 turns that hit its cap was recorded as a 4-second, two-turn success, with
  inferred cost 96% low and the answer graded `no_answer_file`. The re-entry was also past the
  turn cap, where the CLI answers every tool call with a cancelled-permission denial — including
  `echo "test"` and reads inside the run's own working directory — so the agent sat waiting for
  an operator who does not exist while the harness had already moved on to grading. Two of 44
  background-task runs lost their result to this; it was a race on whether a notification landed
  before the process was torn down. The Claude backend now runs through `ClaudeSDKClient` and
  stops reading at the first result, so the turns, duration, usage and cost a run is judged on
  are the graded prompt's by construction. Teardown then stops every outstanding background task,
  waits briefly for the CLI to confirm each is gone so its output file is flushed before the
  artifacts are collected, interrupts the turn, and disconnects — on every exit path, including a
  crash. Work the agent abandoned after declaring itself done is not resumed. The sandbox-path
  denials are unchanged: those are the containment guard doing its job, not this.
- Give a sandboxed Bash command 600 seconds before the CLI moves it to the background, up from
  the default 120, with a 1800-second ceiling an agent can still ask for. A command that outruns
  its timeout is not failed but backgrounded, and a benchmark target downloads its own datasets
  and priors — one such fetch measured over 300 seconds — so almost every one of them was being
  backgrounded, which is what created the session-lifecycle problem above in the first place.
- Record what a capped or crashed Codex run actually spent. `codex exec --json` reports usage
  once, in `turn.completed`, which a run acumen stops at its turn cap never reaches, so every
  turn-capped run recorded zero tokens and a cost of `$0.00` after minutes of real work. acumen
  now follows the running total Codex writes to its rollout session file, which means dropping
  `--ephemeral` so that file exists; it lands inside the run-local `CODEX_HOME` and is discarded
  with the sandbox. A shipper run given no config directory will now leave a session in the
  operator's own `~/.codex`, as a plain `codex exec` would. A capped run is also stopped with an
  interrupt rather than killed outright, which is what lets Codex finish writing the record for
  the response that has just landed; a hard kill loses it. The recovered figure is still a lower
  bound, since the very last response can be cut off before it is recorded, but a measured
  turn-capped run went from `$0.00` on nothing to a real cost on 30k real tokens.
  `turn.completed` stays authoritative whenever it arrives, and a rollout that cannot be read or
  parsed leaves the run exactly as it was before. The same total is what `max_usd` is now checked
  against, once per model response rather than once per turn, so a Codex budget cap stops the run
  partway through the turn instead of only labelling the overspend after the fact. A single
  response can still overshoot.
- Grade a capped run on the answer it managed to write. A run stopped at its turn or budget cap
  was failed on the cap alone, so an agent that had already written a correct `answer.md` before
  being cut off was recorded as a failure with its answer sitting unscored in `result.json`. The
  cap now defers to the grade whenever there is an answer to grade; with no `answer.md`, or an
  empty one, the run still fails as `max_turns` or `budget`. The cap remains visible in the run's
  `subtype` and error list. Applies to Claude and Codex alike, and does not change how a crash, a
  failed sandbox or an exhausted account is classified: those still override the grade.
- Give isolated agents unrestricted internet again. Sandboxing every run also put an egress
  policy in front of it, and the policy denied the hosts a target actually needs: EBI, Zenodo,
  figshare, OmniPath, NCBI and cellxgene were all unreachable while GitHub and PyPI were not.
  A refused host does not stop an agent — it improvises from memory and returns a confident
  wrong answer — so the runs scored as evidence that the models had got worse. Codex now runs
  its proxy in `full` rather than `limited` mode, which had also been rejecting every request
  that was not GET, HEAD or OPTIONS. Claude runs without an OS sandbox, whose proxy cannot be
  opened from the settings file the SDK passes, and is confined to its sandbox directory by a
  tool-layer guard instead: it can use system paths and the target venv, and cannot list or
  read anywhere else on the host.
- Record a run whose sandbox could not execute anything as an infrastructure failure rather
  than a wrong answer. bubblewrap reports its own startup failure as the command's output, not
  on the agent CLI's stderr, so the existing check never saw it and a run where every single
  command died still entered the report as a graded result.
- Let `acumen tasks` generate over the untouched `tasks.yaml` placeholder that `acumen init`
  writes, instead of demanding `--force` — the two documented first steps of the loop
  contradicted each other. A file the user has edited is still protected.
- Stop feeding Codex transcripts to `claude-code-log`, which reads the SDK-native format
  only: it skipped every line, exited 0, and wrote an empty page that was then recorded as a
  successfully rendered transcript.
- Drop `Claude` from the drafting and improving prompts, which described the artifact as a
  "Claude Skill" even when a Codex agent was writing it for a non-Claude skills directory.
- Remove `check_auth`/`auth_available`, which only ever looked for Claude credentials and
  would report a Codex-only setup as unauthenticated. `resolve_auth_mode` replaced them.
