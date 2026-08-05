# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [0.0.1dev]

### Added

- Run Claude and Codex models side by side in benchmark matrices and use either
  provider for drafting, improving, task generation, and shipping.
- Compute `cost_usd` from each run's token breakdown rather than the provider's own
  figure, so both providers are priced by one arithmetic path and cached input is billed
  at its own rate. The rates used are frozen into `result.json`.
- Add `acumen prices` to show the rate table and `acumen prices --refresh` to diff it
  against the providers' published pricing, plus a `prices:` config key to override it.
- Render an HTML transcript for Codex runs too, from the `codex exec` event stream.

### Changed

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
