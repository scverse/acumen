# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [0.0.2]

### Added

- Added `acumen fit --ablation {no-optimize,no-wiki}` to run ablation arms. A report without a `valid` split now uses `test` instead
- Added `acumen prepare` to build task datasets once into `cache/`, and a `data:` key in `tasks.yaml` that links them into each run

### Changed

- Restructured the docs, with a scverse-style README, a Getting started tutorial and one API page per `acumen` subcommand
- `--agent codex` now installs shipped skills into `~/.agents/skills`, where current Codex looks for them
- The report now tests success rate and cost separately, with a Benjamini–Hochberg FDR correction per axis
- Updated the GPT colour ramp in the report so `astra` is the darkest and the other models lighten in even steps
- Merged the per-command model keys (`draft_model`, `improve_model`, `tasks_model`, `ship_model`, `check_model`) into a single `meta_model` key
- Transcripts from every provider are now rendered through one shared trajectory model, and each run also writes a `trajectory.json`. `claude-code-log` is no longer a dependency
- Claude agents now run with the `claude` CLI on `PATH`, falling back to the one bundled in `claude-agent-sdk`
- `result.json` now records the agent CLI's version as `cli_version`, replacing `sdk_version`

### Fixed

- Fixed agent failures reporting a bare "errored: success" instead of the provider's reason
- Fixed `acumen ship --force` leaving the old installer's tests broken. The ship agent now ports them and runs them before delivering
- Fixed `acumen ship` failing to push and open the pull request because the agent could not find the operator's git and `gh` credentials

## [0.0.1dev]

### Added

- `acumen check` now also reviews each task for inconsistencies between its prompt, answer and reproducer. `--no-review` skips this step
- Added `acumen check` to verify each task's answer by rerunning its reproducer at `tasks/<id>-<split>.py`
- `acumen tasks` now keeps one reproducer script per split in `tasks/`
- Added an optional `needs_script` key to `tasks.yaml` for tasks that need no code
- Claude and Codex models can now run side by side in benchmarks and drive every meta-agent command
- `cost_usd` is now computed from each run's token counts, so both providers are priced the same way
- Added `acumen prices` to show and refresh the rate table, and a `prices:` config key to override it
- Rates are now read from the providers' pricing pages instead of a table shipped with the release
- `acumen bench` now fails before spending anything when the pricing pages cannot be read
- `result.json` now records where each run's prices came from and their date
- The report now flags arms that were priced on different dates
- Added HTML transcripts for Codex runs
- `acumen bench` without an arm now benches the baseline and every version in `skills/` in one pass

### Changed

- The report's cost vs. success figure now joins each model's versions with arrows instead of using marker shapes
- The Pareto frontier in the report is now dashed and runs to the right edge of the panel
- Both backends are now optional, installed with `acumen[claude]`, `acumen[all]` or just the `codex` CLI for Codex
- `max_turns` and `max_usd` are now enforced for Codex
- `acumen bench` now supports `--auth {auto,session,api}` and defaults to the provider's subscription
- Runs that fail because provider usage or credit ran out are now treated as invalid and retried on resume

### Fixed

- Fixed baseline runs finding a skill that the target package ships inside itself. The target venv is now cleaned of agent guidance before any agent runs
- Fixed Claude runs being recorded with the wrong result when a background command kept the session alive
- Increased the Bash timeout for sandboxed commands to 600 seconds, so long downloads are no longer sent to the background
- Fixed capped or crashed Codex runs recording a cost of `$0.00`
- Fixed capped runs being failed even when they had already written a correct answer
- Fixed sandboxed agents being blocked from hosts that targets need, such as EBI, Zenodo and NCBI
- Fixed runs whose sandbox could not execute anything being graded as wrong answers instead of infrastructure failures
- `acumen tasks` now overwrites the untouched `tasks.yaml` placeholder from `acumen init` without `--force`
- Fixed Codex transcripts being rendered as empty pages
- Removed "Claude" from the drafting and improving prompts, which are also used by Codex
- Removed `check_auth` and `auth_available`, which ignored Codex credentials
