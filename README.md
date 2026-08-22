<p align="center">
  <img src="docs/_static/images/banner.svg" alt="acumen" width="360">
</p>

<p align="center"><b>Build, benchmark, and optimize agentic skills for your Python package.</b></p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/acumen/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/acumen/badge/

Agentic skills, tool instructions writen in plain text, allow agents to use tools more succesfuly and efficient.
However most python packages do not ship skills with them because developers have no easy way to build and benchmark skills for their tools.
Acumen closes this gap. Point it at a Python package and a few evaluation tasks, and it drafts a skill, benchmarks it and improves it across a train/test split so the gains are generalizable.


Many good tools are unusable by coding agents because their maintainers have no way to write
a skill for them — or, having written one, no way to tell whether it helps. acumen closes
that loop: point it at a Python package and a few tasks, and it drafts a skill, benchmarks it
against a no-skill baseline, and improves it across a train/test split so the gains are real
generalization, not memorized answers.

- **`acumen check`** — rerun the script behind each task's answer, then have an agent judge
  whether the prompt actually asks for what that script and answer produce. Both before a
  benchmark pass pays to find out.
- **`acumen draft`** — write `skills/v1` from the package's own source.
- **`acumen bench`** — score a skill against a no-skill baseline, in a scrubbed sandbox where
  the skill is the only difference between arms. Agent guidance the target ships itself is
  removed from the venv first, so the baseline really is skill-free.
- **`acumen improve`** — refine the skill from its train results, then benchmark again.
- **`acumen report`** — aggregate every run into one self-contained `report.html`: success
  rate per version, train vs. test. Bars are coloured by model, with a grey bar pooling all
  of them; pass `--palette claude-opus-5=#3b7ea1` (repeatable) to recolour any of them.

You decide when to stop. Every version is benchmarked on both splits, and only train results
reach the improver — so a widening train/test gap is a visible sign a skill is overfitting
rather than genuinely helping.

## Quickstart

```bash
# 1. Scaffold a starter config.yaml and tasks.yaml
acumen init

# 2. Fill in config.yaml (repo). Write tasks.yaml by hand, or generate it:
acumen tasks                     # mine the package for real analyses -> tasks.yaml + tasks/
acumen check                     # is the ground truth right, and does each prompt ask for it?

# 3. Then run the loop:
acumen bench --no-skill          # the baseline arm
acumen draft                     # generate skills/v1 from the package source, or write by hand
acumen bench --skill v1          # benchmark the skill against the baseline
acumen improve                   # generate skills/v2 from v1's train results, or write by hand
acumen bench --skill v2
acumen bench                     # or: every arm at once (baseline + each skills/vN)
acumen report                    # aggregate every run into report.html

# 4. Once a version proves out, ship it into the package itself:
acumen ship --skill v2           # add a <dist>-install-skills console script (PR, or local edit)
```

`acumen ship` packages the chosen skill version into the target: the package gains a
`<dist>-install-skills` command that installs the skill into the skills directory of whichever
agent the user names — `--agent {claude,codex,agents,claude-science}`, or an explicit `--dest` —
so the package's own users get the guidance with one command, wherever they run their agent. The
same bundle installs verbatim into every framework.

## Checking the ground truth

A task is only worth benchmarking if its recorded answer is actually correct. A wrong answer makes
every model fail that task: real money spent, and the failure reads in the report as the model's
fault rather than the task's. `acumen check` catches the two ways that happens.

**Does the answer still come out of running the code?** Each task keeps a **reproducer** at
`tasks/<id>-<split>.py`, a self-contained script that redoes the analysis in the target venv and
writes its answer to `answer.md` — the same contract a benchmark run has, graded the same way.
`acumen tasks` writes them as it generates the tasks; `acumen check` reruns them:

```bash
acumen check                                 # every task, both splits
acumen check --task bulk --split train       # one cell, while you fix it
acumen check --jobs 8 --timeout 600          # or: --keep to inspect what a script wrote
```

You get one row per task and split — reproduced, wrong answer, script error, timed out, or no
script at all — then the summary statistics: how much of the task set has a reproducer, how much
of it reproduces, and how many tasks reproduce on both splits. Before running anything it checks
that the package imports in the venv at all, since that one failure would otherwise be reported
once per task.

**Does the prompt actually ask for what the script and answer produce?** Reproducing an answer
proves the code and the answer agree. It says nothing about the prompt, and a prompt describing
something else fails every agent that reads it correctly. A real example:

> Find the 3 most deactivated PROGENy pathways in Megakaryocytes … Report only the pathway names
> sorted by score **(ascending)**.

The script sorted descending, the recorded answer was descending, the reproducer check said `ok` —
and every agent that honoured the prompt produced the reverse order and was graded wrong. So after
the scripts run, one agent reads every split's prompt, recorded answer and reproducer together and
adds a `review` column of `ok` or `mismatch`, with one line naming the contradiction and one naming
the fix. It never edits `tasks.yaml`: which of the three artifacts to repair is your call.

```
task     split  status  review    detail
scell    train  ok      ok        MAPK;Estrogen;TGFb
scell    test   ok      mismatch  Trail;JAK-STAT;Estrogen

1 split the review flagged
  scell/test    prompt says ascending; script and answer are descending
                fix: say descending in the prompt, or reverse the answer
```

The review is on by default and picks its model from `check_model`; it is the one phase that costs
money, so `acumen check --no-review` runs the reproducers alone and spends nothing — what you want
while iterating on a script. `check` takes the same `--auth`, `--stream` and `--log-dir` flags as
the other agentic commands, and `--max-turns`/`--max-usd` bound the reviewer.

Either phase failing exits non-zero, so `acumen check` works as a gate before a pass.

A task that needs no code to answer (a licence, a supported species, a documented default) sets
`needs_script: false`; its reproducer column reads `n/a` rather than counting as a gap, and its
prompt and answer are still reviewed.

The reproducers hold the answers to the held-out test split, so nothing must feed them to an agent
under test. They are safe where they are: `bench`, `draft`, and `improve` confine their agents to
explicit read roots that never include your project directory, and the reviewer reads a staged copy
with no path back to `tasks.yaml`.

`acumen tasks`, `acumen draft`, and `acumen improve` each accept `--feedback "…"` to steer the
agent with context it can't infer — which functionality to skip when generating tasks, what a
skill should emphasise or fix. The guidance is added to the prompt without overriding the
train/test isolation, and for `draft`/`improve` it is recorded in the version's `meta.json` and
shown in the report. (Don't paste held-out test answers into `improve` feedback — that would
defeat the split.)

Claude and Codex can run side by side. Put both model families in `models` to compare them
in one matrix; model IDs beginning with `claude` use Claude Code, while `gpt-*`, `o1`,
`o3`, `o4`, and `codex-*` use Codex:

```yaml
models:
  - claude-opus-5
  - claude-sonnet-5
  - claude-haiku-4-5-20251001
  - gpt-5.6-sol
  - gpt-5.6-terra
  - gpt-5.6-luna
```

This spans each provider's quality/cost range; it is not a claim that the tiers are
one-to-one equivalents.

Neither backend is required. Claude is an optional dependency and Codex is an external CLI,
so install only the one you run — `pip install acumen[claude]`, or plain `acumen` plus the
`codex` CLI on `PATH`. Selecting a model whose backend is missing fails immediately, with the
install command, before acumen prepares a target or spends anything.

Claude API runs use `ANTHROPIC_API_KEY`; Codex API runs use `CODEX_API_KEY` (or
`OPENAI_API_KEY`). The meta-agent commands also accept a Codex model through their
`*_model` config keys or `--model`.

Every agentic command — `bench` included — takes `--auth {auto,session,api}` and defaults to
the provider's logged-in subscription, falling back to its API key. Both billing modes report
tokens, so Acumen can calculate the same API-rate estimate for either. Under `session`, that
estimate is what the run *would* have cost at API rates, not money billed — so each run records
its `auth_mode` alongside the figure.

If the selected subscription runs out of usage or the API account runs out of credit, Acumen
invalidates the pass instead of scoring that as an agent failure: it prints the provider error,
cancels remaining cells for that provider, lets other providers finish all running and queued
cells, and exits non-zero. Replenish the credential and rerun the same command; automatic resume
retries the invalid and cancelled cells. Reports and `improve` refuse invalid quota/credit
evidence.

`max_turns` and `max_usd` apply to both providers, but they are not equally strict for Codex,
which has no cap of its own — acumen enforces both against its event stream:

- **`max_turns` bounds the run.** One `codex exec` is a single Codex turn however much work
  happens inside it, so turns are counted in completed model actions (a message, a command, a
  file change, a tool or search call) and the agent is stopped at the cap.
- **`max_usd` cannot.** Codex reports usage once, when the turn ends, so a breach is only
  visible after the money is spent. The run is recorded as a budget failure — the same outcome
  Claude gives it — but bound Codex spend with `max_turns`. acumen prints this before the pass.

**Every cost acumen shows is inferred from tokens.** Each run records its breakdown (fresh
input, cache reads, cache writes, and output) and Acumen prices it with the rate table stored
in `result.json`. That gives Claude and Codex one comparable basis and prevents an old
benchmark from being silently re-priced, so it is what `cost_usd` holds and what every figure,
table, CSV column and console line reports. Where a backend supplies a dollar figure of its own
it is recorded beside it as `provider_cost_usd` (`recorded_cost_usd` in the report's sidecar
CSV), with the gap between the two, but nothing is plotted or tallied from it: Claude's SDK
total covers nested subagents that the run's own usage block does not, so a console reading it
would disagree with the report it summarises. A model no layer prices stays unpriced even when
the provider reported dollars, since one run on a basis the rest of the pass is not on is worse
than a visible gap.

**Rates are read from the providers' pricing pages, never shipped with the package.** Prices
move, and each run's cost is frozen into its `result.json` and never recomputed, so a table
compiled into a release would store numbers that were already wrong. `bench` resolves rates
before it spends anything and **fails the pass** if the pages cannot be read: cost is a headline
metric, and a benchmark that cannot establish rates has not earned the numbers it would print.
`draft`, `improve`, `tasks`, `check`, and `ship` fetch too but degrade to unpriced instead — their
cost line is progress reporting, not stored evidence.

Alongside the rates themselves each run records `price_source` (`config` or `fetched`) and
`price_rates_as_of`, so a pass run in August and another in October stay individually
attributable and one report can cover both without restating either. When arms in a report were
priced on different dates, the report says so: the cost gap between them includes the price
change, not only the skill's effect.

```bash
acumen prices              # the rates in use today, and where each came from
acumen prices --refresh    # check pinned rates against what the providers publish
```

Pin rates with a `prices:` block in `config.yaml` to price a model the providers don't publish,
to price a gateway, or to record negotiated rates — pins outrank a live fetch, since only you
know what you are billed. They are also the only rates that can drift unnoticed, which is what
`--refresh` checks; it prints a diff for you to accept and never rewrites anything, because
picking the wrong tier or context band would silently misprice future runs. A model no layer
prices records its tokens and leaves report cost unavailable — never zero, which would read as
free.

> One consequence worth knowing: Codex's `max_usd` cap is enforced from these same rates, so an
> unpriced model under Codex has no enforceable budget cap. Bound those runs with `max_turns`,
> or pin the rates.

`draft`, `improve`, `tasks`, `ship`, and `check`'s review phase each drive an autonomous agent.
Every run writes a live `logs/acumen-<command>-<datetime>.jsonl` (one event per step, flushed as it
goes — so you can watch progress by reading the file) and a rendered `.html` transcript. Add
`--stream` to mirror the conversation to the terminal, or `--log-dir` to change where the logs
land.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

Install the backend you actually run — both are optional, and either alone is a complete
install:

| you run | install | also needs |
|---|---|---|
| Claude only | `pip install acumen[claude]` | an Anthropic key or a `claude` login |
| Codex only | `pip install acumen` | the `codex` CLI on `PATH`, plus a Codex login or key |
| both | `pip install acumen[all]` | both of the above |

<!--
1) Install the latest release of `acumen` from [PyPI][]:

```bash
pip install acumen
```
-->

And to install the acumen skill that ships with the package into your agent's skills directory,
run `acumen-install-skills --agent {claude,codex,agents,claude-science}` (or `--dest <dir>` to
choose the directory yourself):

```bash
acumen-install-skills --agent claude
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/acumen/issues
[tests]: https://github.com/scverse/acumen/actions/workflows/test.yaml
[documentation]: https://acumen.readthedocs.io
[changelog]: https://acumen.readthedocs.io/page/changelog.html
[api documentation]: https://acumen.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/acumen
