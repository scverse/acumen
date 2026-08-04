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

- **`acumen draft`** — write `skills/v1` from the package's own source.
- **`acumen bench`** — score a skill against a no-skill baseline, in a scrubbed sandbox where
  the skill is the only difference between arms.
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
acumen tasks                     # mine the package for real analyses -> tasks.yaml

# 3. Then run the loop:
acumen bench --no-skill          # the baseline arm
acumen draft                     # generate skills/v1 from the package source, or write by hand
acumen bench --skill v1          # benchmark the skill against the baseline
acumen improve                   # generate skills/v2 from v1's train results, or write by hand
acumen bench --skill v2
acumen report                    # aggregate every run into report.html

# 4. Once a version proves out, ship it into the package itself:
acumen ship --skill v2           # add a <dist>-install-skills console script (PR, or local edit)
```

`acumen ship` packages the chosen skill version into the target: the package gains a
`<dist>-install-skills` command that installs the skill into the skills directory of whichever
agent the user names — `--agent {claude,codex,agents,claude-science}`, or an explicit `--dest` —
so the package's own users get the guidance with one command, wherever they run their agent. The
same bundle installs verbatim into every framework.

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

`max_turns` and `max_usd` apply to both providers, but they are not equally strict for Codex,
which has no cap of its own — acumen enforces both against its event stream:

- **`max_turns` bounds the run.** One `codex exec` is a single Codex turn however much work
  happens inside it, so turns are counted in completed model actions (a message, a command, a
  file change, a tool or search call) and the agent is stopped at the cap.
- **`max_usd` cannot.** Codex reports usage once, when the turn ends, so a breach is only
  visible after the money is spent. The run is recorded as a budget failure — the same outcome
  Claude gives it — but bound Codex spend with `max_turns`. acumen prints this before the pass.

**Reports compare inferred cost per run.** Every run records its token breakdown — fresh
input, cache reads, cache writes, and output — and Acumen prices it with the frozen rate table
stored in `result.json`. That gives Claude and Codex one comparable basis and prevents an old
benchmark from being silently re-priced. The result itself retains both
`provider_cost_usd` (when the backend supplies one) and `inferred_cost_usd`; its compatibility
field `cost_usd` remains provider-first. The report's sidecar CSV calls the former
`recorded_cost_usd` and keeps it separate from `inferred_cost_usd`, while every displayed cost
and comparison uses the inferred value.

```bash
acumen prices              # the rates in use, and where each came from
acumen prices --refresh    # re-check them against the providers' pricing pages
```

`--refresh` fetches both providers' published tables and prints a diff for you to accept —
it never rewrites anything, because picking the wrong tier or context band would silently
misprice every future run. Adopt changes by pasting the emitted block into `config.yaml`
under `prices:`, which is also how you price a model acumen doesn't ship a rate for, or
override rates for a gateway. A model with no rate records its tokens and leaves inferred
report cost unavailable — never zero, which would read as free.

`draft`, `improve`, `tasks`, and `ship` each drive a long autonomous agent. Every run writes a
live `logs/acumen-<command>-<datetime>.jsonl` (one event per step, flushed as it goes — so you
can watch progress by reading the file) and a rendered `.html` transcript. Add `--stream` to
mirror the conversation to the terminal, or `--log-dir` to change where the logs land.

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
