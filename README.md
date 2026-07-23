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
  rate per version, train vs. test.

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
`<dist>-install-skills` command that installs the skill into `~/.claude/skills/`, so the
package's own users get the guidance with one command.

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

<!--
1) Install the latest release of `acumen` from [PyPI][]:

```bash
pip install acumen
```
-->

And to install the acumen skills needed to run acumen:

```bash
pip install acumen
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
