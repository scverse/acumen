<p align="center">
  <img src="docs/_static/images/banner.svg" alt="acumen" width="360">
</p>

<p align="center"><b>Build, benchmark, and optimize agentic skills for your Python package.</b></p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/acumen/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/acumen/badge/

An agentic skill is a short, plain-text set of instructions that helps a coding agent use a tool
correctly. Most Python packages ship none — maintainers have no easy way to write one, or to tell
whether it actually helps. acumen closes that loop: point it at a package and a few tasks, and it
learns a skill in **training epochs**, each one benchmarking the current skill, distilling what went
right and wrong into a knowledge wiki, and rewriting the skill — always measured against a no-skill
baseline on a held-out split, so the gains generalize instead of memorizing answers.

## Installation

Python 3.12+ is required. Install the backend you actually run — both are optional, and either
alone is a complete install:

| you run | install | also needs |
|---|---|---|
| Claude only | `pip install acumen[claude]` | an Anthropic key or a `claude` login |
| Codex only | `pip install acumen` | the `codex` CLI on `PATH`, plus a Codex login or key |
| both | `pip install acumen[all]` | both of the above |

To install the acumen skill (guidance for using acumen itself) into your agent:

```bash
acumen-install-skills --agent claude   # or codex, agents, claude-science; or --dest <dir>
```

## Quickstart

```bash
acumen init                 # scaffold config.yaml + tasks.yaml
# edit config.yaml: point `repo` at your package (a GitHub URL or local path)

acumen tasks                # mine the package for real tasks + ground-truth reproducers
acumen check                # verify each task's answer reproduces before you spend

acumen epoch                # one training round: bench train → wiki → create/improve skill → bench held-out
acumen epoch                # run again for the next round (v2, v3, …)

acumen report               # aggregate every run into a self-contained report.html
acumen ship --skill v2      # ship the chosen skill into your package
```

`acumen epoch` is the loop. Each round benchmarks the current arm on the training tasks, records
what happened per task in `wiki/`, creates or improves the skill from that wiki, and benchmarks the
new version on the held-out validation tasks. Re-run it to keep going; you decide when to stop.

The stages also run on their own — `acumen bench`, `acumen wiki`, and `acumen improve`.

`acumen ship` gives your package a `<dist>-install-skills` command so its users can install the
skill into whichever agent they use (`--agent {claude,codex,agents,claude-science}`, or `--dest`).

## Getting started

Please refer to the [documentation][], in particular the [API documentation][].

## Release notes

See the [changelog][].

## Contact

For questions and help requests, reach out on the [scverse discourse][].
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
