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
whether it actually helps. acumen closes that loop: point it at a package and a few tasks, and
`acumen fit` optimizes a skill the way you would train a model. Each **epoch** benchmarks the
current skill, distils what went right and wrong into a knowledge wiki, rewrites the skill, and
scores the new version on a validation split. Training stops once validation stops improving, and
the best version gets one final score on an untouched test split. Every score sits next to a
no-skill baseline, so the gains carry over to new inputs instead of memorizing answers.

## Installation

Python 3.12+ is required. Install the backend you actually run — both are optional, and either
alone is a complete install:

| you run | install | also needs |
|---|---|---|
| Claude only | `pip install acumen[claude]` | an Anthropic key or a `claude` login |
| Codex only | `pip install acumen` | the `codex` CLI on `PATH`, plus a Codex login or key |
| both | `pip install acumen[all]` | both of the above |

## Quickstart

```bash
acumen init                 # scaffold config.yaml + tasks.yaml
# edit config.yaml: point `repo` at your package (a GitHub URL or local path)

acumen tasks                # mine the package for real tasks + ground-truth reproducers
acumen check                # verify each task's answer reproduces before you spend

acumen fit                  # optimize the skill epoch by epoch until validation plateaus, then test
                            # the best version; writes training.csv and report.html

acumen ship --skill v2      # ship the best version into your package
```

## The optimization loop

Every task in `tasks.yaml` comes in three splits:

| split | used for |
|---|---|
| `train` | the only results the improver sees; the skill is learned from these |
| `valid` | scores each new version; picks the best one and drives early stopping |
| `test` | benched once, at the end, for the best version only |

`acumen fit` repeats one **epoch** at a time. An epoch has four steps:

1. **bench** the current skill on `train` (the first epoch benches the no-skill baseline instead,
   and also on `valid`, so there is a baseline to compare against);
2. **wiki**: record what happened per task in `wiki/`;
3. **improve**: create or rewrite the skill from that wiki, giving `skills/v1`, `v2`, …;
4. **validate**: bench the new version on `valid`.

After each epoch, fit prints a progress line and updates the training curve in `training.csv`. It
stops when any of these happens:

- `--patience` epochs (default 2) pass without validation mean success rising by more than
  `--min-delta` (default 0.05, i.e. 5 percentage points) over the last improvement;
- `--max-epochs` is reached (default 10);
- validation reaches 100%.

Pass `--epochs N` to run exactly N epochs with no early stopping. Afterwards fit benches the best
version on `test` and writes `report.html`. Run `acumen report` to rebuild the report later.

Fit can be resumed. Re-run `acumen fit` and it continues from the last finished epoch.
`--epochs`/`--max-epochs` set the total number of epochs, not how many more to run.

Each stage also runs on its own: `acumen bench`, `acumen wiki`, and `acumen improve`.

To measure what the wiki adds, `acumen fit --ablate-wiki` writes `v1` from the package source
alone. It uses the same create prompt with no wiki or transcripts, and runs no train benches. It
then benches that `v1` and the no-skill baseline on `valid` and stops. The test split is left
untouched. One project root holds one ablation run. Compare its `v1` with the first epoch of a
normal fit run in another root.

## Shipping

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
