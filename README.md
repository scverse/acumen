<p align="center">
  <img src="docs/_static/images/banner_plate.svg" alt="acumen" width="340">
</p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/acumen/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/acumen/badge/

**Build, benchmark, and optimize Claude skills for your Python package.**

Many good tools are unusable by coding agents today because their maintainers have no way
to write a skill for them — or, having written one, no way to tell whether it actually
helps. acumen closes that loop. Point it at a Python package and a handful of tasks and it
will:

1. **Draft** a skill from the package's own source.
2. **Benchmark** that skill against a no-skill baseline on your tasks, in a scrubbed sandbox
   where the only difference between arms is the skill itself.
3. **Improve** the skill from its results and benchmark again — with a train/test split, so
   the gains are real generalization and not memorized answers.
4. **Report** success rate per version, train vs. test, in one self-contained `report.html`.

```
noskill baseline ──▶ draft v1 ──▶ bench v1 ──▶ report
                                     │
                                     ▼
                        improve (train results only) ──▶ v2 ──▶ bench v2 ──▶ report ──▶ …
```

You decide when to stop. Every version is benchmarked on both splits; only train results
are ever shown to the improver, so a widening train/test gap is a visible signal that a
skill is overfitting rather than genuinely helping.

## Quickstart

```bash
# 1. Scaffold a starter config.yaml and tasks.yaml
acumen init

# 2. Fill in config.yaml (repo, skill_name). Write tasks.yaml by hand, or generate it:
acumen tasks                     # mine the package for real analyses -> tasks.yaml

# 3. Then run the loop:
acumen bench --no-skill          # the baseline arm
acumen draft                     # write skills/v1 from the package source
acumen bench --skill v1          # benchmark the skill against the baseline
acumen improve                   # write skills/v2 from v1's train results
acumen bench --skill v2
acumen report                    # aggregate every run into report.html

# 4. Once a version proves out, ship it into the package itself:
acumen ship --skill v2           # add a <dist>-install-skills console script (PR, or local edit)
```

`acumen ship` packages the chosen skill version into the target: the package gains a
`<dist>-install-skills` command that installs the skill into `~/.claude/skills/`, so the
package's own users get the guidance with one command. For a GitHub target the change arrives
as a pull request; for a local path it is written straight into the working tree.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install acumen:

<!--
1) Install the latest release of `acumen` from [PyPI][]:

```bash
pip install acumen
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/scverse/acumen.git@main
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
