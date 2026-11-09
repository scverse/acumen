<p align="center">
  <img src="docs/_static/images/banner.svg" alt="acumen" width="360">
</p>

<!-- docs-start -->

<p align="center"><b>Build, benchmark, and optimize agentic skills for Python packages.</b></p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[![Issues][badge-issues]][issue tracker]
[![Coverage][badge-coverage]][codecoverage]
[![Stars][badge-stars]](https://github.com/scverse/acumen/stargazers)

[![PyPI][badge-pypi]][pypi]
[![Downloads month][badge-mdown]][down]
[![Downloads all][badge-adown]][down]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/acumen/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/acumen-skills
[badge-issues]: https://img.shields.io/github/issues/scverse/acumen
[badge-coverage]: https://codecov.io/gh/scverse/acumen/branch/main/graph/badge.svg
[badge-pypi]: https://img.shields.io/pypi/v/acumen.svg
[badge-mdown]: https://static.pepy.tech/badge/acumen/month
[badge-adown]: https://static.pepy.tech/badge/acumen
[badge-stars]: https://img.shields.io/github/stars/scverse/acumen?style=flat&logo=github&color=yellow

`acumen` is a python package to build, benchmark and optimize agentic skills for any Python package.
An agentic skill is a short set of plain-text instructions that helps a coding agent use a package correctly.
Given a package and a few tasks, `acumen` trains a skill in the same way one would train a model.
In every epoch, coding agents attempt the training tasks, and `acumen` learns from the errors they make to rewrite the skill.
Each new version is scored on held-out validation tasks, training stops once validation no longer improves, and the best version is evaluated one last time on an untouched test set.
With the final skill, coding agents are more successful at solving tasks that require using the Python package than they are without the skill.

[//]: # "numfocus-fiscal-sponsor-attribution"

`acumen` is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

## Getting started

Please refer to the [documentation][],
in particular, the [Getting started][getting started] tutorial and the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

`acumen` runs its agents with Claude, Codex, or both, and you only need to install the backend you use.
There are several alternative options to install `acumen`:

1. Install the latest stable release from [PyPI][pypi] to run Claude models (requires an Anthropic API key or a `claude` login):

```bash
pip install acumen[claude]
```

2. Install the latest stable release from [PyPI][pypi] to run Codex models (requires the `codex` CLI on your `PATH` and a Codex login or API key):

```bash
pip install acumen
```

3. Install the latest stable release from [PyPI][pypi] to run both:

```bash
pip install acumen[all]
```

4. Install the latest development version:

```bash
pip install "acumen[all] @ git+https://github.com/scverse/acumen.git@main"
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
[scverse]: https://scverse.org/
[issue tracker]: https://github.com/scverse/acumen/issues
[tests]: https://github.com/scverse/acumen/actions/workflows/test.yaml
[documentation]: https://acumen-skills.readthedocs.io/
[getting started]: https://acumen-skills.readthedocs.io/page/getting_started.html
[changelog]: https://acumen-skills.readthedocs.io/page/changelog.html
[api documentation]: https://acumen-skills.readthedocs.io/page/api/index.html
[pypi]: https://pypi.org/project/acumen
[down]: https://pepy.tech/project/acumen
[codecoverage]: https://codecov.io/gh/scverse/acumen
