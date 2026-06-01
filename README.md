# NetFlip

NetFlip is a neural-network bit-flip reliability evaluation framework for comparing random soft errors, adversarial bit-flip attacks, and hardening approaches under a shared experiment format.

## MVP Scope

The first usable release focuses on CIFAR-10 classification with a ResNet-20 benchmark model, BFA-compatible int8 quantization, uniform random soft-error injection, BFA/PBS-style attack injection, per-bit perturbation traces, and reproducible run manifests.

## Local Setup

Recommended local workflow:

```bash
uv python install 3.14
uv sync --all-groups
```

NetFlip supports Python 3.10 and newer, with CI covering Python 3.10 through
3.14. The local workflow is uv-first and uses the checked-in `uv.lock` file for
reproducible installs.

## Run

Current CLI:

```bash
uv run netflip --version
```

The experiment runner is planned for a later MVP work item.

## Test And Lint

Default checks:

```bash
uv run pytest
uv run pytest --xdoctest src/netflip
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Coverage report:

```bash
uv run coverage run -m pytest
uv run coverage xml
uv run coverage report
```

Build documentation and package artifacts:

```bash
uv run sphinx-build -W -b html docs docs/_build/html
uv run python -m build
```

Run the complete automated workflow:

```bash
uv run nox
```

The Nox workflow runs pytest, xdoctest, coverage, Ruff lint and format checks,
Pyright, Sphinx documentation builds, and package builds. It is configured to
skip unavailable Python interpreters so contributors can still run the default
checks on their local development environment.

`pytest-mock` is not part of the development dependencies yet. The current test
suite uses pytest fixtures and Click's `CliRunner` directly, and there is no
shared mock-heavy test setup that needs the plugin. Prefer adding `pytest-mock`
only when a concrete mocking use case appears.
