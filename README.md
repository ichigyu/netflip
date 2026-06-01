# NetFlip

NetFlip is a neural-network bit-flip reliability evaluation framework for comparing random soft errors, adversarial bit-flip attacks, and hardening approaches under a shared experiment format.

## MVP Scope

The first usable release focuses on CIFAR-10 classification with a ResNet-20 benchmark model, BFA-compatible int8 quantization, uniform random soft-error injection, BFA/PBS-style attack injection, per-bit perturbation traces, and reproducible run manifests.

## Local Setup

Recommended local workflow:

```bash
pyenv install 3.11
pyenv local 3.11
poetry config virtualenvs.in-project true --local
poetry install --with dev
```

NetFlip supports Python 3.10 and newer. The local development workflow pins
Python 3.11 as the recommended version for contributors.

## Run

Current CLI:

```bash
poetry run netflip --version
```

The experiment runner is planned for a later MVP work item.

## Test And Lint

Default checks:

```bash
poetry run pytest
poetry run ruff check .
```

Coverage report:

```bash
poetry run coverage run -m pytest
poetry run coverage report
```

Run the automated test workflow:

```bash
poetry run nox
```

The Nox workflow runs pytest, coverage, and Ruff linting. It is configured to
skip unavailable Python interpreters so contributors can still run the default
checks on the local Python 3.11 development environment.

`pytest-mock` is not part of the development dependencies yet. The current test
suite uses pytest fixtures and Click's `CliRunner` directly, and there is no
shared mock-heavy test setup that needs the plugin. Prefer adding `pytest-mock`
only when a concrete mocking use case appears.
