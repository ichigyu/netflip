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
