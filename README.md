# NetFlip

NetFlip is a neural-network bit-flip reliability evaluation framework for comparing random soft errors, adversarial bit-flip attacks, and hardening approaches under a shared experiment format.

## MVP Scope

The first usable release focuses on CIFAR-10 classification with a ResNet-20 benchmark model, BFA-compatible int8 quantization, uniform random soft-error injection, BFA/PBS-style attack injection, per-bit perturbation traces, and reproducible run manifests.

## Local Setup

Recommended local workflow:

```bash
conda create -n netflip python=3.11
conda activate netflip
pip install -e ".[dev]"
```

## Run

Current CLI:

```bash
netflip --version
```

The experiment runner is planned for a later MVP work item.

## Test And Lint

Default checks:

```bash
pytest
ruff check .
```
