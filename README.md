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

### Linux Server Setup

If `uv` is not installed on a Linux server, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If the server does not have `curl`, use `wget`:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Restart the shell, or make sure the uv install directory is on `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Then set up NetFlip from a fresh clone:

```bash
cd netflip
uv python install 3.14
uv sync --extra benchmark
```

For CUDA servers, request CUDA explicitly so the command fails early if the GPU
runtime is unavailable:

```bash
uv run netflip prepare-cifar10-resnet20 --download --device cuda
uv run netflip run examples/cifar10_resnet20/bfa_pbs.yaml
```

For a quick server smoke test before full training:

```bash
uv run netflip prepare-cifar10-resnet20 \
  --download \
  --epochs 0 \
  --train-sample-limit 128 \
  --evaluation-sample-limit 128 \
  --device cuda
```

Use `--device cpu` on CPU-only Linux servers. `--device auto` selects CUDA when
available, then MPS, then CPU.

## Run

Current CLI:

```bash
uv run netflip --version
```

Prepare CIFAR-10 ResNet-20 artifacts for the example BFA/PBS Run:

```bash
uv sync --extra benchmark
uv run netflip prepare-cifar10-resnet20 --download
uv run netflip run examples/cifar10_resnet20/bfa_pbs.yaml
```

The default artifact-preparation command uses the CIFAR-10 ResNet-20 training
configuration from the upstream BFA script. Local datasets, checkpoints and Run
outputs live under git-ignored directories such as `data/`, `checkpoints/` and
`runs/`.

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
