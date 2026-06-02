from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from netflip import ExperimentSpec, load_experiment_spec, parse_experiment_spec

EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "cifar10_resnet20"


def test_random_soft_error_example_spec_parses() -> None:
    spec = load_experiment_spec(EXAMPLES_DIR / "random_soft_error.yaml")

    assert isinstance(spec, ExperimentSpec)
    assert spec.model.benchmark == "cifar10-resnet20"
    assert spec.model.checkpoint.path == "checkpoints/cifar10/resnet20-int8.pt"
    assert spec.dataset.root == "data/cifar10"
    assert spec.scenario.type == "soft_error"
    assert spec.scenario.fault_budget.max_flip_count == 128


def test_bfa_pbs_example_spec_parses() -> None:
    spec = load_experiment_spec(EXAMPLES_DIR / "bfa_pbs.yaml")

    assert spec.scenario.type == "attack"
    assert spec.scenario.strategy_name == "bfa-pbs"
    assert spec.scenario.target_policy == "ground-truth"
    assert spec.scenario.max_flip_count == 20
    assert spec.scenario.selection_batch_size == 128


def test_spec_rejects_missing_configurable_checkpoint_path() -> None:
    with pytest.raises(ValidationError, match="checkpoint"):
        parse_experiment_spec(
            """
            schema_version: "2026.1"
            run_id: missing-checkpoint
            model:
              benchmark: cifar10-resnet20
              architecture: resnet20
              num_classes: 10
              quantization:
                codec: signed-int8-two-complement
            dataset:
              name: cifar10
              root: data/cifar10
            scenario:
              type: soft_error
              fault_budget:
                max_flip_count: 1
              rng_seed: 1
            output_dir: runs/missing-checkpoint
            """
        )


def test_fallback_yaml_parser_reports_unsupported_sequences() -> None:
    with pytest.raises(ValueError, match="does not support sequences"):
        parse_experiment_spec(
            """
            schema_version: "2026.1"
            run_id: unsupported-list
            tags:
              - cifar10
            """
        )
