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
    assert spec.device == "auto"
    assert spec.dataset.root == "data/cifar10"
    assert spec.scenario.type == "soft_error"
    assert spec.scenario.fault_budget.max_flip_count is None
    assert spec.scenario.fault_budget.max_bit_flip_ratio == 0.001


def test_bfa_pbs_example_spec_parses() -> None:
    spec = load_experiment_spec(EXAMPLES_DIR / "bfa_pbs.yaml")

    assert spec.scenario.type == "attack"
    assert spec.scenario.strategy_name == "bfa-pbs"
    assert spec.scenario.target_policy == "ground-truth"
    assert spec.scenario.max_flip_count == 20
    assert spec.scenario.selection_batch_size == 128


def test_spec_accepts_dataset_sample_limits() -> None:
    spec = parse_experiment_spec(
        """
        schema_version: "2026.1"
        run_id: small-validation
        model:
          benchmark: cifar10-resnet20
          architecture: resnet20
          num_classes: 10
          checkpoint:
            path: checkpoints/cifar10/resnet20-int8.pt
            format: pytorch-state-dict
          quantization:
            codec: signed-int8-two-complement
        dataset:
          name: cifar10
          root: data/cifar10
          selection_split: train
          evaluation_split: test
          selection_sample_limit: 64
          evaluation_sample_limit: 128
        scenario:
          type: soft_error
          fault_budget:
            max_flip_count: 1
          rng_seed: 1
        output_dir: runs/small-validation
        """
    )

    assert spec.dataset.selection_sample_limit == 64
    assert spec.dataset.evaluation_sample_limit == 128


def test_spec_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        parse_experiment_spec(
            """
            schema_version: "2025.1"
            run_id: invalid-schema-version
            model:
              benchmark: cifar10-resnet20
              architecture: resnet20
              num_classes: 10
              checkpoint:
                path: checkpoints/cifar10/resnet20-int8.pt
                format: pytorch-state-dict
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
            output_dir: runs/invalid-schema-version
            """
        )


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


def test_spec_accepts_explicit_runtime_device_request() -> None:
    spec = parse_experiment_spec(
        """
        schema_version: "2026.1"
        run_id: explicit-device
        device: mps
        model:
          benchmark: cifar10-resnet20
          architecture: resnet20
          num_classes: 10
          checkpoint:
            path: checkpoints/cifar10/resnet20-int8.pt
            format: pytorch-state-dict
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
        output_dir: runs/explicit-device
        """
    )

    assert spec.device == "mps"


def test_spec_rejects_unknown_runtime_device_request() -> None:
    with pytest.raises(ValidationError, match="device"):
        parse_experiment_spec(
            """
            schema_version: "2026.1"
            run_id: invalid-device
            device: tpu
            model:
              benchmark: cifar10-resnet20
              architecture: resnet20
              num_classes: 10
              checkpoint:
                path: checkpoints/cifar10/resnet20-int8.pt
                format: pytorch-state-dict
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
            output_dir: runs/invalid-device
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
