"""Experiment Spec validation and lightweight YAML loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field

from netflip.benchmarks import CIFAR10_CLASSES, CIFAR_RESNET20_BENCHMARK_ID

EXPERIMENT_SPEC_SCHEMA_VERSION = "2026.1"


class QuantizationSpec(BaseModel):
    """BFA-compatible int8 quantization settings for a benchmark model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    codec: Literal["signed-int8-two-complement"]
    scale_granularity: Literal["per-tensor", "per-channel"] = "per-tensor"
    scale_path: str | None = None


class CheckpointSpec(BaseModel):
    """Configurable model checkpoint artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    format: str = Field(default="pytorch-state-dict", min_length=1)


class BenchmarkModelSpec(BaseModel):
    """Model configuration for an Experiment Spec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: Literal["cifar10-resnet20"] = CIFAR_RESNET20_BENCHMARK_ID
    architecture: Literal["resnet20"] = "resnet20"
    num_classes: int = Field(default=CIFAR10_CLASSES, gt=0)
    checkpoint: CheckpointSpec
    quantization: QuantizationSpec


class DatasetSpec(BaseModel):
    """Configurable Evaluation Dataset paths for CIFAR-10 benchmark runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["cifar10"] = "cifar10"
    root: str = Field(min_length=1)
    selection_split: str = Field(default="train", min_length=1)
    evaluation_split: str = Field(default="test", min_length=1)


class FaultBudgetSpec(BaseModel):
    """Fault Budget limits from an Experiment Spec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_flip_count: int | None = Field(default=None, ge=0)
    max_bit_flip_ratio: float | None = Field(default=None, ge=0, le=1)


class RandomSoftErrorScenarioSpec(BaseModel):
    """Uniform random soft-error scenario configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["soft_error"]
    fault_model: Literal["uniform-eligible-bit"] = "uniform-eligible-bit"
    fault_schedule: Literal["one-bit-step"] = "one-bit-step"
    fault_budget: FaultBudgetSpec
    rng_seed: int


class BfaPbsScenarioSpec(BaseModel):
    """BFA/PBS Attack Scenario configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["attack"]
    strategy_name: Literal["bfa-pbs"] = "bfa-pbs"
    attack_objective: str = Field(min_length=1)
    target_policy: str = Field(min_length=1)
    max_flip_count: int = Field(gt=0)
    selection_batch_size: int = Field(gt=0)
    rng_seed: int


class ExperimentSpec(BaseModel):
    """Auditable YAML Experiment Spec for a NetFlip Run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2026.1"] = EXPERIMENT_SPEC_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    model: BenchmarkModelSpec
    dataset: DatasetSpec
    scenario: RandomSoftErrorScenarioSpec | BfaPbsScenarioSpec
    output_dir: str = Field(min_length=1)


@beartype
def load_experiment_spec(path: str | PathLike[str]) -> ExperimentSpec:
    """Load and validate a YAML Experiment Spec."""
    spec_path = Path(path)
    return parse_experiment_spec(spec_path.read_text(encoding="utf-8"))


@beartype
def parse_experiment_spec(text: str) -> ExperimentSpec:
    """Parse and validate a YAML Experiment Spec string."""
    raw_spec = _load_yaml_mapping(text)
    return ExperimentSpec.model_validate(raw_spec)


def _load_yaml_mapping(text: str) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        loaded = _load_simple_yaml_mapping(text)
    else:  # pragma: no cover - exercised only when PyYAML is installed
        loaded = yaml.safe_load(text)

    if not isinstance(loaded, Mapping):
        msg = "experiment spec must be a YAML mapping"
        raise ValueError(msg)
    return loaded


def _load_simple_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = _parse_indented_mapping(text)
    if not isinstance(loaded, dict):
        msg = "experiment spec must be a YAML mapping"
        raise ValueError(msg)
    return loaded


def _parse_indented_mapping(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            msg = (
                f"tabs are not supported in fallback YAML parser at line {line_number}"
            )
            raise ValueError(msg)

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        key, separator, raw_value = stripped.partition(":")
        if not separator or not key.strip():
            msg = f"expected 'key: value' YAML mapping entry at line {line_number}"
            raise ValueError(msg)

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            msg = f"invalid indentation at line {line_number}"
            raise ValueError(msg)

        current = stack[-1][1]
        normalized_key = key.strip()
        value_text = raw_value.strip()
        if value_text:
            current[normalized_key] = _parse_scalar(value_text)
            continue

        nested: dict[str, Any] = {}
        current[normalized_key] = nested
        stack.append((indent, nested))

    return root


def _parse_scalar(value_text: str) -> Any:
    if value_text in {"null", "Null", "NULL", "~"}:
        return None
    if value_text in {"true", "True", "TRUE"}:
        return True
    if value_text in {"false", "False", "FALSE"}:
        return False
    if (value_text.startswith('"') and value_text.endswith('"')) or (
        value_text.startswith("'") and value_text.endswith("'")
    ):
        return value_text[1:-1]
    try:
        return int(value_text)
    except ValueError:
        pass
    try:
        return float(value_text)
    except ValueError:
        return value_text
