"""End-to-end Experiment Spec execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path
from typing import Any

from netflip import __version__
from netflip.benchmarks import (
    build_cifar10_dataloaders,
    evaluate_classification_metrics,
    load_cifar_resnet20_quantized_artifact,
)
from netflip.experiment_spec import ExperimentSpec, load_experiment_spec
from netflip.manifest import (
    OUTPUT_SCHEMA_VERSION,
    JSONScalar,
    build_run_manifest,
    write_run_manifest,
)
from netflip.model_adapter import ModelAdapter
from netflip.runtime_device import resolve_torch_device
from netflip.soft_error import (
    SOFT_ERROR_SCENARIO_TYPE,
    FaultBudget,
    run_uniform_random_soft_error_baseline,
)
from netflip.trace import write_perturbation_trace


@dataclass(frozen=True)
class ExperimentRunOutput:
    """Filesystem outputs and summary metadata for one Run."""

    output_dir: Path
    manifest_path: Path
    perturbation_trace_path: Path
    clean_baseline_metrics: Mapping[str, JSONScalar]
    flip_count: int
    stopped_because: str
    device: str


class UnsupportedScenarioError(ValueError):
    """Raised when ``netflip run`` cannot execute the configured scenario."""


def execute_experiment_run(spec_path: str | PathLike[str]) -> ExperimentRunOutput:
    """Load and execute one supported Experiment Spec."""
    resolved_spec_path = Path(spec_path)
    spec = load_experiment_spec(resolved_spec_path)
    if spec.scenario.type != SOFT_ERROR_SCENARIO_TYPE:
        msg = (
            f"unsupported scenario type {spec.scenario.type!r}; "
            "netflip run currently supports only 'soft_error'"
        )
        raise UnsupportedScenarioError(msg)

    device = resolve_torch_device(spec.device)
    artifact = _load_benchmark_artifact(spec)
    dataloaders = build_cifar10_dataloaders(
        root=spec.dataset.root,
        selection_split=spec.dataset.selection_split,
        evaluation_split=spec.dataset.evaluation_split,
        selection_sample_limit=spec.dataset.selection_sample_limit,
        evaluation_sample_limit=spec.dataset.evaluation_sample_limit,
    )
    clean_baseline_metrics = _classification_metrics(
        artifact.model,
        dataloaders.evaluation,
        device=device,
    )

    def metric_evaluator(adapter: ModelAdapter) -> Mapping[str, JSONScalar]:
        return _classification_metrics(
            artifact.model,
            dataloaders.evaluation,
            device=device,
        )

    fault_budget = FaultBudget(
        max_flip_count=spec.scenario.fault_budget.max_flip_count,
        max_bit_flip_ratio=spec.scenario.fault_budget.max_bit_flip_ratio,
    )
    run_result = run_uniform_random_soft_error_baseline(
        adapter=artifact.adapter,
        metric_evaluator=metric_evaluator,
        fault_budget=fault_budget,
        rng_seed=spec.scenario.rng_seed,
    )

    output_dir = Path(spec.output_dir)
    perturbation_trace_path = write_perturbation_trace(
        run_result.perturbation_trace,
        output_dir,
    )
    manifest_path = write_run_manifest(
        _build_manifest(
            spec=spec,
            spec_path=resolved_spec_path,
            artifact=artifact,
            device=device,
        ),
        output_dir,
    )

    return ExperimentRunOutput(
        output_dir=output_dir,
        manifest_path=manifest_path,
        perturbation_trace_path=perturbation_trace_path,
        clean_baseline_metrics=clean_baseline_metrics,
        flip_count=run_result.flip_count,
        stopped_because=run_result.stopped_because,
        device=device,
    )


def _load_benchmark_artifact(spec: ExperimentSpec) -> Any:
    if spec.model.quantization.scale_path is None:
        msg = "soft-error CIFAR-10 ResNet-20 runs require model.quantization.scale_path"
        raise ValueError(msg)
    return load_cifar_resnet20_quantized_artifact(
        checkpoint_path=spec.model.checkpoint.path,
        scale_path=spec.model.quantization.scale_path,
        num_classes=spec.model.num_classes,
    )


def _classification_metrics(
    model: Any,
    dataloader: Any,
    *,
    device: str,
) -> dict[str, JSONScalar]:
    return _json_scalar_mapping(
        evaluate_classification_metrics(model, dataloader, device=device),
        context="classification metrics",
    )


def _build_manifest(
    *,
    spec: ExperimentSpec,
    spec_path: Path,
    artifact: Any,
    device: str,
) -> Any:
    return build_run_manifest(
        run_id=spec.run_id,
        created_at=datetime.now(timezone.utc),
        git_commit=_current_git_commit(),
        experiment_spec_hash=_file_sha256(spec_path),
        model_artifact_id=spec.model.benchmark,
        model_checkpoint_path=str(artifact.checkpoint_path),
        model_checkpoint_checksum=_file_sha256(Path(artifact.checkpoint_path)),
        quantization_metadata=_quantization_metadata(artifact),
        selection_dataset_id=_dataset_id(
            spec.dataset.name,
            spec.dataset.selection_split,
            spec.dataset.selection_sample_limit,
        ),
        selection_dataset_checksum=_dataset_request_checksum(
            dataset_name=spec.dataset.name,
            root=spec.dataset.root,
            split=spec.dataset.selection_split,
            sample_limit=spec.dataset.selection_sample_limit,
        ),
        evaluation_dataset_id=_dataset_id(
            spec.dataset.name,
            spec.dataset.evaluation_split,
            spec.dataset.evaluation_sample_limit,
        ),
        evaluation_dataset_checksum=_dataset_request_checksum(
            dataset_name=spec.dataset.name,
            root=spec.dataset.root,
            split=spec.dataset.evaluation_split,
            sample_limit=spec.dataset.evaluation_sample_limit,
        ),
        rng_seeds={"python": spec.scenario.rng_seed},
        device=device,
        netflip_version=__version__,
        dependencies=_dependency_versions(),
        output_schema_version=OUTPUT_SCHEMA_VERSION,
    )


def _quantization_metadata(artifact: Any) -> dict[str, JSONScalar]:
    quantization = artifact.quantization
    metadata: dict[str, JSONScalar] = {
        "codec": quantization.codec,
        "scale_granularity": quantization.scale_granularity,
        "tensor_count": len(quantization.tensors),
    }
    scale_path = getattr(artifact, "scale_path", None)
    if scale_path is not None:
        resolved_scale_path = Path(scale_path)
        metadata["scale_path"] = str(resolved_scale_path)
        metadata["scale_checksum"] = _file_sha256(resolved_scale_path)
    return metadata


def _dataset_id(dataset_name: str, split: str, sample_limit: int | None) -> str:
    if sample_limit is None:
        return f"{dataset_name}-{split}"
    return f"{dataset_name}-{split}-{sample_limit}"


def _dataset_request_checksum(
    *,
    dataset_name: str,
    root: str,
    split: str,
    sample_limit: int | None,
) -> str:
    payload = {
        "dataset_name": dataset_name,
        "root": root,
        "sample_limit": sample_limit,
        "split": split,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return _sha256_bytes(encoded)


def _json_scalar_mapping(
    values: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, JSONScalar]:
    normalized: dict[str, JSONScalar] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            msg = f"{context} keys must be non-empty strings"
            raise TypeError(msg)
        if not _is_json_scalar(value):
            msg = f"{context} value for {key!r} must be a JSON scalar"
            raise TypeError(msg)
        normalized[key] = value
    return normalized


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def _dependency_versions() -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for package_name in ("beartype", "click", "pydantic", "torch", "torchvision"):
        try:
            dependencies[package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    return dependencies
