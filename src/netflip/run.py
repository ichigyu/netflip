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
    CIFAR_RESNET20_BENCHMARK_ID,
    build_cifar10_dataloaders,
    evaluate_classification_metrics,
    load_cifar_resnet20_quantized_artifact,
)
from netflip.bfa_pbs import (
    ATTACK_SCENARIO_TYPE,
    run_bfa_pbs_attack_strategy,
    validate_bfa_pbs_scenario_config,
)
from netflip.experiment_spec import ExperimentSpec, load_experiment_spec
from netflip.manifest import (
    OUTPUT_SCHEMA_VERSION,
    JSONScalar,
    build_run_manifest,
    write_run_manifest,
)
from netflip.model_adapter import ModelAdapter
from netflip.runtime_device import RuntimeDeviceUnavailableError, resolve_torch_device
from netflip.soft_error import (
    SOFT_ERROR_SCENARIO_TYPE,
    FaultBudget,
    run_uniform_random_soft_error_baseline,
)
from netflip.summary import build_run_summary, write_run_summary
from netflip.trace import write_candidate_trace, write_perturbation_trace


@dataclass(frozen=True)
class ExperimentRunOutput:
    """Filesystem outputs and summary metadata for one Run."""

    output_dir: Path
    manifest_path: Path
    perturbation_trace_path: Path
    candidate_trace_path: Path | None
    summary_json_path: Path
    summary_csv_path: Path
    clean_baseline_metrics: Mapping[str, JSONScalar]
    flip_count: int
    stopped_because: str
    device: str


class ExperimentRunError(ValueError):
    """Raised for expected, user-facing run execution errors."""


class UnsupportedScenarioError(ExperimentRunError):
    """Raised when ``netflip run`` cannot execute the configured scenario."""


class UnsupportedBenchmarkError(ExperimentRunError):
    """Raised when ``netflip run`` cannot execute the configured benchmark."""


def execute_experiment_run(spec_path: str | PathLike[str]) -> ExperimentRunOutput:
    """Load and execute one supported Experiment Spec."""
    resolved_spec_path = Path(spec_path).resolve()
    try:
        spec = load_experiment_spec(resolved_spec_path)
    except (OSError, ValueError) as exc:
        msg = f"failed to load Experiment Spec {resolved_spec_path}: {exc}"
        raise ExperimentRunError(msg) from exc

    if spec.scenario.type not in {SOFT_ERROR_SCENARIO_TYPE, ATTACK_SCENARIO_TYPE}:
        msg = f"unsupported scenario type {spec.scenario.type!r}"
        raise UnsupportedScenarioError(msg)
    if spec.scenario.type == ATTACK_SCENARIO_TYPE:
        try:
            validate_bfa_pbs_scenario_config(
                attack_objective=spec.scenario.attack_objective,
                target_policy=spec.scenario.target_policy,
            )
        except ValueError as exc:
            raise ExperimentRunError(str(exc)) from exc

    try:
        device = str(resolve_torch_device(spec.device))
    except (RuntimeDeviceUnavailableError, ValueError) as exc:
        raise ExperimentRunError(str(exc)) from exc

    artifact = _load_benchmark_artifact(spec)
    try:
        dataloaders = build_cifar10_dataloaders(
            root=spec.dataset.root,
            selection_split=spec.dataset.selection_split,
            evaluation_split=spec.dataset.evaluation_split,
            batch_size=_dataloader_batch_size(spec),
            selection_sample_limit=spec.dataset.selection_sample_limit,
            evaluation_sample_limit=spec.dataset.evaluation_sample_limit,
        )
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        raise ExperimentRunError(str(exc)) from exc

    try:
        clean_baseline_metrics = _classification_metrics(
            artifact.model,
            dataloaders.evaluation,
            device=device,
        )
    except TypeError as exc:
        raise ExperimentRunError(str(exc)) from exc

    def metric_evaluator(adapter: ModelAdapter) -> Mapping[str, JSONScalar]:
        return _classification_metrics(
            _adapter_model(adapter, fallback=artifact.model),
            dataloaders.evaluation,
            device=device,
        )

    try:
        if spec.scenario.type == SOFT_ERROR_SCENARIO_TYPE:
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
        else:
            selection_batch = _first_selection_batch(dataloaders.selection)

            def objective_evaluator(adapter: ModelAdapter) -> float:
                return _classification_cross_entropy(
                    _adapter_model(adapter, fallback=artifact.model),
                    selection_batch,
                    device=device,
                )

            run_result = run_bfa_pbs_attack_strategy(
                adapter=artifact.adapter,
                objective_evaluator=objective_evaluator,
                metric_evaluator=metric_evaluator,
                attack_objective=spec.scenario.attack_objective,
                target_policy=spec.scenario.target_policy,
                max_flip_count=spec.scenario.max_flip_count,
                rng_seed=spec.scenario.rng_seed,
                record_candidate_trace=spec.scenario.emit_candidate_trace,
            )
    except (TypeError, ValueError) as exc:
        raise ExperimentRunError(str(exc)) from exc

    output_dir = Path(spec.output_dir)
    try:
        perturbation_trace_path = write_perturbation_trace(
            run_result.perturbation_trace,
            output_dir,
        )
        candidate_trace_output_path = None
        candidate_trace = getattr(run_result, "candidate_trace", ())
        if candidate_trace:
            candidate_trace_output_path = write_candidate_trace(
                candidate_trace,
                output_dir,
            )
        summary = build_run_summary(
            clean_metrics=clean_baseline_metrics,
            perturbation_trace=run_result.perturbation_trace,
            stopped_because=run_result.stopped_because,
            eligible_bit_population=run_result.eligible_bit_population,
        )
        summary_json_path, summary_csv_path = write_run_summary(summary, output_dir)
        manifest_path = write_run_manifest(
            _build_manifest(
                spec=spec,
                spec_path=resolved_spec_path,
                artifact=artifact,
                device=device,
            ),
            output_dir,
        )
    except (OSError, ValueError) as exc:
        raise ExperimentRunError(str(exc)) from exc

    return ExperimentRunOutput(
        output_dir=output_dir,
        manifest_path=manifest_path,
        perturbation_trace_path=perturbation_trace_path,
        candidate_trace_path=candidate_trace_output_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        clean_baseline_metrics=clean_baseline_metrics,
        flip_count=run_result.flip_count,
        stopped_because=run_result.stopped_because,
        device=device,
    )


def _load_benchmark_artifact(spec: ExperimentSpec) -> Any:
    if (
        spec.model.benchmark != CIFAR_RESNET20_BENCHMARK_ID
        or spec.model.architecture != "resnet20"
    ):
        msg = (
            "unsupported benchmark model configuration "
            f"{spec.model.benchmark!r}/{spec.model.architecture!r}; "
            "netflip run currently supports 'cifar10-resnet20'/'resnet20'"
        )
        raise UnsupportedBenchmarkError(msg)
    if spec.model.quantization.scale_path is None:
        msg = "CIFAR-10 ResNet-20 runs require model.quantization.scale_path"
        raise ExperimentRunError(msg)
    try:
        return load_cifar_resnet20_quantized_artifact(
            checkpoint_path=spec.model.checkpoint.path,
            scale_path=spec.model.quantization.scale_path,
            num_classes=spec.model.num_classes,
        )
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        raise ExperimentRunError(str(exc)) from exc


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


def _classification_cross_entropy(
    model: Any,
    batch: Any,
    *,
    device: str,
) -> float:
    try:
        import torch  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = "classification attack objective requires PyTorch to be installed"
        raise ModuleNotFoundError(msg) from exc

    try:
        inputs, targets = batch
    except (TypeError, ValueError) as exc:
        msg = "selection batch must contain inputs and ground-truth targets"
        raise TypeError(msg) from exc

    move_model = getattr(model, "to", None)
    if device is not None and move_model is not None:
        move_model(device)
    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)

    was_training = getattr(model, "training", False)
    set_eval = getattr(model, "eval", None)
    if set_eval is not None:
        set_eval()
    try:
        with torch.no_grad():
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                reduction="mean",
            )
    finally:
        set_train = getattr(model, "train", None)
        if was_training and set_train is not None:
            set_train()

    return float(loss.item())


def _first_selection_batch(dataloader: Any) -> Any:
    try:
        return next(iter(dataloader))
    except StopIteration as exc:
        msg = "selection batch requires at least one sample"
        raise ValueError(msg) from exc


def _dataloader_batch_size(spec: ExperimentSpec) -> int:
    if spec.scenario.type == ATTACK_SCENARIO_TYPE:
        return spec.scenario.selection_batch_size
    return 128


def _adapter_model(adapter: ModelAdapter, *, fallback: Any) -> Any:
    return getattr(adapter, "model", fallback)


def _build_manifest(
    *,
    spec: ExperimentSpec,
    spec_path: Path,
    artifact: Any,
    device: str,
) -> Any:
    resolved_spec_path = spec_path.resolve()
    checkpoint_path = Path(artifact.checkpoint_path).resolve()
    return build_run_manifest(
        run_id=spec.run_id,
        created_at=datetime.now(timezone.utc),
        git_commit=_current_git_commit(),
        experiment_spec_hash=_file_sha256(resolved_spec_path),
        model_artifact_id=spec.model.benchmark,
        model_checkpoint_path=str(checkpoint_path),
        model_checkpoint_checksum=_file_sha256(checkpoint_path),
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
        resolved_scale_path = Path(scale_path).resolve()
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
    root: str | PathLike[str],
    split: str,
    sample_limit: int | None,
) -> str:
    payload = {
        "dataset_name": dataset_name,
        "root": str(root),
        "sample_limit": sample_limit,
        "split": split,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
