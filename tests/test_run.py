from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import netflip.run as run_module
from netflip.run import ExperimentRunError, UnsupportedBenchmarkError


def _write_soft_error_spec(tmp_path: Path, *, include_scale_path: bool = True) -> Path:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    scale_path = tmp_path / "scales.json"
    scale_path.write_text("{}", encoding="utf-8")
    quantization = {
        "codec": "signed-int8-two-complement",
        "scale_granularity": "per-tensor",
    }
    if include_scale_path:
        quantization["scale_path"] = str(scale_path)
    spec_path = tmp_path / "soft-error.yaml"
    spec = {
        "schema_version": "2026.1",
        "run_id": "run-errors",
        "device": "cpu",
        "model": {
            "benchmark": "cifar10-resnet20",
            "architecture": "resnet20",
            "num_classes": 10,
            "checkpoint": {
                "path": str(checkpoint_path),
                "format": "pytorch-state-dict",
            },
            "quantization": quantization,
        },
        "dataset": {
            "name": "cifar10",
            "root": str(tmp_path / "data"),
            "selection_split": "train",
            "evaluation_split": "test",
        },
        "scenario": {
            "type": "soft_error",
            "fault_model": "uniform-eligible-bit",
            "fault_schedule": "one-bit-step",
            "fault_budget": {"max_flip_count": 1},
            "rng_seed": 7,
        },
        "output_dir": str(tmp_path / "run-output"),
    }
    spec_path.write_text(run_module.json.dumps(spec), encoding="utf-8")
    return spec_path


def _fake_artifact(tmp_path: Path) -> Any:
    return SimpleNamespace(
        model=object(),
        adapter=SimpleNamespace(model="adapter-model"),
        checkpoint_path=tmp_path / "checkpoint.pt",
        scale_path=tmp_path / "scales.json",
        quantization=SimpleNamespace(
            codec="signed-int8-two-complement",
            scale_granularity="per-tensor",
            tensors={"features.weight": object()},
        ),
    )


def test_execute_experiment_run_wraps_missing_spec_path(tmp_path: Path) -> None:
    with pytest.raises(ExperimentRunError, match="failed to load Experiment Spec"):
        run_module.execute_experiment_run(tmp_path / "missing.yaml")


def test_execute_experiment_run_wraps_device_resolution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_soft_error_spec(tmp_path)

    def unavailable_device(requested: str) -> str:
        raise run_module.RuntimeDeviceUnavailableError("no accelerator")

    monkeypatch.setattr(run_module, "resolve_torch_device", unavailable_device)

    with pytest.raises(ExperimentRunError, match="no accelerator"):
        run_module.execute_experiment_run(spec_path)


def test_execute_experiment_run_wraps_dataloader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_soft_error_spec(tmp_path)
    monkeypatch.setattr(run_module, "resolve_torch_device", lambda requested: "cpu")
    monkeypatch.setattr(
        run_module,
        "_load_benchmark_artifact",
        lambda spec: _fake_artifact(tmp_path),
    )

    def missing_dataloaders(**kwargs: Any) -> object:
        raise FileNotFoundError("missing cifar data")

    monkeypatch.setattr(run_module, "build_cifar10_dataloaders", missing_dataloaders)

    with pytest.raises(ExperimentRunError, match="missing cifar data"):
        run_module.execute_experiment_run(spec_path)


def test_execute_experiment_run_wraps_soft_error_run_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_soft_error_spec(tmp_path)
    monkeypatch.setattr(run_module, "resolve_torch_device", lambda requested: "cpu")
    monkeypatch.setattr(
        run_module,
        "_load_benchmark_artifact",
        lambda spec: _fake_artifact(tmp_path),
    )
    monkeypatch.setattr(
        run_module,
        "build_cifar10_dataloaders",
        lambda **kwargs: SimpleNamespace(evaluation=["eval"]),
    )
    monkeypatch.setattr(
        run_module,
        "evaluate_classification_metrics",
        lambda model, dataloader, *, device: {"accuracy": 1.0},
    )

    def broken_soft_error_run(**kwargs: Any) -> object:
        raise ValueError("no eligible int8 tensors")

    monkeypatch.setattr(
        run_module,
        "run_uniform_random_soft_error_baseline",
        broken_soft_error_run,
    )

    with pytest.raises(ExperimentRunError, match="no eligible int8 tensors"):
        run_module.execute_experiment_run(spec_path)


def test_execute_experiment_run_wraps_output_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_soft_error_spec(tmp_path)
    monkeypatch.setattr(run_module, "resolve_torch_device", lambda requested: "cpu")
    monkeypatch.setattr(
        run_module,
        "_load_benchmark_artifact",
        lambda spec: _fake_artifact(tmp_path),
    )
    monkeypatch.setattr(
        run_module,
        "build_cifar10_dataloaders",
        lambda **kwargs: SimpleNamespace(evaluation=["eval"]),
    )
    monkeypatch.setattr(
        run_module,
        "evaluate_classification_metrics",
        lambda model, dataloader, *, device: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        run_module,
        "run_uniform_random_soft_error_baseline",
        lambda **kwargs: SimpleNamespace(
            perturbation_trace=(),
            flip_count=0,
            stopped_because="fault_budget",
        ),
    )

    def broken_trace_writer(entries: object, output_dir: object) -> Path:
        raise OSError("cannot write trace")

    monkeypatch.setattr(run_module, "write_perturbation_trace", broken_trace_writer)

    with pytest.raises(ExperimentRunError, match="cannot write trace"):
        run_module.execute_experiment_run(spec_path)


def test_load_benchmark_artifact_reports_unsupported_benchmark(
    tmp_path: Path,
) -> None:
    spec = run_module.load_experiment_spec(_write_soft_error_spec(tmp_path))
    unsupported_model = spec.model.model_copy(update={"benchmark": "other-benchmark"})
    unsupported_spec = spec.model_copy(update={"model": unsupported_model})

    with pytest.raises(UnsupportedBenchmarkError, match="unsupported benchmark"):
        run_module._load_benchmark_artifact(unsupported_spec)


def test_load_benchmark_artifact_requires_scale_path(tmp_path: Path) -> None:
    spec = run_module.load_experiment_spec(
        _write_soft_error_spec(tmp_path, include_scale_path=False)
    )

    with pytest.raises(ExperimentRunError, match="scale_path"):
        run_module._load_benchmark_artifact(spec)


def test_load_benchmark_artifact_wraps_loader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = run_module.load_experiment_spec(_write_soft_error_spec(tmp_path))

    def broken_loader(**kwargs: Any) -> object:
        raise ValueError("bad checkpoint")

    monkeypatch.setattr(
        run_module,
        "load_cifar_resnet20_quantized_artifact",
        broken_loader,
    )

    with pytest.raises(ExperimentRunError, match="bad checkpoint"):
        run_module._load_benchmark_artifact(spec)


def test_run_metadata_helpers_cover_optional_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = run_module._quantization_metadata(
        SimpleNamespace(
            quantization=SimpleNamespace(
                codec="signed-int8-two-complement",
                scale_granularity="per-tensor",
                tensors={"features.weight": object()},
            )
        )
    )
    assert metadata == {
        "codec": "signed-int8-two-complement",
        "scale_granularity": "per-tensor",
        "tensor_count": 1,
    }
    assert run_module._dataset_id("cifar10", "test", 12) == "cifar10-test-12"
    assert run_module._dataset_request_checksum(
        dataset_name="cifar10",
        root=tmp_path,
        split="test",
        sample_limit=None,
    ) == run_module._dataset_request_checksum(
        dataset_name="cifar10",
        root=str(tmp_path),
        split="test",
        sample_limit=None,
    )
    monkeypatch.setattr(
        run_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert run_module._current_git_commit() == "unknown"


def test_json_scalar_mapping_rejects_invalid_metric_shapes() -> None:
    invalid_keys: Any = {1: "ok"}
    with pytest.raises(TypeError, match="keys"):
        run_module._json_scalar_mapping(invalid_keys, context="metrics")

    with pytest.raises(TypeError, match="JSON scalar"):
        run_module._json_scalar_mapping({"nested": {"bad": "shape"}}, context="metrics")


def test_dependency_versions_skips_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(package_name: str) -> str:
        if package_name == "click":
            return "8.4.1"
        raise run_module.PackageNotFoundError(package_name)

    monkeypatch.setattr(run_module, "version", fake_version)

    assert run_module._dependency_versions() == {"click": "8.4.1"}
