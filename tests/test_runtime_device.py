from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from netflip.manifest import build_run_manifest
from netflip.runtime_device import (
    RuntimeDeviceUnavailableError,
    resolve_torch_device,
)


class _Availability:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, *, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = _Availability(cuda)
        self.backends = SimpleNamespace(mps=_Availability(mps))


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    mps: bool = False,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda=cuda, mps=mps))


def _manifest_kwargs() -> dict[str, Any]:
    return {
        "run_id": "run-001",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "git_commit": "abc1234",
        "experiment_spec_hash": "sha256:spec",
        "model_artifact_id": "resnet20-cifar10",
        "model_checkpoint_path": "models/resnet20.pt",
        "model_checkpoint_checksum": "sha256:checkpoint",
        "quantization_metadata": {"codec": "signed-int8-two-complement"},
        "scenario_metadata": {"scenario_type": "soft_error"},
        "selection_dataset_id": "cifar10-train",
        "selection_dataset_checksum": "sha256:selection",
        "evaluation_dataset_id": "cifar10-test",
        "evaluation_dataset_checksum": "sha256:evaluation",
        "rng_seeds": {"python": 1, "torch": 2},
        "device": "cpu",
        "netflip_version": "0.1.0",
        "dependencies": {"click": "8.1.8"},
        "output_schema_version": "2026.1",
    }


def test_auto_chooses_cuda_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=True, mps=True)

    assert resolve_torch_device("auto") == "cuda"


def test_auto_chooses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=False, mps=True)

    assert resolve_torch_device("auto") == "mps"


def test_auto_chooses_cpu_when_accelerators_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)

    assert resolve_torch_device("auto") == "cpu"


def test_auto_chooses_cpu_when_torch_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.runtime_device as runtime_device

    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(runtime_device, "import_module", missing_import)

    assert resolve_torch_device("auto") == "cpu"


def test_explicit_cpu_resolves_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.runtime_device as runtime_device

    def forbidden_import(name: str) -> Any:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(runtime_device, "import_module", forbidden_import)

    assert resolve_torch_device("cpu") == "cpu"


def test_unsupported_device_request_fails_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.runtime_device as runtime_device

    def forbidden_import(name: str) -> Any:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(runtime_device, "import_module", forbidden_import)

    with pytest.raises(ValueError, match="unsupported PyTorch device request"):
        resolve_torch_device("tpu")


def test_explicit_cuda_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=False, mps=True)

    with pytest.raises(RuntimeDeviceUnavailableError, match=r"cuda.*unavailable"):
        resolve_torch_device("cuda")


def test_explicit_mps_fails_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=True, mps=False)

    with pytest.raises(RuntimeDeviceUnavailableError, match=r"mps.*unavailable"):
        resolve_torch_device("mps")


def test_explicit_accelerator_fails_when_torch_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.runtime_device as runtime_device

    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(runtime_device, "import_module", missing_import)

    with pytest.raises(RuntimeDeviceUnavailableError, match="PyTorch is not installed"):
        resolve_torch_device("cuda")


def test_selected_device_serializes_into_run_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=False, mps=True)
    kwargs = _manifest_kwargs()
    kwargs["device"] = resolve_torch_device("auto")

    manifest = build_run_manifest(**kwargs)

    assert manifest.device == "mps"
    assert '"device":"mps"' in manifest.model_dump_json()
