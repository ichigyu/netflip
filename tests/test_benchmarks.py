from __future__ import annotations

import sys
from typing import Any

import pytest

from netflip.benchmarks import (
    CIFAR10_CLASSES,
    CIFAR_RESNET20_BENCHMARK_ID,
    ResNet20Config,
    build_cifar_resnet20,
)


def test_resnet20_config_describes_cifar_benchmark() -> None:
    config = ResNet20Config()

    assert CIFAR_RESNET20_BENCHMARK_ID == "cifar10-resnet20"
    assert config.depth == 20
    assert config.input_channels == 3
    assert config.num_classes == CIFAR10_CLASSES


def test_build_cifar_resnet20_requires_pytorch_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cifar_resnet20, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match="requires PyTorch"):
        build_cifar_resnet20()


def test_build_cifar_resnet20_forward_pass_smoke() -> None:
    torch = pytest.importorskip("torch")

    model = build_cifar_resnet20()
    outputs = model(torch.zeros(2, 3, 32, 32))

    assert model.benchmark_id == CIFAR_RESNET20_BENCHMARK_ID
    assert model.config.depth == 20
    assert tuple(outputs.shape) == (2, CIFAR10_CLASSES)
