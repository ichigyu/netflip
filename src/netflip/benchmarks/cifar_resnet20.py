"""CIFAR-compatible ResNet-20 benchmark model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

CIFAR_RESNET20_BENCHMARK_ID = "cifar10-resnet20"
CIFAR10_CLASSES = 10


@dataclass(frozen=True)
class ResNet20Config:
    """Construction settings for the CIFAR-10 ResNet-20 benchmark model."""

    input_channels: int = 3
    num_classes: int = CIFAR10_CLASSES
    blocks_per_stage: int = 3
    base_channels: int = 16

    @property
    def depth(self) -> int:
        """Return the nominal ResNet depth for this CIFAR configuration."""
        return 6 * self.blocks_per_stage + 2


class TorchModule(Protocol):
    """Minimal structural type for lazily imported PyTorch modules."""

    training: bool

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the module."""
        ...


class CifarResNet20Model(TorchModule, Protocol):
    """Structural return type for the CIFAR-10 ResNet-20 benchmark model."""

    benchmark_id: str
    config: ResNet20Config


def build_cifar_resnet20(
    *,
    num_classes: int = CIFAR10_CLASSES,
    input_channels: int = 3,
) -> CifarResNet20Model:
    """Construct the CIFAR-10 ResNet-20 benchmark model.

    PyTorch is intentionally optional for NetFlip's core package. This
    constructor imports it lazily so spec parsing and trace tooling remain
    available in environments that do not install benchmark runtime packages.
    """
    torch = _require_pytorch()
    config = ResNet20Config(
        input_channels=input_channels,
        num_classes=num_classes,
    )
    return _make_cifar_resnet(config, torch)


@dataclass(frozen=True)
class _BlockConfig:
    in_channels: int
    out_channels: int
    stride: int = 1


def _make_basic_block(config: _BlockConfig, torch: Any) -> TorchModule:
    nn = torch.nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = _conv3x3(
                config.in_channels,
                config.out_channels,
                stride=config.stride,
                torch=torch,
            )
            self.bn1 = nn.BatchNorm2d(config.out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = _conv3x3(
                config.out_channels,
                config.out_channels,
                torch=torch,
            )
            self.bn2 = nn.BatchNorm2d(config.out_channels)
            if config.stride != 1 or config.in_channels != config.out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        config.in_channels,
                        config.out_channels,
                        kernel_size=1,
                        stride=config.stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(config.out_channels),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, inputs: Any) -> Any:
            residual = self.shortcut(inputs)
            outputs = self.relu(self.bn1(self.conv1(inputs)))
            outputs = self.bn2(self.conv2(outputs))
            return self.relu(outputs + residual)

    return BasicBlock()


def _make_cifar_resnet(config: ResNet20Config, torch: Any) -> CifarResNet20Model:
    nn = torch.nn

    class CifarResNet20(nn.Module):
        benchmark_id = CIFAR_RESNET20_BENCHMARK_ID

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.in_channels = config.base_channels
            self.conv1 = _conv3x3(
                config.input_channels,
                config.base_channels,
                torch=torch,
            )
            self.bn1 = nn.BatchNorm2d(config.base_channels)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_stage(config.base_channels, stride=1)
            self.layer2 = self._make_stage(config.base_channels * 2, stride=2)
            self.layer3 = self._make_stage(config.base_channels * 4, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(config.base_channels * 4, config.num_classes)
            self._initialize_weights()

        def _make_stage(self, out_channels: int, *, stride: int) -> TorchModule:
            blocks = [
                _make_basic_block(
                    _BlockConfig(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=stride,
                    ),
                    torch,
                )
            ]
            self.in_channels = out_channels
            for _ in range(1, config.blocks_per_stage):
                blocks.append(
                    _make_basic_block(
                        _BlockConfig(
                            in_channels=self.in_channels,
                            out_channels=out_channels,
                        ),
                        torch,
                    )
                )
            return nn.Sequential(*blocks)

        def _initialize_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(self, inputs: Any) -> Any:
            outputs = self.relu(self.bn1(self.conv1(inputs)))
            outputs = self.layer1(outputs)
            outputs = self.layer2(outputs)
            outputs = self.layer3(outputs)
            outputs = self.avgpool(outputs)
            outputs = torch.flatten(outputs, 1)
            return self.fc(outputs)

    return CifarResNet20()


def _conv3x3(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    torch: Any,
) -> TorchModule:
    return torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _require_pytorch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = "build_cifar_resnet20 requires PyTorch to be installed"
        raise ModuleNotFoundError(msg) from exc
