"""CIFAR-compatible ResNet-20 benchmark model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Protocol

CIFAR_RESNET20_BENCHMARK_ID = "cifar10-resnet20"
CIFAR10_CLASSES = 10
CIFAR10_NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_SPLITS = ("train", "test")


class Cifar10DatasetRole(str, Enum):
    """Role of a CIFAR-10 dataset in a NetFlip run."""

    SELECTION = "selection"
    EVALUATION = "evaluation"


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


@dataclass(frozen=True)
class Cifar10DatasetRequest:
    """Configuration for one CIFAR-10 Dataset role."""

    role: Cifar10DatasetRole
    root: str | PathLike[str]
    split: Literal["train", "test"]
    sample_limit: int | None = None
    download: bool = False


@dataclass(frozen=True)
class Cifar10DataLoaders:
    """Selection and Evaluation Dataset loaders for CIFAR-10 runs."""

    selection: Any
    evaluation: Any


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


def build_cifar10_dataset(request: Cifar10DatasetRequest) -> Any:
    """Build a CIFAR-10 Dataset for a selection or evaluation role."""
    if request.split not in CIFAR10_SPLITS:
        msg = f"CIFAR-10 split must be one of {CIFAR10_SPLITS}; got {request.split!r}"
        raise ValueError(msg)
    if request.sample_limit is not None and request.sample_limit < 0:
        msg = "CIFAR-10 sample_limit must be greater than or equal to 0"
        raise ValueError(msg)

    root = Path(request.root)
    if not request.download and not root.exists():
        msg = (
            "CIFAR-10 dataset root does not exist: "
            f"{root}. Set download=True only when automatic download is intended."
        )
        raise FileNotFoundError(msg)

    torchvision = _require_torchvision()
    dataset = _load_cifar10_dataset(
        torchvision=torchvision,
        root=root,
        split=request.split,
        download=request.download,
    )
    if request.sample_limit is None:
        return dataset

    torch = _require_pytorch()
    return torch.utils.data.Subset(
        dataset, range(min(request.sample_limit, len(dataset)))
    )


def build_cifar10_dataloader(
    request: Cifar10DatasetRequest,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    """Build a PyTorch DataLoader for one CIFAR-10 Dataset role."""
    if batch_size <= 0:
        msg = "CIFAR-10 dataloader batch_size must be greater than 0"
        raise ValueError(msg)
    dataset = build_cifar10_dataset(request)
    torch = _require_pytorch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_cifar10_dataloaders(
    *,
    root: str | PathLike[str],
    selection_split: Literal["train", "test"] = "train",
    evaluation_split: Literal["train", "test"] = "test",
    batch_size: int = 128,
    selection_sample_limit: int | None = None,
    evaluation_sample_limit: int | None = None,
    download: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Cifar10DataLoaders:
    """Build selection and evaluation DataLoaders for CIFAR-10."""
    return Cifar10DataLoaders(
        selection=build_cifar10_dataloader(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.SELECTION,
                root=root,
                split=selection_split,
                sample_limit=selection_sample_limit,
                download=download,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        evaluation=build_cifar10_dataloader(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.EVALUATION,
                root=root,
                split=evaluation_split,
                sample_limit=evaluation_sample_limit,
                download=download,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def cifar10_evaluation_transform() -> Any:
    """Return the standard CIFAR-10 tensor normalization transform."""
    torchvision = _require_torchvision()
    return _make_cifar10_evaluation_transform(torchvision)


def compute_top1_accuracy(
    model: TorchModule,
    dataloader: Any,
    *,
    device: Any | None = None,
) -> float:
    """Compute top-1 classification accuracy for a model and DataLoader."""
    metrics = evaluate_classification_metrics(model, dataloader, device=device)
    return metrics["top1_accuracy"]


def compute_cross_entropy_loss(
    model: TorchModule,
    dataloader: Any,
    *,
    device: Any | None = None,
) -> float:
    """Compute mean cross-entropy loss for a model and DataLoader."""
    metrics = evaluate_classification_metrics(model, dataloader, device=device)
    return metrics["cross_entropy"]


def evaluate_classification_metrics(
    model: TorchModule,
    dataloader: Any,
    *,
    device: Any | None = None,
) -> dict[str, float]:
    """Compute top-1 accuracy and mean cross-entropy for classification."""
    torch = _require_pytorch()
    move_model = getattr(model, "to", None)
    if device is not None and move_model is not None:
        move_model(device)

    was_training = getattr(model, "training", False)
    set_eval = getattr(model, "eval", None)
    if set_eval is not None:
        set_eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    try:
        with torch.no_grad():
            for inputs, targets in dataloader:
                if device is not None:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                outputs = model(inputs)
                batch_size = int(targets.shape[0])
                loss = torch.nn.functional.cross_entropy(
                    outputs,
                    targets,
                    reduction="sum",
                )
                predictions = outputs.argmax(dim=1)
                correct += int((predictions == targets).sum().item())
                total += batch_size
                loss_sum += float(loss.item())
    finally:
        set_train = getattr(model, "train", None)
        if was_training and set_train is not None:
            set_train()

    if total == 0:
        msg = "classification metrics require at least one sample"
        raise ValueError(msg)
    return {
        "top1_accuracy": correct / total,
        "cross_entropy": loss_sum / total,
    }


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


def _load_cifar10_dataset(
    *,
    torchvision: Any,
    root: Path,
    split: Literal["train", "test"],
    download: bool,
) -> Any:
    try:
        return torchvision.datasets.CIFAR10(
            root=str(root),
            train=split == "train",
            transform=_make_cifar10_evaluation_transform(torchvision),
            download=download,
        )
    except RuntimeError as exc:
        msg = (
            f"CIFAR-10 {split} split was not found under dataset root {root}. "
            "Provide an existing CIFAR-10 root or set download=True explicitly."
        )
        raise FileNotFoundError(msg) from exc


def _make_cifar10_evaluation_transform(torchvision: Any) -> Any:
    transforms = torchvision.transforms
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORMALIZATION_MEAN, CIFAR10_NORMALIZATION_STD),
        ]
    )


def _require_pytorch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = "CIFAR-10 benchmark runtime requires PyTorch to be installed"
        raise ModuleNotFoundError(msg) from exc


def _require_torchvision() -> Any:
    try:
        return import_module("torchvision")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = (
            "CIFAR-10 benchmark data loading requires torchvision to be installed; "
            "install NetFlip with the benchmark extra"
        )
        raise ModuleNotFoundError(msg) from exc
