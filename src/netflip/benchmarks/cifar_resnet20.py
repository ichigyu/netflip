"""CIFAR-compatible ResNet-20 benchmark model."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from netflip.pytorch_adapter import PyTorchModelAdapter

CIFAR_RESNET20_BENCHMARK_ID = "cifar10-resnet20"
CIFAR10_CLASSES = 10
CIFAR10_NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_SPLITS = ("train", "test")
SIGNED_INT8_TWO_COMPLEMENT_CODEC = "signed-int8-two-complement"
PER_TENSOR_SCALE_GRANULARITY = "per-tensor"


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

    role: Cifar10DatasetRole | str
    root: str | PathLike[str]
    split: Literal["train", "test"]
    sample_limit: int | None = None
    download: bool = False


@dataclass(frozen=True)
class Cifar10DataLoaders:
    """Selection and Evaluation Dataset loaders for CIFAR-10 runs."""

    selection: Any
    evaluation: Any


@dataclass(frozen=True)
class PerTensorScale:
    """Quantization metadata for one per-tensor scale."""

    tensor_name: str
    scale: float
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class PerTensorScaleMetadata:
    """BFA-compatible int8 per-tensor scale metadata."""

    codec: str
    scale_granularity: str
    tensors: Mapping[str, PerTensorScale]

    def scale_for(self, tensor_name: str) -> float:
        """Return the scale for a tensor name."""
        try:
            return self.tensors[tensor_name].scale
        except KeyError as exc:
            msg = f"unknown per-tensor scale metadata tensor: {tensor_name}"
            raise KeyError(msg) from exc


@dataclass(frozen=True)
class CifarResNet20QuantizedArtifact:
    """Loaded CIFAR-10 ResNet-20 Quantized Model Artifact."""

    model: CifarResNet20Model
    adapter: PyTorchModelAdapter
    checkpoint_path: Path
    scale_path: Path
    quantization: PerTensorScaleMetadata


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


def load_cifar_resnet20_quantized_artifact(
    *,
    checkpoint_path: str | PathLike[str],
    scale_path: str | PathLike[str],
    num_classes: int = CIFAR10_CLASSES,
    input_channels: int = 3,
) -> CifarResNet20QuantizedArtifact:
    """Load a CIFAR-10 ResNet-20 Quantized Model Artifact."""
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.is_file():
        msg = (
            "CIFAR-10 ResNet-20 checkpoint path is not a file: "
            f"{resolved_checkpoint_path}"
        )
        raise FileNotFoundError(msg)
    resolved_scale_path = Path(scale_path)
    if not resolved_scale_path.is_file():
        msg = (
            "CIFAR-10 ResNet-20 scale metadata path is not a file: "
            f"{resolved_scale_path}"
        )
        raise FileNotFoundError(msg)

    torch = _require_pytorch()
    model = build_cifar_resnet20(
        num_classes=num_classes,
        input_channels=input_channels,
    )
    expected_state = _state_dict_mapping(model)
    perturbable_tensor_names = _perturbable_weight_tensor_names(model)
    quantization = load_per_tensor_scale_metadata(
        resolved_scale_path,
        expected_tensor_names=perturbable_tensor_names,
    )
    checkpoint_state = _load_pytorch_state_dict(resolved_checkpoint_path, torch)

    _validate_checkpoint_state(
        checkpoint_state,
        expected_state=expected_state,
        quantization=quantization,
    )
    _validate_scale_tensor_metadata(
        quantization,
        checkpoint_state=checkpoint_state,
    )
    _load_validated_checkpoint_state(
        model,
        checkpoint_state,
        quantized_tensor_names=quantization.tensors.keys(),
        torch=torch,
    )

    return CifarResNet20QuantizedArtifact(
        model=model,
        adapter=PyTorchModelAdapter(model),
        checkpoint_path=resolved_checkpoint_path,
        scale_path=resolved_scale_path,
        quantization=quantization,
    )


def load_per_tensor_scale_metadata(
    path: str | PathLike[str],
    *,
    expected_tensor_names: Iterable[str] | None = None,
) -> PerTensorScaleMetadata:
    """Load and validate signed-int8 per-tensor scale metadata from JSON."""
    metadata_path = Path(path)
    try:
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"scale metadata must be valid JSON: {exc.msg}"
        raise ValueError(msg) from exc

    if not isinstance(raw_metadata, Mapping):
        msg = "scale metadata must be a JSON object"
        raise ValueError(msg)

    codec = _require_metadata_string(raw_metadata, "codec")
    if codec != SIGNED_INT8_TWO_COMPLEMENT_CODEC:
        msg = (
            "scale metadata codec must be "
            f"{SIGNED_INT8_TWO_COMPLEMENT_CODEC!r}; got {codec!r}"
        )
        raise ValueError(msg)

    scale_granularity = _require_metadata_string(raw_metadata, "scale_granularity")
    if scale_granularity != PER_TENSOR_SCALE_GRANULARITY:
        msg = (
            "scale metadata scale_granularity must be "
            f"{PER_TENSOR_SCALE_GRANULARITY!r}; got {scale_granularity!r}"
        )
        raise ValueError(msg)

    raw_tensors = raw_metadata.get("tensors")
    if not isinstance(raw_tensors, Mapping):
        msg = "scale metadata must contain a 'tensors' object"
        raise ValueError(msg)

    tensors = _parse_per_tensor_scales(raw_tensors)
    if not tensors:
        msg = "scale metadata must contain at least one tensor scale"
        raise ValueError(msg)

    if expected_tensor_names is not None:
        _validate_expected_scale_tensor_names(
            tensors.keys(),
            expected_tensor_names=expected_tensor_names,
        )

    return PerTensorScaleMetadata(
        codec=codec,
        scale_granularity=scale_granularity,
        tensors=tensors,
    )


def build_cifar10_dataset(request: Cifar10DatasetRequest) -> Any:
    """Build a CIFAR-10 Dataset for a selection or evaluation role."""
    role = _normalize_cifar10_dataset_role(request.role)
    if request.split not in CIFAR10_SPLITS:
        msg = f"CIFAR-10 split must be one of {CIFAR10_SPLITS}; got {request.split!r}"
        raise ValueError(msg)
    if request.sample_limit is not None and request.sample_limit < 0:
        msg = f"CIFAR-10 {role.value} sample_limit must be greater than or equal to 0"
        raise ValueError(msg)

    root = Path(request.root)
    if not request.download and not root.exists():
        msg = (
            f"CIFAR-10 {role.value} dataset root does not exist: "
            f"{root}. Set download=True only when automatic download is intended."
        )
        raise FileNotFoundError(msg)

    torchvision = _require_torchvision()
    dataset = _load_cifar10_dataset(
        torchvision=torchvision,
        role=role,
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
    role: Cifar10DatasetRole,
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
        if "Dataset not found" not in str(exc):
            raise
        msg = (
            f"CIFAR-10 {role.value} {split} split was not found under "
            f"dataset root {root}. "
            "Provide an existing CIFAR-10 root or set download=True explicitly."
        )
        raise FileNotFoundError(msg) from exc


def _require_metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        msg = f"scale metadata field {key!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def _parse_per_tensor_scales(
    raw_tensors: Mapping[str, Any],
) -> dict[str, PerTensorScale]:
    tensors: dict[str, PerTensorScale] = {}
    for tensor_name, raw_tensor in raw_tensors.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            msg = "scale metadata tensor names must be non-empty strings"
            raise ValueError(msg)
        if not isinstance(raw_tensor, Mapping):
            msg = f"scale metadata for tensor {tensor_name!r} must be an object"
            raise ValueError(msg)

        scale = _parse_positive_scale(raw_tensor.get("scale"), tensor_name)
        shape = _parse_optional_shape(raw_tensor.get("shape"), tensor_name)
        dtype = _parse_optional_dtype(raw_tensor.get("dtype"), tensor_name)
        tensors[tensor_name] = PerTensorScale(
            tensor_name=tensor_name,
            scale=scale,
            shape=shape,
            dtype=dtype,
        )
    return tensors


def _parse_positive_scale(value: Any, tensor_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"scale metadata for tensor {tensor_name!r} must include a numeric scale"
        raise ValueError(msg)
    scale = float(value)
    if not isfinite(scale) or scale <= 0:
        msg = f"scale for tensor {tensor_name!r} must be a finite positive number"
        raise ValueError(msg)
    return scale


def _parse_optional_shape(value: Any, tensor_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        msg = f"shape metadata for tensor {tensor_name!r} must be a list of integers"
        raise ValueError(msg)
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            msg = (
                f"shape metadata for tensor {tensor_name!r} must be a list of integers"
            )
            raise ValueError(msg)
        if dimension < 0:
            msg = f"shape metadata for tensor {tensor_name!r} must be non-negative"
            raise ValueError(msg)
        shape.append(dimension)
    return tuple(shape)


def _parse_optional_dtype(value: Any, tensor_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"dtype metadata for tensor {tensor_name!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def _validate_expected_scale_tensor_names(
    tensor_names: Iterable[str],
    *,
    expected_tensor_names: Iterable[str],
) -> None:
    actual = set(tensor_names)
    expected = set(expected_tensor_names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        msg = "scale metadata is missing per-tensor scales for: " + ", ".join(missing)
        raise ValueError(msg)
    if extra:
        msg = (
            "scale metadata contains tensors that are not perturbable "
            "CIFAR-10 ResNet-20 weights: " + ", ".join(extra)
        )
        raise ValueError(msg)


def _load_pytorch_state_dict(
    checkpoint_path: Path,
    torch: Any,
) -> Mapping[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, Mapping) and _looks_like_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping) and _looks_like_state_dict(nested):
                return nested

    msg = (
        f"checkpoint at {checkpoint_path} must be a PyTorch state_dict mapping "
        "tensor names to tensors"
    )
    raise ValueError(msg)


def _looks_like_state_dict(checkpoint: Mapping[str, Any]) -> bool:
    return all(
        isinstance(name, str) and _is_tensor_like(value)
        for name, value in checkpoint.items()
    )


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _state_dict_mapping(model: Any) -> Mapping[str, Any]:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        msg = "CIFAR-10 ResNet-20 model must provide a callable state_dict"
        raise TypeError(msg)
    state = state_dict()
    if not isinstance(state, Mapping):
        msg = "CIFAR-10 ResNet-20 model state_dict must return a mapping"
        raise TypeError(msg)
    return state


def _perturbable_weight_tensor_names(model: Any) -> tuple[str, ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        msg = "CIFAR-10 ResNet-20 model must provide callable named_parameters"
        raise TypeError(msg)
    parameters = cast("Iterable[tuple[str, Any]]", named_parameters())
    names = tuple(
        name
        for name, parameter in parameters
        if _is_weight_parameter_name(name) and int(parameter.numel()) > 0
    )
    if not names:
        msg = "CIFAR-10 ResNet-20 model exposes no perturbable weight tensors"
        raise ValueError(msg)
    return names


def _is_weight_parameter_name(name: str) -> bool:
    return name == "weight" or name.endswith(".weight")


def _validate_checkpoint_state(
    checkpoint_state: Mapping[str, Any],
    *,
    expected_state: Mapping[str, Any],
    quantization: PerTensorScaleMetadata,
) -> None:
    actual_names = set(checkpoint_state)
    expected_names = set(expected_state)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        msg = "checkpoint is missing tensors: " + ", ".join(missing)
        raise ValueError(msg)
    if extra:
        msg = "checkpoint contains unexpected tensors: " + ", ".join(extra)
        raise ValueError(msg)

    quantized_tensor_names = set(quantization.tensors)
    for tensor_name, checkpoint_tensor in checkpoint_state.items():
        expected_tensor = expected_state[tensor_name]
        _validate_tensor_shape(
            tensor_name,
            checkpoint_tensor,
            expected_tensor=expected_tensor,
        )
        if tensor_name in quantized_tensor_names:
            _validate_signed_int8_checkpoint_dtype(tensor_name, checkpoint_tensor)
        else:
            _validate_checkpoint_dtype(
                tensor_name,
                checkpoint_tensor,
                expected_tensor=expected_tensor,
            )


def _validate_tensor_shape(
    tensor_name: str,
    tensor: Any,
    *,
    expected_tensor: Any,
) -> None:
    actual_shape = _tensor_shape(tensor)
    expected_shape = _tensor_shape(expected_tensor)
    if actual_shape != expected_shape:
        msg = (
            f"checkpoint tensor {tensor_name!r} has shape {actual_shape}; "
            f"expected {expected_shape}"
        )
        raise ValueError(msg)


def _validate_signed_int8_checkpoint_dtype(tensor_name: str, tensor: Any) -> None:
    dtype = _tensor_dtype(tensor)
    if not _is_signed_int8_dtype(dtype):
        msg = (
            f"checkpoint tensor {tensor_name!r} must have signed int8 dtype; "
            f"got {dtype}"
        )
        raise ValueError(msg)


def _validate_checkpoint_dtype(
    tensor_name: str,
    tensor: Any,
    *,
    expected_tensor: Any,
) -> None:
    actual_dtype = _tensor_dtype(tensor)
    expected_dtype = _tensor_dtype(expected_tensor)
    if actual_dtype != expected_dtype:
        msg = (
            f"checkpoint tensor {tensor_name!r} has dtype {actual_dtype}; "
            f"expected {expected_dtype}"
        )
        raise ValueError(msg)


def _validate_scale_tensor_metadata(
    quantization: PerTensorScaleMetadata,
    *,
    checkpoint_state: Mapping[str, Any],
) -> None:
    for tensor_name, metadata in quantization.tensors.items():
        checkpoint_tensor = checkpoint_state[tensor_name]
        if metadata.shape is not None and metadata.shape != _tensor_shape(
            checkpoint_tensor
        ):
            msg = (
                f"scale metadata for tensor {tensor_name!r} has shape "
                f"{metadata.shape}; expected {_tensor_shape(checkpoint_tensor)}"
            )
            raise ValueError(msg)
        if metadata.dtype is not None and not _dtype_metadata_matches(
            metadata.dtype,
            _tensor_dtype(checkpoint_tensor),
        ):
            msg = (
                f"scale metadata for tensor {tensor_name!r} has dtype "
                f"{metadata.dtype}; expected {_tensor_dtype(checkpoint_tensor)}"
            )
            raise ValueError(msg)


def _load_validated_checkpoint_state(
    model: Any,
    checkpoint_state: Mapping[str, Any],
    *,
    quantized_tensor_names: Iterable[str],
    torch: Any,
) -> None:
    quantized_names = set(quantized_tensor_names)
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        msg = "CIFAR-10 ResNet-20 model must provide callable load_state_dict"
        raise TypeError(msg)
    non_quantized_state = {
        name: tensor
        for name, tensor in checkpoint_state.items()
        if name not in quantized_names
    }
    try:
        load_state_dict(non_quantized_state, strict=False)
    except RuntimeError as exc:
        msg = f"checkpoint could not be loaded into the CIFAR-10 ResNet-20 model: {exc}"
        raise ValueError(msg) from exc

    for tensor_name in quantized_names:
        _replace_module_state_tensor(
            model,
            tensor_name,
            checkpoint_state[tensor_name],
            torch=torch,
        )


def _replace_module_state_tensor(
    model: Any,
    tensor_name: str,
    tensor: Any,
    *,
    torch: Any,
) -> None:
    module, attribute_name = _resolve_state_tensor_parent(model, tensor_name)
    clone = _clone_detached_tensor(tensor)
    if _is_model_parameter_name(model, tensor_name):
        parameter = torch.nn.Parameter(clone, requires_grad=False)
        setattr(module, attribute_name, parameter)
        return
    register_buffer = getattr(module, "register_buffer", None)
    if callable(register_buffer):
        register_buffer(attribute_name, clone)
        return
    setattr(module, attribute_name, clone)


def _is_model_parameter_name(model: Any, tensor_name: str) -> bool:
    return any(name == tensor_name for name, _parameter in model.named_parameters())


def _resolve_state_tensor_parent(model: Any, tensor_name: str) -> tuple[Any, str]:
    parts = tensor_name.split(".")
    module = model
    for part in parts[:-1]:
        try:
            module = getattr(module, part)
        except AttributeError as exc:
            msg = (
                f"checkpoint tensor {tensor_name!r} does not map to a model "
                f"module path at {part!r}"
            )
            raise ValueError(msg) from exc
    return module, parts[-1]


def _clone_detached_tensor(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        return clone()
    return tensor


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _tensor_dtype(tensor: Any) -> str:
    return str(tensor.dtype)


def _is_signed_int8_dtype(dtype: str) -> bool:
    return dtype in {"int8", "torch.int8"}


def _dtype_metadata_matches(metadata_dtype: str, actual_dtype: str) -> bool:
    return metadata_dtype == actual_dtype or (
        metadata_dtype.rsplit(".", maxsplit=1)[-1]
        == actual_dtype.rsplit(".", maxsplit=1)[-1]
    )


def _normalize_cifar10_dataset_role(
    role: Cifar10DatasetRole | str,
) -> Cifar10DatasetRole:
    try:
        return Cifar10DatasetRole(role)
    except ValueError as exc:
        allowed_roles = tuple(role.value for role in Cifar10DatasetRole)
        msg = f"CIFAR-10 dataset role must be one of {allowed_roles}; got {role!r}"
        raise ValueError(msg) from exc


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
        msg = "benchmark runtime requires PyTorch to be installed"
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
