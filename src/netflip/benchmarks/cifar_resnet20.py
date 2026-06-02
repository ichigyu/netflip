"""CIFAR-compatible ResNet-20 benchmark model."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from netflip.pytorch_adapter import PyTorchModelAdapter
from netflip.runtime_device import resolve_torch_device

CIFAR_RESNET20_BENCHMARK_ID = "cifar10-resnet20"
CIFAR10_CLASSES = 10
CIFAR10_NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_SPLITS = ("train", "test")
SIGNED_INT8_TWO_COMPLEMENT_CODEC = "signed-int8-two-complement"
PER_TENSOR_SCALE_GRANULARITY = "per-tensor"
BFA_CIFAR_RESNET20_EPOCHS = 160
BFA_CIFAR_RESNET20_BATCH_SIZE = 128
BFA_CIFAR_RESNET20_LEARNING_RATE = 0.1
BFA_CIFAR_RESNET20_LR_SCHEDULE = (80, 120)
BFA_CIFAR_RESNET20_LR_GAMMAS = (0.1, 0.1)
BFA_CIFAR_RESNET20_MOMENTUM = 0.9
BFA_CIFAR_RESNET20_WEIGHT_DECAY = 0.0003
BFA_CIFAR_RESNET20_NUM_WORKERS = 4

ProgressReporter = Callable[[str], None]


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


@dataclass(frozen=True)
class CifarResNet20ArtifactPreparationOutput:
    """Paths and metrics emitted by CIFAR-10 ResNet-20 artifact preparation."""

    fp32_checkpoint_path: Path
    int8_checkpoint_path: Path
    scale_path: Path
    evaluation_metrics: Mapping[str, float]
    device: str
    epochs: int


class TorchModule(Protocol):
    """Minimal structural type for lazily imported PyTorch modules."""

    training: bool

    def eval(self) -> Any:
        """Put the module in evaluation mode."""
        ...

    def train(self) -> Any:
        """Put the module in training mode."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the module."""
        ...


class CifarResNet20Model(TorchModule, Protocol):
    """Structural return type for the CIFAR-10 ResNet-20 benchmark model."""

    benchmark_id: str
    config: ResNet20Config

    def parameters(self) -> Iterable[Any]:
        """Return model parameters."""
        ...

    def named_parameters(self) -> Iterable[tuple[str, Any]]:
        """Return named model parameters."""
        ...


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


def prepare_cifar_resnet20_artifacts(
    *,
    dataset_root: str | PathLike[str],
    output_dir: str | PathLike[str] = "checkpoints/cifar10",
    download: bool = False,
    epochs: int = BFA_CIFAR_RESNET20_EPOCHS,
    batch_size: int = BFA_CIFAR_RESNET20_BATCH_SIZE,
    learning_rate: float = BFA_CIFAR_RESNET20_LEARNING_RATE,
    schedule: Iterable[int] = BFA_CIFAR_RESNET20_LR_SCHEDULE,
    gammas: Iterable[float] = BFA_CIFAR_RESNET20_LR_GAMMAS,
    momentum: float = BFA_CIFAR_RESNET20_MOMENTUM,
    weight_decay: float = BFA_CIFAR_RESNET20_WEIGHT_DECAY,
    train_sample_limit: int | None = None,
    evaluation_sample_limit: int | None = None,
    num_workers: int = BFA_CIFAR_RESNET20_NUM_WORKERS,
    device: str = "auto",
    rng_seed: int = 2026,
    progress: ProgressReporter | None = None,
) -> CifarResNet20ArtifactPreparationOutput:
    """Train and quantize CIFAR-10 ResNet-20 benchmark artifacts.

    The emitted int8 checkpoint and per-tensor scale metadata match the
    default paths used by the example CIFAR-10 Experiment Specs.
    """
    learning_rate_schedule = tuple(schedule)
    learning_rate_gammas = tuple(gammas)
    _validate_artifact_preparation_settings(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        schedule=learning_rate_schedule,
        gammas=learning_rate_gammas,
        momentum=momentum,
        weight_decay=weight_decay,
        train_sample_limit=train_sample_limit,
        evaluation_sample_limit=evaluation_sample_limit,
        num_workers=num_workers,
    )
    torch = _require_pytorch()
    resolved_device = resolve_torch_device(device)
    torch.manual_seed(rng_seed)

    _report_progress(progress, "== Setup ==")
    _report_progress(progress, f"  device: {resolved_device}")
    _report_progress(progress, f"  rng_seed: {rng_seed}")
    _report_progress(progress, f"  epochs: {epochs}")
    _report_progress(progress, f"  batch_size: {batch_size}")
    _report_progress(progress, f"  learning_rate: {learning_rate:.6g}")
    schedule_description = _learning_rate_schedule_description(
        learning_rate_schedule,
        learning_rate_gammas,
    )
    _report_progress(
        progress,
        f"  lr_schedule: {schedule_description}",
    )

    _report_progress(progress, "")
    _report_progress(progress, "== Model ==")
    _report_progress(progress, "  building: CIFAR-10 ResNet-20")
    model = build_cifar_resnet20()
    move_model = getattr(model, "to", None)
    if callable(move_model):
        move_model(resolved_device)

    if epochs > 0:
        _report_progress(progress, "")
        _report_progress(progress, "== Training Data ==")
        _report_dataset_settings(progress, root=dataset_root, download=download)
        train_loader = _build_cifar10_training_dataloader(
            root=dataset_root,
            batch_size=batch_size,
            download=download,
            sample_limit=train_sample_limit,
            num_workers=num_workers,
        )
        _report_dataloader_settings(progress, train_loader)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        _report_progress(progress, "")
        _report_progress(progress, "== Training ==")
        _report_progress(progress, "  epoch        lr       loss       top1")
        for epoch_index in range(epochs):
            epoch_learning_rate = _apply_learning_rate_schedule(
                optimizer,
                base_learning_rate=learning_rate,
                epoch_index=epoch_index,
                schedule=learning_rate_schedule,
                gammas=learning_rate_gammas,
            )
            epoch_metrics = _train_cifar_resnet20_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                torch=torch,
                device=resolved_device,
                progress=progress,
                epoch_index=epoch_index,
                epochs=epochs,
                learning_rate=epoch_learning_rate,
            )
            _report_progress(
                progress,
                (
                    f"  {epoch_index + 1:03d}/{epochs:03d}  "
                    f"{epoch_learning_rate:8.6g}  "
                    f"{epoch_metrics['loss']:9.6g}  "
                    f"{_format_percent(epoch_metrics['top1_accuracy']):>7}"
                ),
            )
    else:
        _report_progress(progress, "")
        _report_progress(progress, "== Training ==")
        _report_progress(progress, "  skipped: epochs=0")

    _report_progress(progress, "")
    _report_progress(progress, "== Evaluation Data ==")
    _report_dataset_settings(progress, root=dataset_root, download=download)
    evaluation_loader = build_cifar10_dataloader(
        Cifar10DatasetRequest(
            role=Cifar10DatasetRole.EVALUATION,
            root=dataset_root,
            split="test",
            sample_limit=evaluation_sample_limit,
            download=download,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    _report_dataloader_settings(progress, evaluation_loader)
    _report_progress(progress, "")
    _report_progress(progress, "== Clean Baseline ==")
    _report_progress(progress, "  evaluating...")
    evaluation_metrics = evaluate_classification_metrics(
        model,
        evaluation_loader,
        device=resolved_device,
    )
    _report_progress(
        progress,
        f"  top1: {_format_percent(evaluation_metrics['top1_accuracy'])}",
    )
    _report_progress(
        progress,
        f"  cross_entropy: {evaluation_metrics['cross_entropy']:.6g}",
    )

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    fp32_checkpoint_path = resolved_output_dir / "resnet20-fp32.pt"
    int8_checkpoint_path = resolved_output_dir / "resnet20-int8.pt"
    scale_path = resolved_output_dir / "resnet20-int8-scales.json"

    _report_progress(progress, "")
    _report_progress(progress, "== Artifacts ==")
    _report_progress(progress, f"  output_dir: {resolved_output_dir}")
    _report_progress(progress, f"  fp32_checkpoint: {fp32_checkpoint_path}")
    torch.save(_cpu_state_dict(model), str(fp32_checkpoint_path))
    _report_progress(progress, "  quantization: signed int8 per-tensor")
    int8_state, scale_metadata = quantize_cifar_resnet20_state_dict(model)
    _report_progress(progress, f"  int8_checkpoint: {int8_checkpoint_path}")
    torch.save(int8_state, str(int8_checkpoint_path))
    _report_progress(progress, f"  scale_metadata: {scale_path}")
    write_per_tensor_scale_metadata(scale_metadata, scale_path)
    _report_progress(progress, "  validation: loading quantized artifact")
    load_cifar_resnet20_quantized_artifact(
        checkpoint_path=int8_checkpoint_path,
        scale_path=scale_path,
    )
    _report_progress(progress, "")
    _report_progress(progress, "== Done ==")

    return CifarResNet20ArtifactPreparationOutput(
        fp32_checkpoint_path=fp32_checkpoint_path,
        int8_checkpoint_path=int8_checkpoint_path,
        scale_path=scale_path,
        evaluation_metrics=evaluation_metrics,
        device=resolved_device,
        epochs=epochs,
    )


def quantize_cifar_resnet20_state_dict(
    model: CifarResNet20Model,
) -> tuple[dict[str, Any], PerTensorScaleMetadata]:
    """Return a BFA-compatible int8 state dict and scale metadata for a model."""
    torch = _require_pytorch()
    state = _state_dict_mapping(model)
    perturbable_tensor_names = set(_perturbable_weight_tensor_names(model))
    quantized_state: dict[str, Any] = {}
    scale_tensors: dict[str, PerTensorScale] = {}

    for tensor_name, tensor in state.items():
        detached_tensor = _clone_detached_tensor(tensor)
        if tensor_name not in perturbable_tensor_names:
            quantized_state[tensor_name] = _cpu_tensor_clone(detached_tensor)
            continue

        cpu_tensor = _cpu_tensor_clone(detached_tensor).float()
        scale = _per_tensor_int8_scale(cpu_tensor, tensor_name=tensor_name, torch=torch)
        quantized_tensor = _quantize_tensor_to_int8(
            cpu_tensor,
            scale=scale,
            torch=torch,
        )
        quantized_state[tensor_name] = quantized_tensor
        scale_tensors[tensor_name] = PerTensorScale(
            tensor_name=tensor_name,
            scale=scale,
            shape=_tensor_shape(quantized_tensor),
            dtype="int8",
        )

    return (
        quantized_state,
        PerTensorScaleMetadata(
            codec=SIGNED_INT8_TWO_COMPLEMENT_CODEC,
            scale_granularity=PER_TENSOR_SCALE_GRANULARITY,
            tensors=scale_tensors,
        ),
    )


def write_per_tensor_scale_metadata(
    metadata: PerTensorScaleMetadata,
    path: str | PathLike[str],
) -> Path:
    """Write per-tensor scale metadata as validated JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_metadata = {
        "codec": metadata.codec,
        "scale_granularity": metadata.scale_granularity,
        "tensors": {
            tensor_name: {
                "scale": tensor.scale,
                "shape": list(tensor.shape) if tensor.shape is not None else None,
                "dtype": tensor.dtype,
            }
            for tensor_name, tensor in metadata.tensors.items()
        },
    }
    output_path.write_text(
        json.dumps(raw_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    load_per_tensor_scale_metadata(output_path)
    return output_path


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
        quantization=quantization,
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


def _build_cifar10_training_dataloader(
    *,
    root: str | PathLike[str],
    batch_size: int,
    download: bool,
    sample_limit: int | None,
    num_workers: int,
) -> Any:
    root_path = Path(root)
    if not download and not root_path.exists():
        msg = (
            "CIFAR-10 training dataset root does not exist: "
            f"{root_path}. Set download=True only when automatic download is intended."
        )
        raise FileNotFoundError(msg)
    torchvision = _require_torchvision()
    torch = _require_pytorch()
    dataset = torchvision.datasets.CIFAR10(
        root=str(root_path),
        train=True,
        transform=_make_cifar10_training_transform(torchvision),
        download=download,
    )
    if sample_limit is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(sample_limit, len(dataset))),
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


def _train_cifar_resnet20_epoch(
    model: CifarResNet20Model,
    dataloader: Any,
    *,
    optimizer: Any,
    torch: Any,
    device: str,
    progress: ProgressReporter | None,
    epoch_index: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    train = getattr(model, "train", None)
    if callable(train):
        train()
    sample_count = 0
    correct = 0
    loss_sum = 0.0
    batch_count = _len_or_none(dataloader)
    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        batch_size = int(targets.shape[0])
        batch_loss = torch.nn.functional.cross_entropy(
            outputs,
            targets,
            reduction="sum",
        )
        loss = batch_loss / batch_size
        loss.backward()
        optimizer.step()
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        sample_count += batch_size
        loss_sum += float(batch_loss.item())
        _report_progress(
            progress,
            "\r"
            + _training_batch_progress_message(
                epoch_index=epoch_index,
                epochs=epochs,
                batch_index=batch_index,
                batch_count=batch_count,
                learning_rate=learning_rate,
                loss=loss_sum / sample_count,
                top1_accuracy=correct / sample_count,
            ),
        )
    if sample_count == 0:
        msg = "CIFAR-10 training requires at least one sample"
        raise ValueError(msg)
    _report_progress(progress, "")
    return {
        "loss": loss_sum / sample_count,
        "top1_accuracy": correct / sample_count,
    }


def _validate_artifact_preparation_settings(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    schedule: tuple[int, ...],
    gammas: tuple[float, ...],
    momentum: float,
    weight_decay: float,
    train_sample_limit: int | None,
    evaluation_sample_limit: int | None,
    num_workers: int,
) -> None:
    if epochs < 0:
        msg = "epochs must be greater than or equal to 0"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be greater than 0"
        raise ValueError(msg)
    if not isfinite(learning_rate) or learning_rate <= 0:
        msg = "learning_rate must be a finite positive number"
        raise ValueError(msg)
    if len(schedule) != len(gammas):
        msg = "schedule and gammas must contain the same number of values"
        raise ValueError(msg)
    previous_milestone = -1
    for milestone in schedule:
        if milestone <= previous_milestone:
            msg = "schedule milestones must be strictly increasing"
            raise ValueError(msg)
        previous_milestone = milestone
    for gamma in gammas:
        if not isfinite(gamma) or gamma <= 0:
            msg = "gammas must be finite positive numbers"
            raise ValueError(msg)
    if not isfinite(momentum) or momentum < 0:
        msg = "momentum must be a finite non-negative number"
        raise ValueError(msg)
    if not isfinite(weight_decay) or weight_decay < 0:
        msg = "weight_decay must be a finite non-negative number"
        raise ValueError(msg)
    if train_sample_limit is not None and train_sample_limit <= 0:
        msg = "train_sample_limit must be greater than 0 when provided"
        raise ValueError(msg)
    if evaluation_sample_limit is not None and evaluation_sample_limit <= 0:
        msg = "evaluation_sample_limit must be greater than 0 when provided"
        raise ValueError(msg)
    if num_workers < 0:
        msg = "num_workers must be greater than or equal to 0"
        raise ValueError(msg)


def _apply_learning_rate_schedule(
    optimizer: Any,
    *,
    base_learning_rate: float,
    epoch_index: int,
    schedule: tuple[int, ...],
    gammas: tuple[float, ...],
) -> float:
    learning_rate = base_learning_rate
    for milestone, gamma in zip(schedule, gammas, strict=True):
        if epoch_index >= milestone:
            learning_rate *= gamma
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def _report_progress(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _report_dataset_settings(
    progress: ProgressReporter | None,
    *,
    root: str | PathLike[str],
    download: bool,
) -> None:
    _report_progress(progress, f"  root: {Path(root)}")
    download_description = _download_progress_description(download)
    _report_progress(progress, f"  download: {download_description}")


def _report_dataloader_settings(
    progress: ProgressReporter | None,
    dataloader: Any,
) -> None:
    sample_count = _len_or_unknown(getattr(dataloader, "dataset", None))
    batch_count = _len_or_unknown(dataloader)
    _report_progress(progress, f"  samples: {sample_count}")
    _report_progress(progress, f"  batches: {batch_count}")


def _download_progress_description(download: bool) -> str:
    if download:
        return "enabled (torchvision may show a 0-100% progress bar)"
    return "disabled"


def _learning_rate_schedule_description(
    schedule: tuple[int, ...],
    gammas: tuple[float, ...],
) -> str:
    if not schedule:
        return "none"
    return ", ".join(
        f"epoch {milestone} x{gamma:g}"
        for milestone, gamma in zip(schedule, gammas, strict=True)
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _training_batch_progress_message(
    *,
    epoch_index: int,
    epochs: int,
    batch_index: int,
    batch_count: int | None,
    learning_rate: float,
    loss: float,
    top1_accuracy: float,
) -> str:
    return (
        f"  epoch {epoch_index + 1:03d}/{epochs:03d} "
        f"{_batch_progress_bar(batch_index, batch_count)} "
        f"batch={_batch_progress_count(batch_index, batch_count)} "
        f"lr={learning_rate:.6g} "
        f"loss={loss:.6g} "
        f"top1={_format_percent(top1_accuracy)}"
    )


def _batch_progress_bar(batch_index: int, batch_count: int | None) -> str:
    if batch_count is None or batch_count <= 0:
        return "[????????????????????]"
    width = 20
    completed = min(width, int(width * batch_index / batch_count))
    remaining = width - completed
    return "[" + ("#" * completed) + ("-" * remaining) + "]"


def _batch_progress_count(batch_index: int, batch_count: int | None) -> str:
    if batch_count is None:
        return f"{batch_index}/?"
    return f"{batch_index}/{batch_count}"


def _len_or_unknown(value: Any) -> str:
    try:
        return str(len(value))
    except TypeError:
        return "unknown"


def _len_or_none(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _cpu_state_dict(model: CifarResNet20Model) -> dict[str, Any]:
    return {
        tensor_name: _cpu_tensor_clone(tensor)
        for tensor_name, tensor in _state_dict_mapping(model).items()
    }


def _cpu_tensor_clone(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        return clone()
    return tensor


def _per_tensor_int8_scale(tensor: Any, *, tensor_name: str, torch: Any) -> float:
    if bool(torch.isnan(tensor).any()) or bool(torch.isinf(tensor).any()):
        msg = f"tensor {tensor_name!r} contains non-finite values"
        raise ValueError(msg)
    max_abs = float(tensor.abs().max().item())
    if max_abs == 0.0:
        return 1.0
    return max_abs / 127.0


def _quantize_tensor_to_int8(tensor: Any, *, scale: float, torch: Any) -> Any:
    scaled = tensor / scale
    rounded = torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    return torch.clamp(rounded, -128, 127).to(torch.int8)


def _make_cifar10_training_transform(torchvision: Any) -> Any:
    transforms = torchvision.transforms
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORMALIZATION_MEAN, CIFAR10_NORMALIZATION_STD),
        ]
    )


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
    quantization: PerTensorScaleMetadata,
    torch: Any,
) -> None:
    quantized_names = set(quantization.tensors)
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
            scale=quantization.scale_for(tensor_name),
            torch=torch,
        )


def _replace_module_state_tensor(
    model: Any,
    tensor_name: str,
    tensor: Any,
    *,
    scale: float | None = None,
    torch: Any,
) -> None:
    module, attribute_name = _resolve_state_tensor_parent(model, tensor_name)
    clone = _clone_detached_tensor(tensor)
    if _is_model_parameter_name(model, tensor_name):
        parameter = torch.nn.Parameter(clone, requires_grad=False)
        setattr(module, attribute_name, parameter)
        if attribute_name == "weight" and _is_signed_int8_dtype(_tensor_dtype(clone)):
            _install_int8_weight_forward(module, scale=scale, torch=torch)
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


def _install_int8_weight_forward(
    module: Any,
    *,
    scale: float | None,
    torch: Any,
) -> None:
    if scale is None:
        msg = "quantized int8 weight forward requires a per-tensor scale"
        raise ValueError(msg)

    nn = getattr(torch, "nn", None)
    conv2d_type = getattr(nn, "Conv2d", ())
    linear_type = getattr(nn, "Linear", ())
    batchnorm2d_type = getattr(nn, "BatchNorm2d", ())
    functional = getattr(nn, "functional", None)
    if functional is None:
        return

    if isinstance(module, conv2d_type):

        def conv2d_forward(inputs: Any) -> Any:
            return functional.conv2d(
                inputs,
                _dequantized_int8_weight(module, scale=scale),
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )

        module.forward = conv2d_forward
        return

    if isinstance(module, linear_type):

        def linear_forward(inputs: Any) -> Any:
            return functional.linear(
                inputs,
                _dequantized_int8_weight(module, scale=scale),
                module.bias,
            )

        module.forward = linear_forward
        return

    if isinstance(module, batchnorm2d_type):

        def batchnorm2d_forward(inputs: Any) -> Any:
            return functional.batch_norm(
                inputs,
                module.running_mean,
                module.running_var,
                _dequantized_int8_weight(module, scale=scale),
                module.bias,
                module.training,
                module.momentum,
                module.eps,
            )

        module.forward = batchnorm2d_forward


def _dequantized_int8_weight(module: Any, *, scale: float) -> Any:
    return module.weight.float() * scale


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
