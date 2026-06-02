from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from netflip.benchmarks import (
    CIFAR10_CLASSES,
    CIFAR10_NORMALIZATION_MEAN,
    CIFAR10_NORMALIZATION_STD,
    CIFAR_RESNET20_BENCHMARK_ID,
    Cifar10DatasetRequest,
    Cifar10DatasetRole,
    ResNet20Config,
    build_cifar10_dataloader,
    build_cifar10_dataloaders,
    build_cifar10_dataset,
    build_cifar_resnet20,
    compute_cross_entropy_loss,
    compute_top1_accuracy,
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


def test_build_cifar_resnet20_constructs_model_with_lazy_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    fake_torch = object()
    built_configs: list[ResNet20Config] = []

    def fake_make_cifar_resnet(config: ResNet20Config, torch: Any) -> str:
        assert torch is fake_torch
        built_configs.append(config)
        return "fake-model"

    monkeypatch.setattr(cifar_resnet20, "_require_pytorch", lambda: fake_torch)
    monkeypatch.setattr(cifar_resnet20, "_make_cifar_resnet", fake_make_cifar_resnet)

    model = build_cifar_resnet20(num_classes=7, input_channels=1)

    assert model == "fake-model"
    assert built_configs == [
        ResNet20Config(
            input_channels=1,
            num_classes=7,
        )
    ]


def test_build_cifar_resnet20_forward_pass_smoke() -> None:
    torch = pytest.importorskip("torch")

    model = build_cifar_resnet20()
    outputs = model(torch.zeros(2, 3, 32, 32))

    assert model.benchmark_id == CIFAR_RESNET20_BENCHMARK_ID
    assert model.config.depth == 20
    assert tuple(outputs.shape) == (2, CIFAR10_CLASSES)


def test_build_cifar10_dataset_requires_torchvision_when_missing(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    def missing_import(name: str) -> Any:
        if name == "torchvision":
            raise ModuleNotFoundError(name)
        return __import__(name)

    monkeypatch.setattr(cifar_resnet20, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match="benchmark extra"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.EVALUATION,
                root=tmp_path,
                split="test",
            )
        )


def test_build_cifar10_dataset_missing_root_reports_clear_error(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match="selection dataset root"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.SELECTION,
                root=tmp_path / "missing-cifar10",
                split="train",
            )
        )


def test_build_cifar10_dataset_rejects_unknown_role(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="dataset role"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role="calibration",
                root=tmp_path,
                split="train",
            )
        )


def test_build_cifar10_dataset_rejects_invalid_split(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="split must be one"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.SELECTION,
                root=tmp_path,
                split="validation",  # type: ignore[arg-type]
            )
        )


def test_build_cifar10_dataset_rejects_negative_sample_limit(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="selection sample_limit"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.SELECTION,
                root=tmp_path,
                split="train",
                sample_limit=-1,
            )
        )


def test_build_cifar10_dataset_without_sample_limit_returns_raw_dataset(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    class FakeCifar10Dataset:
        pass

    fake_dataset = FakeCifar10Dataset()

    def fake_cifar10(**kwargs: Any) -> FakeCifar10Dataset:
        return fake_dataset

    fake_torchvision = SimpleNamespace(
        datasets=SimpleNamespace(CIFAR10=fake_cifar10),
        transforms=_fake_cifar10_transforms(),
    )
    monkeypatch.setattr(
        cifar_resnet20, "_require_torchvision", lambda: fake_torchvision
    )

    dataset = build_cifar10_dataset(
        Cifar10DatasetRequest(
            role=Cifar10DatasetRole.EVALUATION,
            root=tmp_path,
            split="test",
        )
    )

    assert dataset is fake_dataset


def test_build_cifar10_dataloaders_use_splits_and_sample_limits(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    calls: list[dict[str, Any]] = []

    class FakeCifar10Dataset:
        def __init__(
            self,
            *,
            root: str,
            train: bool,
            transform: Any,
            download: bool,
        ) -> None:
            calls.append(
                {
                    "root": root,
                    "train": train,
                    "transform": transform,
                    "download": download,
                }
            )
            self.train = train

        def __len__(self) -> int:
            return 5 if self.train else 4

    class FakeNormalize:
        def __init__(self, mean: Any, std: Any) -> None:
            self.mean = mean
            self.std = std

    class FakeTransforms:
        @staticmethod
        def Compose(transforms: list[Any]) -> Any:
            return transforms

        @staticmethod
        def ToTensor() -> str:
            return "to-tensor"

        Normalize = FakeNormalize

    class FakeSubset:
        def __init__(self, dataset: Any, indices: range) -> None:
            self.dataset = dataset
            self.indices = tuple(indices)

        def __len__(self) -> int:
            return len(self.indices)

    class FakeDataLoader:
        def __init__(
            self,
            dataset: Any,
            *,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool,
        ) -> None:
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory

    fake_torchvision = SimpleNamespace(
        datasets=SimpleNamespace(CIFAR10=FakeCifar10Dataset),
        transforms=FakeTransforms,
    )
    fake_torch = SimpleNamespace(
        utils=SimpleNamespace(
            data=SimpleNamespace(
                DataLoader=FakeDataLoader,
                Subset=FakeSubset,
            )
        )
    )
    monkeypatch.setattr(
        cifar_resnet20, "_require_torchvision", lambda: fake_torchvision
    )
    monkeypatch.setattr(cifar_resnet20, "_require_pytorch", lambda: fake_torch)

    dataloaders = build_cifar10_dataloaders(
        root=tmp_path,
        selection_split="train",
        evaluation_split="test",
        batch_size=2,
        selection_sample_limit=3,
        evaluation_sample_limit=2,
    )

    assert len(dataloaders.selection.dataset) == 3
    assert len(dataloaders.evaluation.dataset) == 2
    assert dataloaders.selection.batch_size == 2
    assert dataloaders.evaluation.shuffle is False
    assert [call["train"] for call in calls] == [True, False]
    assert calls[0]["root"] == str(tmp_path)
    assert calls[0]["download"] is False
    normalize = calls[0]["transform"][1]
    assert normalize.mean == CIFAR10_NORMALIZATION_MEAN
    assert normalize.std == CIFAR10_NORMALIZATION_STD


def test_cifar10_evaluation_transform_uses_standard_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    fake_torchvision = SimpleNamespace(transforms=_fake_cifar10_transforms())
    monkeypatch.setattr(
        cifar_resnet20, "_require_torchvision", lambda: fake_torchvision
    )

    transform = cifar_resnet20.cifar10_evaluation_transform()

    normalize = transform[1]
    assert normalize.mean == CIFAR10_NORMALIZATION_MEAN
    assert normalize.std == CIFAR10_NORMALIZATION_STD


def test_build_cifar10_dataloader_rejects_invalid_batch_size(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build_cifar10_dataloader(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.EVALUATION,
                root=tmp_path,
                split="test",
            ),
            batch_size=0,
        )


def test_build_cifar10_dataset_wraps_only_missing_dataset_runtime_errors(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    class MissingCifar10Dataset:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("Dataset not found or corrupted.")

    fake_torchvision = SimpleNamespace(
        datasets=SimpleNamespace(CIFAR10=MissingCifar10Dataset),
        transforms=_fake_cifar10_transforms(),
    )
    monkeypatch.setattr(
        cifar_resnet20, "_require_torchvision", lambda: fake_torchvision
    )

    with pytest.raises(FileNotFoundError, match="evaluation test split"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.EVALUATION,
                root=tmp_path,
                split="test",
            )
        )


def test_build_cifar10_dataset_preserves_unexpected_runtime_errors(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    class BrokenCifar10Dataset:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("checksum mismatch while reading archive")

    fake_torchvision = SimpleNamespace(
        datasets=SimpleNamespace(CIFAR10=BrokenCifar10Dataset),
        transforms=_fake_cifar10_transforms(),
    )
    monkeypatch.setattr(
        cifar_resnet20, "_require_torchvision", lambda: fake_torchvision
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_cifar10_dataset(
            Cifar10DatasetRequest(
                role=Cifar10DatasetRole.EVALUATION,
                root=tmp_path,
                split="test",
            )
        )


def test_metric_wrappers_use_pytorch_runtime_without_real_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.benchmarks.cifar_resnet20 as cifar_resnet20

    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeComparison:
        def __init__(self, correct: int) -> None:
            self.correct = correct

        def sum(self) -> FakeScalar:
            return FakeScalar(self.correct)

    class FakePredictions:
        def __init__(self, correct: int) -> None:
            self.correct = correct

        def __eq__(self, other: object) -> Any:
            return FakeComparison(self.correct)

    class FakeOutputs:
        def __init__(self, correct: int, loss: float) -> None:
            self.correct = correct
            self.loss = loss

        def argmax(self, *, dim: int) -> FakePredictions:
            assert dim == 1
            return FakePredictions(self.correct)

    class FakeInputs:
        def __init__(self, outputs: FakeOutputs) -> None:
            self.outputs = outputs
            self.device = None

        def to(self, device: str) -> FakeInputs:
            self.device = device
            return self

    class FakeTargets:
        shape = (2,)

        def __init__(self) -> None:
            self.device = None

        def to(self, device: str) -> FakeTargets:
            self.device = device
            return self

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    class FakeFunctional:
        @staticmethod
        def cross_entropy(
            outputs: FakeOutputs,
            targets: FakeTargets,
            *,
            reduction: str,
        ) -> FakeScalar:
            assert reduction == "sum"
            return FakeScalar(outputs.loss)

    class FakeModel:
        training = True

        def __init__(self) -> None:
            self.device = None

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def __call__(self, inputs: FakeInputs) -> FakeOutputs:
            return inputs.outputs

    fake_torch = SimpleNamespace(
        no_grad=FakeNoGrad,
        nn=SimpleNamespace(functional=FakeFunctional),
    )
    monkeypatch.setattr(cifar_resnet20, "_require_pytorch", lambda: fake_torch)

    dataloader = [
        (FakeInputs(FakeOutputs(correct=2, loss=1.0)), FakeTargets()),
        (FakeInputs(FakeOutputs(correct=1, loss=3.0)), FakeTargets()),
    ]
    model = FakeModel()

    assert compute_top1_accuracy(model, dataloader, device="cpu") == pytest.approx(0.75)
    assert model.training is True
    assert model.device == "cpu"
    assert compute_cross_entropy_loss(model, dataloader) == pytest.approx(1.0)


def test_classification_metric_helpers_compute_accuracy_and_cross_entropy() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 5.0],
            [2.0, 3.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 2, 2, 1])
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(logits, targets),
        batch_size=2,
    )
    model = torch.nn.Identity()

    accuracy = compute_top1_accuracy(model, dataloader)
    cross_entropy = compute_cross_entropy_loss(model, dataloader)

    expected_loss = torch.nn.functional.cross_entropy(logits, targets).item()
    assert accuracy == pytest.approx(0.75)
    assert cross_entropy == pytest.approx(expected_loss)
    assert model.training is True


def test_classification_metrics_reject_empty_dataloader() -> None:
    torch = pytest.importorskip("torch")

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.empty(0, 3),
            torch.empty(0, dtype=torch.long),
        ),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="at least one sample"):
        compute_top1_accuracy(torch.nn.Identity(), dataloader)


def _fake_cifar10_transforms() -> Any:
    class FakeNormalize:
        def __init__(self, mean: Any, std: Any) -> None:
            self.mean = mean
            self.std = std

    class FakeTransforms:
        @staticmethod
        def Compose(transforms: list[Any]) -> Any:
            return transforms

        @staticmethod
        def ToTensor() -> str:
            return "to-tensor"

        Normalize = FakeNormalize

    return FakeTransforms
