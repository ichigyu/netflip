from __future__ import annotations

from typing import Any

import pytest

from netflip import PerturbableTensor, PyTorchModelAdapter


@pytest.fixture
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture
def tiny_classifier(torch_module: Any) -> type[Any]:
    class TinyClassifier(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch_module.nn.Linear(2, 2)
            self.register_buffer("running_count", torch_module.ones(1))

        def forward(self, inputs: Any) -> Any:
            return self.features(inputs)

    return TinyClassifier


def test_adapter_can_wrap_a_small_pytorch_module(tiny_classifier: type[Any]) -> None:
    model = tiny_classifier()

    adapter = PyTorchModelAdapter(model)

    assert adapter.model is model


def test_adapter_enumerates_eligible_weight_tensors_by_name(
    tiny_classifier: type[Any],
) -> None:
    adapter = PyTorchModelAdapter(tiny_classifier())

    tensors = adapter.perturbable_tensors()

    assert tensors == (
        PerturbableTensor(
            name="features.weight",
            shape=(2, 2),
            dtype="torch.float32",
            requires_grad=True,
            numel=4,
            layer_name="features",
        ),
    )


def test_adapter_reads_and_writes_perturbable_tensor_values(
    tiny_classifier: type[Any],
) -> None:
    adapter = PyTorchModelAdapter(tiny_classifier())

    before = adapter.read_tensor_value("features.weight", (0, 1))
    adapter.write_tensor_value("features.weight", (0, 1), before + 1.0)

    assert adapter.read_tensor_value("features.weight", (0, 1)) == pytest.approx(
        before + 1.0
    )


def test_adapter_rejects_non_weight_parameters_as_perturbable_tensors(
    tiny_classifier: type[Any],
) -> None:
    adapter = PyTorchModelAdapter(tiny_classifier())

    with pytest.raises(KeyError, match="unknown perturbable tensor"):
        adapter.read_tensor_value("features.bias", (0,))


def test_adapter_validates_tensor_indexes(tiny_classifier: type[Any]) -> None:
    adapter = PyTorchModelAdapter(tiny_classifier())

    with pytest.raises(ValueError, match="2 dimensions"):
        adapter.read_tensor_value("features.weight", (0,))
    with pytest.raises(IndexError, match="out of bounds"):
        adapter.read_tensor_value("features.weight", (0, 2))
    with pytest.raises(TypeError, match="integers"):
        adapter.read_tensor_value("features.weight", (0, True))


def test_inference_runs_in_evaluation_mode_and_restores_training_state(
    torch_module: Any,
    tiny_classifier: type[Any],
) -> None:
    model = tiny_classifier()
    model.train(True)
    adapter = PyTorchModelAdapter(model)

    outputs = adapter.inference(torch_module.tensor([[1.0, 2.0]]))

    assert outputs.shape == (1, 2)
    assert model.training is True
    assert outputs.requires_grad is False


def test_classify_returns_argmax_predictions(
    torch_module: Any,
    tiny_classifier: type[Any],
) -> None:
    model = tiny_classifier()
    with torch_module.no_grad():
        model.features.weight.copy_(torch_module.tensor([[1.0, 0.0], [0.0, 2.0]]))
        model.features.bias.zero_()
    adapter = PyTorchModelAdapter(model)

    predictions = adapter.classify(torch_module.tensor([[3.0, 1.0], [0.0, 2.0]]))

    assert predictions.tolist() == [0, 1]
