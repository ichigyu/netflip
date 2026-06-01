from __future__ import annotations

import sys
from typing import Any

import pytest

from netflip import PerturbableTensor, PyTorchModelAdapter


class _FakeContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeTensorValue:
    def __init__(self, value: Any, dtype: str, device: str) -> None:
        self.value = value
        self.dtype = dtype
        self.device = device

    def item(self) -> Any:
        return self.value


class _FakeParameter:
    def __init__(
        self,
        values: Any,
        *,
        dtype: str = "fake.float32",
        device: str = "fake:0",
        requires_grad: bool = True,
    ) -> None:
        self.values = values
        self.shape = _infer_shape(values)
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def numel(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    def detach(self) -> _FakeParameter:
        return self

    def __getitem__(self, index: tuple[int, ...]) -> _FakeTensorValue:
        value = _nested_get(self.values, index)
        if isinstance(value, _FakeTensorValue):
            return value
        return _FakeTensorValue(value, self.dtype, self.device)

    def __setitem__(self, index: tuple[int, ...], value: Any) -> None:
        _nested_set(self.values, index, value)


class _FakePredictions:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]))

    def tolist(self) -> list[list[int]]:
        return self.values


class _FakeScores:
    def __init__(self) -> None:
        self.shape = (2, 2, 3)
        self.requires_grad = False
        self.argmax_dim: int | None = None

    def argmax(self, *, dim: int) -> _FakePredictions:
        self.argmax_dim = dim
        return _FakePredictions([[1, 0], [2, 1]])


class _FakeTorchModule:
    def __init__(self) -> None:
        self.as_tensor_calls: list[dict[str, Any]] = []

    def as_tensor(self, value: Any, *, dtype: str, device: str) -> _FakeTensorValue:
        self.as_tensor_calls.append({"value": value, "dtype": dtype, "device": device})
        return _FakeTensorValue(value, dtype, device)

    def no_grad(self) -> _FakeContext:
        return _FakeContext()

    def inference_mode(self) -> _FakeContext:
        return _FakeContext()


class _FakeModel:
    def __init__(self, output: Any | None = None) -> None:
        self.weight = _FakeParameter([[1.0, 2.0], [3.0, 4.0]])
        self.bias = _FakeParameter([0.25, -0.25])
        self.empty_weight = _FakeParameter([])
        self.output = output if output is not None else _FakeScores()
        self.training = True
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def named_parameters(self) -> list[tuple[str, _FakeParameter]]:
        return [
            ("features.weight", self.weight),
            ("features.bias", self.bias),
            ("empty.weight", self.empty_weight),
        ]

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.output


def _infer_shape(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    return (len(values), *_infer_shape(values[0]))


def _nested_get(values: Any, index: tuple[int, ...]) -> Any:
    current = values
    for coordinate in index:
        current = current[coordinate]
    return current


def _nested_set(values: Any, index: tuple[int, ...], value: Any) -> None:
    current = values
    for coordinate in index[:-1]:
        current = current[coordinate]
    current[index[-1]] = value


@pytest.fixture
def fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeTorchModule:
    fake_module = _FakeTorchModule()
    monkeypatch.setitem(sys.modules, "torch", fake_module)

    import netflip.pytorch_adapter as pytorch_adapter

    monkeypatch.setattr(pytorch_adapter, "_TORCH_MODULE", None)
    return fake_module


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


def test_adapter_exposes_and_mutates_torch_like_weight_tensors(
    fake_torch: _FakeTorchModule,
) -> None:
    model = _FakeModel()
    adapter = PyTorchModelAdapter(model)

    assert adapter.model is model
    assert adapter.perturbable_tensors() == (
        PerturbableTensor(
            name="features.weight",
            shape=(2, 2),
            dtype="fake.float32",
            requires_grad=True,
            numel=4,
            layer_name="features",
        ),
    )
    assert adapter.read_tensor_value("features.weight", (0, 1)) == 2.0

    adapter.write_tensor_value("features.weight", (0, 1), 9.0)

    assert adapter.read_tensor_value("features.weight", (0, 1)) == 9.0
    assert fake_torch.as_tensor_calls == [
        {"value": 9.0, "dtype": "fake.float32", "device": "fake:0"}
    ]


def test_adapter_rejects_invalid_torch_like_model(fake_torch: _FakeTorchModule) -> None:
    with pytest.raises(TypeError, match="missing PyTorch module attributes"):
        PyTorchModelAdapter(object())


def test_adapter_validates_torch_like_tensor_indexes(
    fake_torch: _FakeTorchModule,
) -> None:
    adapter = PyTorchModelAdapter(_FakeModel())

    with pytest.raises(ValueError, match="2 dimensions"):
        adapter.read_tensor_value("features.weight", (0,))
    with pytest.raises(IndexError, match="out of bounds"):
        adapter.read_tensor_value("features.weight", (0, 2))
    with pytest.raises(TypeError, match="integers"):
        adapter.read_tensor_value("features.weight", (0, True))
    with pytest.raises(KeyError, match="unknown perturbable tensor"):
        adapter.read_tensor_value("features.bias", (0,))


def test_inference_uses_context_managers_and_restores_training_state(
    fake_torch: _FakeTorchModule,
) -> None:
    model = _FakeModel()
    adapter = PyTorchModelAdapter(model)

    output = adapter.inference("batch", named="value")

    assert output is model.output
    assert model.training is True
    assert model.calls == [(("batch",), {"named": "value"})]


def test_classify_uses_last_dimension_for_higher_rank_outputs(
    fake_torch: _FakeTorchModule,
) -> None:
    scores = _FakeScores()
    adapter = PyTorchModelAdapter(_FakeModel(output=scores))

    predictions = adapter.classify("batch")

    assert scores.argmax_dim == -1
    assert predictions.shape == (2, 2)
    assert predictions.tolist() == [[1, 0], [2, 1]]


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


def test_classify_reduces_only_the_last_dimension_for_real_pytorch(
    torch_module: Any,
) -> None:
    class TemporalClassifier(torch_module.nn.Module):
        def forward(self, inputs: Any) -> Any:
            return inputs

    adapter = PyTorchModelAdapter(TemporalClassifier())

    predictions = adapter.classify(
        torch_module.tensor(
            [
                [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
                [[0.0, 0.2, 0.8], [0.1, 0.7, 0.2]],
            ]
        )
    )

    assert tuple(predictions.shape) == (2, 2)
    assert predictions.tolist() == [[1, 0], [2, 1]]
