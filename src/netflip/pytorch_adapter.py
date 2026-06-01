"""PyTorch Model Adapter implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from netflip.model_adapter import PerturbableTensor


class PyTorchModelAdapter:
    """Wrap a PyTorch model and expose perturbable weight tensors."""

    def __init__(self, model: Any) -> None:
        _require_pytorch()
        _validate_pytorch_model(model)
        self._model = model

    @property
    def model(self) -> Any:
        """Return the wrapped PyTorch model."""
        return self._model

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        """Enumerate named weight parameters eligible for perturbation."""
        tensors: list[PerturbableTensor] = []
        for name, parameter in self._named_parameters().items():
            if not _is_weight_parameter_name(name) or parameter.numel() == 0:
                continue
            tensors.append(
                PerturbableTensor(
                    name=name,
                    shape=tuple(int(dimension) for dimension in parameter.shape),
                    dtype=str(parameter.dtype),
                    requires_grad=bool(parameter.requires_grad),
                    numel=int(parameter.numel()),
                    layer_name=_layer_name(name),
                )
            )
        return tuple(tensors)

    def read_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
    ) -> Any:
        """Read one scalar value from a Perturbable Tensor."""
        parameter = self._parameter(tensor_name)
        index = _validate_tensor_index(tensor_index, parameter.shape)
        return parameter.detach()[index].item()

    def write_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
        value: Any,
    ) -> None:
        """Write one scalar value in a Perturbable Tensor."""
        torch = _require_pytorch()
        parameter = self._parameter(tensor_name)
        index = _validate_tensor_index(tensor_index, parameter.shape)
        with torch.no_grad():
            parameter[index] = value

    @contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        """Temporarily put the model in evaluation mode and restore training state."""
        was_training = bool(self._model.training)
        self._model.eval()
        try:
            yield
        finally:
            self._model.train(was_training)

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the wrapped model in evaluation mode without gradients."""
        torch = _require_pytorch()
        with self.evaluation_mode(), torch.inference_mode():
            return self._model(*args, **kwargs)

    def classify(self, *args: Any, **kwargs: Any) -> Any:
        """Return class predictions from model scores using the last dimension."""
        scores = self.inference(*args, **kwargs)
        return scores.argmax(dim=-1)

    def _named_parameters(self) -> dict[str, Any]:
        return dict(self._model.named_parameters())

    def _parameter(self, tensor_name: str) -> Any:
        parameter = self._named_parameters().get(tensor_name)
        if parameter is None or not _is_weight_parameter_name(tensor_name):
            msg = f"unknown perturbable tensor: {tensor_name}"
            raise KeyError(msg)
        return parameter


def _require_pytorch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = "PyTorchModelAdapter requires PyTorch to be installed"
        raise ModuleNotFoundError(msg) from exc


def _validate_pytorch_model(model: Any) -> None:
    required_attributes = ("named_parameters", "eval", "train", "__call__")
    missing = [
        attribute for attribute in required_attributes if not hasattr(model, attribute)
    ]
    if missing:
        msg = f"model is missing PyTorch module attributes: {', '.join(missing)}"
        raise TypeError(msg)


def _is_weight_parameter_name(name: str) -> bool:
    return name == "weight" or name.endswith(".weight")


def _layer_name(tensor_name: str) -> str | None:
    if "." not in tensor_name:
        return None
    return tensor_name.rsplit(".", maxsplit=1)[0]


def _validate_tensor_index(
    tensor_index: Sequence[int],
    shape: Sequence[int],
) -> tuple[int, ...]:
    index = tuple(tensor_index)
    if len(index) != len(shape):
        msg = f"tensor_index must have {len(shape)} dimensions"
        raise ValueError(msg)

    for axis, (coordinate, dimension) in enumerate(zip(index, shape, strict=True)):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            msg = "tensor_index coordinates must be integers"
            raise TypeError(msg)
        if coordinate < 0 or coordinate >= int(dimension):
            msg = (
                f"tensor_index coordinate {coordinate} is out of bounds "
                f"for axis {axis} with size {int(dimension)}"
            )
            raise IndexError(msg)
    return index
