"""Framework-neutral model adapter contracts."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PerturbableTensor:
    """Metadata for a tensor exposed as eligible for bit-flip perturbation."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool
    numel: int
    layer_name: str | None = None


class ModelAdapter(Protocol):
    """Interface NetFlip uses to evaluate a model and perturb Model State Bits."""

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        """Return tensors eligible for bit-flip perturbation."""
        ...

    def read_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
    ) -> Any:
        """Read one scalar value from a Perturbable Tensor."""
        ...

    def write_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
        value: Any,
    ) -> None:
        """Write one scalar value in a Perturbable Tensor."""
        ...

    def evaluation_mode(self) -> AbstractContextManager[None]:
        """Temporarily put the model in evaluation mode."""
        ...

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the model without recording gradients."""
        ...

    def classify(self, *args: Any, **kwargs: Any) -> Any:
        """Return class predictions from model scores."""
        ...
