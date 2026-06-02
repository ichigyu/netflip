"""Runtime PyTorch device selection."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

from beartype import beartype

TorchDeviceRequest = Literal["auto", "cpu", "mps", "cuda"]
ResolvedTorchDevice = Literal["cpu", "mps", "cuda"]


class RuntimeDeviceUnavailableError(RuntimeError):
    """Raised when an explicit runtime device request cannot be satisfied."""


@beartype
def resolve_torch_device(
    requested: TorchDeviceRequest = "auto",
) -> ResolvedTorchDevice:
    """Resolve a PyTorch runtime device from actual environment capabilities.

    ``auto`` prefers CUDA, then MPS, and falls back to CPU. Explicit CPU never
    imports PyTorch, which keeps lightweight validation paths available without
    installing benchmark runtime dependencies.
    """
    if requested == "cpu":
        return "cpu"

    torch = _load_torch_or_none()
    if requested == "auto":
        if torch is not None and _cuda_is_available(torch):
            return "cuda"
        if torch is not None and _mps_is_available(torch):
            return "mps"
        return "cpu"

    if torch is None:
        _raise_unavailable(requested, "PyTorch is not installed")
    if requested == "cuda" and not _cuda_is_available(torch):
        _raise_unavailable(requested, "torch.cuda.is_available() is false")
    if requested == "mps" and not _mps_is_available(torch):
        _raise_unavailable(requested, "torch.backends.mps.is_available() is false")
    return requested


def _load_torch_or_none() -> Any | None:
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        return None


def _cuda_is_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return callable(is_available) and bool(is_available())


def _mps_is_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return callable(is_available) and bool(is_available())


def _raise_unavailable(requested: str, reason: str) -> None:
    msg = (
        f"requested PyTorch device '{requested}' is unavailable: {reason}; "
        "choose 'auto' or 'cpu', or run in an environment with that device"
    )
    raise RuntimeDeviceUnavailableError(msg)
