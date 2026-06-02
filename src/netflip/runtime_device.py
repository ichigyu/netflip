"""Runtime PyTorch device selection."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final, Literal, NoReturn, overload

from beartype import beartype

TorchDeviceRequest = Literal["auto", "cpu", "mps", "cuda"]
ResolvedTorchDevice = Literal["cpu", "mps", "cuda"]
AUTO_DEVICE: Final = "auto"
CPU_DEVICE: Final = "cpu"
CUDA_DEVICE: Final = "cuda"
MPS_DEVICE: Final = "mps"
_EXPLICIT_ACCELERATOR_DEVICES = frozenset({CUDA_DEVICE, MPS_DEVICE})
_TORCH_DEVICE_REQUESTS = frozenset(
    {AUTO_DEVICE, CPU_DEVICE, *_EXPLICIT_ACCELERATOR_DEVICES}
)


class RuntimeDeviceUnavailableError(RuntimeError):
    """Raised when an explicit runtime device request cannot be satisfied."""


@overload
def resolve_torch_device(
    requested: TorchDeviceRequest = AUTO_DEVICE,
) -> ResolvedTorchDevice: ...


@overload
def resolve_torch_device(requested: str) -> ResolvedTorchDevice: ...


@beartype
def resolve_torch_device(requested: str = AUTO_DEVICE) -> ResolvedTorchDevice:
    """Resolve a PyTorch runtime device from actual environment capabilities.

    ``auto`` prefers CUDA, then MPS, and falls back to CPU. Explicit CPU never
    imports PyTorch, which keeps lightweight validation paths available without
    installing benchmark runtime dependencies.
    """
    if requested == CPU_DEVICE:
        return CPU_DEVICE
    if requested not in _TORCH_DEVICE_REQUESTS:
        _raise_unsupported(requested)

    torch = _load_torch_or_none()
    if requested == AUTO_DEVICE:
        return _preferred_torch_device(torch)

    if torch is None:
        _raise_unavailable(requested, "PyTorch is not installed")
    if requested == CUDA_DEVICE:
        if not _cuda_is_available(torch):
            _raise_unavailable(requested, "torch.cuda.is_available() is false")
        return CUDA_DEVICE
    if requested == MPS_DEVICE:
        if not _mps_is_available(torch):
            _raise_unavailable(requested, "torch.backends.mps.is_available() is false")
        return MPS_DEVICE

    _raise_unsupported(requested)


def _load_torch_or_none() -> Any | None:
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        return None


def _preferred_torch_device(torch: Any | None) -> ResolvedTorchDevice:
    if torch is not None and _cuda_is_available(torch):
        return CUDA_DEVICE
    if torch is not None and _mps_is_available(torch):
        return MPS_DEVICE
    return CPU_DEVICE


def _cuda_is_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return callable(is_available) and bool(is_available())


def _mps_is_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return callable(is_available) and bool(is_available())


def _raise_unsupported(requested: str) -> NoReturn:
    msg = (
        f"unsupported PyTorch device request '{requested}'; "
        "choose 'auto', 'cpu', 'cuda', or 'mps'"
    )
    raise ValueError(msg)


def _raise_unavailable(requested: str, reason: str) -> NoReturn:
    msg = (
        f"requested PyTorch device '{requested}' is unavailable: {reason}; "
        "choose 'auto' or 'cpu', or run in an environment with that device"
    )
    raise RuntimeDeviceUnavailableError(msg)
