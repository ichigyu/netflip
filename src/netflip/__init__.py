"""NetFlip package."""

from importlib.metadata import PackageNotFoundError, version

from netflip.int8_codec import (
    INT8_BIT_WIDTH,
    INT8_MAX,
    INT8_MIN,
    INT8_MODULUS,
    INT8_SIGN_BIT_MASK,
    UINT8_MAX,
    BitMetadata,
    BitRole,
    SignedInt8TwoComplementCodec,
)
from netflip.manifest import (
    OUTPUT_SCHEMA_VERSION,
    RUN_MANIFEST_FILENAME,
    RunManifest,
    build_run_manifest,
    write_run_manifest,
)
from netflip.model_adapter import ModelAdapter, PerturbableTensor
from netflip.pytorch_adapter import PyTorchModelAdapter
from netflip.trace import (
    CANDIDATE_TRACE_FILENAME,
    PERTURBATION_TRACE_FILENAME,
    PerturbationTraceEntry,
    candidate_trace_path,
    write_perturbation_trace,
)

try:
    __version__ = version("netflip")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CANDIDATE_TRACE_FILENAME",
    "INT8_BIT_WIDTH",
    "INT8_MAX",
    "INT8_MIN",
    "INT8_MODULUS",
    "INT8_SIGN_BIT_MASK",
    "OUTPUT_SCHEMA_VERSION",
    "PERTURBATION_TRACE_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "UINT8_MAX",
    "BitMetadata",
    "BitRole",
    "ModelAdapter",
    "PerturbableTensor",
    "PerturbationTraceEntry",
    "PyTorchModelAdapter",
    "RunManifest",
    "SignedInt8TwoComplementCodec",
    "__version__",
    "build_run_manifest",
    "candidate_trace_path",
    "write_perturbation_trace",
    "write_run_manifest",
]
