"""NetFlip package."""

from importlib.metadata import PackageNotFoundError, version

from netflip.benchmarks import (
    CifarResNet20QuantizedArtifact,
    PerTensorScale,
    PerTensorScaleMetadata,
    load_cifar_resnet20_quantized_artifact,
    load_per_tensor_scale_metadata,
)
from netflip.experiment_spec import (
    EXPERIMENT_SPEC_SCHEMA_VERSION,
    BenchmarkModelSpec,
    BfaPbsScenarioSpec,
    CheckpointSpec,
    DatasetSpec,
    ExperimentSpec,
    FaultBudgetSpec,
    QuantizationSpec,
    RandomSoftErrorScenarioSpec,
    load_experiment_spec,
    parse_experiment_spec,
)
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
from netflip.runtime_device import (
    ResolvedTorchDevice,
    RuntimeDeviceUnavailableError,
    TorchDeviceRequest,
    resolve_torch_device,
)
from netflip.soft_error import (
    FAILURE_CRITERION_STOP_REASON,
    FAULT_BUDGET_STOP_REASON,
    MODEL_STATE_BITS_ARTIFACT_KIND,
    SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION,
    SOFT_ERROR_SCENARIO_TYPE,
    UNIFORM_ELIGIBLE_BIT_STRATEGY_NAME,
    EligibleBitPopulation,
    FailureCriterion,
    FaultBudget,
    MetricEvaluator,
    SoftErrorRunResult,
    UniformEligibleBitSelection,
    run_uniform_random_soft_error_baseline,
    sample_uniform_eligible_bit,
)
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
    "EXPERIMENT_SPEC_SCHEMA_VERSION",
    "FAILURE_CRITERION_STOP_REASON",
    "FAULT_BUDGET_STOP_REASON",
    "INT8_BIT_WIDTH",
    "INT8_MAX",
    "INT8_MIN",
    "INT8_MODULUS",
    "INT8_SIGN_BIT_MASK",
    "MODEL_STATE_BITS_ARTIFACT_KIND",
    "OUTPUT_SCHEMA_VERSION",
    "PERTURBATION_TRACE_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION",
    "SOFT_ERROR_SCENARIO_TYPE",
    "UINT8_MAX",
    "UNIFORM_ELIGIBLE_BIT_STRATEGY_NAME",
    "BenchmarkModelSpec",
    "BfaPbsScenarioSpec",
    "BitMetadata",
    "BitRole",
    "CheckpointSpec",
    "CifarResNet20QuantizedArtifact",
    "DatasetSpec",
    "EligibleBitPopulation",
    "ExperimentSpec",
    "FailureCriterion",
    "FaultBudget",
    "FaultBudgetSpec",
    "MetricEvaluator",
    "ModelAdapter",
    "PerTensorScale",
    "PerTensorScaleMetadata",
    "PerturbableTensor",
    "PerturbationTraceEntry",
    "PyTorchModelAdapter",
    "QuantizationSpec",
    "RandomSoftErrorScenarioSpec",
    "ResolvedTorchDevice",
    "RunManifest",
    "RuntimeDeviceUnavailableError",
    "SignedInt8TwoComplementCodec",
    "SoftErrorRunResult",
    "TorchDeviceRequest",
    "UniformEligibleBitSelection",
    "__version__",
    "build_run_manifest",
    "candidate_trace_path",
    "load_cifar_resnet20_quantized_artifact",
    "load_experiment_spec",
    "load_per_tensor_scale_metadata",
    "parse_experiment_spec",
    "resolve_torch_device",
    "run_uniform_random_soft_error_baseline",
    "sample_uniform_eligible_bit",
    "write_perturbation_trace",
    "write_run_manifest",
]
