"""Uniform random soft-error baseline over signed int8 model state bits."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral
from random import Random
from typing import TypeAlias

from netflip.int8_codec import INT8_BIT_WIDTH, SignedInt8TwoComplementCodec
from netflip.manifest import JSONScalar
from netflip.model_adapter import ModelAdapter, PerturbableTensor
from netflip.trace import PerturbationTraceEntry

SOFT_ERROR_SCENARIO_TYPE = "soft_error"
UNIFORM_ELIGIBLE_BIT_STRATEGY_NAME = "uniform-eligible-bit"
MODEL_STATE_BITS_ARTIFACT_KIND = "model_state_bits"
SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION = "signed-int8-two-complement"
FAULT_BUDGET_STOP_REASON = "fault_budget"
FAILURE_CRITERION_STOP_REASON = "failure_criterion"

MetricEvaluator: TypeAlias = Callable[[ModelAdapter], Mapping[str, JSONScalar]]
FailureCriterion: TypeAlias = Callable[[PerturbationTraceEntry], bool]


@dataclass(frozen=True)
class FaultBudget:
    """Maximum perturbation allowed for one soft-error Run."""

    max_flip_count: int | None = None
    max_bit_flip_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.max_flip_count is None and self.max_bit_flip_ratio is None:
            msg = "max_flip_count or max_bit_flip_ratio is required"
            raise ValueError(msg)
        if self.max_flip_count is not None:
            _validate_non_negative_int(self.max_flip_count, "max_flip_count")
        if self.max_bit_flip_ratio is not None:
            ratio = self.max_bit_flip_ratio
            if not isfinite(ratio) or ratio < 0 or ratio > 1:
                msg = "max_bit_flip_ratio must be a finite value in range [0, 1]"
                raise ValueError(msg)

    def max_steps(self, eligible_bit_population: int) -> int:
        """Return the maximum one-bit perturbation steps this budget permits."""
        _validate_positive_int(
            eligible_bit_population,
            "eligible_bit_population",
        )
        candidates: list[int] = []
        if self.max_flip_count is not None:
            candidates.append(self.max_flip_count)
        if self.max_bit_flip_ratio is not None:
            candidates.append(floor(self.max_bit_flip_ratio * eligible_bit_population))
        return min(min(candidates), eligible_bit_population)


@dataclass(frozen=True)
class UniformEligibleBitSelection:
    """One uniformly sampled bit from the Eligible Bit Population."""

    population_ordinal: int
    tensor_name: str
    tensor_index: tuple[int, ...]
    layer_name: str | None
    bit_index: int
    bit_role: str


@dataclass(frozen=True)
class SoftErrorRunResult:
    """Trace and stop metadata for one random soft-error Run."""

    perturbation_trace: tuple[PerturbationTraceEntry, ...]
    stopped_because: str
    eligible_bit_population: int

    @property
    def flip_count(self) -> int:
        """Number of committed bit flips in this Run."""
        return len(self.perturbation_trace)

    @property
    def bit_flip_ratio(self) -> float:
        """Committed Flip Count divided by the Eligible Bit Population size."""
        if self.eligible_bit_population == 0:
            return 0
        return self.flip_count / self.eligible_bit_population


@dataclass(frozen=True)
class _EligibleTensorBits:
    tensor: PerturbableTensor
    start_bit_offset: int
    bit_count: int

    @property
    def stop_bit_offset(self) -> int:
        return self.start_bit_offset + self.bit_count


class EligibleBitPopulation:
    """Eligible int8 model state bits addressable by population ordinal."""

    def __init__(
        self,
        tensors: Sequence[_EligibleTensorBits],
        *,
        total_bits: int,
    ) -> None:
        if not tensors:
            msg = "at least one eligible int8 tensor is required"
            raise ValueError(msg)
        _validate_positive_int(total_bits, "total_bits")
        self._tensors = tuple(tensors)
        self._stop_offsets = tuple(tensor.stop_bit_offset for tensor in tensors)
        self._total_bits = total_bits

    @classmethod
    def from_model_adapter(cls, adapter: ModelAdapter) -> EligibleBitPopulation:
        """Build the Eligible Bit Population from int8 perturbable tensors."""
        tensors: list[_EligibleTensorBits] = []
        next_offset = 0
        for tensor in adapter.perturbable_tensors():
            if not _is_signed_int8_dtype(tensor.dtype):
                continue
            bit_count = tensor.numel * INT8_BIT_WIDTH
            if bit_count <= 0:
                continue
            tensors.append(
                _EligibleTensorBits(
                    tensor=tensor,
                    start_bit_offset=next_offset,
                    bit_count=bit_count,
                )
            )
            next_offset += bit_count
        return cls(tensors, total_bits=next_offset)

    @property
    def size(self) -> int:
        """Number of bits in this Eligible Bit Population."""
        return self._total_bits

    def selection_from_ordinal(
        self,
        population_ordinal: int,
        *,
        codec: SignedInt8TwoComplementCodec | None = None,
    ) -> UniformEligibleBitSelection:
        """Map a population ordinal to a tensor index and bit index."""
        ordinal = _validate_population_ordinal(population_ordinal, self.size)
        tensor_position = bisect_right(self._stop_offsets, ordinal)
        eligible_tensor = self._tensors[tensor_position]
        tensor_bit_offset = ordinal - eligible_tensor.start_bit_offset
        value_ordinal = tensor_bit_offset // INT8_BIT_WIDTH
        bit_index = tensor_bit_offset % INT8_BIT_WIDTH
        selected_codec = codec if codec is not None else SignedInt8TwoComplementCodec()
        bit_metadata = selected_codec.bit_metadata(bit_index)
        return UniformEligibleBitSelection(
            population_ordinal=ordinal,
            tensor_name=eligible_tensor.tensor.name,
            tensor_index=_unravel_index(value_ordinal, eligible_tensor.tensor.shape),
            layer_name=eligible_tensor.tensor.layer_name,
            bit_index=bit_index,
            bit_role=bit_metadata.role.value,
        )


def sample_uniform_eligible_bit(
    population: EligibleBitPopulation,
    rng: Random,
    *,
    excluded_ordinals: set[int] | frozenset[int] | None = None,
    codec: SignedInt8TwoComplementCodec | None = None,
) -> UniformEligibleBitSelection:
    """Sample one unexcluded bit uniformly from the Eligible Bit Population."""
    excluded = excluded_ordinals if excluded_ordinals is not None else frozenset()
    if len(excluded) >= population.size:
        msg = "no uncommitted eligible bits remain"
        raise ValueError(msg)

    while True:
        population_ordinal = rng.randrange(population.size)
        if population_ordinal not in excluded:
            return population.selection_from_ordinal(
                population_ordinal,
                codec=codec,
            )


def run_uniform_random_soft_error_baseline(
    *,
    adapter: ModelAdapter,
    metric_evaluator: MetricEvaluator,
    fault_budget: FaultBudget,
    rng_seed: int,
    failure_criterion: FailureCriterion | None = None,
    codec: SignedInt8TwoComplementCodec | None = None,
) -> SoftErrorRunResult:
    """Run a cumulative random soft-error baseline with one bit per step."""
    seed = _validate_int(rng_seed, "rng_seed")
    selected_codec = codec if codec is not None else SignedInt8TwoComplementCodec()
    population = EligibleBitPopulation.from_model_adapter(adapter)
    max_steps = fault_budget.max_steps(population.size)
    if max_steps == 0:
        return SoftErrorRunResult(
            perturbation_trace=(),
            stopped_because=FAULT_BUDGET_STOP_REASON,
            eligible_bit_population=population.size,
        )

    rng = Random(seed)
    committed_ordinals: set[int] = set()
    entries: list[PerturbationTraceEntry] = []
    metric_before = _evaluate_metric(metric_evaluator, adapter)

    for step_index in range(max_steps):
        selection = sample_uniform_eligible_bit(
            population,
            rng,
            excluded_ordinals=committed_ordinals,
            codec=selected_codec,
        )
        committed_ordinals.add(selection.population_ordinal)

        value_before = adapter.read_tensor_value(
            selection.tensor_name,
            selection.tensor_index,
        )
        value_after = selected_codec.flip_value_bit(value_before, selection.bit_index)
        adapter.write_tensor_value(
            selection.tensor_name,
            selection.tensor_index,
            value_after,
        )
        metric_after = _evaluate_metric(metric_evaluator, adapter)
        flip_count = step_index + 1
        entry = PerturbationTraceEntry(
            step_index=step_index,
            scenario_type=SOFT_ERROR_SCENARIO_TYPE,
            strategy_name=UNIFORM_ELIGIBLE_BIT_STRATEGY_NAME,
            artifact_kind=MODEL_STATE_BITS_ARTIFACT_KIND,
            tensor_name=selection.tensor_name,
            tensor_index=selection.tensor_index,
            representation=SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION,
            bit_index=selection.bit_index,
            bit_role=selection.bit_role,
            value_before=value_before,
            value_after=value_after,
            flip_count=flip_count,
            bit_flip_ratio=flip_count / population.size,
            metric_before=metric_before,
            metric_after=metric_after,
            selection_score=None,
            rng_seed=seed,
            layer_name=selection.layer_name,
            eligible_bit_population=population.size,
        )
        entries.append(entry)
        if failure_criterion is not None and failure_criterion(entry):
            return SoftErrorRunResult(
                perturbation_trace=tuple(entries),
                stopped_because=FAILURE_CRITERION_STOP_REASON,
                eligible_bit_population=population.size,
            )
        metric_before = metric_after

    return SoftErrorRunResult(
        perturbation_trace=tuple(entries),
        stopped_because=FAULT_BUDGET_STOP_REASON,
        eligible_bit_population=population.size,
    )


def _evaluate_metric(
    metric_evaluator: MetricEvaluator,
    adapter: ModelAdapter,
) -> dict[str, JSONScalar]:
    return dict(metric_evaluator(adapter))


def _is_signed_int8_dtype(dtype: str) -> bool:
    return dtype.rsplit(".", maxsplit=1)[-1] == "int8"


def _unravel_index(value_ordinal: int, shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        return ()

    remaining = _validate_non_negative_int(value_ordinal, "value_ordinal")
    coordinates_reversed: list[int] = []
    for dimension in reversed(shape):
        dimension_int = _validate_positive_int(dimension, "shape dimension")
        coordinates_reversed.append(remaining % dimension_int)
        remaining //= dimension_int
    if remaining != 0:
        msg = "value_ordinal is out of bounds for tensor shape"
        raise IndexError(msg)
    return tuple(reversed(coordinates_reversed))


def _validate_population_ordinal(population_ordinal: int, population_size: int) -> int:
    ordinal = _validate_non_negative_int(population_ordinal, "population_ordinal")
    size = _validate_positive_int(population_size, "population_size")
    if ordinal >= size:
        msg = "population_ordinal is out of bounds"
        raise IndexError(msg)
    return ordinal


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    return int(value)


def _validate_non_negative_int(value: int, name: str) -> int:
    integer = _validate_int(value, name)
    if integer < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    return integer


def _validate_positive_int(value: int, name: str) -> int:
    integer = _validate_int(value, name)
    if integer <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return integer
