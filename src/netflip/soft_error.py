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
from netflip.progress import (
    ProgressReporter,
    format_progress_metrics,
    report_progress,
)
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

    def ordinal_from_selection(
        self,
        *,
        tensor_name: str,
        tensor_index: Sequence[int],
        bit_index: int,
    ) -> int:
        """Map a tensor index and bit index back to a population ordinal."""
        for eligible_tensor in self._tensors:
            if eligible_tensor.tensor.name != tensor_name:
                continue
            value_ordinal = _ravel_index(
                tensor_index,
                eligible_tensor.tensor.shape,
            )
            if not 0 <= bit_index < INT8_BIT_WIDTH:
                msg = f"bit_index must be in [0, {INT8_BIT_WIDTH}), got {bit_index}"
                raise ValueError(msg)
            return (
                eligible_tensor.start_bit_offset
                + value_ordinal * INT8_BIT_WIDTH
                + bit_index
            )
        msg = f"unknown eligible tensor: {tensor_name}"
        raise KeyError(msg)


def sample_uniform_eligible_bit(
    population: EligibleBitPopulation,
    rng: Random,
    *,
    excluded_ordinals: set[int] | frozenset[int] | None = None,
    codec: SignedInt8TwoComplementCodec | None = None,
) -> UniformEligibleBitSelection:
    """Sample one unexcluded bit uniformly from the Eligible Bit Population."""
    excluded = excluded_ordinals if excluded_ordinals is not None else frozenset()
    _validate_excluded_ordinals(excluded, population.size)
    remaining_count = population.size - len(excluded)
    if remaining_count <= 0:
        msg = "no uncommitted eligible bits remain"
        raise ValueError(msg)

    population_ordinal = _population_ordinal_from_remaining_offset(
        rng.randrange(remaining_count),
        excluded,
    )
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
    progress: ProgressReporter | None = None,
) -> SoftErrorRunResult:
    """Run a cumulative random soft-error baseline with one bit per step."""
    seed = _validate_int(rng_seed, "rng_seed")
    selected_codec = codec if codec is not None else SignedInt8TwoComplementCodec()
    population = EligibleBitPopulation.from_model_adapter(adapter)
    max_steps = fault_budget.max_steps(population.size)
    report_progress(progress, f"  eligible_bits: {population.size}")
    report_progress(progress, f"  max_flip_count: {max_steps}")
    if max_steps == 0:
        report_progress(progress, f"  stopped: {FAULT_BUDGET_STOP_REASON}")
        return SoftErrorRunResult(
            perturbation_trace=(),
            stopped_because=FAULT_BUDGET_STOP_REASON,
            eligible_bit_population=population.size,
        )

    sampler = _RemainingOrdinalSampler(population.size)
    entries: list[PerturbationTraceEntry] = []
    rng = Random(seed)

    with adapter.evaluation_mode():
        metric_before = _evaluate_metric(metric_evaluator, adapter)

        for step_index in range(max_steps):
            remaining_bits = population.size - step_index
            report_progress(
                progress,
                (
                    f"  step {step_index + 1:03d}/{max_steps:03d}: "
                    f"sampling uniform eligible bit from {remaining_bits} "
                    "uncommitted bits"
                ),
            )
            selection = population.selection_from_ordinal(
                sampler.sample(rng),
                codec=selected_codec,
            )
            value_before = adapter.read_tensor_value(
                selection.tensor_name,
                selection.tensor_index,
            )
            value_after = selected_codec.flip_value_bit(
                value_before,
                selection.bit_index,
            )
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
            report_progress(
                progress,
                _committed_flip_progress_message(
                    entry=entry,
                    total_steps=max_steps,
                    metric_after=metric_after,
                ),
            )
            if failure_criterion is not None and failure_criterion(entry):
                report_progress(progress, f"  stopped: {FAILURE_CRITERION_STOP_REASON}")
                return SoftErrorRunResult(
                    perturbation_trace=tuple(entries),
                    stopped_because=FAILURE_CRITERION_STOP_REASON,
                    eligible_bit_population=population.size,
                )
            metric_before = metric_after

    report_progress(progress, f"  stopped: {FAULT_BUDGET_STOP_REASON}")
    return SoftErrorRunResult(
        perturbation_trace=tuple(entries),
        stopped_because=FAULT_BUDGET_STOP_REASON,
        eligible_bit_population=population.size,
    )


def _committed_flip_progress_message(
    *,
    entry: PerturbationTraceEntry,
    total_steps: int,
    metric_after: dict[str, JSONScalar],
) -> str:
    layer = entry.layer_name if entry.layer_name is not None else entry.tensor_name
    return (
        f"  flip {entry.flip_count:03d}/{total_steps:03d}: "
        f"layer={layer} "
        f"tensor={entry.tensor_name} "
        f"index={entry.tensor_index} "
        f"bit={entry.bit_index} "
        f"metrics={format_progress_metrics(metric_after)}"
    )


class _RemainingOrdinalSampler:
    """Sample without replacement from ``range(size)`` using sparse swaps."""

    def __init__(self, size: int) -> None:
        self._remaining_count = _validate_positive_int(size, "size")
        self._swaps: dict[int, int] = {}

    def sample(self, rng: Random) -> int:
        """Return one remaining ordinal and remove it from future draws."""
        if self._remaining_count <= 0:
            msg = "no uncommitted eligible bits remain"
            raise ValueError(msg)

        draw_index = rng.randrange(self._remaining_count)
        selected_ordinal = self._swaps.get(draw_index, draw_index)
        last_index = self._remaining_count - 1
        last_ordinal = self._swaps.get(last_index, last_index)
        self._swaps[draw_index] = last_ordinal
        self._swaps.pop(last_index, None)
        self._remaining_count -= 1
        return selected_ordinal


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


def _ravel_index(tensor_index: Sequence[int], shape: Sequence[int]) -> int:
    if not shape:
        if tuple(tensor_index) != ():
            msg = "tensor_index is out of bounds for scalar tensor"
            raise IndexError(msg)
        return 0

    if len(tensor_index) != len(shape):
        msg = "tensor_index rank does not match tensor shape"
        raise IndexError(msg)

    value_ordinal = 0
    for coordinate, dimension in zip(tensor_index, shape, strict=True):
        coordinate_int = _validate_non_negative_int(coordinate, "tensor_index")
        dimension_int = _validate_positive_int(dimension, "shape dimension")
        if coordinate_int >= dimension_int:
            msg = "tensor_index is out of bounds for tensor shape"
            raise IndexError(msg)
        value_ordinal = value_ordinal * dimension_int + coordinate_int
    return value_ordinal


def _validate_excluded_ordinals(
    excluded_ordinals: set[int] | frozenset[int],
    population_size: int,
) -> None:
    for population_ordinal in excluded_ordinals:
        _validate_population_ordinal(population_ordinal, population_size)


def _population_ordinal_from_remaining_offset(
    remaining_offset: int,
    excluded_ordinals: set[int] | frozenset[int],
) -> int:
    population_ordinal = _validate_non_negative_int(
        remaining_offset,
        "remaining_offset",
    )
    for excluded_ordinal in sorted(excluded_ordinals):
        if excluded_ordinal > population_ordinal:
            break
        population_ordinal += 1
    return population_ordinal


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
