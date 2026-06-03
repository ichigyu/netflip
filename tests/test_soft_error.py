from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import pairwise
from random import Random
from typing import Any

import pytest

from netflip import (
    FAILURE_CRITERION_STOP_REASON,
    FAULT_BUDGET_STOP_REASON,
    EligibleBitPopulation,
    FaultBudget,
    ModelAdapter,
    PerturbableTensor,
    PerturbationTraceEntry,
    run_uniform_random_soft_error_baseline,
    sample_uniform_eligible_bit,
)


class _Int8ListAdapter:
    def __init__(
        self,
        values: Any,
        *,
        dtype: str = "int8",
        tensor_name: str = "features.weight",
    ) -> None:
        self.values = values
        self.shape = _infer_shape(values)
        self.dtype = dtype
        self.tensor_name = tensor_name
        self.evaluation_depth = 0
        self.metric_evaluations_in_eval_mode: list[bool] = []

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        return (
            PerturbableTensor(
                name=self.tensor_name,
                shape=self.shape,
                dtype=self.dtype,
                requires_grad=True,
                numel=_numel(self.shape),
                layer_name="features",
            ),
        )

    def read_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
    ) -> int:
        assert tensor_name == self.tensor_name
        return _nested_get(self.values, tuple(tensor_index))

    def write_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
        value: Any,
    ) -> None:
        assert tensor_name == self.tensor_name
        _nested_set(self.values, tuple(tensor_index), value)

    @contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        self.evaluation_depth += 1
        try:
            yield
        finally:
            self.evaluation_depth -= 1

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        return self.values

    def classify(self, *args: Any, **kwargs: Any) -> Any:
        return self.values


def _sum_metric(adapter: ModelAdapter) -> dict[str, int]:
    assert isinstance(adapter, _Int8ListAdapter)
    adapter.metric_evaluations_in_eval_mode.append(adapter.evaluation_depth > 0)
    return {"sum": _nested_sum(adapter.values)}


def _run_trace(values: Sequence[int], seed: int) -> list[dict[str, Any]]:
    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter(values),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=4),
        rng_seed=seed,
    )
    return [entry.model_dump(mode="json") for entry in result.perturbation_trace]


def _infer_shape(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    return (len(values), *_infer_shape(values[0]))


def _numel(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _nested_get(values: Any, index: tuple[int, ...]) -> int:
    current = values
    for coordinate in index:
        current = current[coordinate]
    return current


def _nested_set(values: Any, index: tuple[int, ...], value: int) -> None:
    current = values
    for coordinate in index[:-1]:
        current = current[coordinate]
    current[index[-1]] = value


def _nested_sum(values: Any) -> int:
    if isinstance(values, list):
        return sum(_nested_sum(value) for value in values)
    return values


def test_sampler_is_uniform_over_a_small_eligible_bit_population() -> None:
    population = EligibleBitPopulation.from_model_adapter(_Int8ListAdapter([0]))
    rng = Random(2026)

    counts = Counter(
        sample_uniform_eligible_bit(population, rng).population_ordinal
        for _ in range(80_000)
    )

    assert set(counts) == set(range(8))
    assert min(counts.values()) > 9_600
    assert max(counts.values()) < 10_400


def test_run_commits_one_bit_per_step_and_records_trace_fields() -> None:
    adapter = _Int8ListAdapter([0, 0])

    result = run_uniform_random_soft_error_baseline(
        adapter=adapter,
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=3),
        rng_seed=7,
    )

    entries = result.perturbation_trace
    committed_bits = {(entry.tensor_index, entry.bit_index) for entry in entries}
    assert result.stopped_because == FAULT_BUDGET_STOP_REASON
    assert result.eligible_bit_population == 16
    assert result.flip_count == 3
    assert result.bit_flip_ratio == 3 / 16
    assert [entry.step_index for entry in entries] == [0, 1, 2]
    assert [entry.flip_count for entry in entries] == [1, 2, 3]
    assert [entry.bit_flip_ratio for entry in entries] == [1 / 16, 2 / 16, 3 / 16]
    assert len(committed_bits) == 3
    assert all(entry.scenario_type == "soft_error" for entry in entries)
    assert all(entry.strategy_name == "uniform-eligible-bit" for entry in entries)
    assert all(entry.artifact_kind == "model_state_bits" for entry in entries)
    assert all(entry.rng_seed == 7 for entry in entries)
    assert all(entry.eligible_bit_population == 16 for entry in entries)
    assert entries[0].metric_before == {"sum": 0}
    assert all(
        previous.metric_after == current.metric_before
        for previous, current in pairwise(entries)
    )
    assert adapter.evaluation_depth == 0
    assert all(adapter.metric_evaluations_in_eval_mode)


def test_run_reports_soft_error_progress() -> None:
    messages: list[str] = []

    run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0, 0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=2),
        rng_seed=7,
        progress=messages.append,
    )

    assert messages[:2] == [
        "  eligible_bits: 16",
        "  max_flip_count: 2",
    ]
    assert messages[2] == (
        "  step 001/002: sampling uniform eligible bit from 16 uncommitted bits"
    )
    assert messages[3].startswith(
        "  flip 001/002: layer=features tensor=features.weight index=("
    )
    assert " bit=" in messages[3]
    assert " metrics=sum=" in messages[3]
    assert messages[4] == (
        "  step 002/002: sampling uniform eligible bit from 15 uncommitted bits"
    )
    assert messages[5].startswith(
        "  flip 002/002: layer=features tensor=features.weight index=("
    )
    assert messages[-1] == "  stopped: fault_budget"


def test_fixed_seeds_produce_deterministic_perturbation_traces() -> None:
    assert _run_trace([0, 0, 0], seed=11) == _run_trace([0, 0, 0], seed=11)
    assert _run_trace([0, 0, 0], seed=11) != _run_trace([0, 0, 0], seed=12)


def test_run_stops_at_bit_flip_ratio_budget() -> None:
    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_bit_flip_ratio=0.25),
        rng_seed=1,
    )

    assert result.stopped_because == FAULT_BUDGET_STOP_REASON
    assert result.flip_count == 2
    assert result.bit_flip_ratio == 0.25


def test_run_stops_at_failure_criterion() -> None:
    def stop_after_first_flip(entry: PerturbationTraceEntry) -> bool:
        return entry.flip_count == 1

    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0, 0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=10),
        rng_seed=5,
        failure_criterion=stop_after_first_flip,
    )

    assert result.stopped_because == FAILURE_CRITERION_STOP_REASON
    assert result.flip_count == 1


def test_fault_budget_allows_zero_committed_flips() -> None:
    messages: list[str] = []

    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=0),
        rng_seed=1,
        progress=messages.append,
    )

    assert result.stopped_because == FAULT_BUDGET_STOP_REASON
    assert result.perturbation_trace == ()
    assert messages == [
        "  eligible_bits: 8",
        "  max_flip_count: 0",
        "  stopped: fault_budget",
    ]


def test_run_reports_failure_criterion_stop_progress() -> None:
    messages: list[str] = []

    def stop_after_first_flip(entry: PerturbationTraceEntry) -> bool:
        return entry.flip_count == 1

    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0, 0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=10),
        rng_seed=5,
        failure_criterion=stop_after_first_flip,
        progress=messages.append,
    )

    assert result.stopped_because == FAILURE_CRITERION_STOP_REASON
    assert messages[-1] == "  stopped: failure_criterion"


def test_population_rejects_models_without_eligible_int8_weight_bits() -> None:
    with pytest.raises(ValueError, match="eligible int8"):
        EligibleBitPopulation.from_model_adapter(_Int8ListAdapter([0], dtype="float32"))


def test_population_selection_from_ordinal_maps_multidimensional_tensor() -> None:
    population = EligibleBitPopulation.from_model_adapter(
        _Int8ListAdapter([[1, 2, 3], [4, 5, 6]])
    )

    first_selection = population.selection_from_ordinal(0)
    second_row_selection = population.selection_from_ordinal(3 * 8)
    last_selection = population.selection_from_ordinal(population.size - 1)

    assert population.size == 2 * 3 * 8
    assert first_selection.tensor_index == (0, 0)
    assert first_selection.bit_index == 0
    assert second_row_selection.tensor_index == (1, 0)
    assert second_row_selection.bit_index == 0
    assert last_selection.tensor_index == (1, 2)
    assert last_selection.bit_index == 7


def test_population_selection_from_ordinal_rejects_out_of_bounds_ordinals() -> None:
    population = EligibleBitPopulation.from_model_adapter(_Int8ListAdapter([0]))

    with pytest.raises(IndexError, match="out of bounds"):
        population.selection_from_ordinal(population.size)

    with pytest.raises(IndexError, match="out of bounds"):
        population.selection_from_ordinal(population.size + 10)
