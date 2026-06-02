from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from contextlib import nullcontext
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
        values: Sequence[int],
        *,
        dtype: str = "int8",
        tensor_name: str = "features.weight",
    ) -> None:
        self.values = list(values)
        self.dtype = dtype
        self.tensor_name = tensor_name

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        return (
            PerturbableTensor(
                name=self.tensor_name,
                shape=(len(self.values),),
                dtype=self.dtype,
                requires_grad=True,
                numel=len(self.values),
                layer_name="features",
            ),
        )

    def read_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
    ) -> int:
        assert tensor_name == self.tensor_name
        return self.values[tensor_index[0]]

    def write_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
        value: Any,
    ) -> None:
        assert tensor_name == self.tensor_name
        self.values[tensor_index[0]] = value

    def evaluation_mode(self) -> nullcontext[None]:
        return nullcontext()

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        return self.values

    def classify(self, *args: Any, **kwargs: Any) -> Any:
        return self.values


def _sum_metric(adapter: ModelAdapter) -> dict[str, int]:
    assert isinstance(adapter, _Int8ListAdapter)
    return {"sum": sum(adapter.values)}


def _run_trace(values: Sequence[int], seed: int) -> list[dict[str, Any]]:
    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter(values),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=4),
        rng_seed=seed,
    )
    return [entry.model_dump(mode="json") for entry in result.perturbation_trace]


def test_sampler_is_uniform_over_a_small_eligible_bit_population() -> None:
    population = EligibleBitPopulation.from_model_adapter(_Int8ListAdapter([0]))
    rng = __import__("random").Random(2026)

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
    result = run_uniform_random_soft_error_baseline(
        adapter=_Int8ListAdapter([0]),
        metric_evaluator=_sum_metric,
        fault_budget=FaultBudget(max_flip_count=0),
        rng_seed=1,
    )

    assert result.stopped_because == FAULT_BUDGET_STOP_REASON
    assert result.perturbation_trace == ()


def test_population_rejects_models_without_eligible_int8_weight_bits() -> None:
    with pytest.raises(ValueError, match="eligible int8"):
        EligibleBitPopulation.from_model_adapter(_Int8ListAdapter([0], dtype="float32"))
