from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

import netflip.bfa_pbs as bfa_pbs_module
from netflip import (
    ATTACK_BUDGET_STOP_REASON,
    ELIGIBLE_BITS_EXHAUSTED_STOP_REASON,
    GROUND_TRUTH_TARGET_POLICY,
    MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
    NO_IMPROVING_CANDIDATE_STOP_REASON,
    BfaPbsCandidateScore,
    BfaPbsRunResult,
    ModelAdapter,
    PerturbableTensor,
    run_bfa_pbs_attack_strategy,
    score_bfa_pbs_candidates,
    validate_bfa_pbs_scenario_config,
)

_InvalidObjectiveEvaluator = Callable[[ModelAdapter], Any]


class _TinyInt8Adapter:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.evaluation_depth = 0

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        return (
            PerturbableTensor(
                name="features.weight",
                shape=(len(self.values),),
                dtype="int8",
                requires_grad=False,
                numel=len(self.values),
                layer_name="features",
            ),
        )

    def read_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
    ) -> int:
        assert tensor_name == "features.weight"
        return self.values[tensor_index[0]]

    def write_tensor_value(
        self,
        tensor_name: str,
        tensor_index: Sequence[int],
        value: Any,
    ) -> None:
        assert tensor_name == "features.weight"
        self.values[tensor_index[0]] = int(value)

    @contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        self.evaluation_depth += 1
        try:
            yield
        finally:
            self.evaluation_depth -= 1

    def inference(self, *args: Any, **kwargs: Any) -> list[int]:
        return self.values

    def classify(self, *args: Any, **kwargs: Any) -> list[int]:
        return self.values


class _EmptyEligibleBitPopulation:
    size = 0


def _sum_objective(adapter: ModelAdapter) -> float:
    assert isinstance(adapter, _TinyInt8Adapter)
    return float(sum(adapter.values))


def _sum_metric(adapter: ModelAdapter) -> dict[str, int]:
    assert isinstance(adapter, _TinyInt8Adapter)
    return {"sum": sum(adapter.values)}


class _CustomBadObjective:
    """Custom non-numeric objective value used by validation tests."""


def _string_objective(adapter: ModelAdapter) -> str:
    return "not-a-number"


def _bool_objective(adapter: ModelAdapter) -> bool:
    return True


def _object_objective(adapter: ModelAdapter) -> _CustomBadObjective:
    return _CustomBadObjective()


def _score_dump(scores: tuple[BfaPbsCandidateScore, ...]) -> list[dict[str, Any]]:
    return [score.__dict__ for score in scores]


def test_candidate_scoring_is_deterministic_and_restores_model_state() -> None:
    adapter = _TinyInt8Adapter([0, 0])

    first_scores = score_bfa_pbs_candidates(
        adapter=adapter,
        objective_evaluator=_sum_objective,
    )
    second_scores = score_bfa_pbs_candidates(
        adapter=adapter,
        objective_evaluator=_sum_objective,
    )
    best_score = max(
        first_scores,
        key=lambda score: (score.selection_score, -score.population_ordinal),
    )

    assert _score_dump(first_scores) == _score_dump(second_scores)
    assert adapter.values == [0, 0]
    assert best_score.tensor_name == "features.weight"
    assert best_score.tensor_index == (0,)
    assert best_score.bit_index == 6
    assert best_score.value_before == 0
    assert best_score.value_after == 64
    assert best_score.selection_score == 64.0


def test_attack_commits_one_bit_per_step_and_records_trace_output() -> None:
    adapter = _TinyInt8Adapter([0])
    progress_messages: list[str] = []

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=_sum_objective,
        metric_evaluator=_sum_metric,
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=2,
        rng_seed=2026,
        record_candidate_trace=True,
        progress=progress_messages.append,
    )

    entries = result.perturbation_trace
    assert result.stopped_because == ATTACK_BUDGET_STOP_REASON
    assert result.flip_count == 2
    assert result.bit_flip_ratio == 2 / 8
    assert adapter.values == [96]
    assert [entry.step_index for entry in entries] == [0, 1]
    assert [entry.bit_index for entry in entries] == [6, 5]
    assert [entry.value_before for entry in entries] == [0, 64]
    assert [entry.value_after for entry in entries] == [64, 96]
    assert [entry.selection_score for entry in entries] == [64.0, 32.0]
    assert entries[0].scenario_type == "attack"
    assert entries[0].strategy_name == "bfa-pbs"
    assert entries[0].metric_before == {"sum": 0}
    assert entries[1].metric_before == {"sum": 64}
    assert len(result.candidate_trace) == 15
    assert result.candidate_trace[0].objective_before == 0.0
    assert result.candidate_trace[0].eligible_bit_population == 8
    assert adapter.evaluation_depth == 0
    assert "  eligible_bits: 8" in progress_messages
    assert "  step 001/002: scoring 8 candidate bits" in progress_messages
    assert any(
        "flip 001/002" in message
        and "layer=features" in message
        and "bit=6" in message
        and "metrics=sum=64" in message
        for message in progress_messages
    )
    assert f"  stopped: {ATTACK_BUDGET_STOP_REASON}" in progress_messages


def test_attack_stops_clearly_when_no_candidate_improves_objective() -> None:
    adapter = _TinyInt8Adapter([0])

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=lambda adapter: 1.0,
        metric_evaluator=_sum_metric,
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=2,
        rng_seed=2026,
    )

    assert result.stopped_because == NO_IMPROVING_CANDIDATE_STOP_REASON
    assert result.perturbation_trace == ()
    assert adapter.values == [0]


@pytest.mark.parametrize(
    "objective_evaluator",
    [_string_objective, _bool_objective, _object_objective],
)
def test_candidate_scoring_rejects_invalid_objective_return_types(
    objective_evaluator: _InvalidObjectiveEvaluator,
) -> None:
    adapter = _TinyInt8Adapter([0])

    with pytest.raises(TypeError, match="objective evaluator must return"):
        score_bfa_pbs_candidates(
            adapter=adapter,
            objective_evaluator=objective_evaluator,
        )

    assert adapter.values == [0]


@pytest.mark.parametrize(
    "objective_evaluator",
    [_string_objective, _bool_objective, _object_objective],
)
def test_attack_run_rejects_invalid_objective_return_types(
    objective_evaluator: _InvalidObjectiveEvaluator,
) -> None:
    adapter = _TinyInt8Adapter([0])

    with pytest.raises(TypeError, match="objective evaluator must return"):
        run_bfa_pbs_attack_strategy(
            adapter=adapter,
            objective_evaluator=objective_evaluator,
            metric_evaluator=_sum_metric,
            attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
            target_policy=GROUND_TRUTH_TARGET_POLICY,
            max_flip_count=1,
            rng_seed=2026,
        )

    assert adapter.values == [0]


def test_run_result_bit_flip_ratio_handles_empty_population() -> None:
    result = BfaPbsRunResult(
        perturbation_trace=(),
        stopped_because=ATTACK_BUDGET_STOP_REASON,
        eligible_bit_population=0,
    )

    assert result.bit_flip_ratio == 0


def test_attack_run_stops_clearly_for_empty_eligible_bit_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TinyInt8Adapter([0])
    monkeypatch.setattr(
        bfa_pbs_module.EligibleBitPopulation,
        "from_model_adapter",
        lambda adapter: _EmptyEligibleBitPopulation(),
    )

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=_sum_objective,
        metric_evaluator=_sum_metric,
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=1,
        rng_seed=2026,
    )

    assert result.stopped_because == ELIGIBLE_BITS_EXHAUSTED_STOP_REASON
    assert result.flip_count == 0
    assert result.eligible_bit_population == 0
    assert result.bit_flip_ratio == 0
    assert adapter.evaluation_depth == 0


def test_candidate_scoring_rejects_invalid_excluded_ordinals() -> None:
    adapter = _TinyInt8Adapter([0])

    with pytest.raises(ValueError, match="population_ordinal must be non-negative"):
        score_bfa_pbs_candidates(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            excluded_ordinals={-1},
        )

    with pytest.raises(IndexError, match="population_ordinal is out of bounds"):
        score_bfa_pbs_candidates(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            excluded_ordinals={8},
        )


def test_attack_run_rejects_invalid_integer_inputs() -> None:
    adapter = _TinyInt8Adapter([0])

    with pytest.raises(ValueError, match="max_flip_count must be positive"):
        run_bfa_pbs_attack_strategy(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            metric_evaluator=_sum_metric,
            attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
            target_policy=GROUND_TRUTH_TARGET_POLICY,
            max_flip_count=0,
            rng_seed=2026,
        )

    with pytest.raises(TypeError, match="rng_seed must be an integer"):
        run_bfa_pbs_attack_strategy(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            metric_evaluator=_sum_metric,
            attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
            target_policy=GROUND_TRUTH_TARGET_POLICY,
            max_flip_count=1,
            rng_seed=True,
        )


def test_unsupported_attack_objectives_and_target_policies_fail_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported BFA/PBS attack_objective"):
        validate_bfa_pbs_scenario_config(
            attack_objective="maximize-loss",
            target_policy=GROUND_TRUTH_TARGET_POLICY,
        )

    with pytest.raises(ValueError, match="unsupported BFA/PBS target_policy"):
        validate_bfa_pbs_scenario_config(
            attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
            target_policy="clean-prediction",
        )
