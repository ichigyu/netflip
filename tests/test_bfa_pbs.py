from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

import netflip.bfa_pbs as bfa_pbs_module
import netflip.pytorch_bfa as pytorch_bfa_module
from netflip import (
    ATTACK_BUDGET_STOP_REASON,
    ELIGIBLE_BITS_EXHAUSTED_STOP_REASON,
    GROUND_TRUTH_TARGET_POLICY,
    MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
    NO_IMPROVING_CANDIDATE_STOP_REASON,
    BfaPbsCandidatePlan,
    BfaPbsCandidateScore,
    BfaPbsRunResult,
    ModelAdapter,
    PerturbableTensor,
    PyTorchModelAdapter,
    SignedInt8TwoComplementCodec,
    build_gradient_bfa_pbs_candidate_scorer,
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


def test_eligible_bit_population_round_trips_selection_to_ordinal() -> None:
    adapter = _TinyInt8Adapter([0, 0])
    population = bfa_pbs_module.EligibleBitPopulation.from_model_adapter(adapter)
    selection = population.selection_from_ordinal(13)

    assert (
        population.ordinal_from_selection(
            tensor_name=selection.tensor_name,
            tensor_index=selection.tensor_index,
            bit_index=selection.bit_index,
        )
        == 13
    )


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


def test_attack_candidate_scorer_can_commit_multi_bit_plan() -> None:
    adapter = _TinyInt8Adapter([0])

    def candidate_scorer(
        *,
        adapter: ModelAdapter,
        objective_evaluator: Callable[[ModelAdapter], float],
        population: Any,
        excluded_ordinals: frozenset[int],
        remaining_flip_budget: int,
        codec: Any,
    ) -> tuple[BfaPbsCandidatePlan, ...]:
        assert excluded_ordinals == frozenset()
        assert remaining_flip_budget == 2
        objective_before = objective_evaluator(adapter)
        first_selection = population.selection_from_ordinal(6, codec=codec)
        second_selection = population.selection_from_ordinal(5, codec=codec)
        return (
            BfaPbsCandidatePlan(
                scores=(
                    BfaPbsCandidateScore(
                        population_ordinal=6,
                        tensor_name=first_selection.tensor_name,
                        tensor_index=first_selection.tensor_index,
                        layer_name=first_selection.layer_name,
                        bit_index=first_selection.bit_index,
                        bit_role=first_selection.bit_role,
                        value_before=0,
                        value_after=64,
                        objective_before=objective_before,
                        objective_after=96,
                        selection_score=96 - objective_before,
                    ),
                    BfaPbsCandidateScore(
                        population_ordinal=5,
                        tensor_name=second_selection.tensor_name,
                        tensor_index=second_selection.tensor_index,
                        layer_name=second_selection.layer_name,
                        bit_index=second_selection.bit_index,
                        bit_role=second_selection.bit_role,
                        value_before=64,
                        value_after=96,
                        objective_before=objective_before,
                        objective_after=96,
                        selection_score=96 - objective_before,
                    ),
                )
            ),
        )

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=_sum_objective,
        metric_evaluator=_sum_metric,
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=2,
        rng_seed=2026,
        record_candidate_trace=True,
        candidate_scorer=candidate_scorer,
    )

    assert result.stopped_because == ATTACK_BUDGET_STOP_REASON
    assert result.flip_count == 2
    assert adapter.values == [96]
    assert [entry.step_index for entry in result.perturbation_trace] == [0, 0]
    assert [entry.flip_count for entry in result.perturbation_trace] == [1, 2]
    assert [entry.value_after for entry in result.perturbation_trace] == [64, 96]
    assert len(result.candidate_trace) == 2


def test_candidate_plan_requires_at_least_one_score() -> None:
    with pytest.raises(ValueError, match="candidate plan"):
        BfaPbsCandidatePlan(scores=())


def test_attack_stops_when_candidate_scorer_returns_no_plans() -> None:
    adapter = _TinyInt8Adapter([0])

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=_sum_objective,
        metric_evaluator=_sum_metric,
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=1,
        rng_seed=2026,
        candidate_scorer=lambda **kwargs: (),
    )

    assert result.stopped_because == ELIGIBLE_BITS_EXHAUSTED_STOP_REASON
    assert result.perturbation_trace == ()


def test_attack_rejects_candidate_plan_that_exceeds_remaining_budget() -> None:
    adapter = _TinyInt8Adapter([0])

    def oversized_scorer(**kwargs: Any) -> tuple[BfaPbsCandidatePlan, ...]:
        population = kwargs["population"]
        objective_before = kwargs["objective_evaluator"](adapter)
        scores: list[BfaPbsCandidateScore] = []
        for ordinal in (6, 5):
            selection = population.selection_from_ordinal(ordinal)
            scores.append(
                BfaPbsCandidateScore(
                    population_ordinal=ordinal,
                    tensor_name=selection.tensor_name,
                    tensor_index=selection.tensor_index,
                    layer_name=selection.layer_name,
                    bit_index=selection.bit_index,
                    bit_role=selection.bit_role,
                    value_before=0,
                    value_after=64,
                    objective_before=objective_before,
                    objective_after=64,
                    selection_score=64,
                )
            )
        return (BfaPbsCandidatePlan(scores=tuple(scores)),)

    with pytest.raises(ValueError, match="remaining Flip Count budget"):
        run_bfa_pbs_attack_strategy(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            metric_evaluator=_sum_metric,
            attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
            target_policy=GROUND_TRUTH_TARGET_POLICY,
            max_flip_count=1,
            rng_seed=2026,
            candidate_scorer=oversized_scorer,
        )


def test_gradient_ranked_bfa_scorer_commits_one_pytorch_candidate() -> None:
    torch = pytest.importorskip("torch")

    class TinyInt8LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(2, 2, bias=False)
            self.classifier.weight = torch.nn.Parameter(
                torch.tensor([[0, 0], [0, 0]], dtype=torch.int8),
                requires_grad=False,
            )

            def dequantized_forward(inputs: Any) -> Any:
                return torch.nn.functional.linear(
                    inputs,
                    self.classifier.weight.float(),
                    None,
                )

            self.classifier.forward = dequantized_forward

        def forward(self, inputs: Any) -> Any:
            return self.classifier(inputs)

    model = TinyInt8LinearModel()
    adapter = PyTorchModelAdapter(model)
    inputs = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([0])
    scorer = build_gradient_bfa_pbs_candidate_scorer(
        model=model,
        selection_batch=(inputs, targets),
        tensor_scales={"classifier.weight": 1.0},
        device="cpu",
    )
    progress_messages: list[str] = []

    def objective(adapter: ModelAdapter) -> float:
        assert isinstance(adapter, PyTorchModelAdapter)
        with torch.no_grad():
            outputs = adapter.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        return float(loss.item())

    result = run_bfa_pbs_attack_strategy(
        adapter=adapter,
        objective_evaluator=objective,
        metric_evaluator=lambda adapter: {"loss": objective(adapter)},
        attack_objective=MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
        target_policy=GROUND_TRUTH_TARGET_POLICY,
        max_flip_count=1,
        rng_seed=2026,
        candidate_scorer=scorer,
        progress=progress_messages.append,
    )

    assert result.stopped_because == ATTACK_BUDGET_STOP_REASON
    assert result.flip_count == 1
    assert result.perturbation_trace[0].tensor_name == "classifier.weight"
    selection_score = result.perturbation_trace[0].selection_score
    assert isinstance(selection_score, (int, float))
    assert not isinstance(selection_score, bool)
    assert selection_score > 0
    assert any(
        "selecting gradient-ranked candidates" in message
        for message in progress_messages
    )


def test_pytorch_candidate_weight_modules_filter_to_int8_conv_and_linear() -> None:
    torch = pytest.importorskip("torch")

    class MixedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
            self.conv.weight = torch.nn.Parameter(
                torch.tensor([[[[1]]]], dtype=torch.int8),
                requires_grad=False,
            )
            self.float_linear = torch.nn.Linear(1, 1, bias=False)

    model = MixedModule()

    candidates = pytorch_bfa_module._candidate_weight_modules(
        model,
        tensor_scales={
            "conv.weight": 0.5,
            "float_linear.weight": 0.25,
            "missing.weight": 0.25,
            "conv.bias": 0.25,
        },
        torch=torch,
    )

    assert len(candidates) == 1
    assert candidates[0].tensor_name == "conv.weight"
    assert candidates[0].layer_name == "conv"
    assert candidates[0].scale == pytest.approx(0.5)


def test_pytorch_weight_proxy_forward_supports_conv2d_and_restores_forward() -> None:
    torch = pytest.importorskip("torch")
    conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    conv.weight = torch.nn.Parameter(
        torch.tensor([[[[2]]]], dtype=torch.int8),
        requires_grad=False,
    )
    original_forward = conv.forward
    candidate = pytorch_bfa_module._CandidateWeightModule(
        tensor_name="conv.weight",
        layer_name="conv",
        module=conv,
        scale=0.5,
    )

    with pytorch_bfa_module._patched_dequantized_weight_proxies(
        (candidate,),
        torch=torch,
    ) as proxies:
        proxy = proxies["conv.weight"]
        output = conv(torch.ones((1, 1, 1, 1)))
        output.sum().backward()

    assert output.item() == pytest.approx(1.0)
    assert proxy.grad.item() == pytest.approx(1.0)
    assert conv.forward == original_forward


def test_pytorch_best_layer_candidate_plan_respects_excluded_ordinals() -> None:
    torch = pytest.importorskip("torch")
    adapter = _TinyInt8Adapter([0, 0])
    population = bfa_pbs_module.EligibleBitPopulation.from_model_adapter(adapter)

    class WeightHolder:
        weight = torch.tensor([0, 0], dtype=torch.int8)

    proxy = torch.tensor([1.0, 2.0], requires_grad=True)
    proxy.grad = torch.tensor([1.0, 2.0])
    candidate = pytorch_bfa_module._CandidateWeightModule(
        tensor_name="features.weight",
        layer_name="features",
        module=WeightHolder(),
        scale=1.0,
    )

    plan = pytorch_bfa_module._best_layer_candidate_plan(
        candidate,
        proxy=proxy,
        population=population,
        excluded_ordinals=frozenset({14}),
        codec=SignedInt8TwoComplementCodec(),
        torch=torch,
        k_top=1,
        bit_count=2,
    )

    assert plan is not None
    assert [selection.population_ordinal for selection in plan.candidates] == [13, 12]
    assert [selection.bit_index for selection in plan.candidates] == [5, 4]


def test_pytorch_gradient_scorer_returns_empty_when_no_candidate_modules() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.ReLU()
    scorer = build_gradient_bfa_pbs_candidate_scorer(
        model=model,
        selection_batch=(torch.tensor([1.0]), torch.tensor([0])),
        tensor_scales={"missing.weight": 1.0},
        device="cpu",
        k_top=1,
    )
    adapter = _TinyInt8Adapter([0])
    population = bfa_pbs_module.EligibleBitPopulation.from_model_adapter(adapter)

    assert (
        scorer(
            adapter=adapter,
            objective_evaluator=_sum_objective,
            population=population,
            excluded_ordinals=frozenset(),
            remaining_flip_budget=1,
            codec=SignedInt8TwoComplementCodec(),
        )
        == ()
    )


def test_pytorch_gradient_scorer_validates_helper_inputs() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="k_top"):
        build_gradient_bfa_pbs_candidate_scorer(
            model=torch.nn.ReLU(),
            selection_batch=(torch.tensor([1.0]), torch.tensor([0])),
            tensor_scales={},
            device="cpu",
            k_top=0,
        )
    with pytest.raises(TypeError, match="selection batch"):
        pytorch_bfa_module._selection_inputs_and_targets(object())
    with pytest.raises(TypeError, match="objective evaluator"):
        pytorch_bfa_module._evaluate_objective(
            lambda adapter: True, _TinyInt8Adapter([0])
        )
    with pytest.raises(TypeError, match="candidate module"):
        pytorch_bfa_module._make_proxy_forward(object(), proxy=object(), torch=torch)
    with pytest.raises(IndexError, match="out of bounds"):
        pytorch_bfa_module._unravel_index(3, (2,))


def test_eligible_bit_population_rejects_invalid_reverse_bit_index() -> None:
    adapter = _TinyInt8Adapter([0])
    population = bfa_pbs_module.EligibleBitPopulation.from_model_adapter(adapter)

    with pytest.raises(ValueError, match="bit_index"):
        population.ordinal_from_selection(
            tensor_name="features.weight",
            tensor_index=(0,),
            bit_index=8,
        )


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
    progress_messages: list[str] = []
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
        progress=progress_messages.append,
    )

    assert result.stopped_because == ELIGIBLE_BITS_EXHAUSTED_STOP_REASON
    assert result.flip_count == 0
    assert result.eligible_bit_population == 0
    assert result.bit_flip_ratio == 0
    assert adapter.evaluation_depth == 0
    assert progress_messages == [
        "  eligible_bits: 0",
        f"  stopped: {ELIGIBLE_BITS_EXHAUSTED_STOP_REASON}",
    ]


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
