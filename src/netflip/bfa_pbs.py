"""BFA/PBS Attack Strategy over signed int8 model state bits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, TypeAlias

from netflip.int8_codec import SignedInt8TwoComplementCodec
from netflip.manifest import JSONScalar
from netflip.model_adapter import ModelAdapter
from netflip.progress import (
    ProgressReporter,
    format_progress_metrics,
    report_progress,
)
from netflip.soft_error import (
    MODEL_STATE_BITS_ARTIFACT_KIND,
    SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION,
    EligibleBitPopulation,
    MetricEvaluator,
)
from netflip.trace import CandidateTraceEntry, PerturbationTraceEntry

ATTACK_SCENARIO_TYPE = "attack"
BFA_PBS_STRATEGY_NAME = "bfa-pbs"
MAXIMIZE_CROSS_ENTROPY_OBJECTIVE = "maximize-cross-entropy"
GROUND_TRUTH_TARGET_POLICY = "ground-truth"
ATTACK_BUDGET_STOP_REASON = "attack_budget"
NO_IMPROVING_CANDIDATE_STOP_REASON = "no_improving_candidate"
ELIGIBLE_BITS_EXHAUSTED_STOP_REASON = "eligible_bits_exhausted"

ObjectiveEvaluator: TypeAlias = Callable[[ModelAdapter], float]


@dataclass(frozen=True)
class BfaPbsCandidateScore:
    """Score for one Candidate Bit Flip considered by BFA/PBS."""

    population_ordinal: int
    tensor_name: str
    tensor_index: tuple[int, ...]
    layer_name: str | None
    bit_index: int
    bit_role: str
    value_before: int
    value_after: int
    objective_before: float
    objective_after: float
    selection_score: float

    def to_trace_entry(
        self,
        *,
        step_index: int,
        rng_seed: int,
        eligible_bit_population: int,
    ) -> CandidateTraceEntry:
        """Convert this score to a Candidate Trace Entry."""
        return CandidateTraceEntry(
            step_index=step_index,
            scenario_type=ATTACK_SCENARIO_TYPE,
            strategy_name=BFA_PBS_STRATEGY_NAME,
            artifact_kind=MODEL_STATE_BITS_ARTIFACT_KIND,
            population_ordinal=self.population_ordinal,
            tensor_name=self.tensor_name,
            tensor_index=self.tensor_index,
            representation=SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION,
            bit_index=self.bit_index,
            bit_role=self.bit_role,
            value_before=self.value_before,
            value_after=self.value_after,
            objective_before=self.objective_before,
            objective_after=self.objective_after,
            selection_score=self.selection_score,
            rng_seed=rng_seed,
            layer_name=self.layer_name,
            eligible_bit_population=eligible_bit_population,
        )


@dataclass(frozen=True)
class BfaPbsCandidatePlan:
    """One BFA/PBS candidate plan, possibly containing multiple bit flips."""

    scores: tuple[BfaPbsCandidateScore, ...]

    def __post_init__(self) -> None:
        if not self.scores:
            msg = "BFA/PBS candidate plan must contain at least one score"
            raise ValueError(msg)

    @property
    def objective_before(self) -> float:
        """Objective value before applying this candidate plan."""
        return self.scores[0].objective_before

    @property
    def objective_after(self) -> float:
        """Objective value after applying this candidate plan."""
        return self.scores[-1].objective_after

    @property
    def selection_score(self) -> float:
        """Objective delta produced by applying this candidate plan."""
        return self.objective_after - self.objective_before

    @property
    def first_population_ordinal(self) -> int:
        """Stable tie-breaker ordinal for this candidate plan."""
        return min(score.population_ordinal for score in self.scores)


class BfaPbsCandidateScorer(Protocol):
    """Select a reduced set of BFA/PBS Candidate Bit Flip plans."""

    def __call__(
        self,
        *,
        adapter: ModelAdapter,
        objective_evaluator: ObjectiveEvaluator,
        population: EligibleBitPopulation,
        excluded_ordinals: frozenset[int],
        remaining_flip_budget: int,
        codec: SignedInt8TwoComplementCodec,
    ) -> tuple[BfaPbsCandidatePlan, ...]:
        """Return scored candidate plans for one BFA/PBS step."""
        ...


@dataclass(frozen=True)
class BfaPbsRunResult:
    """Trace and stop metadata for one BFA/PBS Attack Scenario Run."""

    perturbation_trace: tuple[PerturbationTraceEntry, ...]
    stopped_because: str
    eligible_bit_population: int
    candidate_trace: tuple[CandidateTraceEntry, ...] = ()

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


def validate_bfa_pbs_scenario_config(
    *,
    attack_objective: str,
    target_policy: str,
) -> None:
    """Raise clear errors for unsupported BFA/PBS configuration values."""
    if attack_objective != MAXIMIZE_CROSS_ENTROPY_OBJECTIVE:
        msg = (
            "unsupported BFA/PBS attack_objective "
            f"{attack_objective!r}; supported objective is "
            f"{MAXIMIZE_CROSS_ENTROPY_OBJECTIVE!r}"
        )
        raise ValueError(msg)
    if target_policy != GROUND_TRUTH_TARGET_POLICY:
        msg = (
            "unsupported BFA/PBS target_policy "
            f"{target_policy!r}; supported target policy is "
            f"{GROUND_TRUTH_TARGET_POLICY!r}"
        )
        raise ValueError(msg)


def score_bfa_pbs_candidates(
    *,
    adapter: ModelAdapter,
    objective_evaluator: ObjectiveEvaluator,
    attack_objective: str = MAXIMIZE_CROSS_ENTROPY_OBJECTIVE,
    target_policy: str = GROUND_TRUTH_TARGET_POLICY,
    excluded_ordinals: frozenset[int] | set[int] | None = None,
    codec: SignedInt8TwoComplementCodec | None = None,
) -> tuple[BfaPbsCandidateScore, ...]:
    """Score every uncommitted signed int8 Candidate Bit Flip.

    Scores are exact objective deltas on the configured selection batch. This is
    deterministic under fixed model state and inputs, and keeps the committed
    bit flip in the same two's-complement representation used by soft-error
    runs.
    """
    validate_bfa_pbs_scenario_config(
        attack_objective=attack_objective,
        target_policy=target_policy,
    )
    selected_codec = codec if codec is not None else SignedInt8TwoComplementCodec()
    population = EligibleBitPopulation.from_model_adapter(adapter)
    excluded = frozenset(excluded_ordinals or frozenset())
    _validate_excluded_ordinals(excluded, population.size)
    objective_before = _evaluate_objective(objective_evaluator, adapter)
    scores: list[BfaPbsCandidateScore] = []

    for population_ordinal in range(population.size):
        if population_ordinal in excluded:
            continue
        selection = population.selection_from_ordinal(
            population_ordinal,
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
        try:
            objective_after = _evaluate_objective(objective_evaluator, adapter)
        finally:
            adapter.write_tensor_value(
                selection.tensor_name,
                selection.tensor_index,
                value_before,
            )
        scores.append(
            BfaPbsCandidateScore(
                population_ordinal=population_ordinal,
                tensor_name=selection.tensor_name,
                tensor_index=selection.tensor_index,
                layer_name=selection.layer_name,
                bit_index=selection.bit_index,
                bit_role=selection.bit_role,
                value_before=value_before,
                value_after=value_after,
                objective_before=objective_before,
                objective_after=objective_after,
                selection_score=objective_after - objective_before,
            )
        )

    return tuple(scores)


def run_bfa_pbs_attack_strategy(
    *,
    adapter: ModelAdapter,
    objective_evaluator: ObjectiveEvaluator,
    metric_evaluator: MetricEvaluator,
    attack_objective: str,
    target_policy: str,
    max_flip_count: int,
    rng_seed: int,
    record_candidate_trace: bool = False,
    candidate_scorer: BfaPbsCandidateScorer | None = None,
    codec: SignedInt8TwoComplementCodec | None = None,
    progress: ProgressReporter | None = None,
) -> BfaPbsRunResult:
    """Run cumulative BFA/PBS, committing at most one bit per step."""
    validate_bfa_pbs_scenario_config(
        attack_objective=attack_objective,
        target_policy=target_policy,
    )
    max_steps = _validate_positive_int(max_flip_count, "max_flip_count")
    seed = _validate_int(rng_seed, "rng_seed")
    selected_codec = codec if codec is not None else SignedInt8TwoComplementCodec()
    population = EligibleBitPopulation.from_model_adapter(adapter)
    if population.size == 0:
        report_progress(progress, "  eligible_bits: 0")
        report_progress(progress, f"  stopped: {ELIGIBLE_BITS_EXHAUSTED_STOP_REASON}")
        return BfaPbsRunResult(
            perturbation_trace=(),
            stopped_because=ELIGIBLE_BITS_EXHAUSTED_STOP_REASON,
            eligible_bit_population=0,
        )
    committed_ordinals: set[int] = set()
    entries: list[PerturbationTraceEntry] = []
    candidate_entries: list[CandidateTraceEntry] = []
    total_steps = min(max_steps, population.size)

    report_progress(progress, f"  eligible_bits: {population.size}")
    report_progress(progress, f"  max_flip_count: {max_steps}")

    with adapter.evaluation_mode():
        metric_before = _evaluate_metric(metric_evaluator, adapter)
        step_index = 0

        while len(entries) < total_steps:
            remaining_candidates = population.size - len(committed_ordinals)
            remaining_flip_budget = total_steps - len(entries)
            if candidate_scorer is None:
                report_progress(
                    progress,
                    (
                        f"  step {step_index + 1:03d}/{total_steps:03d}: "
                        f"scoring {remaining_candidates} candidate bits"
                    ),
                )
                scores = score_bfa_pbs_candidates(
                    adapter=adapter,
                    objective_evaluator=objective_evaluator,
                    attack_objective=attack_objective,
                    target_policy=target_policy,
                    excluded_ordinals=committed_ordinals,
                    codec=selected_codec,
                )
                candidate_plans = tuple(
                    BfaPbsCandidatePlan(scores=(score,)) for score in scores
                )
            else:
                report_progress(
                    progress,
                    (
                        f"  step {step_index + 1:03d}/{total_steps:03d}: "
                        f"selecting gradient-ranked candidates from "
                        f"{remaining_candidates} eligible bits"
                    ),
                )
                candidate_plans = candidate_scorer(
                    adapter=adapter,
                    objective_evaluator=objective_evaluator,
                    population=population,
                    excluded_ordinals=frozenset(committed_ordinals),
                    remaining_flip_budget=remaining_flip_budget,
                    codec=selected_codec,
                )
                scores = tuple(
                    score
                    for candidate_plan in candidate_plans
                    for score in candidate_plan.scores
                )
            if record_candidate_trace:
                candidate_entries.extend(
                    score.to_trace_entry(
                        step_index=step_index,
                        rng_seed=seed,
                        eligible_bit_population=population.size,
                    )
                    for score in scores
                )
            selected_plan = _select_best_candidate_plan(candidate_plans)
            if selected_plan is None:
                report_progress(
                    progress, f"  stopped: {ELIGIBLE_BITS_EXHAUSTED_STOP_REASON}"
                )
                return BfaPbsRunResult(
                    perturbation_trace=tuple(entries),
                    stopped_because=ELIGIBLE_BITS_EXHAUSTED_STOP_REASON,
                    eligible_bit_population=population.size,
                    candidate_trace=tuple(candidate_entries),
                )
            if selected_plan.selection_score <= 0:
                report_progress(
                    progress, f"  stopped: {NO_IMPROVING_CANDIDATE_STOP_REASON}"
                )
                return BfaPbsRunResult(
                    perturbation_trace=tuple(entries),
                    stopped_because=NO_IMPROVING_CANDIDATE_STOP_REASON,
                    eligible_bit_population=population.size,
                    candidate_trace=tuple(candidate_entries),
                )
            _validate_selected_candidate_plan(
                selected_plan,
                remaining_flip_budget=remaining_flip_budget,
                committed_ordinals=committed_ordinals,
            )

            for selected_score in selected_plan.scores:
                adapter.write_tensor_value(
                    selected_score.tensor_name,
                    selected_score.tensor_index,
                    selected_score.value_after,
                )
                committed_ordinals.add(selected_score.population_ordinal)
            metric_after = _evaluate_metric(metric_evaluator, adapter)
            for selected_score in selected_plan.scores:
                flip_count = len(entries) + 1
                entries.append(
                    PerturbationTraceEntry(
                        step_index=step_index,
                        scenario_type=ATTACK_SCENARIO_TYPE,
                        strategy_name=BFA_PBS_STRATEGY_NAME,
                        artifact_kind=MODEL_STATE_BITS_ARTIFACT_KIND,
                        tensor_name=selected_score.tensor_name,
                        tensor_index=selected_score.tensor_index,
                        representation=SIGNED_INT8_TWO_COMPLEMENT_REPRESENTATION,
                        bit_index=selected_score.bit_index,
                        bit_role=selected_score.bit_role,
                        value_before=selected_score.value_before,
                        value_after=selected_score.value_after,
                        flip_count=flip_count,
                        bit_flip_ratio=flip_count / population.size,
                        metric_before=metric_before,
                        metric_after=metric_after,
                        selection_score=selected_plan.selection_score,
                        rng_seed=seed,
                        layer_name=selected_score.layer_name,
                        eligible_bit_population=population.size,
                    )
                )
                report_progress(
                    progress,
                    _committed_flip_progress_message(
                        flip_count=flip_count,
                        total_steps=total_steps,
                        score=selected_score,
                        metric_after=metric_after,
                    ),
                )
            metric_before = metric_after
            step_index += 1

    report_progress(progress, f"  stopped: {ATTACK_BUDGET_STOP_REASON}")
    return BfaPbsRunResult(
        perturbation_trace=tuple(entries),
        stopped_because=ATTACK_BUDGET_STOP_REASON,
        eligible_bit_population=population.size,
        candidate_trace=tuple(candidate_entries),
    )


def _select_best_candidate(
    scores: tuple[BfaPbsCandidateScore, ...],
) -> BfaPbsCandidateScore | None:
    if not scores:
        return None
    return max(
        scores,
        key=lambda score: (score.selection_score, -score.population_ordinal),
    )


def _select_best_candidate_plan(
    candidate_plans: tuple[BfaPbsCandidatePlan, ...],
) -> BfaPbsCandidatePlan | None:
    if not candidate_plans:
        return None
    return max(
        candidate_plans,
        key=lambda plan: (plan.selection_score, -plan.first_population_ordinal),
    )


def _validate_selected_candidate_plan(
    selected_plan: BfaPbsCandidatePlan,
    *,
    remaining_flip_budget: int,
    committed_ordinals: set[int],
) -> None:
    if len(selected_plan.scores) > remaining_flip_budget:
        msg = "BFA/PBS candidate plan exceeds remaining Flip Count budget"
        raise ValueError(msg)
    seen_ordinals: set[int] = set()
    for score in selected_plan.scores:
        if score.population_ordinal in committed_ordinals:
            msg = "BFA/PBS candidate plan includes an already committed bit"
            raise ValueError(msg)
        if score.population_ordinal in seen_ordinals:
            msg = "BFA/PBS candidate plan includes a duplicate bit"
            raise ValueError(msg)
        seen_ordinals.add(score.population_ordinal)


def _evaluate_metric(
    metric_evaluator: MetricEvaluator,
    adapter: ModelAdapter,
) -> dict[str, JSONScalar]:
    return dict(metric_evaluator(adapter))


def _committed_flip_progress_message(
    *,
    flip_count: int,
    total_steps: int,
    score: BfaPbsCandidateScore,
    metric_after: dict[str, JSONScalar],
) -> str:
    layer = score.layer_name if score.layer_name is not None else score.tensor_name
    return (
        f"  flip {flip_count:03d}/{total_steps:03d}: "
        f"layer={layer} "
        f"tensor={score.tensor_name} "
        f"index={score.tensor_index} "
        f"bit={score.bit_index} "
        f"score={score.selection_score:.6g} "
        f"metrics={format_progress_metrics(metric_after)}"
    )


def _evaluate_objective(
    objective_evaluator: ObjectiveEvaluator,
    adapter: ModelAdapter,
) -> float:
    objective = objective_evaluator(adapter)
    if isinstance(objective, bool) or not isinstance(objective, (int, float)):
        msg = "BFA/PBS objective evaluator must return a numeric value"
        raise TypeError(msg)
    return float(objective)


def _validate_excluded_ordinals(
    excluded_ordinals: frozenset[int],
    population_size: int,
) -> None:
    for population_ordinal in excluded_ordinals:
        ordinal = _validate_non_negative_int(population_ordinal, "population_ordinal")
        if ordinal >= population_size:
            msg = "population_ordinal is out of bounds"
            raise IndexError(msg)


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
