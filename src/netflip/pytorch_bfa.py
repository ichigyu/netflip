"""PyTorch helpers for efficient BFA/PBS Candidate Bit Flip selection."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from typing import Any

from netflip.bfa_pbs import (
    BfaPbsCandidatePlan,
    BfaPbsCandidateScore,
    ObjectiveEvaluator,
)
from netflip.int8_codec import INT8_BIT_WIDTH, SignedInt8TwoComplementCodec
from netflip.model_adapter import ModelAdapter
from netflip.soft_error import EligibleBitPopulation


@dataclass(frozen=True)
class GradientBfaPbsCandidateScorer:
    """Select BFA/PBS candidates using the original gradient-ranked PBS shape."""

    model: Any
    selection_batch: Any
    tensor_scales: Mapping[str, float]
    device: str
    k_top: int | None = None

    def __post_init__(self) -> None:
        if self.k_top is not None:
            _validate_positive_int(self.k_top, "k_top")

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
        torch = _require_pytorch()
        remaining_budget = _validate_positive_int(
            remaining_flip_budget,
            "remaining_flip_budget",
        )
        inputs, targets = _selection_inputs_and_targets(self.selection_batch)
        move_model = getattr(self.model, "to", None)
        if callable(move_model):
            move_model(self.device)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        candidates = _candidate_weight_modules(
            self.model,
            tensor_scales=self.tensor_scales,
            torch=torch,
        )
        if not candidates:
            return ()

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        objective_before = _evaluate_objective(objective_evaluator, adapter)
        for bit_count in range(1, remaining_budget + 1):
            self.model.eval()
            try:
                with _patched_dequantized_weight_proxies(
                    candidates,
                    torch=torch,
                ) as proxies:
                    zero_grad = getattr(self.model, "zero_grad", None)
                    if callable(zero_grad):
                        zero_grad(set_to_none=True)
                    outputs = self.model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outputs,
                        targets,
                        reduction="mean",
                    )
                    loss.backward()
                    layer_candidates = tuple(
                        _best_layer_candidate_plan(
                            candidate,
                            proxy=proxies[candidate.tensor_name],
                            population=population,
                            excluded_ordinals=excluded_ordinals,
                            codec=codec,
                            torch=torch,
                            k_top=self.k_top,
                            bit_count=bit_count,
                        )
                        for candidate in candidates
                    )
            finally:
                if was_training:
                    self.model.train()

            candidate_plans = _score_layer_candidate_plans(
                layer_candidates,
                adapter=adapter,
                objective_evaluator=objective_evaluator,
                objective_before=objective_before,
            )
            if (
                _best_candidate_plan_score(candidate_plans) > 0
                or bit_count == remaining_budget
            ):
                return candidate_plans
        return ()


@dataclass(frozen=True)
class _CandidateWeightModule:
    tensor_name: str
    layer_name: str
    module: Any
    scale: float


@dataclass(frozen=True)
class _LayerCandidate:
    population_ordinal: int
    tensor_name: str
    tensor_index: tuple[int, ...]
    layer_name: str | None
    bit_index: int
    bit_role: str
    value_before: int
    value_after: int


@dataclass(frozen=True)
class _LayerCandidatePlan:
    candidates: tuple[_LayerCandidate, ...]


def build_gradient_bfa_pbs_candidate_scorer(
    *,
    model: Any,
    selection_batch: Any,
    tensor_scales: Mapping[str, float],
    device: str,
    k_top: int | None = None,
) -> GradientBfaPbsCandidateScorer:
    """Build a gradient-ranked BFA/PBS Candidate Bit Flip scorer."""
    return GradientBfaPbsCandidateScorer(
        model=model,
        selection_batch=selection_batch,
        tensor_scales=tensor_scales,
        device=device,
        k_top=k_top,
    )


def _candidate_weight_modules(
    model: Any,
    *,
    tensor_scales: Mapping[str, float],
    torch: Any,
) -> tuple[_CandidateWeightModule, ...]:
    named_modules = dict(model.named_modules())
    module_types = (torch.nn.Conv2d, torch.nn.Linear)
    candidates: list[_CandidateWeightModule] = []
    for tensor_name, scale in tensor_scales.items():
        if not tensor_name.endswith(".weight"):
            continue
        layer_name = tensor_name[: -len(".weight")]
        module = named_modules.get(layer_name)
        if module is None or not isinstance(module, module_types):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != torch.int8:
            continue
        candidates.append(
            _CandidateWeightModule(
                tensor_name=tensor_name,
                layer_name=layer_name,
                module=module,
                scale=float(scale),
            )
        )
    return tuple(candidates)


@contextmanager
def _patched_dequantized_weight_proxies(
    candidates: tuple[_CandidateWeightModule, ...],
    *,
    torch: Any,
) -> Iterator[dict[str, Any]]:
    proxies: dict[str, Any] = {}
    originals: list[tuple[Any, Any]] = []
    try:
        for candidate in candidates:
            module = candidate.module
            proxy = (
                module.weight.detach().float().mul(candidate.scale).requires_grad_(True)
            )
            proxies[candidate.tensor_name] = proxy
            originals.append((module, module.forward))
            module.forward = _make_proxy_forward(module, proxy=proxy, torch=torch)
        yield proxies
    finally:
        for module, original_forward in reversed(originals):
            module.forward = original_forward


def _make_proxy_forward(module: Any, *, proxy: Any, torch: Any) -> Any:
    functional = torch.nn.functional
    if isinstance(module, torch.nn.Conv2d):

        def conv2d_forward(inputs: Any) -> Any:
            return functional.conv2d(
                inputs,
                proxy,
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )

        return conv2d_forward

    if isinstance(module, torch.nn.Linear):

        def linear_forward(inputs: Any) -> Any:
            return functional.linear(inputs, proxy, module.bias)

        return linear_forward

    msg = f"unsupported BFA/PBS candidate module type: {type(module).__name__}"
    raise TypeError(msg)


def _best_layer_candidate_plan(
    candidate: _CandidateWeightModule,
    *,
    proxy: Any,
    population: EligibleBitPopulation,
    excluded_ordinals: frozenset[int],
    codec: SignedInt8TwoComplementCodec,
    torch: Any,
    k_top: int | None,
    bit_count: int,
) -> _LayerCandidatePlan | None:
    grad = proxy.grad
    if grad is None:
        return None

    weight = candidate.module.weight.detach()
    shape = tuple(int(dimension) for dimension in weight.shape)
    value_count = int(weight.numel())
    if value_count <= 0:
        return None

    grad_flat = grad.detach().reshape(-1)
    value_ordinals = _ranked_value_ordinals(grad_flat, torch=torch, k_top=k_top)
    if value_ordinals.numel() == 0:
        return None

    quantized_flat = weight.reshape(-1).to(torch.int16)
    selected_values = quantized_flat[value_ordinals]
    selected_grads = grad_flat[value_ordinals]
    bit_delta_scores = _bit_delta_scores(
        selected_values,
        selected_grads,
        scale=candidate.scale,
        torch=torch,
    )
    rank_scores = torch.clamp(bit_delta_scores, min=0)

    start_ordinal = population.ordinal_from_selection(
        tensor_name=candidate.tensor_name,
        tensor_index=tuple(0 for _dimension in shape),
        bit_index=0,
    )
    local_bit_offsets = value_ordinals.unsqueeze(1) * INT8_BIT_WIDTH + torch.arange(
        INT8_BIT_WIDTH, device=grad_flat.device
    )
    if excluded_ordinals:
        excluded_local = [
            ordinal - start_ordinal
            for ordinal in excluded_ordinals
            if start_ordinal <= ordinal < start_ordinal + value_count * INT8_BIT_WIDTH
        ]
        if excluded_local:
            excluded_tensor = torch.tensor(excluded_local, device=grad_flat.device)
            excluded_mask = (
                local_bit_offsets.unsqueeze(-1).eq(excluded_tensor).any(dim=-1)
            )
            rank_scores = rank_scores.masked_fill(excluded_mask, -torch.inf)

    rank_scores = rank_scores.reshape(-1)
    if float(torch.max(rank_scores).item()) == 0:
        return None
    finite_scores = torch.isfinite(rank_scores)
    finite_count = int(finite_scores.sum().item())
    if finite_count <= 0:
        return None
    selected_count = min(bit_count, finite_count)
    _top_scores, top_positions = torch.topk(rank_scores, k=selected_count)
    current_values: dict[int, int] = {}
    selected_candidates: list[_LayerCandidate] = []
    for top_position in top_positions.tolist():
        row = int(top_position // INT8_BIT_WIDTH)
        bit_index = int(top_position % INT8_BIT_WIDTH)
        value_ordinal = int(value_ordinals[row].item())
        tensor_index = _unravel_index(value_ordinal, shape)
        population_ordinal = population.ordinal_from_selection(
            tensor_name=candidate.tensor_name,
            tensor_index=tensor_index,
            bit_index=bit_index,
        )
        selection = population.selection_from_ordinal(
            population_ordinal,
            codec=codec,
        )
        value_before = current_values.get(
            value_ordinal,
            int(weight.reshape(-1)[value_ordinal].item()),
        )
        value_after = codec.flip_value_bit(value_before, bit_index)
        current_values[value_ordinal] = value_after
        selected_candidates.append(
            _LayerCandidate(
                population_ordinal=population_ordinal,
                tensor_name=candidate.tensor_name,
                tensor_index=selection.tensor_index,
                layer_name=selection.layer_name,
                bit_index=selection.bit_index,
                bit_role=selection.bit_role,
                value_before=value_before,
                value_after=value_after,
            )
        )
    return _LayerCandidatePlan(candidates=tuple(selected_candidates))


def _ranked_value_ordinals(grad_flat: Any, *, torch: Any, k_top: int | None) -> Any:
    if k_top is None or k_top >= int(grad_flat.numel()):
        return torch.arange(int(grad_flat.numel()), device=grad_flat.device)
    _values, indices = torch.topk(torch.abs(grad_flat), k=k_top)
    return indices


def _score_layer_candidate_plans(
    layer_candidates: tuple[_LayerCandidatePlan | None, ...],
    *,
    adapter: ModelAdapter,
    objective_evaluator: ObjectiveEvaluator,
    objective_before: float,
) -> tuple[BfaPbsCandidatePlan, ...]:
    candidate_plans: list[BfaPbsCandidatePlan] = []
    for layer_candidate in layer_candidates:
        if layer_candidate is None:
            continue
        applied_candidates: list[_LayerCandidate] = []
        for candidate in layer_candidate.candidates:
            adapter.write_tensor_value(
                candidate.tensor_name,
                candidate.tensor_index,
                candidate.value_after,
            )
            applied_candidates.append(candidate)
        try:
            objective_after = _evaluate_objective(objective_evaluator, adapter)
        finally:
            for candidate in reversed(applied_candidates):
                adapter.write_tensor_value(
                    candidate.tensor_name,
                    candidate.tensor_index,
                    candidate.value_before,
                )
        candidate_plans.append(
            BfaPbsCandidatePlan(
                scores=tuple(
                    BfaPbsCandidateScore(
                        population_ordinal=candidate.population_ordinal,
                        tensor_name=candidate.tensor_name,
                        tensor_index=candidate.tensor_index,
                        layer_name=candidate.layer_name,
                        bit_index=candidate.bit_index,
                        bit_role=candidate.bit_role,
                        value_before=candidate.value_before,
                        value_after=candidate.value_after,
                        objective_before=objective_before,
                        objective_after=objective_after,
                        selection_score=objective_after - objective_before,
                    )
                    for candidate in layer_candidate.candidates
                )
            )
        )
    return tuple(candidate_plans)


def _best_candidate_plan_score(
    candidate_plans: tuple[BfaPbsCandidatePlan, ...],
) -> float:
    if not candidate_plans:
        return float("-inf")
    return max(candidate_plan.selection_score for candidate_plan in candidate_plans)


def _bit_delta_scores(
    values: Any,
    grads: Any,
    *,
    scale: float,
    torch: Any,
) -> Any:
    bit_indices = torch.arange(INT8_BIT_WIDTH, device=values.device)
    masks = (1 << bit_indices).to(torch.int16)
    bit_basis = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, -128],
        dtype=grads.dtype,
        device=grads.device,
    )
    encoded_values = torch.bitwise_and(values, 255)
    bit_is_set = torch.bitwise_and(encoded_values.unsqueeze(1), masks).ne(0)
    delta_quantized = torch.where(bit_is_set, -bit_basis, bit_basis)
    return grads.unsqueeze(1) * delta_quantized * scale


def _selection_inputs_and_targets(selection_batch: Any) -> tuple[Any, Any]:
    try:
        inputs, targets = selection_batch
    except (TypeError, ValueError) as exc:
        msg = "selection batch must contain inputs and ground-truth targets"
        raise TypeError(msg) from exc
    return inputs, targets


def _evaluate_objective(
    objective_evaluator: ObjectiveEvaluator,
    adapter: ModelAdapter,
) -> float:
    objective = objective_evaluator(adapter)
    if isinstance(objective, bool) or not isinstance(objective, (int, float)):
        msg = "BFA/PBS objective evaluator must return a numeric value"
        raise TypeError(msg)
    return float(objective)


def _unravel_index(value_ordinal: int, shape: tuple[int, ...]) -> tuple[int, ...]:
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


def _validate_positive_int(value: int, name: str) -> int:
    integer = _validate_int(value, name)
    if integer <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return integer


def _validate_non_negative_int(value: int, name: str) -> int:
    integer = _validate_int(value, name)
    if integer < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    return integer


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    return int(value)


def _require_pytorch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        msg = "gradient BFA/PBS candidate scoring requires PyTorch to be installed"
        raise ModuleNotFoundError(msg) from exc
