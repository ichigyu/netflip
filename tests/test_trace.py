from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from netflip.trace import (
    CandidateTraceEntry,
    PerturbationTraceEntry,
    candidate_trace_path,
    write_candidate_trace,
    write_perturbation_trace,
)


def trace_entry() -> PerturbationTraceEntry:
    return PerturbationTraceEntry(
        step_index=0,
        scenario_type="attack",
        strategy_name="bfa-pbs",
        artifact_kind="model_state_bits",
        tensor_name="layer1.weight",
        tensor_index=(0, 1, 2, 3),
        representation="signed-int8-two-complement",
        bit_index=7,
        bit_role="sign_msb",
        value_before=12,
        value_after=-116,
        flip_count=1,
        bit_flip_ratio=0.0001,
        metric_before={"accuracy": 0.91},
        metric_after={"accuracy": 0.42},
        selection_score=4.75,
        selection_seed=2026,
        layer_name="layer1",
        eligible_bit_population=10_000,
    )


def test_perturbation_trace_entry_records_required_contract_fields() -> None:
    entry = trace_entry()

    dumped = entry.model_dump(mode="json")

    assert {
        "artifact_kind",
        "bit_flip_ratio",
        "bit_index",
        "bit_role",
        "flip_count",
        "metric_after",
        "metric_before",
        "output_schema_version",
        "representation",
        "scenario_type",
        "selection_score",
        "selection_seed",
        "step_index",
        "strategy_name",
        "tensor_index",
        "tensor_name",
        "value_after",
        "value_before",
    } <= set(dumped)
    assert dumped["step_index"] == 0
    assert dumped["scenario_type"] == "attack"
    assert dumped["strategy_name"] == "bfa-pbs"
    assert dumped["tensor_index"] == [0, 1, 2, 3]
    assert dumped["bit_index"] == 7
    assert dumped["bit_role"] == "sign_msb"
    assert dumped["flip_count"] == 1
    assert dumped["output_schema_version"] == "2026.1"
    assert dumped["selection_seed"] == 2026


def test_perturbation_trace_entry_requires_reproducibility_seed() -> None:
    kwargs = trace_entry().model_dump()
    kwargs["selection_seed"] = None

    with pytest.raises(ValidationError, match="rng_seed or selection_seed"):
        PerturbationTraceEntry.model_validate(kwargs)


def test_write_perturbation_trace_emits_one_committed_bit_flip_per_line(
    tmp_path: Path,
) -> None:
    first_entry = trace_entry()
    second_entry = first_entry.model_copy(
        update={
            "step_index": 1,
            "bit_index": 0,
            "bit_role": "lsb",
            "value_before": -116,
            "value_after": -115,
            "flip_count": 2,
            "bit_flip_ratio": 0.0002,
            "selection_score": 3.25,
        }
    )

    trace_path = write_perturbation_trace([first_entry, second_entry], tmp_path)

    assert trace_path == tmp_path / "perturbation_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step_index"] == 0
    assert json.loads(lines[1])["step_index"] == 1
    assert json.loads(lines[1])["flip_count"] == 2


def test_candidate_trace_path_reserves_feature_gated_output_path(
    tmp_path: Path,
) -> None:
    assert candidate_trace_path(tmp_path) == tmp_path / "candidate_trace.jsonl"


def test_write_candidate_trace_emits_one_candidate_bit_flip_per_line(
    tmp_path: Path,
) -> None:
    entry = CandidateTraceEntry(
        step_index=0,
        scenario_type="attack",
        strategy_name="bfa-pbs",
        artifact_kind="model_state_bits",
        population_ordinal=7,
        tensor_name="layer1.weight",
        tensor_index=(0, 1),
        representation="signed-int8-two-complement",
        bit_index=7,
        bit_role="sign_msb",
        value_before=0,
        value_after=-128,
        objective_before=0.5,
        objective_after=4.25,
        selection_score=3.75,
        rng_seed=2026,
        layer_name="layer1",
        eligible_bit_population=128,
    )

    trace_path = write_candidate_trace([entry], tmp_path)

    assert trace_path == tmp_path / "candidate_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["population_ordinal"] == 7
    assert written["selection_score"] == 3.75
