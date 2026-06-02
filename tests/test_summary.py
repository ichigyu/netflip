from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from netflip.summary import (
    FAILURE_CRITERION_STOP_REASON,
    RUN_SUMMARY_CSV_FILENAME,
    RUN_SUMMARY_JSON_FILENAME,
    build_run_summary,
    write_run_summary,
)
from netflip.trace import PerturbationTraceEntry


def trace_entry() -> PerturbationTraceEntry:
    return PerturbationTraceEntry(
        step_index=0,
        scenario_type="soft_error",
        strategy_name="uniform-eligible-bit",
        artifact_kind="model_state_bits",
        tensor_name="features.weight",
        tensor_index=(0,),
        representation="signed-int8-two-complement",
        bit_index=7,
        bit_role="sign_msb",
        value_before=12,
        value_after=-116,
        flip_count=1,
        bit_flip_ratio=0.1,
        metric_before={"accuracy": 0.91},
        metric_after={"accuracy": 0.42},
        selection_score=None,
        rng_seed=2026,
        layer_name="features",
        eligible_bit_population=10,
    )


def test_build_run_summary_derives_metrics_from_non_empty_trace() -> None:
    first_entry = trace_entry()
    second_entry = first_entry.model_copy(
        update={
            "step_index": 1,
            "bit_index": 0,
            "bit_role": "lsb",
            "value_before": -116,
            "value_after": -115,
            "flip_count": 2,
            "bit_flip_ratio": 0.2,
            "metric_before": {"accuracy": 0.42},
            "metric_after": {"accuracy": 0.25},
        }
    )

    summary = build_run_summary(
        clean_metrics={"accuracy": 0.91},
        perturbation_trace=[first_entry, second_entry],
        stopped_because="fault_budget",
        eligible_bit_population=10,
    )

    assert summary.clean_metrics == {"accuracy": 0.91}
    assert summary.final_metrics == {"accuracy": 0.25}
    assert summary.flip_count == 2
    assert summary.bit_flip_ratio == 0.2
    assert summary.eligible_bit_population == 10
    assert summary.failure_flip_count is None
    assert summary.stop_reason == "fault_budget"


def test_build_run_summary_handles_empty_trace() -> None:
    summary = build_run_summary(
        clean_metrics={"accuracy": 0.91},
        perturbation_trace=[],
        stopped_because="fault_budget",
        eligible_bit_population=10,
    )

    assert summary.clean_metrics == {"accuracy": 0.91}
    assert summary.final_metrics == {"accuracy": 0.91}
    assert summary.flip_count == 0
    assert summary.bit_flip_ratio == 0
    assert summary.failure_flip_count is None


def test_build_run_summary_records_failure_flip_count() -> None:
    summary = build_run_summary(
        clean_metrics={"accuracy": 0.91},
        perturbation_trace=[trace_entry()],
        stopped_because=FAILURE_CRITERION_STOP_REASON,
        eligible_bit_population=10,
    )

    assert summary.flip_count == 1
    assert summary.failure_flip_count == 1
    assert summary.stop_reason == "failure_criterion"


def test_build_run_summary_rejects_trace_inconsistent_with_clean_metrics() -> None:
    with pytest.raises(ValueError, match="clean metrics"):
        build_run_summary(
            clean_metrics={"accuracy": 1.0},
            perturbation_trace=[trace_entry()],
            stopped_because="fault_budget",
            eligible_bit_population=10,
        )


def test_write_run_summary_emits_json_and_one_row_csv(tmp_path: Path) -> None:
    summary = build_run_summary(
        clean_metrics={"accuracy": 0.91},
        perturbation_trace=[trace_entry()],
        stopped_because="fault_budget",
        eligible_bit_population=10,
    )

    json_path, csv_path = write_run_summary(summary, tmp_path)

    assert json_path == tmp_path / RUN_SUMMARY_JSON_FILENAME
    assert csv_path == tmp_path / RUN_SUMMARY_CSV_FILENAME
    written_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert written_json["clean_metrics"] == {"accuracy": 0.91}
    assert written_json["final_metrics"] == {"accuracy": 0.42}
    assert written_json["flip_count"] == 1
    assert written_json["bit_flip_ratio"] == 0.1
    assert written_json["failure_flip_count"] is None
    assert written_json["stop_reason"] == "fault_budget"

    with csv_path.open(encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert rows == [
        {
            "output_schema_version": "2026.1",
            "stop_reason": "fault_budget",
            "flip_count": "1",
            "bit_flip_ratio": "0.1",
            "eligible_bit_population": "10",
            "failure_flip_count": "",
            "clean_accuracy": "0.91",
            "final_accuracy": "0.42",
        }
    ]
