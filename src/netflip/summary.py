"""Run Summary validation models and JSON/CSV writers."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field, field_validator

from netflip.manifest import OUTPUT_SCHEMA_VERSION, JSONScalar
from netflip.trace import PerturbationTraceEntry

RUN_SUMMARY_JSON_FILENAME = "summary.json"
RUN_SUMMARY_CSV_FILENAME = "summary.csv"
FAILURE_CRITERION_STOP_REASON = "failure_criterion"


class RunSummary(BaseModel):
    """Aggregate metrics for one completed Run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clean_metrics: dict[str, JSONScalar]
    final_metrics: dict[str, JSONScalar]
    flip_count: int = Field(ge=0)
    bit_flip_ratio: float = Field(ge=0)
    eligible_bit_population: int = Field(gt=0)
    failure_flip_count: int | None = Field(default=None, ge=1)
    stop_reason: str = Field(min_length=1)
    output_schema_version: str = Field(
        default=OUTPUT_SCHEMA_VERSION,
        min_length=1,
    )

    @field_validator("clean_metrics", "final_metrics")
    @classmethod
    def require_metric_names(
        cls,
        metrics: dict[str, JSONScalar],
    ) -> dict[str, JSONScalar]:
        """Require non-empty metric names so JSON and CSV stay addressable."""
        for metric_name in metrics:
            if not metric_name:
                msg = "metric names must be non-empty strings"
                raise ValueError(msg)
        return metrics


@beartype
def build_run_summary(
    *,
    clean_metrics: Mapping[str, JSONScalar],
    perturbation_trace: Sequence[PerturbationTraceEntry],
    stopped_because: str,
    eligible_bit_population: int,
) -> RunSummary:
    """Build a Run Summary from the completed Run result and trace entries."""
    entries = tuple(perturbation_trace)
    clean_metrics_dict = dict(clean_metrics)
    if not entries:
        return RunSummary(
            clean_metrics=clean_metrics_dict,
            final_metrics=clean_metrics_dict,
            flip_count=0,
            bit_flip_ratio=0,
            eligible_bit_population=eligible_bit_population,
            failure_flip_count=None,
            stop_reason=stopped_because,
        )

    first_entry = entries[0]
    last_entry = entries[-1]
    trace_clean_metrics = dict(first_entry.metric_before)
    if clean_metrics_dict != trace_clean_metrics:
        msg = "clean metrics must match the first trace entry metric_before"
        raise ValueError(msg)
    if (
        last_entry.eligible_bit_population is not None
        and last_entry.eligible_bit_population != eligible_bit_population
    ):
        msg = "eligible bit population must match the perturbation trace"
        raise ValueError(msg)

    failure_flip_count = None
    if stopped_because == FAILURE_CRITERION_STOP_REASON:
        failure_flip_count = last_entry.flip_count

    return RunSummary(
        clean_metrics=trace_clean_metrics,
        final_metrics=dict(last_entry.metric_after),
        flip_count=last_entry.flip_count,
        bit_flip_ratio=last_entry.bit_flip_ratio,
        eligible_bit_population=eligible_bit_population,
        failure_flip_count=failure_flip_count,
        stop_reason=stopped_because,
    )


@beartype
def write_run_summary_json(
    summary: RunSummary,
    output_dir: str | PathLike[str],
) -> Path:
    """Write ``summary.json`` in a run output directory."""
    summary_path = Path(output_dir) / RUN_SUMMARY_JSON_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        f"{summary.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return summary_path


@beartype
def write_run_summary_csv(
    summary: RunSummary,
    output_dir: str | PathLike[str],
) -> Path:
    """Write one-row ``summary.csv`` for quick run comparisons."""
    summary_path = Path(output_dir) / RUN_SUMMARY_CSV_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    row = _summary_csv_row(summary)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return summary_path


@beartype
def write_run_summary(
    summary: RunSummary,
    output_dir: str | PathLike[str],
) -> tuple[Path, Path]:
    """Write Run Summary JSON and CSV outputs."""
    return (
        write_run_summary_json(summary, output_dir),
        write_run_summary_csv(summary, output_dir),
    )


def _summary_csv_row(summary: RunSummary) -> dict[str, JSONScalar]:
    row: dict[str, JSONScalar] = {
        "output_schema_version": summary.output_schema_version,
        "stop_reason": summary.stop_reason,
        "flip_count": summary.flip_count,
        "bit_flip_ratio": summary.bit_flip_ratio,
        "eligible_bit_population": summary.eligible_bit_population,
        "failure_flip_count": summary.failure_flip_count,
    }
    for metric_name in sorted(summary.clean_metrics):
        row[f"clean_{metric_name}"] = summary.clean_metrics[metric_name]
    for metric_name in sorted(summary.final_metrics):
        row[f"final_{metric_name}"] = summary.final_metrics[metric_name]
    return row
