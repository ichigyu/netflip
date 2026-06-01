"""Perturbation Trace validation models and JSONL writer."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netflip.manifest import OUTPUT_SCHEMA_VERSION, JSONScalar

PERTURBATION_TRACE_FILENAME = "perturbation_trace.jsonl"
CANDIDATE_TRACE_FILENAME = "candidate_trace.jsonl"


class PerturbationTraceEntry(BaseModel):
    """Ordered record for one committed bit flip in a Run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    scenario_type: str = Field(min_length=1)
    strategy_name: str = Field(min_length=1)
    artifact_kind: str = Field(min_length=1)
    tensor_name: str = Field(min_length=1)
    tensor_index: tuple[int, ...]
    representation: str = Field(min_length=1)
    bit_index: int = Field(ge=0)
    bit_role: str = Field(min_length=1)
    value_before: JSONScalar
    value_after: JSONScalar
    flip_count: int = Field(ge=1)
    bit_flip_ratio: float = Field(ge=0)
    metric_before: dict[str, JSONScalar]
    metric_after: dict[str, JSONScalar]
    selection_score: JSONScalar
    output_schema_version: str = Field(
        default=OUTPUT_SCHEMA_VERSION,
        min_length=1,
    )
    rng_seed: int | None = None
    selection_seed: int | None = None
    layer_name: str | None = None
    eligible_bit_population: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_selection_reproducibility_seed(self) -> PerturbationTraceEntry:
        """Require at least one seed used to select the committed bit flip."""
        if self.rng_seed is None and self.selection_seed is None:
            msg = "rng_seed or selection_seed is required"
            raise ValueError(msg)
        return self


@beartype
def write_perturbation_trace(
    entries: Iterable[PerturbationTraceEntry],
    output_dir: str | PathLike[str],
) -> Path:
    """Write ``perturbation_trace.jsonl`` with one committed bit flip per line."""
    trace_path = Path(output_dir) / PERTURBATION_TRACE_FILENAME
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for entry in entries:
            trace_file.write(f"{entry.model_dump_json()}\n")
    return trace_path


@beartype
def candidate_trace_path(output_dir: str | PathLike[str]) -> Path:
    """Return the reserved optional Candidate Trace path.

    Candidate trace emission is feature-gated until Candidate Bit Flip search
    internals exist.
    """
    return Path(output_dir) / CANDIDATE_TRACE_FILENAME
