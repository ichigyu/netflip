"""Run Manifest validation models.

Example:
    >>> from datetime import datetime, timezone
    >>> manifest = build_run_manifest(
    ...     run_id="demo-run",
    ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ...     git_commit="abc1234",
    ...     experiment_spec_hash="sha256:example",
    ...     model_artifact_id="resnet20-cifar10",
    ...     model_checkpoint_path="models/resnet20.pt",
    ...     model_checkpoint_checksum="sha256:checkpoint",
    ...     quantization_metadata={"codec": "signed-int8-two-complement"},
    ...     selection_dataset_id="cifar10-train",
    ...     selection_dataset_checksum="sha256:selection",
    ...     evaluation_dataset_id="cifar10-test",
    ...     evaluation_dataset_checksum="sha256:evaluation",
    ...     rng_seeds={"python": 1},
    ...     device="cpu",
    ...     netflip_version="0.1.0",
    ...     dependencies={"click": "8.1.8"},
    ...     output_schema_version="2026.1",
    ... )
    >>> manifest.run_id
    'demo-run'
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field

JSONScalar = str | int | float | bool | None


class RunManifest(BaseModel):
    """Reproducibility record for one NetFlip run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    created_at: datetime
    git_commit: str = Field(min_length=1)
    experiment_spec_hash: str = Field(min_length=1)
    model_artifact_id: str = Field(min_length=1)
    model_checkpoint_path: str = Field(min_length=1)
    model_checkpoint_checksum: str = Field(min_length=1)
    quantization_metadata: dict[str, JSONScalar]
    selection_dataset_id: str = Field(min_length=1)
    selection_dataset_checksum: str = Field(min_length=1)
    evaluation_dataset_id: str = Field(min_length=1)
    evaluation_dataset_checksum: str = Field(min_length=1)
    rng_seeds: dict[str, int]
    device: str = Field(min_length=1)
    netflip_version: str = Field(min_length=1)
    dependencies: dict[str, str]
    output_schema_version: str = Field(min_length=1)


@beartype
def build_run_manifest(
    *,
    run_id: str,
    created_at: datetime,
    git_commit: str,
    experiment_spec_hash: str,
    model_artifact_id: str,
    model_checkpoint_path: str,
    model_checkpoint_checksum: str,
    quantization_metadata: Mapping[str, JSONScalar],
    selection_dataset_id: str,
    selection_dataset_checksum: str,
    evaluation_dataset_id: str,
    evaluation_dataset_checksum: str,
    rng_seeds: Mapping[str, int],
    device: str,
    netflip_version: str,
    dependencies: Mapping[str, str],
    output_schema_version: str,
) -> RunManifest:
    """Build a validated Run Manifest from mapping-friendly inputs."""
    return RunManifest(
        run_id=run_id,
        created_at=created_at,
        git_commit=git_commit,
        experiment_spec_hash=experiment_spec_hash,
        model_artifact_id=model_artifact_id,
        model_checkpoint_path=model_checkpoint_path,
        model_checkpoint_checksum=model_checkpoint_checksum,
        quantization_metadata=dict(quantization_metadata),
        selection_dataset_id=selection_dataset_id,
        selection_dataset_checksum=selection_dataset_checksum,
        evaluation_dataset_id=evaluation_dataset_id,
        evaluation_dataset_checksum=evaluation_dataset_checksum,
        rng_seeds=dict(rng_seeds),
        device=device,
        netflip_version=netflip_version,
        dependencies=dict(dependencies),
        output_schema_version=output_schema_version,
    )
