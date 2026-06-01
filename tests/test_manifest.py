from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pydantic import ValidationError

from netflip.manifest import RunManifest, build_run_manifest


def manifest_kwargs() -> dict[str, Any]:
    return {
        "run_id": "run-001",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "git_commit": "abc1234",
        "experiment_spec_hash": "sha256:spec",
        "model_artifact_id": "resnet20-cifar10",
        "model_checkpoint_path": "models/resnet20.pt",
        "model_checkpoint_checksum": "sha256:checkpoint",
        "quantization_metadata": {"codec": "signed-int8-two-complement"},
        "selection_dataset_id": "cifar10-train",
        "selection_dataset_checksum": "sha256:selection",
        "evaluation_dataset_id": "cifar10-test",
        "evaluation_dataset_checksum": "sha256:evaluation",
        "rng_seeds": {"python": 1, "torch": 2},
        "device": "cpu",
        "netflip_version": "0.1.0",
        "dependencies": {"click": "8.1.8"},
        "output_schema_version": "2026.1",
    }


def test_build_run_manifest_validates_reproducibility_record() -> None:
    manifest = build_run_manifest(**manifest_kwargs())

    assert manifest.run_id == "run-001"
    assert manifest.rng_seeds == {"python": 1, "torch": 2}
    assert manifest.model_dump(mode="json")["created_at"] == "2026-01-01T00:00:00Z"


def test_run_manifest_rejects_empty_required_fields() -> None:
    kwargs = manifest_kwargs()
    kwargs["run_id"] = ""

    with pytest.raises(ValidationError, match="run_id"):
        RunManifest.model_validate(kwargs)


def test_build_run_manifest_rejects_wrong_boundary_types() -> None:
    kwargs = manifest_kwargs()
    kwargs["created_at"] = "2026-01-01T00:00:00Z"

    with pytest.raises(BeartypeCallHintParamViolation):
        build_run_manifest(**kwargs)
