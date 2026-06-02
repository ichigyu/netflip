from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from netflip import __version__
from netflip.console import main
from netflip.model_adapter import PerturbableTensor


def test_cli_version() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert f"netflip, version {__version__}" in result.output


def test_cli_without_args_prints_help() -> None:
    runner = CliRunner()

    result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert "Neural-network bit-flip reliability" in result.output


def test_cli_help_options_print_help() -> None:
    runner = CliRunner()

    for help_option in ("-h", "--help"):
        result = runner.invoke(main, [help_option])

        assert result.exit_code == 0
        assert "Neural-network bit-flip reliability" in result.output


class _TinyInt8Adapter:
    def __init__(self) -> None:
        self.values = [0, 0]

    def perturbable_tensors(self) -> tuple[PerturbableTensor, ...]:
        return (
            PerturbableTensor(
                name="features.weight",
                shape=(2,),
                dtype="int8",
                requires_grad=False,
                numel=2,
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
        yield

    def inference(self, *args: Any, **kwargs: Any) -> list[int]:
        return self.values

    def classify(self, *args: Any, **kwargs: Any) -> list[int]:
        return self.values


def test_cli_run_soft_error_spec_writes_manifest_and_trace(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import netflip.run as run_module

    adapter = _TinyInt8Adapter()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(
        json.dumps(
            {
                "codec": "signed-int8-two-complement",
                "scale_granularity": "per-tensor",
                "tensors": {"features.weight": {"scale": 0.25}},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run-output"
    spec_path = tmp_path / "soft-error.yaml"
    spec_path.write_text(
        f"""
        schema_version: "2026.1"
        run_id: cli-soft-error
        device: cpu
        model:
          benchmark: cifar10-resnet20
          architecture: resnet20
          num_classes: 10
          checkpoint:
            path: {checkpoint_path}
            format: pytorch-state-dict
          quantization:
            codec: signed-int8-two-complement
            scale_granularity: per-tensor
            scale_path: {scale_path}
        dataset:
          name: cifar10
          root: {tmp_path / "data"}
          selection_split: train
          evaluation_split: test
        scenario:
          type: soft_error
          fault_model: uniform-eligible-bit
          fault_schedule: one-bit-step
          fault_budget:
            max_flip_count: 2
          rng_seed: 7
        output_dir: {output_dir}
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(run_module, "resolve_torch_device", lambda requested: "cpu")
    monkeypatch.setattr(
        run_module,
        "load_cifar_resnet20_quantized_artifact",
        lambda **kwargs: SimpleNamespace(
            model=object(),
            adapter=adapter,
            checkpoint_path=checkpoint_path,
            scale_path=scale_path,
            quantization=SimpleNamespace(
                codec="signed-int8-two-complement",
                scale_granularity="per-tensor",
                tensors={"features.weight": object()},
            ),
        ),
    )
    monkeypatch.setattr(
        run_module,
        "build_cifar10_dataloaders",
        lambda **kwargs: SimpleNamespace(selection=["selection"], evaluation=["eval"]),
    )
    monkeypatch.setattr(
        run_module,
        "evaluate_classification_metrics",
        lambda model, dataloader, *, device: {"sum": sum(adapter.values)},
    )

    runner = CliRunner()

    result = runner.invoke(main, ["run", str(spec_path)])

    assert result.exit_code == 0
    assert "Run output directory:" in result.output
    assert "Resolved device: cpu" in result.output
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "cli-soft-error"
    assert manifest["device"] == "cpu"
    assert manifest["rng_seeds"] == {"python": 7}
    trace_lines = (
        (output_dir / "perturbation_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(trace_lines) == 2
    assert json.loads(trace_lines[0])["scenario_type"] == "soft_error"


def test_cli_run_unsupported_scenario_type_fails_clearly(tmp_path: Path) -> None:
    spec_path = tmp_path / "attack.yaml"
    spec_path.write_text(
        f"""
        schema_version: "2026.1"
        run_id: cli-attack
        device: cpu
        model:
          benchmark: cifar10-resnet20
          architecture: resnet20
          num_classes: 10
          checkpoint:
            path: {tmp_path / "checkpoint.pt"}
            format: pytorch-state-dict
          quantization:
            codec: signed-int8-two-complement
            scale_granularity: per-tensor
            scale_path: {tmp_path / "scales.json"}
        dataset:
          name: cifar10
          root: {tmp_path / "data"}
        scenario:
          type: attack
          strategy_name: bfa-pbs
          attack_objective: maximize-loss
          target_policy: ground-truth
          max_flip_count: 1
          selection_batch_size: 1
          rng_seed: 7
        output_dir: {tmp_path / "run-output"}
        """,
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(main, ["run", str(spec_path)])

    assert result.exit_code == 1
    assert "unsupported scenario type 'attack'" in result.stderr


def test_cli_run_allows_unexpected_errors_to_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import netflip.console as console_module

    def broken_execute_experiment_run(spec: str) -> object:
        raise RuntimeError("internal bug")

    monkeypatch.setattr(
        console_module,
        "execute_experiment_run",
        broken_execute_experiment_run,
    )
    runner = CliRunner()

    with pytest.raises(RuntimeError, match="internal bug"):
        runner.invoke(main, ["run", "spec.yaml"], catch_exceptions=False)


def test_cli_run_requires_spec_argument() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["run"])

    assert result.exit_code != 0
    assert "Missing argument 'SPEC'" in result.stderr
