"""Console interface for NetFlip."""

from __future__ import annotations

import click

from netflip import __version__
from netflip.benchmarks import prepare_cifar_resnet20_artifacts
from netflip.run import ExperimentRunError, execute_experiment_run
from netflip.runtime_device import RuntimeDeviceUnavailableError


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(__version__, prog_name="netflip")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Neural-network bit-flip reliability evaluation framework."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("spec")
def run(spec: str) -> None:
    """Run an experiment spec."""
    try:
        output = execute_experiment_run(spec)
    except ExperimentRunError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Run output directory: {output.output_dir}")
    click.echo(f"Manifest: {output.manifest_path}")
    click.echo(f"Perturbation trace: {output.perturbation_trace_path}")
    if output.candidate_trace_path is not None:
        click.echo(f"Candidate trace: {output.candidate_trace_path}")
    click.echo(f"Summary JSON: {output.summary_json_path}")
    click.echo(f"Summary CSV: {output.summary_csv_path}")
    click.echo(f"Resolved device: {output.device}")
    click.echo(f"Committed flip count: {output.flip_count}")
    click.echo(f"Stopped because: {output.stopped_because}")


@main.command("prepare-cifar10-resnet20")
@click.option(
    "--dataset-root",
    default="data/cifar10",
    show_default=True,
    help="CIFAR-10 dataset root.",
)
@click.option(
    "--output-dir",
    default="checkpoints/cifar10",
    show_default=True,
    help="Directory for prepared checkpoint artifacts.",
)
@click.option(
    "--download/--no-download",
    default=False,
    show_default=True,
    help="Allow torchvision to download CIFAR-10 into the dataset root.",
)
@click.option(
    "--epochs",
    default=1,
    show_default=True,
    type=int,
    help="Number of FP32 training epochs before quantization.",
)
@click.option(
    "--batch-size",
    default=128,
    show_default=True,
    type=int,
    help="Training and evaluation batch size.",
)
@click.option(
    "--learning-rate",
    default=0.1,
    show_default=True,
    type=float,
    help="SGD learning rate.",
)
@click.option(
    "--momentum",
    default=0.9,
    show_default=True,
    type=float,
    help="SGD momentum.",
)
@click.option(
    "--weight-decay",
    default=5e-4,
    show_default=True,
    type=float,
    help="SGD weight decay.",
)
@click.option(
    "--train-sample-limit",
    type=int,
    default=None,
    help="Optional sample limit for a quick training smoke run.",
)
@click.option(
    "--evaluation-sample-limit",
    type=int,
    default=None,
    help="Optional sample limit for a quick evaluation smoke run.",
)
@click.option(
    "--num-workers",
    default=0,
    show_default=True,
    type=int,
    help="PyTorch DataLoader worker count.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    help="PyTorch runtime device request.",
)
@click.option(
    "--rng-seed",
    default=2026,
    show_default=True,
    type=int,
    help="Random seed used before model construction and training.",
)
def prepare_cifar10_resnet20(
    dataset_root: str,
    output_dir: str,
    download: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    train_sample_limit: int | None,
    evaluation_sample_limit: int | None,
    num_workers: int,
    device: str,
    rng_seed: int,
) -> None:
    """Prepare CIFAR-10 ResNet-20 artifacts for BFA/PBS runs."""
    try:
        output = prepare_cifar_resnet20_artifacts(
            dataset_root=dataset_root,
            output_dir=output_dir,
            download=download,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            train_sample_limit=train_sample_limit,
            evaluation_sample_limit=evaluation_sample_limit,
            num_workers=num_workers,
            device=device,
            rng_seed=rng_seed,
        )
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        RuntimeDeviceUnavailableError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"FP32 checkpoint: {output.fp32_checkpoint_path}")
    click.echo(f"Int8 checkpoint: {output.int8_checkpoint_path}")
    click.echo(f"Scale metadata: {output.scale_path}")
    click.echo(f"Resolved device: {output.device}")
    click.echo(f"Training epochs: {output.epochs}")
    for metric_name, metric_value in output.evaluation_metrics.items():
        click.echo(f"{metric_name}: {metric_value:.6g}")
