"""Console interface for NetFlip."""

from __future__ import annotations

import click

from netflip import __version__
from netflip.run import ExperimentRunError, execute_experiment_run


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
    click.echo(f"Resolved device: {output.device}")
    click.echo(f"Committed flip count: {output.flip_count}")
    click.echo(f"Stopped because: {output.stopped_because}")
