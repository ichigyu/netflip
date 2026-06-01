"""Console interface for NetFlip."""

from __future__ import annotations

import click

from netflip import __version__


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
    raise click.ClickException(
        "The 'run' command is planned but not implemented yet "
        f"(received spec: {spec})."
    )
