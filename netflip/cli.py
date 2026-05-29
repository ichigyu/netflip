"""Command-line interface for NetFlip."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from netflip import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the NetFlip command-line parser."""
    parser = argparse.ArgumentParser(
        prog="netflip",
        description="Neural-network bit-flip reliability evaluation framework.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment spec.",
    )
    run_parser.add_argument(
        "spec",
        help="Path to a YAML experiment spec.",
    )
    run_parser.set_defaults(func=_run_placeholder)

    return parser


def _run_placeholder(args: argparse.Namespace) -> int:
    raise SystemExit(
        "The 'run' command is planned but not implemented yet "
        f"(received spec: {args.spec})."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NetFlip CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)
