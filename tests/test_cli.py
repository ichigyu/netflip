from __future__ import annotations

from click.testing import CliRunner

from netflip import __version__
from netflip.console import main


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


def test_cli_run_placeholder_returns_nonzero() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["run", "spec.yaml"])

    assert result.exit_code == 1
    assert "received spec: spec.yaml" in result.stderr


def test_cli_run_requires_spec_argument() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["run"])

    assert result.exit_code != 0
    assert "Missing argument 'SPEC'" in result.stderr
