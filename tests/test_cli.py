from __future__ import annotations

import pytest

from netflip import __version__
from netflip.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert f"netflip {__version__}" in capsys.readouterr().out


def test_cli_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Neural-network bit-flip reliability" in capsys.readouterr().out


def test_cli_run_placeholder_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["run", "spec.yaml"]) == 1
    assert "received spec: spec.yaml" in capsys.readouterr().err
