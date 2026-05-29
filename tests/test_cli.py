from __future__ import annotations

import pytest

from netflip.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert "netflip 0.1.0" in capsys.readouterr().out


def test_cli_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Neural-network bit-flip reliability" in capsys.readouterr().out
