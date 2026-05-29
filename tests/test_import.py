from __future__ import annotations

import netflip


def test_package_imports() -> None:
    assert isinstance(netflip.__version__, str)
    assert netflip.__version__
