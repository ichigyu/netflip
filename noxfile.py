from __future__ import annotations

import nox

nox.options.default_venv_backend = "venv"
nox.options.error_on_missing_interpreters = False
nox.options.sessions = ["tests-3.11", "coverage", "lint"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite."""
    session.install("--editable", ".")
    session.install("pytest>=8,<9")
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    """Run pytest and report coverage for the netflip package."""
    session.install("--editable", ".")
    session.install("coverage[toml]>=7,<8", "pytest>=8,<9")
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "report")


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run Ruff lint checks."""
    session.install("ruff>=0.4,<1")
    session.run("ruff", "check", ".")
