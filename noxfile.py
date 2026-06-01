from __future__ import annotations

from pathlib import Path

import nox
import tomllib

nox.options.default_venv_backend = "venv"
nox.options.error_on_missing_interpreters = False
nox.options.sessions = ["tests-3.11", "coverage", "lint"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
DEV_DEPENDENCIES = tomllib.loads(Path("pyproject.toml").read_text())[
    "dependency-groups"
]["dev"]


def dev_dependency(name: str) -> str:
    return next(
        dependency
        for dependency in DEV_DEPENDENCIES
        if dependency == name
        or dependency.startswith(f"{name}>")
        or dependency.startswith(f"{name}<")
        or dependency.startswith(f"{name}=")
        or dependency.startswith(f"{name}[")
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite."""
    session.install("--editable", ".")
    session.install(dev_dependency("pytest"))
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    """Run pytest and report coverage for the netflip package."""
    session.install("--editable", ".")
    session.install(dev_dependency("coverage"), dev_dependency("pytest"))
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "report")


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run Ruff lint checks."""
    session.install(dev_dependency("ruff"))
    session.run("ruff", "check", ".")
