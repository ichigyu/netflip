from __future__ import annotations

import shutil
from pathlib import Path

import nox

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.error_on_missing_interpreters = False
nox.options.sessions = [
    "tests",
    "doctests",
    "coverage",
    "lint",
    "format",
    "typecheck",
    "docs",
    "build",
]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
DEPENDENCY_GROUPS = tomllib.loads(Path("pyproject.toml").read_text())[
    "dependency-groups"
]


def group_dependencies(*groups: str) -> list[str]:
    dependencies: list[str] = []
    for group in groups:
        dependencies.extend(DEPENDENCY_GROUPS[group])
    return dependencies


def install_package(
    session: nox.Session,
    *groups: str,
    extras: tuple[str, ...] = (),
) -> None:
    session.install(*group_dependencies(*groups))
    package = ".[{}]".format(",".join(extras)) if extras else "."
    session.install("--editable", package)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite."""
    install_package(session, "test")
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def doctests(session: nox.Session) -> None:
    """Run executable documentation examples with xdoctest."""
    install_package(session, "test")
    session.run("pytest", "--xdoctest", "src/netflip", *session.posargs)


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    """Run pytest and write terminal and XML coverage reports."""
    install_package(session, "test", extras=("benchmark",))
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "xml")
    session.run("coverage", "report")


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run Ruff lint checks."""
    session.install(*group_dependencies("dev"))
    session.run("ruff", "check", ".")


@nox.session(python="3.11")
def format(session: nox.Session) -> None:
    """Check Ruff formatting."""
    session.install(*group_dependencies("dev"))
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run Pyright static type checks."""
    install_package(session, "typing", "test")
    session.run("pyright")


@nox.session(python="3.11")
def docs(session: nox.Session) -> None:
    """Build Sphinx documentation with warnings treated as errors."""
    install_package(session, "docs")
    session.run("sphinx-build", "-W", "-b", "html", "docs", "docs/_build/html")


@nox.session(python="3.11")
def build(session: nox.Session) -> None:
    """Build package artifacts."""
    session.install(*group_dependencies("build"))
    shutil.rmtree("dist", ignore_errors=True)
    session.run("python", "-m", "build")
