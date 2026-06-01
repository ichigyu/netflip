Development
===========

NetFlip uses uv for dependency management and Nox for repeatable local and CI
checks.

Setup
-----

Install uv, then create the local environment and install all dependency groups:

.. code-block:: bash

   uv sync --all-groups

Common Checks
-------------

Run the unit tests:

.. code-block:: bash

   uv run pytest

Run executable examples:

.. code-block:: bash

   uv run pytest --xdoctest src/netflip

Run lint, format, type, docs, and release checks:

.. code-block:: bash

   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run sphinx-build -W -b html docs docs/_build/html
   uv run python -m build
   uv run twine check dist/*

Run the complete Nox workflow:

.. code-block:: bash

   uv run nox
