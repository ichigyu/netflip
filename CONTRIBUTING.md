# Contributing

NetFlip uses GitHub issues for work items and pull requests for reviewed changes.

## Before Starting

- Read `CONTEXT.md` for project terminology.
- Read relevant ADRs in `docs/adr/`.
- Keep work scoped to a GitHub issue.
- Create a topic branch from `main`.

## Branches

Use short topic branches:

```text
feature/<short-name>
fix/<short-name>
chore/<short-name>
```

## Commits

Prefer concise conventional-style commits:

```text
feat: add int8 codec
fix: preserve bit index ordering
test: cover trace writer
docs: record quantization decision
```

## Verification

Install dependencies with uv:

```bash
uv sync --all-groups
```

Run the relevant checks before opening a pull request:

```bash
uv run pytest
uv run pytest --xdoctest src/netflip
uv run coverage run -m pytest
uv run coverage report
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run sphinx-build -W -b html docs docs/_build/html
uv run python -m build
uv run twine check dist/*
uv run nox
```

## Pull Requests

- Link the related GitHub issue.
- Summarize the behavior change.
- List verification commands and results.
- Include screenshots or recordings only when UI or visual artifacts are affected.
- Wait for review and passing checks before merge.

## Documentation

- Update `CONTEXT.md` when a domain term is resolved.
- Add an ADR in `docs/adr/` only for decisions that are hard to reverse, surprising without context, and based on a real trade-off.
- Keep agent-local workflow details out of public docs unless they are relevant to project contributors.
