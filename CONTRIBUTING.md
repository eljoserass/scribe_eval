# Contributing

[AGENTS.md](AGENTS.md) is the full contribution standard: code style, module
boundaries, result contracts, and data handling. This page covers the workflow.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

See the [README](README.md#usage) for the Scribnic checkout and the optional extras.

## Checks

Run these before each commit:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m unittest discover -t . -s tests
```

Unit tests must not download weights or data, need a GPU, or contact inference
services. Keep live model checks opt-in.

## Branches and pull requests

- Never commit to `main`. Branch from `main`, push, and open a pull request.
- Prefix the branch with the kind of change: `feat/`, `fix/`, `refactor/`, `docs/`,
  `ci/` for workflows, or `chore/` for project files and tooling. Commit messages
  use the same prefix, as in `fix: ...`.
- Keep a branch to one topic. The commit and pull request say what changed and why.
- A change that alters behavior updates its docstrings and the README or `docs/`
  page that describes it.

## Data

Datasets, runs, and recordings stay out of git. Do not send dataset content to
external services without an explicit decision. Clinical data stays local; a
remote judge is only for public or synthetic data. Never paste clinical or patient
data into issues or pull requests.
