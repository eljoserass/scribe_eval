#!/usr/bin/env bash
# Install scribe_eval with Scribnic, prepare Scribnic's English workflow, and
# convert PriMock57. Linux and macOS (Apple Silicon).
#
# Usage: scripts/setup_scribnic.sh [CONSULTATIONS]
#   CONSULTATIONS  How many PriMock57 consultations to convert; all 57 by default.
#
# Temporary and Scribnic-specific: the harness itself knows no scribe. It expects
# Scribnic checked out next to this repository (../scribnic), as pyproject.toml
# says, and takes the English workflow from the scribnic-workflows branch.
set -euo pipefail

cd "$(dirname "$0")/.."
consultations="${1:-}"
workflow="workflows/whisper_small_bonsai_en.toml"

if ! command -v uv >/dev/null; then
  echo "uv is missing: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [ ! -d ../scribnic ]; then
  echo "Scribnic is missing: clone it next to this repository, as ../scribnic." >&2
  exit 1
fi

echo "==> Installing scribe_eval with Scribnic"
uv sync --locked --extra scribnic

if [ ! -f "$workflow" ]; then
  echo "==> Fetching the English workflow from the scribnic-workflows branch"
  git fetch --quiet origin scribnic-workflows
  mkdir -p workflows
  for file in whisper_small_bonsai_en.toml soap_en.md; do
    git show "origin/scribnic-workflows:workflows/$file" > "workflows/$file"
  done
fi

echo "==> Downloading Scribnic's engines and weights for $workflow"
uv run scribnic setup --workflow "$workflow"

echo "==> Converting PriMock57"
if [ -n "$consultations" ]; then
  uv run scribe-eval prepare primock57 --limit "$consultations"
else
  uv run scribe-eval prepare primock57
fi

echo "Done. Try: uv run scribe-eval run experiments/primock57/scribnic_smoke.toml"
