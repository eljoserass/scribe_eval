#!/usr/bin/env bash
# Install Berta AI Scribe's own environment, download Whisper small, link the
# llama-server and Bonsai model Scribnic downloads, and convert PriMock57.
# Linux and macOS (Apple Silicon).
#
# Usage: scripts/setup_berta.sh [CONSULTATIONS]
#   CONSULTATIONS  How many PriMock57 consultations to convert; all 57 by default.
#
# Environment:
#   BERTA_DIR            Berta checkout; ../berta-ai-scribe by default, cloned if
#                        missing.
#   SCRIBNIC_DIR         Scribnic checkout; ../scribnic by default.
#   BERTA_TORCH_BACKEND  uv's PyTorch backend for Berta's environment; cpu by
#                        default, so Linux doesn't download CUDA builds.
#
# Temporary and Berta-specific: the harness itself knows no scribe. Berta runs in
# its own Python 3.11 environment, never in scribe_eval's. The note model is the
# one Scribnic's English workflow uses, served by the same Prism llama-server, so
# the script asks Scribnic to download both. vendor/berta/ ends up with links to
# Berta's web-api, llama-server, and the model, which the Berta experiments use.
set -euo pipefail

cd "$(dirname "$0")/.."
consultations="${1:-}"
berta="${BERTA_DIR:-../berta-ai-scribe}"
scribnic="${SCRIBNIC_DIR:-../scribnic}"
backend="${BERTA_TORCH_BACKEND:-cpu}"
whisper_model="small"
model_file="Ternary-Bonsai-2-2B-PQ2_0.gguf"
links="vendor/berta"

for tool in uv ffmpeg git; do
  if ! command -v "$tool" >/dev/null; then
    echo "$tool is missing. On macOS: brew install $tool" >&2
    exit 1
  fi
done
if [ ! -d "$scribnic" ]; then
  echo "Scribnic is missing: clone it next to this repository, or set SCRIBNIC_DIR." >&2
  exit 1
fi

if [ ! -d "$berta" ]; then
  echo "==> Cloning Berta AI Scribe into $berta"
  git clone https://github.com/eljoserass/berta-ai-scribe.git "$berta"
fi
web_api="$(cd "$berta/web-api" && pwd)"
python="$web_api/.venv/bin/python"

echo "==> Installing Berta's environment in $web_api/.venv"
if [ ! -x "$python" ]; then
  uv venv --python 3.11 "$web_api/.venv"
fi
uv pip install --python "$python" --torch-backend "$backend" \
  -r "$web_api/requirements.txt"

echo "==> Downloading faster-whisper $whisper_model for Berta's WhisperX service"
"$python" -c "import faster_whisper, sys; faster_whisper.download_model(sys.argv[1])" \
  "$whisper_model"

if [ -n "${SCRIBNIC_CACHE:-}" ]; then
  cache="$SCRIBNIC_CACHE"
elif [ "$(uname -s)" = "Darwin" ]; then
  cache="$HOME/Library/Caches/scribnic"
else
  cache="${XDG_CACHE_HOME:-$HOME/.cache}/scribnic"
fi
release="$(sed -n 's/^release = "\(.*\)"/\1/p' \
  "$scribnic/src/scribnic/presets/runtimes/llama_cpp_prism.toml")"

# Print the first file with this name under a directory, or nothing.
first_file() {
  find "$1" -name "$2" -type f 2>/dev/null | head -n 1 || true
}

find_assets() {
  server="$(first_file "$cache/runtimes/llama-cpp/$release" llama-server)"
  model="$(first_file "$cache/models" "$model_file")"
}

find_assets
if [ -z "$server" ] || [ -z "$model" ]; then
  echo "==> Downloading Prism llama-server $release and $model_file through Scribnic"
  if [ -d "$scribnic/.venv" ]; then
    uv run --project "$scribnic" --no-sync scribnic setup --workflow bonsai_note
  else
    uv run --project "$scribnic" --no-dev scribnic setup --workflow bonsai_note
  fi
  find_assets
fi
if [ -z "$server" ] || [ -z "$model" ]; then
  echo "Scribnic's llama-server or $model_file is not in $cache." >&2
  exit 1
fi

echo "==> Linking Berta, llama-server, and the model in $links"
mkdir -p "$links"
ln -sfn "$web_api" "$links/web-api"
ln -sfn "$server" "$links/llama-server"
ln -sfn "$model" "$links/$model_file"

# Berta adds nothing to scribe_eval's environment. --inexact keeps Scribnic if
# setup_scribnic.sh installed it, and --frozen, as in CI, needs no Scribnic checkout.
echo "==> Installing scribe_eval"
uv sync --frozen --inexact --no-dev --group checks

if [ -f data/primock57/dataset.json ] && [ -z "$consultations" ]; then
  echo "==> PriMock57 is already converted"
elif [ -n "$consultations" ]; then
  echo "==> Converting PriMock57"
  uv run --no-sync scribe-eval prepare primock57 --limit "$consultations"
else
  echo "==> Converting PriMock57"
  uv run --no-sync scribe-eval prepare primock57
fi

echo "Done. Try: uv run --no-sync scribe-eval run experiments/berta_primock57_smoke.toml"
