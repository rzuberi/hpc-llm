#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/env/llm_ollama.yml"
ENV_NAME="llm_ollama"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniforge3/condabin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/mambaforge/condabin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/miniconda3/bin/conda"; do
    if [ -x "$candidate" ]; then
      CONDA_CMD="$candidate"
      break
    fi
  done
else
  CONDA_CMD="$(command -v conda)"
fi

if [ -z "${CONDA_CMD:-}" ]; then
  echo "ERROR: conda command not found in PATH or common install locations." >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: missing environment file: $ENV_FILE" >&2
  exit 1
fi

if "$CONDA_CMD" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Updating conda env '$ENV_NAME' from $ENV_FILE"
  "$CONDA_CMD" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating conda env '$ENV_NAME' from $ENV_FILE"
  "$CONDA_CMD" env create -f "$ENV_FILE"
fi

"$CONDA_CMD" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$CONDA_CMD" run -n "$ENV_NAME" python -m pip install -e "$PROJECT_ROOT"

chmod +x "$PROJECT_ROOT/bin/llm"

echo
echo "Environment ready."
echo "Activate: conda activate $ENV_NAME"
echo "Run CLI:  $PROJECT_ROOT/bin/llm ask \"hello\" --model llama3.1:8b --no-system"
