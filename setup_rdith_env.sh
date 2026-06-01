#!/usr/bin/env bash
set -euo pipefail

# Create a Python environment for roi_selection.
# Prefer conda if available, otherwise use venv.

ENV_NAME="rdith"
REQUIREMENTS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

if command -v conda >/dev/null 2>&1; then
  echo "Using conda to create environment: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.10 -y
  echo "Activating conda environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  echo "Installing requirements..."
  pip install --upgrade pip
  pip install -r "$REQUIREMENTS_FILE"
  echo "Done. Activate with: conda activate $ENV_NAME"
else
  echo "Conda not found. Falling back to Python venv."
  VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$REQUIREMENTS_FILE"
  echo "Done. Activate with: source $VENV_DIR/bin/activate"
fi
