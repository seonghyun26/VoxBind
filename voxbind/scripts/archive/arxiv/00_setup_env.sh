#!/bin/bash
# Setup the voxbind conda environment and install the package.
# Run from the project root: bash scripts/00_setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "==> Creating conda environment from env.yaml..."
conda env create -f env.yaml

echo "==> Activating voxbind environment and installing package..."
conda run -n voxbind pip install -e .

echo "==> Done. Activate with: conda activate voxbind"
