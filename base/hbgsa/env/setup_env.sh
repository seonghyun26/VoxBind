#!/usr/bin/env bash
# Build the dedicated, isolated `hbgsa` conda env for the HBGSA baseline.
# Self-contained: does NOT touch the voxbind / base envs.
set -euo pipefail

ENV_NAME="hbgsa"
CONDA_BASE="/compuworks/anaconda3"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=== [1/3] conda create -n ${ENV_NAME} (conda-forge: pymol-open-source, rdkit, biopython, scientific stack) ==="
conda create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.11 \
    pymol-open-source \
    rdkit \
    biopython \
    numpy pandas scipy scikit-learn tqdm

echo "=== [2/3] pip install torch (cu124) + torch_geometric ==="
conda activate "${ENV_NAME}"
pip install --no-input torch --index-url https://download.pytorch.org/whl/cu124
pip install --no-input torch_geometric

echo "=== [3/3] verify imports ==="
python - <<'PY'
import importlib
mods = ["pymol", "rdkit", "Bio", "torch", "torch_geometric", "numpy", "pandas", "scipy", "sklearn"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  OK  {m:16s} {getattr(mod, '__version__', '?')}")
import torch
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
# sanity: PyMOL headless launch
import pymol
from pymol import cmd
pymol.finish_launching(['pymol', '-qc'])
print("  PyMOL headless launch: OK")
PY
echo "=== DONE: env '${ENV_NAME}' ready ==="
