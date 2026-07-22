#!/usr/bin/env bash
# Recreate the voxbind conda env after a container restart wiped /opt/conda/envs/voxbind.
# Pinned to the versions captured by wandb from the last working run (run-20260617_020449).
# Box has NO C compiler -> install training-relevant deps from prebuilt wheels only; skip the
# plotting/Qt/PDF transitive cruft (pycairo/gmpy2/reportlab/kaleido/PySide6) that would source-build
# and isn't needed for training. openbabel comes from conda-forge (binary). Verify with a smoke test.
set -uo pipefail
CONDA=/opt/conda/bin/conda
ENVDIR=/opt/conda/envs/voxbind
PIP=$ENVDIR/bin/pip
PY=$ENVDIR/bin/python
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] (1/4) conda create voxbind: python=3.12.13 + openbabel (conda-forge)"
$CONDA create -n voxbind -c conda-forge python=3.12.13 openbabel -y || { echo "FAILED: conda create"; exit 1; }

echo "[$(ts)] (2/4) pip install torch (cu124 wheel) first to pin the CUDA stack"
$PIP install --no-input "torch==2.5.1" || { echo "FAILED: torch"; exit 1; }

echo "[$(ts)] (3/4) pip install pinned training deps"
$PIP install --no-input \
  "torchmetrics==1.9.0" \
  "numpy==2.4.6" \
  "scipy==1.17.1" \
  "scikit-image==0.26.0" \
  "pandas==2.2.3" \
  "rdkit==2025.3.6" \
  "gemmi==0.7.5" \
  "wandb==0.27.2" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "tqdm==4.68.2" \
  "matplotlib==3.10.9" \
  "seaborn==0.13.2" || { echo "FAILED: core deps"; exit 1; }
# pyuul last so it can't drag torch/numpy to a different version
$PIP install --no-input "pyuul==0.4.2" || { echo "FAILED: pyuul"; exit 1; }

echo "[$(ts)] (3b/4) pip install -e . (voxbind local package)"
$PIP install --no-input -e /home1/irteam/VoxBind || { echo "FAILED: voxbind -e"; exit 1; }

echo "[$(ts)] (4/4) smoke test: torch+cuda, then voxbind training import surface"
$PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
import numpy, scipy, skimage, pandas, rdkit, gemmi, wandb, hydra, omegaconf, torchmetrics, tqdm, pyuul
print("base imports OK; numpy", numpy.__version__)
# exact import surface of train_ddp.py
from voxbind.dataset import create_dataloaders
from voxbind.metrics import create_metrics_for_training
from voxbind.models import create_model
from voxbind.models.adamw import AdamW
from voxbind.models.ema import ModelEma
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.voxelizer import Voxelizer
print("VOXBIND TRAIN IMPORTS OK")
PY
RC=$?
echo "[$(ts)] smoke test exit=$RC"
[ "$RC" -eq 0 ] && echo "[$(ts)] ENV REBUILD COMPLETE" || echo "[$(ts)] ENV REBUILD: smoke test FAILED (rc=$RC)"
exit $RC
