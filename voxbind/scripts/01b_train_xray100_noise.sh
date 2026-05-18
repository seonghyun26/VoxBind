#!/bin/bash
# 01b_train_xray100_noise.sh — Smoothed-Gaussian-noise control for the X-ray
# density conditioning A/B test.
#
# Trains VoxBind from scratch on the SAME 100-train + 100-val sample subset as
# exps/260515_voxbind_100ep_density, but with the real 2Fo-Fc density crops
# replaced by structureless smoothed-Gaussian-noise crops produced by
# dataset/00c_make_noise_density.py. The noise reproduces the real density's
# spatial texture (smoothness) but carries NO ligand/pocket information.
#
# Every hyperparameter mirrors 260515_voxbind_100ep_density exactly:
#   8000 epochs, bsz=4 per-rank x 4 ranks (effective 16), lr=1e-5, wd=1e-2,
#   smooth_sigma=0.9, seed=42, aug=true, with_density=true.
# The ONLY difference vs the density run is dset.crops_dir.
#
# Usage:
#   nohup bash scripts/01b_train_xray100_noise.sh \
#       > log/260516_voxbind_100ep_noise.log 2>&1 &
#
# Env vars:
#   GPUS    comma-separated GPU ids   (default: 4,5,6,7)
#   NPROC   DDP ranks                 (default: 4)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOXBIND_DIR="$(dirname "$SCRIPT_DIR")"   # .../VoxBind/voxbind  (where train_ddp.py lives)
cd "$VOXBIND_DIR"

GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-4}"
DATA_DIR="${VOXBIND_DIR}/dataset/data"
CROPS_DIR="${DATA_DIR}/xray_crops_noise"
EXP_NAME="260516_voxbind_100ep_noise"
OUT_DIR="${VOXBIND_DIR}/exps/${EXP_NAME}"

if [ ! -d "${CROPS_DIR}/train" ]; then
    echo "ERROR: missing ${CROPS_DIR}/train — run dataset/00c_make_noise_density.py first"
    exit 1
fi

echo "==> noise-control training (smoothed-Gaussian-noise density)"
echo "    GPUs=${GPUS}  ranks=${NPROC}"
echo "    crops=${CROPS_DIR}"
echo "    out=${OUT_DIR}"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="$NPROC" \
    train_ddp.py \
    dset=crossdocked_xray \
    dset.data_dir="${DATA_DIR}" \
    dset.crops_dir="${CROPS_DIR}" \
    dset.subset_n=100 \
    dset.subset_xray_only=true \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    num_epochs=8000 \
    bsz=4 \
    seed=42 \
    wjs.n_targets=0 \
    model.with_density=true \
    exp_name="${EXP_NAME}" \
    output_dir="${OUT_DIR}"

echo "==> noise-control training done."
