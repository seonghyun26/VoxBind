#!/bin/bash
# PoC: full fine-tune VoxBind on a single X-ray sample.
# Run from the voxbind/ directory: bash scripts/08_poc_density_overfit.sh [mode]
#
# Usage:
#   bash scripts/08_poc_density_overfit.sh [mode]
#
# Arguments (optional):
#   mode    ccp4 | crops  (default: crops)
#             ccp4  — load CCP4 maps on-the-fly
#             crops — load precomputed crops (faster)
#
# Environment variables (optional):
#   NO_DENSITY      set to 1 to fine-tune without density conditioning (default: 0)
#                     0 → exps/poc_xray/overfit_ckpt.pth.tar  (density-conditioned)
#                     1 → exps/poc_xray/finetune_ckpt.pth.tar (no density)
#   N_TRAIN_SAMPLES number of samples to overfit on (default: 1)
#   NORMALIZE       set to 0 to skip per-sample z-score normalization (default: 1)
#                     use 0 when crops_dir already contains pre-normalized crops
#                     (output of scripts/05b_renormalize_crops.py)
#   ITERS           training iterations (default: 2000)
#   LR              LR for density_encoder+density_proj (default: 1e-3)
#   LR_BACKBONE     LR for pretrained backbone (default: 1e-4)
#
# Prerequisites:
#   ccp4  mode: bash scripts/06_download_xray.sh
#   crops mode: bash scripts/06_download_xray.sh && bash scripts/07_process_xray.sh
#
# Outputs (density mode, default):
#   exps/poc_xray/overfit_ckpt.pth.tar
# Outputs (no-density mode, NO_DENSITY=1):
#   exps/poc_xray/finetune_ckpt.pth.tar

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-crops}"
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
NO_DENSITY="${NO_DENSITY:-0}"
PRETRAINED="${PROJECT_ROOT}/exps/exp_sig0.9"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-1}"
NORMALIZE="${NORMALIZE:-1}"
ITERS="${ITERS:-2000}"
LR="${LR:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-4}"

# Auto-set output path based on mode
if [ "$NO_DENSITY" = "1" ]; then
    OUT="${PROJECT_ROOT}/exps/poc_xray/finetune_ckpt.pth.tar"
    NO_DENSITY_FLAG="--no_density"
    MODE_LABEL="no-density fine-tune"
else
    OUT="${PROJECT_ROOT}/exps/poc_xray/overfit_ckpt.pth.tar"
    NO_DENSITY_FLAG=""
    MODE_LABEL="density-conditioned fine-tune"
fi

cd "$PROJECT_ROOT"

if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: No checkpoint found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

echo "==> PoC fine-tune  (${MODE_LABEL}, data=${MODE})"
echo "    Pretrained   : ${PRETRAINED}"
echo "    Output       : ${OUT}"
echo "    GPUs         : ${GPUS}"
echo "    iters=${ITERS}  n_train_samples=${N_TRAIN_SAMPLES}  normalize=${NORMALIZE}  lr=${LR}  lr_backbone=${LR_BACKBONE}"
echo

case "$MODE" in
    ccp4)
        if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
            echo "Run scripts/06_download_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_density_overfit.py \
            --pretrained_path "$PRETRAINED" \
            --data_dir        "$DATA_DIR" \
            --ccp4_dir        "$CCP4_DIR" \
            --n_samples       "$N_TRAIN_SAMPLES" \
            --normalize       "$NORMALIZE" \
            --iters           "$ITERS" \
            --lr              "$LR" \
            --lr_backbone     "$LR_BACKBONE" \
            $NO_DENSITY_FLAG
        ;;

    crops)
        if [ ! -d "$CROPS_DIR" ] || [ -z "$(ls -A "$CROPS_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed crops found in ${CROPS_DIR}"
            echo "Run scripts/07_process_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_density_overfit.py \
            --pretrained_path "$PRETRAINED" \
            --data_dir        "$DATA_DIR" \
            --crops_dir       "$CROPS_DIR" \
            --n_samples       "$N_TRAIN_SAMPLES" \
            --normalize       "$NORMALIZE" \
            --iters           "$ITERS" \
            --lr              "$LR" \
            --lr_backbone     "$LR_BACKBONE" \
            $NO_DENSITY_FLAG
        ;;

    *)
        echo "ERROR: Unknown mode '${MODE}'. Choose: ccp4 | crops"
        exit 1
        ;;
esac

echo
echo "==> Done. Checkpoint saved to ${OUT}"
