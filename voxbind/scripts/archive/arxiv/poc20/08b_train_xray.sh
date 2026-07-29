#!/bin/bash
# Full-dataset X-ray density-conditioned VoxBind finetuning.
# Run from the voxbind/ directory: bash scripts/08b_train_xray.sh [mode]
#
# Usage:
#   bash scripts/08b_train_xray.sh [mode]
#
# Arguments (optional):
#   mode    ccp4 | crops  (default: crops)
#             ccp4  — load CCP4 maps on-the-fly
#             crops — load precomputed crops (faster)
#
# Environment variables (optional):
#   CUDA_VISIBLE_DEVICES   GPUs to use (default: 6,7)
#   EPOCHS                 number of training epochs (default: 30)
#   BSZ                    batch size (default: 32)
#   LR                     LR for density branch (default: 1e-3)
#   LR_BACKBONE            LR for pretrained backbone (default: 1e-4)
#   NUM_WORKERS            dataloader workers (default: 4)
#   OUT_DIR                output directory (default: exps/xray_full)
#   EXP_NAME               W&B run name (default: xray_full)
#
# Prerequisites:
#   ccp4  mode: bash scripts/06_download_xray.sh
#   crops mode: bash scripts/06_download_xray.sh && bash scripts/07_process_xray.sh
#
# Outputs:
#   exps/xray_full/checkpoint.pth.tar        (every epoch)
#   exps/xray_full/best_checkpoint.pth.tar   (best val loss)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODE="${1:-crops}"
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
EPOCHS="${EPOCHS:-30}"
BSZ="${BSZ:-32}"
LR="${LR:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/exps/xray_full}"
EXP_NAME="${EXP_NAME:-xray_full}"

PRETRAINED="${PROJECT_ROOT}/exps/exp_sig0.9"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"

cd "$PROJECT_ROOT"

if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: Baseline checkpoint not found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

echo "==> Full-dataset X-ray finetuning  (mode=${MODE})"
echo "    Pretrained   : ${PRETRAINED}"
echo "    Output       : ${OUT_DIR}"
echo "    GPUs         : ${GPUS}"
echo "    epochs=${EPOCHS}  bsz=${BSZ}  lr=${LR}  lr_backbone=${LR_BACKBONE}  workers=${NUM_WORKERS}"
echo

case "$MODE" in
    ccp4)
        if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
            echo "Run scripts/06_download_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/08b_train_xray.py" \
            --pretrained_path "$PRETRAINED" \
            --data_dir        "$DATA_DIR" \
            --ccp4_dir        "$CCP4_DIR" \
            --out_dir         "$OUT_DIR" \
            --epochs          "$EPOCHS" \
            --bsz             "$BSZ" \
            --lr              "$LR" \
            --lr_backbone     "$LR_BACKBONE" \
            --num_workers     "$NUM_WORKERS" \
            --exp_name        "$EXP_NAME"
        ;;

    crops)
        if [ ! -d "$CROPS_DIR" ] || [ -z "$(ls -A "$CROPS_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed crops found in ${CROPS_DIR}"
            echo "Run scripts/07_process_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/08b_train_xray.py" \
            --pretrained_path "$PRETRAINED" \
            --data_dir        "$DATA_DIR" \
            --crops_dir       "$CROPS_DIR" \
            --out_dir         "$OUT_DIR" \
            --epochs          "$EPOCHS" \
            --bsz             "$BSZ" \
            --lr              "$LR" \
            --lr_backbone     "$LR_BACKBONE" \
            --num_workers     "$NUM_WORKERS" \
            --exp_name        "$EXP_NAME"
        ;;

    *)
        echo "ERROR: Unknown mode '${MODE}'. Choose: ccp4 | crops"
        exit 1
        ;;
esac

echo
echo "==> Done. Checkpoints saved to ${OUT_DIR}"
