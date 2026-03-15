#!/bin/bash
# Three-way sampling comparison: baseline / fine-tuned / density-conditioned.
# Run from the voxbind/ directory: bash scripts/09_poc_compare_samples.sh [mode]
#
# Usage:
#   bash scripts/09_poc_compare_samples.sh [mode]
#
# Arguments (optional):
#   mode    ccp4 | crops  (default: crops)
#             ccp4  — load CCP4 maps on-the-fly
#             crops — load precomputed crops (faster)
#
# Environment variables (optional):
#   CUDA_VISIBLE_DEVICES   GPUs to use (default: 6,7)
#   FINETUNE_CKPT          path to no-density fine-tuned ckpt (enables three-way comparison)
#                          default: exps/poc_xray/finetune_ckpt.pth.tar (used only if exists)
#   N_SAMPLES              molecules to save per model (default: 40)
#   N_CHAINS               WJS chains per iteration (default: 10)
#   WARMUP                 WJS warmup steps (default: 400)
#   STEPS                  walk steps between each jump (default: 100)
#   MAX_STEPS              total walk budget; candidates = N_CHAINS*(MAX_STEPS/STEPS) (default: 100)
#
# Prerequisites:
#   - exps/exp_sig0.9/checkpoint.pth.tar          (pretrained baseline)
#   - exps/poc_xray/overfit_ckpt.pth.tar          (density-finetuned)
#   - exps/poc_xray/finetune_ckpt.pth.tar         (no-density fine-tuned, optional)
#   - ccp4 mode : dataset/data/ccp4/              (from 06_download_xray.sh)
#   - crops mode: dataset/data/xray_crops/        (from 07_process_xray.sh)
#
# Outputs:
#   exps/poc_xray/compare/
#       baseline/samples.sdf   original model, no density
#       finetuned/samples.sdf  fine-tuned, no density  (if FINETUNE_CKPT exists)
#       xray_cond/samples.sdf  fine-tuned + X-ray density

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-crops}"
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
N_SAMPLES="${N_SAMPLES:-40}"
N_CHAINS="${N_CHAINS:-10}"
WARMUP="${WARMUP:-400}"
STEPS="${STEPS:-100}"
MAX_STEPS="${MAX_STEPS:-100}"

PRETRAINED="${PROJECT_ROOT}/exps/exp_sig0.9"
XRAY_CKPT="${PROJECT_ROOT}/exps/poc_xray/overfit_ckpt.pth.tar"
# Use FINETUNE_CKPT env var if set; otherwise auto-detect default path
DEFAULT_FT_CKPT="${PROJECT_ROOT}/exps/poc_xray/finetune_ckpt.pth.tar"
FINETUNE_CKPT="${FINETUNE_CKPT:-$DEFAULT_FT_CKPT}"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"
OUT_DIR="${PROJECT_ROOT}/exps/poc_xray/compare"

cd "$PROJECT_ROOT"

# ── Prerequisite checks ───────────────────────────────────────────────────────
if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: Baseline checkpoint not found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

if [ ! -f "$XRAY_CKPT" ]; then
    echo "ERROR: X-ray checkpoint not found at ${XRAY_CKPT}"
    echo "Run scripts/08_poc_density_overfit.sh first."
    exit 1
fi

# Build optional --finetune_ckpt flag
FINETUNE_FLAG=""
if [ -f "$FINETUNE_CKPT" ]; then
    FINETUNE_FLAG="--finetune_ckpt ${FINETUNE_CKPT}"
    echo "==> Three-way comparison  (mode=${MODE})"
    echo "    Fine-tuned : ${FINETUNE_CKPT}"
else
    echo "==> Two-way comparison  (mode=${MODE})"
    echo "    (no finetune_ckpt found at ${FINETUNE_CKPT} — skipping finetuned condition)"
    echo "    Run: NO_DENSITY=1 bash scripts/08_poc_density_overfit.sh  to create it"
fi

echo "    Baseline   : ${PRETRAINED}/checkpoint.pth.tar"
echo "    X-ray ckpt : ${XRAY_CKPT}"
echo "    Output     : ${OUT_DIR}"
echo "    GPUs       : ${GPUS}"
echo "    n_samples=${N_SAMPLES}  n_chains=${N_CHAINS}  warmup=${WARMUP}  steps=${STEPS}  max_steps=${MAX_STEPS}"
echo

case "$MODE" in
    ccp4)
        if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
            echo "Run scripts/06_download_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_compare_samples.py \
            --pretrained_path "$PRETRAINED" \
            --xray_ckpt       "$XRAY_CKPT" \
            $FINETUNE_FLAG \
            --data_dir        "$DATA_DIR" \
            --ccp4_dir        "$CCP4_DIR" \
            --out_dir         "$OUT_DIR" \
            --n_samples       "$N_SAMPLES" \
            --n_chains        "$N_CHAINS" \
            --warmup          "$WARMUP" \
            --steps           "$STEPS" \
            --max_steps       "$MAX_STEPS"
        ;;

    crops)
        if [ ! -d "$CROPS_DIR" ] || [ -z "$(ls -A "$CROPS_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed crops found in ${CROPS_DIR}"
            echo "Run scripts/07_process_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_compare_samples.py \
            --pretrained_path "$PRETRAINED" \
            --xray_ckpt       "$XRAY_CKPT" \
            $FINETUNE_FLAG \
            --data_dir        "$DATA_DIR" \
            --crops_dir       "$CROPS_DIR" \
            --out_dir         "$OUT_DIR" \
            --n_samples       "$N_SAMPLES" \
            --n_chains        "$N_CHAINS" \
            --warmup          "$WARMUP" \
            --steps           "$STEPS" \
            --max_steps       "$MAX_STEPS"
        ;;

    *)
        echo "ERROR: Unknown mode '${MODE}'. Choose: ccp4 | crops"
        exit 1
        ;;
esac

echo
echo "==> Done."
echo "    baseline/samples.sdf  → ${OUT_DIR}/baseline/samples.sdf"
if [ -f "$FINETUNE_CKPT" ]; then
    echo "    finetuned/samples.sdf → ${OUT_DIR}/finetuned/samples.sdf"
fi
echo "    xray_cond/samples.sdf → ${OUT_DIR}/xray_cond/samples.sdf"
