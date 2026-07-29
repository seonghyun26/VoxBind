#!/bin/bash
# Sample comparison for 08c-trained test-split checkpoints.
# Uses exps/poc_xray/{overfit_test,finetune_test}/ and evaluates on the test split.
#
# Usage:
#   bash scripts/09c_poc_compare_test.sh [mode]
#
# Arguments (optional):
#   mode    ccp4 | crops  (default: crops)
#
# Environment variables (optional):
#   CUDA_VISIBLE_DEVICES   GPUs to use (default: 6,7)
#   N_POCKETS              number of test-set pockets to evaluate (default: 1)
#                            1  → flat output: compare_test/baseline/, compare_test/xray_cond/, ...
#                            >1 → per-pocket subdirs: compare_test/pocket_{idx:04d}/baseline/, ...
#   N_SAMPLES              molecules to save per model per pocket (default: 10)
#   N_CHAINS               WJS chains per iteration (default: 50)
#   WARMUP                 WJS warmup steps (default: 400)
#   STEPS                  walk steps between each jump (default: 100)
#   MAX_STEPS              total walk budget; candidates = N_CHAINS*(MAX_STEPS/STEPS) (default: 100)
#   THRESHOLD              denoising threshold for baseline/finetuned (default: 0.2)
#   THRESHOLD_XRAY         denoising threshold for xray_cond (default: 0.2)
#
# Prerequisites:
#   - exps/poc_xray/overfit_test/overfit_ckpt.pth.tar     (from 08c_density_overfit_test.sh)
#   - exps/poc_xray/finetune_test/finetune_ckpt.pth.tar   (from 08c_density_overfit_test.sh, optional)
#
# Outputs:
#   exps/poc_xray/compare_test/
#       baseline/samples.sdf   original model, no density
#       finetuned/samples.sdf  fine-tuned on test set, no density (if ckpt exists)
#       xray_cond/samples.sdf  fine-tuned on test set + X-ray density
#       random_density/samples.sdf  fine-tuned + random noise density (negative control)
#   When N_POCKETS > 1, results are written to per-pocket subdirs:
#       pocket_{idx:04d}/baseline/samples.sdf
#       pocket_{idx:04d}/finetuned/samples.sdf
#       pocket_{idx:04d}/xray_cond/samples.sdf
#       pocket_{idx:04d}/random_density/samples.sdf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODE="${1:-crops}"
GPUS="${CUDA_VISIBLE_DEVICES:-7}"
N_POCKETS="${N_POCKETS:-1}"
N_SAMPLES="${N_SAMPLES:-10}"
N_CHAINS="${N_CHAINS:-50}"
WARMUP="${WARMUP:-400}"
STEPS="${STEPS:-100}"
MAX_STEPS="${MAX_STEPS:-100}"
THRESHOLD="${THRESHOLD:-0.2}"
THRESHOLD_XRAY="${THRESHOLD_XRAY:-0.2}"

PRETRAINED="${PROJECT_ROOT}/exps/exp_sig0.9"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"

XRAY_CKPT="${XRAY_CKPT:-${PROJECT_ROOT}/exps/poc_xray/overfit_test/overfit_ckpt.pth.tar}"
FINETUNE_CKPT="${FINETUNE_CKPT:-${PROJECT_ROOT}/exps/poc_xray/finetune_test/finetune_ckpt.pth.tar}"
RANDOM_CKPT="${RANDOM_CKPT:-${PROJECT_ROOT}/exps/poc_xray/random_test/overfit_ckpt.pth.tar}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/exps/poc_xray/compare_test}"

cd "$PROJECT_ROOT"

# ── Prerequisite checks ───────────────────────────────────────────────────────
if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: Baseline checkpoint not found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

if [ ! -f "$XRAY_CKPT" ]; then
    echo "ERROR: X-ray checkpoint not found at ${XRAY_CKPT}"
    echo "Run: bash scripts/08c_density_overfit_test.sh"
    exit 1
fi

# Build optional --finetune_ckpt flag
FINETUNE_FLAG=""
if [ -f "$FINETUNE_CKPT" ]; then
    FINETUNE_FLAG="--finetune_ckpt ${FINETUNE_CKPT}"
    echo "==> Comparison  (mode=${MODE}, split=test)"
    echo "    Fine-tuned : ${FINETUNE_CKPT}"
else
    echo "==> Comparison  (mode=${MODE}, split=test)"
    echo "    (no finetune_ckpt found at ${FINETUNE_CKPT} — skipping finetuned condition)"
fi

# Build optional --random_ckpt flag
RANDOM_FLAG=""
if [ -f "$RANDOM_CKPT" ]; then
    RANDOM_FLAG="--random_ckpt ${RANDOM_CKPT}"
    echo "    Random ckpt: ${RANDOM_CKPT}"
else
    echo "    (no random_ckpt found at ${RANDOM_CKPT} — skipping random_trained condition)"
fi

echo "    Baseline   : ${PRETRAINED}/checkpoint.pth.tar"
echo "    X-ray ckpt : ${XRAY_CKPT}"
echo "    Output     : ${OUT_DIR}"
echo "    GPUs       : ${GPUS}  split=test"
echo "    n_pockets=${N_POCKETS}  n_samples=${N_SAMPLES}  n_chains=${N_CHAINS}  warmup=${WARMUP}  steps=${STEPS}  max_steps=${MAX_STEPS}"
echo "    threshold=${THRESHOLD}  threshold_xray=${THRESHOLD_XRAY}"
echo

case "$MODE" in
    ccp4)
        if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
            echo "Run scripts/06_download_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/poc_compare_samples.py" \
            --pretrained_path  "$PRETRAINED" \
            --xray_ckpt        "$XRAY_CKPT" \
            $FINETUNE_FLAG \
            $RANDOM_FLAG \
            --data_dir         "$DATA_DIR" \
            --ccp4_dir         "$CCP4_DIR" \
            --out_dir          "$OUT_DIR" \
            --split            test \
            --n_pockets        "$N_POCKETS" \
            --n_samples        "$N_SAMPLES" \
            --n_chains         "$N_CHAINS" \
            --warmup           "$WARMUP" \
            --steps            "$STEPS" \
            --max_steps        "$MAX_STEPS" \
            --threshold        "$THRESHOLD" \
            --threshold_xray   "$THRESHOLD_XRAY"
        ;;

    crops)
        if [ ! -d "$CROPS_DIR/test" ] || [ -z "$(ls -A "$CROPS_DIR/test" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed test crops found in ${CROPS_DIR}/test"
            echo "Run scripts/07_process_xray.sh test first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/poc_compare_samples.py" \
            --pretrained_path  "$PRETRAINED" \
            --xray_ckpt        "$XRAY_CKPT" \
            $FINETUNE_FLAG \
            $RANDOM_FLAG \
            --data_dir         "$DATA_DIR" \
            --crops_dir        "$CROPS_DIR" \
            --out_dir          "$OUT_DIR" \
            --split            test \
            --n_pockets        "$N_POCKETS" \
            --n_samples        "$N_SAMPLES" \
            --n_chains         "$N_CHAINS" \
            --warmup           "$WARMUP" \
            --steps            "$STEPS" \
            --max_steps        "$MAX_STEPS" \
            --threshold        "$THRESHOLD" \
            --threshold_xray   "$THRESHOLD_XRAY"
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
echo "    random_density/samples.sdf → ${OUT_DIR}/random_density/samples.sdf"
if [ -f "$RANDOM_CKPT" ]; then
    echo "    random_trained/samples.sdf → ${OUT_DIR}/random_trained/samples.sdf"
fi
