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
#   TRAIN_N                number of training samples used in fine-tuning: 1 | 20 (default: 1)
#                            1  → exps/poc_xray/{overfit,finetune}_ckpt.pth.tar
#                            20 → exps/poc_xray/{overfit,finetune}_20/{overfit,finetune}_ckpt.pth.tar
#   XRAY_CKPT              override path to density-conditioned ckpt (auto-set from TRAIN_N if unset)
#   FINETUNE_CKPT          override path to no-density fine-tuned ckpt (auto-set from TRAIN_N if unset)
#                          set to "" to disable the finetuned condition
#   N_POCKETS              number of X-ray available pockets to evaluate (default: 1)
#                            1  → flat output: compare/baseline/, compare/xray_cond/, ...
#                            >1 → per-pocket subdirs: compare/pocket_{idx:04d}/baseline/, ...
#   N_SAMPLES              molecules to save per model per pocket (default: 10)
#   N_CHAINS               WJS chains per iteration (default: 50)
#   WARMUP                 WJS warmup steps (default: 400)
#   STEPS                  walk steps between each jump (default: 100)
#   MAX_STEPS              total walk budget; candidates = N_CHAINS*(MAX_STEPS/STEPS) (default: 100)
#                          increase MAX_STEPS or N_CHAINS to reduce duplicate SMILES
#
# Prerequisites (TRAIN_N=1):
#   - exps/poc_xray/overfit_ckpt.pth.tar          (from 08_poc_density_overfit.sh)
#   - exps/poc_xray/finetune_ckpt.pth.tar         (from NO_DENSITY=1 08_poc_density_overfit.sh, optional)
# Prerequisites (TRAIN_N=20):
#   - exps/poc_xray/overfit_20/overfit_ckpt.pth.tar    (from 08_density_overfit_20.sh)
#   - exps/poc_xray/finetune_20/finetune_ckpt.pth.tar  (from 08_density_overfit_20.sh, optional)
#
# Outputs:
#   exps/poc_xray/compare/      (TRAIN_N=1, N_POCKETS=1)
#   exps/poc_xray/compare_20/   (TRAIN_N=20, N_POCKETS=1)
#       baseline/samples.sdf   original model, no density
#       finetuned/samples.sdf  fine-tuned, no density  (if FINETUNE_CKPT exists)
#       xray_cond/samples.sdf  fine-tuned + X-ray density
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
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
TRAIN_N="${TRAIN_N:-1}"
# Auto-set SPLIT from TRAIN_N: test checkpoints → test split, otherwise train
if [ -z "${SPLIT:-}" ]; then
    [ "$TRAIN_N" = "test" ] && SPLIT="test" || SPLIT="train"
fi
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

# ── Resolve checkpoint paths and output dir from TRAIN_N ──────────────────────
case "$TRAIN_N" in
    1)
        DEFAULT_XRAY_CKPT="${PROJECT_ROOT}/exps/poc_xray/overfit_ckpt.pth.tar"
        DEFAULT_FT_CKPT="${PROJECT_ROOT}/exps/poc_xray/finetune_ckpt.pth.tar"
        DEFAULT_OUT_DIR="${PROJECT_ROOT}/exps/poc_xray/compare"
        TRAIN_HINT="bash scripts/08_poc_density_overfit.sh"
        ;;
    20)
        DEFAULT_XRAY_CKPT="${PROJECT_ROOT}/exps/poc_xray/overfit_20/overfit_ckpt.pth.tar"
        DEFAULT_FT_CKPT="${PROJECT_ROOT}/exps/poc_xray/finetune_20/finetune_ckpt.pth.tar"
        DEFAULT_OUT_DIR="${PROJECT_ROOT}/exps/poc_xray/compare_20"
        TRAIN_HINT="bash scripts/08_density_overfit_20.sh"
        ;;
    test)
        DEFAULT_XRAY_CKPT="${PROJECT_ROOT}/exps/poc_xray/overfit_test/overfit_ckpt.pth.tar"
        DEFAULT_FT_CKPT="${PROJECT_ROOT}/exps/poc_xray/finetune_test/finetune_ckpt.pth.tar"
        DEFAULT_OUT_DIR="${PROJECT_ROOT}/exps/poc_xray/compare_test"
        TRAIN_HINT="bash scripts/08c_density_overfit_test.sh"
        ;;
    *)
        echo "ERROR: TRAIN_N must be 1 | 20 | test (got '${TRAIN_N}')"
        exit 1
        ;;
esac

XRAY_CKPT="${XRAY_CKPT:-$DEFAULT_XRAY_CKPT}"
FINETUNE_CKPT="${FINETUNE_CKPT:-$DEFAULT_FT_CKPT}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"

cd "$PROJECT_ROOT"

# ── Prerequisite checks ───────────────────────────────────────────────────────
if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: Baseline checkpoint not found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

if [ ! -f "$XRAY_CKPT" ]; then
    echo "ERROR: X-ray checkpoint not found at ${XRAY_CKPT}"
    echo "Run: ${TRAIN_HINT}"
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
    echo "    Run: ${TRAIN_HINT}  to create it"
fi

echo "    Baseline   : ${PRETRAINED}/checkpoint.pth.tar"
echo "    X-ray ckpt : ${XRAY_CKPT}"
echo "    Output     : ${OUT_DIR}"
echo "    GPUs       : ${GPUS}  train_n=${TRAIN_N}"
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
            --data_dir         "$DATA_DIR" \
            --ccp4_dir         "$CCP4_DIR" \
            --out_dir          "$OUT_DIR" \
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
        if [ ! -d "$CROPS_DIR" ] || [ -z "$(ls -A "$CROPS_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed crops found in ${CROPS_DIR}"
            echo "Run scripts/07_process_xray.sh first."
            exit 1
        fi
        CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/poc_compare_samples.py" \
            --pretrained_path  "$PRETRAINED" \
            --xray_ckpt        "$XRAY_CKPT" \
            $FINETUNE_FLAG \
            --data_dir         "$DATA_DIR" \
            --crops_dir        "$CROPS_DIR" \
            --out_dir          "$OUT_DIR" \
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
