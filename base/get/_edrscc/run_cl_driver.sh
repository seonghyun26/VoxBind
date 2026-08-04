#!/bin/bash
# run_cl_driver.sh — Train GET / EGNN / EGNN+TargetDiff on CL splits and aggregate results.
#
# Usage:
#   METHOD=GET    SPLIT=cl1  GPU=5 bash _edrscc/run_cl_driver.sh
#   METHOD=EGNN   SPLIT=cl12 GPU=6 bash _edrscc/run_cl_driver.sh
#   METHOD=EGNN_TD SPLIT=cl123 GPU=7 bash _edrscc/run_cl_driver.sh
#   METHOD=EGNN   SPLIT=v2   GPU=5 bash _edrscc/run_cl_driver.sh   # v2 re-run
#
# Runs 3 seeds sequentially on the given GPU (each ~15-40 min depending on split size).
# After training, runs inference on the test pkl for each seed, then aggregates to
#   _edrscc/results_{METHOD}_{SPLIT}.json
#
# GPUs 0-3 are off-limits; use GPU in {4,5,6,7}.

set -euo pipefail
cd /home/shpark/prj-denovo/VoxBind/base/get
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate get

METHOD="${METHOD:?set METHOD=GET|EGNN|EGNN_TD}"
SPLIT="${SPLIT:?set SPLIT=v2|cl1|cl12|cl123}"
GPU="${GPU:?set GPU=4..7}"

PY=/home/shpark/.conda/envs/get/bin/python
LOG_DIR=_edrscc/logs
mkdir -p "$LOG_DIR"

# Select the test pkl (same edrscc_XX dir as the train pkl)
if [ "$SPLIT" = "v2" ]; then
    TEST_PKL="datasets/edrscc/test.pkl"
else
    TEST_PKL="datasets/edrscc_${SPLIT}/test.pkl"
fi

MLOW="$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')"
PREDS_LIST=""

for SEED in 0 1 2; do
    SEED_SUFFIX=""
    [ "$SEED" -gt 0 ] && SEED_SUFFIX="_seed${SEED}"
    CFG="_edrscc/${MLOW}_${SPLIT}${SEED_SUFFIX}.json"
    LOGF="${LOG_DIR}/train_${METHOD}_${SPLIT}_seed${SEED}.log"

    echo "[driver] Training ${METHOD}/${SPLIT} seed${SEED} on GPU${GPU} -> ${LOGF}"
    GPU=${GPU} PORT=$((9940 + SEED)) bash scripts/train/train.sh "$CFG" > "$LOGF" 2>&1

    # Find best checkpoint (lowest validation loss)
    CKPT=$(grep -oE "Validation: [0-9.]+, save path: [^ ]+\.ckpt" "$LOGF" \
           | sed -E 's/Validation: ([0-9.]+), save path: (.*)/\1 \2/' \
           | sort -n | head -1 | awk '{print $2}')
    echo "[driver] seed${SEED} best ckpt: ${CKPT}"

    PRED_FILE="_edrscc/preds_${METHOD}_${SPLIT}_seed${SEED}.jsonl"
    CUDA_VISIBLE_DEVICES=${GPU} $PY inference.py \
        --test_set "$TEST_PKL" \
        --task PDBBind \
        --ckpt "$CKPT" \
        --gpu 0 \
        --save_path "$PRED_FILE"
    PREDS_LIST="$PREDS_LIST $PRED_FILE"
done

echo "[driver] Aggregating 3-seed results..."
$PY _edrscc/aggregate_results.py --method "$METHOD" --split "$SPLIT"
echo "[driver] DONE — results at _edrscc/results_${METHOD}_${SPLIT}.json"
