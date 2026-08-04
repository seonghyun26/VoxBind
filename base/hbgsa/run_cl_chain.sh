#!/bin/bash
# run_cl_chain.sh — train HBGSA 3.06M on the 3 CL-filtered PDBbind splits:
#   lp_edrscc_v2_cl1   (train 2721 / val 680 / test 1166)
#   lp_edrscc_v2_cl12  (train 2643 / val 659 / test 1149)
#   lp_edrscc_v2_cl123 (train 1559 / val 410 / test  733)
# Each split CSV is the single source of truth (membership + bucket assignment).
# H-bond cache is already fully built for all CL pids — no rebuild needed.
# Run on server with /home/shpark/prj-ligand/ mounted (structure access required).
set -u

HB=/home/shpark/prj-denovo/VoxBind/base/hbgsa
PY=/home/shpark/.conda/envs/hbgsa/bin/python
SPLITS=/home/shpark/prj-denovo/VoxBind/voxbind/splits
GPU=${CUDA_VISIBLE_DEVICES:-5}          # override with e.g. CUDA_VISIBLE_DEVICES=6 bash run_cl_chain.sh

cd "$HB/src" || exit 1
ts(){ date "+%F %T"; }

echo "[$(ts)] === HBGSA CL-split retrain (GPU $GPU) ==="

# H-bond cache already built for all CL pids (verified: 0 missing / 16430 cached).
# Skip rebuild to save time. Un-comment below if adding new pids ever becomes needed:
# echo "[$(ts)] H-bond cache check (skips existing)"
# $PY hbonds.py 2>&1 | tail -3

# ── cl1 (train~2721 raw, test~1166 raw) ─────────────────────────────────────
echo "[$(ts)] [1/3] train 3.06M on lp_edrscc_v2_cl1 (seeds 0,1,2)"
CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    --split_csv "$SPLITS/lp_edrscc_v2_cl1.csv" \
    --tag edrscc_v2_cl1_3p06m \
    --seeds 0,1,2 \
    --epochs 150 \
    2>&1 | tee "$HB/logs/train_cl1_3p06m.log"

echo "[$(ts)] [1/3] DONE → results/hbgsa_results_edrscc_v2_cl1_3p06m.csv"

# ── cl12 (train~2643 raw, test~1149 raw) ────────────────────────────────────
echo "[$(ts)] [2/3] train 3.06M on lp_edrscc_v2_cl12 (seeds 0,1,2)"
CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    --split_csv "$SPLITS/lp_edrscc_v2_cl12.csv" \
    --tag edrscc_v2_cl12_3p06m \
    --seeds 0,1,2 \
    --epochs 150 \
    2>&1 | tee "$HB/logs/train_cl12_3p06m.log"

echo "[$(ts)] [2/3] DONE → results/hbgsa_results_edrscc_v2_cl12_3p06m.csv"

# ── cl123 (train~1559 raw, test~733 raw) ────────────────────────────────────
echo "[$(ts)] [3/3] train 3.06M on lp_edrscc_v2_cl123 (seeds 0,1,2)"
CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    --split_csv "$SPLITS/lp_edrscc_v2_cl123.csv" \
    --tag edrscc_v2_cl123_3p06m \
    --seeds 0,1,2 \
    --epochs 150 \
    2>&1 | tee "$HB/logs/train_cl123_3p06m.log"

echo "[$(ts)] [3/3] DONE → results/hbgsa_results_edrscc_v2_cl123_3p06m.csv"

echo "[$(ts)] === HBGSA CL-split retrain COMPLETE ==="
echo "         Aggregate with: python agg_results.py"
