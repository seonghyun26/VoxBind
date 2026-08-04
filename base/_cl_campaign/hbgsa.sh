#!/bin/bash
# HBGSA lane — 3.06M, 3 CL splits x 3 seeds, via --split_csv (CL manifest = bucket source).
# Structures resolved from the local VoxBind copy (config.py STRUCT_BASES fixed). Shares GPU 6
# (HBGSA 3.06M is tiny, ~1-2GB). H-bond cache already covers all pids.
set -u
HB=/home/shpark/prj-denovo/VoxBind/base/hbgsa
PY=/home/shpark/.conda/envs/hbgsa/bin/python
SPLITS=/home/shpark/prj-denovo/VoxBind/voxbind/splits
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/hbgsa.log
ts(){ date "+%F %T"; }
cd "$HB/src"
echo "[$(ts)] === HBGSA CL (3 splits x 3 seeds) on GPU 6 ===" | tee -a "$LOG"
for cl in cl1 cl12 cl123; do
  echo "[$(ts)] HBGSA $cl" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=6 $PY train.py --split_csv "$SPLITS/lp_edrscc_v2_${cl}.csv" \
    --tag "edrscc_v2_${cl}_3p06m" --seeds 0,1,2 --epochs 150 \
    > "$HB/logs/train_${cl}_3p06m.log" 2>&1 \
    && echo "[$(ts)]   ok $cl" | tee -a "$LOG" \
    || echo "[$(ts)]   FAIL HBGSA $cl (see logs/train_${cl}_3p06m.log)" | tee -a "$LOG"
done
cd "$HB"
$PY agg_results.py >> "$LOG" 2>&1 && echo "[$(ts)] aggregated" | tee -a "$LOG" || echo "[$(ts)] FAIL agg" | tee -a "$LOG"
echo "[$(ts)] === HBGSA lane DONE ===" | tee -a "$LOG"
