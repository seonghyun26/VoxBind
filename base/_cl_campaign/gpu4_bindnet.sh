#!/bin/bash
# GPU4 lane — BindNet (trained from scratch) on the 3 CL splits, 3 seeds each.
set -u
BN=/home/shpark/prj-denovo/VoxBind/base/bindnet
PY=/home/shpark/.conda/envs/bindnet/bin/python
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/gpu4_bindnet.log
ts(){ date "+%F %T"; }
cd "$BN"
echo "[$(ts)] === GPU4 lane: BindNet CL (9 runs) ===" | tee -a "$LOG"
for SPLIT in cl1 cl12 cl123; do
  for SEED in 0 1 2; do
    PORT=$((10120 + SEED))
    echo "[$(ts)] BindNet $SPLIT seed$SEED (port $PORT)" | tee -a "$LOG"
    GPU=4 SEED=$SEED SPLIT=$SPLIT LBA_DATA=_edrscc/data/edrscc_${SPLIT}/lba PORT=$PORT \
      bash _edrscc/train_bindnet.sh > _edrscc/logs/train_${SPLIT}_seed${SEED}.log 2>&1 \
      && echo "[$(ts)]   ok $SPLIT seed$SEED" | tee -a "$LOG" \
      || echo "[$(ts)]   FAIL BindNet $SPLIT seed$SEED" | tee -a "$LOG"
  done
  $PY _edrscc/aggregate_results.py --split lp_edrscc_v2_${SPLIT} >> "$LOG" 2>&1 \
    && echo "[$(ts)] aggregated $SPLIT" | tee -a "$LOG" \
    || echo "[$(ts)] FAIL agg $SPLIT" | tee -a "$LOG"
done
echo "[$(ts)] === GPU4 lane DONE ===" | tee -a "$LOG"
