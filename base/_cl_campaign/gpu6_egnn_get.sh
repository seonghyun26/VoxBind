#!/bin/bash
# GPU6 lane — EGNN (v2 + 3 CL, replaces paper v2) then GET (3 CL) via the GET repo driver.
# run_cl_driver.sh self-activates the `get` env, trains 3 seeds, infers, and aggregates to
# base/get/_edrscc/results_{METHOD}_{SPLIT}.json.
set -u
GET=/home/shpark/prj-denovo/VoxBind/base/get
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/gpu6_egnn_get.log
ts(){ date "+%F %T"; }
cd "$GET"
echo "[$(ts)] === GPU6 lane: EGNN (v2+cl) then GET (cl) ===" | tee -a "$LOG"
for SPLIT in v2 cl1 cl12 cl123; do
  echo "[$(ts)] EGNN $SPLIT" | tee -a "$LOG"
  METHOD=EGNN SPLIT=$SPLIT GPU=6 bash _edrscc/run_cl_driver.sh >> "$LOG" 2>&1 \
    && echo "[$(ts)]   ok EGNN $SPLIT" | tee -a "$LOG" \
    || echo "[$(ts)]   FAIL EGNN $SPLIT" | tee -a "$LOG"
done
for SPLIT in cl1 cl12 cl123; do
  echo "[$(ts)] GET $SPLIT" | tee -a "$LOG"
  METHOD=GET SPLIT=$SPLIT GPU=6 bash _edrscc/run_cl_driver.sh >> "$LOG" 2>&1 \
    && echo "[$(ts)]   ok GET $SPLIT" | tee -a "$LOG" \
    || echo "[$(ts)]   FAIL GET $SPLIT" | tee -a "$LOG"
done
echo "[$(ts)] === GPU6 lane DONE ===" | tee -a "$LOG"
