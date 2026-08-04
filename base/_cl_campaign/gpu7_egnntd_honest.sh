#!/bin/bash
# GPU7 lane — EGNN+TargetDiff (v2 + 3 CL, replaces paper v2) then HonestAffinity (4 splits x 3 seeds).
set -u
GET=/home/shpark/prj-denovo/VoxBind/base/get
HA=/home/shpark/prj-denovo/VoxBind/base/honestaffinity
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/gpu7_egnntd_honest.log
ts(){ date "+%F %T"; }
cd "$GET"
echo "[$(ts)] === GPU7 lane: EGNN_TD (v2+cl) then HonestAffinity (all) ===" | tee -a "$LOG"
for SPLIT in v2 cl1 cl12 cl123; do
  echo "[$(ts)] EGNN_TD $SPLIT" | tee -a "$LOG"
  METHOD=EGNN_TD SPLIT=$SPLIT GPU=7 bash _edrscc/run_cl_driver.sh >> "$LOG" 2>&1 \
    && echo "[$(ts)]   ok EGNN_TD $SPLIT" | tee -a "$LOG" \
    || echo "[$(ts)]   FAIL EGNN_TD $SPLIT" | tee -a "$LOG"
done
echo "[$(ts)] HonestAffinity full sweep (4 splits x 3 seeds)" | tee -a "$LOG"
bash "$HA/run_full.sh" 7 >> "$LOG" 2>&1 \
  && echo "[$(ts)]   ok HonestAffinity" | tee -a "$LOG" \
  || echo "[$(ts)]   FAIL HonestAffinity" | tee -a "$LOG"
echo "[$(ts)] === GPU7 lane DONE ===" | tee -a "$LOG"
