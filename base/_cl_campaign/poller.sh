#!/bin/bash
# Rebuild results.html every 5 min so cells fill as campaign result JSONs land. Exits when all
# 48 method×split cells are filled (or after ~12h safety). Rebuild-only — launches no training.
cd /home/shpark/prj-denovo/VoxBind
PLOG=base/_cl_campaign/poller.log
: > "$PLOG"
for i in $(seq 1 145); do
  OUT=$(python3 notebook/html/build_results.py 2>/dev/null | grep -oE "filled [0-9]+/48")
  echo "[$(date +%F\ %H:%M:%S)] $OUT  (iter $i)" >> "$PLOG"
  N=$(echo "$OUT" | grep -oE "[0-9]+/48" | cut -d/ -f1)
  FAILS=$(grep -c FAIL base/_cl_campaign/gpu*.log base/_cl_campaign/hbgsa.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  [ "${FAILS:-0}" -gt 0 ] && echo "[$(date +%H:%M:%S)]   note: $FAILS FAIL marker(s) in lane logs" >> "$PLOG"
  [ "${N:-0}" = "48" ] && { echo "[$(date +%F\ %H:%M:%S)] ALL 48 CELLS FILLED — campaign complete" >> "$PLOG"; break; }
  sleep 300
done
