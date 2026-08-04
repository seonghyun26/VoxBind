#!/bin/bash
# Rebuild results.html every 5 min so the two CASF-2016 columns fill as base/_casf/*.json land.
# Exits when all 24 CASF cells (12 methods x 2 cols) are filled (or after ~3h).
cd /home/shpark/prj-denovo/VoxBind
PLOG=base/_cl_campaign/poller_casf.log
: > "$PLOG"
for i in $(seq 1 36); do
  OUT=$(python3 notebook/html/build_results.py 2>/dev/null | grep -oE "CASF [0-9]+/24")
  echo "[$(date +%F\ %H:%M:%S)] $OUT (iter $i)" >> "$PLOG"
  N=$(echo "$OUT" | grep -oE "[0-9]+/24" | cut -d/ -f1)
  [ "${N:-0}" = "24" ] && { echo "[$(date +%F\ %H:%M:%S)] ALL 24 CASF CELLS FILLED" >> "$PLOG"; break; }
  sleep 300
done
