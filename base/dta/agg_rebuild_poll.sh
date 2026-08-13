#!/bin/bash
# Periodically aggregate DTA preds -> results json, then rebuild results.html. ~6h.
for i in $(seq 1 45); do
  ( cd /home/shpark/prj-denovo/VoxBind/base/dta && python3 aggregate_dta.py >/dev/null 2>&1 )
  ( cd /home/shpark/prj-denovo/VoxBind/notebook/html && python3 build_results.py >/dev/null 2>&1 )
  sleep 480
done
echo "poll done"
