#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/dta/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/dta
for MODEL in deepdta moltrans; do
  for SPLIT in v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
    for SEED in 0 1 2 3 4; do
      echo "===== $MODEL $SPLIT seed$SEED ====="
      $PY run_deeppurpose.py --model "$MODEL" --seed "$SEED" --split "$SPLIT" || echo "FAILED $MODEL $SPLIT s$SEED"
    done
  done
done
echo "=== aggregate DTA preds -> json ==="
$PY aggregate_dta.py || echo "agg FAILED"
echo "ALL DTA DONE"
