#!/usr/bin/env bash
# AEV-PLIG 5-seed on all 4 CL tiers. tag matches existing result filenames (v2->edrscc).
set -u
PY=/home/shpark/.conda/envs/aevplig/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/aevplig/src
declare -A TAG=( [lp_edrscc_v2]=edrscc [lp_edrscc_v2_cl1]=lp_edrscc_v2_cl1 \
                 [lp_edrscc_v2_cl12]=lp_edrscc_v2_cl12 [lp_edrscc_v2_cl123]=lp_edrscc_v2_cl123 )
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== AEV-PLIG $SPLIT (tag ${TAG[$SPLIT]}) 5 seeds ====="
  $PY train_edrscc.py --split "$SPLIT" --tag "${TAG[$SPLIT]}" --seeds 0 1 2 3 4 --epochs 200 || echo "FAILED $SPLIT"
done
echo "ALL AEV-PLIG DONE"
