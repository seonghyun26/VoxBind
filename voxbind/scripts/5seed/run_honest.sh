#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/dsmbind/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/honestaffinity/src
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== HonestAffinity $SPLIT 5 seeds ====="
  $PY train.py --split "$SPLIT" --seeds 0 1 2 3 4 --gpu 0 || echo "FAILED $SPLIT"
done
echo "ALL HonestAffinity DONE"
