#!/usr/bin/env bash
set -u
cd /home/shpark/prj-denovo/VoxBind/base/cheapnet/_edrscc || exit 1
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== CheapNet $SPLIT 5 seeds ====="
  CUDA_VISIBLE_DEVICES=0 bash cheappy train_edrscc.py \
    --data_root "data_${SPLIT}" --tag "${SPLIT}" --seeds 0 1 2 3 4 || echo "FAILED $SPLIT"
done
echo "ALL CheapNet DONE"
