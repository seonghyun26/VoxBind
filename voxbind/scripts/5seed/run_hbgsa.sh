#!/usr/bin/env bash
set -u
HB=/home/shpark/prj-denovo/VoxBind/base/hbgsa
PY=/home/shpark/.conda/envs/hbgsa/bin/python
SP=/home/shpark/prj-denovo/VoxBind/voxbind/splits
cd "$HB/src" || exit 1
declare -A TAG=( [lp_edrscc_v2]=edrscc_v2_3p06m [lp_edrscc_v2_cl1]=edrscc_v2_cl1_3p06m \
                 [lp_edrscc_v2_cl12]=edrscc_v2_cl12_3p06m [lp_edrscc_v2_cl123]=edrscc_v2_cl123_3p06m )
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== HBGSA $SPLIT (tag ${TAG[$SPLIT]}) 5 seeds ====="
  $PY train.py --split_csv "$SP/$SPLIT.csv" --tag "${TAG[$SPLIT]}" --seeds 0,1,2,3,4 --epochs 150 \
    || echo "FAILED $SPLIT"
done
echo "=== aggregating ==="
$PY ../agg_results.py --tags edrscc_v2_3p06m edrscc_v2_cl1_3p06m edrscc_v2_cl12_3p06m edrscc_v2_cl123_3p06m || echo "agg FAILED"
echo "ALL HBGSA DONE"
