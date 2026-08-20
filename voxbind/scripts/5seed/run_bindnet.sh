#!/usr/bin/env bash
# BindNet -> 5 seeds: keep existing seed0-2 logs, train ONLY seed3,4, re-aggregate 0-4.
set -u
HB=/home/shpark/prj-denovo/VoxBind/base/bindnet
PY=/home/shpark/.conda/envs/bindnet/bin/python
GPU="${GPU:-2}"
# spec: TAG:SPLIT_ENV:FULL_SPLIT   (v2 -> empty SPLIT env, default LBA)
for SPEC in "v2::lp_edrscc_v2" "cl1:cl1:lp_edrscc_v2_cl1" "cl12:cl12:lp_edrscc_v2_cl12" "cl123:cl123:lp_edrscc_v2_cl123"; do
  IFS=: read -r TAG SPLIT FULL <<< "$SPEC"
  for SEED in 3 4; do
    if [ "$TAG" = v2 ]; then
      LOGF="$HB/_edrscc/logs/train_seed${SEED}.log"
      echo "[bindnet] train v2 seed$SEED (GPU$GPU)"
      GPU=$GPU SEED=$SEED bash "$HB/_edrscc/train_bindnet.sh" > "$LOGF" 2>&1 || echo "FAIL v2 s$SEED"
    else
      LOGF="$HB/_edrscc/logs/train_${TAG}_seed${SEED}.log"
      echo "[bindnet] train $TAG seed$SEED (GPU$GPU)"
      GPU=$GPU SEED=$SEED SPLIT=$TAG LBA_DATA="_edrscc/data/edrscc_${TAG}/lba" \
        bash "$HB/_edrscc/train_bindnet.sh" > "$LOGF" 2>&1 || echo "FAIL $TAG s$SEED"
    fi
  done
  echo "[bindnet] aggregate $FULL (5 seeds)"
  $PY "$HB/_edrscc/aggregate_results.py" --split "$FULL" --seeds 0,1,2,3,4 || echo "FAIL agg $FULL"
done
echo "ALL BINDNET DONE"
