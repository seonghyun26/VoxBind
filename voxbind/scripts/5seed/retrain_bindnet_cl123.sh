#!/usr/bin/env bash
# Re-train BindNet cl123 all 5 seeds FRESH (keeps checkpoint_best.pt per seed) so per-complex
# cl123-test preds can be inferred for Table-1b CL3 novelty. Re-aggregate metrics for consistency.
set -u
HB=/home/shpark/prj-denovo/VoxBind/base/bindnet
PY=/home/shpark/.conda/envs/bindnet/bin/python
GPU="${GPU:-2}"
for SEED in 0 1 2 3 4; do
  LOGF="$HB/_edrscc/logs/train_cl123_seed${SEED}.log"
  echo "[bindnet-cl123] train seed$SEED (GPU$GPU)"
  GPU=$GPU SEED=$SEED SPLIT=cl123 LBA_DATA="_edrscc/data/edrscc_cl123/lba" \
    bash "$HB/_edrscc/train_bindnet.sh" > "$LOGF" 2>&1 || echo "FAIL cl123 s$SEED"
done
echo "[bindnet-cl123] aggregate"
$PY "$HB/_edrscc/aggregate_results.py" --split lp_edrscc_v2_cl123 --seeds 0,1,2,3,4 || echo "agg FAIL"
echo "BINDNET CL123 RETRAIN DONE"
