#!/usr/bin/env bash
set -u
HB=/home/shpark/prj-denovo/VoxBind/base/bindnet
PY=/home/shpark/.conda/envs/bindnet/bin/python
GPU="${GPU:-2}"
cd "$HB" || exit 1
for S in 0 1 2 3 4; do
  CK="$HB/_edrscc/results/bindnet_scratch_cl123_seed${S}/checkpoint_best.pt"
  [ -f "$CK" ] || { echo "MISSING ckpt seed$S"; continue; }
  echo "[bindnet-cl123-infer] seed$S"
  $PY _edrscc/src/cl123_inference.py --checkpoint "$CK" --seed "$S" --gpu "$GPU" || echo "FAIL infer s$S"
done
echo "BINDNET CL123 INFER DONE -> base/_casf/novel_preds/BindNet_cl123_seed*.csv"
