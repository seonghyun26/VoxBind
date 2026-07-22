#!/usr/bin/env bash
# Build the PLINDER on-the-fly (resample) pretraining set on THIS box, reproducing
# shpark's build: 03c precompute (→ data_train_plinder.pt + frozen crops) then
# 03c --mode resample (→ xray_resample_plinder manifest). Uses v5 stats as the
# canonical density scale (03c's --v6_stats arg; v6 dir absent on this box).
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin/python
V5STATS=$ROOT/dataset/data/xray_crops_aligned_v5/stats.json
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
cd "$ROOT" || exit 1

echo "[$(ts)] STAGE 3a: 03c precompute (full 20,498) → data_train_plinder.pt + crops"
"$PY" dataset/legacy/03c_plinder_preprocess.py --mode precompute --v6_stats "$V5STATS" \
  > log/03c_plinder_precompute_full.log 2>&1
echo "[$(ts)] precompute exit=$?  (data_train_plinder.pt $( [ -f dataset/data/data_train_plinder.pt ] && echo OK || echo MISSING ))"

echo "[$(ts)] STAGE 3b: 03c --mode resample → xray_resample_plinder manifest"
"$PY" dataset/legacy/03c_plinder_preprocess.py --mode resample --v6_stats "$V5STATS" \
  > log/03c_plinder_resample.log 2>&1
echo "[$(ts)] resample exit=$?  (xray_resample_plinder $( [ -d dataset/data/xray_resample_plinder ] && echo OK || echo MISSING ))"
echo "[$(ts)] BUILD COMPLETE"
