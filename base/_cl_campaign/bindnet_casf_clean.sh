#!/bin/bash
# ONE clean single-seed BindNet CASF eval (weakest baseline; 3-seed retrain was chaotic/OOM-prone).
# Train seed 0 on lp_edrscc_v2 (batch 8 to avoid OOM, 8000 steps like the v2 run), eval CASF-214,
# aggregate (1 seed), rebuild the table. Result labeled 1-seed in results.html.
set -u
BN=/home/shpark/prj-denovo/VoxBind/base/bindnet
PY=/home/shpark/.conda/envs/bindnet/bin/python
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/bindnet_casf_clean.log
ts(){ date "+%F %T"; }
cd "$BN"
# clear any stale casfeval seed dirs/logs so the aggregator sees only this clean run
rm -rf _edrscc/results/bindnet_casfeval_seed0 _edrscc/tmp_casfeval_seed0 2>/dev/null
: > _edrscc/logs/eval_casfeval_seed0.log
{ echo "[$(ts)] clean BindNet CASF seed0 (batch8, 8000 steps) on GPU6"
  GPU=6 SEED=0 BATCH=8 MAXSTEPS=8000 VAL=200 PATIENCE=10 bash _edrscc/train_and_eval_casf.sh \
    && echo "[$(ts)] train+eval done" || echo "[$(ts)] FAIL train/eval"
  $PY _edrscc/aggregate_casf_results.py --seeds 0 && echo "[$(ts)] aggregated (1 seed)" || echo "[$(ts)] FAIL agg"
  cd /home/shpark/prj-denovo/VoxBind
  python3 notebook/html/build_results.py 2>/dev/null | grep -oE "CASF [0-9]+/24" || true
  echo "[$(ts)] DONE"; } >> "$LOG" 2>&1
