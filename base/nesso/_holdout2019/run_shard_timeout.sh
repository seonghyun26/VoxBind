#!/usr/bin/env bash
# timeout-guarded Nesso shard runner — skips large-protein ESM hangs (150s cap)
SHARD=$1; GPU=$2
NESSO=/home/shpark/.conda/envs/nesso/bin/nesso
BASE=/home/shpark/prj-denovo/VoxBind/base/nesso/_holdout2019
OUT=$BASE/outputs
while IFS= read -r pid; do
  [ -z "$pid" ] && continue
  [ -f "$OUT/predictions/$pid/affinity.json" ] && continue
  [ ! -f "$BASE/yamls/$pid.yaml" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU timeout 150 $NESSO predict "$BASE/yamls/$pid.yaml" --out_dir "$OUT" --cache "$BASE/.cache" >/dev/null 2>&1
done < "$BASE/$SHARD"
echo "[shard $SHARD gpu$GPU] done"
