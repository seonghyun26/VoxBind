SHARD_ID=$1; GPU_ID=$2
NESSO_BIN=/home/shpark/.conda/envs/nesso/bin/nesso
BASE=/home/shpark/prj-denovo/VoxBind/base/nesso/_holdout2019
YAMLS=$BASE/yamls; OUT=$BASE/outputs; mkdir -p $OUT
while IFS= read -r pid; do
  [ -f "$OUT/predictions/$pid/affinity.json" ] && continue
  [ ! -f "$YAMLS/$pid.yaml" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU_ID $NESSO_BIN predict "$YAMLS/$pid.yaml" --out_dir "$OUT" --cache "$BASE/.cache" >/dev/null 2>&1
done < $BASE/shard_0$SHARD_ID
echo "[shard $SHARD_ID] done"
