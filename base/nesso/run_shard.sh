#!/bin/bash
# run_shard.sh SHARD_ID GPU_ID
# Runs nesso predict on all YAMLs in _edrscc/shard_{SHARD_ID}.txt on GPU {GPU_ID}
# Results go to _edrscc/outputs/ (shared output dir, one subdir per pid = safe to merge)

SHARD_ID=$1
GPU_ID=$2

if [ -z "$SHARD_ID" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 SHARD_ID GPU_ID"
    exit 1
fi

NESSO_BIN=/home/shpark/.conda/envs/nesso/bin/nesso
BASE=/home/shpark/prj-denovo/VoxBind/base/nesso
YAMLS_DIR=$BASE/_edrscc/yamls
OUT_DIR=$BASE/_edrscc/outputs
LOG_DIR=$BASE/_edrscc/logs
CACHE_DIR=$BASE/.cache
SHARD_FILE=$BASE/_edrscc/shard_${SHARD_ID}.txt

mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE=$LOG_DIR/shard_${SHARD_ID}.log

echo "[shard $SHARD_ID] Starting on GPU $GPU_ID at $(date)" | tee "$LOG_FILE"
echo "[shard $SHARD_ID] Processing $(wc -l < $SHARD_FILE) pids" | tee -a "$LOG_FILE"

total=0
done=0
failed=0

while IFS= read -r pid; do
    total=$((total + 1))
    yaml=$YAMLS_DIR/$pid.yaml
    pred_dir=$OUT_DIR/predictions/$pid

    # Skip if already done (affinity.json exists)
    if [ -f "$pred_dir/affinity.json" ]; then
        done=$((done + 1))
        continue
    fi

    if [ ! -f "$yaml" ]; then
        echo "[shard $SHARD_ID] SKIP $pid (no yaml)" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID $NESSO_BIN predict "$yaml" \
        --out_dir "$OUT_DIR" \
        --cache "$CACHE_DIR" \
        --devices 1 --accelerator gpu --no_kernels \
        --seed 42 \
        2>>"$LOG_FILE" 1>>"$LOG_FILE"

    if [ $? -eq 0 ]; then
        done=$((done + 1))
    else
        echo "[shard $SHARD_ID] FAILED $pid" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
    fi

    # Progress every 50 pids
    if [ $((total % 50)) -eq 0 ]; then
        echo "[shard $SHARD_ID] progress: $total processed, $done ok, $failed failed at $(date)" | tee -a "$LOG_FILE"
    fi
done < "$SHARD_FILE"

echo "[shard $SHARD_ID] DONE: $total pids, $done ok, $failed failed at $(date)" | tee -a "$LOG_FILE"
