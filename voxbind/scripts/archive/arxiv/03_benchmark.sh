#!/bin/bash
# Short batch-level benchmark for VoxBind training loop.
#
# Usage:
#   bash scripts/03_benchmark.sh [batches] [warmup] [batch_size] [num_workers] [gpu_ids]
#
# Example:
#   bash scripts/03_benchmark.sh 20 5 64 4 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BATCHES="${1:-10}"
WARMUP="${2:-5}"
BSZ="${3:-16}"
NUM_WORKERS="${4:-4}"
GPUS="${5:-${CUDA_VISIBLE_DEVICES:-5,6}}"
EXTRA_ARGS=("${@:6}")

cd "$PROJECT_ROOT"

echo "==> VoxBind short benchmark"
echo "    GPUs        : ${GPUS}"
echo "    Batches     : ${BATCHES} (+${WARMUP} warmup)"
echo "    Batch size  : ${BSZ}"
echo "    Num workers : ${NUM_WORKERS}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate voxbind
else
    echo "ERROR: conda is not installed or not on PATH."
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS" python test/benchmark_train_short.py \
    --batches "$BATCHES" \
    --warmup "$WARMUP" \
    --batch-size "$BSZ" \
    --num-workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"
