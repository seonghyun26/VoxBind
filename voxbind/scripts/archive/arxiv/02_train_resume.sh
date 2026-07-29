#!/bin/bash
# Train VoxBind on the Crossdocked dataset.
# Run from the project root: bash scripts/02_train.sh [0.9|1.0]
#
# Usage:
#   bash scripts/02_train.sh        # trains both sigma=0.9 and sigma=1.0 sequentially
#   bash scripts/02_train.sh 0.9    # trains only sigma=0.9
#   bash scripts/02_train.sh 1.0    # trains only sigma=1.0
#
# Results are saved in voxbind/exps/exp_sig<sigma>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SIGMA="${1:-0.9}"  # default: train both
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"
RESUME_EPOCH="${2:-20}"

cd "$PROJECT_ROOT"

train_sigma() {
    local sigma="$1"
    echo "==> Training VoxBind with smooth_sigma=${sigma} on GPUs: ${GPUS} resuming from epoch ${RESUME_EPOCH}"
    CUDA_VISIBLE_DEVICES="$GPUS" python train.py smooth_sigma="${sigma}" resume="exps/exp_sig${sigma}" resume_epoch="${RESUME_EPOCH}"
    echo "==> Checkpoint saved in exps/exp_sig${sigma}/"
}

if [ "$SIGMA" = "all" ]; then
    train_sigma 0.9
    train_sigma 1.0
elif [ "$SIGMA" = "0.9" ] || [ "$SIGMA" = "1.0" ]; then
    train_sigma "$SIGMA"
else
    echo "ERROR: Unknown sigma '${SIGMA}'. Use 0.9, 1.0, or omit for both."
    exit 1
fi

echo "==> Training complete."
