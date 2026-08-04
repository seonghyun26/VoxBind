#!/usr/bin/env bash
# Launch the full HonestAffinity-Pocket sweep: 4 splits x 3 seeds on one GPU.
# Usage: bash run_full.sh <gpu>      (GPUs 0-3 are off-limits; use 4/5/6/7)
# Data prep + ESM cache must already be done (they are, for all 4 splits).
set -e
GPU=${1:?usage: run_full.sh <gpu>}
PY=/home/shpark/.conda/envs/dsmbind/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/src"
mkdir -p "$HERE/logs"
for SPLIT in lp_edrscc_v2_cl123 lp_edrscc_v2_cl12 lp_edrscc_v2_cl1 lp_edrscc_v2; do
  echo "=== training $SPLIT on GPU $GPU ($(date)) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py --split "$SPLIT" --seeds 0 1 2 --gpu 0 \
    2>&1 | tee "$HERE/logs/train_$SPLIT.log"
done
echo "=== all splits done ($(date)) ==="
