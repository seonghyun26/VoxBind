#!/bin/bash
# Precompute 64³ X-ray density crops for all CrossDocked2020 samples.
# Run from the voxbind/ directory: bash scripts/07_process_xray.sh
#
# Usage:
#   bash scripts/07_process_xray.sh [splits]
#
# Arguments (optional):
#   splits      space-separated list of splits to process (default: "train test")
#               choices: train, val, test
#
# Prerequisites:
#   CCP4 maps must be downloaded first (bash scripts/06_download_xray.sh).
#
# Outputs:
#   dataset/data/xray_crops/train/{:06d}.npy   float16 (64, 64, 64) per sample
#   dataset/data/xray_crops/test/{:06d}.npy
#   dataset/data/xray_crops/train_available.npy
#   dataset/data/xray_crops/test_available.npy
#
# The processing is resumable — already-existing crop files are skipped.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SPLITS="${1:-train test}"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
OUT_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"

cd "$PROJECT_ROOT"

if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
    echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
    echo "Run scripts/06_download_xray.sh first."
    exit 1
fi

echo "==> Precomputing X-ray density crops"
echo "    Data dir  : ${DATA_DIR}"
echo "    CCP4 dir  : ${CCP4_DIR}"
echo "    Output    : ${OUT_DIR}"
echo "    Splits    : ${SPLITS}"

# shellcheck disable=SC2086
python scripts/05_process_xray.py \
    --data_dir "$DATA_DIR" \
    --ccp4_dir "$CCP4_DIR" \
    --out_dir  "$OUT_DIR" \
    --splits   $SPLITS

echo "==> Processing complete. Crops saved in ${OUT_DIR}"
