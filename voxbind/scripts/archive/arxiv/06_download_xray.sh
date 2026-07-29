#!/bin/bash
# Download 2Fo-Fc CCP4 electron density maps from PDBe EDS.
# Run from the voxbind/ directory: bash scripts/06_download_xray.sh
#
# Usage:
#   bash scripts/06_download_xray.sh [workers]
#
# Arguments (optional):
#   workers     number of parallel download threads (default: 16)
#
# Prerequisites:
#   notebook/data/check_xray_cache.json must exist.
#   Run notebook/check_xray_data.py first if it does not.
#
# Outputs:
#   dataset/data/ccp4/{pdb_id}.map   (~13,000 files, ~200 GB total)
#   dataset/data/ccp4/download_failures.txt   (if any downloads fail)
#
# The download is resumable — already-existing files are skipped.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WORKERS="${1:-16}"
EDS_CACHE="${PROJECT_ROOT}/../notebook/data/check_xray_cache.json"
OUT_DIR="${PROJECT_ROOT}/dataset/data/ccp4"

cd "$PROJECT_ROOT"

if [ ! -f "$EDS_CACHE" ]; then
    echo "ERROR: EDS cache not found at ${EDS_CACHE}"
    echo "Run notebook/check_xray_data.py first to build the cache."
    exit 1
fi

echo "==> Downloading X-ray CCP4 maps"
echo "    EDS cache : ${EDS_CACHE}"
echo "    Output    : ${OUT_DIR}"
echo "    Workers   : ${WORKERS}"

python "$SCRIPT_DIR/04_download_xray.py" \
    --eds_cache "$EDS_CACHE" \
    --out_dir   "$OUT_DIR" \
    --workers   "$WORKERS"

echo "==> Download complete. Maps saved in ${OUT_DIR}"
