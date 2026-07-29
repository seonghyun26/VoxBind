#!/bin/bash
# Download 2Fo-Fc map coefficients from RCSB validation reports and convert
# to CCP4 maps via gemmi.  Drop-in replacement for 06_download_xray.sh.
#
# Run from the voxbind/ directory: bash scripts/06b_download_xray_mtz.sh [workers]
#
# Usage:
#   bash scripts/06b_download_xray_mtz.sh [workers]
#
# Arguments (optional):
#   workers     parallel download/conversion threads (default: 8)
#
# Environment variables (optional):
#   PDB_IDS     space-separated PDB IDs to download (skips EDS cache lookup)
#               e.g. PDB_IDS="5p9s 1a2b" bash scripts/06b_download_xray_mtz.sh
#
# Prerequisites:
#   - gemmi installed in the active conda environment (pip install gemmi)
#   - Either PDB_IDS is set, or notebook/data/check_xray_cache.json exists.
#     (Run notebook/check_xray_data.py first to build the cache.)
#
# Outputs:
#   dataset/data/ccp4_mtz/{pdb_id}.map
#   dataset/data/ccp4_mtz/mtz_failures.txt   (if any downloads fail)
#
# To precompute 64³ crops from these maps, pass --ccp4_dir to 05_process_xray.py:
#   python scripts/05_process_xray.py \
#       --data_dir dataset/data \
#       --ccp4_dir dataset/data/ccp4_mtz \
#       --out_dir  dataset/data/xray_crops_mtz \
#       --splits   train test
#
# Key difference from 06_download_xray.sh
# ----------------------------------------
#   06  — downloads pre-computed CCP4 maps directly from PDBe EDS
#   06b — downloads Fourier map coefficients (CIF.gz) from RCSB validation
#          reports and computes the real-space map on-the-fly via gemmi
#
# The resulting maps are equivalent (same 2Fo-Fc content) but the RCSB
# validation maps use a higher grid oversampling (sample_rate=3.0 vs ~2x for
# PDBe EDS), which may yield slightly better interpolation quality for the
# VoxBind 64³ crop.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

WORKERS="${1:-8}"
EDS_CACHE="${PROJECT_ROOT}/../notebook/data/check_xray_cache.json"
OUT_DIR="${PROJECT_ROOT}/dataset/data/ccp4_mtz"

cd "$PROJECT_ROOT"

echo "==> Downloading X-ray map coefficients from RCSB (CIF → CCP4)"
echo "    Output  : ${OUT_DIR}"
echo "    Workers : ${WORKERS}"

if [ -n "${PDB_IDS:-}" ]; then
    # Explicit PDB IDs provided via environment variable
    echo "    Mode    : explicit PDB IDs (${PDB_IDS})"
    # shellcheck disable=SC2086
    python "$SCRIPT_DIR/04b_download_xray_mtz.py" \
        --pdb_ids  $PDB_IDS \
        --out_dir  "$OUT_DIR" \
        --workers  "$WORKERS"
else
    # Use EDS cache (same source as 06_download_xray.sh)
    if [ ! -f "$EDS_CACHE" ]; then
        echo "ERROR: EDS cache not found at ${EDS_CACHE}"
        echo "Either set PDB_IDS env var or run notebook/check_xray_data.py first."
        exit 1
    fi
    echo "    Mode    : EDS cache (${EDS_CACHE})"
    python "$SCRIPT_DIR/04b_download_xray_mtz.py" \
        --eds_cache "$EDS_CACHE" \
        --out_dir   "$OUT_DIR" \
        --workers   "$WORKERS"
fi

echo "==> Download + conversion complete. Maps saved in ${OUT_DIR}"
echo "    Next: python scripts/05_process_xray.py \\"
echo "              --data_dir dataset/data \\"
echo "              --ccp4_dir ${OUT_DIR} \\"
echo "              --out_dir  dataset/data/xray_crops_mtz \\"
echo "              --splits   train test"
