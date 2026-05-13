#!/bin/bash
# Full data + X-ray density processing pipeline for VoxBind.
# Run from the project root: bash scripts/00_data_density_process.sh [mode]
#
# Usage:
#   bash scripts/00_data_density_process.sh [mode]
#
# Modes (default: all):
#   all         run every step: preprocess  →  download  →  crops
#   preprocess  preprocess CrossDocked (dataset/preprocess_crossdocked.py)
#   download    download 2Fo-Fc CCP4 maps from PDBe EDS (dataset/00a_data_density_download.py)
#   crops       precompute 64³ density crops (dataset/00b_data_density_preprocess.py)
#
# Environment variables (optional):
#   WORKERS     parallel download threads        (default: 16)
#   SPLITS      crop splits to process           (default: "train test")
#   EDS_CACHE   PDBe EDS cache JSON              (default: ../notebook/data/check_xray_cache.json)
#
# Prerequisites (preprocess step):
#   Manually download the following into voxbind/dataset/data/:
#     - split_by_name.pt
#     - crossdocked_pocket10.tar.gz   (then: tar xvzf ... -C dataset/data/)
#   Source: https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM
#
# Prerequisites (download step):
#   notebook/data/check_xray_cache.json must exist
#   (run notebook/check_xray_data.py first to build the cache).
#
# Outputs:
#   preprocess  →  dataset/data/data_train.pt, dataset/data/data_test.pt
#   download    →  dataset/data/ccp4/{pdb_id}.map
#   crops       →  dataset/data/xray_crops/{train,test}/{:06d}.npy
#                  dataset/data/xray_crops/{train,test}_available.npy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-all}"
WORKERS="${WORKERS:-16}"
SPLITS="${SPLITS:-train test}"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"
EDS_CACHE="${EDS_CACHE:-${PROJECT_ROOT}/../notebook/data/check_xray_cache.json}"

cd "$PROJECT_ROOT"

run_preprocess() {
    if [ ! -f "${DATA_DIR}/split_by_name.pt" ]; then
        echo "ERROR: ${DATA_DIR}/split_by_name.pt not found."
        echo "  Download from https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM"
        exit 1
    fi
    if [ ! -d "${DATA_DIR}/crossdocked_pocket10" ]; then
        echo "ERROR: ${DATA_DIR}/crossdocked_pocket10/ not found."
        echo "  Download crossdocked_pocket10.tar.gz and decompress into ${DATA_DIR}/."
        exit 1
    fi

    echo "==> [preprocess] CrossDocked preprocessing (may take a couple of hours)"
    python dataset/preprocess_crossdocked.py --data_dir "$DATA_DIR"
    echo "    output: ${DATA_DIR}/data_train.pt, ${DATA_DIR}/data_test.pt"
}

run_download() {
    if [ ! -f "$EDS_CACHE" ]; then
        echo "ERROR: EDS cache not found at ${EDS_CACHE}"
        echo "  Run notebook/check_xray_data.py first to build the cache."
        exit 1
    fi

    echo "==> [download] X-ray CCP4 maps from PDBe EDS"
    echo "    EDS cache : ${EDS_CACHE}"
    echo "    Output    : ${CCP4_DIR}"
    echo "    Workers   : ${WORKERS}"
    python dataset/00a_data_density_download.py \
        --eds_cache "$EDS_CACHE" \
        --out_dir   "$CCP4_DIR" \
        --workers   "$WORKERS"
}

run_crops() {
    if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
        echo "ERROR: No CCP4 maps in ${CCP4_DIR}"
        echo "  Run with mode=download first."
        exit 1
    fi

    echo "==> [crops] Precompute X-ray density crops"
    echo "    Data dir  : ${DATA_DIR}"
    echo "    CCP4 dir  : ${CCP4_DIR}"
    echo "    Output    : ${CROPS_DIR}"
    echo "    Splits    : ${SPLITS}"
    # shellcheck disable=SC2086
    python dataset/00b_data_density_preprocess.py \
        --data_dir "$DATA_DIR" \
        --ccp4_dir "$CCP4_DIR" \
        --out_dir  "$CROPS_DIR" \
        --splits   $SPLITS
}

case "$MODE" in
    all)
        run_preprocess
        run_download
        run_crops
        ;;
    preprocess)
        run_preprocess
        ;;
    download)
        run_download
        ;;
    crops)
        run_crops
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: all | preprocess | download | crops"
        exit 1
        ;;
esac

echo "==> Done."
