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
#   download    download 2Fo-Fc CCP4 maps + deposited PDBs (dataset/00a_density_download.py)
#   crops       align + crop + normalise density crops (dataset/00b_density_preprocess.py)
#
# Environment variables (optional):
#   WORKERS     parallel workers (download threads / crop processes)  (default: 16)
#   SPLITS      splits to process                                     (default: "train test")
#   VERSION     crop version(s) for the crops step                    (default: "v3")
#                 v1 → xray_crops_aligned (load-time ±3σ z-score)
#                 v2 → xray_crops_aligned_v2 (pool z-score)
#                 v3 → xray_crops_aligned_v3 (pool max-abs → [−1,1])
#                 v4 → xray_crops_aligned_v4 (pool clip + z-score)
#               Space-separate to build several at once, e.g. VERSION="v2 v3 v4".
#   EDS_CACHE   optional PDBe EDS cache JSON to restrict MAP downloads to
#               confirmed-EDS PDB IDs (default: ../notebook/data/check_xray_cache.json
#               if present; otherwise maps are attempted for every dataset PDB ID)
#
# Prerequisites (preprocess step):
#   Manually download the following into voxbind/dataset/data/:
#     - split_by_name.pt
#     - crossdocked_pocket10.tar.gz   (then: tar xvzf ... -C dataset/data/)
#   Source: https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM
#
# Outputs:
#   preprocess  →  dataset/data/data_train.pt, dataset/data/data_test.pt
#   download    →  dataset/data/ccp4/{pdb}.ccp4 , dataset/data/pdb/{pdb}.pdb
#   crops       →  dataset/data/xray_crops_aligned[_vN]/{train,test}/{:06d}.npy
#                  dataset/data/xray_crops_aligned[_vN]/{train,test}_available.npy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-all}"
WORKERS="${WORKERS:-16}"
SPLITS="${SPLITS:-train test}"
VERSION="${VERSION:-v3}"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${DATA_DIR}/ccp4"
PDB_DIR="${DATA_DIR}/pdb"
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
    echo "==> [download] X-ray CCP4 maps + deposited PDBs from PDBe"
    echo "    Maps    : ${CCP4_DIR}"
    echo "    PDBs    : ${PDB_DIR}"
    echo "    Splits  : ${SPLITS}"
    echo "    Workers : ${WORKERS}"
    local eds_arg=()
    if [ -f "$EDS_CACHE" ]; then
        echo "    EDS cache (map filter): ${EDS_CACHE}"
        eds_arg=(--eds_cache "$EDS_CACHE")
    else
        echo "    EDS cache not found — attempting maps for every dataset PDB ID."
    fi
    # shellcheck disable=SC2086
    python dataset/00a_density_download.py \
        --data_dir "$DATA_DIR" \
        --ccp4_dir "$CCP4_DIR" \
        --pdb_dir  "$PDB_DIR" \
        --splits   $SPLITS \
        --workers  "$WORKERS" \
        "${eds_arg[@]}"
}

run_crops() {
    if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
        echo "ERROR: No CCP4 maps in ${CCP4_DIR}"
        echo "  Run with mode=download first."
        exit 1
    fi

    echo "==> [crops] Align + crop + normalise X-ray density"
    echo "    Data dir : ${DATA_DIR}"
    echo "    Version  : ${VERSION}"
    echo "    Splits   : ${SPLITS}"
    echo "    Workers  : ${WORKERS}"
    # shellcheck disable=SC2086
    python dataset/00b_density_preprocess.py \
        --version  $VERSION \
        --data_dir "$DATA_DIR" \
        --ccp4_dir "$CCP4_DIR" \
        --pdb_dir  "$PDB_DIR" \
        --out_root "$DATA_DIR" \
        --splits   $SPLITS \
        --workers  "$WORKERS"
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
