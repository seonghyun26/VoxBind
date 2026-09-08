#!/usr/bin/env bash
# 02_preprocess_data.sh — turn the downloaded raw data into training-ready tensors.
#
# With the prepared copy (01_download_data.sh), data_train.pt / data_test.pt and
# the density crops usually arrive pre-built, so this script is mostly a verify +
# fill-in-the-gaps step. It is idempotent: anything already present is skipped.
#
#   bash script/02_preprocess_data.sh              # verify; build only what's missing
#   MODE=preprocess bash script/02_preprocess_data.sh   # (re)build data_{train,test}.pt
#   MODE=crops      bash script/02_preprocess_data.sh   # (re)build X-ray density crops
#   FORCE=1 MODE=preprocess bash script/02_preprocess_data.sh   # rebuild even if present
#
# Modes:
#   verify      (default) report what exists, build data_{train,test}.pt if absent
#   preprocess  CrossDocked SDF/PDB -> data_train.pt / data_test.pt (dataset/preprocess_crossdocked.py)
#   crops       download 2Fo-Fc maps + align/crop density (scripts/00_data_density_process.sh)
#               — only needed if you are REBUILDING the density corpus from raw maps;
#                 the prepared copy already ships xray_crops_aligned_v5.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

MODE="${MODE:-verify}"
FORCE="${FORCE:-}"
require_env "$VOXBIND_ENV"

run_preprocess() {
    if [ -f "$DATA_ROOT/data_train.pt" ] && [ -f "$DATA_ROOT/data_test.pt" ] && [ -z "$FORCE" ]; then
        log "data_train.pt + data_test.pt already present — skipping (FORCE=1 to rebuild)"; return
    fi
    require_data "split_by_name.pt"        "download the prepared copy (01) or the TargetDiff CrossDocked release"
    require_data "crossdocked_pocket10"    "download + untar crossdocked_pocket10 into dataset/data/"
    banner "preprocessing CrossDocked -> data_train.pt / data_test.pt (can take a couple of hours)"
    ( cd "$CODE_ROOT" && conda_run "$VOXBIND_ENV" python dataset/preprocess_crossdocked.py --data_dir "$DATA_ROOT" ) \
        || die "preprocess_crossdocked.py failed"
    log "wrote $DATA_ROOT/data_{train,test}.pt"
}

run_crops() {
    banner "building X-ray density crops from raw 2Fo-Fc maps (heavy; advanced)"
    log "delegating to voxbind/scripts/00_data_density_process.sh (download + crops)"
    log "NOTE: this rebuilds the BASE crop versions (v1-v4). The v5 / PLINDER"
    log "      ligand-matched corpus the fusion model uses is built by the"
    log "      dataset/00*_*.py + plinder stages and ships in the prepared copy —"
    log "      pull it via 01_download_data.sh rather than rebuilding here."
    ( cd "$CODE_ROOT" \
      && conda_run "$VOXBIND_ENV" env WORKERS="${WORKERS:-16}" bash scripts/00_data_density_process.sh download \
      && conda_run "$VOXBIND_ENV" env WORKERS="${WORKERS:-16}" VERSION="${VERSION:-v3}" bash scripts/00_data_density_process.sh crops ) \
        || die "density crop build failed"
}

case "$MODE" in
    preprocess) run_preprocess ;;
    crops)      run_crops ;;
    verify)     run_preprocess ;;   # build tensors if missing; then just report below
    *) die "unknown MODE '$MODE' (use: verify | preprocess | crops)" ;;
esac

banner "data status"
report() { if [ -e "$DATA_ROOT/$1" ]; then echo "  OK   $2"; else echo "  MISS $2"; fi; }
report "data_train.pt"                                   "preprocessed train tensor"
report "data_test.pt"                                    "preprocessed test tensor"
report "pretrain/xray_crops_aligned_v5/train"            "density crops (train)"
report "pretrain/xray_crops_aligned_v5/test"             "density crops (test)"
report "pretrain/xray_crops_aligned_v5/train_available.npy" "density availability mask (train)"
[ -f "$CODE_ROOT/model_zoo/CDG_v2/checkpoint_e0025.pth.tar" ] && echo "  OK   CDG_v2 encoder" || echo "  MISS CDG_v2 encoder (rerun 01)"
echo
log "done. Next: bash script/03_train.sh"
