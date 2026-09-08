#!/usr/bin/env bash
# 01_download_data.sh — fetch the prepared VoxBind data + weights bundle.
#
# The density-conditioned pipeline needs several large artifacts that are NOT in
# git. This script pulls a PREPARED COPY (already preprocessed) so the coworker
# does not rebuild the electron-density corpus from scratch. Two sources:
#
#   A) rclone from the lab Dropbox/Drive folder  (default; DATA_RCLONE_REMOTE)
#   B) a single HTTP(S) tar bundle               (set DATA_HTTP_URL)
#
#   # A — rclone (configure the 'dropbox' remote once, see dropbox-sync.md):
#   bash script/01_download_data.sh
#
#   # B — one-URL bundle:
#   DATA_HTTP_URL=https://host/voxbind_share.tar.zst bash script/01_download_data.sh
#
# What lands where (all under voxbind/dataset/data unless noted):
#   crossdocked_pocket10/                 raw CrossDocked pockets+ligands
#   split_by_name.pt                      CrossDocked train/test split
#   data_train.pt, data_test.pt           preprocessed tensors (02 regenerates if absent)
#   pretrain/xray_crops_aligned_v5/       aligned X-ray density crops (train/test + *_available.npy)
#   pretrain/data_train_plinder*.pt       PLINDER density corpus (only if reproducing the encoder)
#   ../../model_zoo/CDG_v2/               frozen density encoder (checkpoint + cfg.yaml)
#   ../../model_zoo/<base voxbind ckpt>/  optional base-denoiser warm start
#   <sibling>/targetdiff/data/test_set/   full receptors for docking/pose eval
#
# NOTE: replace the placeholder DATA_RCLONE_REMOTE / DATA_HTTP_URL in _common.sh
# (or export them) with the real link before running.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

mkdir -p "$DATA_ROOT" "$CODE_ROOT/model_zoo"

CROSSDOCKED_URL="https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM"

pull_rclone() {
    export PATH="$HOME/.local/bin:$PATH"
    command -v rclone >/dev/null 2>&1 || die "rclone not on PATH — install per dropbox-sync.md, or use DATA_HTTP_URL"
    local remote="$DATA_RCLONE_REMOTE"
    [ -n "$remote" ] || die "DATA_RCLONE_REMOTE is empty — set it in _common.sh or the environment"
    rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
        || die "rclone remote 'dropbox' not configured — see dropbox-sync.md (copy ~/.config/rclone/rclone.conf from the source server)"

    # 1) Weights — confirmed on Dropbox under model_zoo/. Excludes the git-carried
    #    metadata files so they never clobber what arrives via git. Incremental.
    log "rclone copy weights:  $remote/model_zoo/  ->  $CODE_ROOT/model_zoo/"
    rclone copy "$remote/model_zoo/" "$CODE_ROOT/model_zoo/" \
        --exclude "*.sh" --exclude "*.html" --exclude "*.md" --exclude ".gitignore" \
        --exclude "model_zoo_bundle.tar*" \
        --transfers 4 --checkers 8 --progress "$@"

    # 2) Dataset — only if a data/ folder has been uploaded to the same remote.
    if rclone lsd "$remote/data" >/dev/null 2>&1; then
        log "rclone copy dataset:  $remote/data/  ->  $DATA_ROOT/"
        rclone copy "$remote/data/" "$DATA_ROOT/" --transfers 4 --checkers 8 --progress "$@"
    else
        log "NOTE: no prepared dataset at $remote/data — only weights were pulled."
        log "      Get the CrossDocked raw data from the public release:"
        log "        $CROSSDOCKED_URL"
        log "      Download split_by_name.pt + crossdocked_pocket10.tar.gz into"
        log "        $DATA_ROOT/   (untar the tarball there), then run 02_preprocess_data.sh."
        log "      Density crops (xray_crops_aligned_v5) must also be provided; if you"
        log "      have them, upload to $remote/data/ and re-run this script."
    fi

    # 3) Optional: TargetDiff full receptors, if mirrored on the remote.
    if rclone lsd "$remote/targetdiff_test_set" >/dev/null 2>&1; then
        mkdir -p "$FULL_RECEPTOR_ROOT"
        rclone copy "$remote/targetdiff_test_set/" "$FULL_RECEPTOR_ROOT/" --transfers 4 --checkers 8 --progress "$@"
    fi
}

pull_http() {
    local url="$DATA_HTTP_URL"
    local tmp="$DATA_ROOT/_bundle.$(basename "$url")"
    log "downloading bundle: $url"
    command -v curl >/dev/null 2>&1 || die "curl not found"
    curl -fL --retry 3 -o "$tmp" "$url" || die "download failed"
    log "extracting into $REPO_ROOT (bundle is expected to contain voxbind/dataset/data + voxbind/model_zoo)"
    case "$url" in
        *.tar.zst|*.tzst) command -v zstd >/dev/null 2>&1 || die "zstd needed to unpack $url"
                          zstd -dc "$tmp" | tar -C "$REPO_ROOT" -xf - ;;
        *.tar.gz|*.tgz)   tar -C "$REPO_ROOT" -xzf "$tmp" ;;
        *.tar)            tar -C "$REPO_ROOT" -xf  "$tmp" ;;
        *) die "unrecognized bundle extension for $url (want .tar / .tar.gz / .tar.zst)" ;;
    esac
    rm -f "$tmp"
}

banner "downloading prepared VoxBind data"
if [ -n "$DATA_HTTP_URL" ]; then
    pull_http
else
    pull_rclone "$@"
fi

banner "verifying key artifacts"
ok=1
check() { if [ -e "$DATA_ROOT/$1" ] || [ -e "$CODE_ROOT/$1" ] || [ -e "$1" ]; then echo "  OK   $2"; else echo "  MISS $2"; ok=0; fi; }
check "crossdocked_pocket10"                       "CrossDocked pockets            (dataset/data/crossdocked_pocket10/)"
check "split_by_name.pt"                           "CrossDocked split              (dataset/data/split_by_name.pt)"
check "pretrain/xray_crops_aligned_v5/train"       "density crops (train)          (dataset/data/pretrain/xray_crops_aligned_v5/train/)"
check "pretrain/xray_crops_aligned_v5/train_available.npy" "density availability mask (…/train_available.npy)"
check "model_zoo/CDG_v2/checkpoint_e0025.pth.tar"  "CDG_v2 frozen encoder          (voxbind/model_zoo/CDG_v2/)"
check "model_zoo/voxbind_sig0.9_crossdocked/checkpoint.pth.tar" "base-denoiser warm start (voxbind/model_zoo/voxbind_sig0.9_crossdocked/)"
[ -d "$FULL_RECEPTOR_ROOT" ] && echo "  OK   TargetDiff full receptors ($FULL_RECEPTOR_ROOT)" || echo "  MISS TargetDiff full receptors ($FULL_RECEPTOR_ROOT) — needed for docking/pose eval only"

echo
if [ "$ok" = 1 ]; then
    log "core artifacts present. Next: bash script/02_preprocess_data.sh"
else
    log "some artifacts missing — check the source link / rclone remote and re-run (copy is incremental)."
fi
