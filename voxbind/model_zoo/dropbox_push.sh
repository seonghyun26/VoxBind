#!/usr/bin/env bash
# Upload this model_zoo/ folder to the SPML Dropbox:  박성현/VoxBind/model_zoo
#
# Only the MODEL FOLDERS (weights + cfg.yaml/.hydra/train log) are uploaded.
# Files that travel via git -- .gitignore, *.sh, *.html, *.md -- and the stale,
# oversized model_zoo_bundle.tar (+ .sha256) are excluded. Only this folder's OWN
# contents go up (trailing "/" on the source); nothing above model_zoo is touched.
# NOTE: the checkpoint weights are *.pth.tar -- we exclude only the exact filename
# model_zoo_bundle.tar, NOT the glob *.tar, so the weights DO upload.
#
# Lists the folders that will upload, shows a dry-run preview, then asks to confirm.
# `rclone copy` is INCREMENTAL & RESUMABLE: files already on Dropbox (same size +
# content) are SKIPPED, and `copy` never deletes anything there.
#
# Prereq: rclone remote `dropbox` configured (see ../../dropbox-sync.md),
#         root_namespace_id = 12221840097.
# Usage:  bash model_zoo/dropbox_push.sh [-y|--yes] [extra rclone flags]
#           -y / --yes   skip the confirmation prompt (non-interactive)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../voxbind/model_zoo
DEST="dropbox:/박성현/VoxBind/model_zoo"
EXCLUDES=(
  --exclude "model_zoo_bundle.tar"
  --exclude "model_zoo_bundle.tar.sha256"
  --exclude "*.sh"
  --exclude "*.html"
  --exclude "*.md"
  --exclude ".gitignore"
)

ASSUME_YES=0; ARGS=()
for a in "$@"; do
  case "$a" in
    -y|--yes) ASSUME_YES=1 ;;
    *)        ARGS+=("$a") ;;
  esac
done

command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not in PATH"; exit 1; }
rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
  || { echo "ERROR: rclone remote 'dropbox' not configured -- see dropbox-sync.md"; exit 1; }

echo ">> source : $HERE/"
echo ">> dest   : $DEST/"
echo ">> model folders to upload (git carries .gitignore/*.sh/*.html/*.md; bundle tar skipped):"
( cd "$HERE" && ls -1p | grep '/$' ) | sed 's/^/     /' || true
echo
echo ">> dry-run preview (what would actually transfer vs. what's already on Dropbox):"
rclone copy "$HERE/" "$DEST/" "${EXCLUDES[@]}" --dry-run "${ARGS[@]}"
echo

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p ">> proceed with upload? [y/N] " ans
  [[ "$ans" == [yY] || "$ans" == [yY][eE][sS] ]] || { echo "aborted."; exit 0; }
fi

rclone copy "$HERE/" "$DEST/" "${EXCLUDES[@]}" --transfers 4 --checkers 8 --progress "${ARGS[@]}"

echo
echo ">> done. verify integrity (hash compare) with:"
echo "   rclone check \"$HERE/\" \"$DEST/\" \\"
echo "     --exclude model_zoo_bundle.tar --exclude model_zoo_bundle.tar.sha256 \\"
echo "     --exclude '*.sh' --exclude '*.html' --exclude '*.md' --exclude .gitignore"
