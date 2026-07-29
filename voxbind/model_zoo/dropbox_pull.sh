#!/usr/bin/env bash
# Download the model_zoo/ folder FROM the SPML Dropbox (박성현/VoxBind/model_zoo).
# Run this on the target server. By default it downloads INTO the directory this
# script lives in -- so at <checkout>/voxbind/model_zoo/ the weights land right
# there, mirroring the source layout.
#
# Only the MODEL FOLDERS come down. The git-carried files (.gitignore/*.sh/*.html/
# *.md) and the stale bundle tar are excluded, so Dropbox never clobbers what
# arrives via git and you don't re-download the 4.5G bundle.
#
# Re-run anytime -- `rclone copy` is INCREMENTAL & RESUMABLE: files already present
# locally (same size + content) are SKIPPED, only missing/changed files download.
#
# Prereq: rclone remote `dropbox` configured on this host. Fastest is to copy
#         ~/.config/rclone/rclone.conf from the source server (kaistSvr7);
#         otherwise re-auth per dropbox-sync.md.
#
# Usage:  bash dropbox_pull.sh [extra rclone flags, e.g. --dry-run]
#         MODEL_ZOO_DEST=/data/model_zoo bash dropbox_pull.sh   # custom target dir
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SRC="dropbox:/박성현/VoxBind/model_zoo"
DEST="${MODEL_ZOO_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXCLUDES=(
  --exclude "model_zoo_bundle.tar"
  --exclude "model_zoo_bundle.tar.sha256"
  --exclude "*.sh"
  --exclude "*.html"
  --exclude "*.md"
  --exclude ".gitignore"
)

command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not in PATH"; exit 1; }
rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
  || { echo "ERROR: rclone remote 'dropbox' not configured -- see dropbox-sync.md"; exit 1; }

echo ">> downloading  $SRC/  ->  $DEST/"
mkdir -p "$DEST"
rclone copy "$SRC/" "$DEST/" "${EXCLUDES[@]}" --transfers 4 --checkers 8 --progress "$@"

echo
echo ">> done. verify integrity (hash compare) with:"
echo "   rclone check \"$DEST/\" \"$SRC/\" \\"
echo "     --exclude model_zoo_bundle.tar --exclude model_zoo_bundle.tar.sha256 \\"
echo "     --exclude '*.sh' --exclude '*.html' --exclude '*.md' --exclude .gitignore"
