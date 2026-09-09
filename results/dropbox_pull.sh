#!/usr/bin/env bash
# Download the results/ bundle FROM the SPML Dropbox (박성현/VoxBind/results).
# Run this on the target server. By default it downloads INTO the directory this
# script lives in -- so at <checkout>/results/ the content lands right there.
#
# The three git-carried bootstrap files (dropbox_push.sh, dropbox_pull.sh,
# README.md) are excluded so Dropbox never clobbers what arrives via git.
#
# Re-run anytime -- `rclone copy` is INCREMENTAL & RESUMABLE: files already present
# locally (same size + content) are SKIPPED, only missing/changed files download.
#
# Prereq: rclone remote `dropbox` configured on this host (root_namespace_id
#         12221840097). Fastest is to copy ~/.config/rclone/rclone.conf from a
#         configured server; otherwise re-auth per notebook/html/dropbox-sync.md.
#
# Usage:  bash results/dropbox_pull.sh [extra rclone flags, e.g. --dry-run]
#         RESULTS_DEST=/data/results bash results/dropbox_pull.sh   # custom target dir
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SRC="dropbox:/박성현/VoxBind/results"
DEST="${RESULTS_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXCLUDES=(
  --exclude "/dropbox_push.sh"
  --exclude "/dropbox_pull.sh"
  --exclude "/dropbox_pull_baselines.sh"
  --exclude "/README.md"
  --exclude ".gitignore"
)

command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not in PATH"; exit 1; }
rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
  || { echo "ERROR: rclone remote 'dropbox' not configured -- see notebook/html/dropbox-sync.md"; exit 1; }

echo ">> downloading  $SRC/  ->  $DEST/"
mkdir -p "$DEST"
rclone copy "$SRC/" "$DEST/" "${EXCLUDES[@]}" --transfers 4 --checkers 8 --progress "$@"

echo
echo ">> done. verify integrity (hash compare) with:"
echo "   rclone check \"$DEST/\" \"$SRC/\" \\"
echo "     --exclude /dropbox_push.sh --exclude /dropbox_pull.sh --exclude /dropbox_pull_baselines.sh --exclude /README.md --exclude .gitignore"
