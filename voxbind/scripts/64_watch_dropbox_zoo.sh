#!/usr/bin/env bash
# Poll the SPML Dropbox model_zoo and pull whenever a new/changed checkpoint appears.
#
# Emits ONE line per event on stdout (so it can drive a Monitor):
#   NEW <folder>/<file> ...        newly seen remote objects, before pulling
#   PULLED ...                     pull finished, with what landed locally
#   PROBLEM ...                    rclone/listing failure
# Silent when nothing changed.
#
# Uses model_zoo/dropbox_pull.sh for the actual transfer (incremental + resumable),
# so the exclude rules and destination stay in one place.
#
# Usage:  bash scripts/64_watch_dropbox_zoo.sh [poll_seconds]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ZOO=/home1/irteam/VoxBind/voxbind/model_zoo
SRC="dropbox:/박성현/VoxBind/model_zoo"
POLL="${1:-900}"                       # 15 min default — remote API, checkpoints are rare
STATE=/tmp/claude-500/-home1-irteam-VoxBind/f728513b-c34b-4a14-8006-b8671512085f/scratchpad/dropbox_zoo_state.txt

# "path size modtime" for every remote object, excluding the git-carried/bundle files
# that dropbox_pull.sh also excludes -- otherwise doc edits would look like new models.
#
# MODTIME IS ESSENTIAL, not decoration: a training job that overwrites
# <exp>/checkpoint.pth.tar at a later epoch produces a file of the SAME BYTE SIZE
# (same tensors, same layout). A size-only snapshot silently misses every such update
# -- which is the common case, since checkpoints are rewritten in place each epoch.
snapshot() {
  timeout 180 rclone lsl "$SRC" \
    --exclude "model_zoo_bundle.tar" --exclude "model_zoo_bundle.tar.sha256" \
    --exclude "*.sh" --exclude "*.html" --exclude "*.md" --exclude ".gitignore" \
    2>/dev/null | awk '{print $4" "$1" "$2"_"$3}' | sort
}

[ -f "$STATE" ] || snapshot > "$STATE"

while true; do
  sleep "$POLL"
  now=$(snapshot)
  if [ -z "$now" ]; then
    echo "PROBLEM: rclone listing failed or returned empty (network/auth?)"
    continue
  fi
  new=$(comm -13 "$STATE" <(echo "$now") | awk '{print $1}')
  if [ -n "$new" ]; then
    echo "NEW on Dropbox: $(echo "$new" | paste -sd' ' | cut -c1-300)"
    if timeout 3600 bash "$ZOO/dropbox_pull.sh" >/tmp/dbx_pull.$$ 2>&1; then
      landed=$(echo "$new" | while read -r f; do
                 [ -f "$ZOO/$f" ] && printf '%s(%s) ' "$f" \
                   "$(du -h "$ZOO/$f" 2>/dev/null | cut -f1)"
               done)
      echo "PULLED: ${landed:-<nothing landed - check excludes>}"
      echo "$now" > "$STATE"
    else
      echo "PROBLEM: dropbox_pull.sh failed: $(tail -3 /tmp/dbx_pull.$$ | paste -sd' ')"
    fi
    rm -f /tmp/dbx_pull.$$
  fi
done
