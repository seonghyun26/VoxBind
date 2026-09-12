#!/usr/bin/env bash
# dropbox_sync.sh — pull or push the expensive artifacts of this folder.
#
#   bash dropbox_sync.sh pull          # take what Dropbox has; skip what is already here
#   bash dropbox_sync.sh push          # send what this box has
#   bash dropbox_sync.sh status        # what is where, nothing moved
#   DRY=1 bash dropbox_sync.sh push    # print the transfer and stop
#
# WHAT IS WORTH SYNCING AND WHAT IS NOT. The figures, the CSV and interactions.json rebuild
# from `build_interactions.py` in seconds off data that is already on the box, so they stay
# out of this: syncing a derived file invites a stale copy to win over a fresh rebuild.
# What goes is only what costs real time to make again:
#
#   interactions_<Baseline>.json     ~9 min   the five baselines' fingerprints, recomputed
#                                             by compute_baseline_interactions.py because
#                                             the svr12 export never carried them
#   interactions_<Arm>-{gen,docked}.json      the generated/redocked pair
#   poses/<Arm>-{gen,docked}/*.sdf   ~hours   the staged poses those were scored from --
#                                             2,130 Vina docks at exhaustiveness 16
#
# CHECKED BEFORE THIS EXISTED, 2026-09-12: Dropbox has NONE of it. `_shared/260910_posecheck/`
# holds posecheck_<M>.json, but those carry only `p`/`n`/`s`/`c` -- pocket, heavy atoms,
# strain, clashes. No fingerprint, which is the whole reason this folder recomputes them.
# `_shared/posecheck_interactions_78.csv` has interaction MEANS for the local arms only, not
# per molecule and not for the baselines. Nothing docked is staged anywhere. So the first
# run of this script is a push, not a pull.
#
# PULL NEVER OVERWRITES. It is `rclone copy --ignore-existing`: a file already on this box
# wins, because the box is where things are computed and Dropbox is the backup. Delete the
# local file first if you actually want the remote one. PUSH is `rclone copy` too -- neither
# direction ever deletes at the far end, which `rclone sync` would.
#
# The remote root and the SPML namespace setup are in ../../dropbox-sync.md.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="dropbox:/박성현/VoxBind/results/task2-drugdesign/_shared/260910_interaction"
RC=(rclone --checksum)
[[ -n "${DRY:-}" ]] && RC+=(--dry-run)

command -v rclone >/dev/null || { echo "rclone not on PATH — see ../../dropbox-sync.md"; exit 1; }
rclone listremotes | grep -qx "dropbox:" || { echo "no 'dropbox:' remote — see ../../dropbox-sync.md"; exit 1; }

# The JSON this folder cannot cheaply rebuild. interactions.json (no suffix) is derived and
# is deliberately NOT matched by this pattern.
JSON_GLOB='interactions_*.json'
# poses/_tmp/ is the per-worker Vina scratch dock_core_poses.py makes -- receptor prep,
# tmp_h.sdf, one *_ligand.sdf per dock. It is regenerated every run and is most of the
# files, so it never crosses the wire in either direction.
POSE_EXCLUDE=(--exclude "_tmp/**")

case "${1:-status}" in
  pull)
    echo "pull  $REMOTE  ->  $HERE"
    "${RC[@]}" copy "$REMOTE/json" "$HERE" --include "$JSON_GLOB" \
        --ignore-existing --progress
    "${RC[@]}" copy "$REMOTE/poses" "$HERE/poses" "${POSE_EXCLUDE[@]}" \
        --ignore-existing --progress
    echo
    echo "pulled. Rebuild the figures with:"
    echo "  /opt/conda/envs/voxbind/bin/python $HERE/build_interactions.py"
    ;;
  push)
    echo "push  $HERE  ->  $REMOTE"
    "${RC[@]}" copy "$HERE" "$REMOTE/json" --include "$JSON_GLOB" --progress
    [[ -d "$HERE/poses" ]] && "${RC[@]}" copy "$HERE/poses" "$REMOTE/poses" \
        "${POSE_EXCLUDE[@]}" --progress
    ;;
  status)
    echo "local   $HERE"
    ls -1 "$HERE"/$JSON_GLOB 2>/dev/null | sed 's|.*/|  |' || echo "  (no interactions_*.json)"
    [[ -d "$HERE/poses" ]] && echo "  poses/ $(find "$HERE/poses" -path '*_tmp*' -prune -o \
        -name '*.sdf' -print | wc -l) sdf (excluding _tmp scratch)"
    echo
    echo "remote  $REMOTE"
    rclone lsf "$REMOTE" -R 2>/dev/null | sed 's|^|  |' || echo "  (nothing there yet)"
    ;;
  *)
    echo "usage: bash dropbox_sync.sh {pull|push|status}" >&2
    exit 2
    ;;
esac
