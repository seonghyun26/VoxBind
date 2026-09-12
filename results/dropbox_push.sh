#!/usr/bin/env bash
# Upload this results/ bundle to the SPML Dropbox:  박성현/VoxBind/results
#
# The whole results/ folder is git-ignored EXCEPT the three bootstrap files
# (dropbox_push.sh, dropbox_pull.sh, README.md) which travel via git -- so those
# are excluded here to let git own them and Dropbox own the heavy content
# (reports/, samples/, metrics/, docker/).
#
# `rclone copy` is INCREMENTAL & RESUMABLE: files already on Dropbox (same size +
# content) are SKIPPED, and `copy` never deletes anything there.
#
# Prereq: rclone remote `dropbox` configured (see ../notebook/html/dropbox-sync.md),
#         root_namespace_id = 12221840097.
# Usage:  bash results/dropbox_push.sh [-y|--yes] [extra rclone flags]
#           -y / --yes   skip the confirmation prompt (non-interactive)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../VoxBind/results
DEST="dropbox:/박성현/VoxBind/results"
EXCLUDES=(
  --exclude "/dropbox_push.sh"
  --exclude "/dropbox_pull.sh"
  --exclude "/dropbox_pull_baselines.sh"
  --exclude "/README.md"
  --exclude ".gitignore"
  # agent scratch state that keeps appearing inside the bundle -- never data
  --exclude ".omc/**"
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
  || { echo "ERROR: rclone remote 'dropbox' not configured -- see notebook/html/dropbox-sync.md"; exit 1; }

echo ">> source : $HERE/"
echo ">> dest   : $DEST/"
echo ">> top-level content to upload (dropbox_*.sh/README.md carried by git, skipped):"
( cd "$HERE" && ls -1p | grep -vE '^(dropbox_push\.sh|dropbox_pull\.sh|dropbox_pull_baselines\.sh|README\.md)$' ) | sed 's/^/     /' || true
echo
# The consolidated per-evaluation results (task*/<method>/eval/<evaluation>/results.json,
# written by voxbind/scripts/tools/collect_task2_eval.py) ride inside those folders and are
# the smallest, most-often-changed thing in the bundle -- the whole point of pushing. They
# were silently part of the payload before; this counts them so a push SHOWS they went up.
echo ">> consolidated evaluation JSONs in the payload:"
n_eval=$(find "$HERE" -path '*/eval/*/results.json' -type f 2>/dev/null | wc -l)
n_index=$(find "$HERE" -path '*/eval/index.json' -type f 2>/dev/null | wc -l)
echo "     $n_eval  <task>/<method>/eval/<evaluation>/results.json"
echo "     $n_index  <task>/<method>/eval/index.json"
for st in "$HERE"/*/EVAL_STATUS.json; do
  [ -f "$st" ] || continue
  # Counted out of the JSON, not by grepping the table: the markdown has a second table
  # (the generated-but-not-evaluated runs) whose rows start the same way.
  n_m=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["methods"]), sum(1 for m in d["methods"].values() for k,v in m["evaluations"].items() if v))' "$st" 2>/dev/null)
  echo "     2  ${st#$HERE/}(+.md)   ${n_m:+(${n_m% *} methods, ${n_m#* } populated cells)}"
done
if [ "$n_eval" -eq 0 ]; then
  echo "     NOTE: none found -- run 'python3 voxbind/scripts/tools/collect_task2_eval.py' first"
fi
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
echo "     --exclude /dropbox_push.sh --exclude /dropbox_pull.sh --exclude /dropbox_pull_baselines.sh \\"
echo "     --exclude /README.md --exclude .gitignore --exclude '.omc/**'"
