#!/usr/bin/env bash
# Download ONLY the generated samples of the methods you name, from the SPML
# Dropbox bundle (박성현/VoxBind/results). The cheap slice of `dropbox_pull.sh`,
# which pulls the whole 3.8 GiB bundle.
#
# Per method it takes:   samples/**  +  metrics.json  +  SOURCE.txt
# and deliberately skips: run/ (cfg + hydra + train logs), representations/
#   (task1 cached features), eval/ (per-pocket vina/posecheck/posebusters --
#   add it with --with-eval) and -- unless --with-raw -- the raw generator dumps
#   under samples/outputs_*/. That last one is the whole point: DecompDiff is
#   3.47 GiB of outputs_* against 12.7 MiB of samples/meta/*.pt, and the .pt is
#   what every downstream analysis actually reads.
#
# Method names are resolved against the remote across task1-affinity /
# task2-drugdesign / task3-mcp, so you pass a bare name, never a path.
#
# Re-run anytime -- `rclone copy` is INCREMENTAL & RESUMABLE (files already here
# with the same size+content are skipped) and never deletes anything.
#
# Prereq: rclone remote `dropbox` configured on this host (root_namespace_id
#         12221840097) -- see notebook/html/dropbox-sync.md.
#
# Usage:
#   bash results/dropbox_pull_baselines.sh                   # the 5 CrossDocked baselines
#   bash results/dropbox_pull_baselines.sh AR DiffSBDD GET   # any number of methods
#   bash results/dropbox_pull_baselines.sh -l                # what does the remote have?
#   bash results/dropbox_pull_baselines.sh --with-eval       # samples + per-pocket eval
#   bash results/dropbox_pull_baselines.sh -n DecompDiff     # dry run
#   bash results/dropbox_pull_baselines.sh -a --task task3-mcp
#
# Options:
#   -l, --list          list every method on the remote, grouped by task, then exit
#   -a, --all           every method of --task (of all tasks when --task is unset)
#   -n, --dry-run       preview the transfer, write nothing
#       --with-eval     also take eval/ -- the per-pocket vina / posecheck /
#                       posebusters results behind each method's metrics.json
#       --with-raw      also take samples/outputs_*/  (DecompDiff: +3.45 GiB)
#       --with-shared   also take each touched task's _shared/ aggregates
#       --task <name>   restrict resolution to one task (task1-affinity |
#                       task2-drugdesign | task3-mcp); also disambiguates a name
#                       that exists in more than one task
#   -h, --help          this text
#   anything else       passed straight through to `rclone copy`
#
# Env:
#   RESULTS_DEST=/path  write under this dir instead of the script's own dir
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

BASE="dropbox:/박성현/VoxBind/results"
DEST="${RESULTS_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TASKS=(task1-affinity task2-drugdesign task3-mcp)
DEFAULT_METHODS=(AR Pocket2Mol DiffSBDD DecompDiff FuncBind)

DO_LIST=0; DO_ALL=0; WITH_RAW=0; WITH_EVAL=0; WITH_SHARED=0; TASK_FILTER=""
METHODS=(); PASSTHRU=()

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--list)     DO_LIST=1 ;;
    -a|--all)      DO_ALL=1 ;;
    -n|--dry-run)  PASSTHRU+=(--dry-run) ;;
    --with-eval)   WITH_EVAL=1 ;;
    --with-raw)    WITH_RAW=1 ;;
    --with-shared) WITH_SHARED=1 ;;
    --task)        TASK_FILTER="${2:?--task needs a value}"; shift ;;
    --task=*)      TASK_FILTER="${1#*=}" ;;
    -h|--help)     usage; exit 0 ;;
    -*)            PASSTHRU+=("$1") ;;
    *)             METHODS+=("$1") ;;
  esac
  shift
done

command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not in PATH"; exit 1; }
rclone listremotes 2>/dev/null | grep -qx "dropbox:" \
  || { echo "ERROR: rclone remote 'dropbox' not configured -- see notebook/html/dropbox-sync.md"; exit 1; }

if [[ -n "$TASK_FILTER" ]]; then
  printf '%s\n' "${TASKS[@]}" | grep -qx "$TASK_FILTER" \
    || { echo "ERROR: unknown --task '$TASK_FILTER' (expected: ${TASKS[*]})"; exit 1; }
fi

# --- index the remote once: every "<task>/<method>" ("_shared" is not a method) --
INDEX=()
for t in "${TASKS[@]}"; do
  [[ -n "$TASK_FILTER" && "$t" != "$TASK_FILTER" ]] && continue
  while IFS= read -r d; do
    d="${d%/}"
    [[ -z "$d" || "$d" == "_shared" || "$d" == _legacy* ]] && continue
    INDEX+=("$t/$d")
  done < <(rclone lsf "$BASE/$t/" --dirs-only --max-depth 1 2>/dev/null)
done
[[ ${#INDEX[@]} -gt 0 ]] || { echo "ERROR: remote listing came back empty -- is $BASE reachable?"; exit 1; }

if [[ "$DO_LIST" -eq 1 ]]; then
  echo ">> methods on $BASE"
  for t in "${TASKS[@]}"; do
    [[ -n "$TASK_FILTER" && "$t" != "$TASK_FILTER" ]] && continue
    echo "   $t/"
    for e in "${INDEX[@]}"; do [[ "$e" == "$t/"* ]] && echo "      ${e##*/}"; done
  done
  exit 0
fi

# --- decide which methods --------------------------------------------------
if [[ "$DO_ALL" -eq 1 ]]; then
  TARGETS=("${INDEX[@]}")
else
  [[ ${#METHODS[@]} -gt 0 ]] || METHODS=("${DEFAULT_METHODS[@]}")
  TARGETS=()
  for m in "${METHODS[@]}"; do
    hit=()
    for e in "${INDEX[@]}"; do [[ "${e##*/}" == "$m" ]] && hit+=("$e"); done
    if [[ ${#hit[@]} -eq 0 ]]; then   # retry case-insensitively
      shopt -s nocasematch
      for e in "${INDEX[@]}"; do [[ "${e##*/}" == "$m" ]] && hit+=("$e"); done
      shopt -u nocasematch
    fi
    if [[ ${#hit[@]} -eq 0 ]]; then
      echo "ERROR: no method named '$m' on the remote."
      near=$(printf '%s\n' "${INDEX[@]}" | grep -i -- "$m" || true)
      [[ -n "$near" ]] && { echo "   did you mean:"; printf '     %s\n' $near; } \
                       || echo "   run with -l to list what is there."
      exit 1
    fi
    if [[ ${#hit[@]} -gt 1 ]]; then
      echo "ERROR: '$m' exists in more than one task: ${hit[*]}"
      echo "   disambiguate with --task <name>"
      exit 1
    fi
    TARGETS+=("${hit[0]}")
  done
fi

# --- filters: ORDER MATTERS, first rule that matches wins ------------------
# (--include/--exclude are parsed in an indeterminate order and would let the
#  outputs_*/ exclusion lose to the samples/** inclusion -- rclone warns about
#  exactly this. --filter rules are applied strictly in the order given.)
FILTERS=( --filter "+ /metrics.json" --filter "+ /SOURCE.txt" )
[[ "$WITH_RAW"  -eq 1 ]] || FILTERS+=( --filter "- /samples/outputs_*/**" )
[[ "$WITH_EVAL" -ne 1 ]] || FILTERS+=( --filter "+ /eval/**" )
FILTERS+=( --filter "+ /samples/**" --filter "- **" )

human() { awk -v b="$1" 'BEGIN{ split("B KiB MiB GiB TiB",u," "); i=1;
  while (b>=1024 && i<5) { b/=1024; i++ } printf (i==1 ? "%d %s" : "%.2f %s"), b, u[i] }'; }

echo ">> source  : $BASE"
echo ">> dest    : $DEST"
echo ">> filter  : samples/** + metrics.json + SOURCE.txt$([[ "$WITH_EVAL" -eq 1 ]] && echo " + eval/**")$([[ "$WITH_RAW" -eq 1 ]] && echo "  (INCLUDING raw samples/outputs_*/)" || echo "  (raw samples/outputs_*/ skipped)")"
echo ">> methods : ${#TARGETS[@]}"
echo

total=0
for e in "${TARGETS[@]}"; do
  j=$(rclone size "$BASE/$e/" "${FILTERS[@]}" --json 2>/dev/null || echo '{}')
  c=$(printf '%s' "$j" | grep -o '"count":[0-9]*'  | head -1 | cut -d: -f2)
  b=$(printf '%s' "$j" | grep -o '"bytes":-\{0,1\}[0-9]*' | head -1 | cut -d: -f2)
  c="${c:-0}"; b="${b:-0}"; (( b > 0 )) && total=$(( total + b ))
  printf "   %-44s %5s files  %10s\n" "$e" "$c" "$(human "$b")"
done
printf "   %-44s %5s         %10s\n" "TOTAL" "" "$(human "$total")"
echo

for e in "${TARGETS[@]}"; do
  echo ">> $e"
  rclone copy "$BASE/$e/" "$DEST/$e/" "${FILTERS[@]}" \
    --transfers 4 --checkers 8 --progress ${PASSTHRU+"${PASSTHRU[@]}"}
done

if [[ "$WITH_SHARED" -eq 1 ]]; then
  for t in $(printf '%s\n' "${TARGETS[@]}" | cut -d/ -f1 | sort -u); do
    echo ">> $t/_shared"
    rclone copy "$BASE/$t/_shared/" "$DEST/$t/_shared/" \
      --transfers 4 --checkers 8 --progress ${PASSTHRU+"${PASSTHRU[@]}"}
  done
fi

echo
echo ">> done. verify integrity (hash compare) with:"
for e in "${TARGETS[@]}"; do
  echo "   rclone check \"$DEST/$e/\" \"$BASE/$e/\" --filter '+ /metrics.json' --filter '+ /SOURCE.txt' \\"
  echo "     $([[ "$WITH_RAW" -eq 1 ]] || echo "--filter '- /samples/outputs_*/**' ")$([[ "$WITH_EVAL" -eq 1 ]] && echo "--filter '+ /eval/**' ")--filter '+ /samples/**' --filter '- **' --one-way"
done
