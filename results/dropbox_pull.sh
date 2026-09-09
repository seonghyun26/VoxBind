#!/usr/bin/env bash
# Download the whole results bundle or selected model folders from Dropbox.
# Copies are incremental and never delete local files.
set -euo pipefail
export PATH="$PATH:$HOME/.local/bin"

SRC="dropbox:/박성현/VoxBind/results"
DEST="${RESULTS_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TASKS=(task1-affinity task2-drugdesign task3-mcp)
METHODS=()
RCLONE_ARGS=()
TASK_FILTER=""
DO_LIST=0
DO_ALL=0
EXCLUDES=(
  --exclude "/dropbox_push.sh"
  --exclude "/dropbox_pull.sh"
  --exclude "/dropbox_pull_baselines.sh"
  --exclude "/README.md"
  --exclude ".gitignore"
)

usage() {
  printf '%s\n' \
    'Usage: bash results/dropbox_pull.sh [MODEL ...] [options] [-- rclone flags]' \
    '' \
    'No model/task selector: download the entire bundle (legacy behavior).' \
    'MODEL ...                 Download only these model folders, across tasks.' \
    '--method MODEL            Same as a positional model; repeatable.' \
    '--model MODEL             Alias for --method.' \
    '--task TASK               Restrict to one task; without models, copy that task.' \
    '-l, --list                List remote model folders without downloading.' \
    '-a, --all                 Explicitly request all models; cannot mix with MODEL.' \
    '-n, --dry-run             Preview without writing files.' \
    '-h, --help                Show this help without accessing Dropbox.' \
    '--                        Pass remaining arguments directly to rclone.' \
    '' \
    'Examples:' \
    '  bash results/dropbox_pull.sh VoxBind-Ours' \
    '  bash results/dropbox_pull.sh VoxBind VoxBind-Ours' \
    '  bash results/dropbox_pull.sh --task task2-drugdesign --list' \
    '  bash results/dropbox_pull.sh VoxBind-Ours --dry-run' \
    '  bash results/dropbox_pull.sh VoxBind-Ours -- --checksum --transfers 8' \
    '  RESULTS_DEST=/data/results bash results/dropbox_pull.sh VoxBind-Ours' \
    '' \
    'Model names are case-insensitive; quote names containing spaces.' \
    'All model artifacts are included, unlike the narrower dropbox_pull_baselines.sh.' \
    'Place selectors before extra rclone flags. Unknown options and everything' \
    'after them are forwarded to rclone for compatibility with the old script.'
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -l|--list) DO_LIST=1; shift ;;
    -a|--all) DO_ALL=1; shift ;;
    -n|--dry-run) RCLONE_ARGS+=(--dry-run); shift ;;
    --method|--model|--task)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || fail "$1 needs a value"
      if [[ "$1" == --task ]]; then TASK_FILTER="$2"; else METHODS+=("$2"); fi
      shift 2 ;;
    --method=*|--model=*) METHODS+=("${1#*=}"); shift ;;
    --task=*) TASK_FILTER="${1#*=}"; [[ -n "$TASK_FILTER" ]] || fail "--task needs a value"; shift ;;
    --) shift; RCLONE_ARGS+=("$@"); break ;;
    -*) RCLONE_ARGS+=("$@"); break ;;
    *) METHODS+=("$1"); shift ;;
  esac
done

if [[ -n "$TASK_FILTER" ]]; then
  case "$TASK_FILTER" in
    task1-affinity|task2-drugdesign|task3-mcp) ;;
    *) fail "Unknown task: $TASK_FILTER" ;;
  esac
fi
(( ! DO_ALL || ${#METHODS[@]} == 0 )) || fail "--all cannot be combined with model names"
(( ! DO_LIST || ${#METHODS[@]} == 0 )) || fail "--list cannot be combined with model names"
for method in "${METHODS[@]}"; do
  [[ -n "$method" && "$method" != */* && "$method" != "." && "$method" != ".." && "$method" != -* ]] \
    || fail "Expected a bare model folder name, not a path: $method"
done

command -v rclone >/dev/null 2>&1 || fail "rclone is not in PATH"
remotes=$(rclone listremotes) || fail "Cannot read rclone remotes"
has_dropbox=0
while IFS= read -r remote; do
  if [[ "$remote" == "dropbox:" ]]; then has_dropbox=1; fi
done <<< "$remotes"
(( has_dropbox )) || fail "rclone remote 'dropbox' is not configured; see notebook/html/dropbox-sync.md"

# With no model selector, retain the full-bundle operation without a remote scan.
if (( ! DO_LIST && ${#METHODS[@]} == 0 )); then
  source_path="$SRC"
  dest_path="$DEST"
  if [[ -n "$TASK_FILTER" ]]; then
    source_path="$SRC/$TASK_FILTER"
    dest_path="$DEST/$TASK_FILTER"
  fi
  printf '>> downloading %s/ -> %s/\n' "$source_path" "$dest_path"
  rclone copy "$source_path/" "$dest_path/" "${EXCLUDES[@]}" \
    --transfers 4 --checkers 8 --progress "${RCLONE_ARGS[@]}"
  exit 0
fi

# Resolve every requested name before starting any copy. Do not hide listing errors.
INDEX=()
for task in "${TASKS[@]}"; do
  [[ -z "$TASK_FILTER" || "$task" == "$TASK_FILTER" ]] || continue
  listing=$(rclone lsf "$SRC/$task/" --dirs-only --max-depth 1) \
    || fail "Cannot list $SRC/$task/"
  while IFS= read -r folder; do
    folder="${folder%/}"
    [[ -n "$folder" && "$folder" != _* ]] || continue
    [[ "$folder" != */* && "$folder" != "." && "$folder" != ".." ]] \
      || fail "Unexpected remote folder: $folder"
    INDEX+=("$task/$folder")
  done <<< "$listing"
done

if (( DO_LIST )); then
  printf '>> Models on %s\n' "$SRC"
  if (( ${#INDEX[@]} )); then printf '  %s\n' "${INDEX[@]}"; else printf '  (none)\n'; fi
  exit 0
fi

TARGETS=()
for method in "${METHODS[@]}"; do
  matches=()
  for entry in "${INDEX[@]}"; do
    folder="${entry##*/}"
    [[ "${folder,,}" == "${method,,}" ]] && matches+=("$entry")
  done
  (( ${#matches[@]} > 0 )) || fail "Model '$method' was not found. Use --list."
  (( ${#matches[@]} == 1 )) || fail "Model '$method' exists in multiple tasks: ${matches[*]}. Use --task."
  already_selected=0
  for entry in "${TARGETS[@]}"; do
    [[ "$entry" != "${matches[0]}" ]] || already_selected=1
  done
  (( already_selected )) || TARGETS+=("${matches[0]}")
done

printf '>> Selected model folders:\n'
printf '  %s\n' "${TARGETS[@]}"
for entry in "${TARGETS[@]}"; do
  printf '>> downloading %s/ -> %s/\n' "$SRC/$entry" "$DEST/$entry"
  rclone copy "$SRC/$entry/" "$DEST/$entry/" \
    --exclude ".gitignore" --transfers 4 --checkers 8 --progress "${RCLONE_ARGS[@]}"
done
printf '>> Finished. Existing unrelated models were not changed.\n'
