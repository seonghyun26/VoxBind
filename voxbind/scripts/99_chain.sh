#!/usr/bin/env bash
# Wait for GPUs to remain idle, reserve them with advisory locks, then execute
# one command. The selected physical GPU IDs are exported through both
# VOXBIND_GPUS and CUDA_VISIBLE_DEVICES.
#
# Fixed GPU set:
#   bash scripts/99_chain.sh --gpus 4-7 -- \
#     bash scripts/03_pretrain.sh --experiment cha_gradmag
#
# First four available GPUs:
#   bash scripts/99_chain.sh --count 4 --timeout 24h -- \
#     bash scripts/03_pretrain.sh --experiment cha_gradmag -- num_epochs=200
#
# Wait for a dataset marker as well:
#   bash scripts/99_chain.sh --after-file dataset/data/READY --count 4 -- COMMAND
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/gpu.sh
source "$SCRIPT_DIR/lib/gpu.sh"

VOXBIND_CALLER="99_chain.sh"
VOX="$(voxbind_repo_root)"
GPU_SPEC=""
COUNT=1
COUNT_SET=0
MAX_USED_MIB=2500
MAX_UTIL=10
POLL=60
STABLE_CHECKS=2
TIMEOUT=0
LOCK_DIR="${VOXBIND_LOCK_DIR:-/tmp}"
CHDIR="$VOX"
RUN_LOG=""
DRY_RUN=0
COMMAND=()
SELECTED=()
GPU_LOCK_FDS=()
AFTER_FILES=()
AFTER_PIDS=()
CONDITION_SCRIPTS=()

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --gpus SPEC          Wait for this exact set, e.g. 0,1 or 4-7.
  --count N            Select the first N available GPUs (default: 1).
  --max-used-mib N     Maximum memory.used for an idle GPU (default: 2500).
  --max-util N         Maximum utilization.gpu percentage (default: 10).
  --stable-checks N    Required consecutive idle samples (default: 2).
  --poll DURATION      Poll interval: 30s, 2m, 1h (default: 60s).
  --timeout DURATION   Stop waiting after this duration; 0 disables (default).
  --after-file FILE    Also wait until FILE exists; may be repeated.
  --after-pid PID      Also wait until PID exits; may be repeated.
  --condition-script F Also require executable F to return zero; may be repeated.
  --lock-dir DIR       Advisory GPU lock directory (default: /tmp).
  --chdir DIR          Working directory for the command (default: repo root).
  --log FILE           Append command stdout/stderr to FILE.
  --dry-run            Validate and print without querying or launching.
  -- COMMAND [ARGS...] Command to execute; arguments are never eval'd.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPU_SPEC="${2:?--gpus requires a value}"; shift 2;;
        --count)         COUNT="${2:?--count requires a value}"; COUNT_SET=1; shift 2;;
        --max-used-mib)  MAX_USED_MIB="${2:?--max-used-mib requires a value}"; shift 2;;
        --max-util)      MAX_UTIL="${2:?--max-util requires a value}"; shift 2;;
        --stable-checks) STABLE_CHECKS="${2:?--stable-checks requires a value}"; shift 2;;
        --poll)          POLL="${2:?--poll requires a value}"; shift 2;;
        --timeout)       TIMEOUT="${2:?--timeout requires a value}"; shift 2;;
        --after-file)    AFTER_FILES+=("${2:?--after-file requires a value}"); shift 2;;
        --after-pid)     AFTER_PIDS+=("${2:?--after-pid requires a value}"); shift 2;;
        --condition-script)
                         CONDITION_SCRIPTS+=("${2:?--condition-script requires a value}"); shift 2;;
        --lock-dir)      LOCK_DIR="${2:?--lock-dir requires a value}"; shift 2;;
        --chdir)         CHDIR="${2:?--chdir requires a value}"; shift 2;;
        --log)           RUN_LOG="${2:?--log requires a value}"; shift 2;;
        --dry-run)       DRY_RUN=1; shift;;
        -h|--help)       usage; exit 0;;
        --)              shift; COMMAND=("$@"); break;;
        *)               voxbind_die "unknown argument '$1' (use --help)";;
    esac
done

[[ ${#COMMAND[@]} -gt 0 ]] || voxbind_die "missing command after --"
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || voxbind_die "--count must be a positive integer"
[[ "$MAX_USED_MIB" =~ ^[0-9]+$ ]] ||
    voxbind_die "--max-used-mib must be a non-negative integer"
if [[ ! "$MAX_UTIL" =~ ^[0-9]+$ ]] || (( MAX_UTIL > 100 )); then
    voxbind_die "--max-util must be an integer from 0 to 100"
fi
[[ "$STABLE_CHECKS" =~ ^[1-9][0-9]*$ ]] ||
    voxbind_die "--stable-checks must be a positive integer"
for pid in "${AFTER_PIDS[@]}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || voxbind_die "--after-pid must be a positive integer"
done

POLL_SECONDS="$(voxbind_parse_duration "$POLL")" ||
    voxbind_die "invalid --poll '$POLL' (use values such as 30s, 2m, or 1h)"
TIMEOUT_SECONDS="$(voxbind_parse_duration "$TIMEOUT")" ||
    voxbind_die "invalid --timeout '$TIMEOUT' (use 0, 30s, 2m, or 24h)"

FIXED_GPUS=""
if [[ -n "$GPU_SPEC" ]]; then
    FIXED_GPUS="$(voxbind_expand_gpus "$GPU_SPEC")" ||
        voxbind_die "invalid --gpus specification '$GPU_SPEC'"
    FIXED_COUNT="$(voxbind_gpu_count "$FIXED_GPUS")"
    if [[ "$COUNT_SET" -eq 1 && "$COUNT" -ne "$FIXED_COUNT" ]]; then
        voxbind_die "--count=$COUNT conflicts with the $FIXED_COUNT GPUs in --gpus=$GPU_SPEC"
    fi
    COUNT="$FIXED_COUNT"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ -n "$FIXED_GPUS" ]]; then
        DISPLAY_GPUS="$FIXED_GPUS"
    else
        DISPLAY_GPUS="<first-${COUNT}-free>"
    fi
    voxbind_log "dry-run: GPUs=$DISPLAY_GPUS mem<=${MAX_USED_MIB}MiB util<=${MAX_UTIL}%"
    voxbind_log "working directory: $CHDIR"
    for file in "${AFTER_FILES[@]}"; do voxbind_log "after-file: $file"; done
    for pid in "${AFTER_PIDS[@]}"; do voxbind_log "after-pid: $pid"; done
    for script in "${CONDITION_SCRIPTS[@]}"; do
        voxbind_log "condition-script: $script"
    done
    [[ -z "$RUN_LOG" ]] || voxbind_log "log: $RUN_LOG"
    voxbind_print_command "${COMMAND[@]}"
    exit 0
fi

command -v flock >/dev/null 2>&1 || voxbind_die "flock is required"
command -v "${VOXBIND_NVIDIA_SMI:-nvidia-smi}" >/dev/null 2>&1 ||
    voxbind_die "${VOXBIND_NVIDIA_SMI:-nvidia-smi} is required"
[[ -d "$CHDIR" ]] || voxbind_die "--chdir does not exist: $CHDIR"
mkdir -p "$LOCK_DIR"

release_gpu_locks() {
    local fd
    for fd in "${GPU_LOCK_FDS[@]}"; do
        if [[ -n "$fd" ]]; then
            flock -u "$fd" 2>/dev/null || true
            exec {fd}>&-
        fi
    done
    GPU_LOCK_FDS=()
    SELECTED=()
}

trap 'release_gpu_locks; exit 130' INT
trap 'release_gpu_locks; exit 143' TERM

conditions_ready() {
    local item path
    for item in "${AFTER_FILES[@]}"; do
        if [[ "$item" = /* ]]; then path="$item"; else path="$CHDIR/$item"; fi
        [[ -e "$path" ]] || return 1
    done
    for item in "${AFTER_PIDS[@]}"; do
        kill -0 "$item" 2>/dev/null && return 1
    done
    for item in "${CONDITION_SCRIPTS[@]}"; do
        if [[ "$item" = /* ]]; then
            [[ -x "$item" ]] || voxbind_die "condition script is not executable: $item"
            "$item" >/dev/null 2>&1 || return 1
        else
            [[ -x "$CHDIR/$item" ]] ||
                voxbind_die "condition script is not executable: $CHDIR/$item"
            (cd "$CHDIR" && "./$item") >/dev/null 2>&1 || return 1
        fi
    done
}

try_reserve() {
    local snapshot gpu fd queue_fd
    local -a candidate_array=()
    snapshot="$(voxbind_gpu_snapshot)" || return 1

    if [[ -n "$FIXED_GPUS" ]]; then
        IFS=',' read -r -a candidate_array <<< "$FIXED_GPUS"
    else
        mapfile -t candidate_array < <(
            voxbind_free_gpus "$snapshot" "$MAX_USED_MIB" "$MAX_UTIL"
        )
    fi
    (( ${#candidate_array[@]} >= COUNT )) || return 1

    exec {queue_fd}>"$LOCK_DIR/voxbind-gpu-selection.lock"
    flock "$queue_fd"
    release_gpu_locks
    for gpu in "${candidate_array[@]}"; do
        if ! voxbind_gpu_is_free "$snapshot" "$gpu" "$MAX_USED_MIB" "$MAX_UTIL"; then
            [[ -n "$FIXED_GPUS" ]] && break
            continue
        fi
        exec {fd}>"$LOCK_DIR/voxbind-gpu-${gpu}.lock"
        if flock -n "$fd"; then
            SELECTED+=("$gpu")
            GPU_LOCK_FDS+=("$fd")
        elif [[ -n "$FIXED_GPUS" ]]; then
            exec {fd}>&-
            break
        else
            exec {fd}>&-
        fi
        (( ${#SELECTED[@]} == COUNT )) && break
    done
    flock -u "$queue_fd"
    exec {queue_fd}>&-

    if (( ${#SELECTED[@]} != COUNT )); then
        release_gpu_locks
        return 1
    fi
}

selection_still_free() {
    local snapshot gpu
    snapshot="$(voxbind_gpu_snapshot)" || return 1
    for gpu in "${SELECTED[@]}"; do
        voxbind_gpu_is_free "$snapshot" "$gpu" "$MAX_USED_MIB" "$MAX_UTIL" || return 1
    done
}

STARTED_AT="$(date +%s)"
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    if conditions_ready && try_reserve; then
        STABLE=1
        while (( STABLE < STABLE_CHECKS )); do
            sleep "$POLL_SECONDS"
            if conditions_ready && selection_still_free; then
                STABLE=$((STABLE + 1))
            else
                release_gpu_locks
                break
            fi
        done
        (( STABLE == STABLE_CHECKS )) && break
    fi

    NOW="$(date +%s)"
    if (( TIMEOUT_SECONDS > 0 && NOW - STARTED_AT >= TIMEOUT_SECONDS )); then
        voxbind_die "timed out after ${TIMEOUT_SECONDS}s waiting for $COUNT free GPU(s)"
    fi
    if (( ATTEMPT == 1 || ATTEMPT % 10 == 0 )); then
        voxbind_log "waiting for $COUNT GPU(s): mem<=${MAX_USED_MIB}MiB util<=${MAX_UTIL}%"
    fi
    sleep "$POLL_SECONDS"
done

SELECTED_CSV="$(IFS=,; printf '%s' "${SELECTED[*]}")"
export VOXBIND_GPUS="$SELECTED_CSV"
export CUDA_VISIBLE_DEVICES="$SELECTED_CSV"
cd "$CHDIR" || voxbind_die "cannot cd to $CHDIR"

voxbind_log "launching on physical GPUs [$SELECTED_CSV]"
voxbind_print_command "${COMMAND[@]}"
if [[ -n "$RUN_LOG" ]]; then
    mkdir -p "$(dirname "$RUN_LOG")"
    exec "${COMMAND[@]}" >> "$RUN_LOG" 2>&1
else
    exec "${COMMAND[@]}"
fi
