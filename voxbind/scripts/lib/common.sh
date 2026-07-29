#!/usr/bin/env bash
# Shared shell helpers for VoxBind launchers. Source this file; do not execute it.

voxbind_repo_root() {
    if [[ -n "${VOXBIND_ROOT:-}" ]]; then
        printf '%s\n' "$VOXBIND_ROOT"
    else
        cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
    fi
}

voxbind_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

voxbind_log() {
    printf '[%s] %s\n' "$(voxbind_timestamp)" "$*"
}

voxbind_die() {
    printf '%s: %s\n' "${VOXBIND_CALLER:-voxbind}" "$*" >&2
    exit 1
}

voxbind_expand_gpus() {
    local spec="${1//[[:space:]]/}" item lo hi gpu
    local -a items=() result=()
    local -A seen=()

    [[ -n "$spec" ]] || return 1
    IFS=',' read -r -a items <<< "$spec"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo="${BASH_REMATCH[1]}"
            hi="${BASH_REMATCH[2]}"
            (( lo <= hi )) || return 1
            for ((gpu=lo; gpu<=hi; gpu++)); do
                [[ -z "${seen[$gpu]:-}" ]] || continue
                result+=("$gpu")
                seen[$gpu]=1
            done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            [[ -z "${seen[$item]:-}" ]] || continue
            result+=("$item")
            seen[$item]=1
        else
            return 1
        fi
    done

    local IFS=,
    printf '%s\n' "${result[*]}"
}

voxbind_gpu_count() {
    local expanded
    expanded="$(voxbind_expand_gpus "$1")" || return 1
    awk -F, '{print NF}' <<< "$expanded"
}

voxbind_parse_duration() {
    local value="$1" amount unit multiplier
    [[ "$value" =~ ^([0-9]+)([smhd]?)$ ]] || return 1
    amount="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "$unit" in
        ""|s) multiplier=1;;
        m) multiplier=60;;
        h) multiplier=3600;;
        d) multiplier=86400;;
        *) return 1;;
    esac
    printf '%s\n' "$((amount * multiplier))"
}

voxbind_print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}
