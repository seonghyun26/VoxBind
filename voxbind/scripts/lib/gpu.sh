#!/usr/bin/env bash
# GPU inspection helpers used by scripts/99_chain.sh.

if ! declare -F voxbind_die >/dev/null 2>&1; then
    # shellcheck source=common.sh
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

voxbind_gpu_snapshot() {
    local nvidia_smi="${VOXBIND_NVIDIA_SMI:-nvidia-smi}"
    "$nvidia_smi" \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F, '{
            gsub(/[[:space:]]/, "", $1)
            gsub(/[[:space:]]/, "", $2)
            gsub(/[[:space:]]/, "", $3)
            print $1, $2, $3
        }'
}

voxbind_gpu_is_free() {
    local snapshot="$1" gpu="$2" max_used_mib="$3" max_util="$4"
    awk -v gpu="$gpu" -v max_mem="$max_used_mib" -v max_util="$max_util" '
        $1 == gpu {
            found = 1
            if (($2 + 0) <= max_mem && ($3 + 0) <= max_util) ok = 1
        }
        END { exit !(found && ok) }
    ' <<< "$snapshot"
}

voxbind_free_gpus() {
    local snapshot="$1" max_used_mib="$2" max_util="$3"
    awk -v max_mem="$max_used_mib" -v max_util="$max_util" '
        ($2 + 0) <= max_mem && ($3 + 0) <= max_util { print $1 }
    ' <<< "$snapshot"
}
