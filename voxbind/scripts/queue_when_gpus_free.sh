#!/bin/bash
# queue_when_gpus_free.sh — wait for a free N-GPU set, then launch a train_density.py
# pretrain on it. Generic queue utility (no contention with running jobs).
#
#   bash scripts/queue_when_gpus_free.sh <config-name> <exp-name> [nproc]
#   env: MEM_FREE_MIB (default 2500 = "GPU free if used<this"), POLL (default 120s)
set -u
VOX="${VOXBIND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${VOXBIND_PY:-/home/shpark/.conda/envs/voxbind/bin}"
LOG="$VOX/log"
CFG="${1:?usage: queue_when_gpus_free.sh <config-name> <exp-name> [nproc]}"
EXP="${2:?need exp-name}"
NPROC="${3:-4}"
MEM_FREE_MIB="${MEM_FREE_MIB:-2500}"
POLL="${POLL:-120}"
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"

echo "[$(ts)] queue: $EXP — waiting for $NPROC GPUs with used<${MEM_FREE_MIB} MiB (poll ${POLL}s)"
while true; do
    mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | awk -v t="$MEM_FREE_MIB" -F, '($2+0) < t {print $1+0}')
    if [ "${#FREE[@]}" -ge "$NPROC" ]; then
        SET=$(IFS=,; echo "${FREE[*]:0:NPROC}")
        echo "[$(ts)] $NPROC GPUs free: [$SET] — launching $EXP → $LOG/${EXP}.log"
        cd "$VOX" || exit 1
        CUDA_VISIBLE_DEVICES="$SET" "$PY/torchrun" --standalone --nproc_per_node="$NPROC" \
            train_density.py --config-name="$CFG" >> "$LOG/${EXP}.log" 2>&1
        echo "[$(ts)] $EXP finished (exit $?)  ->  exps/$EXP"
        break
    fi
    sleep "$POLL"
done
