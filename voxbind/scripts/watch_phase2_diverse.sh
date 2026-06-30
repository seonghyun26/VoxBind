#!/usr/bin/env bash
# watch_phase2_diverse.sh — 260624
# Wait until GPUs 4-7 are ALL free, then launch the Phase-2 ROLE-SPLIT DIVERSE-ATOM
# ChannelViT PLINDER-OTF pretrain on exactly that 4-GPU set (pinned, not first-free).
#
#   role-split C+D+G, ChannelViT [1,1,1,1], diverse-atom corpus
#   (data_train_plinder_diverse.pt, 17,530 train positions, per-atom vdW radius)
#   → exps/260623_plinder_otf_roleblob_diverse_cdg_channelvit_mask050_pretrain
#
# Why a dedicated watcher (not queue_when_gpus_free.sh): that grabs the first N free
# GPUs globally, so it could land on 0-3 if those free first. The user asked for 4-7
# specifically (0-3 run the ar_cvit_hcs job), so we gate on {4,5,6,7} and pin them.
set -u
VOX="${VOXBIND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${VOXBIND_PY:-/home/shpark/.conda/envs/voxbind/bin}"
LOG="$VOX/log"
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
EXP=260623_plinder_otf_roleblob_diverse_cdg_channelvit_mask050_pretrain
GPUS="${GPUS:-4,5,6,7}"
NPROC=4
MEM_FREE_MIB="${MEM_FREE_MIB:-2500}"   # a GPU counts as free if memory.used < this
POLL="${POLL:-120}"                    # seconds between checks
STABLE="${STABLE:-2}"                  # require this many consecutive all-free polls
RDZV_PORT="${RDZV_PORT:-29561}"        # distinct from the ar_cvit job (29541)

mkdir -p "$LOG"
WLOG="$LOG/260624_phase2_diverse_watch.log"
TLOG="$LOG/${EXP}.log"
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
say(){ echo "[$(ts)] $*" | tee -a "$WLOG"; }

# guard: don't double-launch if this pretrain is already running or already finished e99
if pgrep -f "$CFG" >/dev/null 2>&1; then
    say "ABORT: a process for $CFG is already running"; exit 1
fi
if [ -f "$VOX/exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    say "ABORT: $EXP already has checkpoint_e0099 — nothing to do"; exit 0
fi

say "watcher start — waiting for GPUs [$GPUS] all <${MEM_FREE_MIB} MiB (poll ${POLL}s, need ${STABLE}x stable)"
ok=0
while true; do
    busy=""
    for g in ${GPUS//,/ }; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
        if [ -z "$m" ] || [ "$m" -ge "$MEM_FREE_MIB" ]; then busy="$busy $g(${m:-NA}MiB)"; fi
    done
    if [ -z "$busy" ]; then
        ok=$((ok+1)); say "all of [$GPUS] free (${ok}/${STABLE})"
        [ "$ok" -ge "$STABLE" ] && break
    else
        [ "$ok" -ne 0 ] && say "reset: still busy →$busy"
        ok=0
    fi
    sleep "$POLL"
done

say "launching $EXP on GPUs [$GPUS]  → $TLOG"
cd "$VOX" || { say "ERROR: cd $VOX failed"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPUS" "$PY/torchrun" \
    --nnodes=1 --nproc_per_node="$NPROC" \
    --rdzv-backend=c10d --rdzv-endpoint="localhost:${RDZV_PORT}" --rdzv-id=roleblobdiv \
    train_density.py --config-name="$CFG" >> "$TLOG" 2>&1
rc=$?
say "$EXP finished (exit $rc)  →  exps/$EXP"
