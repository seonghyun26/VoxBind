#!/usr/bin/env bash
# 71_supervise_voxbind_base_8gpu.sh
#   Crash-resume supervisor around 70_train_voxbind_base_8gpu.sh.
#
#   This box is shared and has been OOM-killed repeatedly (the PoseCheck eval
#   fork-bombed it to load-avg 22,730 on 2026-08-27 and the kernel SIGKILL'd all
#   8 training ranks at 16:18). Losing a 77-hour run to someone else's job is the
#   failure mode this guards against: on any non-zero exit it waits for the box to
#   settle, then relaunches from the last checkpoint.
#
#   TOTAL_EPOCHS is the ABSOLUTE target (epoch indices 0..TOTAL_EPOCHS-1). Because
#   train_ddp.py's loop is range(start_epoch, start_epoch + num_epochs) and
#   load_checkpoint returns the checkpoint's own epoch, a resume needs
#   num_epochs = TOTAL_EPOCHS - ckpt_epoch to land on the same final epoch.
#
#   Env knobs: EXP_NAME, TOTAL_EPOCHS, MAX_RETRIES, SETTLE_SECONDS, MIN_FREE_GB.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP_NAME="${EXP_NAME:-260827_voxbind_base_8gpu}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-350}"
MAX_RETRIES="${MAX_RETRIES:-20}"
SETTLE_SECONDS="${SETTLE_SECONDS:-180}"
MIN_FREE_GB="${MIN_FREE_GB:-200}"

OUT="$ROOT/voxbind/exps/$EXP_NAME"
CKPT="$OUT/checkpoint.pth.tar"
SLOG="$ROOT/voxbind/logs/${EXP_NAME}.supervisor.log"
mkdir -p "$ROOT/voxbind/logs"

log() { echo "[$(date '+%F %T')] $*" >> "$SLOG"; }

ckpt_epoch() {
    [ -f "$CKPT" ] || { echo -1; return; }
    "$HOME/miniforge3/envs/voxbind/bin/python" - "$CKPT" <<'PY' 2>/dev/null || echo -1
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"]))
except Exception:
    print(-1)
PY
}

log "supervisor start: exp=$EXP_NAME total_epochs=$TOTAL_EPOCHS max_retries=$MAX_RETRIES"

for (( attempt=0; attempt<=MAX_RETRIES; attempt++ )); do
    e=$(ckpt_epoch)
    if [ "$attempt" -eq 0 ] && [ "$e" -lt 0 ]; then
        log "attempt $attempt: fresh start, $TOTAL_EPOCHS epochs"
        EXP_NAME="$EXP_NAME" NUM_EPOCHS="$TOTAL_EPOCHS" \
            bash scripts/70_train_voxbind_base_8gpu.sh
        rc=$?
    elif [ "$e" -ge $((TOTAL_EPOCHS - 1)) ]; then
        log "checkpoint already at epoch $e >= $((TOTAL_EPOCHS - 1)) — training complete"
        exit 0
    else
        remaining=$((TOTAL_EPOCHS - e))
        log "attempt $attempt: resuming from epoch $e, $remaining epochs remaining"
        EXP_NAME="$EXP_NAME" NUM_EPOCHS="$remaining" RESUME="exps/$EXP_NAME" \
            bash scripts/70_train_voxbind_base_8gpu.sh
        rc=$?
    fi

    if [ "$rc" -eq 0 ]; then
        log "training exited cleanly (rc=0) after attempt $attempt"
        exit 0
    fi

    log "training died (rc=$rc) on attempt $attempt; ckpt_epoch=$(ckpt_epoch)"
    [ "$attempt" -ge "$MAX_RETRIES" ] && { log "max retries reached — giving up"; exit 1; }

    # Let whatever killed us drain before piling back on.
    log "settling for ${SETTLE_SECONDS}s"
    sleep "$SETTLE_SECONDS"
    while :; do
        free_gb=$(free -g | awk '/^Mem:/{print $7}')
        gpu_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
        [ "$free_gb" -ge "$MIN_FREE_GB" ] && [ "$gpu_busy" -eq 0 ] && break
        log "waiting: available=${free_gb}GB (need ${MIN_FREE_GB}) gpu_procs=${gpu_busy}"
        sleep 60
    done
done
