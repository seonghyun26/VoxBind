#!/bin/bash
# 33a_watch_then_launch_atomblob_weighted.sh — chain the WEIGHTED atomblob
# ViT-MAE run onto GPUs 1-5 once the currently-running atomblob+density
# pretraining (260526_atomblob_density_vit_mae_40m_pretrain) finishes.
#
# Same pattern as scripts/32a_watch_then_launch_blobndensity.sh: poll for the
# final-epoch checkpoint; abort if the process exits without producing it.
#
# Log: voxbind/log/33a_watcher.log — process-level only; the child training
# script writes its own per-run log under log/260528_*.log.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
LOG=$VOX/log
WATCH_LOG=$LOG/33a_watcher.log

PRECEDING=260526_atomblob_density_vit_mae_40m_pretrain
PRECEDING_CKPT=$VOX/exps/$PRECEDING/checkpoint_e0099.pth.tar

WEIGHTED=260528_atomblob_vit_mae_40m_weighted_pretrain
WEIGHTED_CKPT=$VOX/exps/$WEIGHTED/checkpoint_e0099.pth.tar

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"

wait_for_completion() {
    local run_name="$1"
    local target_ckpt="$2"
    echo "[$(ts)] waiting for $run_name (gate: $target_ckpt)" >> "$WATCH_LOG"
    while true; do
        if [ -f "$target_ckpt" ]; then
            echo "[$(ts)] $run_name complete (checkpoint present)" >> "$WATCH_LOG"
            return 0
        fi
        if ! pgrep -af "$run_name" >/dev/null 2>&1; then
            echo "[$(ts)] ABORT: $run_name process gone but checkpoint missing" >> "$WATCH_LOG"
            return 1
        fi
        sleep 120
    done
}

echo "[$(ts)] watcher started (PID $$)" >> "$WATCH_LOG"

# Wait for the currently-running atomblob+density run to land its final ckpt.
wait_for_completion "$PRECEDING" "$PRECEDING_CKPT" || exit 1

# Grace period for DDP/wandb teardown + GPU memory release.
sleep 30

# Launch the weighted atomblob run (blocks until the training script exits).
echo "[$(ts)] launching 34_train_atomblob_vit_mae_40m_weighted.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/34_train_atomblob_vit_mae_40m_weighted.sh"
RC=$?
echo "[$(ts)] script 34 returned (exit $RC)" >> "$WATCH_LOG"

if [ $RC -ne 0 ] || [ ! -f "$WEIGHTED_CKPT" ]; then
    echo "[$(ts)] WARN: $WEIGHTED finished with rc=$RC, ckpt present=$([ -f "$WEIGHTED_CKPT" ] && echo y || echo n)" >> "$WATCH_LOG"
fi

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
