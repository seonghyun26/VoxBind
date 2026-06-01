#!/bin/bash
# 37a_watch_then_launch_dualhead.sh — chain the dual_head + bigger-head + atom_pos
# variant (script 38) onto GPUs 4-7 once the current v4 single-head pretraining
# (260601_atomblob_merged_density_vit_mae_40m_weighted_v4_pretrain) finishes.
#
# Same pattern as scripts/32a_watch_then_launch_blobndensity.sh: poll for the
# final-epoch checkpoint; abort if the process exits without producing it.
#
# Log: voxbind/log/37a_watcher.log — process-level only; child training log
# goes to log/260602_*_dualhead_pretrain.log.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
LOG=$VOX/log
WATCH_LOG=$LOG/37a_watcher.log

PRECEDING=260601_atomblob_merged_density_vit_mae_40m_weighted_v4_pretrain
PRECEDING_CKPT=$VOX/exps/$PRECEDING/checkpoint_e0099.pth.tar

DUALHEAD=260602_atomblob_merged_density_vit_mae_40m_weighted_v4_dualhead_pretrain
DUALHEAD_CKPT=$VOX/exps/$DUALHEAD/checkpoint_e0099.pth.tar

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

# Wait for v4 single-head to land its final ckpt
wait_for_completion "$PRECEDING" "$PRECEDING_CKPT" || exit 1

# Grace period for DDP/wandb teardown + GPU memory release
sleep 30

# Launch dual_head + bigger head + atom_pos_weight run (blocks until done)
echo "[$(ts)] launching 38_train_atomblob_merged_density_v4_dualhead.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/38_train_atomblob_merged_density_v4_dualhead.sh"
RC=$?
echo "[$(ts)] script 38 returned (exit $RC)" >> "$WATCH_LOG"

if [ $RC -ne 0 ] || [ ! -f "$DUALHEAD_CKPT" ]; then
    echo "[$(ts)] WARN: $DUALHEAD finished with rc=$RC, ckpt present=$([ -f "$DUALHEAD_CKPT" ] && echo y || echo n)" >> "$WATCH_LOG"
fi

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
