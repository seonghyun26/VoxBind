#!/bin/bash
# 32a_watch_then_launch_blobndensity.sh — chain the two new ViT-MAE
# pretraining runs (atomblob, then atomblob+density) onto the freed GPUs
# right after the currently-running density-only 40M baseline finishes.
#
# Same pattern as scripts/19a_watch_then_launch_voxbind_vit.sh:
#   1. Wait for 260526_density_vit_mae_40m_xray_pretrain to land its final
#      checkpoint (checkpoint_e0099.pth.tar). Abort if the process exits
#      without producing it (likely crash).
#   2. Launch 32_train_atomblob_vit_mae_40m.sh (blocks until that run ends).
#   3. If atomblob produced its e0099 checkpoint, launch
#      33_train_atomblob_density_vit_mae_40m.sh.
#
# Log: voxbind/log/32a_watcher.log — process-level only; per-run training
# logs go to log/260526_atomblob_*_40m_pretrain.log from each child script.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
LOG=$VOX/log
WATCH_LOG=$LOG/32a_watcher.log

BASELINE=260526_density_vit_mae_40m_xray_pretrain
BASELINE_CKPT=$VOX/exps/$BASELINE/checkpoint_e0099.pth.tar

ATOMBLOB=260526_atomblob_vit_mae_40m_pretrain
ATOMBLOB_CKPT=$VOX/exps/$ATOMBLOB/checkpoint_e0099.pth.tar

ATOMBLOB_DENS=260526_atomblob_density_vit_mae_40m_pretrain
ATOMBLOB_DENS_CKPT=$VOX/exps/$ATOMBLOB_DENS/checkpoint_e0099.pth.tar

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"

wait_for_completion() {
    # Block until $2 (checkpoint) exists. If at any point the process matching
    # $1 (output-dir name) is gone but the checkpoint isn't there, return 1
    # so the caller can abort.
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

# Phase 1: wait for the density-only baseline currently running on GPUs 1-5.
wait_for_completion "$BASELINE" "$BASELINE_CKPT" || exit 1

# Brief grace for the parent process to fully tear down (DDP/wandb cleanup,
# GPU memory release).
sleep 30

# Phase 2: launch atomblob (blocks until the training script exits).
echo "[$(ts)] launching 32_train_atomblob_vit_mae_40m.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/32_train_atomblob_vit_mae_40m.sh"
RC=$?
echo "[$(ts)] script 32 returned (exit $RC)" >> "$WATCH_LOG"

if [ $RC -ne 0 ] || [ ! -f "$ATOMBLOB_CKPT" ]; then
    echo "[$(ts)] ABORT: $ATOMBLOB did not produce final checkpoint (rc=$RC)" >> "$WATCH_LOG"
    exit 1
fi

sleep 30

# Phase 3: launch atomblob+density.
echo "[$(ts)] launching 33_train_atomblob_density_vit_mae_40m.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/33_train_atomblob_density_vit_mae_40m.sh"
RC=$?
echo "[$(ts)] script 33 returned (exit $RC)" >> "$WATCH_LOG"

if [ $RC -ne 0 ] || [ ! -f "$ATOMBLOB_DENS_CKPT" ]; then
    echo "[$(ts)] WARN: $ATOMBLOB_DENS finished with rc=$RC, ckpt present=$([ -f "$ATOMBLOB_DENS_CKPT" ] && echo y || echo n)" >> "$WATCH_LOG"
fi

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
