#!/bin/bash
# 19a_watch_then_launch_voxbind_vit.sh — wait for the ViT-MAE pretrain
# (260522_density_vit_mae_pretrain) to finish, then launch
# 19_train_10k_density_vit_mae.sh on the freed GPUs.
#
# Termination condition: no more `train_density_vit_mae.py` processes AND
# the final checkpoint (checkpoint_e0099.pth.tar) is on disk. If the process
# disappears without the checkpoint (crash / OOM), abort instead of launching
# downstream training against a partial pretrain.
#
# Log: voxbind/log/19a_watcher.log
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
LOG=$VOX/log
CKPT=$VOX/exps/260522_density_vit_mae_pretrain/checkpoint_e0099.pth.tar
WATCH_LOG=$LOG/19a_watcher.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
echo "[$(ts)] watcher started; waiting for train_density_vit_mae.py to finish" >> "$WATCH_LOG"
echo "[$(ts)]   condition: process gone AND $CKPT exists" >> "$WATCH_LOG"

# Wait until the pretrain process is gone. Poll every 2 min — the run takes
# hours, no point checking faster.
while pgrep -f train_density_vit_mae.py >/dev/null 2>&1; do
    sleep 120
done
echo "[$(ts)] train_density_vit_mae.py no longer running" >> "$WATCH_LOG"

# Brief grace for the final checkpoint rename (os.replace is atomic but the
# parent training process might be tearing down DDP/wandb when we poll).
sleep 5

if [ ! -f "$CKPT" ]; then
    echo "[$(ts)] ABORT: pretrain process gone but $CKPT not present" >> "$WATCH_LOG"
    echo "[$(ts)] (likely a crash; check log/260522_density_vit_mae_pretrain.log)" >> "$WATCH_LOG"
    exit 1
fi

echo "[$(ts)] pretrain complete; launching 19_train_10k_density_vit_mae.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/19_train_10k_density_vit_mae.sh" >> "$WATCH_LOG" 2>&1
echo "[$(ts)] 19_train_10k_density_vit_mae.sh returned (exit $?)" >> "$WATCH_LOG"
