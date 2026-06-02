#!/bin/bash
# 37a_watch_then_launch_v2.sh — chain the V2 retrain onto GPU 4-7 once the
# currently-running 260530 atomblob_merged_density weighted run finishes.
#
# Three sequential gates:
#   1. wait for exps/260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain/checkpoint_e0099.pth.tar
#   2. run dataset/00e_v2_pool_norm_crossdocked.py (~1 hour, produces v2 + v3 crops)
#   3. launch scripts/37_train_atomblob_merged_density_vit_mae_40m_weighted_v2.sh
#
# Logs:
#   voxbind/log/37a_watcher.log            — watcher-level events
#   voxbind/log/00e_v2_precompute.log      — precompute output
#   voxbind/log/260530_*_v2_pretrain.log   — training script writes its own
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
LOG=$VOX/log
WATCH_LOG=$LOG/37a_watcher.log

PRECEDING=260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain
PRECEDING_CKPT=$VOX/exps/$PRECEDING/checkpoint_e0099.pth.tar

V2=260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain
V2_CKPT=$VOX/exps/$V2/checkpoint_e0099.pth.tar

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"
echo "[$(ts)] watcher started (PID $$)" >> "$WATCH_LOG"

# ── 1. Wait for the preceding run's final checkpoint ───────────────────────────
echo "[$(ts)] waiting for $PRECEDING (gate: $PRECEDING_CKPT)" >> "$WATCH_LOG"
while true; do
    if [ -f "$PRECEDING_CKPT" ]; then
        echo "[$(ts)] $PRECEDING complete (checkpoint present)" >> "$WATCH_LOG"
        break
    fi
    if ! pgrep -af "$PRECEDING" >/dev/null 2>&1; then
        echo "[$(ts)] ABORT: $PRECEDING process gone but checkpoint missing" >> "$WATCH_LOG"
        exit 1
    fi
    sleep 120
done

# Grace period for DDP / wandb teardown + GPU memory release.
sleep 30

# ── 2. Run v2 precompute (produces v2 AND v3 in one pass) ──────────────────────
echo "[$(ts)] running 00e_v2_pool_norm_crossdocked.py" >> "$WATCH_LOG"
cd "$VOX" || exit 1
$PY/python dataset/00e_v2_pool_norm_crossdocked.py \
    > "$LOG/00e_v2_precompute.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "[$(ts)] ABORT: precompute failed (rc=$RC); see $LOG/00e_v2_precompute.log" >> "$WATCH_LOG"
    exit 1
fi
echo "[$(ts)] precompute complete" >> "$WATCH_LOG"

# Quick sanity check: stats.json present + non-trivial sample count
STATS=$VOX/dataset/data/xray_crops_aligned_v2/stats.json
if [ ! -f "$STATS" ]; then
    echo "[$(ts)] ABORT: $STATS missing after precompute" >> "$WATCH_LOG"
    exit 1
fi

# ── 3. Launch v2 training ──────────────────────────────────────────────────────
echo "[$(ts)] launching scripts/37_train_atomblob_merged_density_vit_mae_40m_weighted_v2.sh" >> "$WATCH_LOG"
bash "$VOX/scripts/37_train_atomblob_merged_density_vit_mae_40m_weighted_v2.sh"
RC=$?
echo "[$(ts)] script 37 returned (exit $RC)" >> "$WATCH_LOG"

if [ $RC -ne 0 ] || [ ! -f "$V2_CKPT" ]; then
    echo "[$(ts)] WARN: $V2 finished with rc=$RC, ckpt present=$([ -f "$V2_CKPT" ] && echo y || echo n)" >> "$WATCH_LOG"
fi

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
