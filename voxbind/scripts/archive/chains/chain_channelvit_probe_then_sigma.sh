#!/usr/bin/env bash
# Orchestrate: wait for the ChannelViT pretraining to finish -> probe binding
# affinity (vs the fused atomblob_density_gradmag baseline, same test pool) ->
# resume the sig=1.0 VoxBind denoiser to 1200 epochs.
# Launch detached:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_channelvit_probe_then_sigma.sh \
#     > log/chain_channelvit_probe_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
EXP=atomblob_density_gradmag_channelvit_g7411_40m_v5
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
cd "$ROOT" || exit 1
# v5 voxel caches are the legacy schema (no metadata_version) — opt into the
# warn-not-raise path so fresh feature extraction works (same data the fused
# atomblob_density_gradmag=0.477 baseline used).
export VOXBIND_ALLOW_LEGACY_VOXEL_META=1

# ── 1) wait for the ChannelViT pretraining process to exit ───────────────────
echo "[$(ts)] CHAIN START — waiting for ChannelViT pretraining ($EXP)..."
while pgrep -f "exp_name=$EXP" >/dev/null 2>&1; do sleep 60; done
last=$(grep -oE "epoch: [0-9]+" "$LOG/${EXP}.log" 2>/dev/null | tail -1)
echo "[$(ts)] pretraining process gone (last $last)"

# ── 2) probe binding affinity (only if the e99 encoder exists) ───────────────
if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
  echo "[$(ts)] ERROR: exps/$EXP/checkpoint_e0099.pth.tar missing — skipping probe"
else
  echo "[$(ts)] extracting ChannelViT frozen features (GPU 0)..."
  CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
    --condition atomblob_density_gradmag_channelvit --voxel_version v5 --epoch 99 \
    --device cuda --batch_size 32 > "$LOG/probe_features_${EXP}.log" 2>&1
  echo "[$(ts)] features exit=$? ($(grep -oE 'saved [0-9,]+ features' "$LOG/probe_features_${EXP}.log" | tail -1))"

  echo "[$(ts)] probing channelvit vs fused atomblob_density_gradmag (3 seeds, CPU)..."
  CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
    --conditions atomblob_density_gradmag atomblob_density_gradmag_channelvit \
    --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag channelvit \
    --allow_stale_features > "$LOG/probe_${EXP}.log" 2>&1
  echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_channelvit.csv"
fi

# ── 3) resume the sig=1.0 VoxBind denoiser (ep707 -> 1200) ────────────────────
echo "[$(ts)] resuming sig=1.0 denoiser (resume_epoch=707, +493 -> 1200)"
SIG=1.0 NUM_EPOCHS=493 RESUME_EPOCH=707 bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
  > "$LOG/sig1.0_resume_after_channelvit.log" 2>&1
echo "[$(ts)] sig=1.0 launcher exit=$?  -> $LOG/sig1.0_resume_after_channelvit.log"
echo "[$(ts)] CHAIN COMPLETE"
