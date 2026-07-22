#!/usr/bin/env bash
# Density-ablation controls, then resume the sig=1.0 denoiser.
#   Exp1: atoms + NOISE density          (12ch, config_train_atomblob_noisedensity_vit_mae_40m_v5)
#   Exp2: atoms + NOISE density + gradmag (13ch, gradmag DERIVED from the noise density)
# Both on GPUs 0-3, one at a time; then probe both vs the real-density baselines;
# then resume sig=1.0 to 1200.
#
# NOTE: assumes sig=1.0 has ALREADY been stopped and GPUs are free before launch
# (done manually so the checkpoint is clean and GPUs verified empty).
# Launch:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/chain_noisedensity_then_sigma.sh \
#     > log/chain_noisedensity_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
NOISE_PROBE=$ROOT/dataset/data/pdbbind/voxels_v5_noise
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
cd "$ROOT" || exit 1
export VOXBIND_ALLOW_LEGACY_VOXEL_META=1

train() {  # $1=config-name  $2=exp_name
  local CFG=$1 EXP=$2 OUT=$ROOT/exps/$2
  if pgrep -f "exp_name=$EXP" >/dev/null 2>&1; then echo "[$(ts)] SKIP $EXP (running)"; return 0; fi
  mkdir -p "$OUT"
  echo "[$(ts)] START $EXP ($CFG)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
    "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
    exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
    > "$LOG/${EXP}.log" 2>&1
  echo "[$(ts)] END $EXP exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"
}

# ── 1) two ablation trainings, sequentially ──────────────────────────────────
train config_train_atomblob_noisedensity_vit_mae_40m_v5         atomblob_noisedensity_vit_mae_40m_v5
train config_train_atomblob_noisedensity_gradmag_vit_mae_40m_v5 atomblob_noisedensity_gradmag_vit_mae_40m_v5

# ── 2) features (GPU now free) — noise voxels; exp2 DERIVES gradmag from noise ─
echo "[$(ts)] extracting features (noise voxels)..."
CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
  --condition atomblob_noisedensity --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
  --noise_voxels_dir "$NOISE_PROBE" > "$LOG/probe_features_atomblob_noisedensity.log" 2>&1
echo "[$(ts)] exp1 features exit=$?"
CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
  --condition atomblob_noisedensity_gradmag --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
  --noise_density_dir "$NOISE_PROBE/density" > "$LOG/probe_features_atomblob_noisedensity_gradmag.log" 2>&1
echo "[$(ts)] exp2 features exit=$?"

# ── 3) one probe over noise + real baselines (same intersected test pool) ─────
echo "[$(ts)] probing noise vs real (3 seeds, CPU)..."
CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density atomblob_noisedensity atomblob_density_gradmag atomblob_noisedensity_gradmag \
  --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag noisedensity --allow_stale_features \
  > "$LOG/probe_noisedensity.log" 2>&1
echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_noisedensity.csv"

# ── 4) resume sig=1.0 denoiser to 1200 (dynamic resume epoch from its log) ────
SLOG=exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/train_ddp.log
LASTEP=$(grep -oE "epoch: [0-9]+" "$SLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
RES=$(( ${LASTEP:-706} + 1 )); REM=$(( 1200 - RES ))
echo "[$(ts)] resuming sig=1.0 (last ep $LASTEP -> resume $RES, +$REM -> 1200)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_noisedensity.log" 2>&1
  echo "[$(ts)] sig=1.0 launcher exit=$?"
else
  echo "[$(ts)] sig=1.0 already at/over 1200 — not resuming"
fi
echo "[$(ts)] CHAIN COMPLETE"
