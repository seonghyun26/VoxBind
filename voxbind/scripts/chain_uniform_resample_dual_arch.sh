#!/usr/bin/env bash
# UNIFORM-weight + ON-THE-FLY-RESAMPLE, two architectures back-to-back:
#   coord (atom-blob lig/poc) + v5 X-ray density + gradmag, 13 ch, ALL channels
#   weighted equally, density resampled from the FULL CCP4 map at the augmented pose.
#   arch 1 = ChannelViT (per-channel-group patch embed, train_density_vit_mae.py)
#   arch 2 = ChA-MAEViT (token-drop DCP MAE,            train_density_cha_mae.py)
# Flow: stop the sig=1.0 denoiser → pretrain+probe arch1 → pretrain+probe arch2 →
#       RESUME the sig=1.0 denoiser to 1200 ("resume voxbind training" afterward).
# Mirrors scripts/chain_resample_then_sigma.sh.
#
# Launch (detached) — own session/pgid so the denoiser-group kill can't hit it:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/chain_uniform_resample_dual_arch.sh \
#     > log/chain_uniform_resample_dual_arch.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
BASE_COND=atomblob_density_gradmag                 # frozen-crop baseline (260606, probe 0.477)

# arch 1 — ChannelViT
EXP1=atomblob_density_gradmag_channelvit_uniform_resample_v5
CFG1=config_train_atomblob_density_gradmag_channelvit_uniform_resample_v5
COND1=atomblob_density_gradmag_channelvit_uniform_resample
# arch 2 — ChA-MAEViT
EXP2=atomblob_density_gradmag_cha_mae_uniform_resample_v5
CFG2=config_train_atomblob_density_gradmag_cha_mae_uniform_resample_v5
COND2=atomblob_density_gradmag_cha_mae_uniform_resample

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1   # no C compiler on this box → neutralize torch.compile (belt+braces w/ compile.enabled=false)
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"; cd "$ROOT" || exit 1

# ── 1) stop the sig=1.0 denoiser (kill its process group) ──────────────────────
echo "[$(ts)] stopping sig=1.0 denoiser ..."
SIG_PGID=$(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | head -1 | tr -d ' ')
[ -n "${SIG_PGID:-}" ] && { echo "[$(ts)] kill -TERM pgroup $SIG_PGID"; kill -TERM -- -"$SIG_PGID" 2>/dev/null; }
sleep 15
[ -n "${SIG_PGID:-}" ] && kill -KILL -- -"$SIG_PGID" 2>/dev/null
# wait until the GPUs actually drain (NCCL teardown) before launching, up to ~3 min
for _ in $(seq 1 18); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 4000 ] && break
  sleep 10
done
echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"

# ── helper: pretrain (4 GPU) then probe one arm ────────────────────────────────
run_arm() {
  local TRAINER="$1" CFG="$2" EXP="$3" COND="$4"
  local OUT="$ROOT/exps/$EXP"
  mkdir -p "$OUT"
  echo "[$(ts)] START pretrain $EXP via $TRAINER (100 ep, bsz=32 accum=1 x4 = eff 128)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
    "$ROOT/$TRAINER" --config-name="$CFG" \
    exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
    > "$LOG/${EXP}.log" 2>&1
  echo "[$(ts)] END pretrain $EXP exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"

  if [ -f "$OUT/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] extracting features $COND (GPU)..."
    CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
      --condition "$COND" --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
      > "$LOG/probe_features_${EXP}.log" 2>&1
    echo "[$(ts)] features $COND exit=$?"
  else
    echo "[$(ts)] ERROR: $EXP e99 checkpoint missing — skipping feature extraction"
  fi
}

# ── 2) arch 1: ChannelViT ──────────────────────────────────────────────────────
run_arm train_density_vit_mae.py "$CFG1" "$EXP1" "$COND1"
# ── 3) arch 2: ChA-MAEViT ──────────────────────────────────────────────────────
run_arm train_density_cha_mae.py "$CFG2" "$EXP2" "$COND2"

# ── 4) joint probe: baseline vs both uniform-resample arms (CPU; non-fatal) ─────
echo "[$(ts)] probing $BASE_COND vs $COND1 vs $COND2 ..."
CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
  --conditions "$BASE_COND" "$COND1" "$COND2" \
  --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag uniform_resample --allow_stale_features \
  > "$LOG/probe_uniform_resample_dual_arch.log" 2>&1
echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_uniform_resample.csv"

# ── 5) resume sig=1.0 to 1200 (resume epoch = checkpoint epoch + 1) ────────────
LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
RES=$(( ${LASTEP:-1057} + 1 )); REM=$(( 1200 - RES ))
echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> 1200)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_uniform_resample.log" 2>&1
  echo "[$(ts)] sig=1.0 launcher exit=$?"
else
  echo "[$(ts)] sig=1.0 already at/past 1200 (last $LASTEP) — not resuming"
fi
echo "[$(ts)] CHAIN COMPLETE"
