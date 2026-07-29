#!/usr/bin/env bash
# ON-THE-FLY RESAMPLE ablation: stop the sig=1.0 denoiser, pretrain the resample-aug
# 3D ViT (twin of 260606 invfreq v5, density cropped from the FULL map at the augmented
# pose), probe it vs the frozen-crop baseline (0.477), then RESUME the denoiser to 1200.
# Mirrors scripts/archive/chains/chain_combined_then_sigma.sh.
#
# Launch (detached) — gives this chain its own session/pgid so the denoiser-group kill
# below cannot hit it:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_resample_then_sigma.sh \
#     > log/chain_resample_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
EXP=atomblob_density_gradmag_vit_mae_40m_v5_resample
CFG=config_train_atomblob_density_gradmag_vit_mae_40m_v5_resample
COND=atomblob_density_gradmag_resample          # 01c registry → exps/$EXP
BASE_COND=atomblob_density_gradmag              # frozen-crop baseline (260606, probe 0.477)
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1   # this box has NO C compiler → neutralize ALL torch.compile (belt+braces w/ compile.enabled=false)
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"; cd "$ROOT" || exit 1

# ── 1) stop the sig=1.0 denoiser + its chain watcher (kill the process groups) ─
echo "[$(ts)] stopping sig=1.0 denoiser + chain watcher ..."
SIG_PGID=$(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | head -1 | tr -d ' ')
# EXCLUDE this script's own pgid — otherwise the chain-watcher grep matches itself and self-kills.
CHN_PGID=$(ps -eo pgid,cmd | grep "[c]hain_.*_then_sigma" | grep -v "chain_resample_then_sigma" | awk '{print $1}' | head -1 | tr -d ' ')
for g in "${SIG_PGID:-}" "${CHN_PGID:-}"; do
  [ -n "$g" ] && { echo "[$(ts)] kill -TERM pgroup $g"; kill -TERM -- -"$g" 2>/dev/null; }
done
sleep 15
for g in "${SIG_PGID:-}" "${CHN_PGID:-}"; do [ -n "$g" ] && kill -KILL -- -"$g" 2>/dev/null; done
# wait until the GPUs actually drain (NCCL teardown) before launching, up to ~3 min
for _ in $(seq 1 18); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 4000 ] && break
  sleep 10
done
echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"

# ── 2) pretrain the resample arm (4 GPU, effective batch 128 = baseline) ───────
OUT=$ROOT/exps/$EXP
mkdir -p "$OUT"
echo "[$(ts)] START resample pretrain $EXP (100 ep, bsz=32 accum=1 x4 = eff 128)"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
  "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
  exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
  > "$LOG/${EXP}.log" 2>&1
echo "[$(ts)] END resample pretrain exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"

# ── 3) probe resample vs frozen-crop baseline (non-fatal; denoiser resumes regardless) ─
if [ -f "$OUT/checkpoint_e0099.pth.tar" ]; then
  echo "[$(ts)] extracting resample features (GPU)..."
  CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
    --condition "$COND" --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
    > "$LOG/probe_features_${EXP}.log" 2>&1
  echo "[$(ts)] features exit=$?"
  echo "[$(ts)] probing $COND vs $BASE_COND ..."
  CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
    --conditions "$BASE_COND" "$COND" \
    --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag resample --allow_stale_features \
    > "$LOG/probe_${EXP}.log" 2>&1
  echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_resample.csv"
else
  echo "[$(ts)] ERROR: resample e99 checkpoint missing — skipping probe"
fi

# ── 4) resume sig=1.0 to 1200 (resume epoch = checkpoint epoch + 1) ────────────
LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
RES=$(( ${LASTEP:-1057} + 1 )); REM=$(( 1200 - RES ))
echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> 1200)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_resample.log" 2>&1
  echo "[$(ts)] sig=1.0 launcher exit=$?"
else
  echo "[$(ts)] sig=1.0 already at/past 1200 (last $LASTEP) — not resuming"
fi
echo "[$(ts)] CHAIN COMPLETE"
