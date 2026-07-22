#!/usr/bin/env bash
# Combined-data pretrain (CrossDocked + PDBbind-train experimental density), inserted
# AFTER the noise-density exp2 probe and BEFORE sig=1.0 makes real progress.
#
# Hands off from chain_noisedensity_then_sigma.sh: waits for that chain's exp2 probe
# CSV, then KILLS that chain's process group (stopping it + any sig=1.0 it launched),
# runs the combined pretrain + probe, and resumes sig=1.0 to 1200.
#
# Launch (detached) any time after the combined data/config are ready:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/chain_combined_then_sigma.sh \
#     > log/chain_combined_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
EXP=atomblob_density_gradmag_combined_cdpdb_v5
COND=atomblob_density_gradmag_combined_cdpdb
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
cd "$ROOT" || exit 1
export VOXBIND_ALLOW_LEGACY_VOXEL_META=1

# ── 1) wait for the noise-density chain to finish exp2 + its probe ────────────
echo "[$(ts)] waiting for exp2 probe (probe_results_e99_v5_noisedensity.csv)..."
while [ ! -f dataset/data/pdbbind/probe_results_e99_v5_noisedensity.csv ]; do sleep 60; done
echo "[$(ts)] exp2 probe done; taking over the GPUs"

# ── 2) stop the old chain + any sig=1.0 it started (kill its process group) ───
sleep 30   # let the old chain reach its sig=1.0 launch so we kill it too
OLD_PGID=$(ps -eo pgid,cmd | grep "[c]hain_noisedensity_then_sigma" | awk '{print $1}' | head -1 | tr -d ' ')
if [ -n "${OLD_PGID:-}" ]; then echo "[$(ts)] killing old chain pgroup $OLD_PGID"; kill -TERM -- -"$OLD_PGID" 2>/dev/null; fi
# also catch a sig=1.0 in any other group
SIG_PGID=$(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | head -1 | tr -d ' ')
if [ -n "${SIG_PGID:-}" ] && [ "${SIG_PGID:-}" != "${OLD_PGID:-}" ]; then kill -TERM -- -"$SIG_PGID" 2>/dev/null; fi
sleep 12
# verify GPUs free; SIGKILL stragglers
for g in "${OLD_PGID:-}" "${SIG_PGID:-}"; do [ -n "$g" ] && kill -KILL -- -"$g" 2>/dev/null; done
sleep 5
# wait until the GPUs actually drain (NCCL teardown) before launching, up to ~3 min
for _ in $(seq 1 18); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 2000 ] && break
  sleep 10
done
echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"

# ── 3) combined pretrain (4 GPU) ─────────────────────────────────────────────
OUT=$ROOT/exps/$EXP
mkdir -p "$OUT"
echo "[$(ts)] START combined pretrain $EXP"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
  "$ROOT/train_density_vit_mae.py" --config-name=config_train_atomblob_density_gradmag_combined_cdpdb_v5 \
  exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
  > "$LOG/${EXP}.log" 2>&1
echo "[$(ts)] END combined pretrain exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"

# ── 4) probe combined vs CrossDocked-only baseline (0.477) ────────────────────
if [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
  echo "[$(ts)] extracting combined features (GPU)..."
  CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
    --condition "$COND" --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
    > "$LOG/probe_features_${EXP}.log" 2>&1
  echo "[$(ts)] features exit=$?"
  echo "[$(ts)] probing combined vs atomblob_density_gradmag..."
  CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
    --conditions atomblob_density_gradmag "$COND" \
    --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag combined --allow_stale_features \
    > "$LOG/probe_${EXP}.log" 2>&1
  echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_combined.csv"
else
  echo "[$(ts)] ERROR: combined e99 checkpoint missing — skipping probe"
fi

# ── 5) resume sig=1.0 to 1200 (dynamic resume epoch) ─────────────────────────
SLOG=exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/train_ddp.log
LASTEP=$(grep -oE "epoch: [0-9]+" "$SLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
RES=$(( ${LASTEP:-906} + 1 )); REM=$(( 1200 - RES ))
echo "[$(ts)] resuming sig=1.0 (last ep $LASTEP -> resume $RES, +$REM -> 1200)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_combined.log" 2>&1
  echo "[$(ts)] sig=1.0 launcher exit=$?"
fi
echo "[$(ts)] CHAIN COMPLETE"
