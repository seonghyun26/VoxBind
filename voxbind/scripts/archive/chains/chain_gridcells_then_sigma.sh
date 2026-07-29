#!/usr/bin/env bash
# density×gradmag grid completion: 4 pretrains (cells 6-9) + probe, then resume sig=1.0.
#   6 atomblob_realgrad            X density, real gradmag  (gradmag-as-density, 12ch)
#   7 atomblob_noisegrad           X density, noise gradmag (gradmag-as-density, 12ch)
#   8 atomblob_density_noisegrad   real density, noise gradmag (13ch)
#   9 atomblob_noisedensity_realgrad noise density, real gradmag (13ch)
# Probe uses dataset/probe_density_grid.py (standalone — robust to 01c_pdbbind_probe.py resets).
#
# Launch:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_gridcells_then_sigma.sh > log/chain_gridcells_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
cd "$ROOT" || exit 1
export VOXBIND_ALLOW_LEGACY_VOXEL_META=1

# ── 0) stop sig=1.0 (free the GPUs) ──────────────────────────────────────────
SIG_PGID=$(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | head -1 | tr -d ' ')
if [ -n "${SIG_PGID:-}" ]; then echo "[$(ts)] stopping sig=1.0 pgroup $SIG_PGID"; kill -TERM -- -"$SIG_PGID" 2>/dev/null; sleep 12; kill -KILL -- -"$SIG_PGID" 2>/dev/null; fi
for _ in $(seq 1 18); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${U:-99999}" -lt 2000 ] && break; sleep 10
done
echo "[$(ts)] GPU mem used: ${U:-n/a} MiB"

train() {  # $1=config-name $2=exp_name
  local CFG=$1 EXP=$2 OUT=$ROOT/exps/$2
  mkdir -p "$OUT"
  echo "[$(ts)] START $EXP"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
    "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
    exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
    > "$LOG/${EXP}.log" 2>&1
  echo "[$(ts)] END $EXP exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"
}

# ── 1) four pretrains, sequentially on all 4 GPUs ────────────────────────────
train config_train_atomblob_realgrad_vit_mae_40m_v5            atomblob_realgrad_vit_mae_40m_v5
train config_train_atomblob_noisegrad_vit_mae_40m_v5           atomblob_noisegrad_vit_mae_40m_v5
train config_train_atomblob_density_noisegrad_vit_mae_40m_v5   atomblob_density_noisegrad_vit_mae_40m_v5
train config_train_atomblob_noisedensity_realgrad_vit_mae_40m_v5 atomblob_noisedensity_realgrad_vit_mae_40m_v5

# ── 2) probe all 4 (standalone; density/gradmag sources per cell) ─────────────
echo "[$(ts)] probing grid cells (standalone probe_density_grid.py)..."
CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/probe_density_grid.py \
  > "$LOG/probe_density_grid.log" 2>&1
echo "[$(ts)] grid probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_gridcells.csv"

# ── 3) resume sig=1.0 to 1200 ────────────────────────────────────────────────
SLOG=exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/train_ddp.log
LASTEP=$(grep -oE "epoch: [0-9]+" "$SLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
RES=$(( ${LASTEP:-907} + 1 )); REM=$(( 1200 - RES ))
echo "[$(ts)] resuming sig=1.0 (last ep $LASTEP -> resume $RES, +$REM -> 1200)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_gridcells.log" 2>&1
fi
echo "[$(ts)] CHAIN COMPLETE"
