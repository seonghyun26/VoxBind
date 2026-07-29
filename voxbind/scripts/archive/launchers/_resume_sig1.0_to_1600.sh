#!/usr/bin/env bash
# Standalone resume of the sig=1.0 denoiser to ep1600, mirroring resume_denoiser() in
# launch_plinder_otf_when_ready.sh. Safety net for the PLINDER C / C+D+G preemption
# (2026-06-20): if the interactive session dies with the denoiser stopped, run this to
# bring it back. Reads the CURRENT checkpoint epoch (so it's correct whenever it runs).
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
TARGET=1600
LOG=$ROOT/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1

# guard: don't double-launch if a sig=1.0 denoiser is already running
if ps -eo cmd | grep -q "[t]rain_ddp.py smooth_sigma=1.0"; then
  echo "[$(ts)] a sig=1.0 denoiser is ALREADY running — not resuming"; exit 0
fi

LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
if [ -z "${LASTEP:-}" ]; then echo "[$(ts)] ERROR: cannot read denoiser ckpt epoch — resume manually"; exit 1; fi
RES=$(( LASTEP + 1 )); REM=$(( TARGET - RES ))
echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> $TARGET)"
if [ "$REM" -gt 0 ]; then
  SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
    > "$LOG/sig1.0_resume_after_plinder.log" 2>&1
  echo "[$(ts)] sig=1.0 launcher exit=$?"
else
  echo "[$(ts)] sig=1.0 already at/past $TARGET (last $LASTEP) — not resuming"
fi
