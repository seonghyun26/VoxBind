#!/usr/bin/env bash
# WATCHER: when the current sig=1.0 run (resuming -> ep1200) finishes, auto-launch
# another 400 epochs -> ep1600. Robust: reads the actual checkpoint epoch and only
# extends if the run truly reached the ~1200 target (so a crash won't trigger a
# wrong-range relaunch). Convention (35_train_baseline_4gpu.sh): after ep1199 the
# ckpt['epoch']=1199, so RESUME_EPOCH=ckpt+1=1200, NUM_EPOCHS=1600-1200=400 -> ep1200..1599.
#
# Launch (detached, own session so it survives this shell):
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_sig1.0_extend_to_1600.sh \
#     > log/chain_sig1.0_extend_to_1600.log 2>&1 &
#
# NOTE: a container restart wipes the voxbind conda env AND kills this watcher
# (see memory voxbind-env-rebuild). This watcher only covers normal completion.
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
CK=$SDIR/checkpoint.pth.tar
TARGET=1600
FLOOR=1198   # require the run to have reached ~1200 before extending
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
ckpt_epoch(){ "$PY/python" -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null; }
cd "$ROOT" || exit 1

echo "[$(ts)] watcher armed: waiting for current sig=1.0 run (train_ddp.py) to finish ..."
# bracket pattern so this script's own cmdline never self-matches
while pgrep -f '[t]rain_ddp.py' >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] current run ended; letting final ckpt flush + NCCL teardown ..."
sleep 30

LASTEP=$(ckpt_epoch)
echo "[$(ts)] checkpoint epoch = ${LASTEP:-unknown}"
if [ -z "${LASTEP:-}" ]; then echo "[$(ts)] ERROR: cannot read ckpt epoch (env wiped?) — aborting"; exit 1; fi
if [ "$LASTEP" -lt "$FLOOR" ]; then
  echo "[$(ts)] ABORT: run ended early at ep$LASTEP (<$FLOOR, did not reach 1200) — NOT auto-extending"
  exit 1
fi

RES=$(( LASTEP + 1 )); REM=$(( TARGET - RES ))
if [ "$REM" -le 0 ]; then echo "[$(ts)] already at/past $TARGET (ep$LASTEP) — nothing to do"; exit 0; fi

# wait for GPUs to actually drain before relaunching
for _ in $(seq 1 18); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 4000 ] && break
  sleep 10
done
echo "[$(ts)] GPU mem used (MiB): ${USED:-n/a}"

echo "[$(ts)] EXTEND sig=1.0: resume ep$RES, +$REM -> $TARGET"
SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
  > "$LOG/sig1.0_extend_to_1600.log" 2>&1
echo "[$(ts)] extension launcher exit=$? (ckpt now $(ckpt_epoch))"
echo "[$(ts)] WATCHER COMPLETE"
