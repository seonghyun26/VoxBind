#!/usr/bin/env bash
# WATCHER (looping): keep the sig=1.0 baseline going forever. Each time the
# current train_ddp.py run finishes, read the checkpoint epoch and relaunch a
# fresh +CHUNK epochs from it (1600 -> 2000 -> 2400 -> ...), then re-arm and
# wait for THAT run to finish, and so on, until killed.
#
# This supersedes chain_sig1.0_extend_to_1600.sh: the current run already
# targets ep1600 (resume_epoch=1164 + num_epochs=436), so the old "extend to
# 1600" watcher is a no-op (REM=0). This one extends BEYOND 1600 indefinitely.
#
# Robust:
#   * waits on the actual train_ddp.py process (bracket pattern -> no pgrep
#     self-match; this script's own argv never contains the literal train_ddp.py)
#   * reads the real checkpoint epoch; resume = ckpt+1, so a clean run that
#     reached target T (ckpt=T-1) extends T -> T+CHUNK on a round milestone
#   * if the ckpt epoch can't be read (env wiped by a container restart, see
#     memory voxbind-env-rebuild) it ABORTS rather than relaunch blindly
#   * crash-loop guard: 3 consecutive launches that exit in <600s AND make no
#     ckpt progress -> abort, so a broken env can't spin-relaunch forever
#   * single-instance flock so a second copy can't double-launch on the GPUs
#
# Launch (detached, own session so it survives this shell):
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_sig1.0_keepgoing.sh \
#     > log/chain_sig1.0_keepgoing.log 2>&1 &
#
# Stop it (and let the in-flight training keep running):
#   pkill -f '[c]hain_sig1.0_keepgoing.sh'
# Stop it AND the current training: also kill the train_ddp.py procs by PID.
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
CK=$SDIR/checkpoint.pth.tar
CHUNK=400                 # epochs to add each time a run finishes
LOCK=$LOG/chain_sig1.0_keepgoing.lock
# no-C-compiler box: must run fp32 eager (memory voxbind-no-c-compiler)
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1

mkdir -p "$LOG"
cd "$ROOT" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }
ckpt_epoch(){ "$PY/python" -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null; }

# ── single-instance lock ──────────────────────────────────────────────────────
exec 9>"$LOCK" || { log "cannot open lock $LOCK"; exit 1; }
if ! flock -n 9; then log "another chain_sig1.0_keepgoing.sh is already running — exiting"; exit 1; fi

log "watcher armed (PID $$): will keep sig=1.0 going forever, +$CHUNK epochs per completion"
ITER=0
FAILS=0
while true; do
  ITER=$((ITER+1))

  # ── wait for the in-flight run to finish ────────────────────────────────────
  log "[#$ITER] waiting for train_ddp.py to finish ..."
  while pgrep -f '[t]rain_ddp.py' >/dev/null 2>&1; do sleep 60; done
  log "[#$ITER] no train_ddp.py running; grace 30s for ckpt flush + NCCL teardown"
  sleep 30

  L=$(ckpt_epoch)
  if [ -z "${L:-}" ]; then
    log "[#$ITER] ERROR: cannot read ckpt epoch (env wiped / container restart?) — aborting watcher"
    exit 1
  fi
  log "[#$ITER] checkpoint epoch = $L"

  RES=$(( L + 1 ))
  NUM=$CHUNK
  NEXT=$(( RES + NUM ))     # target #epochs; ckpt will reach NEXT-1 on success

  # ── wait for GPU memory to actually drain before relaunching ────────────────
  USED=""
  for _ in $(seq 1 18); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [ "${USED:-99999}" -lt 4000 ] && break
    sleep 10
  done
  log "[#$ITER] GPU mem used before launch (MiB): ${USED:-n/a}"

  RUNLOG="$LOG/sig1.0_keepgoing_to${NEXT}_$(date +%Y%m%d_%H%M%S).log"
  log "[#$ITER] EXTEND sig=1.0: resume ep$RES, +$NUM -> target ep$NEXT  (log: $RUNLOG)"

  # 35_train_baseline_4gpu.sh runs torchrun in the FOREGROUND -> this blocks
  # until that training run exits, then we loop and wait again.
  START=$SECONDS
  SIG=1.0 NUM_EPOCHS=$NUM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" > "$RUNLOG" 2>&1
  RC=$?
  DUR=$(( SECONDS - START ))
  sleep 10                 # let the final ckpt flush before we read it
  L2=$(ckpt_epoch)
  log "[#$ITER] launcher exit=$RC dur=${DUR}s  ckpt now ${L2:-?}"

  # ── crash-loop guard ────────────────────────────────────────────────────────
  if [ "${L2:-0}" -le "$L" ] && [ "$DUR" -lt 600 ]; then
    FAILS=$(( FAILS + 1 ))
    log "[#$ITER] WARN: no ckpt progress and exited in ${DUR}s (consecutive fail $FAILS/3)"
    if [ "$FAILS" -ge 3 ]; then
      log "[#$ITER] ABORT: 3 fast failures with no progress — stopping watcher to avoid spin-relaunch"
      exit 1
    fi
    log "[#$ITER] backoff 120s before retry"
    sleep 120
  else
    FAILS=0
  fi
done
