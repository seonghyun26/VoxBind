#!/usr/bin/env bash
# Cap the frozen-enc restart at absolute epoch 350 (original plan; the resumed run
# would otherwise go to ~ep449). Poll the rolling checkpoint's epoch; when it reaches
# TARGET, kill the train_ddp.py ranks by PID. torchrun then exits and the parent
# chain_frozenenc_restart.sh runs resume_denoiser → the sig=1.0 denoiser resumes.
#   setsid nohup bash scripts/chain_cap350.sh > log/chain_cap350.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CKPT=$VOX/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/checkpoint.pth.tar
TARGET=350
LOCK=$LOG/chain_cap350.lock
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another cap watcher running — exit"; exit 1; }
log "cap watcher armed: stop frozen-enc when checkpoint epoch >= $TARGET"
DEADLINE=$(( $(date +%s) + 5*24*3600 ))
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "timeout (5d) — exit without capping"; exit 1; }
  if ! pgrep -f 'python3.*[t]rain_ddp.py' >/dev/null; then
    log "train_ddp.py not running — nothing to cap; exit."; exit 0
  fi
  ep=$("$B/python" -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
  if [ -n "$ep" ] && [ "$ep" -ge "$TARGET" ] 2>/dev/null; then
    log "checkpoint reached epoch $ep >= $TARGET — stopping training."
    mapfile -t PIDS < <(pgrep -f 'python3.*train_ddp.py')
    log "killing ranks: ${PIDS[*]}"
    for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
    sleep 15
    for p in $(pgrep -f 'python3.*train_ddp.py'); do kill -9 "$p" 2>/dev/null; done
    log "training stopped at epoch $ep. chain_frozenenc_restart.sh will resume the denoiser."
    exit 0
  fi
  log "epoch=$ep (<$TARGET) — waiting ..."
  sleep 600
done
