#!/usr/bin/env bash
# Cap the fresh vanilla sig=0.9 v2 at epoch 350 (user wants 350 to epoch-match the
# frozen-enc, not the config's 400), then run its full 79-pocket eval on the freed GPUs.
#   setsid nohup bash scripts/archive/chains/chain_cap350_v2.sh > log/chain_cap350_v2.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
EXP=exp_sig0.9_v2; CKPT=$VOX/exps/$EXP/checkpoint.pth.tar; TARGET=350
LOCK=$LOG/chain_cap350_v2.lock
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another cap-v2 running — exit"; exit 1; }
log "cap-v2 armed: stop $EXP at epoch >= $TARGET, then run its full eval"
DEADLINE=$(( $(date +%s) + 6*24*3600 ))
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "timeout — exit"; exit 1; }
  pgrep -f "python3.*[t]rain_ddp.py" >/dev/null || { log "train_ddp gone (v2 stopped/crashed) — proceeding to eval"; break; }
  ep=$("$B/python" -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
  { [ -n "$ep" ] && [ "$ep" -ge "$TARGET" ]; } 2>/dev/null && { log "v2 reached epoch $ep >= $TARGET"; break; }
  log "epoch=$ep (<$TARGET) — waiting ..."; sleep 600
done
# stop v2 (kill the train_ddp ranks by PID; file-based script so pattern won't self-match)
for p in $(pgrep -f 'python3.*train_ddp.py'); do kill -TERM "$p" 2>/dev/null; done
sleep 15
for p in $(pgrep -f 'python3.*train_ddp.py'); do kill -9 "$p" 2>/dev/null; done
for _ in $(seq 1 48); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-9}" -lt 3000 ] && break; sleep 5; done
ep=$("$B/python" -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
log "v2 stopped (epoch $ep), GPU drained (${U:-?} MiB). Launching v2 full eval on dedicated GPUs."
bash "$VOX/scripts/archive/workflows/vanilla_full_eval.sh" "$VOX/exps/$EXP" "$VOX/exps/$EXP/samples/full_eval_ep${ep}" 100
log "CAP-V2 CHAIN DONE (v2 eval finished/launched at epoch $ep)."
