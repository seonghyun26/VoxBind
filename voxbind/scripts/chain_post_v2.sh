#!/usr/bin/env bash
# Post-v2 master pipeline (all GPU stages are 4-GPU sequential; dockings run on CPU in
# the background so they overlap the next GPU stage):
#   1) wait exp_sig0.9_v2 → epoch 350, stop it
#   2) EVAL ep923   (fully-trained vanilla sig0.9)  — sample dedicated + dock(bg)
#   3) EVAL v2      (vanilla sig0.9 @ ep350)         — sample dedicated + dock(bg)
#   4) TRAIN fresh vanilla sig=1.0, 350 epochs from scratch  (exp_sig1.0_350ep)
#   5) EVAL sig1.0  (@ ep350)                        — sample + dock(bg)
#   setsid nohup bash scripts/chain_post_v2.sh > log/chain_post_v2.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
V2=exp_sig0.9_v2; V2CK=$VOX/exps/$V2/checkpoint.pth.tar; TARGET=350
EP923=$VOX/exps/_vanilla_ep923
SIG10=exp_sig1.0_350ep; SIG10_DIR=$VOX/exps/$SIG10
LOCK=$LOG/chain_post_v2.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another post-v2 chain running — exit"; exit 1; }
epk(){ "$B/python" -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null; }
drain(){ for _ in $(seq 1 48);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }
kill_train(){ for p in $(pgrep -f 'python3.*train_ddp.py');do kill -TERM "$p" 2>/dev/null;done; sleep 15; for p in $(pgrep -f 'python3.*train_ddp.py');do kill -9 "$p" 2>/dev/null;done; }

# ── 1) wait for the v2 cap ───────────────────────────────────────────────────
DEADLINE=$(( $(date +%s)+7*24*3600 ))
log "waiting for v2 ($V2) to reach epoch $TARGET ..."
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for v2"; exit 1; }
  pgrep -f 'python3.*[t]rain_ddp.py' >/dev/null || { log "v2 train_ddp gone — proceeding"; break; }
  ep=$(epk "$V2CK"); { [ -n "$ep" ] && [ "$ep" -ge "$TARGET" ]; } 2>/dev/null && { log "v2 reached epoch $ep >= $TARGET"; break; }
  log "v2 epoch=$ep (<$TARGET) — waiting ..."; sleep 600
done
kill_train; drain
V2EP=$(epk "$V2CK"); log "v2 stopped at epoch $V2EP."

# ── 2) EVAL ep923 (dedicated) ────────────────────────────────────────────────
log "===== STAGE 2: EVAL ep923 (vanilla sig0.9, fully trained) ====="
bash "$VOX/scripts/vanilla_full_eval.sh" "$EP923" "$EP923/samples/full_eval_ep923" 100

# ── 3) EVAL v2 (dedicated) ───────────────────────────────────────────────────
log "===== STAGE 3: EVAL v2 (vanilla sig0.9 @ ep$V2EP) ====="
bash "$VOX/scripts/vanilla_full_eval.sh" "$VOX/exps/$V2" "$VOX/exps/$V2/samples/full_eval_ep${V2EP}" 100

# ── 4) TRAIN fresh vanilla sig=1.0, 350 epochs ───────────────────────────────
drain
S10LOG="$LOG/${SIG10}_$(date +%Y%m%d_%H%M%S).log"
log "===== STAGE 4: TRAIN fresh sig=1.0 (350 ep, from scratch) exp=$SIG10 → $S10LOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCHDYNAMO_DISABLE=1 LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib \
  "$B/torchrun" --standalone --nproc_per_node=4 "$VOX/train_ddp.py" \
  smooth_sigma=1.0 bsz=32 accum_steps=1 num_workers=12 +prefetch_factor=16 \
  num_epochs=350 wjs.n_targets=0 exp_name="$SIG10" output_dir="$SIG10_DIR" hydra.run.dir="$SIG10_DIR" \
  > "$S10LOG" 2>&1
RC=$?; log "sig1.0 training exit=$RC"

# ── 5) EVAL sig1.0 ───────────────────────────────────────────────────────────
drain
S10EP=$(epk "$SIG10_DIR/checkpoint.pth.tar")
log "===== STAGE 5: EVAL sig1.0 (@ ep$S10EP) ====="
bash "$VOX/scripts/vanilla_full_eval.sh" "$SIG10_DIR" "$SIG10_DIR/samples/full_eval_ep${S10EP}" 100
log "POST-V2 PIPELINE DONE (ep923 + v2 + sig1.0 evals; sig1.0 trained). Add rows to voxbind_results.html."
