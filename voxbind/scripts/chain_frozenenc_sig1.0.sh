#!/usr/bin/env bash
# After the vanilla pipeline (chain_post_v2) fully finishes, train a FROZEN-ENC VoxBind
# at sigma=1.0 for 350 epochs from scratch (same frozen atomblob7 v2.1 density encoder),
# then run its density-conditioned 79-pocket eval.
#   setsid nohup bash scripts/chain_frozenenc_sig1.0.sh > log/chain_frozenenc_sig1.0.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
POSTLOG=$LOG/chain_post_v2.log
CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1
EXP=voxbind_frozenenc_atomblob7_v2p1_sig1.0; EXPDIR=$VOX/exps/$EXP
LOCK=$LOG/chain_frozenenc_sig1.0.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another frozenenc-sig1.0 chain running — exit"; exit 1; }
drain(){ for _ in $(seq 1 48);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }

# 1) wait for the vanilla pipeline to finish
DEADLINE=$(( $(date +%s)+8*24*3600 ))
log "waiting for chain_post_v2 to finish (POST-V2 PIPELINE DONE) ..."
until grep -q "POST-V2 PIPELINE DONE" "$POSTLOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for vanilla pipeline"; exit 1; }
  sleep 600
done
log "vanilla pipeline done. Draining GPUs (its final eval sampling is done; docking is CPU-bg)."
drain

# 2) train sigma=1.0 frozen-enc, 350 ep, from scratch (wjs.n_targets=0 → no in-training sampling crash)
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "===== TRAIN sig=1.0 frozen-enc (350 ep, from scratch) exp=$EXP → $RUNLOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 "$VOX/train_ddp.py" \
  --config-name="$CFG" smooth_sigma=1.0 wjs.n_targets=0 num_epochs=350 \
  exp_name="$EXP" output_dir="$EXPDIR" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
log "sig1.0 frozen-enc train exit=$? (target ep350)"
drain

# 3) density-conditioned eval
EP=$("$B/python" -c "import torch;print(torch.load('$EXPDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
log "===== EVAL sig1.0 frozen-enc (@ ep$EP) ====="
bash "$VOX/scripts/frozenenc_full_eval.sh" "$EXPDIR" "$EXPDIR/samples/full_eval_ep${EP}" 100
log "FROZENENC-SIG1.0 CHAIN DONE (@ ep$EP). Add its row to voxbind_results.html (rerun the generator with the new json)."
