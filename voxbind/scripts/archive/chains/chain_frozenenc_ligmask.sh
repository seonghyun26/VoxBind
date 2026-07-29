#!/usr/bin/env bash
# LEAK-REMOVAL frozen-enc run. After the sig1.0 frozen-enc pipeline finishes
# (FROZENENC-SIG1.0 CHAIN DONE), train the ligand-masked frozen-enc VoxBind at sigma=0.9
# for 350 epochs from scratch (density+gradmag blanked in the clean ligand footprint —
# mask region from the ligand coords), then run its density-conditioned 79-pocket eval.
#   setsid nohup bash scripts/archive/chains/chain_frozenenc_ligmask.sh > log/chain_frozenenc_ligmask.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
POSTLOG=$LOG/chain_frozenenc_sig1.0.log
CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1_ligmask
EXP=voxbind_frozenenc_atomblob7_v2p1_ligmask_sig0.9; EXPDIR=$VOX/exps/$EXP
LOCK=$LOG/chain_frozenenc_ligmask.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another frozenenc-ligmask chain running — exit"; exit 1; }
drain(){ for _ in $(seq 1 48);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }

# 1) wait for the sig1.0 frozen-enc pipeline (#22 train + #23 eval) to finish
DEADLINE=$(( $(date +%s)+8*24*3600 ))
log "waiting for sig1.0 frozen-enc pipeline to finish (FROZENENC-SIG1.0 CHAIN DONE) ..."
until grep -q "FROZENENC-SIG1.0 CHAIN DONE" "$POSTLOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for sig1.0 pipeline"; exit 1; }
  sleep 600
done
log "sig1.0 pipeline done. Draining GPUs (its final eval sampling is done; docking is CPU-bg)."
drain

# 2) train ligand-masked frozen-enc, sigma=0.9, 350 ep, from scratch
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "===== TRAIN ligmask frozen-enc (sig=0.9, 350 ep, from scratch) exp=$EXP → $RUNLOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 "$VOX/train_ddp.py" \
  --config-name="$CFG" smooth_sigma=0.9 wjs.n_targets=0 num_epochs=350 \
  exp_name="$EXP" output_dir="$EXPDIR" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
log "ligmask frozen-enc train exit=$? (target ep350)"
drain

# 3) density-conditioned eval (sample.py loads the exp's cfg.yaml → density_mask_ligand=true,
#    so sample() carves the same hole from the reference ligand — no eval-script change needed)
EP=$("$B/python" -c "import torch;print(torch.load('$EXPDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
log "===== EVAL ligmask frozen-enc (@ ep$EP) ====="
bash "$VOX/scripts/archive/workflows/frozenenc_full_eval.sh" "$EXPDIR" "$EXPDIR/samples/full_eval_ep${EP}" 100
log "FROZENENC-LIGMASK CHAIN DONE (@ ep$EP). Add its row to voxbind_results.html (rerun the generator with the new json)."
