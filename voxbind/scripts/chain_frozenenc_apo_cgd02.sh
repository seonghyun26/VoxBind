#!/usr/bin/env bash
# After the cgd=0.2 encoder pretrain finishes (PRETRAIN-CGD02 DONE), train a σ=0.9
# frozen-enc VoxBind in the APO / leak-removed setting (default ControlNet-like fusion,
# density_mask_ligand=true) but with the FROZEN encoder = the new cgd0.2 checkpoint
# (channel-group dropout → apo input is in-distribution). Goal: better generation with
# NO ligand density. Clean read vs #29's -7.92 (which was OOD-confounded on 260701).
# Then run the density-conditioned 79-pocket eval.
#   setsid nohup bash scripts/chain_frozenenc_apo_cgd02.sh > log/chain_frozenenc_apo_cgd02.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
POSTLOG=$LOG/chain_pretrain_cgd02.log
CGD02DIR=$VOX/exps/260718_plinder_v2p1_box_atomblob7_cdg_channelvit_cgd0p2_pretrain
CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1_ligmask   # apo (density_mask_ligand=true), default fusion
EXP=voxbind_frozenenc_atomblob7_cgd0p2_apo_sig0.9; EXPDIR=$VOX/exps/$EXP
LOCK=$LOG/chain_frozenenc_apo_cgd02.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another apo-cgd02 chain running — exit"; exit 1; }
drain(){ for _ in $(seq 1 60);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }

# 1) wait for the cgd=0.2 encoder pretrain to finish
DEADLINE=$(( $(date +%s)+10*24*3600 ))
log "waiting for cgd=0.2 encoder pretrain to finish (PRETRAIN-CGD02 DONE) ..."
until grep -q "PRETRAIN-CGD02 DONE" "$POSTLOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for cgd02 pretrain"; exit 1; }
  sleep 300
done
log "cgd02 pretrain done. Draining GPUs."
drain

# 2) resolve the cgd0.2 frozen-encoder checkpoint (latest e*)
DENS=$(ls "$CGD02DIR"/checkpoint_e*.pth.tar 2>/dev/null | sort | tail -1)
[ -z "$DENS" ] && { log "ABORT: no cgd0.2 checkpoint found in $CGD02DIR"; exit 1; }
log "frozen encoder = $DENS"

# 3) train apo σ=0.9 frozen-enc on the cgd0.2 encoder, 350 ep, from scratch
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "===== TRAIN apo-cgd0.2 frozen-enc (sig=0.9, 350 ep) exp=$EXP → $RUNLOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 "$VOX/train_ddp.py" \
  --config-name="$CFG" smooth_sigma=0.9 wjs.n_targets=0 num_epochs=350 \
  model.density_pretrained_path="$DENS" \
  exp_name="$EXP" output_dir="$EXPDIR" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
log "apo-cgd0.2 train exit=$? (target ep350)"
drain

# 4) density-conditioned eval (sample.py loads the exp cfg → density_mask_ligand + cgd0.2 encoder both applied)
EP=$("$B/python" -c "import torch;print(torch.load('$EXPDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
log "===== EVAL apo-cgd0.2 frozen-enc (@ ep$EP) ====="
bash "$VOX/scripts/frozenenc_full_eval.sh" "$EXPDIR" "$EXPDIR/samples/full_eval_ep${EP}" 100
log "APO-CGD02 DONE (@ ep$EP). Compare Vina to #29 apo (-7.92, OOD) and holo (-8.24). Add its row to voxbind_results.html."
