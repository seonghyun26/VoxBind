#!/usr/bin/env bash
# v3 FUSION run. After #29 (ligmask/apo) finishes (FROZENENC-LIGMASK CHAIN DONE), train the v3
# frozen-enc VoxBind at sigma=0.9 for 350 epochs from scratch — the frozen encoder REPLACES the
# pocket_encoder (pocket + apo density via the frozen ViT, fused with a normal-init context_proj).
# Then run its density-conditioned 79-pocket eval. Clean A/B vs #29 (apo + default fusion).
#   setsid nohup bash scripts/archive/chains/chain_frozenenc_v3.sh > log/chain_frozenenc_v3.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
POSTLOG=$LOG/chain_frozenenc_ligmask.log
CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1_v3
EXP=voxbind_frozenenc_atomblob7_v2p1_v3_sig0.9; EXPDIR=$VOX/exps/$EXP
LOCK=$LOG/chain_frozenenc_v3.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another frozenenc-v3 chain running — exit"; exit 1; }
drain(){ for _ in $(seq 1 48);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }

# 1) wait for the #29 ligmask (apo) pipeline to finish
DEADLINE=$(( $(date +%s)+8*24*3600 ))
log "waiting for #29 ligmask pipeline to finish (FROZENENC-LIGMASK CHAIN DONE) ..."
until grep -q "FROZENENC-LIGMASK CHAIN DONE" "$POSTLOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for ligmask pipeline"; exit 1; }
  sleep 600
done
log "#29 ligmask done. Draining GPUs (its eval sampling is done; docking is CPU-bg)."
drain

# 2) train v3 frozen-enc, sigma=0.9, 350 ep, from scratch
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "===== TRAIN v3 frozen-enc (sig=0.9, 350 ep, from scratch) exp=$EXP → $RUNLOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 "$VOX/train_ddp.py" \
  --config-name="$CFG" smooth_sigma=0.9 wjs.n_targets=0 num_epochs=350 \
  exp_name="$EXP" output_dir="$EXPDIR" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
log "v3 frozen-enc train exit=$? (target ep350)"
drain

# 3) density-conditioned eval (sample.py loads the exp's cfg.yaml → fusion=v3 + density_mask_ligand
#    both applied automatically; no eval-script change needed)
EP=$("$B/python" -c "import torch;print(torch.load('$EXPDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null)
log "===== EVAL v3 frozen-enc (@ ep$EP) ====="
bash "$VOX/scripts/archive/workflows/frozenenc_full_eval.sh" "$EXPDIR" "$EXPDIR/samples/full_eval_ep${EP}" 100
log "FROZENENC-V3 CHAIN DONE (@ ep$EP). Add its row to voxbind_results.html (rerun the generator with the new json)."
