#!/usr/bin/env bash
# New ChannelViT density-MAE pretrain with channel_group_dropout=0.2 — whole channel-groups
# (INCLUDING the ligand) are dropped ~20% of forwards, so the encoder learns to work with the
# ligand present AND absent. This makes ligand-free (apo) input IN-DISTRIBUTION for the encoder,
# dissolving the leakage-vs-OOD confound in #29: on this encoder, a holo-vs-apo frozen-enc gap
# would be PURE leakage. Same recipe as 260701 otherwise (atomblob7, PLINDER v2.1, C+D+G, MAE 0.5).
# After #29→v3 finish (FROZENENC-V3 CHAIN DONE). Runs before σ0.8.
#   setsid nohup bash scripts/archive/chains/chain_pretrain_cgd02.sh > log/chain_pretrain_cgd02.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; LOG=$VOX/log
POSTLOG=$LOG/chain_frozenenc_v3.log
CFG=config_train_atomblob7_density_gradmag_channelvit_mae_40m_plinder_v2p1_box
EXP=260718_plinder_v2p1_box_atomblob7_cdg_channelvit_cgd0p2_pretrain; EXPDIR=$VOX/exps/$EXP
RD=$VOX/dataset/data/pretrain/xray_resample_plinder_v2p1; MANIFEST=$RD/train_manifest.npz
LOCK=$LOG/chain_pretrain_cgd02.lock
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another cgd02-pretrain chain running — exit"; exit 1; }
drain(){ for _ in $(seq 1 48);do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}');[ "${U:-9}" -lt 3000 ]&&break;sleep 5;done; log "GPU drained (${U:-?} MiB)"; }

# 1) wait for the v3 frozen-enc pipeline to finish
DEADLINE=$(( $(date +%s)+10*24*3600 ))
log "waiting for v3 frozen-enc pipeline to finish (FROZENENC-V3 CHAIN DONE) ..."
until grep -q "FROZENENC-V3 CHAIN DONE" "$POSTLOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for v3 pipeline"; exit 1; }
  sleep 600
done
log "v3 pipeline done. Draining GPUs."
drain

# 2) pretrain the cgd=0.2 encoder (100 ep, 4-GPU) — same recipe as 260701 + channel_group_dropout
NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))" 2>/dev/null)
SUBSET_N=$(( NPOS - 100 ))
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "===== PRETRAIN cgd=0.2 encoder (4-GPU, subset_n=$SUBSET_N) exp=$EXP → $RUNLOG ====="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz=32 accum_steps=1 num_workers=16 \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  model.channel_group_dropout=0.2 \
  exp_name="$EXP" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
log "cgd=0.2 pretrain exit=$? (target ep100)"
EP=$(ls "$EXPDIR"/checkpoint_e*.pth.tar 2>/dev/null | sort | tail -1)
log "PRETRAIN-CGD02 DONE. encoder ckpt=$EP"
log "NEXT: run frozen-enc HOLO + APO on this encoder (density_pretrained_path=$EP) → holo−apo gap = pure leakage."
