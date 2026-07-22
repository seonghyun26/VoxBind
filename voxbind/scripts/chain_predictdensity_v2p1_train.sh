#!/usr/bin/env bash
# Predict-density-from-structure (v2.1 atomblob7): encoder sees ONLY atoms (density+gradmag
# force-masked); decoder reconstructs atoms+density+gradmag. Queued AFTER the v2.1 atomblob7
# (density-as-input) run. wait v2.1 done → stop denoiser → train 100ep → resume denoiser.
# (Probe handled separately — needs density+gradmag zeroed at test time to match training.)
#   setsid nohup bash scripts/chain_predictdensity_v2p1_train.sh > log/chain_predictdensity_v2p1.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_atomblob7_predictdensity_channelvit_mae_40m_plinder_v2p1_box
EXP=260701_plinder_v2p1_predictdens_atomblob7_cdg_channelvit_full_pretrain
MANIFEST=$VOX/dataset/data/pretrain/xray_resample_plinder_v2p1/train_manifest.npz
V2P1_LOG=$LOG/chain_atomblob7_v2p1.log          # the prerequisite run's log
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
BSZ=${BSZ:-32}; NWORK=${NWORK:-16}
LOCK=$LOG/chain_predictdensity_v2p1.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another predictdensity chain running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; }; }

DEADLINE=$(( $(date +%s) + 24*3600 ))
# wait for the v2.1 run to FULLY finish (train+probe+denoiser-resume) — grep-on-file, no pgrep self-match
log "waiting for v2.1 atomblob7 run to finish ..."
until grep -q "ATOMBLOB7 v2.1 CHAIN DONE" "$V2P1_LOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for v2.1"; exit 1; }
  sleep 60
done
log "  v2.1 done."

NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))")
SUBSET_N=$(( NPOS - 100 ))
log "subset_n=$SUBSET_N"

# stop denoiser, drain
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# train (inline) — predict-density (force_mask_groups=[2,3] in the config)
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH predict-density train: 4-GPU bsz=$BSZ workers=$NWORK subset_n=$SUBSET_N → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "predict-density train exit=$RC (ckpt: exps/$EXP)"

resume_denoiser
log "PREDICTDENSITY v2.1 TRAIN DONE (rc=$RC). Probe TODO: feed density+gradmag ZEROED (match masked training)."
