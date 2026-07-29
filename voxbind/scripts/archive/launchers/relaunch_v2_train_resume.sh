#!/usr/bin/env bash
# Relaunch the v2 pretrain at a proper bsz, REUSING the existing build (no re-download/rebuild).
# Why: the original chain's bsz-probe ran train_density.py as plain `python` (no torchrun) →
# KeyError: 'RANK' → every probe insta-failed → it fell back to bsz=8. train_density.py REQUIRES
# torchrun. Here we just relaunch the 4-GPU run at $BSZ, then resume the denoiser on exit.
#   BSZ=32 setsid nohup bash scripts/archive/launchers/relaunch_v2_train_resume.sh > log/relaunch_v2.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_v2_full
EXP=260627_plinder_v2_otf_roleblob_diverse_cdg_channelvit_full_pretrain
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
BSZ=${BSZ:-32}
NWORK=${NWORK:-28}       # 28×4=112 single-threaded workers on 128 cores (no oversubscription)
SUBSET_N=112633          # manifest positions 112733 − 100 val
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
# Thread balance: OMP=1 → workers single-threaded but MAIN process starved (collation/glue
# single-threaded → GPU starves at ~40%). OMP=8 → 960 threads thrash. Middle ground via $OMP.
OMP=${OMP:-1}
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }

# stop any leftover v2 train, drain GPUs
pkill -9 -f 'train_density.py.*v2_full' && log "stopped leftover v2 train" || log "no leftover v2 train"
sleep 5
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB). LAUNCH 4-GPU bsz=$BSZ accum=1 workers=$NWORK subset_n=$SUBSET_N (eff batch=$((BSZ*4)))"

RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" mae.val_every=5 \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "v2 pretrain exit=$RC → $RUNLOG (ckpt: exps/$EXP)"

# resume denoiser
if [ -f "$DENOISER_CK" ]; then
  setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "denoiser resume re-armed."
fi
log "V2 RELAUNCH DONE (rc=$RC)."
