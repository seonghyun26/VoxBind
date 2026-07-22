#!/usr/bin/env bash
# Restart the frozen-enc VoxBind from its epoch-99 checkpoint with the in-training
# sampling eval DISABLED. Root cause of the 2026-07-01 23:02 crash: wjs.n_targets=100
# → the rank-0 sample() at epoch 100 sampled ~100 pockets (~2-4h), ranks 1-3 timed out
# at the 2h NCCL barrier → SIGABRT all ranks. Fix: wjs.n_targets=0 (loop breaks at
# pocket 0, no sampling). On exit (success OR crash) resume the sig=1.0 denoiser.
#   setsid nohup bash scripts/chain_frozenenc_restart.sh > log/chain_frozenenc_restart.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
EXP=voxbind_frozenenc_atomblob7_v2p1_sig0.9
EXPDIR=$VOX/exps/$EXP
CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_frozenenc_restart.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another restart chain running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; } || log "WARN: denoiser ckpt missing — cannot resume"; }

# drain (GPUs should already be free after the crash cleanup)
for _ in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-999999}" -lt 3000 ] && break; sleep 5; done
log "GPU free (${U:-?} MiB). Resuming frozen-enc from $EXPDIR (sampling OFF, num_epochs=251 → ~ep350 total)."

RUNLOG="$LOG/${EXP}_restart_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_ddp.py" --config-name="$CFG" \
  resume="$EXPDIR" wjs.n_targets=0 \
  exp_name="$EXP" output_dir="$EXPDIR" hydra.run.dir="$EXPDIR" > "$RUNLOG" 2>&1
RC=$?; log "frozen-enc restart exit=$RC (ckpt: $EXPDIR)"

[ "$RC" -ne 0 ] && log "RESTART FAILED (rc=$RC) — see $RUNLOG. Resuming denoiser per directive." \
               || log "frozen-enc finished (rc=$RC). Resuming denoiser."
resume_denoiser
log "FROZENENC RESTART CHAIN DONE (rc=$RC)."
