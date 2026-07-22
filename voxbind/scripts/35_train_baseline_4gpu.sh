#!/usr/bin/env bash
# 35_train_baseline_4gpu.sh
#   RECREATED 2026-06-24 — the original was lost in the dataset/scripts refactor, which
#   silently broke the denoiser auto-resume: the chain_sig1.0_* keepgoing watchers call
#   THIS script to (re)launch the sig=SIG diffusion denoiser, so its absence made every
#   relaunch exit 127 ("No such file or directory") and the watcher abort.
#
#   Launches the 4-GPU sig=SIG denoiser (train_ddp.py), resuming from
#   exps/exp_sig${SIG}+prefetch_factor16_wjs.n_targets0. Foreground (blocks) — the
#   watcher relies on that to detect completion.
#
#   Env knobs (the watcher passes these):
#     SIG          smooth_sigma           (default 1.0)
#     NUM_EPOCHS   relative epochs to run (default 400; loop = range(start, start+NUM))
#     RESUME_EPOCH start_epoch override   (optional; = ckpt['epoch']+1)
#
#   Hyperparameters below mirror the original running command (bsz=32, accum_steps=1,
#   num_workers=12, +prefetch_factor=16, wjs.n_targets=0). TORCHDYNAMO_DISABLE=1 because
#   this box has no C compiler (torch.compile/inductor cannot build).
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
SIG="${SIG:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"
RESUME_EPOCH="${RESUME_EPOCH:-}"
EXP="$ROOT/exps/exp_sig${SIG}+prefetch_factor16_wjs.n_targets0"

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$ROOT" || exit 1

echo "[35_train_baseline_4gpu] SIG=$SIG NUM_EPOCHS=$NUM_EPOCHS RESUME_EPOCH=${RESUME_EPOCH:-<none>} EXP=$EXP"
exec "$PY/torchrun" --standalone --nproc_per_node=4 \
  train_ddp.py smooth_sigma="$SIG" bsz=32 accum_steps=1 num_workers=12 \
  +prefetch_factor=16 num_epochs="$NUM_EPOCHS" wjs.n_targets=0 \
  resume="$EXP" ${RESUME_EPOCH:+resume_epoch="$RESUME_EPOCH"}
