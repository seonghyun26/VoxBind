#!/bin/bash
# 35_train_baseline_4gpu.sh — parameterized 4xH200 baseline VoxBind launcher.
# Supersedes the old "phase 3 resume" snapshot of this file; the same script
# now handles both fresh launches and resumes via env vars. Defaults reproduce
# the phase-3 invocation (SIG=0.9 NUM_EPOCHS=400 RESUME_EPOCH=800).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/35_train_baseline_4gpu.sh
#       -> reproduces phase 3: resume exp_sig0.9 from ep799 -> ep800..ep1199
#
#   SIG=1.0 NUM_EPOCHS=1200 bash scripts/35_train_baseline_4gpu.sh
#       -> fresh sig=1.0 baseline, 1200 epochs from scratch
#
#   SIG=0.9 NUM_EPOCHS=400 RESUME_EPOCH=1200 bash scripts/35_train_baseline_4gpu.sh
#       -> further-extend sig=0.9 from ep1199 -> ep1200..ep1599
#
# ── Env vars ──────────────────────────────────────────────────────────────────
#   SIG           smooth_sigma value                            (default 0.9)
#   NUM_EPOCHS    epochs to run THIS launch                     (default 400)
#   RESUME_EPOCH  if set, resume=$VOX/exps/$EXP resume_epoch=N
#                 if unset, fresh launch (no resume args)       (default 800)
#
# Effective batch is locked at 128 (bsz=32 on 4 ranks, accum=1). exp_name auto-
# resolves to exp_sig${SIG}+prefetch_factor16_wjs.n_targets0 via Hydra's
# override_dirname (bsz/accum_steps/num_workers/num_epochs are excluded in
# config_train.yaml, so they don't pollute the dir name).
#
# Per the 15_/22_ lesson: pass the FULL hparam arg set on resume too, since
# create_exp_dir rewrites cfg.yaml from CLI cfg BEFORE the resume branch
# reloads it -- any flag omitted would silently revert to config_train.yaml
# defaults.
set -u

VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$VOX/log

SIG=${SIG:-0.9}
NUM_EPOCHS=${NUM_EPOCHS:-400}
RESUME_EPOCH=${RESUME_EPOCH-800}   # default 800 reproduces phase-3; set empty for fresh launch

EXP=exp_sig${SIG}+prefetch_factor16_wjs.n_targets0
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1

ARGS=(
    smooth_sigma=$SIG
    bsz=32 accum_steps=1
    num_workers=12
    +prefetch_factor=16
    num_epochs=$NUM_EPOCHS
    wjs.n_targets=0
)
if [ -n "$RESUME_EPOCH" ]; then
    ARGS+=( resume=$VOX/exps/$EXP resume_epoch=$RESUME_EPOCH )
    MODE="resume @ep$RESUME_EPOCH +$NUM_EPOCHS epochs"
else
    MODE="fresh $NUM_EPOCHS epochs"
fi

echo "[$(ts)] sig=$SIG  exp=$EXP  $MODE  (4xH200, bsz 32x4=128)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY/torchrun --standalone --nproc_per_node=4 train_ddp.py "${ARGS[@]}"
RC=$?
echo "[$(ts)] sig=$SIG done (exit $RC)  ->  exps/$EXP"
exit $RC
