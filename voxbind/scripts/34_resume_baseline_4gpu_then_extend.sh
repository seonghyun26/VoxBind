#!/bin/bash
# 34_resume_baseline_4gpu_then_extend.sh
# Resume the baseline VoxBind run exp_sig0.9+prefetch_factor16_wjs.n_targets0
# on 4x H200. The original 8-GPU run died at ep67 (last logged val miou 0.5454);
# phase 1 finishes the original 400-epoch budget, phase 2 extends to 800 total.
#
# Effective batch is held at 128 (was 8x16) by going to bsz=32 on 4 ranks --
# H200's 143GB easily absorbs the doubled per-rank batch, and matching the
# effective batch preserves the optimization trajectory the checkpoint encodes.
#
# Per the 15_/22_ resume scripts in this repo: pass the FULL hparam arg set
# even on resume, because create_exp_dir rewrites cfg.yaml from the current
# cfg BEFORE train_ddp.py's resume branch reloads it -- any flag omitted here
# would silently revert to config_train.yaml defaults.
#
# resume_epoch is set explicitly to skip the codebase's "+0" convention
# (without it the loop would re-run the saved ep67 once before continuing).
set -u

VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
EXP=exp_sig0.9+prefetch_factor16_wjs.n_targets0
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1

# ---------- phase 1: ep68..ep399 (332 epochs to finish original 400-epoch plan)
echo "[$(ts)] PHASE 1: resume $EXP ep67 -> ep68..ep399 (332 epochs, 4x H200, bsz 32x4=128)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY/torchrun --standalone --nproc_per_node=4 train_ddp.py \
    smooth_sigma=0.9 \
    bsz=32 accum_steps=1 \
    num_workers=12 \
    +prefetch_factor=16 \
    num_epochs=332 \
    wjs.n_targets=0 \
    resume=$VOX/exps/$EXP \
    resume_epoch=68
P1=$?
echo "[$(ts)] PHASE 1 exit=$P1"
if [ $P1 -ne 0 ]; then
    echo "[$(ts)] phase 1 failed (exit $P1) -- skipping phase 2"
    exit $P1
fi

# ---------- phase 2: ep400..ep799 (another 400 epochs -> 800 total)
echo "[$(ts)] PHASE 2: continue $EXP ep399 -> ep400..ep799 (+400 epochs, total 800)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY/torchrun --standalone --nproc_per_node=4 train_ddp.py \
    smooth_sigma=0.9 \
    bsz=32 accum_steps=1 \
    num_workers=12 \
    +prefetch_factor=16 \
    num_epochs=400 \
    wjs.n_targets=0 \
    resume=$VOX/exps/$EXP \
    resume_epoch=400
P2=$?
echo "[$(ts)] PHASE 2 exit=$P2  ->  exps/$EXP"
exit $P2
