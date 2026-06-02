#!/bin/bash
# 15_train_10k_density_mae_resume.sh — resume 260521_voxbind_10k_density_mae_frozen
# for 100 more epochs (ep100..ep199), same hparams, same MAE-frozen encoder.
#
# Phase 3 ended at ep99 with val mIoU 0.505 / val loss 229 — clearly still
# climbing. Extending to see if it catches up to the no-pretrain baseline
# (260519 aligned: val mIoU 0.575 / val loss 159).
#
# resume_epoch=100 is set explicitly so the loop starts at ep100 (otherwise
# train_ddp.py reuses the saved ep99 once before continuing — the "+0" resume
# convention in this codebase).
#
# IMPORTANT (per script 11 lesson): pass the FULL original arg set even on
# resume — create_exp_dir rewrites cfg.yaml from the current cfg BEFORE the
# resume branch reloads it, so omitting flags reverts them to defaults.
#
# GPU plan: 1-6 (eff batch 4×6×1 = 24).
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260521_voxbind_10k_density_mae_frozen
MAE_CKPT=$VOX/exps/260521_density_mae_posweight/checkpoint.pth.tar
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] RESUMING $EXP from ep99 → ep100..ep199 (GPU 1-6)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $PY/torchrun --standalone --nproc_per_node=6 train_ddp.py \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    num_epochs=100 bsz=4 accum_steps=1 seed=42 \
    wjs.n_targets=0 \
    model.with_density=true \
    model.density_encoder_blocks=4 \
    model.density_pretrained_path=$MAE_CKPT \
    model.density_freeze=true \
    'wandb_tags=[voxbind,10k,density_mae,frozen_encoder,phase3,aligned_density,xray,resume_ep100to199]' \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    resume=$VOX/exps/$EXP \
    resume_epoch=100 \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] resume done (exit $?)  ->  exps/$EXP"
