#!/bin/bash
# 22_train_10k_density_vit_mae_resume.sh — resume 260523_voxbind_10k_density_vit_mae_frozen
# for 180 more epochs (ep200..ep379), same hparams, same ViT-MAE-frozen encoder.
#
# Phase 3 ended at ep199 with val mIoU 0.592 / val loss 147 (cnn-MAE-frozen
# ep199 reached val mIoU 0.567). ViT trajectory still climbing — extending to
# see how high it goes and where it plateaus.
#
# resume_epoch=200 forces start at ep200 (per the codebase '+0' resume
# convention noted in 15_*.sh — without it train_ddp.py re-does the saved
# ep199 once before continuing).
#
# IMPORTANT (per 11_/15_ lesson): pass FULL original arg set even on resume —
# create_exp_dir rewrites cfg.yaml from the current cfg BEFORE the resume
# branch reloads it, so omitting flags reverts them to defaults.
#
# GPU plan: 1-6 (eff batch 4×6×1 = 24). GPU 7 used by 21_sample_*.sh in parallel.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260523_voxbind_10k_density_vit_mae_frozen
MAE_CKPT=$VOX/exps/260522_density_vit_mae_pretrain/checkpoint.pth.tar
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] RESUMING $EXP from ep199 → ep200..ep379 (GPU 1-6)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $PY/torchrun --standalone --nproc_per_node=6 train_ddp.py \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    num_epochs=180 bsz=4 accum_steps=1 seed=42 \
    wjs.n_targets=0 \
    model.with_density=true \
    model.density_encoder_type=vit \
    model.density_pretrained_path=$MAE_CKPT \
    model.density_freeze=true \
    'wandb_tags=[voxbind,10k,density_vit_mae,frozen_encoder,vit,phase3,aligned_density,xray,resume_ep200to379]' \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    resume=$VOX/exps/$EXP \
    resume_epoch=200 \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] resume done (exit $?)  ->  exps/$EXP"
