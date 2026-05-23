#!/bin/bash
# 14_train_10k_density_mae.sh — VoxBind 10k x-ray training with FROZEN
# pre-trained density encoder (260521 density-MAE).
#
# Phase 3 of the density-MAE pretraining plan:
#   * encoder = 4 ResidualBlocks, weights loaded from
#     exps/260521_density_mae_posweight/checkpoint.pth.tar (encoder_state_dict_ema)
#   * encoder is FROZEN (requires_grad=False, dropout disabled)
#   * density_proj zero-init still bites the fusion path; it trains from scratch
#   * direct comparison: 260518_voxbind_10k_density (no pretrain), 260519_voxbind_10k_density_aligned
#
# Same data + hparams as 260519 (aligned x-ray crops, 10k subset, bsz 4, 100 ep)
# so deltas vs that run isolate the value of MAE-pretrained density features.
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
echo "[$(ts)] launching $EXP  (GPU 1-6 | bsz 4×6 = 24 eff | MAE ckpt: $(basename $MAE_CKPT))"

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
    'wandb_tags=[voxbind,10k,density_mae,frozen_encoder,phase3,aligned_density,xray]' \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
