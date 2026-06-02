#!/bin/bash
# 18_train_density_vit_mae.sh — voxel-MAE pre-training for the ViT density encoder.
#
# Same pretext as 13_train_density_mae.sh (synthetic Gaussian-blur density +
# 3D block masking + density-reconstruction + 11-channel atom-structure heads),
# but the encoder is a pure 3D ViT (see models/density_vit.py) instead of the
# CNN-stack. Apples-to-apples vs the 260521 cnn-MAE run.
#
# Output: exps/260522_density_vit_mae_pretrain/ — checkpoint.pth.tar carries
# both `state_dict_ema` (full DensityViTMAE EMA) and `encoder_state_dict_ema`
# (encoder.* slice — drops into VoxBind.density_encoder with
# density_encoder_type=vit via the existing loader in models/__init__.py).
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260522_density_vit_mae_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-6 | dataset=full crossdocked | bsz 16 × 6 = 96 eff)"

# bsz=16/rank to mirror the cnn-MAE precedent (260521_density_mae_posweight) so
# the ViT vs CNN ablation differs only in encoder, not effective batch size.
# ViT activations should be lighter than the 4-block CNN encoder at the same
# input; if memory headroom is large after the first epoch, bump to bsz=32 in a
# follow-up run.
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $PY/torchrun --standalone --nproc_per_node=6 train_density_vit_mae.py \
    dset=crossdocked \
    dset.data_dir=$DATA \
    num_epochs=100 \
    bsz=16 \
    accum_steps=1 \
    'wandb_tags=[pretrain,density_vit_mae,stage_a,posweight_100,crossdocked_train]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
