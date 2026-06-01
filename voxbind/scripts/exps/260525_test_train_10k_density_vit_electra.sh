#!/bin/bash
# 29_train_10k_density_vit_electra.sh — VoxBind 10k x-ray training with the FROZEN
# pre-trained ViT density encoder from ELECTRA pretext (260524 density-ViT-ELECTRA).
#
# Companion to 19_train_10k_density_vit_mae.sh: same data + hparams, only the
# pretext used to train the encoder differs (ELECTRA rule-based corruption +
# per-patch RTD vs MAE zero-mask + voxel reconstruction). Deltas vs the 260523
# ViT-MAE-frozen run isolate the value of ELECTRA features over MAE features.
#
# Encoder = DensityViT (patch=8, dim=192, depth=6, heads=6), loaded from
# exps/260524_density_vit_electra_pretrain/checkpoint.pth.tar (encoder_state_dict_ema),
# FROZEN (requires_grad=False, dropout disabled).
#
# 200 epochs to match the ViT-MAE downstream precedent (long enough to see
# whether ELECTRA features keep climbing on the diffusion task).
#
# GPU plan: 1-5 (eff batch 4×5×1 = 20). One rank lighter than the ViT-MAE run
# (which used 1-6 = 24 eff) since GPU 6 was unavailable; comparable enough for
# the ablation.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260525_voxbind_10k_density_vit_electra_frozen
ELECTRA_CKPT=$VOX/exps/260524_density_vit_electra_pretrain/checkpoint.pth.tar
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | bsz 4×5 = 20 eff | ELECTRA ckpt: $(basename $ELECTRA_CKPT))"

CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_ddp.py \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    num_epochs=200 bsz=4 accum_steps=1 seed=42 \
    wjs.n_targets=0 \
    model.with_density=true \
    model.density_encoder_type=vit \
    model.density_pretrained_path=$ELECTRA_CKPT \
    model.density_freeze=true \
    'wandb_tags=[voxbind,10k,density_vit_electra,frozen_encoder,vit,phase3,aligned_density,xray]' \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
