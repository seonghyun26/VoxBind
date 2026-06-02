#!/bin/bash
# 34_train_atomblob_vit_mae_40m_weighted.sh — ViT-MAE pre-training on ATOM-BLOB
# voxels with TWO new knobs vs scripts/32:
#   1. mae.mask_strategy=atom_biased   — blocks containing atoms are masked
#      first (noisy top-K over per-block atom mass), encouraging the model
#      to spend capacity on informative regions rather than empty space.
#   2. mae.channel_weighting=inv_sqrt_freq — per-channel masked-MSE weighted
#      by 1/sqrt(pos-voxel frequency), normalized to sum=11. Rare atom types
#      (S/F/Cl/P) are up-weighted so they actually contribute to learning.
#
# Same X-ray-available subset (subset_xray_only=true, n=78428) as
# 32_train_atomblob_vit_mae_40m.sh for a clean A/B against the baseline.
# use_xray=false: no per-sample CCP4 read.
#
# Output: exps/260528_atomblob_vit_mae_40m_weighted_pretrain/checkpoint.pth.tar
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260528_atomblob_vit_mae_40m_weighted_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | bsz 8 × 5 × accum 2 = 80 eff | atomblob + atom-biased mask + inv-sqrt-freq ch weights)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_density_vit_mae.py \
    --config-name=config_train_atomblob_vit_mae_40m_weighted \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=false \
    input_mode=atomblob \
    num_epochs=100 \
    bsz=8 \
    accum_steps=2 \
    'wandb_tags=[pretrain,atomblob_vit_mae,40m,weighted,atom_biased_mask,inv_sqrt_freq,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
