#!/bin/bash
# 33_train_atomblob_density_vit_mae_40m.sh — ViT-MAE on ATOM-BLOB + X-RAY DENSITY.
#
# The multi-channel arm of the ablation pair (see 32_train_atomblob_vit_mae_40m.sh).
# Same encoder size (40M, D=512 L=12 H=8) and the same X-ray-available subset
# (subset_xray_only=true, subset_n=78428, subset_val_n=100) as both companion
# runs, so the three pretraining variants (density-only / atomblob /
# atomblob+density) are directly comparable head-to-head.
#
# Encoder input: (B, 12, G³) — 7 ligand + 4 pocket atom-type voxels concatenated
# with the z-scored real 2Fo-Fc density (xray_crops_aligned/).
# Pretext: 3-D block-mask (60% of 8³ cubes zeroed) + masked-voxel MSE across
# all 12 channels jointly. Per-modality recon (atom vs density) is logged to
# wandb (L_dens_atom, L_dens_density) so we can spot scale imbalance.
# Atom channels are never noised (no experimental noise on synthesized atoms);
# only the trailing density channel goes through the sigma_noise pipeline,
# but sigma_noise is set to 0 here to mirror the density-only baseline.
#
# Output: exps/260526_atomblob_density_vit_mae_40m_pretrain/
# (encoder slice loads into a DensityViT(n_in_channels=12) — downstream
# VoxBind integration needs a separate refactor of voxbind.py).
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260526_atomblob_density_vit_mae_40m_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | bsz 8 × 5 × accum 2 = 80 eff | atomblob + real xray)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_vit_mae_40m \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    input_mode=atomblob_density \
    num_epochs=100 \
    bsz=8 \
    accum_steps=2 \
    'wandb_tags=[pretrain,atomblob_density_vit_mae,40m,real_density,xray,stage_a,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
