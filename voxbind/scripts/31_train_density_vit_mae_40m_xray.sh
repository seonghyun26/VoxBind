#!/bin/bash
# 31_train_density_vit_mae_40m_xray.sh — ViT-MAE pre-training on REAL x-ray density.
#
# Successor to 18_train_density_vit_mae.sh (260522, ViT-MAE 4.5M, synthetic
# Gaussian-blur density). Two changes vs that run:
#   (a) encoder bumped from 4.5M (D=192, L=6, H=6) to 40M (D=512, L=12, H=8),
#   (b) pretext input swapped from synthesized-from-atoms to REAL 2Fo-Fc crops
#       loaded from dataset/data/xray_crops_aligned/.
#
# Pretext is otherwise unchanged: 3-D block-mask (60% of 8³ cubes zeroed in
# input) + masked-voxel MSE on the un-masked density + 11-ch atom-structure
# auxiliary head. Additive noise is disabled (real density already has it).
#
# Output: exps/260526_density_vit_mae_40m_xray_pretrain/ — checkpoint.pth.tar
# carries `state_dict_ema` (full DensityViTMAE) and `encoder_state_dict_ema`
# (encoder.* slice — drops into VoxBind.density_encoder with
# density_encoder_type=vit via the existing loader in models/__init__.py).
#
# Apples-to-apples downstream comparison target: re-run 19_train_10k_density_vit_mae.sh
# pointing density_pretrained_path at this run's checkpoint and updating the
# density_vit block in configs/model/voxbind.yaml to D=512/L=12/H=8.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260526_density_vit_mae_40m_xray_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | bsz 8 × 5 × accum 2 = 80 eff | xray_crops_aligned)"

# bsz=8/rank to match the 4.5M synthetic precedent eff batch (96), accum=2 to
# soak up the lighter wall-clock per step (no in-loop conv blur in xray mode).
# 5 ranks × bsz 8 × accum 2 = 80 eff (close enough; the 4.5M run used 6×16=96).
CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_density_vit_mae.py \
    --config-name=config_train_density_vit_mae_40m_xray \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    num_epochs=100 \
    bsz=8 \
    accum_steps=2 \
    'wandb_tags=[pretrain,density_vit_mae,40m,real_density,xray,stage_a,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
