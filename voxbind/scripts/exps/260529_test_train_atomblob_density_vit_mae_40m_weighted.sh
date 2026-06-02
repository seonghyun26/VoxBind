#!/bin/bash
# 35_train_atomblob_density_vit_mae_40m_weighted.sh — fully-weighted atomblob+density.
#
# Stacks three rebalancing knobs on top of the atomblob+density baseline:
#   1. mae.channel_weighting=inv_sqrt_freq     — inv-sqrt-freq atom weighting
#   2. mae.mask_strategy=atom_biased           — atoms-first 60% block mask
#   3. mae.density_channel_weight=0.1          — density down-weighted (~57% gradient share
#                                                vs ~93% in the baseline)
#
# GPU 4 and 5 only → world_size=2; bumped accum_steps=5 keeps effective batch = 80
# (matching the 5-GPU baseline runs for clean A/B). Expected wallclock ~28h
# (vs 11h on 5 GPUs) since the same per-epoch sample count is split across
# fewer devices.
#
# Output: exps/260529_atomblob_density_vit_mae_40m_weighted_pretrain/checkpoint.pth.tar
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260529_atomblob_density_vit_mae_40m_weighted_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 4,5 | bsz 8 × 2 × accum 5 = 80 eff | atomblob+density + inv-sqrt-freq + atom-biased mask + density_weight=0.1)"

CUDA_VISIBLE_DEVICES=4,5 $PY/torchrun --standalone --nproc_per_node=2 train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_vit_mae_40m_weighted \
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
    accum_steps=5 \
    'wandb_tags=[pretrain,atomblob_density_vit_mae,40m,weighted,atom_biased_mask,inv_sqrt_freq,density_downweight,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
