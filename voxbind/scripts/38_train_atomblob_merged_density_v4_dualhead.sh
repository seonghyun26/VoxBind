#!/bin/bash
# 38_train_atomblob_merged_density_v4_dualhead.sh — v4 crops + dual_head heads.
#
# Same as 37 (v4 single-head) EXCEPT model.dual_head=true:
#   - head_atoms   : Conv3d(16→16) → SiLU → Conv3d(16→7)   for the 7 merged atom channels
#   - head_density : Conv3d(16→16) → SiLU → Conv3d(16→1)   for the density channel
# Encoder is shared; only the last 2 conv layers split. Tests whether
# decoupled hidden representations at the head help atom vs density specialise.
#
# Wallclock: ~14h on 4 GPUs (4-7). Schedule AFTER 37 (v4 single-head) finishes
# so we have an apples-to-apples A/B at the end.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260602_atomblob_merged_density_vit_mae_40m_weighted_v4_dualhead_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 4-7 | bsz 8 × 4 × accum 3 = 96 eff | v4 crops + dual_head=true)"

CUDA_VISIBLE_DEVICES=4,5,6,7 $PY/torchrun --standalone --nproc_per_node=4 train_density_vit_mae.py \
    --config-name=config_train_atomblob_merged_density_vit_mae_40m_weighted_v4_dualhead \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned_v4 \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    dset.normalize=false \
    input_mode=atomblob_merged_density \
    model.n_in_channels=8 \
    model.dual_head=true \
    model.head_hidden_dim=32 \
    model.head_depth=3 \
    mae.atom_pos_weight=10.0 \
    mae.atom_pos_thresh=0.05 \
    num_epochs=100 \
    bsz=8 \
    accum_steps=3 \
    'wandb_tags=[pretrain,atomblob_merged_density_vit_mae,40m,weighted,merged7,v4,dualhead,bighead,atompos,clip_zscore,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
