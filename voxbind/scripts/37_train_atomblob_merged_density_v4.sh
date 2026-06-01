#!/bin/bash
# 37_train_atomblob_merged_density_v4.sh — merged + density variant on v4 crops.
#
# v4 = pocket-pool clip + z-score:
#   - clip raw to [-0.578, +1.647] (P_0.1 / P_99.9 of the v2 raw distribution)
#   - then z-score with pool stats recomputed AFTER clipping (μ=0.0015, σ=0.293)
#   - final range observed: [-1.98, +5.62] — preserves atom-peak signal while
#     killing the long-tail outliers that collapsed v3 to ~zero.
#
# Same recipe as 36 (v1 merged-density) and the v2/v3 variants. The only delta
# is dset.crops_dir.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260601_atomblob_merged_density_vit_mae_40m_weighted_v4_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 4,6,7 | bsz 8 × 3 × accum 4 = 96 eff | merged+density on v4 crops)"

CUDA_VISIBLE_DEVICES=4,6,7 $PY/torchrun --standalone --nproc_per_node=3 train_density_vit_mae.py \
    --config-name=config_train_atomblob_merged_density_vit_mae_40m_weighted_v4 \
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
    num_epochs=100 \
    bsz=8 \
    accum_steps=4 \
    'wandb_tags=[pretrain,atomblob_merged_density_vit_mae,40m,weighted,merged7,v4,clip_zscore,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
