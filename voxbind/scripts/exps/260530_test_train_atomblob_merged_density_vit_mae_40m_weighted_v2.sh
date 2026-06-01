#!/bin/bash
# 37_train_atomblob_merged_density_vit_mae_40m_weighted_v2.sh
#
# V2 clone of 36 — same recipe (merged-7 atoms + density + atom-biased mask +
# inv-sqrt-freq + density_weight=0.1) but density crops come from the v2
# pool-z-scored xray_crops_aligned_v2/ (no per-crop ±3σ clip).
#
# Tests the density-clip hypothesis (project memory: density_clip_hypothesis):
# does removing the ±3σ clip and switching to a single dataset-wide z-score
# preserve heavy-atom magnitudes enough to close the gap vs atomblob on the
# PDBbind probe?
#
# GPU 4-7 (4 GPUs) with accum_steps=3 → effective batch 96. ~14-15h.
# Output: exps/260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain/checkpoint.pth.tar
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 4-7 | bsz 8 × 4 × accum 3 = 96 eff | v2 density)"

CUDA_VISIBLE_DEVICES=4,5,6,7 $PY/torchrun --standalone --nproc_per_node=4 train_density_vit_mae.py \
    --config-name=config_train_atomblob_merged_density_vit_mae_40m_weighted_v2 \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned_v2 \
    dset.normalize=false \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    input_mode=atomblob_merged_density \
    model.n_in_channels=8 \
    num_epochs=100 \
    bsz=8 \
    accum_steps=3 \
    'wandb_tags=[pretrain,atomblob_merged_density_vit_mae,40m,weighted,merged7,atom_biased_mask,inv_sqrt_freq,density_downweight,v2_density,pool_zscore,no_clip,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
