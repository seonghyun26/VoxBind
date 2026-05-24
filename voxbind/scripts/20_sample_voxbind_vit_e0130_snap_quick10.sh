#!/bin/bash
# 20_sample_voxbind_vit_e0130_snap_quick10.sh — quick 10-ligand sanity sample
# from a mid-training snapshot of the 260523_voxbind_10k_density_vit_mae_frozen
# run (epoch 129, snapshotted while training continues on GPUs 1-6).
#
# Runs on GPU 7 in parallel with the main training. Same recipe as
# 16_sample_density_mae_quick10.sh so the output is directly comparable to the
# cnn-MAE-frozen ep99 samples (`exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep99_quick10/`).
#
# 1 wjs.n_targets → sample.py off-by-one yields 2 val pockets × 10 = 20 ligands.
#
# Out: exps/260523_voxbind_10k_density_vit_mae_frozen_e0130_snap/samples/res_e0130_quick10/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260523_voxbind_10k_density_vit_mae_frozen_e0130_snap
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_e0130_quick10
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling 10 ligands from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=val wjs.n_targets=1 wjs.n_samples_per_pocket=10 \
    out_dir=res_e0130_quick10 \
    save_dir=$OUT \
    > "$LOG/260523_sample_voxbind_vit_e0130_snap_quick10.log" 2>&1

echo "[$(ts)] sample done (exit $?)  ->  $OUT"
