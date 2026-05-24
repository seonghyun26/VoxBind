#!/bin/bash
# 16_sample_density_mae_quick10.sh — quick 10-ligand sanity sample from the
# 260521_voxbind_10k_density_mae_frozen ep99 checkpoint, on GPU 7.
#
# 1 val pocket × 10 samples = 10 ligands. Val split is the safe choice
# (every val pocket has x-ray density, so the density-aware sample.py
# won't skip).
#
# Out: exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep99_quick10/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260521_voxbind_10k_density_mae_frozen
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_ep99_quick10
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
    out_dir=res_ep99_quick10 \
    save_dir=$OUT \
    > "$LOG/260522_sample_density_mae_quick10.log" 2>&1

echo "[$(ts)] sample done (exit $?)  ->  $OUT"
