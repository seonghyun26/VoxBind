#!/bin/bash
# 21_sample_voxbind_vit_ep199_quick10.sh — quick 10-ligand sample from the
# 260523_voxbind_10k_density_vit_mae_frozen final ep199 checkpoint (snapshotted
# to *_e0199_snap before the resume run overwrites checkpoint.pth.tar).
#
# Apples-to-apples vs:
#   - cnn-MAE ep199 quick10: exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep199_quick10/
#   - cnn-MAE ep99  quick10: exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep99_quick10/
#   - vit-MAE e130  quick10: exps/260523_voxbind_10k_density_vit_mae_frozen_e0130_snap/samples/res_e0130_quick10/
#
# Same recipe (split=val, n_targets=1 → 2 pockets × 10 ligands = 20). GPU 7
# only, so the parallel resume train on GPUs 1-6 isn't disturbed.
#
# Out: exps/260523_voxbind_10k_density_vit_mae_frozen_e0199_snap/samples/res_ep199_quick10/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260523_voxbind_10k_density_vit_mae_frozen_e0199_snap
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_ep199_quick10
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling 10 ligands × 2 pockets from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=val wjs.n_targets=1 wjs.n_samples_per_pocket=10 \
    out_dir=res_ep199_quick10 \
    save_dir=$OUT \
    > "$LOG/260523_sample_voxbind_vit_ep199_quick10.log" 2>&1

echo "[$(ts)] sample done (exit $?)  ->  $OUT"
