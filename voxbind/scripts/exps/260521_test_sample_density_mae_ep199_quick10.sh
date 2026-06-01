#!/bin/bash
# 17_sample_density_mae_ep199_quick10.sh — re-sample the SAME two val pockets
# (target_00 FIMH_ECOLI, target_01 PNMT_HUMAN) from the ep199 checkpoint, so
# we can compare ep99 vs ep199 generation quality side-by-side.
#
# Uses wjs.n_targets=1 (same as 16_*.sh) which, with sample.py's off-by-one,
# produces 2 pockets — exactly the two we already have at ep99.
#
# Out: exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep199_quick10/
# Compare with: exps/260521_voxbind_10k_density_mae_frozen/samples/res_ep99_quick10/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260521_voxbind_10k_density_mae_frozen
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_ep199_quick10
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling 10 ligands × 2 pockets from $EXP @ ep199 on GPU $GPU"
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
    > "$LOG/260522_sample_density_mae_ep199_quick10.log" 2>&1

echo "[$(ts)] sample done (exit $?)  ->  $OUT"
