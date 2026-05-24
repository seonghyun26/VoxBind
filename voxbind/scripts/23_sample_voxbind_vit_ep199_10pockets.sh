#!/bin/bash
# 23_sample_voxbind_vit_ep199_10pockets.sh — broader-coverage sample from the
# 260523_voxbind_10k_density_vit_mae_frozen ep199 snapshot (10 val pockets ×
# 10 ligands = 100 ligands), then Vina dock eval.
#
# Goal: reduce per-pocket noise in the headline mean. The earlier 2-pocket
# quick10 was too small to resolve ViT-vs-CNN or ep199-vs-e130 differences.
#
# Out: exps/260523_voxbind_10k_density_vit_mae_frozen_e0199_snap/samples/res_ep199_10pockets/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260523_voxbind_10k_density_vit_mae_frozen_e0199_snap
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_ep199_10pockets
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling 10 pockets × 10 ligands from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

# Phase 1: WJS sample (GPU)
CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=val wjs.n_targets=9 wjs.n_samples_per_pocket=10 \
    out_dir=res_ep199_10pockets \
    save_dir=$OUT \
    > "$LOG/260524_sample_voxbind_vit_ep199_10pockets.log" 2>&1
SAMPLE_RC=$?
echo "[$(ts)] WJS sample done (exit $SAMPLE_RC)"
[ $SAMPLE_RC -ne 0 ] && exit $SAMPLE_RC

# Phase 2: Vina dock eval (CPU). workers=8 cpu=4 = 32 cores — leaves plenty
# for the parallel resume training on GPUs 1-6.
echo "[$(ts)] starting Vina dock eval (workers=8 cpu=4)"
$PY $VOX/../notebook/webapp/metrics.py "$OUT" \
    --docking vina_dock --workers 8 --cpu 4 --skip-existing \
    >> "$LOG/260524_sample_voxbind_vit_ep199_10pockets.log" 2>&1
EVAL_RC=$?
echo "[$(ts)] vina eval done (exit $EVAL_RC)  ->  $OUT"
