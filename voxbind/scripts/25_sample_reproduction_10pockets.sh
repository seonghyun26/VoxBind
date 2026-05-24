#!/bin/bash
# 25_sample_reproduction_10pockets.sh — sample the reproduced VoxBind baseline
# (exps/reproduction/, no density, ckpt ep86) on the SAME 10 val pockets as
# scripts/23_*.sh and 24_*.sh so the three runs are apples-to-apples.
#
# The reproduction model was trained on plain `crossdocked` (no density branch),
# but we use the same crossdocked_xray dset config here for pocket enumeration —
# the no-density model just ignores the density input.
#
# Out: exps/reproduction/samples/res_repro_10pockets/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=reproduction
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_repro_10pockets
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling 10 pockets × 10 ligands from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=val wjs.n_targets=9 wjs.n_samples_per_pocket=10 \
    out_dir=res_repro_10pockets \
    save_dir=$OUT \
    > "$LOG/260524_sample_reproduction_10pockets.log" 2>&1
SAMPLE_RC=$?
echo "[$(ts)] WJS sample done (exit $SAMPLE_RC)"
[ $SAMPLE_RC -ne 0 ] && exit $SAMPLE_RC

echo "[$(ts)] starting Vina dock eval (workers=16 cpu=4)"
$PY $VOX/../notebook/webapp/metrics.py "$OUT" \
    --docking vina_dock --workers 16 --cpu 4 --skip-existing \
    >> "$LOG/260524_sample_reproduction_10pockets.log" 2>&1
EVAL_RC=$?
echo "[$(ts)] vina eval done (exit $EVAL_RC)  ->  $OUT"
