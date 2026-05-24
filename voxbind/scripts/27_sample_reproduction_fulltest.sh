#!/bin/bash
# 27_sample_reproduction_fulltest.sh — full test-set sample from the
# reproduction baseline (exps/reproduction, no-density VoxBind ep86).
#
# Apples-to-apples vs scripts/26_*.sh (ep379 ViT-MAE-frozen on the same test
# split). Reproduction model has no density branch, so it samples ALL 100 test
# pockets (vs ViT's 79 xray-only); compare on the 79-pocket intersection.
#
# GPU 6 (parallel with the ep379 fulltest on GPU 7).
#
# Out: exps/reproduction/samples/res_repro_fulltest/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=reproduction
GPU=${GPU:-6}
OUT=$VOX/exps/$EXP/samples/res_repro_fulltest
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling full test set from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

# Phase 1: WJS sample (GPU). Same dset config as 26_*.sh so pocket enumeration
# matches. Reproduction model has with_density=False so it processes every
# pocket regardless of xray availability — gives broader coverage but the
# fair comparison is the 79-pocket xray intersection with the ViT run.
CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=test wjs.n_targets=200 wjs.start=0 wjs.end=200 wjs.n_samples_per_pocket=10 \
    out_dir=res_repro_fulltest \
    save_dir=$OUT \
    > "$LOG/260524_sample_reproduction_fulltest.log" 2>&1
SAMPLE_RC=$?
echo "[$(ts)] WJS sample done (exit $SAMPLE_RC)"
[ $SAMPLE_RC -ne 0 ] && exit $SAMPLE_RC

# Phase 2: Vina dock eval (CPU, workers=16 cpu=4)
echo "[$(ts)] starting Vina dock eval (workers=16 cpu=4)"
$PY $VOX/../notebook/webapp/metrics.py "$OUT" \
    --docking vina_dock --workers 16 --cpu 4 --skip-existing \
    >> "$LOG/260524_sample_reproduction_fulltest.log" 2>&1
EVAL_RC=$?
echo "[$(ts)] vina eval done (exit $EVAL_RC)  ->  $OUT"
