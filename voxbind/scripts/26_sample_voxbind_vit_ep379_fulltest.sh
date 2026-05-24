#!/bin/bash
# 26_sample_voxbind_vit_ep379_fulltest.sh — full test-set sample from the
# 260523_voxbind_10k_density_vit_mae_frozen ep379 snapshot.
#
# 100 test pockets total in crossdocked_xray; 79 have x-ray density and will be
# sampled (the density-conditioned model skips the 21 without). 10 ligands per
# pocket = ~790 ligands total.
#
# n_targets=200 and end=200 are intentionally above 100 so the loop iterates
# the full test split (pockets without xray are skipped via `continue` inside
# sample.py, but still count toward pocket_id — the break checks fire on
# pocket_id, not processed count).
#
# GPU 7 only. Approx 5 min WJS per pocket × ~80 sampled ≈ 6.5h, then ~10 min
# for Vina dock eval with workers=16.
#
# Out: exps/260523_voxbind_10k_density_vit_mae_frozen_e0379_snap/samples/res_ep379_fulltest/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260523_voxbind_10k_density_vit_mae_frozen_e0379_snap
GPU=${GPU:-7}
OUT=$VOX/exps/$EXP/samples/res_ep379_fulltest
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] sampling full test set from $EXP on GPU $GPU"
echo "[$(ts)]   out=$OUT"

# Phase 1: WJS sample (GPU)
CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    wjs.split=test wjs.n_targets=200 wjs.start=0 wjs.end=200 wjs.n_samples_per_pocket=10 \
    out_dir=res_ep379_fulltest \
    save_dir=$OUT \
    > "$LOG/260524_sample_voxbind_vit_ep379_fulltest.log" 2>&1
SAMPLE_RC=$?
echo "[$(ts)] WJS sample done (exit $SAMPLE_RC)"
[ $SAMPLE_RC -ne 0 ] && exit $SAMPLE_RC

# Phase 2: Vina dock eval (CPU, workers=16)
echo "[$(ts)] starting Vina dock eval (workers=16 cpu=4)"
$PY $VOX/../notebook/webapp/metrics.py "$OUT" \
    --docking vina_dock --workers 16 --cpu 4 --skip-existing \
    >> "$LOG/260524_sample_voxbind_vit_ep379_fulltest.log" 2>&1
EVAL_RC=$?
echo "[$(ts)] vina eval done (exit $EVAL_RC)  ->  $OUT"
