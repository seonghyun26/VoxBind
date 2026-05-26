#!/bin/bash
# 30_sample_eval_electra_fulltest.sh — full test-set sample + Vina dock eval
# from the 260525_voxbind_10k_density_vit_electra_frozen final checkpoint.
#
# Mirror of 26_sample_voxbind_vit_ep379_fulltest.sh, with the model swapped to
# the ELECTRA-pretrained-encoder downstream run.
#
# 100 test pockets in crossdocked_xray; 79 have x-ray density and will be
# sampled (the density-conditioned model skips the 21 without).
# n_targets=200 and end=200 are deliberately above 100 — the loop iterates
# the full test split (pockets without xray are skipped via `continue` inside
# sample.py, but still count toward pocket_id, so the breaks must trigger on
# pocket_id, not processed count).
#
# GPU 6 by default (1-5 may still be busy / cooling from training).
# Approx 5 min WJS per pocket × ~80 sampled ≈ 6.5h, then ~10 min for Vina
# dock eval with workers=16.
#
# Out: exps/260525_voxbind_10k_density_vit_electra_frozen/samples/res_electra_fulltest/
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260525_voxbind_10k_density_vit_electra_frozen
GPU=${GPU:-6}
OUT=$VOX/exps/$EXP/samples/res_electra_fulltest
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
    out_dir=res_electra_fulltest \
    save_dir=$OUT \
    > "$LOG/260526_sample_electra_fulltest.log" 2>&1
SAMPLE_RC=$?
echo "[$(ts)] WJS sample done (exit $SAMPLE_RC)"
[ $SAMPLE_RC -ne 0 ] && exit $SAMPLE_RC

# Phase 2: Vina dock eval (CPU, workers=16)
echo "[$(ts)] starting Vina dock eval (workers=16 cpu=4)"
$PY $VOX/../notebook/webapp/metrics.py "$OUT" \
    --docking vina_dock --workers 16 --cpu 4 --skip-existing \
    >> "$LOG/260526_sample_electra_fulltest.log" 2>&1
EVAL_RC=$?
echo "[$(ts)] vina eval done (exit $EVAL_RC)  ->  $OUT"
