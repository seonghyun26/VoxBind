#!/bin/bash
# 260721 — SAME-RECIPE SIZE LADDER on the clean v2.1 pool. Answers "does more data help?" cleanly
# by holding the recipe FIXED and varying ONLY size (subsample the same v2.1 crops). v1-vs-v2 was
# confounded (dedup pdb vs pdb_ccd, single-lig, splits, in-vocab ALL differed + size). Here: 40M
# ChannelViT [7,4,2] mask0.5 ema0.9999 100ep — IDENTICAL to the v2.1 ema0.9999 run — at subset_n
# {9000, 18000, 37401=full(already have)}. Monotonic rise → data helps; flat → saturation is real.
# eff-batch 128 held constant (bsz8 × accum = 128/(8·NG)). subset_n keeps first N of the shuffled
# manifest (nested subsets). usage: sizeladder.sh <GPUS csv> <PORT> <SUBSET_N>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; SUBSET_N="$3"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=100; PROBE_EP=99
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
ACCUM=$((128 / (8 * NG)))
EXP=260721_ar_cvit_40m_v2p1_ema9999_n${SUBSET_N}

echo "===== SIZE-LADDER n=$SUBSET_N · 40M v2.1 · mask0.5 · ema0.9999 · 100ep START $(date '+%m-%d %H:%M:%S') GPU=$GPUS (eff=$((8*ACCUM*NG))) ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit40mLadder$SUBSET_N \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=$ACCUM \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] mae.ema_decay=0.9999 \
    dset.data_file=pretrain/data_train_plinder_v2p1_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2p1_perelem \
    dset.subset_n=$SUBSET_N dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (Pearson r / Spearman rho / RMSE in cols) ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== SIZE-LADDER n=$SUBSET_N COMPLETE $(date '+%m-%d %H:%M:%S') ====="
