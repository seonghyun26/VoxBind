#!/bin/bash
# 260701_ar_cvit_big768_plinder_v2.sh — does bigger DATA unlock the big model?
# The dim768 (~98M) ChannelViT [7,4,2] encoder REGRESSED on PLINDER v1 (17.5K):
# ρ 0.626 < 40M's 0.637 → capacity looked wasted at 17.5K pretrain data. Retrain the
# IDENTICAL recipe on PLINDER v2 per-element (~112K crops, 6.4× v1) to test whether the
# extra data lets the width pay off. Everything else identical to 260625_ar_cvit_big768:
# [7,4,2], mask 0.50, dens-wt 0.1, 100 epochs, eff-batch 128 (bsz4×accum8×4gpu). Only the
# dataset swaps (v1 → v2 per-element) + subset_n raised. GPU 4-7, then frozen probe.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7
PORT=29561
RID=cvitBIGv2
EXP=260701_ar_cvit_big768_plinder_v2

echo "===== AR ChannelViT BIG768 · PLINDER v2 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    model.channel_groups=[7,4,2] model.dim=768 model.heads=12 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR ChannelViT BIG768 · PLINDER v2 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
