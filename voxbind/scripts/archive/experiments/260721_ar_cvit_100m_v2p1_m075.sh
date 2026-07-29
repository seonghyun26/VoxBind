#!/bin/bash
# 260721 — CHAMPION RECIPE on the CLEAN corpus. The two positive levers (mask0.75 = +0.030 on
# 100M/v2; clean data) have NEVER been stacked. This = the exact 0.644 champion (100M d640/L18/h10
# ChannelViT [7,4,2] C+D+G, mask0.75, 50ep, ema0.999, eff128) EXCEPT corpus = PLINDER v2.1-clean
# (37.4K: dedup+res≤2.5+ligand&pocket-RSCC≥0.8) instead of v2-noisy-112K. Beats 0.644 → clean
# data + mask0.75 stack; ties/below → clean data doesn't help even the champion recipe.
# usage: 260721_ar_cvit_100m_v2p1_m075.sh <GPUS csv> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49; SUBSET_N=37401
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260721_ar_cvit_100m_v2p1_m075

echo "===== CHAMPION-on-CLEAN: 100M · v2.1 · mask0.75 · 50ep · ema0.999 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV2p1m075 \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 \
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
echo "###### [$EXP] result (Pearson r / Spearman rho / RMSE) ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== CHAMPION-on-CLEAN COMPLETE $(date '+%m-%d %H:%M:%S') ====="
