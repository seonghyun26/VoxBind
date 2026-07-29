#!/bin/bash
# 260722 — WINNING RECIPE on the new v3 corpus (pose-diverse clean, ~73K). Recipe = the 0.644
# champion (100M d640/L18/h10 ChannelViT [7,4,2] C+D+G, mask0.75, 50ep, ema0.999, eff128) PLUS
# channel_group_dropout=0.15 (HCS) for free missing-ligand-channel robustness (ties baseline in
# every prior test). compile OFF (HCS varies token count). v3 = res≤3.0 + pocket-RSCC + density-
# aware pose dedup → clean AND pose-diverse (~2× v2.1). Tests whether clean+pose-diverse scaling
# beats the 0.644 noisy-112K champion. SUBSET_N passed as $3 (from the build manifest).
# usage: 260722_ar_cvit_100m_v3_m075_hcs.sh <GPUS csv> <PORT> <SUBSET_N>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; SUBSET_N="$3"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260722_ar_cvit_100m_v3_m075_hcs015

echo "===== WINNING RECIPE on v3: 100M · mask0.75 · ema0.999 · HCS0.15 · subset=$SUBSET_N START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV3hcs \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    model.channel_group_dropout=0.15 mae.mask_ratio=0.75 compile.enabled=false \
    dset.data_file=pretrain/data_train_plinder_v3_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v3_perelem \
    dset.subset_n=$SUBSET_N dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (r / rho / RMSE) ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== v3 winning-recipe COMPLETE $(date '+%m-%d %H:%M:%S') ====="
