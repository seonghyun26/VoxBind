#!/bin/bash
# 260719 — EXTEND the EMA-decay win to the bigger-data champion. v2.1 40M gave ema0.999→0.9999
# = +0.011 ρ (0.619→0.630, first positive in the scaling sweep). The EMA-window-shrink argument
# is STRONGER at larger data (v2 112K = 875 steps/ep → the 0.999 window is only ~1 epoch), so
# EMA 0.9999 should help MORE here. IDENTICAL to the 0.644 anchor (260705 100M d640/L18/h10,
# mask0.75, v2-112K, 50ep, eff128) EXCEPT mae.ema_decay 0.999→0.9999. If it lifts 0.644, real break.
# usage: 260719_ar_cvit_100m_v2_m075_ema9999.sh <GPUS csv> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260719_ar_cvit_100m_v2_m075_ema9999

echo "===== 100M v2 · mask0.75 · ema0.9999 · 50ep START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV2ema \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 mae.ema_decay=0.9999 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== 100M v2 ema0.9999 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
