#!/bin/bash
# 260720 — DECISIVE EMA×epochs test on the 100M champion. The 50ep champion (ema0.999)=0.644;
# ema0.9999 HURT it at 50ep (0.631) because a slow EMA lags at short budgets. But ema0.9999
# HELPED the 40M v2.1 run at 100ep (+0.014). So the untested cell = the 100M champion at 100ep,
# where the slow EMA can converge. Run both EMA values at 100ep (parameterized) to attribute
# epochs-vs-EMA. IDENTICAL to 260705 100M (d640/L18/h10, mask0.75, v2-112K, eff128) except
# num_epochs=100 + mae.ema_decay=$EMA. Beats 0.644 → EMA+long breaks the plateau on the big model.
# usage: 260720_ar_cvit_100m_v2_m075_ep100.sh <GPUS csv> <PORT> <EMA>   e.g. ... 0,1,2,3 29587 0.9999
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; EMA="$3"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=100; PROBE_EP=99
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
ETAG=$(echo "$EMA" | tr -d '.')
EXP=260720_ar_cvit_100m_v2_m075_ep100_ema${ETAG}

echo "===== 100M v2 · mask0.75 · ema$EMA · 100ep START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV2ep100e$ETAG \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 mae.ema_decay=$EMA \
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
echo "===== 100M v2 ema$EMA 100ep COMPLETE $(date '+%m-%d %H:%M:%S') ====="
