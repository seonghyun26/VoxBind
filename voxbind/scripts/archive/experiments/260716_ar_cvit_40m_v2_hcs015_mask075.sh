#!/bin/bash
# 260716 FOLLOW-UP (fires only if the mask-0.5 v2 run underperforms v1 by >0.02 rho).
# Same 40M ChannelViT [7,4,2] C+D+G + HCS p=0.15 on PLINDER v2, but flips the two levers
# the v2 campaign already showed matter:
#   (1) mask 0.75  — the ONE proven v2 win (+0.030 on 100M v2; 0.656 recipe was mask 0.5).
#   (2) ckpt_every=10 — save e10/20/30/40/49 so the probe can find the PEAK epoch instead of
#       assuming ep49 is best (v2/100M peaked before 50; 6.4× data may overfit the pretext).
# compile OFF (HCS varies token count). eff-batch 128 (bsz4×accum8×4gpu), nw6/pf2 host-safe.
# usage: 260716_ar_cvit_40m_v2_hcs015_mask075.sh <GPUS> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS="$1"; PORT="$2"
EXP=260716_ar_cvit_40m_v2_hcs015_mask075

echo "===== HCS p=0.15 · 40M · mask0.75 · v2 · ckptE10 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit40mV2HCSm075 \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 \
    model.channel_groups=[7,4,2] model.channel_group_dropout=0.15 \
    mae.mask_ratio=0.75 mae.ckpt_every=10 compile.enabled=false \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] EPOCH-PEAK PROBE start $(date '+%H:%M:%S') ######"
for E in 9 19 29 39 49; do
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --epoch $E --gpu "${GPUS%%,*}" --tag "${EXP}_e${E}" \
      --num_workers 0 -- --require_density \
    && echo "[$EXP] PROBE e$E OK" || echo "[$EXP] PROBE e$E FAILED"
done
echo "###### [$EXP] results (per epoch) ######"
for E in 9 19 29 39 49; do
  echo "-- e$E --"; tail -1 "$RES/probe_results_e${E}_v5_lp_edrscc_v2split_${EXP}_e${E}.csv" 2>/dev/null
done
echo "===== 40M v2 HCS0.15 mask075 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
