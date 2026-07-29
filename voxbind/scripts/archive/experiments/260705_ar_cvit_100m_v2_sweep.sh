#!/bin/bash
# 260705_ar_cvit_100m_v2_sweep.sh — BRIEF autoresearch: does a bigger model + bigger data
# want a different masking ratio? New BALANCED ~100M encoder (dim640/depth18/heads10 — expands
# BOTH width 512->640 and depth 12->24->18 vs base) on PLINDER v2 112K, C+D+G ChannelViT [7,4,2].
# Sweeps mae.mask_ratio at a reduced 50-epoch budget (probe e49) for speed; winner gets full 100ep later.
# Memory-safe for CONCURRENT 2-group runs (num_workers 6 + prefetch 2, bsz4/accum8 eff128) — the
# earlier bsz8/nw16 concurrent run OOM-killed the host.
# usage: sweep.sh <GPUS csv> <PORT> <MASK>   e.g. sweep.sh 0,1,2,3 29581 0.50
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; MASK="$3"
MTAG=$(echo "$MASK" | tr -d '.')
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
RID=cvit100mv2m$MTAG
EXP=260705_ar_cvit_100m_v2_mask$MTAG
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "===== 100M (d640/L18) v2 · mask=$MASK START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=$MASK \
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
echo "===== 100M v2 · mask=$MASK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
