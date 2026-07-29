#!/bin/bash
# 260718 — ISOLATE the LR schedule from the batch confound. The 260717 package (cosine +
# 2× batch + 2× LR) landed ρ0.632 < 0.644 anchor, but the 2× batch halved optimizer steps
# → could mask a schedule gain. This run adds ONLY the cosine+warmup SHAPE, everything else
# identical to the 0.644 anchor (260705 mask0.75): eff-batch 128 (bsz4×accum8), peak LR 1e-4
# (== anchor's constant value, so same peak, just warmup-up + cosine-down), 50ep, probe e49.
# If ρ ≈ 0.644 → scheduling itself is inert (plateau real). If ρ > 0.644 → the 260717 batch
# increase was the culprit, schedule helps.
# usage: 260718_ar_cvit_100m_v2_cosine_only.sh <GPUS csv> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260718_ar_cvit_100m_v2_m075_cosine_bs128

echo "===== 100M v2 · mask0.75 · cosine+warmup5 · eff128 · lr1e-4 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV2cosonly \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    lr=1e-4 +lr_schedule=cosine +lr_warmup_epochs=5 +lr_min=0.0 \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 \
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
echo "===== 100M v2 cosine-only COMPLETE $(date '+%m-%d %H:%M:%S') ====="
