#!/bin/bash
# 260717 — "is the ~0.63/0.64 plateau an artifact of constant-LR undertraining at scale?"
# Takes the current v2 BEST recipe (100M d640/L18/h10 ChannelViT [7,4,2] C+D+G, mask 0.75,
# ρ 0.644 @ 50ep) and swaps the fixed-LR optimizer for the standard big-model package:
#   (1) lr_schedule=cosine + lr_warmup_epochs=5  — linear warmup 0→peak, cosine decay→0.
#   (2) eff-batch 256 (bsz4×accum16×4gpu) — 2× the campaign-standard 128; big models want it.
#   (3) peak LR 2e-4 — linear-scaled with the 2× batch (base was constant 1e-4 @ eff128).
# Everything else identical to 260705_ar_cvit_100m_v2_mask075 so the schedule+batch package
# is the ONLY change vs the 0.644 anchor. compile ON (mask corrupts voxels, token count fixed).
# num_workers6/prefetch2 = host-RAM-safe. 50ep, probe e49. All 4 GPUs.
# usage: 260717_ar_cvit_100m_v2_schedule.sh <GPUS csv> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260717_ar_cvit_100m_v2_m075_cosine_bs256

echo "===== 100M v2 · mask0.75 · cosine+warmup5 · eff256 · lr2e-4 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit100mV2sched \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=16 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    lr=2e-4 +lr_schedule=cosine +lr_warmup_epochs=5 +lr_min=0.0 \
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
echo "===== 100M v2 schedule COMPLETE $(date '+%m-%d %H:%M:%S') ====="
