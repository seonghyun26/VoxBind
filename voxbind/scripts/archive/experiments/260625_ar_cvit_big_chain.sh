#!/bin/bash
# 260625_ar_cvit_big_chain.sh — CAPACITY test. ~13 autoresearch trials (grouping,
# mask, recon-wt, lr, wd, head-d, rope3d, HCS channel-dropout) ALL tie-or-hurt the
# [7,4,2] base (ρ 0.637) → the frozen-probe plateau looks data/capacity-bound, not
# SSL-recipe-bound. So scale the WINNING recipe up on width: dim 512→768, heads 8→12
# (head_dim stays 64), depth 12 unchanged → ~2× params (~95M). The frozen affinity
# probe reads the dim-wide mean-pooled feature, so width is the most direct capacity
# lever for this eval. Everything else IDENTICAL to c1_g742: [7,4,2], mask 0.50,
# dens-wt 0.1, 100ep, eff-batch 128 (bsz4×accum8×4gpu — halved per-step batch to stay
# safe on 24GB at the larger width; compile ON since the token count is fixed).
# Single trial on GPU 0-3, then frozen probe on lp_edrscc_v2 (Kd/Ki, n=1320). Beat 0.637.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3
PORT=29551
RID=cvitBIG
EXP=260625_ar_cvit_big768

echo "===== AR ChannelViT BIG (dim768/heads12) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    model.channel_groups=[7,4,2] model.dim=768 model.heads=12 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR ChannelViT BIG COMPLETE $(date '+%m-%d %H:%M:%S') ====="
