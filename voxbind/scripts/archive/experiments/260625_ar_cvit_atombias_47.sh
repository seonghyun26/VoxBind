#!/bin/bash
# 260625_ar_cvit_atombias_47.sh — top of the reordered knob queue, run on GPU 4-7 in
# parallel with the capacity test (big768) on GPU 0-3. atom_biased masking concentrates
# the MAE block-mask on atom/density-occupied voxels (chemistry-rich regions) instead of
# random blocks — the single most distinct SSL lever left, untested on the [7,4,2] base.
# Standard base recipe (46.71M, dim512): [7,4,2], mask 0.50, dw 0.1, 100ep, eff-batch 128
# (bsz8×accum4×4gpu), compile ON (fixed token count). Frozen probe → lp_edrscc_v2. Beat 0.637.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7
PORT=29561
RID=cvitAB
EXP=260624_ar_cvit_atombias

echo "===== AR ChannelViT atom-bias START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[7,4,2] mae.mask_strategy=atom_biased \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR ChannelViT atom-bias COMPLETE $(date '+%m-%d %H:%M:%S') ====="
