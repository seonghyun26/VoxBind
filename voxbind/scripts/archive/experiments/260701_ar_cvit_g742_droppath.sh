#!/bin/bash
# 260701_ar_cvit_g742_droppath.sh — CV/NLP trick A: STOCHASTIC DEPTH (DropPath, DeiT/MAE).
# Controlled change off the ChannelViT [7,4,2] winner (ρ 0.637): the ONLY thing changed is
# +model.drop_path=0.1 — per-sample drop whole residual branches in training, rate ramped 0→0.1
# over depth, eval bit-exact (no params, frozen probe unchanged). Directly targets the campaign's
# one robust finding (capacity/overfitting hurts the frozen probe) with the canonical ViT regularizer.
# Opt-in, original code bit-identical at drop_path=0. GPU 0-3, bsz8×accum4 eff128, compile OFF. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29621; RID=arG742DP
EXP=260701_ar_cvit_g742_droppath

echo "===== AR [7,4,2] DROPPATH=0.1 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 compile.enabled=false \
    model.channel_groups=[7,4,2] +model.drop_path=0.1 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] DROPPATH COMPLETE $(date '+%m-%d %H:%M:%S') ====="
