#!/bin/bash
# 260625 — Probe-head ablation, ChannelViT (best encoder). mean (ref) vs attn on the
# frozen encoder, lp_edrscc_v2 (3850/817/1320), 3 seeds, finetune --ft_scope none path.
# ChannelViT = 260622_..._channelvit_mask050 (channel_group [7,4,1,1], NO memory tokens
# → memtok N/A). Canonical cached-probe mean: test ρ 0.630 / pearson 0.645.
# Coexists on GPU 6 (per user; 4-7 busy with other runs). Skips cells whose CSV exists.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
cd "$VOX" || exit 1
GPU=6
SEEDS=3
SPLIT=lp_edrscc_v2
RES=$VOX/dataset/data/pdbbind/results
LOG=$VOX/log/260625_pool_heads_channelvit.log
CVIT=exps/260622_plinder_otf_cdg_channelvit_mask050_pretrain
mkdir -p "$RES" "$VOX/log"
ts(){ date "+%F %T"; }

run(){  # name pool
  local name=$1 pool=$2
  local csv=$RES/poolheads_${name}.csv
  if [ -f "$csv" ]; then echo "[$(ts)] [$name] CSV exists → skip" | tee -a "$LOG"; return; fi
  echo "[$(ts)] [$name] START pool=$pool (gpu$GPU)" | tee -a "$LOG"
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$GPU nice -19 $PY -u dataset/01c_pdbbind_probe.py finetune \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --exp_dir "$CVIT" --modes finetune --ft_scope none --pool "$pool" --atom_source auto \
    --split "$SPLIT" --seeds "$SEEDS" --num_workers 0 --feat_batch_size 8 \
    --out_csv "$csv" >> "$LOG" 2>&1
  echo "[$(ts)] [$name] exit=$? -> $csv" | tee -a "$LOG"
}

echo "[$(ts)] === ChannelViT pool-head ablation (gpu$GPU, $SPLIT, seeds $SEEDS) ===" | tee -a "$LOG"
run channelvit_mean mean
run channelvit_attn attn
echo "[$(ts)] === DONE ===" | tee -a "$LOG"
