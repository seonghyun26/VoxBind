#!/bin/bash
# 260625 — Probe-head ablation on the 0.637 ChannelViT (autoresearch best, channel_groups
# [7,4,2], exps/260623_ar_cvit_c1_g742; canonical cached-probe mean ρ 0.637 / pearson 0.656).
# mean (ref) vs attn, frozen encoder, lp_edrscc_v2, 3 seeds, finetune --ft_scope none path.
# No memory tokens → memtok N/A. mean on GPU 4, attn on GPU 5 (both idle), in parallel.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
cd "$VOX" || exit 1
SEEDS=3
SPLIT=lp_edrscc_v2
RES=$VOX/dataset/data/pdbbind/results
LOG=$VOX/log/260625_pool_heads_g742.log
G742=exps/260623_ar_cvit_c1_g742
mkdir -p "$RES" "$VOX/log"
ts(){ date "+%F %T"; }

run(){  # gpu name pool
  local gpu=$1 name=$2 pool=$3
  local csv=$RES/poolheads_${name}.csv
  if [ -f "$csv" ]; then echo "[$(ts)] [$name] CSV exists → skip" | tee -a "$LOG"; return; fi
  echo "[$(ts)] [$name] START pool=$pool (gpu$gpu)" | tee -a "$LOG"
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu nice -19 $PY -u dataset/01c_pdbbind_probe.py finetune \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --exp_dir "$G742" --modes finetune --ft_scope none --pool "$pool" --atom_source auto \
    --split "$SPLIT" --seeds "$SEEDS" --num_workers 0 --feat_batch_size 16 \
    --out_csv "$csv" >> "$LOG" 2>&1
  echo "[$(ts)] [$name] exit=$? -> $csv" | tee -a "$LOG"
}

echo "[$(ts)] === g742 ChannelViT pool-head ablation ($SPLIT, seeds $SEEDS) ===" | tee -a "$LOG"
run 4 g742_mean mean &
run 5 g742_attn attn &
wait
echo "[$(ts)] === DONE ===" | tee -a "$LOG"
