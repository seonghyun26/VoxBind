#!/bin/bash
# 260624 — Probe-head ablation: mean (ref) vs attn vs memtok pooling on the frozen
# encoder, lp_edrscc_v2 (3850/817/1320), 3 seeds. finetune --ft_scope none path
# (encode tokens ONCE → train a learnable pool+head; one fwd pass/cell).
#
#   ChA-MAE (260622_..._chamae_mask050, 4 memory tokens) : mean, attn, memtok
#   plain ViT (260622_..._cdg_mask050, headline)         : mean, attn
#
# Parallelised across GPUs 4/6/7 (one heavy ChA-MAE cell per GPU, ViT cells appended).
# Skips any cell whose CSV already exists. Known mean baselines: ChA-MAE 0.578, ViT 0.583.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
cd "$VOX" || exit 1
SEEDS=3
SPLIT=lp_edrscc_v2
RES=$VOX/dataset/data/pdbbind/results
LOG=$VOX/log/260624_pool_heads.log
mkdir -p "$RES" "$VOX/log"
ts(){ date "+%F %T"; }

CHAMAE=exps/260622_plinder_otf_cdg_chamae_mask050_pretrain
VIT=exps/260622_plinder_otf_cdg_mask050_pretrain

run(){  # gpu name exp_dir pool feat_bsz
  local gpu=$1 name=$2 exp=$3 pool=$4 fbs=$5
  local csv=$RES/poolheads_${name}.csv
  if [ -f "$csv" ]; then echo "[$(ts)] [gpu$gpu][$name] CSV exists → skip" | tee -a "$LOG"; return; fi
  echo "[$(ts)] [gpu$gpu][$name] START pool=$pool" | tee -a "$LOG"
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu nice -19 $PY -u dataset/01c_pdbbind_probe.py finetune \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --exp_dir "$exp" --modes finetune --ft_scope none --pool "$pool" --atom_source auto \
    --split "$SPLIT" --seeds "$SEEDS" --num_workers 0 --feat_batch_size "$fbs" \
    --out_csv "$csv" >> "$LOG" 2>&1
  echo "[$(ts)] [gpu$gpu][$name] exit=$? -> $csv" | tee -a "$LOG"
}

# one ChA-MAE (heavy, channel-group) cell per GPU; ViT (light, fused) cells appended.
queue4(){ run 4 chamae_mean   "$CHAMAE" mean   16; run 4 vit_mean "$VIT" mean 32; }
queue6(){ run 6 chamae_attn   "$CHAMAE" attn   16; run 6 vit_attn "$VIT" attn 32; }
queue7(){ run 7 chamae_memtok "$CHAMAE" memtok 16; }

echo "[$(ts)] === pool-head ablation (GPUs 4/6/7, split $SPLIT, seeds $SEEDS) ===" | tee -a "$LOG"
queue4 & queue6 & queue7 &
wait
echo "[$(ts)] === ALL DONE ===" | tee -a "$LOG"
