#!/bin/bash
# 260628_ar_g742_finetune_full.sh — AUTORESEARCH LOOP iter 6 (GPU 0). NEW AXIS: eval protocol.
# Campaign concluded the FROZEN-probe plateaus at ρ≈0.63 across all config-overridable axes.
# Open question: is 0.63 just the FROZEN-probe ceiling? Test end-to-end finetuning of the
# [7,4,2] winner (260623_ar_cvit_c1_g742, e99) on lp_edrscc_v2. --modes frozen finetune runs
# BOTH arms in one job (frozen head-only ≈0.637 baseline + end-to-end), --ft_scope full
# unfreezes patch_embed+pos+all blocks+norm (encoder_lr 1e-5, head_lr 1e-3). 3 seeds (full-FT
# is known high-variance → seeds matter, per the seed-noise finding). Single-GPU subcommand.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin/python
RES=dataset/data/pdbbind/results
EXP_DIR=exps/260623_ar_cvit_c1_g742
OUT=$RES/ft_g742_full.csv

echo "===== AR [7,4,2] FINETUNE (frozen vs end-to-end full) START $(date '+%m-%d %H:%M:%S') GPU=0 ====="
CUDA_VISIBLE_DEVICES=0 "$PY" -u dataset/01c_pdbbind_probe.py finetune \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --exp_dir "$EXP_DIR" --modes frozen finetune --ft_scope full \
    --pool mean --atom_source auto --split lp_edrscc_v2 --seeds 3 \
    --num_workers 0 --feat_batch_size 8 --out_csv "$OUT" \
  && echo "[ft_g742_full] FINETUNE OK $(date '+%H:%M:%S')" \
  || { echo "[ft_g742_full] FINETUNE FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [ft_g742_full] result ######"
tail -8 "$OUT" 2>/dev/null
echo "===== AR [7,4,2] FINETUNE COMPLETE $(date '+%m-%d %H:%M:%S') ====="
