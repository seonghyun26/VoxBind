#!/bin/bash
# Architecture ablation (Table 6) — ChA-MAEViT on PLINDER C+D+G.
# Same INPUT + DATASET as the best ViT (PLINDER C+D+G, ρ0.624): 13ch (7 lig + 4 poc
# atoms + density + on-the-fly gradmag), ligvdw, PLINDER corpus, 100ep, bsz32×4.
# Architecture = config_train_atomblob_density_gradmag_cha_mae_40m: grouped DensityViT
# (channel_group [7,4,1,1]) + 4 memory tokens + channel-aware decoder, token-drop DCP
# masking + pixel+Fourier loss (its OWN objective, no invfreq recon weighting). Then a
# 3-seed frozen affinity probe (EMA encoder slice) on the canonical 2172/480/839.
# Queued on GPU 4-7: waits until that group frees (the noise-control run finishes).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=4,5,6,7
GATEGPU=4
EXP=260617_plinder_chamae_cdg_pretrain
TAG=plinder_chamae
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

# ── gate: GPU group idle for 3 consecutive 60s checks (mem<2GB) ──
echo "[$(ts)] [chamae] waiting for GPU $GATEGPU to free (3×60s idle)..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [chamae] GPU $GATEGPU free -> TRAIN ($EXP)" | tee -a "$LOG"

# expandable_segments REQUIRED: ChA-MAE rides ~24GB/3090 and OOMs ~epoch 3 on fragmentation
# without it (verified on the v5 run). bsz=32 OK — ChA-MAE token-drops (DCP), so the encoder
# sees far fewer than the 4× channel_group tokens (unlike the all-token Channel-ViT above).
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_cha_mae.py \
  --config-name=config_train_atomblob_density_gradmag_cha_mae_40m \
  dset.data_dir="$DATA" \
  dset.crops_dir="$DATA/xray_crops_aligned_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,cha_mae,atomblob_density_gradmag,40m,plinder,channelvit,memory_tokens,dcp,ligvdw,arch_ablation]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [chamae] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] [chamae] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] [chamae] features + 3-seed probe -> $CSV" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] [chamae] DONE -> $CSV" | tee -a "$LOG"
