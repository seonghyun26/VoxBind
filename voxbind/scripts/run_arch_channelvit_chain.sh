#!/bin/bash
# Architecture ablation (Table 6) — Channel-ViT on PLINDER C+D+G.
# Same INPUT + DATASET + recipe as the best ViT (PLINDER C+D+G invfreq, ρ0.624):
# 13ch (7 lig + 4 poc atoms + density + on-the-fly gradmag), ligvdw, invfreq, 100ep,
# bsz32×4. ONLY the patch embed changes: fused -> channel_group groups [7,4,1,1]
# (pos_encoding=learnable, required by channel_group; the best ViT also used learnable,
# so no PE confound). Then a 3-seed frozen affinity probe on the canonical 2172/480/839.
# Queued on GPU 0-3: waits until that group frees (PLINDER OTF size-sweep finishes).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=0,1,2,3
GATEGPU=0
EXP=260617_plinder_channelvit_cdg_invfreq_pretrain
TAG=plinder_channelvit
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

# ── gate: GPU group idle for 3 consecutive 60s checks (mem<2GB; survives transient probe bursts) ──
echo "[$(ts)] [channelvit] waiting for GPU $GATEGPU to free (3×60s idle)..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [channelvit] GPU $GATEGPU free -> TRAIN ($EXP)" | tee -a "$LOG"

# channel_group = 4× tokens (no token-drop) → bsz=8 (matches fused bsz=32 token-throughput,
# eff batch = 8×4×4 = 128, same as the fused ViT); expandable_segments to dodge fragmentation OOM.
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py \
  --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  dset.data_dir="$DATA" \
  dset.crops_dir="$DATA/xray_crops_aligned_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  model.patch_embed_mode=channel_group \
  'model.channel_groups=[7,4,1,1]' \
  num_epochs=100 bsz=8 accum_steps=4 \
  wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,channelvit,channel_group,ligvdw,clip30,arch_ablation]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [channelvit] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] [channelvit] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] [channelvit] features + 3-seed probe -> $CSV" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] [channelvit] DONE -> $CSV" | tee -a "$LOG"
