#!/bin/bash
# Controlled test: 40M PLINDER C+D+G OTF with the rotated-out CORNERS ZEROED
# (VOXBIND_RESAMPLE_BOXMASK=1) — replicates the frozen-crop zero-fill so OTF is
# pocket-focused AND its train-time boundary matches the probe's voxels_v5 (frozen
# crops, zero corners). Tests whether that recovers the OTF deficit
# (OTF 0.600 vs frozen 0.624). Everything else identical to the p100 40M OTF.
# Runs on GPU 4-7 AFTER the 0.4M + 4M param sweep finishes (no CPU contention).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=4,5,6,7
GATEGPU=4
EXP=260618_atomblob_density_gradmag_vit_mae_40m_plinder_otf_boxmask_pretrain
TAG=plinder_otf_boxmask
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
FOURM_CKPT=exps/260618_atomblob_density_gradmag_vit_mae_4m_plinder_otf_pretrain/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

# ── gate: run strictly AFTER the 4M (checkpoint gate avoids racing the 4M's own GPU-4 gate),
#    then wait for GPU 4-7 idle (4M probe + 0.4M all done) ──
echo "[$(ts)] [boxmask] waiting for 4M to finish training (ckpt) ..." | tee -a "$LOG"
while [ ! -f "$FOURM_CKPT" ]; do sleep 120; done
echo "[$(ts)] [boxmask] 4M trained; waiting for GPU $GATEGPU idle (4M probe done) ..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [boxmask] GPU free -> TRAIN $EXP (OTF + corner zero-mask)" | tee -a "$LOG"

VOXBIND_RESAMPLE_BOXMASK=1 CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py \
  --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  dset.data_dir="$DATA" \
  dset.crops_dir="" \
  dset.resample_dir="$DATA/xray_resample_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,otf,boxmask,ligvdw,paramscale]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [boxmask] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] [boxmask] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] [boxmask] === PROBE (features + 3-seed) -> $CSV ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --num_workers 0 --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] [boxmask] === DONE -> $CSV (compare vs OTF 0.600 / frozen 0.624) ===" | tee -a "$LOG"
