#!/bin/bash
# D+G-only (density+gradmag, NO coords) 40M on PLINDER with on-the-fly resample aug.
# 2-channel: input_mode=density + with_gradmag (resampled density + on-the-fly ‖∇ρ‖),
# n_in=2, no atom channels. Train -> 3-seed frozen probe on canonical 839 (density+gradmag
# from voxels_v5_realdens). Runs on GPU 0-3 AFTER the density-only + gradmag-only OTF chain.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GPUS=0,1,2,3; GATEGPU=0
EXP=260619_plinder_dg_otf_vit_mae_40m_pretrain
TAG=plinder_dg_otf
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
GMONLY_CKPT=exps/260619_plinder_gradmagonly_otf_vit_mae_40m_pretrain/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

# ── gate: run after the gradmag-only OTF finishes (ckpt gate avoids racing its GPU-0 use),
#    then wait for GPU 0-3 idle ──
echo "[$(ts)] [dg-otf] waiting for gradmag-only OTF to finish (ckpt) ..." | tee -a "$LOG"
while [ ! -f "$GMONLY_CKPT" ]; do sleep 120; done
echo "[$(ts)] [dg-otf] gradmag-only done; waiting for GPU $GATEGPU idle ..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [dg-otf] GPU free -> TRAIN $EXP (OTF, density+gradmag, n_in=2)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py --config-name=config_train_density_vit_mae_40m_xray \
  dset.data_dir="$DATA" dset.crops_dir="" dset.resample_dir="$DATA/xray_resample_plinder" \
  dset.data_file=data_train_plinder.pt dset.normalize=false \
  dset.subset_xray_only=true dset.subset_n=17430 dset.subset_val_n=100 \
  ++model.input_mode=density ++model.with_gradmag=true ++model.n_in_channels=2 ++mae.gradmag_reconstruct=true \
  num_workers=8 bsz=32 accum_steps=1 num_epochs=100 \
  "wandb_tags=[pretrain,dg_only,40m,plinder,otf,density_ablation,uniform]" \
  exp_name=$EXP output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [dg-otf] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] [dg-otf] ABORT: no e99" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] [dg-otf] PROBE -> $CSV  (density+gradmag from realdens, 839)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features \
  --condition density_gradmag --voxel_version v5 --epoch 99 --num_workers 0 \
  --exp_dir "exps/$EXP" --tag "$TAG" --noise_voxels_dir "$DATA/pdbbind/voxels_v5_realdens" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] [dg-otf] DONE -> $CSV" | tee -a "$LOG"
