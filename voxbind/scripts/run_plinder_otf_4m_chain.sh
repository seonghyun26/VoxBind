#!/bin/bash
# 4M PLINDER C+D+G with on-the-fly (resample) augmentation — param-scaling point.
# Same corpus/recipe/OTF as the 0.4M run; only the model width changes to ~4M
# (dim128/depth8/heads4/head_hidden24). Runs on GPU 4,5 (2 GPUs, lighter on CPU);
# waits for the 0.4M run to free GPU 4, then trains + 3-seed frozen affinity probe
# (canonical 839). 2 GPUs × bsz32 × accum2 = effective batch 128 (same as 0.4M).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=4,5
GATEGPU=4
EXP=260618_atomblob_density_gradmag_vit_mae_4m_plinder_otf_pretrain
TAG=plinder_otf_4m
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

# ── gate: wait until GPU 4 idle for 3 consecutive 60s checks (0.4M run + its probe done) ──
echo "[$(ts)] [4m] waiting for GPU $GATEGPU to free (after 0.4M)..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [4m] GPU free -> TRAIN $EXP (OTF resample)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=2 \
  train_density_vit_mae.py \
  --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  dset.data_dir="$DATA" \
  dset.crops_dir="" \
  dset.resample_dir="$DATA/xray_resample_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  model.model_name=atomblob_density_vit_mae_4m \
  model.dim=128 model.depth=8 model.heads=4 model.head_hidden_dim=24 \
  num_workers=8 num_epochs=100 bsz=32 accum_steps=2 \
  wandb_tags="[pretrain,atomblob_density_gradmag,4m,invfreq,plinder,otf,ligvdw,paramscale]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [4m] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] [4m] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] [4m] === PROBE (features + 3-seed) -> $CSV ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --num_workers 0 --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] [4m] === DONE 4M PLINDER OTF -> $CSV ===" | tee -a "$LOG"
