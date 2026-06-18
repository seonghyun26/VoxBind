#!/bin/bash
# Variance check: re-run the 40M PLINDER C+D+G OTF (identical config to p100, plain
# mode="wrap", NO boxmask) with a fresh seed (0) to see whether the p100 result
# (test ρ 0.600) reproduces or was an unlucky single draw. GPU 0-3 (free now).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=0,1,2,3
EXP=260618_atomblob_density_gradmag_vit_mae_40m_plinder_otf_seed0_pretrain
TAG=plinder_otf_seed0
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

echo "[$(ts)] === TRAIN $EXP (GPU $GPUS, OTF resample, seed 0, no boxmask) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py \
  --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  dset.data_dir="$DATA" \
  dset.crops_dir="" \
  dset.resample_dir="$DATA/xray_resample_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  seed=0 num_workers=8 num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,otf,ligvdw,seed0,variance]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] === PROBE (features + 3-seed) -> $CSV ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --num_workers 0 --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] === DONE 40M OTF seed0 -> $CSV (compare vs p100 seed42 = 0.600) ===" | tee -a "$LOG"
