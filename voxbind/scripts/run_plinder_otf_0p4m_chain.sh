#!/bin/bash
# 0.4M PLINDER C+D+G with on-the-fly (resample) augmentation — param-scaling point.
# Same corpus/recipe as the 40M PLINDER OTF (p100): inv-freq, ligvdW, full PLINDER
# corpus (subset_n=17430), OTF density+gradmag via dset.resample_dir. ONLY the model
# width shrinks to ~0.4M (dim32/depth6/heads4/head_hidden10). Then 3-seed frozen
# affinity probe on the canonical 2172/480/839. GPU 4-7.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }

GPUS=4,5,6,7
EXP=260618_atomblob_density_gradmag_vit_mae_0p4m_plinder_otf_pretrain
TAG=plinder_otf_0p4m
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

echo "[$(ts)] === TRAIN $EXP (GPU $GPUS, OTF resample) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py \
  --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  dset.data_dir="$DATA" \
  dset.crops_dir="" \
  dset.resample_dir="$DATA/xray_resample_plinder" \
  dset.data_file=data_train_plinder.pt \
  dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
  model.model_name=atomblob_density_vit_mae_0p4m \
  model.dim=32 model.depth=6 model.heads=4 model.head_hidden_dim=10 \
  num_workers=8 num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,atomblob_density_gradmag,0p4m,invfreq,plinder,otf,ligvdw,paramscale]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] train exit=$?" | tee -a "$LOG"
[ -f "$CKPT" ] || { echo "[$(ts)] ABORT: no e99 checkpoint" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] === PROBE (features + 3-seed) -> $CSV ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
  --atom_source ligvdw --num_workers 0 --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=4 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
echo "[$(ts)] === DONE 0.4M PLINDER OTF -> $CSV ===" | tee -a "$LOG"
