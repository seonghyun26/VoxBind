#!/bin/bash
# Architecture ablation + coords baseline on the PLINDER ON-THE-FLY (resample) C+D+G line.
# Fires after the dg-OTF run reaches e99, then runs sequentially on GPU 0-3 (each gated on
# GPU-idle). Same OTF input/dataset/recipe as plinder_otf_p100 (the C+D+G OTF, ρ~0.60); ONLY
# the architecture changes:
#   1) Channel-ViT  — patch_embed_mode=channel_group [7,4,1,1] (bsz8×accum4, expandable_segments)
#   2) ChA-MAE-ViT  — train_density_cha_mae.py (token-drop MAE; bsz32, expandable_segments)
#   3) Coords-only  — atoms-only (11ch) OTF, SEED 0 (matched baseline; "same config except coords-only")
# Each: train 100ep -> 3-seed frozen affinity probe on the canonical 2172/480/839.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GPUS=0,1,2,3; GATEGPU=0
MASTER=$VOX/log/260620_plinder_otf_arch_coords_chain.log
DG_CKPT=exps/260619_plinder_dg_otf_vit_mae_40m_pretrain/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

gate_gpu(){  # block until GPU $GATEGPU idle for 3 consecutive 60s checks
  local free=0 m
  while [ "$free" -lt 3 ]; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
    if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
    sleep 60
  done
}

# OTF dataset overrides shared by all three (proven by the seed0 OTF C+D+G run).
OTF_DSET=( dset.data_dir="$DATA" dset.crops_dir="" dset.resample_dir="$DATA/xray_resample_plinder"
           dset.data_file=data_train_plinder.pt dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true )

probe(){  # $1=exp  $2=tag  $3=condition
  local EXP=$1 TAG=$2 COND=$3 LOG=$VOX/log/$1.log
  local CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
  echo "[$(ts)] [$TAG] features + 3-seed probe ($COND) -> $CSV" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
    --condition "$COND" --voxel_version v5 --epoch 99 --atom_source ligvdw \
    --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
  && CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions "$COND" --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv "$CSV" >> "$LOG" 2>&1
  echo "[$(ts)] [$TAG] DONE -> $CSV" | tee -a "$MASTER"
}

# ── gate on dg-OTF completion ────────────────────────────────────────────────────
echo "[$(ts)] [arch-coords] waiting for dg-OTF e99 ($DG_CKPT) ..." | tee -a "$MASTER"
while [ ! -f "$DG_CKPT" ]; do sleep 120; done
echo "[$(ts)] [arch-coords] dg-OTF done." | tee -a "$MASTER"

# ── 1) Channel-ViT (OTF C+D+G, channel_group) ────────────────────────────────────
EXP=260620_plinder_otf_channelvit_vit_mae_40m_pretrain; LOG=$VOX/log/$EXP.log
echo "[$(ts)] [channelvit-otf] GPU gate ..." | tee -a "$MASTER"; gate_gpu
echo "[$(ts)] [channelvit-otf] TRAIN $EXP" | tee -a "$MASTER"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
  "${OTF_DSET[@]}" model.patch_embed_mode=channel_group 'model.channel_groups=[7,4,1,1]' \
  num_epochs=100 bsz=8 accum_steps=4 \
  wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,otf,channelvit,channel_group,ligvdw,arch_ablation]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [channelvit-otf] train exit=$?" | tee -a "$MASTER"
if [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then probe "$EXP" plinder_otf_channelvit atomblob_density_gradmag
else echo "[$(ts)] [channelvit-otf] ABORT: no e99" | tee -a "$MASTER"; fi

# ── 2) ChA-MAE-ViT (OTF C+D+G) ────────────────────────────────────────────────────
EXP=260620_plinder_otf_chamae_40m_pretrain; LOG=$VOX/log/$EXP.log
echo "[$(ts)] [chamae-otf] GPU gate ..." | tee -a "$MASTER"; gate_gpu
echo "[$(ts)] [chamae-otf] TRAIN $EXP" | tee -a "$MASTER"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_cha_mae.py --config-name=config_train_atomblob_density_gradmag_cha_mae_40m \
  "${OTF_DSET[@]}" num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,cha_mae,atomblob_density_gradmag,40m,plinder,otf,channelvit,memory_tokens,dcp,ligvdw,arch_ablation]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [chamae-otf] train exit=$?" | tee -a "$MASTER"
if [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then probe "$EXP" plinder_otf_chamae atomblob_density_gradmag
else echo "[$(ts)] [chamae-otf] ABORT: no e99" | tee -a "$MASTER"; fi

# ── 3) Coords-only (OTF, 11ch atoms, SEED 0) ──────────────────────────────────────
EXP=260620_plinder_coordsonly_otf_seed0_vit_mae_40m_pretrain; LOG=$VOX/log/$EXP.log
echo "[$(ts)] [coords-otf] GPU gate ..." | tee -a "$MASTER"; gate_gpu
echo "[$(ts)] [coords-otf] TRAIN $EXP (seed 0)" | tee -a "$MASTER"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
  train_density_vit_mae.py --config-name=config_train_atomblob_vit_mae_40m_invfreq \
  "${OTF_DSET[@]}" seed=0 num_epochs=100 bsz=32 accum_steps=1 \
  wandb_tags="[pretrain,atomblob,40m,invfreq,plinder,otf,ligvdw,coords_only,seed0,arch_ablation]" \
  exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
echo "[$(ts)] [coords-otf] train exit=$?" | tee -a "$MASTER"
if [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then probe "$EXP" plinder_coordsonly_otf_seed0 atomblob_ligvdw
else echo "[$(ts)] [coords-otf] ABORT: no e99" | tee -a "$MASTER"; fi

echo "[$(ts)] === arch + coords OTF chain COMPLETE ===" | tee -a "$MASTER"
