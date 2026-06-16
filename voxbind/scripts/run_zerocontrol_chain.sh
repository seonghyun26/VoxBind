#!/bin/bash
# run_zerocontrol_chain.sh — CAPACITY control for the density-ablation story.
# Same combined encoder as 260606 / the noise control (260612): dedicated config
# config_train_atomblob_density_gradmag_vit_mae_40m_invfreq (head_style=patch_mlp,
# n_in=13, ligvdw, invfreq), bsz=8/accum=3 — IDENTICAL recipe, ONLY density+gradmag are
# ZERO-filled (train: gradmag derived ‖∇0‖=0; probe: zero voxels). Isolates whether the
# +0.05 affinity gain is added-channel capacity (zeros reproduce it) vs needs varying input.
# Pretrain ~9.5 h on $GPUS, then 3-seed frozen probe on the canonical 2172/480/839 split.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
EXP=260613_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_zerocontrol_pretrain
ZD=$DATA/xray_crops_aligned_v5_zero
ZV=$DATA/pdbbind/voxels_v5_zero
LOG=$VOX/log/$EXP.log
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PROBE_GPU=${GPUS%%,*}
ts(){ date "+%F %T"; }

# gate: zero crops materialized (zero crops + REAL availability masks)
if [ ! -d "$ZD/density/train" ] || [ ! -d "$ZV/density" ]; then
    echo "[$(ts)] ABORT: zero crops not materialized ($ZD / $ZV)" >> "$LOG"; exit 1
fi

# 1. pretrain (~9.5h) — dedicated config (== 260606/noise), data=zero, bsz=8/accum=3
echo "[$(ts)] launching ZERO-control pretrain on GPU $GPUS (config_train_atomblob_density_gradmag_vit_mae_40m_invfreq)" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=$NPROC \
    train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
    dset.data_dir="$DATA" \
    dset.crops_dir="$ZD/density" \
    bsz=8 accum_steps=3 \
    wandb_tags='[pretrain,atomblob_density_gradmag,40m,invfreq,v5,ligvdw,clip30,crossdocked_xray,zero_control,density_ablation]' \
    exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] ABORT: training produced no e99" >> "$LOG"; exit 1
fi

# 2. probe: zero density+gradmag voxels → zero-pretrained encoder (matched 839 split)
CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py features \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --atom_source ligvdw --exp_dir "exps/$EXP" --tag zerocontrol --noise_voxels_dir "$ZV" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag zerocontrol --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv dataset/data/pdbbind/probe_results_e99_v5_filtered_zerocontrol.csv >> "$LOG" 2>&1
echo "[$(ts)] zero-control chain DONE -> probe_results_e99_v5_filtered_zerocontrol.csv" >> "$LOG"
