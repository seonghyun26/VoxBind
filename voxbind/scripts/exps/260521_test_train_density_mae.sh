#!/bin/bash
# 13_train_density_mae.sh — voxel-MAE pre-training for the density encoder.
#
# Stage A: synthetic density (gaussian-blur over voxelized ligand+pocket atoms,
# z-scored, light noise) + 3D block masking. Targets: clean density on masked
# voxels + 11-channel atom voxels (7 ligand + 4 pocket).
#
# Output: exps/260520_density_mae_pretrain/ — checkpoint.pth.tar carries both
# the full DensityMAE EMA state and an `encoder_state_dict_ema` slice for
# easy load into VoxBind (Phase 3).
#
# GPU plan: 1-6 (free during this run; the 10k baseline freed GPUs 1-6 overnight).
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
# v2 — adds struct_pos_weight=100 to fix the constant-zero shortcut on the
# sparse ligand structure target (260520 diagnosis: ligand pred peak 0.012
# vs target peak 0.99). 260520_density_mae_pretrain kept as reference.
EXP=260521_density_mae_posweight
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-6 | dataset=full crossdocked | bsz 16 × 6 = 96 eff)"

# bsz=16/rank: at 64³ × 16 channels × 4-block encoder, backward activations
# scale roughly linearly in bsz. bsz=64 OOMed (~22 GiB/rank); bsz=16 leaves
# ~12 GiB headroom for PyUUL's internal voxelizer intermediate (≈4 GiB peak).
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $PY/torchrun --standalone --nproc_per_node=6 train_density_mae.py \
    dset=crossdocked \
    dset.data_dir=$DATA \
    num_epochs=100 \
    bsz=16 \
    accum_steps=1 \
    'wandb_tags=[pretrain,density_mae,stage_a,posweight_100,crossdocked_train]' \
    lr=1e-4 \
    wd=1e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
