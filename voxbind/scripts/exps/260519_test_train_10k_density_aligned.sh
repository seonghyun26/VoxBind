#!/bin/bash
# 11_train_10k_density_aligned.sh — density arm on FRAME-CORRECTED x-ray crops.
#
# Density-only 10k sanity check for the x-ray crop alignment fix
# (dataset/00d_align_xray_density.py). Run dir exps/260519_voxbind_10k_density_aligned;
# compare to exps/260518_voxbind_10k_density (same hparams, misaligned crops).
#
# History:
#   * launched on GPUs 0-3  (nproc 4, eff batch 4x4x1=16);  epochs 0-3.
#   * RESUMED on GPUs 1-3   (nproc 3, eff batch 4x3x1=12);  epochs 4-80 (GPU 0 freed).
#   * RESUMED on GPUs 1-6   (nproc 6, eff batch 4x6x1=24);  epochs 81-99 — ~2x faster
#     now that GPUs 4-6 are free (the 260518 baseline finished overnight).
#
# IMPORTANT: train_ddp.py::create_exp_dir() rewrites <exp>/cfg.yaml from the
# CURRENT cfg *before* the resume branch reads it back. A resume must therefore
# pass the FULL original arg set (not just resume=) — otherwise the model is
# built from config defaults (with_density=false) and the checkpoint fails to
# load. resume= then continues from <exp>/checkpoint.pth.tar.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260519_voxbind_10k_density_aligned
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

cd "$VOX" || exit 1
echo "[$(ts)] resuming $EXP  (GPU 1-6 | from exps/$EXP/checkpoint.pth.tar)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $PY/torchrun --standalone --nproc_per_node=6 train_ddp.py \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_n=10000 dset.subset_xray_only=true dset.subset_val_n=100 \
    dset.use_xray=true \
    num_epochs=100 bsz=4 accum_steps=1 seed=42 \
    wjs.n_targets=0 \
    model.with_density=true \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    resume=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
