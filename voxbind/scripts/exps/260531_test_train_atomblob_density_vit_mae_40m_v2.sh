#!/bin/bash
# 39_train_atomblob_density_vit_mae_40m_v2.sh
#
# Vanilla 12-ch atomblob_density (NO weighted recipe) on v2 density crops,
# single GPU on cuda:5. Mirrors the 260526 baseline's training setup with
# only the density data swapped.
#
# Effective batch = 8 × 1 GPU × 10 accum = 80 (matches 260526's 5-GPU × 2 accum).
# Single-GPU wallclock ~ 4× the 4-GPU runs: ~55–60h for 100 epochs.
#
# Output: exps/260531_atomblob_density_vit_mae_40m_v2_pretrain/checkpoint.pth.tar
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260531_atomblob_density_vit_mae_40m_v2_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 5 only | bsz 8 × accum 10 = 80 eff | vanilla 12-ch + v2 density)"

CUDA_VISIBLE_DEVICES=5 $PY/torchrun --standalone --nproc_per_node=1 train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_vit_mae_40m_v2 \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned_v2 \
    dset.normalize=false \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    input_mode=atomblob_density \
    model.n_in_channels=12 \
    num_epochs=100 \
    bsz=8 \
    accum_steps=10 \
    'wandb_tags=[pretrain,atomblob_density_vit_mae,40m,vanilla,v2_density,pool_zscore,no_clip,single_gpu,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
