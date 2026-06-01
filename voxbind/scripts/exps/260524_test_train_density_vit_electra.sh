#!/bin/bash
# 28_train_density_vit_electra.sh — ELECTRA-style pre-training for the ViT density encoder.
#
# Same train script as 18_train_density_vit_mae.sh, but with
# `mae.pretext_style=electra`: blocks are CORRUPTED (rule-based) instead of
# zeroed, and the discriminator emits per-patch RTD logits trained with BCE.
# The 11-ch atom-structure aux head is retained.
#
# Output: exps/260524_density_vit_electra_pretrain/ — checkpoint.pth.tar carries
# both `state_dict_ema` (full DensityViTMAE EMA) and `encoder_state_dict_ema`
# (encoder.* slice — drops into VoxBind.density_encoder with
# density_encoder_type=vit via the existing loader in models/__init__.py).
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260524_density_vit_electra_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | dataset=full crossdocked | bsz 16 × 5 = 80 eff)"

# bsz=16/rank to keep effective batch close to the MAE precedent
# (260522_density_vit_mae_pretrain: bsz 16 × 6 = 96 eff). With 5 ranks here we
# get bsz 16 × 5 = 80 eff — close enough for a fair comparison.
CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_density_vit_mae.py \
    --config-name=config_train_density_vit_electra \
    dset=crossdocked \
    dset.data_dir=$DATA \
    num_epochs=100 \
    bsz=16 \
    accum_steps=1 \
    'wandb_tags=[pretrain,density_vit,electra,stage_a,rule_based,crossdocked_train]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
