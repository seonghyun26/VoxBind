#!/bin/bash
# 32_train_atomblob_vit_mae_40m.sh — ViT-MAE pre-training on ATOM-BLOB voxels.
#
# One half of the multi-channel-ablation pair launched alongside
# 33_train_atomblob_density_vit_mae_40m.sh. Same encoder size (40M ViT,
# D=512 L=12 H=8) and the same X-ray-available sample subset
# (subset_xray_only=true, subset_n=78428, subset_val_n=100) as the
# 31_train_density_vit_mae_40m_xray.sh baseline — only the input changes.
#
# Encoder input: (B, 11, G³) — 7 ligand + 4 pocket atom-type voxels.
# Pretext: 3-D block-mask (60% of 8³ cubes zeroed) + masked-voxel MSE
# across all 11 channels. The atom-structure auxiliary head is OFF
# (n_struct_channels=0) — predicting atoms from atoms is degenerate.
#
# use_xray=false: the per-sample CCP4/crop read is skipped entirely; only
# the X-ray-availability MANIFEST is consulted, for subset filtering.
#
# Output: exps/260526_atomblob_vit_mae_40m_pretrain/checkpoint.pth.tar
# carries `state_dict_ema` (full DensityViTMAE) and `encoder_state_dict_ema`
# (encoder.* slice). The encoder slice loads into any DensityViT instantiated
# with n_in_channels=11; the existing VoxBind.density_encoder slot (n_in=1)
# does NOT accept this checkpoint — a follow-up branch will widen that slot.
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260526_atomblob_vit_mae_40m_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 1-5 | bsz 8 × 5 × accum 2 = 80 eff | atomblob, no density I/O)"

CUDA_VISIBLE_DEVICES=1,2,3,4,5 $PY/torchrun --standalone --nproc_per_node=5 train_density_vit_mae.py \
    --config-name=config_train_atomblob_vit_mae_40m \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=false \
    input_mode=atomblob \
    num_epochs=100 \
    bsz=8 \
    accum_steps=2 \
    'wandb_tags=[pretrain,atomblob_vit_mae,40m,stage_a,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
