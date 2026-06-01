#!/bin/bash
# 36_train_atomblob_merged_density_vit_mae_40m_weighted.sh
#
# Merged-atomblob (7 ch, pocket folded into ligand element channels) + X-ray
# density (1 ch) ViT-MAE pretraining with the full weighted recipe:
#   - mae.channel_weighting=inv_sqrt_freq   (operates on the 7 merged channels)
#   - mae.mask_strategy=atom_biased         (atoms-first 60% block mask)
#   - mae.density_channel_weight=0.1        (density downweighted inside joint MAE loss)
#
# Compared to 35_train_atomblob_density_vit_mae_40m_weighted.sh:
#   - input is 8 ch (7 merged atoms + density) instead of 12 (7 lig + 4 poc + density)
#   - lig-vs-pocket disambiguation drops out of the pretext entirely
#   - merged_C/O/N/S channels are dense (pocket-dominated) → no 'predict zero' cheat
#     on the bulk atom signal; rare-ligand-only channels (F/Cl/P) still tiny
#     but heavily up-weighted by inv-sqrt-freq.
#
# GPU 4, 5, 6, 7 (4 GPUs) with accum_steps=3 → effective batch = 8 × 4 × 3 = 96
# (close to the 80 used by the 5-GPU baselines). Expected wallclock ~14-15h.
#
# Output: exps/260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain/checkpoint.pth.tar
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
EXP=260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

mkdir -p "$LOG"
cd "$VOX" || exit 1
echo "[$(ts)] launching $EXP  (GPU 4-7 | bsz 8 × 4 × accum 3 = 96 eff | merged-7 atoms + density + atom-biased mask + inv-sqrt-freq + density_weight=0.1)"

CUDA_VISIBLE_DEVICES=4,5,6,7 $PY/torchrun --standalone --nproc_per_node=4 train_density_vit_mae.py \
    --config-name=config_train_atomblob_merged_density_vit_mae_40m_weighted \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops_aligned \
    dset.subset_xray_only=true \
    dset.subset_n=78428 \
    dset.subset_val_n=100 \
    dset.use_xray=true \
    input_mode=atomblob_merged_density \
    model.n_in_channels=8 \
    num_epochs=100 \
    bsz=8 \
    accum_steps=3 \
    'wandb_tags=[pretrain,atomblob_merged_density_vit_mae,40m,weighted,merged7,atom_biased_mask,inv_sqrt_freq,density_downweight,crossdocked_xray]' \
    lr=1e-4 \
    wd=5e-2 \
    seed=42 \
    exp_name=$EXP \
    output_dir=$VOX/exps/$EXP \
    >> $LOG/${EXP}.log 2>&1

echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
