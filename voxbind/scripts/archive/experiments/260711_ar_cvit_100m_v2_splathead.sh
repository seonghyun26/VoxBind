#!/bin/bash
# 260711 GAUSSIAN-SPLATTING RECONSTRUCTION HEAD (initial test, user req). Replaces the MAE
# patch-MLP voxel-regression recon head with a SPLAT head: each masked patch token predicts K=4
# free Gaussians (sub-voxel μ, isotropic σ, per-channel amplitude) that are analytically splatted
# onto the patch's 8³ grid and summed → reconstruct density as a sum of atom-like Gaussians (the
# physical prior) instead of arbitrary voxels. Head is pretrain-only (NOT in encoder.*), so the
# frozen probe is bit-exact → clean test of "does splatting recon give a better ENCODER for affinity".
# Best recipe otherwise (d640/L18/h10 @ mask0.75, v2, 50ep). vs patch_mlp baseline ~0.63-0.64.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7; PORT=29596
EXP=260711_ar_cvit_100m_v2_splathead_k4

echo "===== SPLAT-HEAD (K=4) · d640/L18/h10 @ mask0.75 START $(date '+%m-%d %H:%M:%S') ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvitSplatK4 \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    model.head_style=splat +model.splat_k=4 mae.mask_ratio=0.75 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu 4 --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== SPLAT-HEAD COMPLETE $(date '+%m-%d %H:%M:%S') ====="
