#!/bin/bash
# 260712 ANISOTROPIC splat-head pilot (variant #1 of the splatting-recon test). Extends the
# isotropic splat head (tied w/ baseline, ρ 0.628) with per-Gaussian anisotropy: each of K=4
# patch-Gaussians gets 3 per-axis scales + a rotation quaternion → full covariance Σ=R·diag(s²)·Rᵀ,
# so blobs can elongate/orient along bonds (3DGS-style) instead of round σ. Same seed 42 as the
# isotropic splat for a clean comparison (iso 0.628 vs aniso ?; patch_mlp baseline 0.631±.010).
# Best recipe (d640/L18/h10 @ mask0.75, v2, 50ep). head_style=splat + splat_aniso=true (opt-in).
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS="$1"; PORT="${2:-29601}"
EXP=260712_ar_cvit_100m_v2_splat_aniso_k4

echo "===== SPLAT-ANISO (K=4) · d640/L18/h10 @ mask0.75 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvitSplatAniso \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    model.head_style=splat +model.splat_k=4 +model.splat_aniso=true mae.mask_ratio=0.75 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu "${GPUS%%,*}" --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== SPLAT-ANISO COMPLETE $(date '+%m-%d %H:%M:%S') ====="
