#!/usr/bin/env bash
# 60_train_frozen_efficient60m_4gpu.sh
#   Launches the generative VoxBind denoiser (train_ddp.py) on all 4 H200s with the
#   *frozen* 60M density encoder fused in — model_zoo/efficient_60m_v3_mask085
#   (ChannelViT [7,4,2], dim512/depth18, 64,619,520 params; test rho 0.641 at 1/3
#   fewer params than the 100M champion).
#
#   Reconstructed verbatim from the overrides of the 2026-07-29 16:17 launch
#   (exps/voxbind_frozen_efficient60m_holo_xrayfull_20260729/.hydra/overrides.yaml),
#   which died 3 minutes in when the container restart wiped /opt/conda/envs/voxbind
#   before any checkpoint was written. Restore the env first:
#     bash scripts/restore_voxbind_env.sh
#
#   Conditioning density MUST be v5 (arcsinh+z) with normalize=false / pocket_radius=-1
#   to match the encoder's pretraining, else the frozen encoder sees OOD input.
#
#   Env knobs:
#     NUM_EPOCHS   relative epochs to run (default 350; loop = range(start, start+NUM))
#     RESUME       exp DIR to resume from (optional; must be a directory)
#     RESUME_EPOCH start_epoch override (optional; = ckpt['epoch']+1)
#
#   TORCHDYNAMO_DISABLE=1 because this box has no C compiler (inductor cannot build).
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
ENCODER="$ROOT/model_zoo/efficient_60m_v3_mask085/checkpoint_e0049.pth.tar"
# WARM_START: vanilla (density-free) VoxBind checkpoint to seed the denoiser from.
#   exps/exp_sig0.9_v2/checkpoint.pth.tar = vanilla sig0.9, epoch 350 (epoch-matched baseline).
#   Set WARM_START="" to train the density model from scratch instead.
# NB: ${VAR-default}, not ${VAR:-default} - an explicitly empty WARM_START must
# mean "from scratch", while leaving it unset keeps the warm-start default.
WARM_START="${WARM_START-$ROOT/exps/exp_sig0.9_v2/checkpoint.pth.tar}"
EXP_NAME="${EXP_NAME:-voxbind_frozen_efficient60m_warmstart_sig0.9_20260729}"
OUT="$ROOT/exps/$EXP_NAME"
NUM_EPOCHS="${NUM_EPOCHS:-350}"
RESUME="${RESUME:-}"
RESUME_EPOCH="${RESUME_EPOCH:-}"

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
# Respect an inherited CUDA_VISIBLE_DEVICES so this composes with scripts/99_chain.sh,
# which reserves GPUs under advisory locks and exports its selection before exec'ing us.
# Hardcoding 0,1,2,3 here would silently steal GPUs the harness did not reserve.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
cd "$ROOT" || exit 1

[ -x "$PY/torchrun" ] || { echo "[60_train] MISSING $PY/torchrun — run scripts/restore_voxbind_env.sh first"; exit 1; }
[ -f "$ENCODER" ]     || { echo "[60_train] MISSING encoder checkpoint $ENCODER"; exit 1; }
[ -z "$WARM_START" ] || [ -f "$WARM_START" ] || {
  echo "[60_train] MISSING warm-start checkpoint $WARM_START"; exit 1; }

echo "[60_train] NUM_EPOCHS=$NUM_EPOCHS RESUME=${RESUME:-<none>} EXP=$OUT"
echo "[60_train] WARM_START=${WARM_START:-<none, from scratch>}"
echo "[60_train] GPUS=$CUDA_VISIBLE_DEVICES (nproc_per_node=$NPROC)"

# DRY_RUN=1 → compose+print the Hydra config and exit (no GPUs, no DDP, no training).
# Pre-flight for override syntax before killing a live run.
if [ -n "${DRY_RUN:-}" ]; then
  set -- "$PY/python" train_ddp.py --cfg job
else
  set -- "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_ddp.py
fi

exec "$@" \
  --config-name config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1 \
  wandb=true \
  wandb_tags='[voxbind,density_cond,frozen_encoder,model_zoo,efficient_60m_v3_mask085,holo,full_density_crossdocked,direct_voxbind_trainer,faithful_direct,efficient_60m]' \
  num_workers=16 prefetch_factor=8 \
  exp_name="$EXP_NAME" output_dir="$OUT" \
  num_epochs="$NUM_EPOCHS" bsz=32 accum_steps=1 lr=1e-5 wd=1e-2 \
  smooth_sigma=0.9 with_gradmag=false \
  dset.crops_dir="$ROOT/dataset/data/xray_crops_aligned_v5" \
  dset.normalize=false dset.pocket_radius=-1 dset.ligand_radius=0.5 \
  dset.use_xray=true dset.subset_xray_only=true \
  dset.subset_n=78512 dset.subset_val_n=100 dset.cache_size=32 \
  model.with_density=true model.density_encoder_type=vit model.density_freeze=true \
  model.density_pretrained_path="$ENCODER" \
  model.density_vit.patch=8 model.density_vit.dim=512 model.density_vit.depth=18 \
  model.density_vit.heads=8 model.density_vit.mlp_ratio=4 model.density_vit.dropout=0.1 \
  model.density_vit.n_in_channels=13 \
  model.density_vit.patch_embed_mode=channel_group \
  model.density_vit.channel_groups='[7,4,2]' \
  model.density_mask_ligand="${MASK_LIGAND:-false}" model.fusion="${FUSION:-default}" \
  model.density_encoder_amp="${ENCODER_AMP:-true}" \
  model.density_attenuate="${ATTENUATE:-false}" \
  model.density_attenuate_sigma="${ATTEN_SIGMA:-7.0}" \
  model.density_attenuate_quantile="${ATTEN_Q:-0.90}" \
  model.density_attenuate_strength="${ATTEN_STRENGTH:-1.0}" \
  model.density_attenuate_noise_sigma="${ATTEN_NOISE_SIGMA:-7.0}" \
  wjs.n_targets=0 \
  ${WARM_START:+pretrained_path="$WARM_START"} \
  ${RESUME:+resume="$RESUME"} ${RESUME_EPOCH:+resume_epoch="$RESUME_EPOCH"}
