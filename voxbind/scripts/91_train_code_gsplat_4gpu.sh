#!/usr/bin/env bash
# 91_train_code_gsplat_4gpu.sh — CoDE (VoxBind + Ours, frozenenc atomblob7 v2.1, sigma 0.9)
#   with its conv output head REPLACED by the Gaussian-splat rasterizer
#   (models/gsplat_head.py, model.decoder=gsplat), warm-started from CoDE's trained weights.
#
#   Question: how well does a "predict Gaussians, then splat them" decoder do on top of an
#   already-trained denoiser? Everything upstream of the head -- the frozen density encoder,
#   density_proj, ligand/pocket encoders, the U-Net -- loads from CoDE; only gsplat_head
#   starts from init (train_ddp.py lets `gsplat_head.` be missing from the warm start).
#   The old final_ligand is still built for key compatibility, frozen, and never called.
#
#   Same config file CoDE was launched with, so data, density conditioning, sigma, bsz and
#   the encoder geometry are CoDE's. Differences are only what is set below.
#
#   Env knobs: EXP_NAME, NUM_EPOCHS, LR, HEAD_LR_MULT, BSZ, STRIDE, K, HIDDEN, WANDB,
#              WARM_START, RESUME, DRY_RUN.
#
#   bash scripts/91_train_code_gsplat_4gpu.sh            # from voxbind/
#   DRY_RUN=1 bash scripts/91_train_code_gsplat_4gpu.sh  # print the composed config only
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin

CODE_EXP="$ROOT/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9"
WARM_START="${WARM_START-$CODE_EXP/checkpoint.pth.tar}"
EXP_NAME="${EXP_NAME:-voxbind_code_gsplat_s4k2_sig0.9_$(date +%y%m%d)}"
OUT="$ROOT/exps/$EXP_NAME"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
LR="${LR:-1e-5}"                  # CoDE's own fine-tune lr for the pretrained body
HEAD_LR_MULT="${HEAD_LR_MULT:-10}" # the head starts from init; 1e-5 would barely move it
BSZ="${BSZ:-32}"                  # per rank, CoDE's value (eff 128 on 4 GPUs)
STRIDE="${STRIDE:-4}"
K="${K:-2}"
HIDDEN="${HIDDEN:-256}"
WANDB="${WANDB:-false}"           # unattended overnight run: no login prompt to hang on
RESUME="${RESUME:-}"

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1      # no C compiler on this box
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
cd "$ROOT" || exit 1

[ -x "$PY/torchrun" ] || { echo "[91] MISSING $PY/torchrun"; exit 1; }
[ -z "$WARM_START" ] || [ -f "$WARM_START" ] || { echo "[91] MISSING warm start $WARM_START"; exit 1; }

echo "[91] EXP=$OUT epochs=$NUM_EPOCHS lr=$LR head_lr_mult=$HEAD_LR_MULT bsz=$BSZ"
echo "[91] gsplat stride=$STRIDE k=$K hidden=$HIDDEN warm_start=${WARM_START:-<none>}"
echo "[91] GPUS=$CUDA_VISIBLE_DEVICES (nproc=$NPROC)"

if [ -n "${DRY_RUN:-}" ]; then
  set -- "$PY/python" train_ddp.py --cfg job
else
  set -- "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_ddp.py
fi

exec "$@" \
  --config-name config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1 \
  wandb="$WANDB" \
  exp_name="$EXP_NAME" output_dir="$OUT" \
  num_epochs="$NUM_EPOCHS" bsz="$BSZ" accum_steps="${ACCUM:-1}" lr="$LR" \
  wjs.n_targets=0 \
  +model.decoder=gsplat \
  +model.gsplat.stride="$STRIDE" \
  +model.gsplat.gaussians_per_anchor="$K" \
  +model.gsplat.hidden="$HIDDEN" \
  +model.gsplat.lr_mult="$HEAD_LR_MULT" \
  ${WARM_START:+pretrained_path="$WARM_START"} \
  ${RESUME:+resume="$RESUME"}
