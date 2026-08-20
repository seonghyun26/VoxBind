#!/usr/bin/env bash
# 65_train_urepa_champion_4gpu.sh
#   U-REPA representation-alignment finetune of VoxBind on all 4 H200s:
#   student = vanilla sig0.9 ep350 (exps/exp_sig0.9_v2, the epoch-matched baseline),
#   teacher = FROZEN champion CDG (model_zoo/champion_100m_v2_mask075, ChannelViT [7,4,2],
#   dim 640, probe rho 0.644).
#
#   The denoiser stays DENSITY-FREE (model.with_density=false). Density reaches the loop
#   only to build the teacher's 13-ch input; inference is stock VoxBind walk-jump. The
#   alignment merely reshapes the U-Net's bottleneck representation during training.
#
#   Pre-check evidence (voxbind/test/results/*, notebook/html/260806/urepa_*.md):
#     A1  teacher(apo) rho 0.601 vs student(ligand-free bottleneck) 0.474, pocket null 0.145
#     B1  R2(teacher<-student) 0.069-0.076 at the bottleneck -> RELATIONAL loss only;
#         a tokenwise cosine/MSE term would chase an unreachable target
#     B2  bottleneck is the alignment point; C1 grids match at 8^3 (no upsample)
#     B3  clean vs sigma=0.9 differ by <=0.017 -> no noise averaging needed
#
#   Teacher runs LIVE (no_grad + bf16) so it sees the student's own AUGMENTED frame —
#   precomputed canonical tokens (dataset/00j) would break the token<->token correspondence
#   under cfg.aug rotation.
#
#   Env knobs:
#     LAM           alignment weight (default 5.0; sweep {0,1,5,20}, 0 = off control)
#     TAU/CENTER    softmax temperature / relation centering. CENTER=false reproduces the
#                   2026-08-05 first attempt, whose teacher target was UNIFORM
#                   (entropy_ratio 1.000) because raw ViT cosines sit at 0.976/0.999.
#     TEACHER_DIR   frozen CDG teacher exp dir (default champion)
#     APO           true = atoms0_dm target (default), false = holo
#     STAGE1        epochs with the U-Net held at lr=0 (projector only). Default 5.
#     W_INTRA/W_INTER  manifold axis weights (explicit, batch-decoupled — not the 1/B of flat)
#     SAMPLING      split (default) | pool (sample-level, the paper's axis) | block | flat
#     REPA_WEIGHT/ML_WEIGHT  L_align = REPA_WEIGHT*tokenwise + ML_WEIGHT*manifold.
#                   0/1 = ours (manifold only); 1/3 = the paper's L_REPA + 3*L_ML.
#     NUM_EPOCHS    relative epochs to run (loop = range(start, start+NUM); default 350)
#     RESUME        exp DIR to resume from (optional; must be a directory)
#     EXP_NAME      override the run name (default encodes LAM)
#
#   TORCHDYNAMO_DISABLE=1 because this box has no C compiler (inductor cannot build).
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin

TEACHER_DIR="${TEACHER_DIR:-model_zoo/champion_100m_v2_mask075}"
WARM_START="${WARM_START-$ROOT/exps/exp_sig0.9_v2/checkpoint.pth.tar}"
LAM="${LAM:-5.0}"        # measured scale: L_denoise ~710/batch vs L_align ~5.2 -> ~3.5%
APO="${APO:-true}"
STAGE1="${STAGE1:-5}"
W_INTRA="${W_INTRA:-1.0}"
W_INTER="${W_INTER:-1.0}"
EXP_NAME="${EXP_NAME:-voxbind_urepa_champion_ep350_lam${LAM}}"
OUT="$ROOT/exps/$EXP_NAME"
NUM_EPOCHS="${NUM_EPOCHS:-350}"
RESUME="${RESUME:-}"
RESUME_EPOCH="${RESUME_EPOCH:-}"

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
cd "$ROOT" || exit 1

[ -x "$PY/torchrun" ] || { echo "[65_urepa] MISSING $PY/torchrun"; exit 1; }
[ -f "$ROOT/$TEACHER_DIR/checkpoint_e0049.pth.tar" ] || {
  echo "[65_urepa] MISSING teacher ckpt $TEACHER_DIR/checkpoint_e0049.pth.tar"; exit 1; }
[ -z "$WARM_START" ] || [ -f "$WARM_START" ] || {
  echo "[65_urepa] MISSING warm-start checkpoint $WARM_START"; exit 1; }

echo "[65_urepa] EXP=$OUT  NUM_EPOCHS=$NUM_EPOCHS  RESUME=${RESUME:-<none>}"
echo "[65_urepa] student=${WARM_START:-<scratch>}"
echo "[65_urepa] teacher=$TEACHER_DIR  apo=$APO  lam=$LAM  stage1=$STAGE1"
echo "[65_urepa] GPUS=$CUDA_VISIBLE_DEVICES (nproc_per_node=$NPROC)"

# DRY_RUN=1 → compose+print the Hydra config and exit (no GPUs, no DDP, no training).
if [ -n "${DRY_RUN:-}" ]; then
  set -- "$PY/python" train_ddp.py --cfg job
else
  set -- "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_ddp.py
fi

exec "$@" \
  --config-name config_train_voxbind_urepa_champion \
  wandb=true \
  wandb_tags='[voxbind,urepa,champion,alignment,ep350]' \
  num_workers=12 prefetch_factor=8 \
  exp_name="$EXP_NAME" output_dir="$OUT" \
  num_epochs="$NUM_EPOCHS" bsz=32 accum_steps=1 lr=1e-5 wd=1e-2 smooth_sigma=0.9 \
  dset.crops_dir="$ROOT/dataset/data/xray_crops_aligned_v5" \
  dset.normalize=false dset.pocket_radius=-1 dset.ligand_radius=0.5 \
  dset.use_xray=true dset.subset_xray_only=true \
  dset.subset_n=78512 dset.subset_val_n=100 dset.cache_size=32 \
  model.with_density=false \
  urepa.enabled=true \
  urepa.exp_dir="$TEACHER_DIR" urepa.epoch=49 \
  urepa.apo="$APO" urepa.lam="$LAM" urepa.stage1_epochs="$STAGE1" \
  urepa.sampling="${SAMPLING:-split}" urepa.w_intra="$W_INTRA" urepa.w_inter="$W_INTER" \
  urepa.repa_weight="${REPA_WEIGHT:-0.0}" urepa.ml_weight="${ML_WEIGHT:-1.0}" \
  urepa.tau="${TAU:-0.1}" urepa.center="${CENTER:-true}" urepa.mode="${MODE:-relkl}" \
  urepa.tokens_per_sample="${TOKENS_PER_SAMPLE:-128}" \
  urepa.proj_lr="${PROJ_LR:-1e-4}" urepa.teacher_amp=true \
  wjs.n_targets=0 \
  ${WARM_START:+pretrained_path="$WARM_START"} \
  ${RESUME:+resume="$RESUME"} ${RESUME_EPOCH:+resume_epoch="$RESUME_EPOCH"}
