#!/usr/bin/env bash
# 67_train_fusion_champion_reference_4gpu.sh
#   Fuse the FROZEN champion CDG encoder into VoxBind's existing ligand+pocket representation,
#   with BOTH the ligand and the density fed to the encoder at train AND sample time.
#
#     x = ligand_encoder(y) + pocket_encoder(pocket)
#     x = x + density_proj(cat([x, champion(enc_in)]))        # fusion=default, zero-init proj
#     enc_in = [ ligand(7) | pocket(4) | rho | ||grad rho|| ] # champion's 13ch layout
#
#   WHAT THIS ADDS, PRECISELY:
#     * experimental 2Fo-Fc density (+ its gradient magnitude) of the holo crystal. This is
#       NOT a separate leak: the CrossDocked SBDD setting already hands every model a HOLO
#       pocket (side chains in the ligand-bound conformation) — every baseline, TargetDiff /
#       DecompDiff / vanilla VoxBind included, is conditioned on that same structure. The map
#       is a higher-resolution view of an information channel the benchmark already uses.
#     * pocket atoms — the denoiser already sees these via pocket_encoder, so no new info.
#     * the target ligand's own atoms (7 channels), because density_encoder_sees_ligand=true.
#       This one IS the target molecule fed in directly, independent of the density.
#
#   Whether that last point matters is an empirical question, not a label: if the model
#   collapses to reproducing the reference, DIVERSITY (across the 100 samples per pocket) and
#   NOVELTY (Tanimoto / RMSD vs the reference ligand) will show it. Report those alongside
#   Vina; if they are healthy the run belongs in the normal de-novo table, annotated with the
#   density_encoder_sees_ligand setting.
#
#   Provenance: the density's ligand blob is usually NOT the target. 86.3% of the 78,612
#   density-bearing train rows and 52 of the 79 density-bearing TEST pockets are CROSS-docked
#   (receptor PDB != ligand-source PDB), so the map shows the receptor's own ligand instead.
#
#   Env knobs:
#     SEES_LIGAND   true (default) = reference; false = the standard masked-ligand Path B
#     MASK_LIGAND   false (default) = keep ligand density; true = blank the ligand footprint
#     FUSION        default (into lig+poc sum) | protein_first | v3
#     NUM_EPOCHS    relative epochs (loop = range(start, start+NUM); default 100)
#     SIGMA         walk-jump smooth_sigma (default 0.9; 1.0 matches exps/exp_sig1.0_350ep)
#     RESUME        exp DIR to resume from (optional; must be a directory)
#
#   TORCHDYNAMO_DISABLE=1 because this box has no C compiler (inductor cannot build).
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
# The encoder and its geometry travel together. model_zoo entries are NOT
# interchangeable at fixed dims -- champion_100m is dim=640/heads=10 while
# efficient_60m is dim=512/heads=8 -- so overriding ENCODER without the matching
# VIT_* values loads a state_dict into the wrong shapes and fails. Read them out of
# the encoder folder's own cfg.yaml.
ENCODER="${ENCODER:-$ROOT/model_zoo/champion_100m_v2_mask075/checkpoint_e0049.pth.tar}"
VIT_PATCH="${VIT_PATCH:-8}"
VIT_DIM="${VIT_DIM:-640}"
VIT_DEPTH="${VIT_DEPTH:-18}"
VIT_HEADS="${VIT_HEADS:-10}"
VIT_MLP_RATIO="${VIT_MLP_RATIO:-4}"
VIT_DROPOUT="${VIT_DROPOUT:-0.1}"
VIT_NCH="${VIT_NCH:-13}"
VIT_GROUPS="${VIT_GROUPS:-[7,4,2]}"
# Fraction of training examples whose density residual is zeroed. The frozen
# encoder's own dropout is inert (it is held in eval()), so this is the only
# regularization on the density path -- and the density-free branch it trains is
# what a classifier-free guidance weight would later interpolate against.
# 0 (default) reproduces every run made before 2026-08-23 exactly.
COND_DROPOUT="${COND_DROPOUT:-0}"
WARM_START="${WARM_START-$ROOT/exps/exp_sig0.9_v2/checkpoint.pth.tar}"
EXP_NAME="${EXP_NAME:-voxbind_fusion_champion_reference}"
OUT="$ROOT/exps/$EXP_NAME"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
SIGMA="${SIGMA:-0.9}"   # walk-jump noise level; vanilla baselines exist at 0.9 and 1.0
RESUME="${RESUME:-}"
RESUME_EPOCH="${RESUME_EPOCH:-}"
CROPS_DIR="${CROPS_DIR:-$ROOT/dataset/data/xray_crops_aligned_v5}"
SUBSET_N="${SUBSET_N:-78512}"
SUBSET_VAL_N="${SUBSET_VAL_N:-100}"
WANDB_TAGS="${WANDB_TAGS:-[voxbind,fusion,champion,reference,ligand_conditioned]}"

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
cd "$ROOT" || exit 1

[ -x "$PY/torchrun" ] || { echo "[67_fusion] MISSING $PY/torchrun"; exit 1; }
[ -f "$ENCODER" ]     || { echo "[67_fusion] MISSING champion ckpt $ENCODER"; exit 1; }
[ -z "$WARM_START" ] || [ -f "$WARM_START" ] || {
  echo "[67_fusion] MISSING warm-start checkpoint $WARM_START"; exit 1; }

echo "[67_fusion] EXP=$OUT NUM_EPOCHS=$NUM_EPOCHS RESUME=${RESUME:-<none>}"
echo "[67_fusion] student=${WARM_START:-<scratch>}  encoder=$ENCODER (frozen, dim=$VIT_DIM depth=$VIT_DEPTH heads=$VIT_HEADS)"
echo "[67_fusion] sees_ligand=${SEES_LIGAND:-true} mask_ligand=${MASK_LIGAND:-false} fusion=${FUSION:-default} sigma=$SIGMA cond_dropout=$COND_DROPOUT"
echo "[67_fusion] GPUS=$CUDA_VISIBLE_DEVICES (nproc_per_node=$NPROC)"
echo "[67_fusion] crops=$CROPS_DIR subset=$SUBSET_N+$SUBSET_VAL_N"

# WARM_START="" must OVERRIDE the config default (pretrained_path=null), not merely
# skip adding an override — otherwise "from scratch" silently warm-starts from ep350.
if [ -n "${DRY_RUN:-}" ]; then
  set -- "$PY/python" train_ddp.py --cfg job
else
  set -- "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_ddp.py
fi

exec "$@" \
  --config-name config_train_voxbind_fusion_champion_reference \
  wandb=true \
  wandb_tags="$WANDB_TAGS" \
  num_workers=12 prefetch_factor=8 \
  exp_name="$EXP_NAME" output_dir="$OUT" \
  num_epochs="$NUM_EPOCHS" bsz=32 accum_steps=1 lr=1e-5 wd=1e-2 smooth_sigma="$SIGMA" \
  dset.crops_dir="$CROPS_DIR" \
  dset.normalize=false dset.pocket_radius=-1 dset.ligand_radius=0.5 \
  dset.use_xray=true dset.subset_xray_only=true \
  dset.subset_n="$SUBSET_N" dset.subset_val_n="$SUBSET_VAL_N" dset.cache_size=32 \
  model.with_density=true model.density_encoder_type=vit model.density_freeze=true \
  model.density_pretrained_path="$ENCODER" \
  model.density_vit.patch="$VIT_PATCH" model.density_vit.dim="$VIT_DIM" model.density_vit.depth="$VIT_DEPTH" \
  model.density_vit.heads="$VIT_HEADS" model.density_vit.mlp_ratio="$VIT_MLP_RATIO" model.density_vit.dropout="$VIT_DROPOUT" \
  model.density_vit.n_in_channels="$VIT_NCH" \
  model.density_vit.patch_embed_mode=channel_group \
  model.density_vit.channel_groups="$VIT_GROUPS" \
  model.density_encoder_sees_ligand="${SEES_LIGAND:-true}" \
  model.density_mask_ligand="${MASK_LIGAND:-false}" \
  model.fusion="${FUSION:-default}" \
  model.density_encoder_amp=true \
  ++model.density_cond_dropout="$COND_DROPOUT" \
  wjs.n_targets=0 \
  pretrained_path="${WARM_START:-null}" \
  ${RESUME:+resume="$RESUME"} ${RESUME_EPOCH:+resume_epoch="$RESUME_EPOCH"}
