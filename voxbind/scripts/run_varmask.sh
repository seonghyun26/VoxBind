#!/usr/bin/env bash
# Single VARIABLE mask-ratio (R2MAE) pretraining on the champion recipe.
#
# Champion 260705 recipe pinned (v2 · 100M dim640/depth18/heads10 · channel_group
# [7,4,2] · vdW radii · uniform mask · 50 ep · lr1e-4/wd0.05). The ONLY addition is
# mae.mask_ratio_min/max → each training batch draws r ~ U[min,max] (train-only; val
# stays at the fixed mae.mask_ratio). One encoder internalises mask-ratio diversity at
# 1× inference cost — the single-model analog of the v3 mask-only ensemble (+0.007).
#
#   bash scripts/run_varmask.sh --gpus 0-3                 # default U[0.6,0.9]
#   bash scripts/run_varmask.sh --gpus 0-3 --min 0.7 --max 0.95
#   bash scripts/run_varmask.sh --gpus 7 --dry-run
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0-3"; RMIN=0.6; RMAX=0.9; EPOCHS=50; DRY=0; INVOCAB=0
while [[ $# -gt 0 ]]; do case "$1" in
  --gpus)     GPUS="${2:?}"; shift 2;;
  --min)      RMIN="${2:?}"; shift 2;;
  --max)      RMAX="${2:?}"; shift 2;;
  --epochs)   EPOCHS="${2:?}"; shift 2;;
  --in_vocab) INVOCAB=1; shift;;          # v2.2: drop out-of-vocab-ligand complexes at load time
  --dry-run)  DRY=1; shift;;
  *) echo "unknown arg '$1'"; exit 1;;
esac; done

MM=$(awk -v a="$RMIN" -v b="$RMAX" 'BEGIN{printf "%02d%02d", a*100, b*100}')   # 0.6,0.9 -> 6090
VER=v2; INV_OVR=""
if [[ $INVOCAB -eq 1 ]]; then VER=v22; INV_OVR="dset.in_vocab_only=true"; fi
NAME="260813_cdg_100m_${VER}_varmask${MM}"
DRYFLAG=(); [[ $DRY -eq 1 ]] && DRYFLAG=(--dry-run)

bash "$SCRIPT_DIR/03_pretrain.sh" --name "$NAME" --gpus "$GPUS" "${DRYFLAG[@]}" -- \
  bsz=4 accum_steps=8 num_workers=6 prefetch_factor=2 num_epochs="$EPOCHS" \
  lr=0.0001 wd=0.05 \
  model.patch_embed_mode=channel_group 'model.channel_groups=[7,4,2]' \
  model.dim=640 model.depth=18 model.heads=10 \
  mae.mask_strategy=uniform mae.mask_ratio=0.75 \
  mae.mask_ratio_min="$RMIN" mae.mask_ratio_max="$RMAX" \
  dset.ligand_radius=-1 dset.pocket_radius=-1 \
  dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
  dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
  dset.subset_n=112000 dset.subset_val_n=100 $INV_OVR \
  wandb=true "wandb_tags=[pretrain,cdg,100m,${VER},varmask,r2mae,${MM}]"
