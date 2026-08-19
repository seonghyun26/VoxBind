#!/usr/bin/env bash
# Per-channel-group masking (mae.mask_strategy=per_group): each ChannelViT group gets an
# INDEPENDENT spatial block mask, so the encoder must inpaint a group's holes from the
# OTHER groups still visible there (cross-channel inference — e.g. recover ligand identity
# from surrounding electron density). Only meaningful WITH channel separation, so groups
# default to [7,4,1,1] (density & gradmag split into their own groups). Champion recipe
# otherwise (100M · vdW · fixed mask 0.75 · 50 ep), on clean v2.2 with --in_vocab.
# It can also use the CASF-clean v2.4 data view and a custom EMA decay.
#
#   bash scripts/run_pergroup.sh --gpus 0-3 --in_vocab
#   bash scripts/run_pergroup.sh --gpus 2-7 --v24 --groups 7,4,2 --ema-decay 0.9999
#   bash scripts/run_pergroup.sh --gpus 7 --in_vocab --dry-run
#   SMOKE=1 bash scripts/run_pergroup.sh --gpus 7            # tiny model, few steps, no wandb
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0-3"; EPOCHS=50; DRY=0; INVOCAB=0; V24=0; CGRP="7,4,1,1"; RATIO=0.75; RMIN=""; RMAX=""; EMA_DECAY=0.999
while [[ $# -gt 0 ]]; do case "$1" in
  --gpus)     GPUS="${2:?}"; shift 2;;
  --epochs)   EPOCHS="${2:?}"; shift 2;;
  --groups)   CGRP="${2:?}"; shift 2;;    # ChannelViT groups WITHOUT brackets, e.g. 7,4,1,1 or 7,4,2
  --ratio)    RATIO="${2:?}"; shift 2;;   # fixed mask ratio (ignored if --min/--max given)
  --min)      RMIN="${2:?}"; shift 2;;    # variable-rate lower bound → r ~ U[min,max] per batch
  --max)      RMAX="${2:?}"; shift 2;;
  --in_vocab) INVOCAB=1; shift;;
  --v24)      V24=1; shift;;              # CASF-2016 ID30-clean v2.4 view
  --ema-decay) EMA_DECAY="${2:?}"; shift 2;;
  --dry-run)  DRY=1; shift;;
  *) echo "unknown arg '$1'"; exit 1;;
esac; done

GTAG=$(echo "$CGRP" | tr -cd '0-9')            # 7,4,1,1 -> 7411 ; 7,4,2 -> 742
VER=v2; INV_OVR=""
if [[ $INVOCAB -eq 1 ]]; then VER=v22; INV_OVR="dset.in_vocab_only=true"; fi
RESAMPLE=xray_resample_plinder_v2_perelem; SUBSET_N=112000
if [[ $V24 -eq 1 ]]; then
  VER=v24; INV_OVR=""; RESAMPLE=xray_resample_plinder_v2p4_perelem; SUBSET_N=101107
fi
# Optional variable mask-ratio (R2MAE) — composes with per_group (each per-group mask
# is drawn at the batch's variable ratio). Empty min/max → fixed RATIO.
VAR_OVR=""; VARTAG=""
if [[ -n "$RMIN" && -n "$RMAX" ]]; then
  MM=$(awk -v a="$RMIN" -v b="$RMAX" 'BEGIN{printf "%02d%02d", a*100, b*100}')
  VAR_OVR="mae.mask_ratio_min=$RMIN mae.mask_ratio_max=$RMAX"
  VARTAG="_varmask${MM}"
fi
EMATAG=$(printf '%s' "$EMA_DECAY" | tr -d '.')
NAME="260818_cdg_100m_${VER}_g${GTAG}_pergroup_m$(printf '%s' "$RATIO" | tr -d '.')_ema${EMATAG}${VARTAG}"
DRYFLAG=(); [[ $DRY -eq 1 ]] && DRYFLAG=(--dry-run)

# SMOKE: tiny model + few samples + no wandb, to validate the per_group path end-to-end.
SMOKE_OVR=""
if [[ "${SMOKE:-0}" -eq 1 ]]; then
  NAME="smoke_${NAME}"
  SMOKE_OVR="model.dim=128 model.depth=2 model.heads=2 dset.subset_n=16 dset.subset_val_n=8 num_epochs=1 wandb=false"
fi

bash "$SCRIPT_DIR/03_pretrain.sh" --name "$NAME" --gpus "$GPUS" "${DRYFLAG[@]}" -- \
  bsz=4 accum_steps=8 num_workers=6 prefetch_factor=2 num_epochs="$EPOCHS" \
  lr=0.0001 wd=0.05 \
  model.patch_embed_mode=channel_group "model.channel_groups=[$CGRP]" \
  model.dim=640 model.depth=18 model.heads=10 \
  mae.mask_strategy=per_group mae.mask_ratio="$RATIO" mae.ema_decay="$EMA_DECAY" $VAR_OVR \
  dset.ligand_radius=-1 dset.pocket_radius=-1 \
  dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
  dset.resample_dir=dataset/data/pretrain/$RESAMPLE \
  dset.subset_n=$SUBSET_N dset.subset_val_n=100 $INV_OVR \
  wandb=true "wandb_tags=[pretrain,cdg,100m,${VER},pergroup,g${GTAG}${VARTAG}]" \
  $SMOKE_OVR
