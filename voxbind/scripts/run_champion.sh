#!/usr/bin/env bash
# Fixed-mask CHAMPION-recipe pretraining (the 260705 reference recipe), optionally on
# clean v2.2 (in-vocab) data. Sibling of run_varmask.sh but FIXED uniform mask 0.75.
#
# Champion 260705 recipe pinned: v2 · 100M dim640/depth18/heads10 · channel_group
# [7,4,2] · vdW radii · uniform mask 0.75 · 50 ep · lr1e-4/wd0.05. The champion was
# trained on DIRTY v2 (7 lig channels → 5,709 OOV-ligand input/target mismatch);
# --in_vocab reruns the SAME recipe on clean v2.2 (dset.in_vocab_only) to test whether
# the ~+0.01 v2→v2.2 cleaning gain (seen on varmask & [7,4,1,1]) lifts the champion too.
#
#   bash scripts/run_champion.sh --gpus 0-3               # champion on v2 (reference replay)
#   bash scripts/run_champion.sh --gpus 0-3 --in_vocab    # champion on clean v2.2
#   bash scripts/run_champion.sh --gpus 0-3 --v24         # champion on CASF-clean v2.4
#   bash scripts/run_champion.sh --gpus 7 --in_vocab --dry-run
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0-3"; EPOCHS=50; DRY=0; INVOCAB=0; V24=0; CGRP="7,4,2"; NOGRAD=0
while [[ $# -gt 0 ]]; do case "$1" in
  --gpus)       GPUS="${2:?}"; shift 2;;
  --epochs)     EPOCHS="${2:?}"; shift 2;;
  --groups)     CGRP="${2:?}"; shift 2;;   # ChannelViT groups WITHOUT brackets, e.g. 7,4,2 or 7,4,1
  --no_gradmag) NOGRAD=1; shift;;          # drop gradmag (CD not CDG): with_gradmag=false, use [7,4,1]
  --in_vocab)   INVOCAB=1; shift;;         # v2.2: drop out-of-vocab-ligand complexes at load time
  --v24)        V24=1; shift;;             # v2.4: CASF-2016 ID30-decontaminated view of v2 (101,207)
  --dry-run)    DRY=1; shift;;
  *) echo "unknown arg '$1'"; exit 1;;
esac; done

# Data-version overrides. v2 = full 113,874 (default). v2.2 = in-vocab load-time filter.
# v2.4 = OTF on the CASF-ID30-clean resample dir (v2 tuples reused; manifest selects 101,207;
# box is a hard-link to v2, position-identical). See dataset/plinder/validate_v2p4.py.
VER=v2; INV_OVR=""
RESAMPLE=xray_resample_plinder_v2_perelem; SUBSET_N=112000
if [[ $INVOCAB -eq 1 ]]; then VER=v22; INV_OVR="dset.in_vocab_only=true"; fi
if [[ $V24 -eq 1 ]];    then VER=v24; RESAMPLE=xray_resample_plinder_v2p4_perelem; SUBSET_N=101107; fi

# Channel grouping + optional no-gradmag. Default = champion [7,4,2] CDG.
# n_in_channels MUST equal sum(channel_groups) (asserted at build); with_gradmag lives under
# model.* (the trainer mirrors it to top-level, so override model.with_gradmag, NOT top-level).
GTAG=$(echo "$CGRP" | tr -cd '0-9')             # 7,4,2 -> 742 ; 7,4,1 -> 741
NIN=$(echo "$CGRP" | awk -F, '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')   # sum of groups
GPART=""; [[ "$GTAG" != "742" ]] && GPART="_g${GTAG}"
CDG="cdg"; GRAD_OVR="model.n_in_channels=$NIN"
[[ $NOGRAD -eq 1 ]] && { CDG="cd"; GRAD_OVR="$GRAD_OVR model.with_gradmag=false mae.gradmag_reconstruct=false"; }
NAME="260813_${CDG}_100m_${VER}${GPART}_mask075"
DRYFLAG=(); [[ $DRY -eq 1 ]] && DRYFLAG=(--dry-run)

# SMOKE: tiny model + few samples + no wandb, to validate the data path end-to-end.
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
  mae.mask_strategy=uniform mae.mask_ratio=0.75 $GRAD_OVR \
  dset.ligand_radius=-1 dset.pocket_radius=-1 \
  dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
  dset.resample_dir=dataset/data/pretrain/$RESAMPLE \
  dset.subset_n=$SUBSET_N dset.subset_val_n=100 $INV_OVR \
  wandb=true "wandb_tags=[pretrain,${CDG},100m,${VER},champion,mask075,g${GTAG}]" \
  $SMOKE_OVR
