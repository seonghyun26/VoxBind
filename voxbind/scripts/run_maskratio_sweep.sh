#!/usr/bin/env bash
# Champion mask-ratio sweep — build a mask-ratio-ONLY ensemble family on v2.
#
# Everything is pinned to the champion recipe (260705_ar_cvit_100m_v2_mask075):
#   v2 data · 100M dim640/depth18/heads10 · channel_group [7,4,2] · vdW radii
#   · UNIFORM masking · 50 ep · lr 1e-4 wd 0.05.  The ONLY thing that varies
#   across runs is mae.mask_ratio, so any later ensemble gain is pure
#   mask-ratio diversity (validated on the v3 family: +0.007 test ρ).
#
#   # dry-run (print resolved commands, launch nothing):
#   bash scripts/run_maskratio_sweep.sh --gpus 0-3 --ratios "0.60 0.85 0.90" --dry-run
#   # smoke (1 ep, 64 samples, no wandb — validates the recipe trains):
#   bash scripts/run_maskratio_sweep.sh --gpus 7 --ratios 0.90 --smoke
#   # production chain on the pretraining lane:
#   bash scripts/run_maskratio_sweep.sh --gpus 0-3 --ratios "0.60 0.85 0.90"
#
# Runs are chained: each waits for the previous process to exit before the next
# launches. Children are started with `setsid` so stopping this orchestrator
# (Ctrl-C / TaskStop) never SIGTERM-cascades into an in-flight training.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0-3"; RATIOS="0.60 0.85 0.90"; EPOCHS=50; DRY=0; SMOKE=0
while [[ $# -gt 0 ]]; do case "$1" in
  --gpus)    GPUS="${2:?}"; shift 2;;
  --ratios)  RATIOS="${2:?}"; shift 2;;
  --epochs)  EPOCHS="${2:?}"; shift 2;;
  --dry-run) DRY=1; shift;;
  --smoke)   SMOKE=1; shift;;
  *) echo "unknown arg '$1'"; exit 1;;
esac; done

# Champion recipe, everything fixed except mae.mask_ratio (arg $1).
champion_overrides () {
  local R="$1"
  printf '%s ' \
    bsz=4 accum_steps=8 num_workers=6 prefetch_factor=2 num_epochs="$EPOCHS" \
    lr=0.0001 wd=0.05 \
    model.patch_embed_mode=channel_group 'model.channel_groups=[7,4,2]' \
    model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_strategy=uniform mae.mask_ratio="$R" \
    dset.ligand_radius=-1 dset.pocket_radius=-1 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100
}

for R in $RATIOS; do
  MM=$(awk -v r="$R" 'BEGIN{printf "m%03d", r*100}')          # 0.60 -> m060
  NAME="260813_cdg_100m_v2_${MM}"
  OV="$(champion_overrides "$R") wandb_tags=[pretrain,cdg,100m,v2,masksweep,${MM}]"

  if [[ $SMOKE -eq 1 ]]; then
    NAME="260813_smoke_${MM}"
    OV="$(champion_overrides "$R") num_epochs=1 dset.subset_n=64 dset.subset_val_n=16 wandb=false"
    echo ">> SMOKE $NAME on GPU $GPUS"
    bash "$SCRIPT_DIR/03_pretrain.sh" --name "$NAME" --gpus "$GPUS" -- $OV
    echo ">> smoke exit=$?"; exit $?
  fi

  if [[ $DRY -eq 1 ]]; then
    echo "### $NAME (mask_ratio=$R) on GPU $GPUS"
    bash "$SCRIPT_DIR/03_pretrain.sh" --name "$NAME" --gpus "$GPUS" --dry-run -- $OV
    echo; continue
  fi

  echo ">> LAUNCH $NAME (mask_ratio=$R) on GPU $GPUS"
  setsid bash "$SCRIPT_DIR/03_pretrain.sh" --name "$NAME" --gpus "$GPUS" -- $OV &
  sleep 120
  while pgrep -f "$NAME" >/dev/null 2>&1; do sleep 120; done
  echo ">> $NAME finished"
done
echo ">> sweep complete: $RATIOS"
