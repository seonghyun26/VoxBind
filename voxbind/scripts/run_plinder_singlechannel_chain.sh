#!/bin/bash
# density-only + gradmag-only 40M on PLINDER (frozen crops), single-channel
# (config_train_density_vit_mae_40m_xray, n_in=1, uniform — no atom-type channels to
# weight), "like before". density-only = PLINDER density crops; gradmag-only = PLINDER
# gradmag crops (00f). Each: train -> patch cfg.yaml (n_in=1/input_mode=density) -> 3-seed
# frozen affinity probe on canonical 839 (density-only on voxels_v5/density; gradmag-only
# on voxels_v5/gradmag via --noise_voxels_dir). Runs on GPU 0-3 after the seed0 run frees it.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GPUS=0,1,2,3; GATEGPU=0
MASTER=$VOX/log/260618_plinder_singlechannel_chain.log
mkdir -p "$VOX/log"

# ── gate: GPU 0-3 idle (seed0 train + probe done) ──
echo "[$(ts)] [singlechannel] waiting for GPU $GATEGPU to free (after seed0) ..." | tee -a "$MASTER"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done
echo "[$(ts)] [singlechannel] GPU free -> starting runs" | tee -a "$MASTER"

run_one(){   # $1=name(densityonly|gradmagonly)  $2=crops_dir  $3=probe_noise_dir("" = use voxels_v5/density)
  local NAME=$1 CROPS=$2 PNOISE=$3
  local EXP=260618_plinder_${NAME}_vit_mae_40m_pretrain
  local TAG=plinder_${NAME}
  local LOG=$VOX/log/$EXP.log
  local CKPT=exps/$EXP/checkpoint_e0099.pth.tar
  local CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv

  echo "[$(ts)] [$NAME] TRAIN $EXP  crops=$CROPS" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
    train_density_vit_mae.py --config-name=config_train_density_vit_mae_40m_xray \
    dset.data_dir="$DATA" dset.crops_dir="$CROPS" dset.normalize=false \
    dset.data_file=data_train_plinder.pt \
    dset.subset_xray_only=true dset.subset_n=17430 dset.subset_val_n=100 \
    num_workers=8 bsz=32 accum_steps=1 num_epochs=100 \
    "wandb_tags=[pretrain,$NAME,40m,plinder,density_ablation,uniform]" \
    exp_name=$EXP output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
  if [ ! -f "$CKPT" ]; then echo "[$(ts)] [$NAME] ABORT: no e99" | tee -a "$MASTER"; return 1; fi

  # patch cfg.yaml so downstream loaders rebuild the 1-channel density encoder
  $PY/python - "$EXP" >> "$LOG" 2>&1 <<'PYEOF'
import sys
p = f"exps/{sys.argv[1]}/cfg.yaml"; t = open(p).read()
if "n_in_channels" not in t: t = t.replace("model:\n", "model:\n  n_in_channels: 1\n", 1)
if "\ninput_mode:" not in t: t = t.replace("\nmae:\n", "\ninput_mode: density\nwith_gradmag: false\nmae:\n", 1)
open(p, "w").write(t); print("patched cfg.yaml (n_in:1, input_mode:density, with_gradmag:false)")
PYEOF

  echo "[$(ts)] [$NAME] PROBE -> $CSV" | tee -a "$MASTER"
  local NOISE_ARG=""; [ -n "$PNOISE" ] && NOISE_ARG="--noise_voxels_dir $PNOISE"
  CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
    --condition density_gradmag --voxel_version v5 --epoch 99 --num_workers 0 \
    --exp_dir "exps/$EXP" --tag "$TAG" $NOISE_ARG >> "$LOG" 2>&1 \
  && CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv "$CSV" >> "$LOG" 2>&1
  echo "[$(ts)] [$NAME] DONE -> $CSV" | tee -a "$MASTER"
}

run_one densityonly "$DATA/xray_crops_aligned_plinder" ""
# gradmag-only needs the PLINDER gradmag train crops (00f); wait if still building
while [ "$(ls $DATA/xray_crops_aligned_plinder/gradmag/train 2>/dev/null | wc -l)" -lt 17000 ]; do
  echo "[$(ts)] [gradmagonly] waiting for PLINDER gradmag crops (00f) ..." | tee -a "$MASTER"; sleep 120; done
run_one gradmagonly "$DATA/xray_crops_aligned_plinder/gradmag" "$DATA/pdbbind/voxels_v5/gradmag"
echo "[$(ts)] === singlechannel chain COMPLETE ===" | tee -a "$MASTER"
