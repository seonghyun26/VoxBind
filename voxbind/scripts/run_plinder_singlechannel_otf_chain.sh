#!/bin/bash
# density-only + gradmag-only 40M on PLINDER with ON-THE-FLY (resample) augmentation.
# Single-channel (config_train_density_vit_mae_40m_xray, n_in=1) via dset.resample_dir:
#   density-only  = resampled density as the single channel.
#   gradmag-only  = on-the-fly ‖∇ρ‖ as the single channel (VOXBIND_OTF_GRADMAG_AS_DENSITY=1).
# Each: train -> patch cfg.yaml -> 3-seed frozen probe on canonical 839 (density-only on
# voxels_v5_realdens = canonical real density; gradmag-only on voxels_v5/gradmag). GPU 0-3.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GPUS=0,1,2,3
MASTER=$VOX/log/260619_plinder_singlechannel_otf_chain.log
mkdir -p "$VOX/log"

run_one(){   # $1=name  $2=extra_env("" or VOXBIND_OTF_GRADMAG_AS_DENSITY=1)  $3=probe_noise_dir
  local NAME=$1 EXTRAENV=$2 PNOISE=$3
  local EXP=260619_plinder_${NAME}_otf_vit_mae_40m_pretrain
  local TAG=plinder_${NAME}_otf
  local LOG=$VOX/log/$EXP.log
  local CKPT=exps/$EXP/checkpoint_e0099.pth.tar
  local CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv

  echo "[$(ts)] [$NAME-otf] TRAIN $EXP  env='${EXTRAENV:-none}'" | tee -a "$MASTER"
  env $EXTRAENV CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=4 \
    train_density_vit_mae.py --config-name=config_train_density_vit_mae_40m_xray \
    dset.data_dir="$DATA" dset.crops_dir="" dset.resample_dir="$DATA/xray_resample_plinder" \
    dset.data_file=data_train_plinder.pt dset.normalize=false \
    dset.subset_xray_only=true dset.subset_n=17430 dset.subset_val_n=100 \
    num_workers=8 bsz=32 accum_steps=1 num_epochs=100 \
    "wandb_tags=[pretrain,$NAME,40m,plinder,otf,density_ablation,uniform]" \
    exp_name=$EXP output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
  if [ ! -f "$CKPT" ]; then echo "[$(ts)] [$NAME-otf] ABORT: no e99" | tee -a "$MASTER"; return 1; fi

  $PY/python - "$EXP" >> "$LOG" 2>&1 <<'PYEOF'
import sys
p = f"exps/{sys.argv[1]}/cfg.yaml"; t = open(p).read()
if "n_in_channels" not in t: t = t.replace("model:\n", "model:\n  n_in_channels: 1\n", 1)
if "\ninput_mode:" not in t: t = t.replace("\nmae:\n", "\ninput_mode: density\nwith_gradmag: false\nmae:\n", 1)
open(p, "w").write(t); print("patched cfg.yaml (n_in:1, input_mode:density, with_gradmag:false)")
PYEOF

  echo "[$(ts)] [$NAME-otf] PROBE -> $CSV" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features \
    --condition density_gradmag --voxel_version v5 --epoch 99 --num_workers 0 \
    --exp_dir "exps/$EXP" --tag "$TAG" --noise_voxels_dir "$PNOISE" >> "$LOG" 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv "$CSV" >> "$LOG" 2>&1
  echo "[$(ts)] [$NAME-otf] DONE -> $CSV" | tee -a "$MASTER"
}

run_one densityonly ""                                  "$DATA/pdbbind/voxels_v5_realdens"
run_one gradmagonly "VOXBIND_OTF_GRADMAG_AS_DENSITY=1"   "$DATA/pdbbind/voxels_v5/gradmag"
echo "[$(ts)] === singlechannel OTF chain COMPLETE ===" | tee -a "$MASTER"
