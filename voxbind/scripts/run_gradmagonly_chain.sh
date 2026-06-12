#!/bin/bash
# run_gradmagonly_chain.sh — gradmag-only ablation = density-only encoder trained on
# real ‖∇ρ‖-as-density crops, then frozen-probed on the matched 839 split. Reuses the
# verified density-only path (no training-code change). Invoked by the watcher AFTER
# the noise-control probe finishes (so GPUs 4-7 are free). Blocks ~9.5 h (train) + ~10 m (probe).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
cd "$VOX" || exit 1
EXP=260612_gradmagonly_density_xray_vit_mae_40m_v5_pretrain
GMAG=$VOX/dataset/data/xray_crops_aligned_v5_gradmag
PGMAG=$VOX/dataset/data/pdbbind/voxels_v5_gradmag
LOG=$VOX/log/$EXP.log
ts(){ date "+%F %T"; }

# gate: gradmag crops must be fully materialized
if [ ! -f "$GMAG/train/078527.npy" ] || [ ! -d "$PGMAG/density" ]; then
    echo "[$(ts)] ABORT: gradmag crops not ready ($GMAG / $PGMAG)" >> "$LOG"; exit 1
fi

# 1. train gradmag-only encoder (density-only path on gradmag-as-density crops), 4-7, blocks to e99
echo "[$(ts)] launching gradmag-only training on 4-7" >> "$LOG"
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_density_vit_mae.py --config-name=config_train_density_vit_mae_40m_xray \
    dset.data_dir=dataset/data dset.crops_dir="$GMAG" dset.normalize=false \
    dset.subset_xray_only=true dset.subset_n=78428 dset.subset_val_n=100 \
    bsz=8 accum_steps=3 num_epochs=100 \
    'wandb_tags=[pretrain,gradmag_only,40m,v5,density_ablation,crossdocked_xray]' \
    exp_name=$EXP output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] ABORT: training produced no e99" >> "$LOG"; exit 1
fi

# 2. patch cfg so the probe's load_encoder can build the 1-ch density encoder
$PY - "$EXP" <<'PYEOF' >> "$LOG" 2>&1
import sys
p = f"exps/{sys.argv[1]}/cfg.yaml"
t = open(p).read()
if "n_in_channels" not in t:
    t = t.replace("model:\n", "model:\n  n_in_channels: 1\n", 1)
if "\ninput_mode:" not in t:
    t = t.replace("\nmae:\n", "\ninput_mode: density\nwith_gradmag: false\nmae:\n", 1)
open(p, "w").write(t)
print("patched cfg.yaml (n_in_channels:1, input_mode:density, with_gradmag:false)")
PYEOF

# 3. probe: density-only encoder, fed gradmag-as-density at probe time too (matched 839 split)
CUDA_VISIBLE_DEVICES=4 $PY dataset/01c_pdbbind_probe.py features \
    --condition density_gradmag --voxel_version v5 --epoch 99 \
    --exp_dir "exps/$EXP" --tag gradmagonly --noise_voxels_dir "$PGMAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=4 $PY dataset/01c_pdbbind_probe.py probe \
    --conditions density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag gradmagonly --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv dataset/data/pdbbind/probe_results_e99_v5_filtered_gradmagonly.csv >> "$LOG" 2>&1
echo "[$(ts)] gradmag-only chain DONE -> probe_results_e99_v5_filtered_gradmagonly.csv" >> "$LOG"
