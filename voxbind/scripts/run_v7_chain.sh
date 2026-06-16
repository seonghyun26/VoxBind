#!/bin/bash
# run_v7_chain.sh — COMBINED ligand-matched corpus point for the density story.
# Same combined C+D+G encoder / recipe as 260606 / v6 (config
# config_train_atomblob_density_gradmag_vit_mae_40m_invfreq, n_in=13, ligvdw,
# invfreq, bsz=8/accum=3) — ONLY the pretraining data changes to v7:
#   v7 = CrossDocked v6 (5,270 ligand-matched) ∪ PDBbind-2020-train matched (~2,704),
#   one unified corpus (data_train_v7.pt + xray_crops_aligned_v7) where density
#   corresponds to BOTH pocket and ligand. ~3x the v6 matched data.
# Tests whether a LARGER ligand-matched density corpus improves the affinity signal
# over v6 (small matched) and the contaminated v5.
#
# data_train_v7.pt is loaded via dset.data_file; subset_n is sized to the v7 pool at
# runtime. REAL density + on-the-fly gradmag. Pretrain on $GPUS, then 3-seed frozen
# probe on the canonical 2172/480/839 affinity split (PDBbind voxels stay v5).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
EXP=260615_atomblob_density_gradmag_vit_mae_40m_invfreq_v7_pretrain
V7=$DATA/xray_crops_aligned_v7
LOG=$VOX/log/$EXP.log
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PROBE_GPU=${GPUS%%,*}
ts(){ date "+%F %T"; }

# gate: v7 corpus materialized
if [ ! -d "$V7/train" ] || [ ! -f "$V7/train_available.npy" ] || [ ! -f "$DATA/data_train_v7.pt" ]; then
    echo "[$(ts)] ABORT: v7 corpus not materialized ($V7 / data_train_v7.pt)" >> "$LOG"; exit 1
fi

# size the subset to the v7 pool: N train crops → subset_n = N-100, val = 100
N=$($PY/python -c "import numpy as np; print(int(np.load('$V7/train_available.npy').shape[0]))")
SUBN=$((N - 100))
echo "[$(ts)] v7 pool N=$N → subset_n=$SUBN subset_val_n=100" >> "$LOG"

# 1. pretrain — C+D+G config, data=v7 (data_file override), subset sized to v7
echo "[$(ts)] launching v7 combined C+D+G pretrain on GPU $GPUS" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=$NPROC \
    train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
    dset.data_dir="$DATA" \
    dset.crops_dir="$V7" \
    dset.data_file=data_train_v7.pt \
    dset.subset_n=$SUBN dset.subset_val_n=100 \
    bsz=8 accum_steps=3 \
    wandb_tags='[pretrain,atomblob_density_gradmag,40m,invfreq,v7,ligvdw,clip30,crossdocked_xray,pdbbind_train,ligand_matched,combined]' \
    exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] ABORT: training produced no e99" >> "$LOG"; exit 1
fi

# 2. probe: REAL PDBbind v5 voxels → v7-pretrained encoder (canonical 2172/480/839)
CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py features \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --atom_source ligvdw --exp_dir "exps/$EXP" --tag v7combined >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag v7combined --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv dataset/data/pdbbind/probe_results_e99_v5_v7_combined.csv >> "$LOG" 2>&1
echo "[$(ts)] v7 combined chain DONE -> probe_results_e99_v5_v7_combined.csv" >> "$LOG"
