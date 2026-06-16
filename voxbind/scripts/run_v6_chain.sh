#!/bin/bash
# run_v6_chain.sh — LIGAND-MATCHED density point for the density-ablation story.
# Same combined C+D+G encoder as 260606 / noise / zero controls: dedicated config
# config_train_atomblob_density_gradmag_vit_mae_40m_invfreq (head_style=patch_mlp,
# n_in=13, ligvdw, invfreq), bsz=8/accum=3 — IDENTICAL recipe, ONLY the CrossDocked
# density data is swapped to v6 (xray_crops_aligned_v6): the tt_min ligand-MATCHED
# subset where the experimental density corresponds to BOTH pocket and ligand
# (vs v5, where 86% are cross-docked and the ligand-region density is a mismatch).
# Tests whether density that genuinely matches the ligand carries real signal —
# the open question from the "density = capacity, not content" finding.
#
# v6 has only 5,270 x-ray-available train crops (vs v5's 78,528), so subset_n is
# resized to 5170 train + 100 val (uses all 5,270). REAL density + on-the-fly gradmag
# (no noise/zero override). Pretrain on $GPUS, then 3-seed frozen probe on the
# canonical 2172/480/839 affinity split (PDBbind voxels stay v5 — same as every probe).
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1
EXP=260615_atomblob_density_gradmag_vit_mae_40m_invfreq_v6_pretrain
V6=$DATA/xray_crops_aligned_v6
LOG=$VOX/log/$EXP.log
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PROBE_GPU=${GPUS%%,*}
ts(){ date "+%F %T"; }

# gate: v6 crops materialized (ligand-matched density + availability mask)
if [ ! -d "$V6/train" ] || [ ! -f "$V6/train_available.npy" ]; then
    echo "[$(ts)] ABORT: v6 crops not materialized ($V6)" >> "$LOG"; exit 1
fi

# 1. pretrain — dedicated C+D+G config (== 260606), data=v6, subset resized to 5,270
echo "[$(ts)] launching v6 ligand-matched C+D+G pretrain on GPU $GPUS" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=$NPROC \
    train_density_vit_mae.py \
    --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
    dset.data_dir="$DATA" \
    dset.crops_dir="$V6" \
    dset.subset_n=5170 dset.subset_val_n=100 \
    bsz=8 accum_steps=3 \
    wandb_tags='[pretrain,atomblob_density_gradmag,40m,invfreq,v6,ligvdw,clip30,crossdocked_xray,ligand_matched,density_ablation]' \
    exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] ABORT: training produced no e99" >> "$LOG"; exit 1
fi

# 2. probe: REAL PDBbind v5 voxels → v6-pretrained encoder (canonical 2172/480/839)
CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py features \
    --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
    --atom_source ligvdw --exp_dir "exps/$EXP" --tag v6match >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py probe \
    --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
    --feature_tag v6match --exp_dir "exps/$EXP" --allow_stale_features \
    --out_csv dataset/data/pdbbind/probe_results_e99_v5_v6_ligandmatched.csv >> "$LOG" 2>&1
echo "[$(ts)] v6 ligand-matched chain DONE -> probe_results_e99_v5_v6_ligandmatched.csv" >> "$LOG"
