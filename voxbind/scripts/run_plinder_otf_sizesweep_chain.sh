#!/bin/bash
# run_plinder_otf_sizesweep_chain.sh — pretrain-corpus SIZE sweep for the
# on-the-fly (resample-aug) C+D+G encoder on the PLINDER ligand-matched density corpus.
#
# Same encoder / recipe as the PLINDER C+D+G row (config
# config_train_atomblob_density_gradmag_vit_mae_40m_invfreq, n_in=13, ligvdw, invfreq,
# 100 epochs) and the SAME on-the-fly density path as the v5 OTF run
# (DatasetCrossDockedDensity via dset.resample_dir) — ONLY the pre-training corpus SIZE
# changes. Each run trains on a random subset of the PLINDER pool, then a 3-seed frozen
# probe on the canonical 2172/480/839 affinity split reports test Spearman.
#
# Random partition = ZERO code change: the corpus is pre-shuffled (seed 1234) at load and
# the resample manifest is stored in that same order, so dset.subset_n=N selects a
# uniformly-random subset (the 5 fractions are NESTED: 1% ⊂ 5% ⊂ … ⊂ 50%).
#
# Fixed 100 epochs for every fraction (data-scaling at fixed training budget). Ascending
# order so the cheap small runs validate the full pretrain→features→probe plumbing first.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
cd "$VOX" || exit 1

RESAMPLE=$DATA/xray_resample_plinder
DATA_FILE=data_train_plinder.pt
MASTER=$VOX/log/260617_plinder_otf_sizesweep_chain.log
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PROBE_GPU=${GPUS%%,*}
FRACS="0.01 0.05 0.10 0.25 0.50"   # ascending: 1% 5% 10% 25% 50%
ts(){ date "+%F %T"; }

mkdir -p "$VOX/log"
echo "[$(ts)] === PLINDER OTF C+D+G size sweep START | GPUs=$GPUS | fracs=$FRACS ===" | tee -a "$MASTER"

# gate: PLINDER OTF corpus materialized
if [ ! -f "$RESAMPLE/resample.json" ] || [ ! -f "$RESAMPLE/train_manifest.npz" ] || [ ! -f "$DATA/$DATA_FILE" ]; then
    echo "[$(ts)] ABORT: PLINDER resample corpus missing ($RESAMPLE / $DATA_FILE)" | tee -a "$MASTER"; exit 1
fi

# train pool size: N available crops, reserve 100 for val → TRAIN_POOL = N-100
N=$($PY/python -c "import numpy as np; print(int(np.load('$RESAMPLE/train_available.npy').shape[0]))")
TRAIN_POOL=$((N - 100))
echo "[$(ts)] PLINDER pool N=$N → train_pool=$TRAIN_POOL (val=100)" | tee -a "$MASTER"

for FRAC in $FRACS; do
    PCT=$($PY/python -c "print(f'{int(round($FRAC*100)):02d}')")
    SUBN=$($PY/python -c "print(int(round($FRAC*$TRAIN_POOL)))")
    EXP=260617_plinder_otf_cdg_invfreq_p${PCT}_pretrain
    TAG=plinder_otf_p${PCT}
    LOG=$VOX/log/$EXP.log
    CSV=dataset/data/pdbbind/probe_results_e99_v5_plinder_otf_p${PCT}.csv

    if [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
        echo "[$(ts)] [p${PCT}] e99 already exists — skipping pretrain (subset_n=$SUBN)" | tee -a "$MASTER"
    else
        echo "[$(ts)] [p${PCT}] pretrain subset_n=$SUBN (frac=$FRAC of $TRAIN_POOL) on GPU $GPUS -> $LOG" | tee -a "$MASTER"
        CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --standalone --nproc_per_node=$NPROC \
            train_density_vit_mae.py \
            --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
            dset.data_dir="$DATA" \
            dset.crops_dir="" \
            dset.resample_dir="$RESAMPLE" \
            dset.data_file=$DATA_FILE \
            dset.subset_n=$SUBN dset.subset_val_n=100 dset.subset_xray_only=true \
            num_epochs=100 bsz=32 accum_steps=1 \
            wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,otf,ligvdw,clip30,sizesweep,p${PCT}]" \
            exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
        if [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
            echo "[$(ts)] [p${PCT}] ABORT: training produced no e99 (see $LOG)" | tee -a "$MASTER"; continue
        fi
    fi

    # frozen-encoder probe on canonical 2172/480/839 (PDBbind v5 voxels)
    echo "[$(ts)] [p${PCT}] features + 3-seed probe -> $CSV" | tee -a "$MASTER"
    CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py features \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
        --atom_source ligvdw --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$PROBE_GPU $PY/python dataset/01c_pdbbind_probe.py probe \
        --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
        --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
        --out_csv "$CSV" >> "$LOG" 2>&1
    echo "[$(ts)] [p${PCT}] DONE subset_n=$SUBN -> $CSV" | tee -a "$MASTER"
done

echo "[$(ts)] === PLINDER OTF C+D+G size sweep COMPLETE ===" | tee -a "$MASTER"
