#!/bin/bash
# reprobe_plinder_otf.sh — salvage the feature-extraction + probe step for the
# PLINDER OTF size-sweep fractions whose probe crashed in the main chain.
#
# Root cause of the crash: run_plinder_otf_sizesweep_chain.sh launched the features
# step with ZERO grace the instant the 4-proc DDP torchrun pretrain exited; the
# half-released GPU-0 CUDA context failed the DataLoader's first pinned-memory alloc
# (CUDA error: invalid argument in the pin_memory thread). The pretrains all succeeded
# (e99 saved) — only the post-torchrun features launch was racy. p01 survived because
# its tiny pretrain tore down fast.
#
# Fix here: wait for the main chain to fully exit (no torchrun on GPU 0), add a grace
# delay so the driver settles, then run features+probe per fraction on the now-idle GPU.
# Idempotent: skips any fraction whose CSV already exists; truthful per-fraction logging.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
cd "$VOX" || exit 1

CHAIN_PID=${CHAIN_PID:-3877769}     # main sweep chain; wait for it to exit first
GPU=${GPU:-0}
GRACE=${GRACE:-45}
MASTER=$VOX/log/260617_plinder_otf_reprobe.log
ts(){ date "+%F %T"; }

mkdir -p "$VOX/log"
echo "[$(ts)] === PLINDER OTF re-probe START (gpu=$GPU, grace=${GRACE}s) ===" | tee -a "$MASTER"

# 1. wait for the main chain to fully exit so no torchrun is tearing down GPU $GPU
if [ -n "$CHAIN_PID" ]; then
    while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 20; done
    echo "[$(ts)] main chain ($CHAIN_PID) exited" | tee -a "$MASTER"
fi
sleep "$GRACE"   # let the driver settle (the actual fix for the dirty-context race)

# 2. re-run features+probe for each missing fraction on the clean GPU
for entry in "05:872" "10:1743" "25:4358" "50:8715"; do
    pp=${entry%%:*}; SUBN=${entry##*:}
    EXP=260617_plinder_otf_cdg_invfreq_p${pp}_pretrain
    TAG=plinder_otf_p${pp}
    CKPT=exps/$EXP/checkpoint_e0099.pth.tar
    CSV=dataset/data/pdbbind/probe_results_e99_v5_plinder_otf_p${pp}.csv
    LOG=$VOX/log/${EXP}.log

    if [ -f "$CSV" ]; then
        echo "[$(ts)] [p${pp}] CSV already present — skip" | tee -a "$MASTER"; continue
    fi
    if [ ! -f "$CKPT" ]; then
        echo "[$(ts)] [p${pp}] e99 MISSING — cannot probe, skip" | tee -a "$MASTER"; continue
    fi

    echo "[$(ts)] [p${pp}] re-features (subset_n=$SUBN) on GPU $GPU -> $LOG" | tee -a "$MASTER"
    CUDA_VISIBLE_DEVICES=$GPU $PY/python dataset/01c_pdbbind_probe.py features \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 \
        --atom_source ligvdw --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1
    feat=dataset/data/pdbbind/features/atomblob_density_gradmag_e99_v5_${TAG}.pt
    if [ ! -f "$feat" ]; then
        echo "[$(ts)] [p${pp}] features FAILED (no $feat) — see $LOG" | tee -a "$MASTER"; continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU $PY/python dataset/01c_pdbbind_probe.py probe \
        --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
        --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
        --out_csv "$CSV" >> "$LOG" 2>&1
    if [ -f "$CSV" ]; then
        echo "[$(ts)] [p${pp}] DONE -> $CSV" | tee -a "$MASTER"
    else
        echo "[$(ts)] [p${pp}] probe FAILED (no CSV) — see $LOG" | tee -a "$MASTER"
    fi
done

echo "[$(ts)] === PLINDER OTF re-probe COMPLETE ===" | tee -a "$MASTER"