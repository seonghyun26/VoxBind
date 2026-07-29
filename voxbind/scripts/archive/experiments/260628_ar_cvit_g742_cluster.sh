#!/bin/bash
# 260628_ar_cvit_g742_cluster.sh — AUTORESEARCH LOOP, DRAMATIC change #1: cluster masking ALGORITHM.
# New code (make_cluster_mask in models/mae_ops.py, opt-in mae.mask_strategy=cluster — uniform stays
# default, atom_biased/uniform untouched). Instead of many small scattered blocks, mask a FEW LARGE
# CONTIGUOUS atom-anchored clusters (n_seeds seeds ∝ atom mass, then mask the ratio·gb³ blocks closest
# to any seed). Forces the encoder to inpaint whole substructures from long-range context — the MAE
# "large contiguous masks → semantic features" regime, atom-anchored. On the [7,4,2] winner.
# Runs on GPU 1-3 (3-GPU, eff-batch 8×5×3=120 ≈ the base 128) in parallel with the iter-6 finetune (GPU 0).
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=1,2,3; PORT=29601; RID=arG742Clu
EXP=260628_ar_cvit_g742_cluster

echo "===== AR [7,4,2] CLUSTER-MASK START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=3 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=5 \
    model.channel_groups=[7,4,2] mae.mask_strategy=cluster +mae.mask_n_seeds=4 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] CLUSTER-MASK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
