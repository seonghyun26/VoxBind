#!/bin/bash
# 260628_ar_cvit_g742_atombias_s1.sh — AUTORESEARCH LOOP iter 4 (GPU 0-3).
# The campaign's ONLY nominal beat was atom-bias ([7,4,2] mask_strategy=atom_biased,
# seed42) ρ 0.641 — but within probe-seed noise of the base 0.637. This re-runs the
# SAME config with a different PRETRAIN seed (seed=1) to test whether 0.641 replicates
# (real, if small, win) or regresses to ~0.637 (pretrain-seed luck). Closes the campaign's
# one open "is there a winner?" question. Same recipe otherwise. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29594; RID=arG742ABs1
EXP=260628_ar_cvit_g742_atombias_s1

echo "===== AR [7,4,2] atom-bias SEED=1 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 seed=1 \
    model.channel_groups=[7,4,2] mae.mask_strategy=atom_biased \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] atom-bias SEED=1 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
