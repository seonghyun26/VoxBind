#!/bin/bash
# 260628_ar_cvit_g742_base_s1.sh — AUTORESEARCH LOOP iter 5 (GPU 0-3). MEASURE THE SEED-NOISE FLOOR.
# iter4 showed atom-bias 0.641 (seed42) → 0.619 (seed1): the lone nominal beat did NOT replicate,
# and the pretrain-seed spread (~0.022) exceeds every knob effect. To make "knob diffs are within
# seed noise" rigorous, re-run the BASE [7,4,2] (uniform mask, the reference) with seed=1 and compare
# to its seed42 0.637 → quantifies the base's own pretrain-seed variance. Completes a clean
# base/atom-bias × seed42/seed1 grid. Same recipe otherwise. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29595; RID=arG742Bs1
EXP=260628_ar_cvit_g742_base_s1

echo "===== AR [7,4,2] BASE SEED=1 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 seed=1 \
    model.channel_groups=[7,4,2] \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] BASE SEED=1 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
