#!/bin/bash
# 260627_ar_roleblob_div_g112.sh — AUTORESEARCH LOOP iter (GPU 0-3).
# Hypothesis: the campaign's one structural win was "group density+gradmag together"
# ([7,4,2] 0.637 > [7,4,1,1] 0.630). The diverse-atom PLINDER-v2 role representation
# (data_train_plinder_diverse.pt; all heavy atoms kept via per-atom vdW size) was run
# only as [1,1,1,1] → ρ 0.585 (already beats non-diverse roleblob 0.554, but < atomblob).
# Apply the grouping win to it: channel_groups [1,1,1,1] → [1,1,2] (lig / poc / D+G).
# Tests whether the D+G-grouping benefit transfers to the diverse role rep + gives that
# arm its best shot. Same recipe otherwise (4ch role, mask 0.50, uniform wt, 100ep,
# eff-batch128, compile ON). Frozen probe → lp_edrscc_v2 (cond roleblob_density_gradmag_channelvit).
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29591; RID=arDivG112
EXP=260627_ar_roleblob_div_g112

echo "===== AR roleblob-diverse [1,1,2] START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[1,1,2] \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition roleblob_density_gradmag_channelvit \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR roleblob-diverse [1,1,2] COMPLETE $(date '+%m-%d %H:%M:%S') ====="
