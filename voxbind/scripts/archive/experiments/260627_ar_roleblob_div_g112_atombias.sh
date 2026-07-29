#!/bin/bash
# 260627_ar_roleblob_div_g112_atombias.sh — AUTORESEARCH LOOP iter 2 (GPU 0-3).
# Prev iter: roleblob-diverse [1,1,2] = 0.592 (+0.007 vs [1,1,1,1] 0.585) → the D+G-grouping
# win TRANSFERS to the diverse role rep (same +0.007 it gave atomblob). Role-split still
# caps ~0.59 < atomblob 0.637. Now stack the campaign's OTHER confirmed lever — atom_biased
# masking (the lone nominal atomblob beat, +0.011) — on the best role config to find its
# ceiling. One change from [1,1,2] (mask_strategy uniform→atom_biased) → isolates atom-bias
# transfer. Diverse PLINDER-v2 data, 4ch role, 100ep, eff-batch128, compile ON. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29592; RID=arDivG112AB
EXP=260627_ar_roleblob_div_g112_atombias

echo "===== AR roleblob-diverse [1,1,2]+atom-bias START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[1,1,2] mae.mask_strategy=atom_biased \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition roleblob_density_gradmag_channelvit \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR roleblob-diverse [1,1,2]+atom-bias COMPLETE $(date '+%m-%d %H:%M:%S') ====="
