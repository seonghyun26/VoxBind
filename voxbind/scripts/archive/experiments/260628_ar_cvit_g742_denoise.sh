#!/bin/bash
# 260628_ar_cvit_g742_denoise.sh — AUTORESEARCH LOOP iter 3 (GPU 0-3). NEW AXIS: SSL objective.
# The whole campaign used masked-reconstruction MAE. Phase 4 closed the role-rep arm
# (diverse atoms help +0.031, D+G-grouping transfers +0.007 → 0.592, atom-bias does NOT
# transfer −0.031; role caps <0.637). All knob/grouping/capacity/PE/mask levers flat.
# The one untouched FRONTIER lever is the pretext itself: try a DENOISING objective on the
# [7,4,2] winner — mae.pretext_style=denoise sets mask=ALL and corrupts density+gradmag with
# N(0,1)·sigma_noise, so the encoder reconstructs the CLEAN density from a noisy one (a
# density-denoising AE, directly targeting density-reading) instead of inpainting holes.
# Same element atomblob [7,4,2], 100ep, eff-batch128, compile ON. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29593; RID=arG742Den
EXP=260628_ar_cvit_g742_denoise

echo "===== AR [7,4,2] DENOISE pretext START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[7,4,2] mae.pretext_style=denoise mae.sigma_noise=0.5 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] DENOISE COMPLETE $(date '+%m-%d %H:%M:%S') ====="
