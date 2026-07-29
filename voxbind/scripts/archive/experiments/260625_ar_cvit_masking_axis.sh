#!/bin/bash
# 260625_ar_cvit_masking_axis.sh — map the MASKING-DISTRIBUTION axis, the one lever that
# moved ρ. Context: ~14 trials flat, capacity REFUTED (big768 0.626, −0.011), but
# atom_biased masking nudged ρ to 0.641 (nominal best). The two remaining masking knobs
# probe the same axis, so run them in PARALLEL on the now-free GPUs:
#   lane A (GPU 0-3): mask 0.4   — mae.mask_ratio 0.50→0.40 (less masking)
#   lane B (GPU 4-7): block 4    — mae.block_size 8→4 (finer mask blocks)
# Both on the [7,4,2] base recipe (46.71M, dim512, mask 0.50 default, dw 0.1, 100ep,
# eff-batch 128 = bsz8×accum4×4gpu, compile ON). Per trial: pretrain → frozen probe
# lp_edrscc_v2 (Kd/Ki, n=1320). Beat ρ 0.637 / RMSE 1.353.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
LOG=log

run_trial () {                 # $1=GPUS $2=PORT $3=RID $4=EXP ; rest = hydra overrides
  local GPUS=$1 PORT=$2 RID=$3 EXP=$4; shift 4
  echo "###### [$EXP] PRETRAIN start $(date '+%m-%d %H:%M:%S') GPU=$GPUS :: $* ######"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
      train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
      model.channel_groups=[7,4,2] "$@" \
    && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
    || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; return 1; }
  echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
      -- --require_density \
    && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
  echo "###### [$EXP] result ######"; tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
}

lane_A () { run_trial 0,1,2,3 29571 cvitMA 260624_ar_cvit_mask04 mae.mask_ratio=0.4; }
lane_B () { run_trial 4,5,6,7 29581 cvitMB 260624_ar_cvit_block4 mae.block_size=4; }

echo "===== AR ChannelViT masking-axis START $(date '+%m-%d %H:%M:%S') (A:mask0.4 GPU0-3 | B:block4 GPU4-7) ====="
lane_A > "$LOG/260625_ar_cvit_mask04_A.log" 2>&1 &
PA=$!
lane_B > "$LOG/260625_ar_cvit_block4_B.log" 2>&1 &
PB=$!
wait "$PA"; rA=$?
wait "$PB"; rB=$?
echo "===== AR ChannelViT masking-axis COMPLETE $(date '+%m-%d %H:%M:%S')  laneA=$rA laneB=$rB ====="
