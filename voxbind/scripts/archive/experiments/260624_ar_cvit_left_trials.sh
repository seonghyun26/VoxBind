#!/bin/bash
# 260624_ar_cvit_left_trials.sh — run the 4 remaining ChannelViT C+D+G [7,4,2]
# auto-research trials (the ones the 260623 chain had queued but never started)
# in 2 PARALLEL lanes:  lane A = GPU 0-3, lane B = GPU 4-7.
# Each lane runs its 2 trials sequentially; the two lanes run concurrently.
# Per trial:  torchrun 4-GPU pretrain (40M ChannelViT, 100ep) -> frozen probe on
# lp_edrscc_v2 (Kd/Ki, n=1320).  Base grouping [7,4,2] = the c1 winner (rho 0.637).
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
LOG=log

run_trial () {                 # $1=GPUS $2=PORT $3=RDZVID $4=EXP ; rest = hydra overrides
  local GPUS=$1 PORT=$2 RID=$3 EXP=$4; shift 4
  echo "###### [$EXP] PRETRAIN start $(date '+%m-%d %H:%M:%S')  GPU=$GPUS ######"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
      train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 "$@" \
    && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
    || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; return 1; }
  echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
      -- --require_density \
    && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
  echo "###### [$EXP] result ######"
  tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
}

lane_A () {   # GPU 0-3
  run_trial 0,1,2,3 29501 cvitA 260623_ar_cvit_c6_g742_rope   'model.channel_groups=[7,4,2]' 'model.pos_encoding=rope3d'
  run_trial 0,1,2,3 29501 cvitA 260623_ar_cvit_c5_g742_mask06 'model.channel_groups=[7,4,2]' 'mae.mask_ratio=0.6'
}
lane_B () {   # GPU 4-7
  run_trial 4,5,6,7 29511 cvitB 260623_ar_cvit_c3_g742_dw03     'model.channel_groups=[7,4,2]' 'mae.density_channel_weight=0.3' 'mae.gradmag_channel_weight=0.3'
  run_trial 4,5,6,7 29511 cvitB 260623_ar_cvit_c7_g742_atombias 'model.channel_groups=[7,4,2]' 'mae.mask_strategy=atom_biased'
}

echo "===== AR ChannelViT [7,4,2] left-trials START $(date '+%m-%d %H:%M:%S') ====="
echo "lane A (GPU0-3): c6_rope -> c5_mask06   |   lane B (GPU4-7): c3_dw03 -> c7_atombias"
lane_A > "$LOG/260624_ar_cvit_laneA.log" 2>&1 &
PA=$!
lane_B > "$LOG/260624_ar_cvit_laneB.log" 2>&1 &
PB=$!
wait "$PA"; rA=$?
wait "$PB"; rB=$?
echo "===== AR ChannelViT left-trials COMPLETE $(date '+%m-%d %H:%M:%S')  laneA=$rA laneB=$rB ====="
