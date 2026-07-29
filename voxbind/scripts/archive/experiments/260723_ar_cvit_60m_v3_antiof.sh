#!/bin/bash
# 260723 — OVERFIT-PREVENTION study. v3-100M (mask0.75, ema0.999, HCS0.15) reached val recon
# L_dens 0.0067 (very low = memorization) but probe plateaued ρ0.632. Hypothesis: the encoder
# overfits the reconstruction pretext. Test a SMALLER 60M model (dim512/depth18/heads8, ~63M —
# depth>width, on-trend) + 2 overfit regularizers, on the v3 corpus (70,725 train). Sequential
# 4-GPU chain (eff-batch 128, bsz4×accum8, compile OFF for HCS). Each: pretrain 50ep → frozen
# probe lp_edrscc_v2. Reference to beat: v3-100M ρ0.632 / champion 0.644.
#   T1 base60M      : does halving capacity alone reduce overfit?
#   T2 +drop_path0.2: stochastic depth (canonical ViT overfit regularizer)
#   T3 +mask0.85    : harder pretext → recon not-trivial → forces transferable features
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29611; SUBSET=70725
BASE="model.channel_groups=[7,4,2] model.dim=512 model.depth=18 model.heads=8 \
model.channel_group_dropout=0.15 mae.mask_ratio=0.75 mae.ema_decay=0.999 compile.enabled=false \
dset.data_file=pretrain/data_train_plinder_v3_perelem.pt \
dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v3_perelem \
dset.subset_n=$SUBSET dset.subset_val_n=100 bsz=4 accum_steps=8 num_workers=6 prefetch_factor=2 num_epochs=50"

run_trial () {   # $1=EXP ; rest = extra overrides
  local EXP=$1; shift
  echo "###### [$EXP] PRETRAIN start $(date '+%m-%d %H:%M:%S') :: $* ######"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$EXP" \
      train_density.py --config-name="$CFG" exp_name="$EXP" $BASE "$@" \
    && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
    || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; return 1; }
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu 0 --tag "$EXP" --num_workers 0 -- --require_density \
    && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
  echo "###### [$EXP] result (r / rho / RMSE) ######"
  tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
}

echo "===== 60M v3 overfit-prevention chain START $(date '+%m-%d %H:%M:%S') ====="
run_trial 260723_ar_cvit_60m_v3_base
run_trial 260723_ar_cvit_60m_v3_droppath02 +model.drop_path=0.2
run_trial 260723_ar_cvit_60m_v3_mask085 mae.mask_ratio=0.85
echo "===== 60M v3 overfit-prevention chain COMPLETE $(date '+%m-%d %H:%M:%S') ====="
