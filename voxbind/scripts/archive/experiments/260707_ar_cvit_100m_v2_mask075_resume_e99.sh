#!/bin/bash
# 260707 — CONTINUE the 0.644 model (d640/L18/h10 @ mask0.75, the round-1 sweep winner) from its
# e49 checkpoint for 50 more epochs → e99, i.e. "the full run FROM the 50-epoch number". Valid
# because LR is CONSTANT (1e-4, no schedule) and same seed → resume-and-extend == fresh 100ep run
# but 2× faster (skips recomputing e0-49). Trainer: range(start_epoch=50, 50+num_epochs); num_epochs=50
# → trains e50..e99. Then frozen probe e99 for the table-comparable 100-epoch number vs v1 40M 0.637.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7
EXP=260705_ar_cvit_100m_v2_mask075          # resume dir = this exp; e49 → e99

echo "===== RESUME 0.644 model → e99 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:29587 --rdzv-id=cvitResumeE99 \
    train_density.py --config-name="$CFG" \
    resume=exps/$EXP num_epochs=50 wandb=false \
  && echo "[$EXP] RESUME→e99 OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] RESUME FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE e99 start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch 99 --gpu "${GPUS%%,*}" \
    --tag "${EXP}_e100" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE e99 OK" || echo "[$EXP] PROBE e99 FAILED"
echo "###### [$EXP] 100ep result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}_e100.csv" 2>/dev/null
echo "===== RESUME 0.644 model → e99 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
