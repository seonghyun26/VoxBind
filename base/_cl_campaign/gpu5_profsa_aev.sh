#!/bin/bash
# GPU5 lane — ProFSA (frozen encoder + head retrain, 3 seeds) then AEV-PLIG (3 seeds) on the 3 CL splits.
set -u
PF=/home/shpark/prj-denovo/VoxBind/base/profsa
PFPY=/home/shpark/.conda/envs/profsa/bin/python
AEV=/home/shpark/prj-denovo/VoxBind/base/aevplig
AEVPY=/home/shpark/.conda/envs/aevplig/bin/python
LOG=/home/shpark/prj-denovo/VoxBind/base/_cl_campaign/gpu5_profsa_aev.log
ts(){ date "+%F %T"; }

# ---- ProFSA ----
cd "$PF"
P=$(pwd); CKPT=$P/data/log/train/profsa/profsa_release/checkpoints/last.ckpt; LOGS=$P/_edrscc/logs
mkdir -p "$LOGS"
echo "[$(ts)] === GPU5 lane: ProFSA CL (9 runs) ===" | tee -a "$LOG"
for cl in cl1 cl12 cl123; do
  DATA=$P/data/dataset/edrscc_${cl}; full=lp_edrscc_v2_${cl}
  for SEED in 0 1 2; do
    if [ "$SEED" -eq 0 ]; then RUN=lba30_${cl}; LOGF=$LOGS/${full}.log
    else RUN=lba30_${cl}_seed${SEED}; LOGF=$LOGS/${full}_seed${SEED}.log; fi
    echo "[$(ts)] ProFSA $full seed$SEED" | tee -a "$LOG"
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 $PFPY train.py experiment=lba30 logging=csv '~callbacks.rich' \
      dataset.dataset_cfg.train.data_dir=$DATA model.cfg.data_dir=$DATA \
      model.cfg.pretrained_weights=$CKPT model.cfg.dropout=0.5 optim.lr=0.0002 scheduler.num_warmup_steps=200 \
      trainer.devices=1 run_test=true +trainer.enable_progress_bar=false seed=$SEED \
      hydra.run.dir=$P/_edrscc/runs/$RUN > "$LOGF" 2>&1 \
      && echo "[$(ts)]   ok $full seed$SEED" | tee -a "$LOG" \
      || echo "[$(ts)]   FAIL ProFSA $full seed$SEED" | tee -a "$LOG"
  done
  $PFPY _edrscc/src/aggregate_results.py --split $full >> "$LOG" 2>&1 \
    && echo "[$(ts)] ProFSA aggregated $full" | tee -a "$LOG" \
    || echo "[$(ts)] FAIL ProFSA agg $full" | tee -a "$LOG"
done

# ---- AEV-PLIG ----
cd "$AEV"
echo "[$(ts)] === GPU5 lane: AEV-PLIG CL (3 runs x 3 seeds) ===" | tee -a "$LOG"
for cl in cl1 cl12 cl123; do
  full=lp_edrscc_v2_${cl}
  echo "[$(ts)] AEV-PLIG $full (3 seeds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=5 $AEVPY src/train_edrscc.py --split $full --seeds 0 1 2 --epochs 200 \
    >> logs/train_${full}.log 2>&1 \
    && echo "[$(ts)]   ok AEV $full" | tee -a "$LOG" \
    || echo "[$(ts)]   FAIL AEV $full" | tee -a "$LOG"
done
echo "[$(ts)] === GPU5 lane DONE ===" | tee -a "$LOG"
