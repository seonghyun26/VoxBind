#!/usr/bin/env bash
# ProFSA -> 5 seeds: keep existing seed0-2 logs, train ONLY seed3,4, re-aggregate 0-4.
# Frozen ProFSA encoders + regression head (hydra). Aggregate reads logs/{split}[_seed{s}].log.
set -u
P=/home/shpark/prj-denovo/VoxBind/base/profsa
PY=/home/shpark/.conda/envs/profsa/bin/python
CKPT=$P/data/log/train/profsa/profsa_release/checkpoints/last.ckpt
GPU="${GPU:-0}"
cd "$P" || exit 1
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  CLTAG="${SPLIT#lp_edrscc_v2}"          # "" / _cl1 / _cl12 / _cl123
  DATA="$P/data/dataset/edrscc${CLTAG}"
  for SEED in 3 4; do
    RUN="$P/_edrscc/runs/lba30${CLTAG}_seed${SEED}"
    LOG="$P/_edrscc/logs/${SPLIT}_seed${SEED}.log"
    echo "[profsa] train $SPLIT seed$SEED (GPU$GPU) data=$DATA"
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$GPU $PY "$P/train.py" \
      experiment=lba30 logging=csv '~callbacks.rich' \
      dataset.dataset_cfg.train.data_dir="$DATA" model.cfg.data_dir="$DATA" \
      model.cfg.pretrained_weights="$CKPT" model.cfg.dropout=0.5 \
      optim.lr=0.0002 scheduler.num_warmup_steps=200 \
      trainer.devices=1 seed=$SEED run_test=true +trainer.enable_progress_bar=false \
      hydra.run.dir="$RUN" > "$LOG" 2>&1 || echo "FAIL $SPLIT s$SEED"
  done
  echo "[profsa] aggregate $SPLIT (5 seeds)"
  $PY "$P/_edrscc/src/aggregate_results.py" --split "$SPLIT" --seeds 0,1,2,3,4 || echo "FAIL agg $SPLIT"
done
echo "ALL PROFSA DONE"
