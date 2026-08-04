#!/usr/bin/env bash
# Train ProFSA head on lp_edrscc_v2 for seeds 1 and 2 on GPU 5.
# Seeds are passed as `seed=N` Hydra override; run_dir is per-seed.

set -e
PY=/home/shpark/.conda/envs/profsa/bin/python
P=/home/shpark/prj-denovo/VoxBind/base/profsa
CKPT=$P/data/log/train/profsa/profsa_release/checkpoints/last.ckpt

for SEED in 1 2; do
  RUN_DIR=$P/_edrscc/runs/lba30_seed${SEED}
  echo "=== Training seed=${SEED} -> ${RUN_DIR} ==="
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 $PY $P/train.py \
    experiment=lba30 \
    logging=csv \
    '~callbacks.rich' \
    dataset.dataset_cfg.train.data_dir=$P/data/dataset/edrscc \
    model.cfg.data_dir=$P/data/dataset/edrscc \
    model.cfg.pretrained_weights=$CKPT \
    model.cfg.dropout=0.5 \
    optim.lr=0.0002 \
    scheduler.num_warmup_steps=200 \
    trainer.devices=1 \
    seed=${SEED} \
    run_test=true \
    +trainer.enable_progress_bar=false \
    hydra.run.dir=$RUN_DIR \
    2>&1 | tee $P/_casf/train_seed${SEED}.log
  echo "=== Seed ${SEED} done ==="
done
echo "Both seeds complete."
