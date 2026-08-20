#!/usr/bin/env bash
set -u
P=/home/shpark/prj-denovo/VoxBind/base/profsa
PY=/home/shpark/.conda/envs/profsa/bin/python
CASF=$P/data/dataset/casf2016
OUTD=$P/_casf; mkdir -p "$OUTD"
cd "$P" || exit 1
for S in 0 1 2 3 4; do
  suf=""; [ $S -gt 0 ] && suf="_seed$S"
  RUN="$P/_edrscc/runs/lba30${suf}"
  OUT="$OUTD/profsa_casf5_seed${S}.csv"
  echo "[profsa-casf] seed$S: $RUN"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU:-3} $PY _casf/infer_casf.py \
    --run_dir "$RUN" --ckpt best --out "$OUT" --casf_data_dir "$CASF" 2>&1 | tail -3 || echo "FAIL s$S"
done
echo "PROFSA CASF 5SEED INFER DONE"
