#!/usr/bin/env bash
# GET/EGNN/EGNN_TD CASF-2016 inference, 5 seeds, from the v2 model ckpts -> per-complex preds jsonl.
set -u
G=/home/shpark/prj-denovo/VoxBind/base/get
PY=/home/shpark/.conda/envs/get/bin/python
CASF=$G/_casf_get/datasets/casf2016/test.pkl
OUTD=$G/_casf_get/preds; mkdir -p "$OUTD"
export CUDA_VISIBLE_DEVICES=${GPU:-3}
cd "$G" || exit 1
for M in GET EGNN EGNN_TD; do
  for S in 0 1 2 3 4; do
    suf=""; [ $S -gt 0 ] && suf="_seed$S"
    CK=$(find "_edrscc/models/${M}_v2${suf}" -name "*.ckpt" 2>/dev/null | head -1)
    [ -n "$CK" ] || { echo "MISSING ckpt $M s$S"; continue; }
    OUT="$OUTD/preds_${M}_casf5_seed${S}.jsonl"
    echo "[casf] $M seed$S: $CK"
    $PY $G/_casf_get/run_casf_inference.py --ckpt "$G/$CK" --test_pkl "$CASF" --out "$OUT" --gpu 0 \
      2>/dev/null || echo "FAIL infer $M s$S"
  done
done
echo "GET-FAMILY CASF 5SEED INFER DONE"
