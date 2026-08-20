#!/usr/bin/env bash
# GET/EGNN/EGNN_TD -> 5 seeds: keep existing seed0-2 preds, train ONLY seed3,4, re-aggregate 0-4.
set -u
cd /home/shpark/prj-denovo/VoxBind/base/get || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate get
PY=/home/shpark/.conda/envs/get/bin/python
GPU="${GPU:-3}"
declare -A ML=( [GET]=get [EGNN]=egnn [EGNN_TD]=egnn_td )
for METHOD in GET EGNN EGNN_TD; do
  ml=${ML[$METHOD]}
  for SPLIT in v2 cl1 cl12 cl123; do
    TEST_PKL="datasets/edrscc/test.pkl"; [ "$SPLIT" != v2 ] && TEST_PKL="datasets/edrscc_${SPLIT}/test.pkl"
    for SEED in 3 4; do
      BASE="_edrscc/${ml}_${SPLIT}.json"
      CFG="_edrscc/${ml}_${SPLIT}_seed${SEED}.json"
      PRED="_edrscc/preds_${METHOD}_${SPLIT}_seed${SEED}.jsonl"
      [ -f "$BASE" ] || { echo "SKIP missing base $BASE"; continue; }
      $PY -c "import json;d=json.load(open('$BASE'));d['seed']=$SEED;d['save_dir']=d['save_dir'].rstrip('/')+'_seed${SEED}';json.dump(d,open('$CFG','w'),indent=2)"
      LOGF="_edrscc/logs/train_${METHOD}_${SPLIT}_seed${SEED}.log"
      echo "[getfam] train $METHOD $SPLIT seed$SEED (GPU$GPU)"
      GPU=$GPU PORT=$((9950+SEED)) bash scripts/train/train.sh "$CFG" > "$LOGF" 2>&1 || { echo "FAIL train $METHOD $SPLIT s$SEED"; continue; }
      CKPT=$(grep -oE "Validation: [0-9.]+, save path: [^ ]+\.ckpt" "$LOGF" | sed -E 's/Validation: ([0-9.]+), save path: (.*)/\1 \2/' | sort -n | head -1 | awk '{print $2}')
      [ -n "$CKPT" ] || { echo "FAIL no ckpt $METHOD $SPLIT s$SEED"; continue; }
      CUDA_VISIBLE_DEVICES=$GPU $PY inference.py --test_set "$TEST_PKL" --task PDBBind --ckpt "$CKPT" --gpu 0 --save_path "$PRED" 2>/dev/null || echo "FAIL infer $METHOD $SPLIT s$SEED"
    done
    echo "[getfam] aggregate $METHOD $SPLIT (5 seeds)"
    $PY _edrscc/aggregate_results.py --method "$METHOD" --split "$SPLIT" --seeds 0 1 2 3 4 || echo "FAIL agg $METHOD $SPLIT"
  done
done
echo "ALL GET-FAMILY DONE"
