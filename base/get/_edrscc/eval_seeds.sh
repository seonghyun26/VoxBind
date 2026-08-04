#!/bin/bash
cd /home/shpark/prj-denovo/VoxBind/base/get
source $(conda info --base)/etc/profile.d/conda.sh; conda activate get
# wait for both seed trainings to finish
while pgrep -f "edrscc_get_seed1" >/dev/null || pgrep -f "edrscc_get_seed2" >/dev/null; do sleep 20; done
echo "BOTH_DONE"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
for s in 1 2; do
  CKPT=$(grep -oE "Validation: [0-9.]+, save path: [^ ]+\.ckpt" _edrscc/logs/train_seed$s.log | sed -E 's/Validation: ([0-9.]+), save path: (.*)/\1 \2/' | sort -n | head -1 | awk '{print $2}')
  echo "=== seed$s ckpt: $CKPT ==="
  CUDA_VISIBLE_DEVICES=$GPU python inference.py --test_set datasets/edrscc/test.pkl --task PDBBind \
      --ckpt "$CKPT" --gpu 0 --save_path _edrscc/preds_test_seed$s.jsonl 2>/dev/null
  echo "--- seed$s metrics ---"
  python evaluate.py --predictions _edrscc/preds_test_seed$s.jsonl 2>/dev/null | grep -E "Pearson|Spearman|RMSE"
done
echo "ALL_EVAL_DONE"
