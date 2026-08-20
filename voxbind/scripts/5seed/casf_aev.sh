#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/aevplig/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/aevplig/src || exit 1
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY casf_eval.py --seeds 0 1 2 3 4
echo "AEV CASF 5SEED DONE"
