#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/dsmbind/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/honestaffinity/src || exit 1
CUDA_VISIBLE_DEVICES=${GPU:-3} $PY casf_eval.py --seeds 0 1 2 3 4 --gpu 0
echo "HONEST CASF 5SEED DONE"
