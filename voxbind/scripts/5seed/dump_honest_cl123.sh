#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/dsmbind/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/honestaffinity/src
CUDA_VISIBLE_DEVICES=0 $PY train.py --split lp_edrscc_v2_cl123 --seeds 0 1 2 3 4 --gpu 0 \
  --dump_dir /home/shpark/prj-denovo/VoxBind/base/honestaffinity/cache/cl123_preds
echo "HONEST CL123 DUMP DONE"
