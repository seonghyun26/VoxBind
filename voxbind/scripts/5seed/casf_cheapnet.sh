#!/usr/bin/env bash
set -u
cd /home/shpark/prj-denovo/VoxBind/base/cheapnet/_edrscc || exit 1
CUDA_VISIBLE_DEVICES=${GPU:-3} bash cheappy train_casf_eval.py --seeds 0 1 2 3 4
echo "CHEAPNET CASF 5SEED DONE"
