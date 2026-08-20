#!/usr/bin/env bash
set -u
PY=/home/shpark/.conda/envs/hbgsa/bin/python
cd /home/shpark/prj-denovo/VoxBind/base/hbgsa || exit 1
CUDA_VISIBLE_DEVICES=${GPU:-3} $PY run_casf2016.py --seeds 0,1,2,3,4
echo "HBGSA CASF 5SEED DONE"
