#!/bin/bash
# Run bench_voxelizer.py on GPU 7 (default).
# Usage:
#   ./run_bench_voxelizer.sh
#   ./run_bench_voxelizer.sh --bsz 8 --n_batches 10

GPU=${CUDA_VISIBLE_DEVICES:-7}
CUDA_VISIBLE_DEVICES=$GPU python bench_voxelizer.py "$@"
