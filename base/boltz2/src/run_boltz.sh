#!/bin/bash
# run_boltz.sh — run Boltz-2 affinity prediction over the prepared YAML inputs.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_boltz.sh [inputs_dir] [out_dir]
# Requires the `boltz` env (pip install boltz; weights auto-download on first run to $BOLTZ_CACHE).
set -u
BASE="$(cd "$(dirname "$0")/.." && pwd)"
INPUTS="${1:-$BASE/inputs}"
OUT="${2:-$BASE/preds}"
# boltz 2.x is a console script (no `python -m boltz`); weights live in $BOLTZ_CACHE (~/.boltz).
BOLTZ_BIN="${BOLTZ_BIN:-/home/shpark/.conda/envs/boltz/bin/boltz}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/.boltz}"

mkdir -p "$OUT" "$BASE/logs"
# --use_msa_server: auto-generate MSAs via the MMseqs2/Colab server (network).
# --diffusion_samples 1 keeps STRUCTURE sampling cheap; the affinity head runs its own
# 5-sample ensemble (--diffusion_samples_affinity default) which drives the pIC50 output.
# --output_format pdb is smaller than mmcif (disk-lean; only the affinity JSON is scored).
"$BOLTZ_BIN" predict "$INPUTS" \
    --use_msa_server \
    --out_dir "$OUT" \
    --output_format pdb \
    --model boltz2 \
    --diffusion_samples 1 \
    --num_workers 2 \
    2>&1 | tee -a "$BASE/logs/run_boltz.log"
