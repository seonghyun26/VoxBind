#!/usr/bin/env bash
# build_plinder_big_03c.sh — wait for the 03b download, then build the big PLINDER
# frozen-crop set (03c precompute, --tag _big) with the protein+ligand coverage gate,
# verify crop↔atom alignment, and regenerate the HTML report. CPU-only — does NOT
# touch the GPU denoiser. Training is launched separately after inspecting this.
#
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/build_plinder_big_03c.sh > log/build_plinder_big_03c.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

# ── 1. wait for 03b to finish downloading maps+cifs ───────────────────────────
log "waiting for 03b_plinder_acquire to finish ..."
while pgrep -f '[0]3b_plinder_acquire' >/dev/null 2>&1; do sleep 30; done
log "03b done. ccp4=$(ls dataset/data/plinder/ccp4/*.ccp4 2>/dev/null | wc -l) cif=$(ls dataset/data/plinder/cif/*.cif 2>/dev/null | wc -l)"

# ── 2. 03c full build (frozen crops + coverage gate + ligand vocab gate) ───────
log "03c precompute (big) starting ..."
"$PY/python" dataset/legacy/03c_plinder_preprocess.py \
  --selection dataset/data/plinder/plinder_selected_big.csv \
  --v6_stats dataset/data/xray_crops_aligned_v5/stats.json \
  --tag _big --pocket_radius 8 \
  --coverage_check --cov_sigma 1.0 --lig_cov_frac 0.75 --poc_cov_frac 0.50 \
  --strict_vocab
RC=$?
log "03c exit=$RC"
[ "$RC" -eq 0 ] || { log "03c FAILED — stopping"; exit 1; }

# ── 3. verify crop↔atom alignment (silent-scramble guard) ─────────────────────
log "verifying alignment ..."
"$PY/python" dataset/verify_plinder_big_alignment.py --n 16 || log "WARN: verification reported issues"

# ── 4. regenerate the HTML report (dataset sections now populated) ─────────────
"$PY/python" dataset/make_plinder_big_report.py || true
log "BUILD COMPLETE. Inspect report_plinder_big.html, then launch training."
