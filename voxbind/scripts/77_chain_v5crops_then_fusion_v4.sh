#!/usr/bin/env bash
# 77_chain_v5crops_then_fusion_v4.sh
#   Unattended chain: build the CrossDocked v5 density crops this box is missing, then
#   launch the fusion='v4' + CDG_v2 training under the crash-resume supervisor.
#
#   WHY THE DATA STEP EXISTS: svr12 has the raw 2Fo-Fc maps (dataset/data/ccp4, 13,364
#   maps / 54G) and the v1 crops (dataset/data/xray_crops), but xray_crops_aligned_v5/
#   holds only a stats.json — the actual v5 crops were never built here. v5 is the
#   arcsinh+z normalisation the CDG encoders were PRETRAINED under, so it is not
#   interchangeable with the v1 load-time +-3sigma z-score: feeding v1 crops to the
#   frozen encoder puts it OOD and would confound the v4 fusion result with a
#   normalisation mismatch. v5 needs the RAW crop (00b's scratch pass), which the v1
#   crops do not preserve, so it is a full align+crop pass.
#
#   Stages, each skipped if already satisfied so the chain is restartable:
#     1. deposited PDBs   (00a --what pdbs)  — 00b's Kabsch alignment needs them
#     2. v5 crops         (00b --version v5) — align + crop + pooled arcsinh+z stats
#     3. v4 GPU smoke     (test/v4_gpu_smoke.py) — refuses to launch on a non-zero exit
#     4. training         (76_supervise_fusion_v4_8gpu.sh)
#
#   Progress: logs/fusion_v4_chain.log
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
PY=/home/shpark/miniforge3/envs/voxbind/bin
cd "$ROOT/voxbind" || exit 1

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1

DATA="$ROOT/voxbind/dataset/data"
V5="$DATA/pretrain/xray_crops_aligned_v5"
ENCODER="$ROOT/voxbind/model_zoo/CDG_v2/checkpoint_e0025.pth.tar"
WORKERS="${WORKERS:-32}"          # box is shared; 32 of 384 cores, well clear of a fork bomb
EXP_NAME="${EXP_NAME:-260831_fusion_v4_cdgv2_8gpu}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"

mkdir -p logs
CLOG="$ROOT/voxbind/logs/fusion_v4_chain.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$CLOG"; }

log "=== chain start: v5 crops -> fusion v4 (CDG_v2) on 8 GPUs ==="

# ── 1. deposited PDBs ─────────────────────────────────────────────────────────
# 00a is resumable (existing non-empty files are skipped), so re-running it is the
# cheapest way to be sure the download finished rather than died mid-way.
log "[1/4] deposited PDBs -> $DATA/pdb"
"$PY/python" dataset/00a_density_download.py --what pdbs --workers "$WORKERS" \
    >> "$CLOG" 2>&1
n_pdb=$(ls "$DATA/pdb" 2>/dev/null | wc -l)
log "[1/4] done: $n_pdb deposited PDBs on disk"
[ "$n_pdb" -gt 1000 ] || { log "[1/4] FAILED: only $n_pdb PDBs — aborting"; exit 1; }

# ── 1b. map naming ────────────────────────────────────────────────────────────
# 00b opens `$ccp4_dir/{pdb_id}.ccp4`, but this box's 13,364 maps were downloaded as
# `{pdb_id}.map` by an older pipeline. The mismatch is SILENT and total: align_pdb's
# first gate returns "no_eds_map" for every sample, so the run "succeeds" in 6 seconds
# with 0/99,881 trustworthy crops. Symlink rather than rename — nothing else that
# already points at the .map names breaks.
n_map=$(ls "$DATA/ccp4"/*.map 2>/dev/null | wc -l)
n_ccp4=$(ls "$DATA/ccp4"/*.ccp4 2>/dev/null | wc -l)
if [ "$n_map" -gt 0 ] && [ "$n_ccp4" -lt "$n_map" ]; then
    log "[1b] $n_map .map maps but only $n_ccp4 .ccp4 — symlinking"
    ( cd "$DATA/ccp4" && for f in *.map; do ln -sfn "$f" "${f%.map}.ccp4"; done )
    n_ccp4=$(ls "$DATA/ccp4"/*.ccp4 2>/dev/null | wc -l)
fi
log "[1b] $n_ccp4 maps visible to 00b as .ccp4"
[ "$n_ccp4" -gt 1000 ] || { log "[1b] FAILED: only $n_ccp4 .ccp4 maps — aborting"; exit 1; }

# ── 2. v5 crops ───────────────────────────────────────────────────────────────
if [ -d "$V5/train" ] && [ "$(ls "$V5/train" 2>/dev/null | wc -l)" -gt 1000 ]; then
    log "[2/4] v5 crops already present ($(ls "$V5/train" | wc -l) train) — skipping"
else
    # A previous aborted attempt caches its transforms, and `ensure_transforms` reuses
    # any cache whose row count matches — INCLUDING one where every row failed. That
    # turns a fixed prerequisite into a permanently-stuck 0-crop run, because the
    # alignment is never retried. Drop a cache that has no usable rows.
    for tx in "$DATA/pretrain/xray_crops_aligned"/{train,test}_transforms.npz; do
        [ -f "$tx" ] || continue
        ok=$("$PY/python" -c "import numpy as np,sys;print(int(np.load(sys.argv[1])['ok'].sum()))" \
             "$tx" 2>/dev/null || echo 0)
        [ "$ok" -eq 0 ] && { log "[2/4] dropping stale zero-ok transform cache $(basename "$tx")"; rm -f "$tx"; }
    done
    log "[2/4] building v5 crops (align + crop + pooled arcsinh+z); this is the long stage"
    "$PY/python" dataset/00b_density_preprocess.py \
        --version v5 \
        --data_dir "$DATA" --ccp4_dir "$DATA/ccp4" --pdb_dir "$DATA/pdb" \
        --out_root "$DATA" --splits train test --workers "$WORKERS" \
        >> "$CLOG" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || { log "[2/4] FAILED: 00b exited $rc — aborting"; exit 1; }
fi
[ -f "$V5/train_available.npy" ] || { log "[2/4] FAILED: no train_available.npy"; exit 1; }
avail=$("$PY/python" -c "import numpy as np,sys;print(int(np.load(sys.argv[1]).sum()))" \
        "$V5/train_available.npy")
log "[2/4] done: $avail density-bearing train rows available"
[ "$avail" -gt 10000 ] || { log "[2/4] FAILED: only $avail available rows — aborting"; exit 1; }

# ── 3. v4 GPU smoke ───────────────────────────────────────────────────────────
# Geometry mismatch / CUDA-only shape bug / OOM would otherwise surface as a dead
# multi-day launch. Refuse to start training on a non-zero exit.
log "[3/4] v4 GPU smoke at the real per-rank batch (bsz=16)"
CUDA_VISIBLE_DEVICES=0 "$PY/python" test/v4_gpu_smoke.py \
    --encoder "$ENCODER" --bsz 16 >> "$CLOG" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { log "[3/4] FAILED: v4 smoke exited $rc — NOT launching"; exit 1; }
log "[3/4] smoke PASS"

# ── 4. training ───────────────────────────────────────────────────────────────
log "[4/4] launching supervisor: exp=$EXP_NAME total_epochs=$TOTAL_EPOCHS"
EXP_NAME="$EXP_NAME" TOTAL_EPOCHS="$TOTAL_EPOCHS" \
    bash scripts/76_supervise_fusion_v4_8gpu.sh
log "[4/4] supervisor returned $?"
