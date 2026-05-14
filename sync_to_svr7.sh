#!/bin/bash
# Sync VoxBind data/checkpoints to kaistSvr7
# Run from the repo root: bash sync_to_svr7.sh

set -e

REMOTE="shpark@59.29.246.29"
PORT=22228
REMOTE_ROOT="/home/shpark/prj-denovo/VoxBind"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

RSYNC="rsync -avz --progress -e 'ssh -p ${PORT}'"

echo "==> Syncing to ${REMOTE}:${REMOTE_ROOT}"
echo

# ── 1. Code (skip large ignored dirs) ────────────────────────────────────────
echo "[1/4] Code..."
eval $RSYNC \
    --exclude='**/__pycache__' \
    --exclude='**/*.egg-info' \
    --exclude='.git' \
    --exclude='voxbind/dataset/data' \
    --exclude='voxbind/exps' \
    "${LOCAL_ROOT}/" \
    "${REMOTE}:${REMOTE_ROOT}/"

# ── 2. Checkpoints ────────────────────────────────────────────────────────────
echo
echo "[2/4] Checkpoints (exp_sig0.9 + poc_xray)..."
ssh -p ${PORT} ${REMOTE} "mkdir -p ${REMOTE_ROOT}/voxbind/exps"
eval $RSYNC \
    "${LOCAL_ROOT}/voxbind/exps/exp_sig0.9/" \
    "${REMOTE}:${REMOTE_ROOT}/voxbind/exps/exp_sig0.9/"
eval $RSYNC \
    "${LOCAL_ROOT}/voxbind/exps/poc_xray/" \
    "${REMOTE}:${REMOTE_ROOT}/voxbind/exps/poc_xray/"

# ── 3. CrossDocked dataset files (small) ─────────────────────────────────────
echo
echo "[3/4] CrossDocked dataset files..."
ssh -p ${PORT} ${REMOTE} "mkdir -p ${REMOTE_ROOT}/voxbind/dataset/data"
for f in data_train.pt data_test.pt split_by_name.pt index.pkl; do
    src="${LOCAL_ROOT}/voxbind/dataset/data/${f}"
    [ -f "$src" ] && eval $RSYNC "$src" "${REMOTE}:${REMOTE_ROOT}/voxbind/dataset/data/"
done
# Pocket PDB files (needed for evaluation)
eval $RSYNC \
    "${LOCAL_ROOT}/voxbind/dataset/data/crossdocked_pocket10/" \
    "${REMOTE}:${REMOTE_ROOT}/voxbind/dataset/data/crossdocked_pocket10/"

# ── 4. X-ray density crops (45G — comment out if not needed) ─────────────────
echo
echo "[4/4] X-ray crops (45G — comment out this section to skip)..."
eval $RSYNC \
    "${LOCAL_ROOT}/voxbind/dataset/data/xray_crops_norm/" \
    "${REMOTE}:${REMOTE_ROOT}/voxbind/dataset/data/xray_crops_norm/"

echo
echo "==> Done."
