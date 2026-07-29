#!/bin/bash
# Wait for the k=8 radius-embed run to fully finish (pretrain+probe), confirm GPU 0-3 are
# free, then launch the distance-conditioned (multi-scale) k=8 run on GPU 0-3.
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
K8_LOG=log/260630_ar_cvit_radembed_k8.log
DIST=scripts/archive/experiments/260630_ar_cvit_radembed_k8_dist.sh
DIST_LOG=log/260630_ar_cvit_radembed_k8_dist.log

echo "[watch] $(date '+%m-%d %H:%M:%S') waiting for k=8 to complete..."
while ! grep -q "radius-embed k=8 COMPLETE" "$K8_LOG" 2>/dev/null; do
    sleep 120
done
echo "[watch] $(date '+%m-%d %H:%M:%S') k=8 done. Waiting for GPU 0 to drain..."
# wait until GPU 0 memory is low (probe released it)
while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    [ "${m:-9999}" -lt 1500 ] && break
    sleep 30
done
echo "[watch] $(date '+%m-%d %H:%M:%S') GPU free → launching distance-conditioned k=8"
nohup bash "$DIST" > "$DIST_LOG" 2>&1 &
echo "[watch] launched dist pid $! → $DIST_LOG"
