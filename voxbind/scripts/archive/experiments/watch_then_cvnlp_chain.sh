#!/bin/bash
# Wait for the distance-conditioned radius-embed run to finish (frees GPU 0-3), then run the
# CV/NLP transferable-trick queue sequentially on GPU 0-3:
#   A = stochastic depth (DropPath)  →  B = data2vec latent-target MAE
# Both are controlled changes off the ChannelViT [7,4,2] winner (one knob each).
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
DIST_LOG=log/260630_ar_cvit_radembed_k8_dist.log

echo "[watch-cvnlp] $(date '+%m-%d %H:%M:%S') waiting for distance-conditioned run to complete..."
while ! grep -q "radius-embed k=8 COMPLETE" "$DIST_LOG" 2>/dev/null; do
    sleep 180
done
echo "[watch-cvnlp] $(date '+%m-%d %H:%M:%S') dist done. Waiting for GPU 0 to drain..."
while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    [ "${m:-9999}" -lt 1500 ] && break
    sleep 30
done

echo "[watch-cvnlp] $(date '+%m-%d %H:%M:%S') === A: stochastic depth ==="
bash scripts/archive/experiments/260701_ar_cvit_g742_droppath.sh > log/260701_ar_cvit_g742_droppath.log 2>&1
echo "[watch-cvnlp] $(date '+%m-%d %H:%M:%S') A done → === B: data2vec ==="
bash scripts/archive/experiments/260701_ar_cvit_g742_data2vec.sh > log/260701_ar_cvit_g742_data2vec.log 2>&1
echo "[watch-cvnlp] $(date '+%m-%d %H:%M:%S') CV/NLP queue (A,B) complete."
