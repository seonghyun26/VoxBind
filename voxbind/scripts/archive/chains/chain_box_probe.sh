#!/usr/bin/env bash
# Frozen-probe the box-pretrained encoder on lp_edrscc_v2, then resume the denoiser.
# stop denoiser → features (GPU0) → probe (3 seeds) → resume denoiser.
#   setsid nohup bash scripts/archive/chains/chain_box_probe.sh > log/chain_box_probe.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
EXP=260629_plinder_v2_box_roleblob_diverse_cdg_channelvit_full_pretrain
COND=roleblob_density_gradmag_channelvit          # role-split C+D+G ChannelViT (matches encoder cfg)
TAG=box_cdg_cvit
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }

resume_denoiser(){
  if [ -f "$DENOISER_CK" ]; then
    setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
    log "denoiser resume re-armed."
  fi
}

# ── stop denoiser + watcher, free the GPUs ───────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# ── features (frozen encoder → 512-D mean-pooled features) ───────────────────
log "FEATURES (cond=$COND, tag=$TAG) ..."
CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py features \
  --condition "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
  --tag "$TAG" --device cuda --batch_size 24 > "$LOG/${TAG}_features.log" 2>&1
RC1=$?; log "features exit=$RC1"
if [ "$RC1" -ne 0 ]; then log "FEATURES FAILED — see $LOG/${TAG}_features.log"; resume_denoiser; exit 1; fi

# ── probe (MLP head on lp_edrscc_v2, 3 seeds) ────────────────────────────────
log "PROBE on lp_edrscc_v2 (3 seeds) ..."
CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py probe \
  --conditions "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
  --feature_tag "$TAG" --tag "$TAG" --split lp_edrscc_v2 --seeds 3 --device cuda \
  > "$LOG/${TAG}_probe.log" 2>&1
RC2=$?; log "probe exit=$RC2"

# ── resume denoiser ──────────────────────────────────────────────────────────
resume_denoiser
log "BOX PROBE CHAIN DONE (features=$RC1 probe=$RC2). Results: dataset/data/pdbbind/results/probe_results_e99_v5_lp_edrscc_v2split_${TAG}.csv"
