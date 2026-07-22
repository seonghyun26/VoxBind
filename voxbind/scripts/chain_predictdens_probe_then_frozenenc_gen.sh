#!/usr/bin/env bash
# After the predict-density v2.1 pretrain finishes (chain_predictdensity_v2p1.sh writes the
# "PREDICTDENSITY v2.1 TRAIN DONE" marker), do — UNATTENDED:
#   1) probe the predict-density encoder on lp_edrscc_v2 (task #16; force_mask auto-detected
#      from the encoder cfg). Best-effort — a probe hiccup does NOT block the gen run.
#   2) launch the FROZEN-encoder generative VoxBind: atomblob7 v2.1 ChannelViT (45.7M) loaded
#      frozen as the conditioning encoder; only UNet3D + density_proj train. 350 ep, sig=0.9.
#   3) ALWAYS resume the original sig=1.0 denoiser — whether gen finishes OR fails (user directive
#      2026-06-30: "if the voxbind training on frozen encoder fails, resume the original denoiser").
#   setsid nohup bash scripts/chain_predictdens_probe_then_frozenenc_gen.sh \
#       > log/chain_predictdens_probe_then_frozenenc_gen.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log

# ── predict-density (prerequisite + task #16 probe) ──────────────────────────────
PD_LOG=$LOG/chain_predictdensity_v2p1.log                                   # prerequisite run's log
PD_EXP=260701_plinder_v2p1_predictdens_atomblob7_cdg_channelvit_full_pretrain
PD_TAG=predictdens_v2p1_cdg_cvit
COND=atomblob_density_gradmag

# ── frozen-encoder generative run ────────────────────────────────────────────────
GEN_CFG=config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1
GEN_EXP=voxbind_frozenenc_atomblob7_v2p1_sig0.9
GEN_OUT=$VOX/exps/$GEN_EXP

DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_predictdens_probe_then_frozenenc_gen.lock

export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another predictdens→gen chain running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; } || log "WARN: denoiser ckpt missing ($DENOISER_CK) — cannot resume"; }
drain(){ for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done; log "GPU drained (${U:-?} MiB)"; }

# ── 1) wait for predict-density training to finish (grep-on-file, NO pgrep self-match) ──
DEADLINE=$(( $(date +%s) + 14*3600 ))
log "waiting for predict-density v2.1 training to finish ..."
until grep -q "PREDICTDENSITY v2.1 TRAIN DONE" "$PD_LOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for predict-density"; exit 1; }
  sleep 60
done
PD_RC_LINE=$(grep "PREDICTDENSITY v2.1 TRAIN DONE" "$PD_LOG" | tail -1)
log "  predict-density done: $PD_RC_LINE"

# the predict-density chain resumes the filler denoiser on finish — stop watcher+denoiser, drain
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "filler denoiser stopped"; sleep 5; fi
drain

# ── 2) probe the predict-density encoder (task #16) — best-effort; never blocks gen ────
if echo "$PD_RC_LINE" | grep -q "rc=0" && [ -f "$VOX/exps/$PD_EXP/checkpoint_e0099.pth.tar" ]; then
  log "FEATURES predict-density (cond=$COND tag=$PD_TAG) — force_mask auto-detected from cfg ..."
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py features \
    --condition "$COND" --exp_dir "exps/$PD_EXP" --epoch 99 --voxel_version v5 \
    --atom_source ligvdw --tag "$PD_TAG" --device cuda --batch_size 24 > "$LOG/${PD_TAG}_features.log" 2>&1
  RCF=$?; log "predict-density features exit=$RCF"
  if [ "$RCF" -eq 0 ]; then
    log "PROBE predict-density lp_edrscc_v2 (3 seeds) ..."
    CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py probe \
      --conditions "$COND" --exp_dir "exps/$PD_EXP" --epoch 99 --voxel_version v5 \
      --feature_tag "$PD_TAG" --tag "$PD_TAG" --split lp_edrscc_v2 --seeds 3 --device cuda \
      > "$LOG/${PD_TAG}_probe.log" 2>&1
    log "predict-density probe exit=$? → dataset/data/pdbbind/results/probe_results_e99_v5_lp_edrscc_v2split_${PD_TAG}.csv"
  else
    log "predict-density FEATURES FAILED — see $LOG/${PD_TAG}_features.log (continuing to gen)"
  fi
  drain   # release any probe-held GPU memory before the gen launch
else
  log "predict-density not rc=0 or e0099 missing — SKIP probe (continuing to gen)"
fi

# ── 3) launch FROZEN-encoder generative VoxBind (atomblob7 v2.1 frozen), 350 ep ────────
mkdir -p "$GEN_OUT"
GENLOG="$LOG/${GEN_EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH frozen-enc gen: 4-GPU cfg=$GEN_CFG exp=$GEN_EXP → $GENLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_ddp.py" --config-name="$GEN_CFG" \
  exp_name="$GEN_EXP" output_dir="$GEN_OUT" hydra.run.dir="$GEN_OUT" \
  > "$GENLOG" 2>&1
RCG=$?; log "frozen-enc gen exit=$RCG (ckpt: exps/$GEN_EXP)"

# ── 4) ALWAYS resume the original denoiser — success OR failure (user directive) ───────
[ "$RCG" -ne 0 ] && log "GEN FAILED (rc=$RCG) — see $GENLOG. Resuming original denoiser per directive." \
                 || log "GEN finished ok (rc=$RCG). Resuming original denoiser."
resume_denoiser
log "PREDICTDENS→FROZENENC-GEN CHAIN DONE (gen rc=$RCG)."
