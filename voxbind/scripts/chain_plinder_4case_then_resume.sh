#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
# chain_plinder_6case_then_resume.sh
#   User-approved GPU protocol for the 6-case PLINDER ablation:
#     stop keepgoing watcher -> stop sig=1.0 denoiser -> run 6 density-MAE pretrains
#     (density / gradmag / D+G  x  ViT / ChannelViT) on the _v2clean PLINDER set,
#     each 4-GPU foreground, THEN re-arm the watcher to auto-resume the denoiser.
#
#   Launch detached:
#     cd /home1/irteam/VoxBind/voxbind && \
#     setsid nohup bash scripts/chain_plinder_6case_then_resume.sh \
#       > log/chain_plinder_6case.log 2>&1 &
#   Stop just this orchestrator: pkill -9 -f '[c]hain_plinder_6case_then_resume.sh'
set -uo pipefail

VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CROPS=$VOX/dataset/data/xray_crops_aligned_plinder_v2clean
STATS=$CROPS/stats.json
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_plinder_6case.lock

# 6-case grid: config-name -> exp-name (exp-name also = config's exp_name)
# 4-case grid (gradmag-only dropped: input_mode='gradmag' unimplemented):
# {density-only, density+gradmag} x {ViT, ChannelViT}
CONFIGS=(
  config_train_density_only_vit_mae_40m_plinder
  config_train_density_gradmag_vit_mae_40m_plinder
  config_train_density_only_channelvit_40m_plinder
  config_train_density_gradmag_channelvit_40m_plinder
)

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1          # no-C-compiler box (memory)
export OMP_NUM_THREADS=8
mkdir -p "$LOG"; cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

exec 9>"$LOCK" || { log "cannot open lock $LOCK"; exit 1; }
if ! flock -n 9; then log "another chain_plinder_6case already running — exiting"; exit 1; fi

# ── PRE-FLIGHT: never stop the denoiser unless the dataset + configs are ready ──
log "pre-flight: dataset + configs ..."
if [ ! -f "$VOX/dataset/data/data_train_plinder_v2clean.pt" ] || [ ! -f "$STATS" ] || [ ! -d "$CROPS/train" ]; then
  log "ABORT: _v2clean dataset not built. Denoiser untouched."; exit 1
fi
for c in "${CONFIGS[@]}"; do
  [ -f "$VOX/configs/$c.yaml" ] || { log "ABORT: missing config $c.yaml. Denoiser untouched."; exit 1; }
done
NTRAIN=$("$PY/python" -c "import json;print(json.load(open('$STATS'))['n_train'])" 2>/dev/null)
if [ -z "${NTRAIN:-}" ] || [ "$NTRAIN" -lt 1100 ]; then
  log "ABORT: bad n_train ('${NTRAIN:-}') from $STATS. Denoiser untouched."; exit 1
fi
SUBSET_N=$(( NTRAIN - 100 ))
log "pre-flight OK: n_train=$NTRAIN -> subset_n=$SUBSET_N subset_val_n=100; 6 configs present."

# ── 1. stop the keep-going watcher FIRST (else it relaunches the denoiser) ──────
log "stopping chain_sig1.0_keepgoing watcher (if any) ..."
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "  watcher killed" || log "  no watcher running"
sleep 3

# ── 2. stop the denoiser (bracket pattern won't self-match; != train_density.py) ─
if pgrep -f '[t]rain_ddp.py' >/dev/null 2>&1; then
  log "stopping denoiser train_ddp.py ..."
  pkill -9 -f '[t]rain_ddp.py'; sleep 5
  pkill -9 -f '[t]orchrun --standalone --nproc_per_node=4 train_ddp.py' 2>/dev/null || true
else
  log "no denoiser train_ddp.py running"
fi

# ── 3. wait for GPU memory to drain ────────────────────────────────────────────
log "waiting for GPU memory to drain ..."
for _ in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 6000 ] && break
  sleep 10
done
log "GPU mem used before launch (MiB total): ${USED:-n/a}"

# ── 4. run the 6 pretrains sequentially, each 4-GPU foreground (continue on fail) ─
OK=0; FAIL=0
for CFG in "${CONFIGS[@]}"; do
  EXP="${CFG#config_train_}"; EXP="${EXP}"          # exp dir name = config tail
  RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
  log "LAUNCH [$((OK+FAIL+1))/${#CONFIGS[@]}] cfg=$CFG subset_n=$SUBSET_N  (log: $RUNLOG)"
  START=$SECONDS
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
    "$VOX/train_density.py" \
    --config-name="$CFG" \
    dset.data_dir="$VOX/dataset/data" \
    dset.subset_n="$SUBSET_N" \
    dset.subset_val_n=100 \
    exp_name="$EXP" \
    hydra.run.dir="$VOX/exps/$EXP" \
    > "$RUNLOG" 2>&1
  RC=$?
  DUR=$(( SECONDS - START ))
  if [ "$RC" -eq 0 ]; then OK=$((OK+1)); log "  OK   $EXP (dur=${DUR}s)"; else FAIL=$((FAIL+1)); log "  FAIL $EXP rc=$RC (dur=${DUR}s) — see $RUNLOG"; fi
done
log "pretrains done: ok=$OK fail=$FAIL"

# ── 5. AUTO-RESUME the denoiser: re-arm the keep-going watcher ──────────────────
if [ -f "$DENOISER_CK" ]; then
  log "re-arming chain_sig1.0_keepgoing watcher to resume the denoiser ..."
  setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" \
    > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "  watcher re-armed (resumes sig=1.0 from latest ckpt)."
else
  log "WARN: denoiser ckpt $DENOISER_CK missing — NOT re-arming (manual resume needed)."
fi
log "done. (probes run separately on lp_edrscc_v2 once encoders exist.)"
