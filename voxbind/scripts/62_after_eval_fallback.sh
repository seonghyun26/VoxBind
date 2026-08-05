#!/usr/bin/env bash
# 62_after_eval_fallback.sh
#   Post-evaluation GPU fallback cascade. Waits for the VoxBind sampling evaluation
#   to finish, then keeps the GPUs busy by trying, in order:
#
#     1. MCP sampling      — FuncBind sample_fb.py (scripts/run_mcpp_paper_run.sh).
#                            If it starts and does real GPU work, we stop: it owns the GPUs.
#     2. frozen-enc VoxBind — scripts/60_train_frozen_efficient60m_4gpu.sh FROM SCRATCH
#                            (WARM_START="" — the ${VAR-default} form makes an explicitly
#                            empty value mean "no warm start"). Gated on 61_cond-style checks.
#     3. original VoxBind   — vanilla denoiser (configs/config_train.yaml defaults:
#                            model.with_density=false, smooth_sigma=0.9), 500 epochs,
#                            from scratch.
#
#   Each stage is *verified* to reach a healthy running state before the cascade stops;
#   a stage that fails to start or dies during the probe window falls through to the next.
#   99_chain.sh runs ONE command under conditions — it has no fallback semantics — hence
#   this separate supervisor. Stage 2/3 launches go through 99_chain.sh so GPU reservation
#   and advisory locking stay consistent with the rest of the pipeline.
#
#   Env knobs:
#     MCP_GRACE       seconds to wait for MCP sampling to appear      (default 1800)
#     MCP_MAX_SM      sm%% below which MCP counts as "not using GPUs" (default 40, matches 61_cond)
#     STAGE_PROBE     seconds to confirm a launched stage is healthy  (default 900)
#     VANILLA_EPOCHS  epochs for stage 3                              (default 500)
#     VANILLA_SIG     smooth_sigma for stage 3                        (default 0.9)
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOGDIR=$ROOT/log
MCP_GRACE=${MCP_GRACE:-1800}
MCP_MAX_SM=${MCP_MAX_SM:-40}
STAGE_PROBE=${STAGE_PROBE:-900}
VANILLA_EPOCHS=${VANILLA_EPOCHS:-500}
VANILLA_SIG=${VANILLA_SIG:-0.9}
mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] [62_fallback] $*"; }

gpu_busy_pct(){   # mean util across GPUs over $1 seconds
  local n=${1:-20}
  for _ in $(seq "$n"); do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd' '
    sleep 1
  done | awk '{for(i=1;i<=NF;i++){s+=$i;c++}} END{if(c)printf "%d", s/c; else print 0}'
}

# ── stage 0: wait for the evaluation's sampling to finish ────────────────────────
log "waiting for VoxBind sampling evaluation to finish"
while pgrep -f "config-name=config_sample" >/dev/null 2>&1; do sleep 60; done
log "sampling evaluation finished"
# docking continues on CPU afterwards; it does not use GPUs, so we do not wait for it.

# ── stage 1: MCP sampling ───────────────────────────────────────────────────────
log "stage 1: waiting up to ${MCP_GRACE}s for MCP sampling (sample_fb.py) to start"
deadline=$(( $(date +%s) + MCP_GRACE ))
mcp_ok=0
while (( $(date +%s) < deadline )); do
  if pgrep -f "sample_fb.py" >/dev/null 2>&1; then
    sm=$(gpu_busy_pct 20)
    if (( sm >= MCP_MAX_SM )); then
      log "stage 1 OK: MCP sampling running and using GPUs (mean util ${sm}%) — cascade stops here"
      mcp_ok=1; break
    fi
    log "  sample_fb.py present but GPU util ${sm}% < ${MCP_MAX_SM}% — still watching"
  fi
  sleep 60
done
if (( mcp_ok )); then log "DONE (MCP owns the GPUs)"; exit 0; fi
log "stage 1 FAILED: no MCP sampling doing GPU work within ${MCP_GRACE}s"

# ── stage 2: frozen-encoder VoxBind, from scratch ───────────────────────────────
EXP2=voxbind_frozen_efficient60m_scratch_sig0.9
log "stage 2: launching frozen-enc VoxBind FROM SCRATCH (exp=$EXP2)"
WARM_START="" EXP_NAME="$EXP2" \
  nohup bash scripts/99_chain.sh --gpus 0-3 --timeout 30m \
    --log "$LOGDIR/${EXP2}.log" -- \
    bash scripts/60_train_frozen_efficient60m_4gpu.sh \
  > "$LOGDIR/${EXP2}_chain.log" 2>&1 &
stage2_pid=$!
log "  chain pid $stage2_pid; probing ${STAGE_PROBE}s for a healthy run"
end=$(( $(date +%s) + STAGE_PROBE ))
stage2_ok=0
while (( $(date +%s) < end )); do
  sleep 60
  if grep -qE 'start training|>> epoch: ' "$ROOT/exps/$EXP2/train_ddp.log" 2>/dev/null \
     && pgrep -f 'train_ddp.py' >/dev/null 2>&1; then
    stage2_ok=1; break
  fi
  kill -0 "$stage2_pid" 2>/dev/null || { log "  stage-2 chain exited early"; break; }
done
if (( stage2_ok )); then
  log "stage 2 OK: frozen-enc training running normally — cascade stops here"
  log "DONE (frozen-enc from-scratch training)"; exit 0
fi
log "stage 2 FAILED: frozen-enc training did not reach a healthy state"
pkill -f "99_chain.sh --gpus 0-3" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -TERM "$p" 2>/dev/null; done
sleep 30

# ── stage 3: original (vanilla) VoxBind, 500 epochs ─────────────────────────────
EXP3=exp_sig${VANILLA_SIG}_${VANILLA_EPOCHS}ep_original
log "stage 3: launching ORIGINAL VoxBind (no density) ${VANILLA_EPOCHS}ep from scratch (exp=$EXP3)"
nohup bash scripts/99_chain.sh --gpus 0-3 --timeout 30m \
  --log "$LOGDIR/${EXP3}.log" -- \
  env LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib TORCHDYNAMO_DISABLE=1 \
  "$PY/torchrun" --standalone --nproc_per_node=4 train_ddp.py \
    smooth_sigma="$VANILLA_SIG" bsz=32 accum_steps=1 num_workers=12 \
    +prefetch_factor=16 num_epochs="$VANILLA_EPOCHS" wjs.n_targets=0 \
    exp_name="$EXP3" output_dir="$ROOT/exps/$EXP3" \
  > "$LOGDIR/${EXP3}_chain.log" 2>&1 &
log "  chain pid $!; probing ${STAGE_PROBE}s"
end=$(( $(date +%s) + STAGE_PROBE ))
while (( $(date +%s) < end )); do
  sleep 60
  if grep -qE 'start training|>> epoch: ' "$ROOT/exps/$EXP3/train_ddp.log" 2>/dev/null; then
    log "stage 3 OK: original VoxBind training running (${VANILLA_EPOCHS} epochs)"
    log "DONE (original VoxBind)"; exit 0
  fi
done
log "stage 3 did not confirm healthy within ${STAGE_PROBE}s — see $LOGDIR/${EXP3}*.log"
exit 1
