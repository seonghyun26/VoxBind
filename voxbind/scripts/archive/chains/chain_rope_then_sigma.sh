#!/usr/bin/env bash
# Orchestrate the RoPE ablation queue, then keep the GPUs busy with the sig=1.0
# denoiser resume. Launch detached:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_rope_then_sigma.sh > log/chain_rope_then_sigma.log 2>&1 &
#
# Sequence (all on GPUs 0,1,2,3, one run at a time):
#   run1 density+rope3d      already running   -> this script only WAITS for it
#   run2 density+learnable   torchrun foreground
#   run3 gradmag+rope3d      torchrun foreground
#   run4 gradmag+learnable   torchrun foreground
#   tail sig=1.0 denoiser    resume ep543 -> 1200 (scripts/archive/launchers/35_train_baseline_4gpu.sh)
#
# Each run blocks until exit; a crash is logged but does NOT abort the chain
# (we keep the GPUs busy and let the user inspect per-run logs).
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"
cd "$ROOT" || exit 1

run_ablation() {  # $1=config-name  $2=exp_name  $3=pos_encoding
  local CFG=$1 EXP=$2 POS=$3
  local OUT=$ROOT/exps/$EXP
  if pgrep -f "exp_name=$EXP" >/dev/null 2>&1; then
    echo "[$(ts)] SKIP $EXP — already running"
    return 0
  fi
  mkdir -p "$OUT"
  echo "[$(ts)] START $EXP  (config=$CFG pos=$POS)  -> $LOG/${EXP}.log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
    "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
    exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
    model.pos_encoding="$POS" bsz=16 accum_steps=2 \
    > "$LOG/${EXP}.log" 2>&1
  local rc=$?
  local last=$(grep -oE "epoch: [0-9]+" "$LOG/${EXP}.log" 2>/dev/null | tail -1)
  echo "[$(ts)] END   $EXP  exit=$rc  (last $last)"
}

# ── 1) wait for run1 (already launched separately) ───────────────────────────
echo "[$(ts)] CHAIN START — waiting for run1 density_only_vit_mae_40m_v5_rope3d ..."
while pgrep -f "exp_name=density_only_vit_mae_40m_v5_rope3d" >/dev/null 2>&1; do
  sleep 60
done
last1=$(grep -oE "epoch: [0-9]+" "$LOG/density_only_vit_mae_40m_v5_rope3d.log" 2>/dev/null | tail -1)
echo "[$(ts)] run1 process gone (last $last1)"

# ── 2) remaining ablation runs, sequentially on all 4 GPUs ────────────────────
run_ablation config_train_density_only_vit_mae_40m_v5 density_only_vit_mae_40m_v5_learnable learnable
run_ablation config_train_gradmag_only_vit_mae_40m_v5 gradmag_only_vit_mae_40m_v5_rope3d    rope3d
run_ablation config_train_gradmag_only_vit_mae_40m_v5 gradmag_only_vit_mae_40m_v5_learnable learnable

# ── 3) keep GPUs busy: resume sig=1.0 denoiser ep543 -> 1200 ──────────────────
echo "[$(ts)] all ablation runs done — resuming sig=1.0 denoiser (ep543 -> 1200)"
SIG=1.0 NUM_EPOCHS=657 RESUME_EPOCH=543 bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
  > "$LOG/sig1.0_resume_chain.log" 2>&1
echo "[$(ts)] sig=1.0 launcher returned exit=$?  -> $LOG/sig1.0_resume_chain.log"
echo "[$(ts)] CHAIN COMPLETE"
