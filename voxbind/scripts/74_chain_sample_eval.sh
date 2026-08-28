#!/usr/bin/env bash
# 74_chain_sample_eval.sh — wait for the supervised training run to finish, then
# sample all 100 test pockets on 8 GPUs (72) and evaluate them (73).
#
#   EXP=260827_voxbind_base_8gpu bash scripts/74_chain_sample_eval.sh
#
# Training is GPU-bound, sampling is GPU-bound, docking is CPU-bound, so chaining
# them here keeps the box busy instead of leaving 8 GPUs idle overnight waiting
# for someone to notice epoch 349 landed.
#
# "Finished" means BOTH: the supervisor process is gone AND the checkpoint reached
# the final epoch. The supervisor also exits non-zero when it gives up after
# MAX_RETRIES, and sampling a half-trained checkpoint silently produces numbers
# that look real — so the epoch check is the gate, not the process exit.
#
# Env knobs: EXP, OUT, TOTAL_EPOCHS, SAMPLES, DOCK, WORKERS, CPU, POLL.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP="${EXP:-260827_voxbind_base_8gpu}"
OUT="${OUT:-samples_ep350_test}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-350}"
SAMPLES="${SAMPLES:-10}"     # configs/wjs/sampling.yaml default
DOCK="${DOCK:-vina_dock}"
WORKERS="${WORKERS:-48}"
CPU="${CPU:-4}"
POLL="${POLL:-300}"

CKPT="$ROOT/voxbind/exps/$EXP/checkpoint.pth.tar"
SAVE="$ROOT/voxbind/exps/$EXP/samples/$OUT"
LOG="$ROOT/voxbind/logs/${EXP}.chain.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

ckpt_epoch() {
    [ -f "$CKPT" ] || { echo -1; return; }
    "$HOME/miniforge3/envs/voxbind/bin/python" - "$CKPT" <<'PY' 2>/dev/null || echo -1
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"]))
except Exception:
    print(-1)
PY
}

say "chain start: exp=$EXP out=$OUT target_epoch=$((TOTAL_EPOCHS - 1)) samples/pocket=$SAMPLES dock=$DOCK"

# ── Phase 0: wait for training ────────────────────────────────────────────────
while :; do
    sup_alive=$(pgrep -fc "71_supervise_voxbind_base_8gpu.sh" || true)
    e=$(ckpt_epoch)
    if [ "$sup_alive" -eq 0 ]; then
        if [ "$e" -ge $((TOTAL_EPOCHS - 1)) ]; then
            say "training complete at epoch $e"
            break
        fi
        say "ABORT: supervisor gone but checkpoint only at epoch $e (< $((TOTAL_EPOCHS - 1)))"
        exit 1
    fi
    say "waiting: training at epoch $e / $((TOTAL_EPOCHS - 1))"
    sleep "$POLL"
done

# GPUs must actually be free before 8 sampling processes pile on.
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do
    say "waiting for GPUs to drain"
    sleep 60
done

# ── Phase 1: sample ───────────────────────────────────────────────────────────
say "phase 1: sampling 100 test pockets across 8 GPUs -> $SAVE"
EXP="$EXP" OUT="$OUT" SAMPLES="$SAMPLES" bash scripts/72_sample_8gpu.sh >> "$LOG" 2>&1
rc=$?
say "phase 1 done (exit $rc)"
[ "$rc" -ne 0 ] && { say "sampling failed — not evaluating"; exit "$rc"; }

# ── Phase 2: evaluate ─────────────────────────────────────────────────────────
say "phase 2: evaluation ($DOCK)"
SAMPLE_DIR="$SAVE" DOCK="$DOCK" WORKERS="$WORKERS" CPU="$CPU" \
    bash scripts/73_evaluate_samples.sh >> "$LOG" 2>&1
rc=$?
say "phase 2 done (exit $rc)"
[ "$rc" -ne 0 ] && exit "$rc"

say "CHAIN COMPLETE -> $SAVE/summary.json"
