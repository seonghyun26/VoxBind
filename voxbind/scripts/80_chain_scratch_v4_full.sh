#!/usr/bin/env bash
# 80_chain_scratch_v4_full.sh
#   fusion='v4' + frozen CDG_v2 trained FROM SCRATCH on 8 GPUs, then sampled and evaluated.
#
#   WHY FROM SCRATCH: the 100-epoch warm-started arm (260831_fusion_v4_cdgv2_8gpu) came out
#   0.48 kcal/mol WORSE than the vanilla e349 baseline on Vina dock over the same 79 pockets,
#   while its val miou rose. Warm-starting from that very baseline and then fine-tuning on the
#   78.5k x-ray subset confounds three things — density conditioning, the subset restriction,
#   and 100 extra epochs of drift away from a converged model. Training from scratch removes
#   the drift term: the density branch is learned jointly with the denoiser instead of being
#   bolted onto a model that already converged without it.
#
#   350 EPOCHS, NOT 100: the vanilla baseline it is compared against is a 350-epoch
#   from-scratch run. A 100-epoch from-scratch arm would be undertrained and any gap would
#   just measure that. At the measured 686 s/epoch this is ~67 h (~2.8 days).
#
#   WARM_START="" is what selects scratch. Note 75 uses ${WARM_START-default}, NOT
#   ${WARM_START:-default}, precisely so an explicit empty string overrides instead of
#   falling through to the warm-start checkpoint — verified via DRY_RUN (pretrained_path: null).
#   The variable is exported so it survives into 76 -> 75.
#
#   Env knobs: EXP_NAME, TOTAL_EPOCHS, OUT, SAMPLES.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP_NAME="${EXP_NAME:-260902_fusion_v4_cdgv2_scratch_8gpu}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-350}"
OUT="${OUT:-samples_ep${TOTAL_EPOCHS}_test79}"
SAMPLES="${SAMPLES:-10}"

CKPT="$ROOT/voxbind/exps/$EXP_NAME/checkpoint.pth.tar"
LOG="$ROOT/voxbind/logs/${EXP_NAME}.chain.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== scratch chain start: exp=$EXP_NAME epochs=$TOTAL_EPOCHS ==="

# ── 1. training (from scratch, crash-resume supervised) ───────────────────────
export WARM_START=""
say "[1/3] training from scratch, $TOTAL_EPOCHS epochs (~$((TOTAL_EPOCHS * 686 / 3600)) h)"
EXP_NAME="$EXP_NAME" TOTAL_EPOCHS="$TOTAL_EPOCHS" \
    bash scripts/76_supervise_fusion_v4_8gpu.sh >>"$LOG" 2>&1
rc=$?
say "[1/3] supervisor exited $rc"

# The supervisor also exits non-zero after MAX_RETRIES, and sampling a half-trained
# checkpoint silently produces numbers that look real — so gate on the EPOCH, not the
# exit code.
ep=$("$HOME/miniforge3/envs/voxbind/bin/python" - "$CKPT" <<'PY' 2>/dev/null || echo -1
import sys, torch
try: print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"]))
except Exception: print(-1)
PY
)
say "[1/3] checkpoint epoch = $ep (need >= $((TOTAL_EPOCHS - 1)))"
[ "$ep" -ge $((TOTAL_EPOCHS - 1)) ] || { say "[1/3] ABORT: training did not reach the final epoch"; exit 1; }

# ── 2+3. sampling + evaluation ────────────────────────────────────────────────
# 79 handles the x-ray dset overrides sample.py needs (without them EVERY pocket is
# skipped and the run still exits 0) and expects 79 density-bearing test pockets.
say "[2/3] sampling + [3/3] eval via 79_chain_sample_eval_fusion_v4.sh"
EXP="$EXP_NAME" OUT="$OUT" SAMPLES="$SAMPLES" \
    bash scripts/79_chain_sample_eval_fusion_v4.sh >>"$LOG" 2>&1
say "[2/3+3/3] sample+eval exited $?"
say "=== scratch chain done: exps/$EXP_NAME/samples/$OUT ==="
