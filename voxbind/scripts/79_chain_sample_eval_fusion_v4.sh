#!/usr/bin/env bash
# 79_chain_sample_eval_fusion_v4.sh
#   Sample the fusion='v4' + CDG_v2 checkpoint on 8 GPUs, then evaluate (Vina + PoseCheck).
#
#   Not 74_chain_sample_eval.sh: that one waits on a running supervisor (this run finished
#   2026-08-31 20:46) and does not pass the x-ray overrides a density model needs.
#
#   WHY XRAY_CROPS IS MANDATORY HERE: sample.py does `cfg = OmegaConf.merge(cfg_model, cfg)`,
#   so the SAMPLE-time config wins over the checkpoint's. config_sample.yaml defaults to
#   `dset: crossdocked`, which silently replaces dset_name=crossdocked_xray -> the batch has
#   no usable map, `xray_available` is False everywhere, and sample.py skips EVERY pocket
#   while still exiting 0. Verified 2026-09-02: with the override the loader yields
#   xray_available for exactly the 79 test pockets that have a v5 crop.
#
#   79, NOT 100: density conditioning only samples pockets that HAVE a map. Test indices 0
#   and 1 have none (first available is 2), 79 of 100 in total. EXPECT_TARGETS=79 keeps 72's
#   completeness check honest instead of failing a correct run. NOTE for the write-up: the
#   density arm therefore covers a 79-pocket SUBSET, so its Vina/PoseCheck numbers are not
#   directly comparable to a 100-pocket baseline — compare on the same 79.
#
#   WORKERS is deliberately below 73's default 48: svr12 is shared and has been fork-bombed
#   into an OOM kill before; at launch the box was already at load 128.
#
#   Env knobs: EXP, OUT, SAMPLES, WORKERS, DOCK, POSE, CPU.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP="${EXP:-260831_fusion_v4_cdgv2_8gpu}"
OUT="${OUT:-samples_ep100_test79}"
SAMPLES="${SAMPLES:-10}"
WORKERS="${WORKERS:-32}"
DOCK="${DOCK:-vina_dock}"
POSE="${POSE:-posecheck}"
CPU="${CPU:-4}"
CROPS="$ROOT/voxbind/dataset/data/pretrain/xray_crops_aligned_v5"
SAVE="$ROOT/voxbind/exps/$EXP/samples/$OUT"
LOG="$ROOT/voxbind/logs/${EXP}.sample_eval.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== chain start: sample -> eval | exp=$EXP out=$OUT ==="

# ── 1. sampling ───────────────────────────────────────────────────────────────
if [ -d "$SAVE" ] && [ "$(ls -d "$SAVE"/target_* 2>/dev/null | wc -l)" -ge 79 ]; then
    say "[1/2] samples already present ($(ls -d "$SAVE"/target_* | wc -l) targets) — skipping"
else
    say "[1/2] sampling 79 density-bearing test pockets x $SAMPLES on 8 GPUs"
    EXP="$EXP" OUT="$OUT" SAMPLES="$SAMPLES" \
      XRAY_CROPS="$CROPS" EXPECT_TARGETS=79 \
      bash scripts/72_sample_8gpu.sh >>"$LOG" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || { say "[1/2] FAILED: 72_sample exited $rc — aborting"; exit 1; }
fi
n_t=$(ls -d "$SAVE"/target_* 2>/dev/null | wc -l)
n_sdf=$(find "$SAVE" -name "*.sdf" 2>/dev/null | wc -l)
say "[1/2] done: $n_t target dirs, $n_sdf sdf files"
[ "$n_t" -ge 1 ] || { say "[1/2] FAILED: no target dirs — aborting"; exit 1; }

# ── 2. evaluation ─────────────────────────────────────────────────────────────
say "[2/2] evaluating (dock=$DOCK pose=$POSE workers=$WORKERS)"
SAMPLE_DIR="$SAVE" DOCK="$DOCK" POSE="$POSE" WORKERS="$WORKERS" CPU="$CPU" \
  bash scripts/73_evaluate_samples.sh >>"$LOG" 2>&1
say "[2/2] eval exited $?"
say "=== chain done -> $SAVE ==="
