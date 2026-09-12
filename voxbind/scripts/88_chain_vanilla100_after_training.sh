#!/usr/bin/env bash
# 88_chain_vanilla100_after_training.sh
#   Re-sample the VANILLA arm at 100 molecules/pocket once the GPUs are free, then
#   evaluate it on the same footing as everything else in the results bundle.
#
#   WHY: 260827_voxbind_base_8gpu is the vanilla VoxBind row -- density-free, ICML'24
#   recipe, trained here for 350 epochs. It is the right vanilla model (exp_sig0.9 /
#   model_zoo/voxbind_sig0.9_crossdocked is only EPOCH 86, and its Vina was scored
#   against the pocket10 crop). What it lacks is scale: it was sampled at the shipped
#   default of 10 molecules/pocket, so 1,000 molecules against the baselines' 6.4k-9.9k.
#   This fills that in: 100 pockets x 100 molecules, full-receptor Vina + PoseCheck +
#   PoseBusters, the protocol the four arms and five baselines already use.
#
#   NOT script 82/79: those carry the x-ray overrides a density model needs and stop at
#   the 79 map-bearing pockets. The vanilla model is density-free, so it takes the plain
#   path -- 72 (sample, all 100 pockets) then 73 (eval) -- and covers all 100, which also
#   lets the density79 subset be cut from it later.
#
#   WAITS FOR THE GPUS. 260908_fusion_default_cv2_scratch_8gpu is training on all 8
#   (~70 GB each); starting now would just OOM. This polls until that run's python
#   processes are gone, then smoke-tests 2 pockets at the new width before committing.
#
#   WIDTH IS THE RISK: sampling_utils sets n_chains = n_samples_per_pocket, so the WJS
#   batch goes from 10 to 100 wide. Every run on this box so far was 10. The smoke test
#   is there to find an OOM in 15 minutes instead of 15 hours. A density-free UNet has
#   far more headroom than the fusion arms (no frozen ViT), so it is expected to fit.
#
#   COST after the wait, extrapolated from the 79x10 run (2.0 h sampling, 2.1 h eval at
#   WORKERS=32): roughly 10-20 h sampling + ~20 h eval. Budget ~2 days.
#
#   Env knobs: EXP, OUT, SAMPLES, WAIT_EXP, POLL, WORKERS, CPU, SKIP_SMOKE, DRY_RUN
#   Cancel:    kill this script's PID (printed at start); nothing is left half-written
#              that 72/73 cannot resume.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP="${EXP:-260827_voxbind_base_8gpu}"
SAMPLES="${SAMPLES:-100}"
OUT="${OUT:-samples_ep350_test100_n${SAMPLES}}"
WAIT_EXP="${WAIT_EXP:-260908_fusion_default_cv2_scratch_8gpu}"
POLL="${POLL:-600}"
WORKERS="${WORKERS:-32}"
CPU="${CPU:-4}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SAVE="$ROOT/voxbind/exps/$EXP/samples/$OUT"
LOG="$ROOT/voxbind/logs/vanilla100.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -f "exps/$EXP/checkpoint.pth.tar" ] || { say "MISSING exps/$EXP/checkpoint.pth.tar"; exit 1; }

say "=== vanilla resample: $EXP @ ${SAMPLES}/pocket -> samples/$OUT (pid $$)"
if [ -n "${DRY_RUN:-}" ]; then
    say "DRY_RUN plan:"
    say "  wait for: $WAIT_EXP (poll ${POLL}s)"
    say "  smoke   : EXP=$EXP OUT=${OUT}_smoke SAMPLES=$SAMPLES N_POCKETS=2 NTARGETS=2 bash scripts/72_sample_8gpu.sh"
    say "  sample  : EXP=$EXP OUT=$OUT SAMPLES=$SAMPLES bash scripts/72_sample_8gpu.sh"
    say "  eval    : SAMPLE_DIR=exps/$EXP/samples/$OUT DOCK=vina_dock POSE=all WORKERS=$WORKERS CPU=$CPU bash scripts/73_evaluate_samples.sh"
    exit 0
fi

# ── wait for the training run to release the GPUs ───────────────────────────────
while pgrep -f "train_ddp.py.*$WAIT_EXP" >/dev/null 2>&1 \
   || pgrep -f "exps/$WAIT_EXP" >/dev/null 2>&1; do
    say "waiting: $WAIT_EXP still on the GPUs"
    sleep "$POLL"
done
# a DDP run can linger a moment while it writes its final checkpoint
sleep 60
free_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
say "GPUs free (max used across cards: ${free_mb} MiB)"

# ── smoke: 2 pockets at the new chain width, to catch an OOM early ──────────────
if [ "$SKIP_SMOKE" != "1" ]; then
    say "smoke: 2 pockets at ${SAMPLES}/pocket"
    EXP="$EXP" OUT="${OUT}_smoke" SAMPLES="$SAMPLES" N_POCKETS=2 NTARGETS=2 \
        bash scripts/72_sample_8gpu.sh >>"$LOG" 2>&1
    rc=$?
    n=$(ls -d "$ROOT/voxbind/exps/$EXP/samples/${OUT}_smoke"/target_*/ 2>/dev/null | wc -l)
    say "smoke exited $rc, $n target dir(s)"
    if [ "$rc" -ne 0 ] || [ "$n" -lt 1 ]; then
        say "SMOKE FAILED — stopping before the long run. Check $LOG (OOM at n_chains=$SAMPLES?)"
        exit 1
    fi
fi

# ── the real run ────────────────────────────────────────────────────────────────
say "sampling: 100 pockets x ${SAMPLES}"
EXP="$EXP" OUT="$OUT" SAMPLES="$SAMPLES" bash scripts/72_sample_8gpu.sh >>"$LOG" 2>&1
say "sampling exited $? ($(ls -d "$SAVE"/target_* 2>/dev/null | wc -l) target dirs)"

say "eval: full-receptor vina_dock + PoseCheck + PoseBusters (workers=$WORKERS cpu=$CPU)"
SAMPLE_DIR="exps/$EXP/samples/$OUT" DOCK=vina_dock POSE=all \
    WORKERS="$WORKERS" CPU="$CPU" nice -n 10 bash scripts/73_evaluate_samples.sh >>"$LOG" 2>&1
say "eval exited $? -> $SAVE/summary.json"

say "=== done. stage + upload with:"
say "    bash voxbind/scripts/86_stage_results_bundle.sh && bash results/dropbox_push.sh"
