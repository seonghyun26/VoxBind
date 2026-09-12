#!/usr/bin/env bash
# 82_chain_resample100.sh
#   Re-sample the finished fusion-v4 arms at 100 molecules/pocket and re-evaluate, so
#   their row in tab:result-drug-design carries an n comparable to the published rows.
#
#   WHY: every arm so far ran the SHIPPED default wjs.n_samples_per_pocket=10, i.e.
#   79 pockets x 10 = 790 draws (788 kept for C_v2 — target_66 exhausted the 500-candidate
#   cap on duplicates). The baselines in that table are 6.4k-7.9k molecules and
#   VoxBind/Ours ~7.9k, all at 100/pocket. 790 vs 7,900 is not a truncated run, it is a
#   different protocol; the arms are fine against EACH OTHER but a 790-molecule row
#   cannot sit in that table.
#
#   EVERYTHING ELSE IS THE SHIPPED CONFIG. Only wjs.n_samples_per_pocket moves, exactly
#   as 72 already supports via SAMPLES; configs/wjs/sampling.yaml is untouched. Two
#   consequences worth knowing before launching, both stock behaviour:
#     * sampling_utils.py sets n_chains = n_samples_per_pocket, so the WJS batch pushed
#       through the U-Net (and, on a density model, the frozen ViT) becomes 100 wide.
#       Every run recorded on this box was 10. If it OOMs, that is where.
#     * max_batches = ceil(500 / n_chains) = 5, and the loop still stops at 500
#       voxel->mol attempts, so the candidate budget per pocket is unchanged. Pockets
#       that cannot yield 100 unique molecules inside it set hit_cap in gen_stats.json —
#       expect a yield below 7,900, the same way the published rows sit below 100 x n.
#
#   COST, extrapolated from the 10/pocket run (2026-09-08: 79x10 = 2.0 h sampling on
#   8 GPUs + 2.1 h eval at WORKERS=32): candidates scale with molecules wanted, so budget
#   roughly 10-20 h sampling and ~20 h eval PER ARM.
#
#   Env knobs:
#     ARMS       space-separated exp dir names   (default: the two v4 scratch arms)
#     SAMPLES    molecules per pocket            (100)
#     OUT_SUFFIX appended to the sample dir name (_n${SAMPLES})
#     BASE_OUT   sample dir stem                 (samples_ep350_test79)
#     WORKERS/CPU/DOCK/POSE  passed through to 79 -> 73
#     DRY_RUN=1  print the plan and the exact commands, run nothing
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

ARMS="${ARMS:-260905_fusion_v4_cv2_scratch_8gpu 260902_fusion_v4_cdgv2_scratch_8gpu}"
SAMPLES="${SAMPLES:-100}"
BASE_OUT="${BASE_OUT:-samples_ep350_test79}"
OUT_SUFFIX="${OUT_SUFFIX:-_n${SAMPLES}}"
WORKERS="${WORKERS:-32}"
CPU="${CPU:-4}"
DOCK="${DOCK:-vina_dock}"
POSE="${POSE:-posecheck}"
EXPECT=79        # density-bearing test pockets; 79 hardcodes the same number

LOG="$ROOT/voxbind/logs/resample${SAMPLES}.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for exp in $ARMS; do
    [ -f "$ROOT/voxbind/exps/$exp/checkpoint.pth.tar" ] \
        || { say "MISSING exps/$exp/checkpoint.pth.tar"; exit 1; }
done

say "=== resample at ${SAMPLES}/pocket ==="
say "  arms : $ARMS"
say "  out  : exps/<arm>/samples/${BASE_OUT}${OUT_SUFFIX}"
say "  eval : dock=$DOCK pose=$POSE workers=$WORKERS cpu=$CPU"

for exp in $ARMS; do
    OUT="${BASE_OUT}${OUT_SUFFIX}"
    SAVE="$ROOT/voxbind/exps/$exp/samples/$OUT"

    # 79 decides "already sampled" from the target_* dir COUNT, but a dir appears with
    # its first molecule — so an interrupted run reads as complete and gets evaluated
    # partial. gen_stats.json is written only after a pocket finishes its WJS loop, so
    # check that here rather than patching the shared script.
    n_dirs=$(ls -d "$SAVE"/target_* 2>/dev/null | wc -l)
    n_done=$(ls "$SAVE"/target_*/gen_stats.json 2>/dev/null | wc -l)
    if [ "$n_dirs" -ge "$EXPECT" ] && [ "$n_done" -lt "$n_dirs" ]; then
        say "[$exp] PARTIAL: $n_dirs target dirs but only $n_done finished."
        say "[$exp] 79 would skip sampling and evaluate the partial set. Remove it first:"
        say "[$exp]   rm -rf $SAVE"
        continue
    fi

    say "[$exp] sampling ${SAMPLES}/pocket + eval via 79"
    cmd=(env EXP="$exp" OUT="$OUT" SAMPLES="$SAMPLES"
         WORKERS="$WORKERS" CPU="$CPU" DOCK="$DOCK" POSE="$POSE"
         bash scripts/79_chain_sample_eval_fusion_v4.sh)
    if [ -n "${DRY_RUN:-}" ]; then
        say "[$exp] DRY_RUN: ${cmd[*]}"
        continue
    fi
    "${cmd[@]}" >>"$LOG" 2>&1
    say "[$exp] exited $? -> $SAVE/summary.json"
done

[ -n "${DRY_RUN:-}" ] && { say "DRY_RUN — nothing executed"; exit 0; }
say "=== done. summaries:"
for exp in $ARMS; do
    f="$ROOT/voxbind/exps/$exp/samples/${BASE_OUT}${OUT_SUFFIX}/summary.json"
    [ -f "$f" ] && say "  $f" || say "  MISSING $f"
done
