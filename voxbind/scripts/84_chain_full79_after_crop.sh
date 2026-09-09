#!/usr/bin/env bash
# 84_chain_full79_after_crop.sh — after the crop/exh16 re-dock finishes, re-check the OTHER
# published protocol: full receptor at exhaustiveness 32.
#
# Why both. The two live results pages quote DIFFERENT protocols off the same molecules:
#   notebook/html/results.html        Table 4 "Ours"  crop (pocket10) / exh 16
#                                     -5.51 / -7.09 / -8.24, HA 67.5%   -> eval_docking_results.json
#   notebook/html/results_drug_design.html            full receptor / exh 32
#                                     -6.58 / -7.64 / -8.49, HA 68.1%   -> eval_docking_results_full79.json
# results_drug_design.html is the baseline-comparable one ("the same protocol the published
# baselines use"), so it is the number that has to hold up; results.html Table 4 is the older
# crop protocol. Verifying only one of them leaves the other unchecked.
#
#   bash scripts/84_chain_full79_after_crop.sh
#
# Runs serially, never alongside the crop job: the box is capped at 32 cores by its cgroup
# quota and a co-running 4-GPU training already takes ~16-20 of them. Two 20-worker docking
# runs at once would thrash both.
set -uo pipefail
cd /home1/irteam/VoxBind

V=voxbind/exps
DIR=${DIR:-$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/redock_full_exh32}
POLL=${POLL:-300}
TIMEOUT=${TIMEOUT:-172800}
W=${W:-20}

mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the crop re-dock driver to exit before starting full/exh32"
waited=0
while pgrep -f "run_docking_eval.py.*full_eval_ep350" >/dev/null 2>&1; do
    if (( waited >= TIMEOUT )); then
        say "TIMEOUT after ${waited}s - crop run still going, not starting full79"
        exit 1
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done
say "crop run finished after ${waited}s of waiting; starting full/exh32"

# Same driver, other protocol. OUT is a _rerun_ file so the published
# eval_docking_results_full79.json stays intact for the diff.
SCOPE=full EXH=32 W="$W" \
    OUT="$DIR/eval_docking_results_full79_rerun.json" \
    bash voxbind/scripts/82_redock_ours_crop_exh16_79.sh
rc=$?
say "full/exh32 re-dock exit=$rc"

if [ "$rc" -eq 0 ] || [ -f "$DIR/eval_docking_results_full79_rerun.json" ]; then
    say "comparing against the published full79 numbers"
    /opt/conda/envs/voxdock/bin/python voxbind/scripts/tools/compare_redock_vs_published.py \
        "$DIR/eval_docking_results_full79.json" \
        "$DIR/eval_docking_results_full79_rerun.json" \
        --md "$LOGDIR/redock_full79_vs_published.md" >>"$LOG" 2>&1
    say "report -> $LOGDIR/redock_full79_vs_published.md"
fi
say "FULL79_CHAIN_DONE"
exit $rc
