#!/usr/bin/env bash
# 88_chain_svr12_arms_eval.sh — finish the evaluation of the two arms that were sampled on
# svr12 and never fully scored anywhere: `VoxBind-vanilla` and `VoxBind-base-ep350-n100`.
#
# WHAT WAS MISSING, out of results/task2-drugdesign/EVAL_STATUS.md (2026-09-12):
#   VoxBind-vanilla            PoseBusters —   PoseCheck —   Vina: metrics.py, crop scope
#   VoxBind-base-ep350-n100    both present                  Vina: metrics.py, crop scope
# so the pose gap is vanilla's alone, and the Vina gap is that neither arm was ever docked
# under the published-baseline protocol. Their Vina numbers therefore do NOT sit on the
# axis the AR / Pocket2Mol / DiffSBDD / DecompDiff / FuncBind rows are on.
#
# BOTH SAMPLE TREES ARE STAGED FROM DROPBOX, not regenerated — the checkpoints are on
# svr12, these are the same molecules the bundle reports:
#   results/dropbox_pull.sh VoxBind-vanilla   -> exps/reproduction/samples/res_test_100
#   results/dropbox_pull.sh VoxBind-base-ep350-n100
#                                -> exps/260827_voxbind_base_8gpu/samples/..._test100_n100
#
# ORDER, and why it is a chain rather than two commands: the pose pass (85) and the docking
# driver both saturate what this box has spare -- a 4-GPU trainer is resident and the cgroup
# quota is 64 cores -- so running them together just makes both slower and starves the
# trainer. This waits for 85 to put POSEFILL_DONE in its progress file, then docks.
#
#   bash voxbind/scripts/88_chain_svr12_arms_eval.sh
#   W=6 C=2 bash voxbind/scripts/88_chain_svr12_arms_eval.sh   # if the box gets busier
#   SKIP_WAIT=1 bash voxbind/scripts/88_chain_svr12_arms_eval.sh   # pose fill already done
#
# Vina here is the BASELINE PROTOCOL, which is what "TargetDiff-standard" means for this
# table: whole *_rec.pdb receptor, exhaustiveness 32, the 79 density pockets, through
# frozenenc_probes/run_docking_eval.py -- the same driver and flags 73_dock_baseline_
# protocol_79.sh and 74_dock_targetdiff_full79.sh used for the arms already on that axis.
# It writes eval_docking_results_full79.json and leaves the existing crop-scope
# metrics.json Vina rows untouched, exactly as 73 does.
set -uo pipefail
cd /home1/irteam/VoxBind

V=voxbind/exps
VANILLA100=$V/reproduction/samples/res_test_100
BASE100=$V/260827_voxbind_base_8gpu/samples/samples_ep350_test100_n100
POSELOG=${POSELOG:-$V/frozenenc_probes/logs/posefill_vanilla100}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/svr12arms}
POLL=${POLL:-120}
TIMEOUT=${TIMEOUT:-86400}
SKIP_WAIT=${SKIP_WAIT:-0}
# Deliberately under 73's W=16 C=3: that was written for an idle box. 8x2 = 16 Vina threads
# leaves the trainer its ~15 and still keeps half the quota free.
W=${W:-8}; C=${C:-2}

mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── 1. wait for the pose fill ────────────────────────────────────────────────────
if [ "$SKIP_WAIT" != "1" ]; then
    say "waiting for the pose fill to finish (poll ${POLL}s, $POSELOG/progress.txt)"
    waited=0
    while ! grep -q POSEFILL_DONE "$POSELOG/progress.txt" 2>/dev/null; do
        if (( waited >= TIMEOUT )); then
            say "TIMEOUT after ${waited}s — pose fill still running, not starting the dock"
            exit 1
        fi
        sleep "$POLL"; waited=$((waited + POLL))
    done
    ok=$(grep -c '^OK'   "$POSELOG/progress.txt" 2>/dev/null || echo 0)
    bad=$(grep -c '^FAIL' "$POSELOG/progress.txt" 2>/dev/null || echo 0)
    say "pose fill done after ${waited}s: ok=$ok failed=$bad"
    [ "$bad" -gt 0 ] && say "NOTE $bad target(s) failed the pose pass — see $POSELOG"
fi

# ── 2. dock both arms under the baseline protocol ────────────────────────────────
ARMS_LIST=""
for pair in "vanilla_res100 $VANILLA100" "base_ep350_n100 $BASE100"; do
    set -- $pair
    if [ -d "$2" ]; then ARMS_LIST+="$1 $2"$'\n'
    else say "SKIP $1 — $2 not staged"; fi
done
[ -n "$ARMS_LIST" ] || { say "nothing staged to dock"; exit 1; }

say "docking (baseline protocol: scope=full exh=32 W=$W C=$C, 79 pockets)"
printf '%s' "$ARMS_LIST" | sed 's/^/    /' | tee -a "$LOG"
ARMS="$ARMS_LIST" W=$W C=$C LOGDIR=$LOGDIR \
    nice -n 15 bash voxbind/scripts/73_dock_baseline_protocol_79.sh
say "docking driver returned $?"

# ── 3. stage the scored trees back into the results bundle ───────────────────────
# collect_task2_eval.py reads <bundle>/<Method>/samples/target_*/metrics.json and
# <bundle>/<Method>/samples/eval_docking_results_full79.json -- NOT voxbind/exps. Both of
# these arms were staged INTO exps to be scored (that is where 85 and the docking driver
# work), so the new pose blocks and the new docking JSON have to come back or the collector
# will re-emit yesterday's numbers and nothing will look like it changed.
#
# The bundle is git-ignored and Dropbox-backed, so this is a plain copy, not a link: a
# symlink here would break results/dropbox_push.sh.
stage_back() {
    local src=$1 dst=$2
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    say "staging $src -> $dst"
    cp -a "$src/." "$dst/"
}
stage_back "$VANILLA100" results/task2-drugdesign/VoxBind-vanilla/samples
stage_back "$BASE100"    results/task2-drugdesign/VoxBind-base-ep350-n100/samples

# ── 4. re-collect the bundle's evaluation JSONs ──────────────────────────────────
say "re-collecting results/task2-drugdesign/*/eval"
/opt/conda/envs/voxbind/bin/python voxbind/scripts/tools/collect_task2_eval.py \
    >> "$LOG" 2>&1 || say "collect_task2_eval.py returned $? — see $LOG"
say "SVR12ARMS_DONE"
echo "SVR12ARMS_DONE" >> "$LOG"
