#!/usr/bin/env bash
# 74_dock_targetdiff_full79.sh — re-dock TargetDiff under the same protocol as
# 73_dock_baseline_protocol_79.sh, so its row can join the primary Table 1.
#
# Protocol (matches 260903/baseline.html and our vanilla / Ours v1 runs):
#   whole receptor (*_rec.pdb), exhaustiveness 32, ligand-centred box + 5 Å,
#   the 79 density pockets in p79_targets.json.
#
# The samples live under base_drug (TargetDiff's own eval tree), which already has
# the target_NN/{samples.sdf,*_ref.sdf,*_pocket10.pdb} layout the driver expects.
# Pre-flighted 2026-08-31: all 79 targets resolve, and every receptor matches the
# one the VoxBind runs used, so the three rows are docked against identical proteins.
#
# The existing crop result (eval_docking_results.json, 100 targets, exhaustiveness 16)
# is left untouched — it still backs Table A5.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/voxdock/bin:$PATH"

V=voxbind/exps
DRIVER=$V/frozenenc_probes/run_docking_eval.py
TARGETS=${TARGETS:-$V/frozenenc_probes/p79_targets.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/baseproto}
DIR=${DIR:-/home1/irteam/base_drug/eval/targetdiff}
mkdir -p "$LOGDIR"
# 64-core cgroup quota; the FuncBind trainer sits at ~12, so 14x3 leaves real headroom.
W=${W:-14}; C=${C:-3}; EXH=${EXH:-32}

echo "[$(date '+%m-%d %H:%M:%S')] START targetdiff (scope=full exh=$EXH W=$W C=$C)" | tee -a "$LOGDIR/driver.log"
/opt/conda/envs/voxdock/bin/python "$DRIVER" "$DIR" \
    --scope full --workers "$W" --cpu "$C" --exhaustiveness "$EXH" \
    --targets "$TARGETS" --skip-existing \
    --out "$DIR/eval_docking_results_full79.json" \
    > "$LOGDIR/targetdiff.log" 2>&1
rc=$?               # capture BEFORE anything else runs, or we log tee's status
echo "[$(date '+%m-%d %H:%M:%S')] DONE  targetdiff rc=$rc" | tee -a "$LOGDIR/driver.log"
[ "$rc" -eq 0 ] || echo "[$(date '+%m-%d %H:%M:%S')] WARNING: targetdiff exited $rc" | tee -a "$LOGDIR/driver.log"
echo "TARGETDIFF_FULL79_DONE" >> "$LOGDIR/driver.log"
