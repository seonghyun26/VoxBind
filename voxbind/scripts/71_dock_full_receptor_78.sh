#!/usr/bin/env bash
# 71_dock_full_receptor_78.sh — re-dock the 78-pocket set against the WHOLE
# CrossDocked receptor instead of the pocket10 crop.
#
# WHY: every Vina number in this project so far was scored on the crop. The paired
# 2026-07-29 audit found crop scoring is biased -- Dock -0.16 overall, and the shift
# is model-dependent (vanilla -0.298 vs the density model -0.020 on the hardest
# pockets), because the crop deletes 35-64% of atoms within Vina's interaction range
# and so HIDES clashes that larger ligands would incur. The density model draws the
# larger molecules, so the crop may be flattering exactly the comparison we care
# about. This settles it on the full 78.
#
# COST: measured 2026-08-26 -- prep is 13.9x slower on the full receptor but the Vina
# search is 1.01x, because the affinity grid is built over the ligand-derived box,
# which is identical for both. At 100 mols/pocket that is +2.7% wall-clock.
#
# Writes eval_docking_results_full.json; the crop results are left untouched so the
# two can be compared rather than one overwriting the other.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/voxdock/bin:$PATH"      # pdb2pqr30 + prepare_receptor4

V=voxbind/exps
DRIVER=$V/frozenenc_probes/run_docking_eval.py
TARGETS=${TARGETS:-/home1/irteam/VoxBind/voxbind/exps/frozenenc_probes/p78_targets.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/fullrec}
mkdir -p "$LOGDIR"

# workers x cpu must stay inside the 64-core cgroup quota; 16x3 = 48 leaves room for
# the FuncBind trainer's dataloaders.
W=${W:-16}
C=${C:-3}

run() {
    local name=$1 dir=$2
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$LOGDIR/driver.log"
    /opt/conda/envs/voxdock/bin/python "$DRIVER" "$dir" \
        --scope full --workers "$W" --cpu "$C" \
        --targets "$TARGETS" --skip-existing \
        > "$LOGDIR/$name.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $name rc=$? -> $dir/eval_docking_results_full.json" \
        | tee -a "$LOGDIR/driver.log"
}

# Sequential across runs, parallel within: two runs at 12 workers each would oversubscribe.
run vanilla "$V/_vanilla_ep923/samples/full_eval_ep923"
run ours_v1 "$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
echo "FULLREC_DONE" >> "$LOGDIR/driver.log"
