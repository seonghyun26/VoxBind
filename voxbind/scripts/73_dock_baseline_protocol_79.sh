#!/usr/bin/env bash
# 73_dock_baseline_protocol_79.sh — re-dock vanilla VoxBind and Ours v1 under the
# protocol 260903/baseline.html used, so our rows and the five baselines can sit in
# one table.
#
# baseline.html: full receptor (*_rec.pdb), AutoDock Vina 1.2.2, exhaustiveness 32,
# ligand-centred box + 5 Å, 79 density pockets INCLUDING target_71 (the crop is what
# made that pocket unscoreable; the whole receptor preps fine).
#
# Our existing numbers differ on all three axes -- pocket10 crop, exhaustiveness 16,
# 78 pockets -- and the gap is not negligible: the same crystal reference ligand scores
# -7.18 our way and -7.32 theirs. Mixing them would hand the baselines 0.14 kcal/mol.
#
# Writes eval_docking_results_full79.json; the crop and the earlier exh-16 full results
# are both left in place.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/voxdock/bin:$PATH"

V=voxbind/exps
DRIVER=$V/frozenenc_probes/run_docking_eval.py
TARGETS=${TARGETS:-$V/frozenenc_probes/p79_targets.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/baseproto}
mkdir -p "$LOGDIR"
W=${W:-16}; C=${C:-3}; EXH=${EXH:-32}

run() {
    local name=$1 dir=$2
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$LOGDIR/driver.log"
    /opt/conda/envs/voxdock/bin/python "$DRIVER" "$dir" \
        --scope full --workers "$W" --cpu "$C" --exhaustiveness "$EXH" \
        --targets "$TARGETS" --skip-existing \
        --out "$dir/eval_docking_results_full79.json" \
        > "$LOGDIR/$name.log" 2>&1
    local rc=$?          # capture BEFORE anything else runs, or we log tee's status
    echo "[$(date '+%H:%M:%S')] DONE  $name rc=$rc" | tee -a "$LOGDIR/driver.log"
    [ "$rc" -eq 0 ] || echo "[$(date '+%H:%M:%S')] WARNING: $name exited $rc" | tee -a "$LOGDIR/driver.log"
}

run vanilla "$V/_vanilla_ep923/samples/full_eval_ep923"
run ours_v1 "$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
echo "BASEPROTO_DONE" >> "$LOGDIR/driver.log"
