#!/usr/bin/env bash
# 70_reseed_posecheck_78.sh — recompute PoseCheck for every run on the 78-pocket set
# with PoseCheck's conformer embedding seeded (randomSeed=0).
#
# WHY: posecheck/utils/strain.py ships the seeded embed call commented out, so strain
# is the difference between a constrained relax and an UNSEEDED global relax. Measured
# twice, the same molecule gives different strain — our 78 reference ligands moved
# median 33.8 vs 34.6 between run roots. pose_eval._seed_posecheck_strain() wraps
# EmbedMolecule to force randomSeed=0; this script re-measures everything under it.
#
# --force is required: the metrics cache keys on samples.sdf, which has not changed,
# so without it every cached (unseeded) posecheck block is reused verbatim.
#
# Targets run in parallel at the TARGET level, N at a time. One target is ~390 s
# single-core, so N=24 turns ~34 h of serial work into ~1.5 h while leaving the box
# far below its 64-core cgroup quota.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/moleval/bin:$PATH"     # hydride + reduce must be findable

V=voxbind/exps
N=${N:-24}
TARGETS=${TARGETS:-/tmp/claude-500/-home1-irteam-VoxBind/0d116ebb-2481-4cc2-9ea9-5e71711288ac/scratchpad/p78_targets.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/reseed}
mkdir -p "$LOGDIR"

ROOTS=(
  "/home1/irteam/base_drug/eval/targetdiff"
  "$V/_vanilla_ep923/samples/full_eval_ep923"
  "$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
  "$V/samples_reference_receptor_ed_ep350"
)

mapfile -t NAMES < <(/opt/conda/envs/moleval/bin/python -c "
import json,sys; print('\n'.join(json.load(open('$TARGETS'))))")
echo "[$(date '+%H:%M:%S')] reseeding ${#NAMES[@]} pockets x ${#ROOTS[@]} runs, $N at a time"

run_one() {
    local d=$1 tag=$2
    /opt/conda/envs/moleval/bin/python notebook/webapp/metrics.py "$d" \
        --pose posecheck --force > "$LOGDIR/$tag.log" 2>&1
    printf '%s %s\n' "$([ $? -eq 0 ] && echo OK || echo FAIL)" "$tag" >> "$LOGDIR/progress.txt"
}

: > "$LOGDIR/progress.txt"
for root in "${ROOTS[@]}"; do
    rn=$(basename "$(dirname "$(dirname "$root")")")_$(basename "$root")
    for t in "${NAMES[@]}"; do
        [ -d "$root/$t" ] || continue
        while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 3; done
        run_one "$root/$t" "${rn}__${t}" &
    done
done
wait
ok=$(grep -c '^OK'   "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
bad=$(grep -c '^FAIL' "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
echo "[$(date '+%H:%M:%S')] reseed done: ok=$ok failed=$bad"
echo "RESEED_DONE" >> "$LOGDIR/progress.txt"
