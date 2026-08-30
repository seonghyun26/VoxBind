#!/usr/bin/env bash
# 72_posecheck_full_receptor_78.sh — re-measure PoseCheck against the WHOLE receptor.
#
# CROP STAYS THE PRIMARY METRIC. PoseCheck's own CrossDocked pipeline keys on the
# pocket10 crop (posecheck/utils/loading.py get_ids_to_pockets reads split entry [0],
# which IS the *_pocket10.pdb), so published PoseCheck numbers -- and the ~10.8
# TargetDiff clash mean our 10.39 reproduces -- are crop-scored. Switching wholesale
# would break that calibration.
#
# This run answers a narrower question: HOW MUCH does the crop under-count? Clashes are
# a ligand-atom x protein-atom distance test, and the 10 Å crop drops 35-64% of atoms
# within van der Waals reach, so the crop count is a structural under-count that should
# grow with ligand size. Strain never sees the protein and must come back unchanged --
# that is the built-in check that the swap did what it claims.
#
# Writes into a SEPARATE tree so the crop metrics.json files are untouched.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/moleval/bin:$PATH"      # hydride + reduce

V=voxbind/exps
OUT=${OUT:-$V/frozenenc_probes/posecheck_full}
TARGETS=${TARGETS:-$V/frozenenc_probes/p78_targets.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/posefull}
N=${N:-16}
mkdir -p "$OUT" "$LOGDIR"

declare -A ROOTS=(
  [targetdiff]="/home1/irteam/base_drug/eval/targetdiff"
  [vanilla]="$V/_vanilla_ep923/samples/full_eval_ep923"
  [ours_v1]="$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
)

mapfile -t NAMES < <(/opt/conda/envs/moleval/bin/python -c "
import json; print('\n'.join(json.load(open('$TARGETS'))))")
echo "[$(date '+%H:%M:%S')] ${#NAMES[@]} pockets x ${#ROOTS[@]} methods, $N at a time" | tee -a "$LOGDIR/driver.log"

: > "$LOGDIR/progress.txt"
for m in "${!ROOTS[@]}"; do
    src=${ROOTS[$m]}
    for t in "${NAMES[@]}"; do
        [ -d "$src/$t" ] || continue
        d=$OUT/$m/$t
        # already done? metrics.json carries the scope it was measured with
        if [ -f "$d/metrics.json" ] && grep -q '"pose_receptor_scope": "full"' "$d/metrics.json" 2>/dev/null; then
            continue
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 3; done
        (
          mkdir -p "$d"
          # hard-link the inputs: same volume, no copy, and the crop PDB must be present
          # because find_full_receptor derives the whole-receptor path from its filename.
          for f in "$src/$t"/samples.sdf "$src/$t"/*_pocket10.pdb; do
              [ -e "$f" ] && ln -f "$f" "$d/$(basename "$f")" 2>/dev/null
          done
          /opt/conda/envs/moleval/bin/python notebook/webapp/metrics.py "$d" \
              --pose posecheck --pose-scope full --force > "$LOGDIR/${m}__${t}.log" 2>&1
          printf '%s %s %s\n' "$([ $? -eq 0 ] && echo OK || echo FAIL)" "$m" "$t" >> "$LOGDIR/progress.txt"
        ) &
    done
done
wait
echo "[$(date '+%H:%M:%S')] done: ok=$(grep -c '^OK' "$LOGDIR/progress.txt") fail=$(grep -c '^FAIL' "$LOGDIR/progress.txt")" | tee -a "$LOGDIR/driver.log"
echo "POSEFULL_DONE" >> "$LOGDIR/progress.txt"
