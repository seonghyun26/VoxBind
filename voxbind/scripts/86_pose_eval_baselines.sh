#!/usr/bin/env bash
# 86_pose_eval_baselines.sh — PoseBusters for the five published baselines.
#
# AR, Pocket2Mol, DiffSBDD, DecompDiff and FuncBind were sampled on another box and only
# their PoseCheck exports travelled with them, so until now the PoseBusters figures could
# only show TargetDiff, VoxBind and CoDE. Their molecules ARE here, as TargetDiff-format
# meta bundles under results/task2-drugdesign/; tools/stage_baseline_samples.py writes
# them out as ordinary sample dirs (samples.sdf + the pocket's own *_pocket10.pdb, hard
# linked from the vanilla run) so this can score them with the same metrics.py the other
# arms went through -- same molecule filter, same 20 scored checks, same definition of
# `valid`. Run that first; this refuses to start otherwise.
#
#   /opt/conda/envs/voxbind/bin/python voxbind/scripts/tools/stage_baseline_samples.py
#   bash voxbind/scripts/86_pose_eval_baselines.sh
#
# SCOPE IS THE CROP, deliberately. The staged receptor is each pocket's *_pocket10.pdb --
# the same file our arms are scored against, hard linked, not copied. That makes these
# numbers comparable with fig-posebusters and NOT with the whole-receptor PoseCheck
# figures in fig-posecheck/build_posecheck_all_by_atom_range.py.
#
# 79 pockets x 5 methods = 395 target dirs, ~30 s each. N=5 niced, as in 85: one target's
# process tree already spreads over 8-25 cores, the cgroup quota is 64, and `nice -n 15`
# means a resident training run keeps its dataloader cores.
#
#   N=2 bash voxbind/scripts/86_pose_eval_baselines.sh      # if the box gets busier
#   DRY=1 bash voxbind/scripts/86_pose_eval_baselines.sh    # print the worklist only
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/moleval/bin:$PATH"      # hydride + reduce, as in 85

PY=/opt/conda/envs/moleval/bin/python
STAGE=voxbind/exps/baselines_pose
N=${N:-5}
DRY=${DRY:-0}
LOGDIR=${LOGDIR:-voxbind/exps/frozenenc_probes/logs/pbbaselines}
export POSE_TIMEOUT_S=${POSE_TIMEOUT_S:-1800}
mkdir -p "$LOGDIR"

[ -d "$STAGE" ] || { echo "[86] $STAGE missing — run tools/stage_baseline_samples.py first"; exit 1; }

PLAN=$LOGDIR/plan.txt
: > "$PLAN"
for m in ar pocket2mol diffsbdd decompdiff funcbind; do
    [ -d "$STAGE/$m" ] || { echo "[86] missing $STAGE/$m"; exit 1; }
    for d in "$STAGE/$m"/target_*; do
        [ -f "$d/samples.sdf" ] || continue
        # already scored? a PoseBusters verdict on any sample means this target is done
        if [ -f "$d/metrics.json" ] && "$PY" - "$d" <<'EOF'
import json, os, sys
rows = json.load(open(os.path.join(sys.argv[1], "metrics.json"))).get("samples", [])
raise SystemExit(0 if any(isinstance(r.get("posebusters"), dict)
                          and isinstance(r["posebusters"].get("valid"), bool)
                          for r in rows) else 1)
EOF
        then continue; fi
        printf '%s %s\n' "$m" "$d" >> "$PLAN"
    done
done

total=$(wc -l < "$PLAN")
echo "[$(date '+%F %H:%M:%S')] $total target(s), $N at a time, timeout ${POSE_TIMEOUT_S}s" \
    | tee -a "$LOGDIR/driver.log"
awk '{print $1}' "$PLAN" | sort | uniq -c | sed 's/^/    /' | tee -a "$LOGDIR/driver.log"
[ "$DRY" = "1" ] && { echo "DRY=1 — plan only: $PLAN"; exit 0; }
[ "$total" -eq 0 ] && { echo "nothing to do"; exit 0; }

: > "$LOGDIR/progress.txt"
while read -r m dir; do
    while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 5; done
    (
      tag="${m}__$(basename "$dir")"
      nice -n 15 "$PY" notebook/webapp/metrics.py "$dir" --pose posebusters \
          > "$LOGDIR/$tag.log" 2>&1
      rc=$?
      # grade on what landed in metrics.json: metrics.py exits 0 having recorded a
      # per-chunk timeout in-band
      ok=$("$PY" - "$dir" <<'EOF'
import json, os, sys
try:
    rows = json.load(open(os.path.join(sys.argv[1], "metrics.json"))).get("samples", [])
except Exception:
    print("no"); raise SystemExit
print("yes" if any(isinstance(r.get("posebusters"), dict)
                   and isinstance(r["posebusters"].get("valid"), bool)
                   for r in rows) else "no")
EOF
)
      printf '%s %s rc=%s\n' "$([ "$ok" = yes ] && echo OK || echo FAIL)" "$tag" "$rc" \
          >> "$LOGDIR/progress.txt"
    ) &
done < "$PLAN"
wait

ok=$(grep -c '^OK'   "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
bad=$(grep -c '^FAIL' "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
echo "[$(date '+%F %H:%M:%S')] baselines done: ok=$ok failed=$bad (logs: $LOGDIR)" \
    | tee -a "$LOGDIR/driver.log"
grep '^FAIL' "$LOGDIR/progress.txt" 2>/dev/null | tee -a "$LOGDIR/driver.log"
echo "PBBASELINES_DONE" >> "$LOGDIR/progress.txt"
