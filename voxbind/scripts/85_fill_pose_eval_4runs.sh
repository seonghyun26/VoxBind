#!/usr/bin/env bash
# 85_fill_pose_eval_4runs.sh — finish the pose-quality eval (PoseCheck + PoseBusters)
# for the four runs that live on THIS box: the two baselines and the two "Ours" arms.
#
#   TargetDiff                /home1/irteam/base_drug/eval/targetdiff        100 pockets
#   VoxBind sigma=0.9         exps/_vanilla_ep923/samples/full_eval_ep923    100 pockets
#   Ours v1                   exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/  79 pockets
#                               samples/full_eval_ep350
#   Ours v2                   exps/samples_reference_receptor_ed_ep350        92 pockets
#
# (Those are the roots build_posecheck_analysis.py plots. The five published baselines --
# AR / Pocket2Mol / DiffSBDD / DecompDiff / FuncBind -- were sampled on the other box and
# only their exported PoseCheck JSONs are here, so they cannot be filled from here.)
#
# WHY: PoseCheck is done everywhere except 6 vanilla pockets that hit the 600 s per-chunk
# timeout back on 2026-07-22, but PoseBusters was only ever run on part of one run and a
# third of another (33/79 Ours v1, 53/100 vanilla, 0/100 TargetDiff, 0/92 Ours v2). This
# fills every gap so all four arms carry both metrics over their full pocket set.
#
# PER-TARGET MODE. Decided from what the target's metrics.json already holds, so a
# re-run is cheap and never recomputes a block it does not have to:
#   both blocks present   -> skipped
#   PoseCheck only        -> `--pose posebusters`  (PoseCheck rows are reused verbatim)
#   neither / PoseCheck errored -> `--pose all`    (the 6 timed-out vanilla pockets)
# `--pose posebusters` is only safe to run over a finished PoseCheck because of the
# reference-block carry-over in metrics.py (compute_target_metrics): _reference_row
# rebuilds a bare chem row every call and the pose step writes back only the blocks the
# requested mode asks for, so without that carry-over this fill would DELETE
# reference.posecheck -- the crystal-ligand baseline the PoseCheck figures plot.
#
# TIMEOUT: 1800 s per 20-ligand chunk, up from the 600 s default. That default is exactly
# what the 6 vanilla pockets tripped over.
#
# PARALLELISM AND NICENESS. N is a SLOT count, not a core count: one PoseBusters target
# is ~20-45 s wall but 3-25 CPU-minutes, because its own process tree already spreads over
# 8-25 cores. The cgroup quota here is 64 cores (cpu.max = 6400000/100000; `nproc` says 128
# and lies, and the 32 in older notes is stale), and the resident 4-GPU FuncBind DDP run
# only wants ~7 of them (4 ranks near 100% plus dataloader workers at ~10% each). So the
# limit that matters is latency per target, not cores: N=2 left one slot stuck behind a
# slow `--pose all` pocket and drained ~1 target every 2 min. N=5 keeps several moving.
# `nice -n 15` is what makes that safe -- whenever the training run wants a core it wins,
# and this fill only ever eats slack. Unniced, N=4 alone drove the 1-min load to 97.
#
#   bash voxbind/scripts/85_fill_pose_eval_4runs.sh          # fill everything missing
#   N=2 bash voxbind/scripts/85_fill_pose_eval_4runs.sh      # if the box gets busier
#   DRY=1 bash voxbind/scripts/85_fill_pose_eval_4runs.sh    # just print the worklist
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/moleval/bin:$PATH"      # hydride + reduce for PoseCheck

PY=/opt/conda/envs/moleval/bin/python
V=voxbind/exps
N=${N:-5}
DRY=${DRY:-0}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/posefill}
export POSE_TIMEOUT_S=${POSE_TIMEOUT_S:-1800}
mkdir -p "$LOGDIR"

# label -> run root. Labels are the log-file prefixes, so keep them path-safe.
declare -A ROOTS=(
  [targetdiff]="/home1/irteam/base_drug/eval/targetdiff"
  [vanilla]="$V/_vanilla_ep923/samples/full_eval_ep923"
  [ours_v1]="$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
  [ours_v2]="$V/samples_reference_receptor_ed_ep350"
)

# ── worklist: one "<label> <target_dir> <pose-mode>" line per target needing work ──
PLAN=$LOGDIR/plan.txt
: > "$PLAN"
for m in "${!ROOTS[@]}"; do
    "$PY" - "$m" "${ROOTS[$m]}" >> "$PLAN" <<'EOF'
import glob, json, os, sys

label, root = sys.argv[1], sys.argv[2]
for d in sorted(glob.glob(os.path.join(root, "target_*"))):
    if not glob.glob(os.path.join(d, "*_pocket10.pdb")):
        continue                                  # nothing to score against
    if not os.path.exists(os.path.join(d, "samples.sdf")):
        continue
    p = os.path.join(d, "metrics.json")
    rows = []
    if os.path.exists(p):
        try:
            rows = json.load(open(p)).get("samples", [])
        except (json.JSONDecodeError, OSError):
            rows = []
    # a block counts as present only if it actually carries a measurement: an
    # errored PoseCheck chunk leaves {"error": ...} behind, which must be redone
    has_pc = any(isinstance(r.get("posecheck"), dict)
                 and r["posecheck"].get("clashes") is not None for r in rows)
    has_pb = any(isinstance(r.get("posebusters"), dict)
                 and isinstance(r["posebusters"].get("valid"), bool) for r in rows)
    if has_pc and has_pb:
        continue
    print(label, d, "posebusters" if has_pc else "all")
EOF
done

total=$(wc -l < "$PLAN")
echo "[$(date '+%F %H:%M:%S')] $total target(s) to fill, $N at a time, timeout ${POSE_TIMEOUT_S}s" \
    | tee -a "$LOGDIR/driver.log"
awk '{print $1, $3}' "$PLAN" | sort | uniq -c | sed 's/^/    /' | tee -a "$LOGDIR/driver.log"
[ "$DRY" = "1" ] && { echo "DRY=1 — plan only: $PLAN"; exit 0; }
[ "$total" -eq 0 ] && { echo "nothing to do"; exit 0; }

: > "$LOGDIR/progress.txt"
while read -r label dir mode; do
    while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 5; done
    (
      tag="${label}__$(basename "$dir")__${mode}"
      nice -n 15 "$PY" notebook/webapp/metrics.py "$dir" --pose "$mode" \
          > "$LOGDIR/$tag.log" 2>&1
      rc=$?
      # the run can exit 0 having recorded a per-chunk failure in-band, so grade on
      # what landed in metrics.json, not on the exit status alone
      ok=$("$PY" - "$dir" "$mode" <<'EOF'
import json, os, sys
d, mode = sys.argv[1], sys.argv[2]
try:
    rows = json.load(open(os.path.join(d, "metrics.json"))).get("samples", [])
except Exception:
    print("no"); raise SystemExit
need = ("posecheck", "posebusters") if mode == "all" else ("posebusters",)
good = all(
    any(isinstance(r.get(b), dict)
        and (r[b].get("clashes") is not None if b == "posecheck"
             else isinstance(r[b].get("valid"), bool)) for r in rows)
    for b in need)
print("yes" if good else "no")
EOF
)
      printf '%s %s %s rc=%s\n' "$([ "$ok" = yes ] && echo OK || echo FAIL)" \
             "$mode" "$tag" "$rc" >> "$LOGDIR/progress.txt"
    ) &
done < "$PLAN"
wait

ok=$(grep -c '^OK'   "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
bad=$(grep -c '^FAIL' "$LOGDIR/progress.txt" 2>/dev/null || echo 0)
echo "[$(date '+%F %H:%M:%S')] posefill done: ok=$ok failed=$bad  (logs: $LOGDIR)" \
    | tee -a "$LOGDIR/driver.log"
grep '^FAIL' "$LOGDIR/progress.txt" 2>/dev/null | tee -a "$LOGDIR/driver.log"
echo "POSEFILL_DONE" >> "$LOGDIR/progress.txt"
