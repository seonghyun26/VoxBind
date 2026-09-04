#!/usr/bin/env bash
# 73_evaluate_samples.sh — TargetDiff-style metrics over a 72_sample_8gpu.sh run dir.
#
#   SAMPLE_DIR=exps/260827_voxbind_base_8gpu/samples/samples_ep350_test100 \
#     bash scripts/73_evaluate_samples.sh
#
# Two passes so a docking failure never costs the cheap numbers:
#   pass 1  --docking none   -> validity, uniqueness, diversity, QED, SA, logP,
#                               Lipinski, sim_to_ref, pocket contacts/clashes
#   pass 2  --docking DOCK   -> Vina score_only / minimize / full dock (the paper's
#                               three affinity columns) + high-affinity %
# Pass 2 uses --skip-existing so a re-run resumes instead of re-docking.
#
# RECEPTOR: metrics.py defaults to the *_pocket10.pdb crop sitting next to the
# samples (~400 atoms). That is NOT what TargetDiff scores against — its
# VinaDockingTask resolves <protein_root>/<subdir>/<ligand[:10]>.pdb, the full
# receptor (~3.5k atoms). A crop cannot show a clash with an atom it does not
# contain, so it flatters Vina and under-counts PoseCheck clashes.
# RECEPTOR_ROOT sets metrics.py's own $FULL_RECEPTOR_ROOT, and both
# --pose-scope full and --dock-scope full are passed so affinity AND pose are
# scored against the whole chain. metrics.py records pose_receptor_scope /
# dock_receptor_scope in every metrics.json, and drops a cached row whose scope
# disagrees, so a run can never silently mix the two.
#
# POSE: PoseCheck (clashes, strain energy, prolif interactions) is what the
# VoxBind paper reports alongside Vina. metrics.py runs it in a subprocess whose
# interpreter defaults to /opt/conda/envs/moleval — a path from a different box.
# MOLEVAL_PY points it at the sbdd env, which already carries posecheck.
#
# Env: `sbdd`, NOT `targetdiff` — metrics.py uses PEP 585 generics
# (Callable[[int,int,str], None]) which are a TypeError on targetdiff's Python 3.8.
# sbdd is 3.12 and carries rdkit + the vina python module + meeko; pdb2pqr30 and
# obabel must be on PATH because TargetDiff's VinaDockingTask shells out to them
# and silently records `None` affinities if they are missing.
#
# Env knobs: SAMPLE_DIR, DOCK, POSE, RECEPTOR_ROOT, WORKERS, CPU, EXH, SKIP_EXISTING.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
WEBAPP="$ROOT/notebook/webapp"
ENV_BIN="$HOME/miniforge3/envs/sbdd/bin"

: "${SAMPLE_DIR:?set SAMPLE_DIR (dir containing target_*/)}"
case "$SAMPLE_DIR" in /*) ;; *) SAMPLE_DIR="$ROOT/voxbind/$SAMPLE_DIR";; esac
DOCK="${DOCK:-vina_dock}"
POSE="${POSE:-posecheck}"        # none | posecheck | posebusters | all
RECEPTOR_ROOT="${RECEPTOR_ROOT:-/home/shpark/prj-denovo/targetdiff/data/test_set}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"   # off by default: a receptor change invalidates cached scores
CPU="${CPU:-4}"
WORKERS="${WORKERS:-48}"     # 48 x 4 = 192 of 384 cores; box is shared
EXH="${EXH:-8}"              # TargetDiff/VoxBind default exhaustiveness

[ -d "$SAMPLE_DIR" ] || { echo "[73_eval] MISSING $SAMPLE_DIR"; exit 1; }
[ -x "$ENV_BIN/python" ] || { echo "[73_eval] MISSING $ENV_BIN/python (sbdd env)"; exit 1; }
export PATH="$ENV_BIN:$PATH"
for b in pdb2pqr30 obabel; do
    command -v "$b" >/dev/null || { echo "[73_eval] MISSING $b on PATH"; exit 1; }
done
export MOLEVAL_PY="${MOLEVAL_PY:-$ENV_BIN/python}"
SCOPE_ARG=()
if [ -n "$RECEPTOR_ROOT" ]; then
    [ -d "$RECEPTOR_ROOT" ] || { echo "[73_eval] MISSING RECEPTOR_ROOT $RECEPTOR_ROOT"; exit 1; }
    export FULL_RECEPTOR_ROOT="$RECEPTOR_ROOT"
    SCOPE_ARG=(--pose-scope full --dock-scope full)
fi

LOG="$SAMPLE_DIR/evaluate.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cd "$WEBAPP" || exit 1

n_targets=$(ls -d "$SAMPLE_DIR"/target_* 2>/dev/null | wc -l)
say "evaluating $n_targets target dirs in $SAMPLE_DIR"
say "receptor: ${FULL_RECEPTOR_ROOT:-pocket10 crop (per-target)} (scope args: ${SCOPE_ARG[*]:-crop})"
say "docking=$DOCK pose=$POSE"

say "pass 1/2: chem + geometry metrics (no docking)"
SKIP_ARG=()
[ "$SKIP_EXISTING" = "1" ] && SKIP_ARG=(--skip-existing)

"$ENV_BIN/python" metrics.py "$SAMPLE_DIR" --docking none \
        "${SCOPE_ARG[@]}" "${SKIP_ARG[@]}" >> "$LOG" 2>&1
rc1=$?
say "pass 1 done (exit $rc1)"
[ "$rc1" -ne 0 ] && exit "$rc1"

if [ "$DOCK" != "none" ]; then
    say "pass 2/2: $DOCK (workers=$WORKERS cpu=$CPU exhaustiveness=$EXH)"
    "$ENV_BIN/python" metrics.py "$SAMPLE_DIR" \
        --docking "$DOCK" --workers "$WORKERS" --cpu "$CPU" \
        --exhaustiveness "$EXH" "${SCOPE_ARG[@]}" "${SKIP_ARG[@]}" >> "$LOG" 2>&1
    rc2=$?
    say "pass 2 done (exit $rc2)"
    [ "$rc2" -ne 0 ] && exit "$rc2"
fi

if [ "$POSE" != "none" ]; then
    say "pass 3/3: pose eval ($POSE) via $MOLEVAL_PY"
    "$ENV_BIN/python" metrics.py "$SAMPLE_DIR" \
        --docking "$DOCK" --pose "$POSE" \
        --workers "$WORKERS" --cpu "$CPU" --exhaustiveness "$EXH" \
        "${SCOPE_ARG[@]}" --skip-existing >> "$LOG" 2>&1
    rc3=$?
    say "pass 3 done (exit $rc3)"
    [ "$rc3" -ne 0 ] && exit "$rc3"
fi

say "aggregating across targets -> $SAMPLE_DIR/summary.json"
"$ENV_BIN/python" - "$SAMPLE_DIR" >> "$LOG" 2>&1 <<'PY'
import json, os, sys, statistics as st
from pathlib import Path

root = Path(sys.argv[1])
per_target, rows = [], []
for mj in sorted(root.glob("target_*/metrics.json")):
    d = json.loads(mj.read_text())
    per_target.append({"target": mj.parent.name, **d.get("aggregates", {})})
    rows.extend(d.get("samples", []))

def over_targets(key):
    """Mean across targets - the axis the CrossDocked papers report."""
    v = [t[key] for t in per_target if isinstance(t.get(key), (int, float))]
    return round(st.mean(v), 4) if v else None

def pooled_pose(field, how):
    """PoseCheck field pooled across every scored molecule."""
    v = [s["posecheck"][field] for s in rows
         if isinstance(s.get("posecheck"), dict)
         and "error" not in s["posecheck"]
         and isinstance(s["posecheck"].get(field), (int, float))]
    if not v:
        return None
    return round(st.median(v) if how == "median" else st.mean(v), 4)

def over_molecules(field):
    """Pooled across all molecules; median too, since Vina scores are skewed."""
    v = [s["vina"][field] for s in rows
         if isinstance(s.get("vina"), dict)
         and isinstance(s["vina"].get(field), (int, float))]
    return ({"mean": round(st.mean(v), 4), "median": round(st.median(v), 4), "n": len(v)}
            if v else {"mean": None, "median": None, "n": 0})

summary = {
    "receptor": os.environ.get("FULL_RECEPTOR_ROOT") or "pocket10 crop",
    "n_targets": len(per_target),
    "n_molecules": len(rows),
    # chem / geometry, averaged over targets
    "validity":        over_targets("validity"),
    "uniqueness":      over_targets("uniqueness"),
    "diversity":       over_targets("diversity"),
    "qed_mean":        over_targets("qed_mean"),
    "sa_mean":         over_targets("sa_mean"),
    "logp_mean":       over_targets("logp_mean"),
    "lipinski_mean":   over_targets("lipinski_mean"),
    "sim_to_ref_mean": over_targets("sim_to_ref_mean"),
    # Vina, averaged over targets (paper axis)
    "vina_score_mean": over_targets("vina_score_mean"),
    "vina_min_mean":   over_targets("vina_min_mean"),
    "vina_dock_mean":  over_targets("vina_dock_mean"),
    # High Affinity % - fraction of samples beating the reference ligand. Only
    # computable per target (it needs that pocket's reference score), so this is
    # the mean of the per-target fractions, not a pooled ratio.
    "high_affinity":   over_targets("high_affinity"),
    "vina_n_docked":   sum(t.get("vina_n_docked", 0) or 0 for t in per_target),
    "vina_n_failed":   sum(t.get("vina_n_failed", 0) or 0 for t in per_target),
    # PoseCheck: steric clashes with the receptor and ligand strain energy.
    # Both are only meaningful against the FULL receptor - a 10 A crop has no
    # atoms to clash with beyond its shell.
    #
    # Pooled over molecules, not averaged over targets: strain is heavy-tailed
    # (p99 ~1e6, max ~1e11 here), so a MEAN of per-target medians re-imports the
    # very outlier problem the median exists to avoid. The mean is kept only as
    # provenance, flagged so nobody quotes it.
    "clashes_median":  pooled_pose("clashes", "median"),
    "clashes_mean":    pooled_pose("clashes", "mean"),
    "strain_median":   pooled_pose("strain", "median"),
    "strain_mean_UNRELIABLE": pooled_pose("strain", "mean"),
    "n_interactions_median": pooled_pose("n_interactions", "median"),
    "n_posecheck":     sum(1 for s_ in rows
                           if isinstance(s_.get("posecheck"), dict)
                           and "error" not in s_["posecheck"]),
    "n_posecheck_err": sum(1 for s_ in rows
                           if isinstance(s_.get("posecheck"), dict)
                           and "error" in s_["posecheck"]),
    # pooled-over-molecules view, for sanity-checking the per-target means
    "pooled": {f: over_molecules(f) for f in ("score_only", "minimize", "dock")},
}

(root / "summary.json").write_text(json.dumps(
    {"summary": summary, "per_target": per_target}, indent=2))
print(json.dumps(summary, indent=2))
PY
say "all done -> $SAMPLE_DIR/summary.json"
