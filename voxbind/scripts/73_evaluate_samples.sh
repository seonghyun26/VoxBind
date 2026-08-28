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
# Env: `sbdd`, NOT `targetdiff` — metrics.py uses PEP 585 generics
# (Callable[[int,int,str], None]) which are a TypeError on targetdiff's Python 3.8.
# sbdd is 3.12 and carries rdkit + the vina python module + meeko; pdb2pqr30 and
# obabel must be on PATH because TargetDiff's VinaDockingTask shells out to them
# and silently records `None` affinities if they are missing.
#
# Env knobs: SAMPLE_DIR, DOCK (vina_dock|vina_min|vina_score|none), WORKERS, CPU, EXH.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
WEBAPP="$ROOT/notebook/webapp"
ENV_BIN="$HOME/miniforge3/envs/sbdd/bin"

: "${SAMPLE_DIR:?set SAMPLE_DIR (dir containing target_*/)}"
case "$SAMPLE_DIR" in /*) ;; *) SAMPLE_DIR="$ROOT/voxbind/$SAMPLE_DIR";; esac
DOCK="${DOCK:-vina_dock}"
CPU="${CPU:-4}"
WORKERS="${WORKERS:-48}"     # 48 x 4 = 192 of 384 cores; box is shared
EXH="${EXH:-8}"              # TargetDiff/VoxBind default exhaustiveness

[ -d "$SAMPLE_DIR" ] || { echo "[73_eval] MISSING $SAMPLE_DIR"; exit 1; }
[ -x "$ENV_BIN/python" ] || { echo "[73_eval] MISSING $ENV_BIN/python (sbdd env)"; exit 1; }
export PATH="$ENV_BIN:$PATH"
for b in pdb2pqr30 obabel; do
    command -v "$b" >/dev/null || { echo "[73_eval] MISSING $b on PATH"; exit 1; }
done

LOG="$SAMPLE_DIR/evaluate.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cd "$WEBAPP" || exit 1

n_targets=$(ls -d "$SAMPLE_DIR"/target_* 2>/dev/null | wc -l)
say "evaluating $n_targets target dirs in $SAMPLE_DIR"

say "pass 1/2: chem + geometry metrics (no docking)"
"$ENV_BIN/python" metrics.py "$SAMPLE_DIR" --docking none >> "$LOG" 2>&1
rc1=$?
say "pass 1 done (exit $rc1)"
[ "$rc1" -ne 0 ] && exit "$rc1"

if [ "$DOCK" != "none" ]; then
    say "pass 2/2: $DOCK (workers=$WORKERS cpu=$CPU exhaustiveness=$EXH)"
    "$ENV_BIN/python" metrics.py "$SAMPLE_DIR" \
        --docking "$DOCK" --workers "$WORKERS" --cpu "$CPU" \
        --exhaustiveness "$EXH" --skip-existing >> "$LOG" 2>&1
    rc2=$?
    say "pass 2 done (exit $rc2)"
    [ "$rc2" -ne 0 ] && exit "$rc2"
fi

say "aggregating across targets -> $SAMPLE_DIR/summary.json"
"$ENV_BIN/python" - "$SAMPLE_DIR" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
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

def over_molecules(field):
    """Pooled across all molecules; median too, since Vina scores are skewed."""
    v = [s["vina"][field] for s in rows
         if isinstance(s.get("vina"), dict)
         and isinstance(s["vina"].get(field), (int, float))]
    return ({"mean": round(st.mean(v), 4), "median": round(st.median(v), 4), "n": len(v)}
            if v else {"mean": None, "median": None, "n": 0})

summary = {
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
    # pooled-over-molecules view, for sanity-checking the per-target means
    "pooled": {f: over_molecules(f) for f in ("score_only", "minimize", "dock")},
}

(root / "summary.json").write_text(json.dumps(
    {"summary": summary, "per_target": per_target}, indent=2))
print(json.dumps(summary, indent=2))
PY
say "all done -> $SAMPLE_DIR/summary.json"
