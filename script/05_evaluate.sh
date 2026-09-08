#!/usr/bin/env bash
# 05_evaluate.sh — full sample evaluation: chem/geometry + Vina docking + PoseCheck.
#
#   SAMPLE_DIR=voxbind/exps/voxbind_density_fusion/samples/samples_test100 \
#     bash script/05_evaluate.sh
#
# Reproduces the VoxBind/TargetDiff table over a sampling run dir (the one 04
# printed): validity, uniqueness, diversity, QED, SA, logP, Lipinski,
# sim-to-ref, Vina score/min/dock, high-affinity %, PoseCheck clashes/strain,
# ProLIF interactions. Writes per-target metrics.json + a run-level summary.json.
#
# Three tools, three passes (a docking failure never costs the cheap numbers):
#   pass 1  chem + geometry (RDKit only)                       — always
#   pass 2  Vina docking (voxdock env: vina 1.2.2)             — DOCK != none
#   pass 3  PoseCheck / PoseBusters (moleval env)              — POSE != none
#
# Requirements (built by 00_setup_env.sh / the Docker image):
#   - voxdock env  (Vina 1.2.2 + TargetDiff eval modules on sys.path)
#   - moleval env  (PoseCheck 1.3.1) — reached via $MOLEVAL_PY subprocess
#   - TargetDiff clone as a sibling of the repo ($TARGETDIFF_ROOT)
#   - full receptors at $FULL_RECEPTOR_ROOT (Vina + clash counts score the WHOLE
#     receptor, not the 10 Å crop)
#
# Key overrides: SAMPLE_DIR, DOCK, POSE, EXH, CPU, WORKERS, RECEPTOR_ROOT, SKIP_EXISTING.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

: "${SAMPLE_DIR:?set SAMPLE_DIR (a sampling run dir containing target_*/ , from 04_sample.sh)}"
case "$SAMPLE_DIR" in /*) ;; *) SAMPLE_DIR="$REPO_ROOT/$SAMPLE_DIR" ;; esac
[ -d "$SAMPLE_DIR" ] || die "no such dir: $SAMPLE_DIR"

DOCK="${DOCK:-vina_dock}"          # none | vina_score | vina_min | vina_dock
POSE="${POSE:-posecheck}"          # none | posecheck | posebusters | all
EXH="${EXH:-8}"                    # Vina exhaustiveness (paper default)
CPU="${CPU:-4}"                    # Vina threads per worker
WORKERS="${WORKERS:-8}"            # docking process pool
RECEPTOR_ROOT="${RECEPTOR_ROOT:-$FULL_RECEPTOR_ROOT}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

WEBAPP="$REPO_ROOT/notebook/webapp"
[ -f "$WEBAPP/metrics.py" ] || die "missing $WEBAPP/metrics.py"
require_env "$VOXDOCK_ENV"

# TargetDiff must be importable + supply the full receptors.
[ -d "$TARGETDIFF_ROOT/utils/evaluation" ] || die "TargetDiff not found at $TARGETDIFF_ROOT (run 00_setup_env.sh targetdiff)"

SCOPE_ARG=()
if [ -n "$RECEPTOR_ROOT" ] && [ -d "$RECEPTOR_ROOT" ]; then
    export FULL_RECEPTOR_ROOT="$RECEPTOR_ROOT"
    SCOPE_ARG=( --pose-scope full --dock-scope full )
    log "scoring against FULL receptors: $RECEPTOR_ROOT"
else
    log "WARN: no full receptors at '$RECEPTOR_ROOT' — falling back to pocket10 crop (flatters Vina, under-counts clashes)"
fi

# Verify Vina version — every reported VoxBind affinity is vina 1.2.2; a
# different build shifts absolute scores, so refuse rather than silently mix.
vina_ver="$(conda_run "$VOXDOCK_ENV" python -c 'import importlib.metadata as m; print(m.version("vina"))' 2>/dev/null || true)"
[ "$vina_ver" = "1.2.2" ] || die "voxdock has vina='${vina_ver:-none}', need 1.2.2 (rebuild: FORCE=1 bash script/00_setup_env.sh voxdock)"

# Pose worker interpreter (moleval). Only required if POSE != none.
if [ "$POSE" != "none" ]; then
    require_env "$MOLEVAL_ENV"
    export MOLEVAL_PY="$(env_python "$MOLEVAL_ENV")"
    log "pose worker interpreter: $MOLEVAL_PY"
fi

# TargetDiff eval modules import from its repo root; docking shells out to
# pdb2pqr30/obabel, which live on the voxdock env's bin (conda_run puts them on PATH).
export PYTHONPATH="$TARGETDIFF_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$SAMPLE_DIR/evaluate.log"
n_t=$(ls -d "$SAMPLE_DIR"/target_* 2>/dev/null | wc -l)
banner "evaluating $n_t target dir(s) in $SAMPLE_DIR   (dock=$DOCK pose=$POSE exh=$EXH)"
SKIP_ARG=(); [ "$SKIP_EXISTING" = "1" ] && SKIP_ARG=(--skip-existing)

run_metrics() { ( cd "$WEBAPP" && conda_run "$VOXDOCK_ENV" python metrics.py "$@" ); }

log "pass 1/3: chem + geometry (no docking)"
run_metrics "$SAMPLE_DIR" --docking none "${SCOPE_ARG[@]}" "${SKIP_ARG[@]}" >> "$LOG" 2>&1 \
    || die "pass 1 failed (see $LOG)"

if [ "$DOCK" != "none" ]; then
    log "pass 2/3: $DOCK (workers=$WORKERS cpu=$CPU exhaustiveness=$EXH)"
    run_metrics "$SAMPLE_DIR" --docking "$DOCK" --workers "$WORKERS" --cpu "$CPU" \
        --exhaustiveness "$EXH" "${SCOPE_ARG[@]}" "${SKIP_ARG[@]}" >> "$LOG" 2>&1 \
        || die "pass 2 (docking) failed (see $LOG)"
fi

if [ "$POSE" != "none" ]; then
    log "pass 3/3: pose eval ($POSE)"
    run_metrics "$SAMPLE_DIR" --docking "$DOCK" --pose "$POSE" --workers "$WORKERS" \
        --cpu "$CPU" --exhaustiveness "$EXH" "${SCOPE_ARG[@]}" --skip-existing >> "$LOG" 2>&1 \
        || die "pass 3 (pose) failed (see $LOG)"
fi

# Aggregate per-target metrics.json -> summary.json (mean over targets, paper axis).
log "aggregating -> $SAMPLE_DIR/summary.json"
conda_run "$VOXDOCK_ENV" python - "$SAMPLE_DIR" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
root = Path(sys.argv[1]); per_target, rows = [], []
for mj in sorted(root.glob("target_*/metrics.json")):
    d = json.loads(mj.read_text())
    per_target.append({"target": mj.parent.name, **d.get("aggregates", {})})
    rows.extend(d.get("samples", []))
def over_t(k):
    v = [t[k] for t in per_target if isinstance(t.get(k), (int, float))]
    return round(st.mean(v), 4) if v else None
def pose(field, how):
    v = [s["posecheck"][field] for s in rows
         if isinstance(s.get("posecheck"), dict) and "error" not in s["posecheck"]
         and isinstance(s["posecheck"].get(field), (int, float))]
    if not v: return None
    return round(st.median(v) if how == "median" else st.mean(v), 4)
summary = {
    "n_targets": len(per_target), "n_molecules": len(rows),
    "validity": over_t("validity"), "uniqueness": over_t("uniqueness"),
    "diversity": over_t("diversity"), "qed_mean": over_t("qed_mean"),
    "sa_mean": over_t("sa_mean"), "logp_mean": over_t("logp_mean"),
    "lipinski_mean": over_t("lipinski_mean"), "sim_to_ref_mean": over_t("sim_to_ref_mean"),
    "vina_score_mean": over_t("vina_score_mean"), "vina_min_mean": over_t("vina_min_mean"),
    "vina_dock_mean": over_t("vina_dock_mean"), "high_affinity": over_t("high_affinity"),
    "clashes_median": pose("clashes", "median"),
    "strain_median": pose("strain", "median"),
    "n_interactions_median": pose("n_interactions", "median"),
}
(root / "summary.json").write_text(json.dumps({"summary": summary, "per_target": per_target}, indent=2))
print(json.dumps(summary, indent=2))
PY

banner "evaluation complete"
[ -f "$SAMPLE_DIR/summary.json" ] && conda_run "$VOXDOCK_ENV" python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['summary'],indent=2))" "$SAMPLE_DIR/summary.json"
log "summary -> $SAMPLE_DIR/summary.json   (full log: $LOG)"
