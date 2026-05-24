#!/bin/bash
# 10_dock_eval_10k_test_ab.sh — wait for the 10k test-split sampling to finish,
# then Vina-dock + chem-eval the density / noise arms (2-arm A/B).
#
# Companion to 09_sample_10k_test_ab.sh. Phase 0 blocks until that run's
# orchestrator log reports "ALL ARMS DONE", then runs notebook/webapp/metrics.py
# on each arm's res_ep99_test/ (density + noise) in parallel. metrics.py writes
# a metrics.json into every target_XX/:
#   - chemical quality : validity, uniqueness, diversity, QED, SA, LogP, Lipinski
#   - Vina (vina_dock) : per-sample score_only / minimize / dock + vina_*_mean
# Docking is CPU-only (GPUs untouched). The 10k baseline is still training, so
# it is NOT in this A/B — dock it separately once it finishes and is sampled.
#
# Launch detached:
#   nohup bash voxbind/scripts/10_dock_eval_10k_test_ab.sh \
#       > voxbind/log/260519_dock_eval_10k_test_ab.log 2>&1 &
#
# Env vars (optional): DOCKING (default vina_dock), EXH (8), CPU (8).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOXBIND_DIR="$(dirname "$SCRIPT_DIR")"     # .../VoxBind/voxbind  (script in scripts/)
REPO_DIR="$(dirname "$VOXBIND_DIR")"       # .../VoxBind
WEBAPP_DIR="${REPO_DIR}/notebook/webapp"
LOG_DIR="${VOXBIND_DIR}/log"
EXPS="${VOXBIND_DIR}/exps"

DOCKING="${DOCKING:-vina_dock}"
EXH="${EXH:-8}"
CPU="${CPU:-8}"
SUB="res_ep99_test"

SAMPLE_LOG="${LOG_DIR}/260519_sample_10k_test_ab.log"   # orchestrator log of 09_*
DONE_MARK="ALL ARMS DONE"
POLL=120          # s between checks while waiting for sampling
MAX_WAIT=64800    # 18 h hard cap (sampling est. ~9 h)

VOXBIND_ENV_BIN="${VOXBIND_ENV_BIN:-/home/shpark/.conda/envs/voxbind/bin}"
export PATH="${VOXBIND_ENV_BIN}:${PATH}"

log() { echo "[dock-eval-10k $(date '+%F %T')] $*"; }

DENSITY_DIR="${EXPS}/260518_voxbind_10k_density/samples/${SUB}"
NOISE_DIR="${EXPS}/260518_voxbind_10k_noise/samples/${SUB}"

echo "$$" > "${LOG_DIR}/260519_dock_eval_10k_test_ab.pid"
log "staged (pid $$) — waiting for sampling, then dock density+noise (docking=$DOCKING exh=$EXH cpu=$CPU)"

# --- Phase 0: wait for 09_sample_10k_test_ab.sh to finish ----------------
waited=0
while ! grep -q "$DONE_MARK" "$SAMPLE_LOG" 2>/dev/null; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        log "FATAL: sampling not done after ${MAX_WAIT}s — aborting (no docking)."
        exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done
log "sampling complete (waited ${waited}s) — proceeding to docking"

# --- preflight -----------------------------------------------------------
cd "$WEBAPP_DIR" || { log "FATAL: webapp dir missing: $WEBAPP_DIR"; exit 1; }
[ -f metrics.py ] || { log "FATAL: metrics.py not found in $WEBAPP_DIR"; exit 1; }
for d in "$DENSITY_DIR" "$NOISE_DIR"; do
    [ -d "$d" ] || { log "FATAL: missing sample dir $d"; exit 1; }
done
log "density targets: $(ls -d "$DENSITY_DIR"/target_* 2>/dev/null | wc -l)  |  noise targets: $(ls -d "$NOISE_DIR"/target_* 2>/dev/null | wc -l)"

# --- Phase 1: dock the 2 arms in parallel (--skip-existing -> resumable) --
log "docking — density / noise in parallel"
python metrics.py "$DENSITY_DIR" --docking "$DOCKING" --exhaustiveness "$EXH" --cpu "$CPU" --skip-existing \
    > "${LOG_DIR}/260519_dock_eval_10k_density_test.log" 2>&1 &
PID_D=$!
python metrics.py "$NOISE_DIR"   --docking "$DOCKING" --exhaustiveness "$EXH" --cpu "$CPU" --skip-existing \
    > "${LOG_DIR}/260519_dock_eval_10k_noise_test.log"   2>&1 &
PID_N=$!
log "  density : pid $PID_D -> log/260519_dock_eval_10k_density_test.log"
log "  noise   : pid $PID_N -> log/260519_dock_eval_10k_noise_test.log"

wait "$PID_D"; RC_D=$?
wait "$PID_N"; RC_N=$?
log "docking finished — rc(density=$RC_D noise=$RC_N)"

# --- Phase 2: 2-arm comparison summary (mean over test pockets) ----------
log "=============================================================="
python - "$DENSITY_DIR" "$NOISE_DIR" <<'PYEOF'
import json, sys, glob, os
import numpy as np

arms = [("density", sys.argv[1]), ("noise", sys.argv[2])]
keys = ["validity", "uniqueness", "diversity", "qed_mean", "sa_mean",
        "vina_score_mean", "vina_min_mean", "vina_dock_mean"]
hdr = f"{'arm':<10}" + "".join(f"{k:>16}" for k in keys)
print(hdr)
print("-" * len(hdr))
for name, root in arms:
    acc = {k: [] for k in keys}
    nt = 0
    for mp in sorted(glob.glob(os.path.join(root, "target_*", "metrics.json"))):
        try:
            agg = json.load(open(mp)).get("aggregates", {})
        except Exception:
            continue
        nt += 1
        for k in keys:
            v = agg.get(k)
            if isinstance(v, (int, float)):
                acc[k].append(v)
    row = f"{name:<10}"
    for k in keys:
        vals = acc[k]
        row += f"{np.mean(vals):>16.3f}" if vals else f"{'n/a':>16}"
    print(row + f"   ({nt} pockets)")
print()
print("Vina affinities in kcal/mol — lower (more negative) = stronger binding.")
PYEOF
log "=============================================================="
log "dock-eval (10k test) done. per-target metrics.json under each ${SUB}/."
