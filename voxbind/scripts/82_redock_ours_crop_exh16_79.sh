#!/usr/bin/env bash
# 82_redock_ours_crop_exh16_79.sh — independently re-dock the SAME sampled ligands that
# produced the "Ours" row of Table 4 in notebook/html/results.html, to confirm the
# published Vina numbers.
#
# What is being reproduced (results.html Table 4, gold "Ours" row):
#   run     voxbind_frozenenc_atomblob7_v2p1_sig0.9 (frozen C+D+G encoder, sigma=0.9,
#           num_epochs=350 -> checkpoint.pth.tar)
#   samples exps/<run>/samples/full_eval_ep350, 79 x-ray-density pockets x 100 mols
#   source  eval_docking_results.json (2026-07-05)
#           Score -5.5134  Min -7.0898  Dock -8.2382  HighAff 67.52%
#
# Protocol is the CROP protocol those numbers were measured under -- pocket10 receptor,
# exhaustiveness 16 -- NOT the 260903 baseline protocol of 73_dock_baseline_protocol_79.sh
# (full receptor, exh 32), which is a different table and gives different absolutes.
#
# Vina's `dock` mode runs with seed=0, i.e. a RANDOM seed (TargetDIff docking_vina.py:133),
# so this cannot reproduce bit-for-bit: score_only/minimize are deterministic and must match
# exactly, while dock carries run-to-run search noise. That is the point of the exercise.
#
# Writes eval_docking_results_rerun.json; the original eval_docking_results.json is left
# untouched so the two can be diffed.
#
#   W=20 bash scripts/82_redock_ours_crop_exh16_79.sh
#
# W is the core budget. The box is capped at 32 cores by its cgroup quota (nproc reports
# 128 and lies), so W must leave room for whatever GPU job is co-running or its dataloader
# starves and GPU util sawtooths. W=20 is the published config (cpu=1 x 20 workers); drop
# to W=10 when sharing the box with a training run.
set -uo pipefail
cd /home1/irteam/VoxBind
export PATH="/opt/conda/envs/voxdock/bin:$PATH"

V=voxbind/exps
DRIVER=$V/frozenenc_probes/run_docking_eval.py
DIR=${DIR:-$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350}
TARGETS=${TARGETS:-$V/frozenenc_probes/p79_targets.json}
SCOPE=${SCOPE:-crop}
OUT=${OUT:-$DIR/eval_docking_results_rerun.json}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/redock_${SCOPE}_exh${EXH:-16}}
W=${W:-20}; C=${C:-1}; EXH=${EXH:-16}
PY=/opt/conda/envs/voxdock/bin/python

# SCOPE=full EXH=32 re-checks the OTHER published protocol instead: results_drug_design.html
# scores against the whole receptor at exhaustiveness 32 ("the same protocol the published
# baselines use"), which is eval_docking_results_full79.json (-6.58/-7.64/-8.49), NOT the
# crop numbers in results.html Table 4 (-5.51/-7.09/-8.24). Always pass a matching OUT.
mkdir -p "$LOGDIR"
LOG="$LOGDIR/ours_v1_${SCOPE}_exh${EXH}.log"

# Every VoxBind Vina number in the paper is produced with vina 1.2.2 (the voxdock env);
# 1.2.7 lives in funcbind/.repro-env and shifts absolute affinities. Refuse rather than
# silently compare across builds -- that would make the whole re-check meaningless.
VINA_REQ=${VINA_REQ:-1.2.2}
vina_ver=$("$PY" -c 'import importlib.metadata as m; print(m.version("vina"))' 2>/dev/null)
[ "$vina_ver" = "$VINA_REQ" ] || { echo "[82] WRONG VINA: have ${vina_ver:-none}, need $VINA_REQ"; exit 1; }

# VinaDockingTask SHELLS OUT to pdb2pqr30/obabel; without the env's bin on PATH those calls
# raise FileNotFoundError, dock_all_modes swallows it into None, and the run "succeeds" with
# every affinity null in 0.0s.
for bin in pdb2pqr30 obabel; do
    command -v "$bin" >/dev/null || { echo "[82] MISSING $bin on PATH"; exit 1; }
done
[ -f "$DRIVER" ]  || { echo "[82] MISSING $DRIVER"; exit 1; }
[ -f "$TARGETS" ] || { echo "[82] MISSING $TARGETS"; exit 1; }

echo "[$(date '+%F %T')] START $SCOPE/exh$EXH W=$W C=$C -> $OUT" | tee -a "$LOG"
"$PY" "$DRIVER" "$DIR" \
    --scope "$SCOPE" --workers "$W" --cpu "$C" --exhaustiveness "$EXH" \
    --targets "$TARGETS" --skip-existing --out "$OUT" \
    >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE rc=$rc -> $OUT" | tee -a "$LOG"
exit $rc
