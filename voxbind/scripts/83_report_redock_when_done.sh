#!/usr/bin/env bash
# 83_report_redock_when_done.sh — wait for 82_redock_ours_crop_exh16_79.sh to finish, then
# diff the re-dock against the published numbers and refresh the model_zoo backup.
#
# Split out from 82 rather than appended to it because 82 was already running when this was
# written, and because the compare is cheap to re-run by hand on the checkpointed JSON:
#   python scripts/tools/compare_redock_vs_published.py OLD.json NEW.json
#
#   bash scripts/83_report_redock_when_done.sh
#
# Polls for the driver PROCESS rather than for the output file: run_docking_eval.py
# checkpoints eval_docking_results_rerun.json after every completed target, so the file
# appearing means "one target done", not "run finished".
set -uo pipefail
cd /home1/irteam/VoxBind

V=voxbind/exps
DIR=${DIR:-$V/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350}
OLD=${OLD:-$DIR/eval_docking_results.json}
NEW=${NEW:-$DIR/eval_docking_results_rerun.json}
ZOO=${ZOO:-voxbind/model_zoo/generated_samples/ours_v1_frozenenc_atomblob7_v2p1_sig0.9_ep350}
LOGDIR=${LOGDIR:-$V/frozenenc_probes/logs/redock_crop_exh16}
REPORT=${REPORT:-$LOGDIR/redock_vs_published.md}
POLL=${POLL:-300}
TIMEOUT=${TIMEOUT:-172800}
PY=/opt/conda/envs/voxdock/bin/python

mkdir -p "$LOGDIR"
LOG="$LOGDIR/report_chain.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the re-dock driver to exit (poll ${POLL}s)"
waited=0
while pgrep -f "run_docking_eval.py $DIR" >/dev/null 2>&1 || pgrep -f "run_docking_eval.py.*full_eval_ep350" >/dev/null 2>&1; do
    if (( waited >= TIMEOUT )); then
        say "TIMEOUT after ${waited}s with the driver still running - not reporting"
        exit 1
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done
say "driver gone after ${waited}s"

[ -f "$NEW" ] || { say "MISSING $NEW - the run produced nothing"; exit 1; }

say "comparing $NEW against $OLD"
"$PY" voxbind/scripts/tools/compare_redock_vs_published.py "$OLD" "$NEW" --md "$REPORT" \
    >>"$LOG" 2>&1
rc=$?
say "compare exit=$rc -> $REPORT"

# Keep the model_zoo backup in step with the sample dir: the re-run JSON is part of the
# provenance of these samples. Same exclusions as the original copy -- .vina_tmp_* is
# multi-hundred-MB Vina scratch, not data.
say "refreshing model_zoo backup"
rsync -a --exclude='.vina_tmp_*' --exclude='__pycache__' --exclude='.vina_cache' \
    "$DIR/" "$ZOO/samples/" >>"$LOG" 2>&1
cp -f "$REPORT" "$ZOO/" 2>/dev/null
say "backup refreshed -> $ZOO"
say "REDOCK_REPORT_DONE"
exit $rc
