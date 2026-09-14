#!/usr/bin/env bash
# 93_redock_coe_n100_chunked.sh
#   Finish the VoxBind+CoE n100 Vina leg that 90's docking step abandoned, then re-stage,
#   re-collect and re-push.
#
#   WHAT WENT WRONG (2026-09-14 06:36): run_docking_eval.py runs ONE shared
#   ProcessPoolExecutor over all 79 targets. A single Vina C++ abort
#   ("terminate called after throwing an instance of 'internal_error'") killed one worker,
#   and every target still queued died instantly with BrokenProcessPool -- 72 of 79 lost,
#   44 minutes in, while the driver still exited 0. It was NOT the pdb2pqr30/PATH null-score
#   trap (73 exports voxdock's bin and the finished molecules carry real scores at
#   17-45 s/mol) and NOT memory (cgroup 525/824 GB, ~1.6 TB free, no OOM traces).
#
#   THE FIX HERE IS ISOLATION, NOT TUNING: dock in chunks of $CHUNK targets, one driver
#   process per chunk, so an abort costs that chunk instead of the whole run. --skip-existing
#   keeps every target already in the output JSON, so the 7 finished targets (365 molecules)
#   and any chunk that succeeded are never re-docked; a retried chunk resumes from them too.
#
#   Waits for chain 90 to exit first: it is still pushing, and two rclone pushes or two
#   stagings of the same bundle folder would race.
#
#   Env knobs: CHUNK, W, C, EXH, PUSH.
set -uo pipefail
ROOT=/home1/irteam/VoxBind
cd "$ROOT" || exit 1

V=voxbind/exps
SAMPLE="$V/260908_fusion_default_cv2_scratch_8gpu/samples/samples_ep350_test79_n100"
OUT="$SAMPLE/eval_docking_results_full79.json"
BUNDLE="results/task2-drugdesign/VoxBind+CoE"
ZOO="voxbind/model_zoo/VoxBind+CoE"
DRIVER="$V/frozenenc_probes/run_docking_eval.py"
PY_DOCK=/opt/conda/envs/voxdock/bin/python
PY_COLLECT=/opt/conda/envs/voxbind/bin/python

CHUNK="${CHUNK:-8}"
W="${W:-8}"; C="${C:-2}"; EXH="${EXH:-32}"     # 16 Vina threads, gentler on the resident trainer
PUSH="${PUSH:-1}"
LOGDIR="$V/frozenenc_probes/logs/coe_n100/redock"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"
PIDFILE="$LOGDIR/chain.pid"
echo $$ > "$PIDFILE"
# Watchers and kills must target THIS pid, never `pgrep -f 93_redock...`: a watcher's own
# command line contains the script name, so a pattern kill takes the watcher down with the
# chain (exit 144). Same family as the pkill self-match trap in the ops notes.
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# pdb2pqr30 + obabel must be on PATH or every Vina call returns None in 0.0s and the run
# still exits 0 with a JSON full of nulls (73's header documents this trap).
export PATH="/opt/conda/envs/voxdock/bin:$PATH"
command -v pdb2pqr30 >/dev/null || { say "ABORT: pdb2pqr30 not on PATH"; exit 1; }

say "=== re-dock chain: wait for 90 -> chunked docking -> stage -> collect -> push ==="

while pgrep -f "90_chain_coe_n100_eval_push.sh" >/dev/null; do
    say "waiting for chain 90 (push in progress)"
    sleep 120
done
say "chain 90 finished"

remaining() {
    "$PY_COLLECT" - "$SAMPLE" "$OUT" <<'PY'
import glob, json, os, sys
sample, out = sys.argv[1], sys.argv[2]
done = set()
if os.path.exists(out):
    done = {t["target"] for t in json.load(open(out)).get("per_target", [])}
all_t = sorted(os.path.basename(p) for p in glob.glob(os.path.join(sample, "target_*")))
print(" ".join(t for t in all_t if t not in done))
PY
}

todo=($(remaining))
say "targets still undocked: ${#todo[@]} (of 79)"

dock_chunk() {   # $1 = comma-separated target names, $2 = tag
    nice -n 15 "$PY_DOCK" "$DRIVER" "$SAMPLE" \
        --scope full --workers "$W" --cpu "$C" --exhaustiveness "$EXH" \
        --targets "$1" --skip-existing --out "$OUT" \
        >"$LOGDIR/$2.log" 2>&1
    return $?
}

i=0
while [ "$i" -lt "${#todo[@]}" ]; do
    chunk=("${todo[@]:i:CHUNK}")
    csv=$(IFS=,; echo "${chunk[*]}")
    tag="chunk_$(printf '%02d' $((i / CHUNK)))"
    say "$tag: ${#chunk[@]} targets ($csv)"
    dock_chunk "$csv" "$tag"; rc=$?
    if [ "$rc" -ne 0 ] || grep -q "BrokenProcessPool" "$LOGDIR/$tag.log" 2>/dev/null; then
        say "$tag: rc=$rc / pool broken — retrying once (resumes from what it already scored)"
        dock_chunk "$csv" "${tag}_retry"; rc=$?
        say "$tag retry rc=$rc"
    fi
    i=$((i + CHUNK))
done

left=($(remaining))
say "docking pass done; still undocked: ${#left[@]} ${left[*]:0:6}"

# ── stage + collect + headline metrics (same steps 90 ran, now on complete Vina) ──
say "staging -> $BUNDLE"
mkdir -p "$BUNDLE/samples" "$BUNDLE/run"
rsync -a --delete --exclude '.vina_cache/' --exclude '.vina_tmp*' "$SAMPLE/" "$BUNDLE/samples/"
cp -u "$ZOO/cfg.yaml" "$BUNDLE/run/cfg.yaml" 2>/dev/null
cp -u "$ZOO/train_ddp.log" "$BUNDLE/run/train_ddp.log" 2>/dev/null

say "collect_task2_eval.py"
"$PY_COLLECT" voxbind/scripts/tools/collect_task2_eval.py >>"$LOG" 2>&1 \
    || say "collect_task2_eval.py returned $?"

say "rebuilding $BUNDLE/metrics.json"
"$PY_COLLECT" - "$BUNDLE" >>"$LOG" 2>&1 <<'PY'
import json, os, sys
bundle = sys.argv[1]
def ev(name):
    p = os.path.join(bundle, "eval", name, "results.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    ps = d.get("pocket_sets") or {}
    # quality/posecheck/posebusters envelopes key their set "all"; the collector's Vina
    # fallback keys it "density79" (collect_native, eval_docking_results_full79.json). Reading
    # only "all" is why the first rebuild wrote vina_* = None even for the targets that HAD
    # docked.
    return ps.get("all") or ps.get("density79") or d
q, v, pc, pb = ev("sample_quality"), ev("vina_docking"), ev("posecheck"), ev("posebusters")
ppm = v.get("per_pocket_mean") or {}
def pooled(k):
    b = v.get(k) or {}
    return {"mean": b.get("mean"), "median": b.get("median"), "n": b.get("n")}
m = json.load(open(os.path.join(bundle, "metrics.json")))
m["eval"].update({
    "vina_score_mean": ppm.get("vina_score"), "vina_min_mean": ppm.get("vina_min"),
    "vina_dock_mean": ppm.get("vina_dock"), "high_affinity": v.get("high_affinity"),
    "vina_n_docked": v.get("n_docked"), "vina_n_failed": v.get("n_failed"),
    "vina_n_pockets": v.get("n_pockets"), "vina_n_molecules": v.get("n_molecules"),
    "pooled": {"score_only": pooled("vina_score"), "minimize": pooled("vina_min"),
               "dock": pooled("vina_dock")},
})
json.dump(m, open(os.path.join(bundle, "metrics.json"), "w"), indent=2)
e = m["eval"]
print(f"   metrics.json: vina pockets={e.get('vina_n_pockets')} mols={e.get('vina_n_molecules')} "
      f"dock={e.get('vina_dock_mean')} HA={e.get('high_affinity')}")
PY

if [ "$PUSH" = "1" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    DEST="dropbox:/박성현/VoxBind/results"
    STAMP=$(date '+%Y%m%d_%H%M')
    say "push: syncing $BUNDLE (superseded -> _archive/VoxBind+CoE_partialvina_$STAMP)"
    rclone sync "$BUNDLE/" "$DEST/task2-drugdesign/VoxBind+CoE/" \
        --backup-dir "$DEST/_archive/VoxBind+CoE_partialvina_$STAMP" \
        --transfers 4 --checkers 8 >>"$LOG" 2>&1
    say "rclone sync returned $?"
    bash results/dropbox_push.sh -y --update >>"$LOG" 2>&1
    say "dropbox_push.sh returned $?"
    # Dropbox rate-limits batches (`too_many_write_operations`), and 90's push already hit it
    # twice -- a sync that "returned 0" can still have dropped files. Prove what landed.
    diffs=$(rclone check "$BUNDLE/" "$DEST/task2-drugdesign/VoxBind+CoE/" --one-way 2>&1 \
            | grep -cE "^[0-9/]* *ERROR|: file not in")
    say "rclone check: $diffs file(s) missing/differing on Dropbox (0 = fully uploaded)"
    if [ "${diffs:-1}" -ne 0 ]; then
        say "re-syncing once to pick up rate-limited files"
        rclone sync "$BUNDLE/" "$DEST/task2-drugdesign/VoxBind+CoE/" \
            --transfers 2 --checkers 4 --tpslimit 6 >>"$LOG" 2>&1
        say "re-sync returned $?"
    fi
fi

say "COE_REDOCK_DONE undocked_left=${#left[@]}"
echo "COE_REDOCK_DONE undocked_left=${#left[@]}" >> "$LOG"
