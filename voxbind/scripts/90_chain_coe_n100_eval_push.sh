#!/usr/bin/env bash
# 90_chain_coe_n100_eval_push.sh
#   Wait for the VoxBind+CoE 100-molecules/pocket sampling to finish, score it, stage it into
#   the results bundle as `VoxBind+CoE`, re-collect, and push to Dropbox.
#
#   WHY NOT 73_evaluate_samples.sh (which does Vina + pose in one go): it needs ONE env with
#   rdkit + vina + meeko AND pdb2pqr30/obabel on PATH, and this box has no such env --
#   `voxdock` carries the docking stack but is python 3.8 (metrics.py uses PEP 585 generics
#   and dies with a TypeError), `voxbind` is 3.12 but has no vina module and no pdb2pqr30.
#   So the two halves run in the two envs that DO work here, exactly as the 2026-09-13
#   svr12-arms chain did:
#     pose + chem  85_fill_pose_eval_4runs.sh  -> moleval py3.10 (posecheck + posebusters)
#     Vina         73_dock_baseline_protocol_79.sh -> voxdock py3.8 (run_docking_eval.py)
#
#   metrics.py defaults to --docking none and ALWAYS computes the chemical block, so the pose
#   fill also produces sample_quality. Vina then lands in
#   samples/eval_docking_results_full79.json, which is precisely the file
#   collect_task2_eval.py falls back to for a native arm with no Vina rows in its per-target
#   metrics.json (collect_native, "vina_docking" not in out). That file is also the published
#   axis -- whole *_rec.pdb receptor, exhaustiveness 32, the 79 density pockets -- i.e. the
#   same protocol behind the "VoxBind + CDG" (CoDE) row of results/reports/table_drug_design.tex,
#   so the two rows are comparable.
#
#   THE BUNDLE FOLDER IS REPLACED, NOT MERGED (user decision, 2026-09-14): VoxBind+CoE on
#   Dropbox currently holds the 10/pocket tree. `rclone copy` never deletes, so a plain push
#   would leave a mix of 10- and 100-molecule files; this syncs that ONE folder with
#   --backup-dir, which MOVES every superseded remote file into
#   results/_archive/VoxBind+CoE_n10_<stamp>/ instead of deleting it. The 10/pocket numbers
#   also survive untouched in Fusion-default-cv2-scratch-ep350/ (same run, same molecules).
#
#   Env knobs: EXP, OUTDIR, FOLDER, LABEL, N, W, C, EXH, POLL, TIMEOUT, PUSH, SKIP_WAIT.
#
#   bash voxbind/scripts/90_chain_coe_n100_eval_push.sh
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 1

V=voxbind/exps
EXP="${EXP:-260908_fusion_default_cv2_scratch_8gpu}"
OUTDIR="${OUTDIR:-samples_ep350_test79_n100}"
SAMPLE="$V/$EXP/samples/$OUTDIR"          # relative: 85 and 73 both run from $ROOT
LABEL="${LABEL:-coe_n100}"
FOLDER="${FOLDER:-VoxBind+CoE}"
BUNDLE="results/task2-drugdesign/$FOLDER"
ZOO="voxbind/model_zoo/VoxBind+CoE"
PY_COLLECT="${PY_COLLECT:-/opt/conda/envs/voxbind/bin/python}"

N="${N:-5}"                                # parallel pose workers (85's default)
W="${W:-14}"; C="${C:-3}"; EXH="${EXH:-32}"  # 14x3 = 42 Vina threads; ~12 h for 79 pockets
POLL="${POLL:-300}"; TIMEOUT="${TIMEOUT:-43200}"
SKIP_WAIT="${SKIP_WAIT:-0}"
PUSH="${PUSH:-1}"
EXPECT_TARGETS="${EXPECT_TARGETS:-79}"

LOGDIR="${LOGDIR:-$V/frozenenc_probes/logs/coe_n100}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== COE n100 chain: sample-wait -> pose+chem -> vina -> stage -> collect -> push ==="
say "sample dir: $SAMPLE   bundle: $BUNDLE   push=$PUSH"

# ── 1. wait for the 4 sampling shards ────────────────────────────────────────────
# The shard exit codes are the completion signal, not the target-dir count: each GPU
# writes target dirs as it goes, so 79 can appear while a shard is still sampling.
if [ "$SKIP_WAIT" != "1" ]; then
    waited=0
    while :; do
        done_n=$(ls "$SAMPLE"/_run_gpu*/exit_code 2>/dev/null | wc -l)
        [ "$done_n" -ge 4 ] && break
        if [ "$waited" -ge "$TIMEOUT" ]; then
            say "TIMEOUT after ${waited}s with $done_n/4 shards finished — nothing scored"
            exit 1
        fi
        n_t=$(ls -d "$SAMPLE"/target_* 2>/dev/null | wc -l)
        say "waiting for sampling: shards=$done_n/4 targets=$n_t"
        sleep "$POLL"; waited=$((waited + POLL))
    done
    bad=0
    for f in "$SAMPLE"/_run_gpu*/exit_code; do
        rc=$(cat "$f" 2>/dev/null || echo 1)
        [ "$rc" -eq 0 ] || { say "shard $f exited $rc"; bad=1; }
    done
    [ "$bad" -eq 0 ] || { say "ABORT: a sampling shard failed"; exit 1; }
fi
n_t=$(ls -d "$SAMPLE"/target_* 2>/dev/null | wc -l)
n_sdf=$(find "$SAMPLE" -name samples.sdf 2>/dev/null | wc -l)
say "sampling done: $n_t target dirs, $n_sdf samples.sdf"
if [ "$n_t" -ne "$EXPECT_TARGETS" ]; then
    say "WARNING: expected $EXPECT_TARGETS target dirs, got $n_t — continuing anyway"
fi
[ "$n_sdf" -ge 1 ] || { say "ABORT: no molecules were written"; exit 1; }

# ── 2. PoseCheck + PoseBusters + chem, whole-receptor scope ──────────────────────
# POSE_SCOPE=full so the clash/interaction terms line up with every other VoxBind arm
# (metrics.py records the scope and DROPS a cached row whose scope disagrees).
say "[1/5] pose + chem fill (moleval, N=$N, scope=full)"
ONLY="$LABEL" EXTRA_ROOTS="$LABEL=$SAMPLE" POSE_SCOPE=full N="$N" \
    LOGDIR="$LOGDIR/posefill" bash voxbind/scripts/85_fill_pose_eval_4runs.sh >>"$LOG" 2>&1
say "[1/5] pose fill returned $?"

# ── 3. Vina under the published baseline protocol ────────────────────────────────
say "[2/5] vina (baseline protocol: scope=full exh=$EXH W=$W C=$C, 79 pockets)"
ARMS="$LABEL $SAMPLE" W="$W" C="$C" EXH="$EXH" LOGDIR="$LOGDIR/baseproto" \
    nice -n 15 bash voxbind/scripts/73_dock_baseline_protocol_79.sh >>"$LOG" 2>&1
say "[2/5] docking driver returned $?"

# ── 4. stage into the bundle ─────────────────────────────────────────────────────
# collect_task2_eval.py reads <bundle>/<Method>/samples/..., NOT voxbind/exps, and the bundle
# is pushed to Dropbox -- so this is a real copy, never a symlink. .vina_cache is receptor-prep
# scratch (~90% of the bytes, regenerated in minutes) and stays out, as in 86.
say "[3/5] staging -> $BUNDLE"
mkdir -p "$BUNDLE/samples" "$BUNDLE/run"
rsync -a --delete --exclude '.vina_cache/' --exclude '.vina_tmp*' "$SAMPLE/" "$BUNDLE/samples/"
# run identity: the ORIGINAL svr12 cfg from the zoo, not the locally path-patched copy
cp -u "$ZOO/cfg.yaml" "$BUNDLE/run/cfg.yaml" 2>/dev/null
cp -u "$ZOO/train_ddp.log" "$BUNDLE/run/train_ddp.log" 2>/dev/null
[ -d "$ZOO/.hydra" ] && rsync -a "$ZOO/.hydra/" "$BUNDLE/run/hydra/"

# ── 5. re-collect, then rebuild this folder's headline metrics.json ──────────────
say "[4/5] collect_task2_eval.py"
"$PY_COLLECT" voxbind/scripts/tools/collect_task2_eval.py >>"$LOG" 2>&1 \
    || say "collect_task2_eval.py returned $?"

# The folder's top-level metrics.json is the headline the reports read. 86 builds it from a
# 73-written samples/summary.json, which this split path never produces -- so compose it from
# the collector's own eval/*/results.json instead. Same schema as the 10/pocket file it
# replaces, same aggregation convention: vina_*_mean are per-pocket means, `pooled` is over
# molecules, high_affinity is the per-molecule rate.
say "[5/5] rebuilding $FOLDER/metrics.json from eval/*/results.json"
"$PY_COLLECT" - "$BUNDLE" "$FOLDER" "$EXP" "$OUTDIR" <<'PY' >>"$LOG" 2>&1
import json, os, socket, sys

bundle, folder, exp, outdir = sys.argv[1:5]


def ev(name):
    p = os.path.join(bundle, "eval", name, "results.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return (d.get("pocket_sets") or {}).get("all") or d


q, v = ev("sample_quality"), ev("vina_docking")
pc, pb = ev("posecheck"), ev("posebusters")
ppm = v.get("per_pocket_mean") or {}


def pooled(key):
    b = v.get(key) or {}
    return {"mean": b.get("mean"), "median": b.get("median"), "n": b.get("n")}


out = {
    "task": "drugdesign",
    "method": "VoxBind + CoE (100/pocket)",
    "folder": folder,
    "benchmark": "CrossDocked",
    "generated_on": socket.gethostname(),
    "source_run": f"voxbind/exps/{exp}",
    "source_samples": f"voxbind/exps/{exp}/samples/{outdir}",
    "what": (
        "the same run as Fusion-default-cv2-scratch-ep350 (C_v2 coords-only FROZEN encoder "
        "through fusion=default, the pathway CoDE uses), RE-SAMPLED at 100 molecules per "
        "pocket so it matches CoDE's draw budget -- the 10/pocket tree this replaces is still "
        "in Fusion-default-cv2-scratch-ep350/. Weights: model_zoo/VoxBind+CoE (frozen encoder "
        "model_zoo/C_v2 required to load them). Vina is the published baseline protocol "
        "(whole receptor, exhaustiveness 32); PoseCheck/PoseBusters at full-receptor scope."
    ),
    "n_targets": q.get("n_pockets") or v.get("n_pockets"),
    "n_molecules": q.get("n_molecules") or v.get("n_molecules"),
    "eval": {
        "receptor": os.environ.get(
            "FULL_RECEPTOR_ROOT",
            "/home1/irteam/VoxBind/targetdiff/data/test_set/test_set"),
        "n_targets": q.get("n_pockets"),
        "n_molecules": q.get("n_molecules"),
        "validity": q.get("validity"),
        "uniqueness": q.get("uniqueness"),
        "diversity": q.get("diversity"),
        "qed_mean": q.get("qed_mean"),
        "sa_mean": q.get("sa_mean"),
        "logp_mean": q.get("logp_mean"),
        "lipinski_mean": q.get("lipinski_mean"),
        "n_atoms_mean": q.get("n_atoms_mean"),
        "sim_to_ref_mean": q.get("sim_to_ref_mean"),
        "vina_score_mean": ppm.get("vina_score"),
        "vina_min_mean": ppm.get("vina_min"),
        "vina_dock_mean": ppm.get("vina_dock"),
        "high_affinity": v.get("high_affinity"),
        "vina_n_docked": v.get("n_docked"),
        "vina_n_failed": v.get("n_failed"),
        "clashes_median": pc.get("clashes_median"),
        "clashes_mean": pc.get("clashes_mean"),
        "strain_median": pc.get("strain_median"),
        "n_interactions_median": pc.get("n_interactions_median"),
        "n_posecheck": pc.get("n_scored"),
        "n_posecheck_err": pc.get("n_errors"),
        "pb_valid_rate": pb.get("pb_valid_rate"),
        "n_posebusters": pb.get("n_scored"),
        "n_posebusters_err": pb.get("n_errors"),
        "pb_checks_pass_rate": pb.get("checks_pass_rate") or {},
        "pooled": {
            "score_only": pooled("vina_score"),
            "minimize": pooled("vina_min"),
            "dock": pooled("vina_dock"),
        },
    },
    # Not a row in the 260910 PoseBusters figure -- missing must read as missing, not zero.
    "posebusters_260910": {"all": None, "density79": None},
    "report": "../../reports/results_drug_design.html",
}
json.dump(out, open(os.path.join(bundle, "metrics.json"), "w"), indent=2)
e = out["eval"]
print(f"   {folder}/metrics.json  ({out['n_targets']} targets, {out['n_molecules']} mols, "
      f"vina_dock {e['vina_dock_mean']}, pb_valid {e['pb_valid_rate']}, "
      f"clash_med {e['clashes_median']})")
PY

# ── 6. push ──────────────────────────────────────────────────────────────────────
if [ "$PUSH" = "1" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    STAMP=$(date '+%Y%m%d')
    DEST="dropbox:/박성현/VoxBind/results"
    say "[push 1/2] replacing $FOLDER on Dropbox (superseded files -> _archive/${FOLDER}_n10_$STAMP)"
    rclone sync "$BUNDLE/" "$DEST/task2-drugdesign/$FOLDER/" \
        --backup-dir "$DEST/_archive/${FOLDER}_n10_$STAMP" \
        --transfers 4 --checkers 8 >>"$LOG" 2>&1
    say "[push 1/2] rclone sync returned $?"
    # --update so a remote file that is NEWER than the local one is never clobbered by this
    # blanket pass (the bundle here was pulled, not authored, for most folders).
    say "[push 2/2] dropbox_push.sh for the rest of the bundle"
    bash results/dropbox_push.sh -y --update >>"$LOG" 2>&1
    say "[push 2/2] dropbox_push.sh returned $?"
else
    say "PUSH=0 — bundle staged and collected, nothing uploaded"
fi

say "COE_N100_DONE"
echo "COE_N100_DONE" >> "$LOG"
