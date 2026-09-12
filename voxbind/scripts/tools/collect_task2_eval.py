#!/usr/bin/env python3
"""collect_task2_eval.py — one canonical home per (method, evaluation) in the results bundle.

WHY THIS EXISTS. task2 was evaluated on at least three boxes and the numbers never landed
in one place: the svr12 arms keep theirs per molecule in voxbind/exps/<run>/samples/
target_*/metrics.json, the five published baselines keep theirs in _shared/baselines_eval/
*.json (one array per metric, arms keyed by yet another naming scheme), the 260910 figure
code reads /home1/irteam/..., and rigid-fragment lives only as a CSV under notebook/html.
The same arm therefore shows up under four names and nothing says which of the five
evaluations a given method actually has. This walks every source that IS in the bundle and
writes, per method:

    results/task2-drugdesign/<Method>/eval/
        index.json                    which evaluations exist, from where, how many molecules
        vina_docking/results.json     Vina score_only / minimize / dock + high-affinity %
        sample_quality/results.json   validity, uniqueness, diversity, QED, SA, logP, Lipinski
        posebusters/results.json      dock-mode validity + the 20 per-check pass rates
        posecheck/results.json        steric clashes, torsional strain, interaction profile
        rigid_fragment/results.json   rigid-fragment RMSD against fragment size

plus EVAL_STATUS.{json,md} at the task root: the method x evaluation matrix, and the list of
runs that were generated but never (fully) evaluated.

IT ONLY EVER COPIES. Nothing is moved out of voxbind/exps/ or out of _shared/, and no
existing <Method>/metrics.json is touched -- the reports read those. Every file this writes
is new and lives under <Method>/eval/.

RE-RUNNABLE ANYWHERE. Sources are read from inside results/ (and notebook/html/260910 for
rigid-fragment), never from /home1/irteam, so this produces the same tree on any box that
has the bundle. A results.json whose payload has not changed is left alone -- its mtime does
not move, so `rclone copy` in results/dropbox_push.sh skips it instead of re-uploading.

    python3 voxbind/scripts/tools/collect_task2_eval.py            # write
    python3 voxbind/scripts/tools/collect_task2_eval.py --check    # verify, write nothing
    python3 voxbind/scripts/tools/collect_task2_eval.py --dry-run  # show what would change
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T2 = ROOT / "results" / "task2-drugdesign"
SHARED = T2 / "_shared"
NB = ROOT / "notebook" / "html" / "260910"
CONSISTENCY = NB / "fig-consistency"

EVALS = ("vina_docking", "sample_quality", "posebusters", "posecheck", "rigid_fragment")
THIS_BOX = socket.gethostname()
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
TOOL = "voxbind/scripts/tools/collect_task2_eval.py"

# The reporting box the 260910 figures and the pre-svr12 arms were evaluated on. Recorded
# verbatim in `evaluated_on` so a number that did NOT come from this machine says so.
OTHER_BOX = "reporting box (/home1/irteam)"


# ----------------------------------------------------------------- small helpers
def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def r4(x, nd: int = 4):
    """Round, tolerating the empty cells the rigid-fragment CSVs carry for absent bins."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def nums(vals):
    return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]


def mean(vals, nd: int = 4):
    v = nums(vals)
    return r4(st.mean(v), nd) if v else None


def median(vals, nd: int = 4):
    v = nums(vals)
    return r4(st.median(v), nd) if v else None


def quantile(vals, q, nd: int = 4):
    v = sorted(nums(vals))
    if not v:
        return None
    i = (len(v) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return r4(v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo), nd)


def envelope(folder, label, evaluation, *, source, generated_on, evaluated_on,
             protocol=None, notes=None, **extra):
    d = {
        "method": label,
        "folder": folder,
        "evaluation": evaluation,
        "generated_on": generated_on,
        "evaluated_on": evaluated_on,
        "source": source,
        "protocol": protocol or {},
    }
    d.update(extra)
    if notes:
        d["notes"] = notes
    d["written_by"] = TOOL
    d["written_at"] = NOW
    return d


STAMPS = ("written_at", "built_at")


def payload(d):
    """The comparison key: everything but the timestamps, so an unchanged file stays put.

    Rewriting a byte-identical file would move its mtime and make `rclone copy` in
    results/dropbox_push.sh re-upload the whole eval tree on every collect.
    """
    return json.dumps({k: v for k, v in d.items() if k not in STAMPS},
                      sort_keys=True, ensure_ascii=False)


class Writer:
    def __init__(self, dry: bool):
        self.dry, self.written, self.unchanged = dry, [], []

    def __call__(self, path: Path, obj: dict):
        if path.exists():
            try:
                if payload(json.loads(path.read_text())) == payload(obj):
                    self.unchanged.append(path)
                    return
            except (json.JSONDecodeError, OSError):
                pass
        self.written.append(path)
        if not self.dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ------------------------------------------------------- the 79 density pockets
def load_p79() -> set[str]:
    """The pocket set every density arm covers, taken from a file already in the bundle.

    pose_common.py resolves this as frozenenc_probes/p79_targets.json under /home1/irteam;
    the identical list rides inside _shared/260910_posecheck/posecheck_summary.json, so
    reading it there keeps this script runnable on a box that has only the bundle.
    """
    f = SHARED / "260910_posecheck" / "posecheck_summary.json"
    return set(json.loads(f.read_text())["pocket_set"]["targets"])


P79 = load_p79()


# -------------------------------------------------- per-target metrics.json trees
def load_run(run_dir: Path) -> dict | None:
    """Read a metrics.py output tree: {target_name: {agg, rows}} plus the run's scopes."""
    mjs = sorted(run_dir.glob("target_*/metrics.json"))
    if not mjs:
        return None
    targets, meta = {}, {}
    for mj in mjs:
        try:
            d = json.loads(mj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        targets[mj.parent.name] = {"agg": d.get("aggregates", {}), "rows": d.get("samples", []),
                                   # the pocket's deposited crystal ligand, scored by the
                                   # same pass -- this is where the Reference row comes from
                                   "ref": d.get("reference")}
        if not meta:
            meta = {k: d.get(k) for k in ("docking", "pose", "pose_receptor_scope",
                                          "dock_receptor_scope", "computed_at", "version")}
    if not targets:
        return None
    return {"dir": run_dir, "targets": targets, "meta": meta}


def sdf_total(run_dir: Path) -> int:
    n = 0
    for f in run_dir.glob("target_*/samples.sdf"):
        try:
            n += sum(1 for line in f.open(errors="ignore") if line.strip() == "$$$$")
        except OSError:
            pass
    return n


def slices_of(run: dict) -> dict[str, list[str]]:
    """`all` is the run's own coverage; `density79` only when the run is not already p79."""
    names = sorted(run["targets"])
    out = {"all": names}
    inter = [t for t in names if t in P79]
    if inter and len(inter) != len(names):
        out["density79"] = inter
    return out


def rows_of(run, targets):
    return [row for t in targets for row in run["targets"][t]["rows"]]


def aggs_of(run, targets):
    return [run["targets"][t]["agg"] for t in targets]


def q_sample_quality(run, targets):
    rows, aggs = rows_of(run, targets), aggs_of(run, targets)
    return {
        "n_pockets": len(targets),
        "n_molecules": len(rows),
        # validity / uniqueness / diversity are per-pocket quantities: averaged over
        # pockets, which is the axis the CrossDocked papers report them on.
        "validity": mean(a.get("validity") for a in aggs),
        "uniqueness": mean(a.get("uniqueness") for a in aggs),
        "diversity": mean(a.get("diversity") for a in aggs),
        # TWO AXES, BOTH NAMED. `*_mean` pools every molecule; `*_mean_over_pockets`
        # averages the per-pocket means, which is the axis summary.json and the
        # CrossDocked papers report. They differ the moment pockets yield unequal
        # numbers of molecules, so neither is allowed to stand in for the other.
        "qed_mean": mean(x.get("qed") for x in rows),
        "qed_median": median(x.get("qed") for x in rows),
        "qed_mean_over_pockets": mean(a.get("qed_mean") for a in aggs),
        "sa_mean": mean(x.get("sa") for x in rows),
        "sa_median": median(x.get("sa") for x in rows),
        "sa_mean_over_pockets": mean(a.get("sa_mean") for a in aggs),
        "logp_mean": mean(x.get("logp") for x in rows),
        "logp_mean_over_pockets": mean(a.get("logp_mean") for a in aggs),
        "lipinski_mean": mean(x.get("lipinski") for x in rows),
        "lipinski_mean_over_pockets": mean(a.get("lipinski_mean") for a in aggs),
        "n_atoms_mean": mean(x.get("n_atoms") for x in rows),
        "n_atoms_median": median(x.get("n_atoms") for x in rows),
        "sim_to_ref_mean": mean(x.get("sim_to_ref") for x in rows),
        "sim_to_ref_mean_over_pockets": mean(a.get("sim_to_ref_mean") for a in aggs),
    }


def q_vina(run, targets):
    rows, aggs = rows_of(run, targets), aggs_of(run, targets)
    out = {"n_pockets": len(targets), "n_molecules": len(rows)}
    any_field = False
    for field, label in (("score_only", "vina_score"), ("minimize", "vina_min"), ("dock", "vina_dock")):
        v = nums(x["vina"].get(field) for x in rows
                 if isinstance(x.get("vina"), dict) and "error" not in x["vina"])
        if v:
            any_field = True
        out[label] = {"mean": mean(v), "median": median(v), "n": len(v)} if v else None
    if not any_field:
        return None
    # Mean over pockets as well as over molecules: the papers report the former, and a
    # pocket that yielded more molecules must not weigh more heavily in it.
    out["per_pocket_mean"] = {
        "vina_score": mean(a.get("vina_score_mean") for a in aggs),
        "vina_min": mean(a.get("vina_min_mean") for a in aggs),
        "vina_dock": mean(a.get("vina_dock_mean") for a in aggs),
    }
    ha_n = sum(a.get("high_affinity_n") or 0 for a in aggs)
    ha_t = sum(a.get("high_affinity_total") or 0 for a in aggs)
    out["high_affinity"] = r4(ha_n / ha_t) if ha_t else None
    out["high_affinity_n"], out["high_affinity_total"] = ha_n, ha_t
    out["n_docked"] = sum(a.get("vina_n_docked") or 0 for a in aggs)
    out["n_failed"] = sum(a.get("vina_n_failed") or 0 for a in aggs)
    return out


def q_posecheck(run, targets):
    rows = rows_of(run, targets)
    ok = [x["posecheck"] for x in rows
          if isinstance(x.get("posecheck"), dict) and "error" not in x["posecheck"]]
    n_err = sum(1 for x in rows
                if isinstance(x.get("posecheck"), dict) and "error" in x["posecheck"])
    if not ok:
        return None
    strain = [p.get("strain") for p in ok]
    inter: dict[str, list[float]] = {}
    for p in ok:
        for k, v in (p.get("interactions") or {}).items():
            inter.setdefault(k, []).append(v)
    return {
        "n_pockets": len(targets),
        "n_molecules": len(rows),
        "n_scored": len(ok),
        "n_errors": n_err,
        "clashes_mean": mean(p.get("clashes") for p in ok),
        "clashes_median": median(p.get("clashes") for p in ok),
        "strain_median": median(strain),
        "strain_q25": quantile(strain, 0.25),
        "strain_q75": quantile(strain, 0.75),
        # The mean is dominated by a handful of 1e9 relaxation blow-ups and is not a
        # statistic anyone should quote -- the name says so rather than hiding it.
        "strain_mean_UNRELIABLE": mean(strain),
        "n_interactions_mean": mean(p.get("n_interactions") for p in ok),
        "n_interactions_median": median(p.get("n_interactions") for p in ok),
        "interactions_mean": {k: r4(sum(v) / len(ok)) for k, v in sorted(inter.items())},
    }


def q_posebusters(run, targets):
    rows = rows_of(run, targets)
    ok = [x["posebusters"] for x in rows
          if isinstance(x.get("posebusters"), dict) and "error" not in x["posebusters"]]
    n_err = sum(1 for x in rows
                if isinstance(x.get("posebusters"), dict) and "error" in x["posebusters"])
    if not ok:
        return None
    seen: dict[str, int] = {}
    passed: dict[str, int] = {}
    for b in ok:
        for k, v in (b.get("checks") or {}).items():
            if k in ("mol_pred_loaded", "mol_cond_loaded") or v is None:
                continue
            seen[k] = seen.get(k, 0) + 1
            passed[k] = passed.get(k, 0) + bool(v)
    valid_n = sum(bool(b.get("valid")) for b in ok)
    return {
        "n_pockets": len(targets),
        "n_molecules": len(rows),
        "n_scored": len(ok),
        "n_errors": n_err,
        "pb_valid_rate": r4(valid_n / len(ok)),
        "pb_valid_n": valid_n,
        "checks_pass_rate": {k: r4(passed[k] / seen[k]) for k in sorted(seen)},
    }


QUERIES = {
    "sample_quality": q_sample_quality,
    "vina_docking": q_vina,
    "posecheck": q_posecheck,
    "posebusters": q_posebusters,
}


# ============================================================ the method registry
# Everything below decides, per folder, WHERE its five evaluations come from. Three
# shapes exist in this bundle and each needs its own reader:
#
#   native     an arm sampled + scored on svr12 by 72_sample_8gpu.sh / 73_evaluate_samples.sh
#              -> per-molecule rows in <folder>/samples/target_*/metrics.json
#   baseline   one of the five published models, scored on svr12 from the targetdiff meta
#              bundles -> aggregate rows in _shared/baselines_eval/*.json
#   imported   a row whose molecules were scored on the reporting box and reached the
#              bundle as a headline only -> <folder>/metrics.json
#
# Folder -> the name the 260910 figures use, so a reader can join the two. `CoDE` is the
# current paper name for what the bundle still files as VoxBind-Ours / ours_v1.
FIGURE_KEY = {
    "VoxBind-Ours": "CoDE (ours_v1)",
    "VoxBind": "VoxBind (vanilla ep923)",
    "Reference": "Reference ligand",
}
# _shared/baselines_eval rows are keyed by the generator's own name, the bundle by folder.
BASELINE_ROW = {
    "AR": "AR", "Pocket2Mol": "Pocket2Mol", "DiffSBDD": "DiffSBDD",
    "DecompDiff": "DecompDiff_ref_prior", "FuncBind": "FuncBind",
}
# Folders that are a headline copy of a run evaluated elsewhere -- no molecules here to
# re-derive from, so they carry what their metrics.json holds and say where it came from.
IMPORTED = {
    "VoxBind": "voxbind/exps/_vanilla_ep923/samples/full_eval_ep923 (reporting box)",
    "Ours-v2": "voxbind/exps/samples_reference_receptor_ed_ep350 (reporting box)",
}


def label_of(folder: Path) -> str:
    mj = folder / "metrics.json"
    if mj.exists():
        try:
            return json.loads(mj.read_text()).get("method") or folder.name
        except (json.JSONDecodeError, OSError):
            pass
    return folder.name


def head(folder: Path) -> dict:
    mj = folder / "metrics.json"
    if not mj.exists():
        return {}
    try:
        return json.loads(mj.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def protocol_of(run: dict, extra=None) -> dict:
    m = run["meta"]
    p = {
        "dock_receptor_scope": m.get("dock_receptor_scope"),
        "pose_receptor_scope": m.get("pose_receptor_scope"),
        "docking": m.get("docking"),
        "pose": m.get("pose"),
        "metrics_version": m.get("version"),
        "computed_at": m.get("computed_at"),
        "engine": "notebook/webapp/metrics.py via voxbind/scripts/73_evaluate_samples.sh",
    }
    p.update(extra or {})
    return p


# ----------------------------------------------------------------- native arms
def collect_native(folder: Path, out: dict):
    run_chem = load_run(folder / "samples")
    if run_chem is None:
        return False
    # VoxBind-Ours keeps its pose eval in a second tree scored against the WHOLE receptor;
    # samples/ holds the pocket10-crop pass, which is not comparable with the other arms.
    pose_dir = folder / "pose_full"
    run_pose = load_run(pose_dir) if pose_dir.is_dir() else None
    src_pose = run_pose or run_chem

    label = label_of(folder)
    hd = head(folder)
    gen_on = hd.get("generated_on") or "svr12"
    sl_chem, sl_pose = slices_of(run_chem), slices_of(src_pose)

    for name, fn, run, slices in (
        ("sample_quality", q_sample_quality, run_chem, sl_chem),
        ("vina_docking", q_vina, run_chem, sl_chem),
        ("posecheck", q_posecheck, src_pose, sl_pose),
        ("posebusters", q_posebusters, src_pose, sl_pose),
    ):
        sets = {k: fn(run, t) for k, t in slices.items()}
        sets = {k: v for k, v in sets.items() if v}
        if not sets:
            continue
        out[name] = envelope(
            folder.name, label, name,
            source=rel(run["dir"]),
            generated_on=gen_on,
            evaluated_on=THIS_BOX if gen_on == "svr12" else OTHER_BOX,
            protocol=protocol_of(run),
            source_run=hd.get("source_run"),
            source_samples=hd.get("source_samples"),
            pocket_sets=sets,
        )

    # Ours' Vina lives in its own docking JSON, not in the per-target metrics.json tree.
    if "vina_docking" not in out:
        dj = folder / "samples" / "eval_docking_results_full79.json"
        if dj.exists():
            d = json.loads(dj.read_text())
            s, per = d["summary"], d["per_target"]
            out["vina_docking"] = envelope(
                folder.name, label, "vina_docking",
                source=rel(dj),
                generated_on=gen_on,
                evaluated_on=OTHER_BOX,
                protocol={"dock_receptor_scope": s.get("receptor_scope"),
                          "exhaustiveness": 32,
                          "engine": "eval_docking (reporting box)"},
                pocket_sets={"density79": {
                    "n_pockets": s.get("n_targets"),
                    "n_molecules": sum(t.get("n_valid") or 0 for t in per),
                    "vina_score": {"mean": r4(s.get("vina_score")),
                                   "median": median(t.get("vina_score") for t in per)},
                    "vina_min": {"mean": r4(s.get("vina_min")),
                                 "median": median(t.get("vina_min") for t in per)},
                    "vina_dock": {"mean": r4(s.get("vina_dock")),
                                  "median": median(t.get("vina_dock") for t in per)},
                    "high_affinity": r4(s.get("high_affinity")),
                }},
                notes=["Means are unweighted over per-target means, the aggregation "
                       "VoxBind-Ours/metrics.json documents; medians are over targets too.",
                       "Cached Vina Dock scores do not imply docked coordinates -- the SDF "
                       "poses are as generated."],
            )
    return True


# -------------------------------------------------------------- five baselines
def load_shared(name):
    f = SHARED / "baselines_eval" / name
    return json.loads(f.read_text()) if f.exists() else []


def row_for(rows, key):
    return next((r for r in rows if r.get("baseline") == key), None)


SHARED_SETS = {
    "all": ("summary.json", "posebusters_summary.json", "posecheck_summary.json"),
    "density79": ("summary_density79.json", "posebusters_summary_density79.json",
                  "posecheck_summary_density79.json"),
}


def strain_quartiles(folder_name: str):
    """q25/q75 per slice from the per-molecule PoseCheck dump, to match the native arms."""
    f = SHARED / "260910_posecheck" / f"posecheck_{folder_name}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    p79 = set(d.get("density79_pockets") or [])
    mols = d.get("molecules") or []
    out = {}
    for key, keep in (("all", None), ("density79", p79)):
        s = [m.get("s") for m in mols if keep is None or m.get("p") in keep]
        if nums(s):
            out[key] = {"strain_q25": quantile(s, 0.25), "strain_q75": quantile(s, 0.75),
                        "n_strain": len(nums(s))}
    return out


def collect_baseline(folder: Path, out: dict):
    key = BASELINE_ROW[folder.name]
    label = label_of(folder)
    quart = strain_quartiles(folder.name)
    src_note = ("prj-denovo/baselines on svr12; molecules in this bundle at "
                f"{folder.name}/samples/meta/*.pt")
    common = dict(generated_on="svr12", evaluated_on=THIS_BOX)

    quality, vina, pb, pc = {}, {}, {}, {}
    for sl, (f_sum, f_pb, f_pc) in SHARED_SETS.items():
        s = row_for(load_shared(f_sum), key)
        if s:
            quality[sl] = {
                "n_pockets": s.get("n_pockets"), "n_molecules": s.get("n_molecules"),
                "diversity": r4(s.get("diversity")),
                "diversity_median": r4(s.get("diversity_median")),
                "qed_mean": r4(s.get("qed_mean")), "qed_median": r4(s.get("qed_median")),
                "sa_mean": r4(s.get("sa_mean")), "sa_median": r4(s.get("sa_median")),
                "n_atoms_mean": r4(s.get("size_mean")), "n_atoms_median": r4(s.get("size_median")),
            }
            vina[sl] = {
                "n_pockets": s.get("n_pockets"), "n_molecules": s.get("n_molecules"),
                "vina_score": {"mean": r4(s.get("vina_score_mean")),
                               "median": r4(s.get("vina_score_median"))},
                "vina_min": {"mean": r4(s.get("vina_min_mean")),
                             "median": r4(s.get("vina_min_median"))},
                "vina_dock": {"mean": r4(s.get("vina_dock_mean")),
                              "median": r4(s.get("vina_dock_median"))},
                # Percentages in the shared rows, fractions in the native arms -- both are
                # kept under their own name rather than silently rescaled.
                "high_affinity_pct": r4(s.get("high_affinity")),
                "high_affinity_pct_per_draw": r4(s.get("high_affinity_per_draw")),
                "success_rate_pct": r4(s.get("success_rate")),
                "success_rate_pct_per_draw": r4(s.get("success_rate_per_draw")),
            }
        b = row_for(load_shared(f_pb), key)
        if b:
            pb[sl] = {"n_pockets": b.get("n_pockets"), "n_molecules": b.get("n_molecules"),
                      "n_scored": (b.get("n_molecules") or 0),
                      "n_errors": b.get("n_errors"),
                      "pb_valid_rate": r4(b.get("pb_valid_rate")),
                      "pb_valid_n": b.get("pb_valid_n"),
                      "checks_pass_rate": {k: r4(v) for k, v in
                                           sorted((b.get("checks_pass_rate") or {}).items())}}
        c = row_for(load_shared(f_pc), key)
        if c:
            pc[sl] = {
                "n_pockets": c.get("n_pockets"), "n_molecules": c.get("n_molecules"),
                "clashes_mean": r4(c.get("clash_mean")), "clashes_median": r4(c.get("clash_median")),
                "strain_median": r4(c.get("strain_median")),
                "strain_mean_UNRELIABLE": r4(c.get("strain_mean")),
                "interaction_coverage_pct": r4(c.get("interaction_coverage")),
                "interactions_mean": {k[:-5]: r4(v) for k, v in sorted(c.items())
                                      if k.endswith("_mean") and k not in
                                      ("clash_mean", "strain_mean")},
                **quart.get(sl, {}),
            }

    for name, sets, src, proto in (
        ("sample_quality", quality, "_shared/baselines_eval/summary*.json", {}),
        ("vina_docking", vina, "_shared/baselines_eval/summary*.json",
         {"dock_receptor_scope": "full", "engine": "TargetDiff VinaDockingTask"}),
        ("posebusters", pb, "_shared/baselines_eval/posebusters_summary*.json",
         {"mode": "dock", "engine": "posebusters"}),
        ("posecheck", pc, "_shared/baselines_eval/posecheck_summary*.json",
         {"pose_receptor_scope": "full",
          "engine": "posecheck 1.3.1 via baselines/_scripts/run_posecheck.py",
          "per_molecule_dump": rel(SHARED / "260910_posecheck" / f"posecheck_{folder.name}.json")}),
    ):
        if sets:
            out[name] = envelope(folder.name, label, name, source=src, protocol=proto,
                                 baseline_row_key=key, source_note=src_note, **common,
                                 pocket_sets=sets)
    # Draw accounting: how many molecules were asked for vs. written vs. scored.
    y = row_for(load_shared("yield.json"), key)
    if y and "sample_quality" in out:
        out["sample_quality"]["yield"] = {k: y[k] for k in
                                          ("draws", "n_written", "n_scored", "scored_pct", "exact")
                                          if k in y}
    return True


# ------------------------------------------------------------------- Reference
def collect_reference(folder: Path, out: dict):
    hd = head(folder)
    v = hd.get("vina") or {}
    if not v:
        return False
    quality, vina = {}, {}
    for sl, s in v.items():
        quality[sl] = {"n_pockets": s.get("n_pockets"), "n_molecules": s.get("n_molecules"),
                       "qed_mean": r4(s.get("qed_mean")), "qed_median": r4(s.get("qed_median")),
                       "sa_mean": r4(s.get("sa_mean")), "sa_median": r4(s.get("sa_median")),
                       "n_atoms_mean": r4(s.get("size_mean")),
                       "n_atoms_median": r4(s.get("size_median"))}
        vina[sl] = {"n_pockets": s.get("n_pockets"), "n_molecules": s.get("n_molecules"),
                    "vina_score": {"mean": r4(s.get("vina_score_mean")),
                                   "median": r4(s.get("vina_score_median"))},
                    "vina_min": {"mean": r4(s.get("vina_min_mean")),
                                 "median": r4(s.get("vina_min_median"))},
                    "vina_dock": {"mean": r4(s.get("vina_dock_mean")),
                                  "median": r4(s.get("vina_dock_median"))},
                    "success_rate_pct": r4(s.get("success_rate"))}
    note = ("Not generated: the deposited CrossDocked2020 crystal ligand of each pocket, put "
            "through the identical Vina protocol. This is the energy the High-Affinity % of "
            "every other row is measured against.")
    common = dict(generated_on="crystal structure (not generated)", evaluated_on="svr12",
                  source=rel(folder / "metrics.json"), notes=[note])
    out["sample_quality"] = envelope(folder.name, label_of(folder), "sample_quality",
                                     protocol={}, pocket_sets=quality, **common)
    out["vina_docking"] = envelope(folder.name, label_of(folder), "vina_docking",
                                   protocol={"dock_receptor_scope": "full",
                                             "engine": "TargetDiff VinaDockingTask"},
                                   per_pocket_dump=rel(folder / "eval" / "per_pocket"),
                                   pocket_sets=vina, **common)

    # PoseBusters on the crystal ligand: metrics.py scores it alongside the generated
    # molecules and parks it in each target's `reference` block, so the benchmark row is
    # already computed -- it just had nowhere to live. Read from the widest arm (100
    # pockets); the ligand is the same molecule in every arm that carries one.
    donor = load_run(T2 / "VoxBind-base-ep350" / "samples")
    if donor:
        synth = {"dir": donor["dir"], "meta": donor["meta"], "targets": {
            t: {"agg": {}, "rows": [v["ref"]] if isinstance(v.get("ref"), dict) else []}
            for t, v in donor["targets"].items()}}
        sets = {k: q_posebusters(synth, t) for k, t in slices_of(synth).items()}
        sets = {k: v for k, v in sets.items() if v}
        if sets:
            out["posebusters"] = envelope(
                folder.name, label_of(folder), "posebusters",
                source=rel(donor["dir"]) + " (per-target `reference` block)",
                generated_on="crystal structure (not generated)", evaluated_on=THIS_BOX,
                protocol=protocol_of(donor, {"mode": "dock"}),
                pocket_sets=sets,
                notes=[note, "One molecule per pocket, so n_molecules == n_pockets.",
                       "Taken from VoxBind-base-ep350's run because it covers all 100 "
                       "pockets; the crystal ligand is identical in every arm."])
    return True


# ------------------------------------- headline-only rows evaluated on the other box
def collect_imported(folder: Path, out: dict):
    hd = head(folder)
    v = hd.get("vina_dock_kcal_mol") or {}
    if not v:
        return False
    out["vina_docking"] = envelope(
        folder.name, label_of(folder), "vina_docking",
        source=rel(folder / "metrics.json"),
        generated_on=OTHER_BOX, evaluated_on=OTHER_BOX,
        protocol={"dock_receptor_scope": "full", "engine": "eval_docking (reporting box)"},
        partial=True,
        pocket_sets={"density79": {"n_pockets": v.get("n_targets"),
                                   "n_molecules": None,
                                   "vina_dock": {"mean": r4(v.get("vina_dock_avg")),
                                                 "median": r4(v.get("vina_dock_med"))}}},
        notes=[f"Headline only -- the molecules were scored at {IMPORTED[folder.name]} and "
               "never reached this bundle, so nothing here can be re-derived.",
               "Listed as PARTIAL in EVAL_STATUS.md for that reason."],
    )
    return True


def collect_shared_posecheck(folder: Path, out: dict):
    """PoseCheck for the arms scored on the reporting box, from the 260910 shared summary.

    posecheck_summary.json carries TargetDiff / VoxBind(ep923) / CoDE at both the p79 and
    the full-coverage slice. Only arms with no per-molecule tree in this bundle need it --
    a native arm's own rows are richer and take precedence.
    """
    if "posecheck" in out:
        return
    f = SHARED / "260910_posecheck" / "posecheck_summary.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    arm = {"VoxBind": "VoxBind", "VoxBind-Ours": "CoDE"}.get(folder.name)
    row = (d.get("arms") or {}).get(arm)
    if not row:
        return
    sets = {}
    for src_key, dst_key in (("p79", "density79"), ("all_pockets", "all")):
        r = row.get(src_key)
        if not r:
            continue
        sets[dst_key] = {
            "n_pockets": 79 if dst_key == "density79" else row.get("pockets_all"),
            "n_molecules": r.get("n_molecules"), "n_scored": r.get("n_strain"),
            "clashes_mean": r4(r.get("clash_mean")), "clashes_median": r4(r.get("clash_median")),
            "strain_median": r4(r.get("strain_median")), "strain_q25": r4(r.get("strain_q25")),
            "strain_q75": r4(r.get("strain_q75")),
            "strain_mean_UNRELIABLE": r4(r.get("strain_mean")),
            "n_atoms_mean": r4(r.get("atoms_mean")),
        }
    if not sets:
        return
    out["posecheck"] = envelope(
        folder.name, label_of(folder), "posecheck", source=rel(f),
        generated_on=OTHER_BOX, evaluated_on=OTHER_BOX,
        protocol={"pose_receptor_scope": "full", "engine": "posecheck 1.3.1",
                  "definition": d.get("definition")},
        figure_arm=arm, run_root=row.get("root"), pocket_sets=sets,
        notes=["Strain uses the posecheck 1.3.1 definition, ~10x smaller than the VoxBind "
               "paper's -- never place the two side by side.",
               "Aggregate only: the per-molecule dump for this arm stayed on the "
               "reporting box."])


# -------------------------------------------------------------- rigid fragment
# The metric: MMFF-optimise the molecule, cut every rotatable bond, and RMSD each rigid
# fragment against its own optimised self after Kabsch superposition. A rigid fragment has
# one correct shape, so movement means the generated geometry was wrong. Lower is better,
# in Angstrom. See notebook/html/260910/fig-consistency/README.md.
RIGID_FOLDER = {
    "VoxBind + Ours": "VoxBind-Ours",
    "VoxBind": "VoxBind",
    "Reference ligand": "Reference",
    # TargetDiff has no folder in this bundle -> _shared/260910_consistency/
    "TargetDiff": None,
}
RIGID_JSON = {
    "VoxBind-Ours": ("VoxBind-Ours/samples/rigid_fragment_results.json", "density79"),
    "Reference": ("VoxBind-Ours/samples/rigid_fragment_reference.json", "density79"),
}


def read_rigid_csvs():
    by_bin: dict[str, dict[str, list]] = {}
    per_size: dict[str, list] = {}
    f = CONSISTENCY / "rigid_fragment_by_bin.csv"
    if f.exists():
        for row in csv.DictReader(f.open()):
            arm = row["arm"]
            by_bin.setdefault(arm, {}).setdefault(row["binning"], []).append({
                "fragment_size_bin": row["fragment_size_bin"], "n": int(row["n"]),
                **{k: r4(row[k]) for k in ("min", "q25", "median", "mean", "q75", "max")},
            })
    f = CONSISTENCY / "rigid_fragment_per_size.csv"
    if f.exists():
        for row in csv.DictReader(f.open()):
            per_size.setdefault(row["arm"], []).append({
                "fragment_atoms": int(row["fragment_atoms"]), "n": int(row["n"]),
                "rmsd_median": r4(row["rmsd_median"]), "rmsd_mean": r4(row["rmsd_mean"]),
            })
    return by_bin, per_size


def rigid_envelope(folder_name, label, arm, by_bin, per_size):
    src = [rel(CONSISTENCY / "rigid_fragment_by_bin.csv"),
           rel(CONSISTENCY / "rigid_fragment_per_size.csv")]
    sets: dict[str, dict] = {}
    jf = RIGID_JSON.get(folder_name)
    if jf and (T2 / jf[0]).exists():
        d = json.loads((T2 / jf[0]).read_text())["summary"]
        src.insert(0, rel(T2 / jf[0]))
        sets[jf[1]] = {
            "n_pockets": d.get("n_targets"), "n_molecules": d.get("n_mols"),
            "n_fragments": d.get("n_fragments"), "n_mmff_failed": d.get("n_mmff_failed"),
            "rmsd_median": r4(d.get("median_rmsd")), "rmsd_mean": r4(d.get("mean_rmsd")),
            "by_size": [{**b, "median": r4(b.get("median")), "mean": r4(b.get("mean")),
                         "q25": r4(b.get("q25")), "q75": r4(b.get("q75"))}
                        for b in d.get("by_size") or []],
        }
    if arm in per_size:
        d79 = sets.setdefault("density79", {})
        # The figures draw every arm on the 79 density pockets; a CSV-only arm has no
        # molecule count here, but the pocket count is not unknown -- state it.
        d79.setdefault("n_pockets", 79)
        d79.setdefault("n_molecules", None)
        d79.setdefault("n_fragments", sum(x["n"] for x in per_size[arm]))
        d79["per_fragment_size"] = per_size[arm]
    if arm in by_bin:
        sets.setdefault("density79", {})["by_fragment_size_bin"] = by_bin[arm]
    if not sets:
        return None
    return envelope(
        folder_name, label, "rigid_fragment", source=" + ".join(src),
        generated_on=OTHER_BOX, evaluated_on=OTHER_BOX,
        protocol={"metric": "RMSD of each rotatable-bond-free fragment against its own "
                            "MMFF-optimised self, after Kabsch superposition; Angstrom, "
                            "lower is better",
                  "engine": "voxbind/exps/frozenenc_probes/eval_rigid_fragments.py "
                            "(reporting box -- NOT present on svr12)",
                  "pocket_set": "the 79 electron-density pockets",
                  "figure": "notebook/html/260910/fig-consistency/"},
        figure_arm=arm,
        pocket_sets=sets,
        notes=["x is the FRAGMENT's heavy-atom count, not the ligand's -- do not read a "
               "point here against fig-posecheck at the same x.",
               "Scored 2026-08-20 (reference 2026-09-09) on the reporting box. The "
               "evaluator is not on svr12; porting it is what the svr12 arms are waiting on."],
    )


# ============================ generated on this box but never (fully) evaluated
EXPS = ROOT / "voxbind" / "exps"
# Sampling smoke tests and the pre-2026-08 probe runs: not paper arms, deliberately not
# reported as gaps. Matched as a substring of the run path relative to exps/.
NOT_AN_ARM = ("_smoke", "smoke_", "260619_", "260623_", "260624_", "260701_",
              "exp_sig0.9/", "cbgbench", "poc_xray", "reproduction/samples/res_test/")


def scan_exps() -> list[dict]:
    """Every sample tree under voxbind/exps, with what each of the four scorers covered."""
    if not EXPS.is_dir():
        return []
    seen, runs = set(), []
    for t in EXPS.rglob("target_*"):
        if not t.is_dir() or t.parent in seen:
            continue
        seen.add(t.parent)
        run = load_run(t.parent)
        rel_run = str(t.parent.relative_to(EXPS))
        n_t = len(list(t.parent.glob("target_*")))
        if run is None:
            runs.append({"run": rel_run, "n_pockets": n_t, "n_molecules": sdf_total(t.parent),
                         "n_scored": 0, "chem": 0, "vina": 0, "posecheck": 0, "posebusters": 0,
                         "summary": (t.parent / "summary.json").exists(),
                         "skip": any(k in rel_run for k in NOT_AN_ARM)})
            continue
        rows = rows_of(run, sorted(run["targets"]))
        ok = lambda k: sum(1 for x in rows if isinstance(x.get(k), dict) and "error" not in x[k])
        runs.append({
            "run": rel_run, "n_pockets": n_t, "n_molecules": sdf_total(t.parent),
            "n_scored": len(rows),
            "chem": sum(1 for x in rows if x.get("qed") is not None),
            "vina": ok("vina"), "posecheck": ok("posecheck"), "posebusters": ok("posebusters"),
            "summary": (t.parent / "summary.json").exists(),
            "skip": any(k in rel_run for k in NOT_AN_ARM),
        })
    runs.sort(key=lambda x: x["run"])
    return runs


def gaps_of(run: dict) -> list[str]:
    return [k for k in ("chem", "vina", "posecheck", "posebusters") if run[k] == 0]


# A blank column is not always a gap. Where the numbers exist but live somewhere else, say
# where -- otherwise the next person re-docks 7,881 molecules that are already scored.
RUN_NOTE = {
    "_posefull_work/ours_v1_ep350":
        "pose-only tree; its Vina is in VoxBind-Ours/samples/eval_docking_results_full79.json "
        "and the pose rows are staged at VoxBind-Ours/pose_full/ -- nothing to re-run",
    "reproduction/samples/res_test_100":
        "published sigma=0.9 at 100 mols/pocket, staged as VoxBind-vanilla/; the pose pass "
        "never ran",
    "reproduction/samples/res_test_100_posefull":
        "full-receptor pose staging for res_test_100 -- the molecules are hard copies, the "
        "pose pass was never started",
    "reproduction/samples/res_test":
        "the 10 mols/pocket vanilla set, superseded by res_test_100; not staged",
    "260908_fusion_default_cv2_scratch_8gpu/samples/samples_ep350_test79":
        "staged as Fusion-default-cv2-scratch-ep350/; PoseBusters is the only missing pass",
}


# ------------------------------------------------------------- index + status
def write_index(folder: Path, evals: dict, w: Writer):
    idx = {
        "method": label_of(folder), "folder": folder.name,
        "figure_arm": FIGURE_KEY.get(folder.name),
        "evaluations": {},
        "missing": [e for e in EVALS if e not in evals],
        "written_by": TOOL, "written_at": NOW,
    }
    for name in EVALS:
        e = evals.get(name)
        if not e:
            continue
        idx["evaluations"][name] = {
            "file": f"eval/{name}/results.json",
            "source": e["source"], "evaluated_on": e["evaluated_on"],
            "partial": e.get("partial", False),
            "pocket_sets": {k: {"n_pockets": v.get("n_pockets"),
                                "n_molecules": v.get("n_molecules")}
                            for k, v in e["pocket_sets"].items()},
        }
    w(folder / "eval" / "index.json", idx)


def cell(e, sl="all"):
    """`n_pockets/n_molecules`, falling back to fragments, then to pockets alone."""
    if not e:
        return "—"
    ps = e["pocket_sets"]
    s = ps.get(sl) or ps.get("density79") or next(iter(ps.values()))
    n_p, n_m = s.get("n_pockets"), s.get("n_molecules")
    tag = "" if sl in ps else "*"
    if n_m:
        body = f"{n_p}/{n_m}{tag}"
    elif s.get("n_fragments"):
        body = f"{n_p} pockets, {s['n_fragments']} frags{tag}"
    else:
        body = f"{n_p} pockets{tag}"
    return f"**{body}**" if e.get("partial") else body


def write_status(all_evals: dict, runs: list[dict], w: Writer):
    folders = sorted(all_evals)
    status = {
        "task": "task2-drugdesign", "built_on": THIS_BOX, "built_at": NOW,
        "written_by": TOOL,
        "legend": {
            "cell": "n_pockets/n_molecules for the widest pocket set the evaluation covers",
            "*": "the arm covers only the 79 density pockets, so the cell is that slice",
            "partial": "a headline copied from another box -- no molecules here to re-derive",
        },
        "methods": {
            f: {"evaluations": {e: (all_evals[f].get(e) is not None) for e in EVALS},
                "missing": [e for e in EVALS if e not in all_evals[f]]}
            for f in folders
        },
        "generated_not_evaluated": [
            {**r, "gaps": gaps_of(r), "note": RUN_NOTE.get(r["run"])}
            for r in runs if not r["skip"] and gaps_of(r)
        ],
        "excluded_as_smoke_or_legacy": [r["run"] for r in runs if r["skip"]],
    }
    w(T2 / "EVAL_STATUS.json", status)

    L = []
    L.append("# task2-drugdesign — evaluation status\n")
    L.append(f"Built by `{TOOL}` on **{THIS_BOX}** at {NOW}.\n")
    L.append("Every cell is `n_pockets/n_molecules`, read out of the per-evaluation JSON it\n"
             "points at — `<Method>/eval/<evaluation>/results.json`. `—` means the evaluation\n"
             "has never been run for that method. `*` means the arm only ever covered the 79\n"
             "electron-density pockets, so the cell is that slice. **Bold** is a headline\n"
             "imported from another box with no molecules here to re-derive it from.\n")
    L.append("| method | vina docking | sample quality | posebusters | posecheck | rigid fragment |")
    L.append("|---|---|---|---|---|---|")
    for f in folders:
        e = all_evals[f]
        L.append("| `" + f + "` | " + " | ".join(cell(e.get(k)) for k in EVALS) + " |")
    L.append("")

    missing = {f: [e for e in EVALS if e not in all_evals[f]] for f in folders}
    L.append("## What is missing, by evaluation\n")
    for ev in EVALS:
        who = [f for f in folders if ev in missing[f]]
        L.append(f"- **{ev}** — missing for: " + (", ".join(f"`{x}`" for x in who) or "_nothing_"))
    L.append("")

    gaps = [r for r in runs if not r["skip"] and gaps_of(r)]
    L.append("## Generated on this box but never (fully) evaluated\n")
    L.append("Counted by walking `voxbind/exps/**/target_*/metrics.json` and counting the\n"
             "per-molecule rows that actually carry each scorer's block — not by file presence,\n"
             "which is how a PoseCheck pass that errored on every molecule looked successful\n"
             "twice before (see the header of `73_evaluate_samples.sh`).\n")
    if gaps:
        L.append("| run (under `voxbind/exps/`) | pockets | mols | quality | vina | posecheck | posebusters | note |")
        L.append("|---|---|---|---|---|---|---|---|")
        mark = lambda n, tot: "—" if n == 0 else ("ok" if tot and n >= tot else str(n))
        for r in gaps:
            L.append(f"| `{r['run']}` | {r['n_pockets']} | {r['n_molecules']} | "
                     f"{mark(r['chem'], r['n_scored'])} | {mark(r['vina'], r['n_scored'])} | "
                     f"{mark(r['posecheck'], r['n_scored'])} | {mark(r['posebusters'], r['n_scored'])} | "
                     f"{RUN_NOTE.get(r['run'], '')} |")
    else:
        L.append("_Nothing: every non-smoke run under `voxbind/exps/` carries all four scorers._")
    L.append("")
    L.append("Smoke and pre-2026-08 probe runs are excluded on purpose: "
             + ", ".join(f"`{r['run']}`" for r in runs if r["skip"]) + ".\n")
    L.append("## Closing the gaps — not run by this script\n")
    L.append("```bash")
    L.append("cd voxbind")
    L.append("# PoseBusters only, on a run that already has vina + posecheck:")
    L.append("SAMPLE_DIR=exps/260908_fusion_default_cv2_scratch_8gpu/samples/samples_ep350_test79 \\")
    L.append("  POSE=posebusters SKIP_EXISTING=1 bash scripts/73_evaluate_samples.sh")
    L.append("# full pose pass (PoseCheck + PoseBusters) on the vanilla 100/pocket set:")
    L.append("SAMPLE_DIR=exps/reproduction/samples/res_test_100 \\")
    L.append("  POSE=all SKIP_EXISTING=1 bash scripts/73_evaluate_samples.sh")
    L.append("# then re-collect:")
    L.append("cd .. && python3 " + TOOL)
    L.append("```")
    L.append("")
    L.append("Rigid-fragment consistency has no runner on this box: its evaluator is\n"
             "`voxbind/exps/frozenenc_probes/eval_rigid_fragments.py` on the reporting box\n"
             "(`notebook/html/260910/fig-consistency/README.md`). Porting it is a prerequisite\n"
             "for every `—` in that column.\n")
    if not w.dry:
        (T2 / "EVAL_STATUS.md").write_text("\n".join(L))
    return "\n".join(L)


# ------------------------------------------------------------------------ check
# Independent re-derivations must reproduce what the bundle already reports. Keys are
# (folder, path into metrics.json, path into the derived `all`/`density79` slice).
CHECKS = [
    ("VoxBind-base-ep350", "eval.vina_dock_mean", "vina_docking.all.per_pocket_mean.vina_dock"),
    ("VoxBind-base-ep350", "eval.pooled.dock.mean", "vina_docking.all.vina_dock.mean"),
    ("VoxBind-base-ep350", "eval.pb_valid_rate", "posebusters.all.pb_valid_rate"),
    ("VoxBind-base-ep350", "eval.strain_median", "posecheck.all.strain_median"),
    ("VoxBind-base-ep350", "eval.clashes_mean", "posecheck.all.clashes_mean"),
    ("VoxBind-base-ep350", "eval.qed_mean", "sample_quality.all.qed_mean_over_pockets"),
    ("VoxBind-base-ep350", "eval.diversity", "sample_quality.all.diversity"),
    ("VoxBind-base-ep350", "posebusters_260910.density79.pb_valid_pct",
     "posebusters.density79.pb_valid_rate", 100.0),
    ("Fusion-v4-cdgv2-warm-ep100", "eval.vina_dock_mean", "vina_docking.all.per_pocket_mean.vina_dock"),
    ("Fusion-v4-cdgv2-warm-ep100", "eval.pb_valid_rate", "posebusters.all.pb_valid_rate"),
    ("Fusion-v4-cdgv2-warm-ep100", "eval.strain_median", "posecheck.all.strain_median"),
    ("Fusion-v4-cdgv2-scratch-ep350", "eval.vina_dock_mean", "vina_docking.all.per_pocket_mean.vina_dock"),
    ("Fusion-v4-cdgv2-scratch-ep350", "eval.sa_mean", "sample_quality.all.sa_mean_over_pockets"),
    ("Fusion-v4-cv2-scratch-ep350", "eval.pooled.dock.mean", "vina_docking.all.vina_dock.mean"),
    ("Fusion-v4-cv2-scratch-ep350", "eval.pb_valid_rate", "posebusters.all.pb_valid_rate"),
    ("Reference", "vina.density79.vina_dock_mean", "vina_docking.density79.vina_dock.mean"),
    ("VoxBind-Ours", "summary.vina_dock", "vina_docking.density79.vina_dock.mean"),
]


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def run_checks(all_evals) -> int:
    bad = 0
    for folder, mpath, dpath, *scale in CHECKS:
        want = dig(head(T2 / folder), mpath)
        got = dig({k: v["pocket_sets"] for k, v in all_evals.get(folder, {}).items()}, dpath)
        if scale:
            got = None if got is None else got * scale[0]
        if want is None or got is None:
            print(f"  MISS  {folder:<32} {mpath} -> {dpath}  (want={want} got={got})")
            bad += 1
        # The derived rates are rounded to 4 dp as fractions, so a percentage comparison
        # has to carry the same factor or 0.619 vs 61.89873 reads as a mismatch.
        elif abs(float(want) - float(got)) > 1e-3 * (scale[0] if scale else 1):
            print(f"  FAIL  {folder:<32} {mpath}={want}  vs derived {dpath}={got}")
            bad += 1
        else:
            print(f"  ok    {folder:<32} {mpath} = {want}")
    return bad


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare against each metrics.json; write nothing")
    ap.add_argument("--dry-run", action="store_true", help="report what would change only")
    a = ap.parse_args()
    w = Writer(dry=a.dry_run or a.check)

    by_bin, per_size = read_rigid_csvs()
    folders = sorted(p for p in T2.iterdir()
                     if p.is_dir() and not p.name.startswith((".", "_")))
    all_evals: dict[str, dict] = {}

    for folder in folders:
        evals: dict[str, dict] = {}
        if folder.name in BASELINE_ROW:
            collect_baseline(folder, evals)
        elif folder.name == "Reference":
            collect_reference(folder, evals)
        elif folder.name in IMPORTED and not (folder / "samples" / "target_00").is_dir():
            collect_imported(folder, evals)
        else:
            collect_native(folder, evals)

        collect_shared_posecheck(folder, evals)

        arm = next((k for k, v in RIGID_FOLDER.items() if v == folder.name), None)
        if arm:
            e = rigid_envelope(folder.name, label_of(folder), arm, by_bin, per_size)
            if e:
                evals["rigid_fragment"] = e

        for name, e in evals.items():
            w(folder / "eval" / name / "results.json", e)
        write_index(folder, evals, w)
        all_evals[folder.name] = evals
        cov = ", ".join(n for n in EVALS if n in evals) or "nothing"
        print(f"{folder.name:<34} {cov}")

    # TargetDiff has no folder of its own; its rigid-fragment row still belongs in the bundle.
    e = rigid_envelope("_shared/260910_consistency", "TargetDiff", "TargetDiff", by_bin, per_size)
    if e:
        w(SHARED / "260910_consistency" / "rigid_fragment_TargetDiff.json", e)

    runs = scan_exps()
    write_status(all_evals, runs, w)

    print(f"\n{len(w.written)} file(s) {'would be ' if w.dry else ''}written, "
          f"{len(w.unchanged)} unchanged")
    if a.dry_run:
        for p in w.written:
            print("   [dry]", rel(p))

    if a.check:
        print("\n-- check: derived vs. the bundle's own metrics.json --")
        bad = run_checks(all_evals)
        print(f"-- {len(CHECKS) - bad}/{len(CHECKS)} agree")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
