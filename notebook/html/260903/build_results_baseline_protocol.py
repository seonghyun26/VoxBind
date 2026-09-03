"""build_results_baseline_protocol.py — assemble the baseline-protocol tables for
results_drug_design.html.

WHY THIS EXISTS. results_drug_design.html was written on our own evaluation protocol
(pocket10 crop, exhaustiveness 16, 78 pockets). 260903/baseline.html ran five published
baselines on a different one (whole *_rec.pdb receptor, exhaustiveness 32, 79 pockets,
Vina 1.2.2). The two cannot share a table: the same crystal reference ligand scores
-7.18 our way and -7.32 theirs, which is a third of the gap we are trying to report.

So vanilla VoxBind and Ours v1 were re-docked under the baseline protocol
(scripts/73_dock_baseline_protocol_79.sh -> eval_docking_results_full79.json) and this
script merges the two sources into one table, with the reference ligand row acting as
the calibration check: if our re-run's reference does not land on baseline.html's -7.32,
the protocols still differ somewhere and the merge is not yet safe.

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_results_baseline_protocol.py
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"

OURS = {
    "VoxBind σ=0.9": f"{E}/_vanilla_ep923/samples/full_eval_ep923",
    "Ours · v1":     f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
    # TargetDiff is not one of baseline.html's five: it was sampled and scored on another
    # box, but its eval json was copied here in the same format, over the same 79 pockets
    # at receptor_scope=full, so it aggregates through exactly the same code path.
    "TargetDiff":    "/home1/irteam/base_drug/eval/targetdiff",
}
BASE = json.load(open(f"{HERE}/baseline_vina.json"))["79"]
# per-molecule QED/SA (pooled) and per-pocket Diversity for our runs, from
# recompute_qed_sa.py -- rerun that first if a new run is added here.
QED_SA_DIV = json.load(open(f"{HERE}/ours_qed_sa_div.json"))
# baseline.html's own row order, most-cited first; ours are appended after
BASE_ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind"]


def ours_row(name, root):
    """Aggregate one of our runs the way TargetDiff's evaluate_diffusion.py does:
    BOTH mean and median over every generated molecule, pooled across pockets.

    TargetDiff builds one flat `results` list with one entry per molecule and calls
    np.mean/np.median on it, and DecompDiff/DiffSBDD follow suit, so this is the
    published convention. An earlier version of this function took the mean over the
    79 *pocket* values instead, which is a different unit; with 97-100 molecules in
    every pocket the two agree to <=0.004 kcal/mol, but the labels then disagreed with
    what the numbers were.

    Vina Score/Min/Dock and the heavy-atom count pool straight out of per_mol. QED and SA
    are stored only as a per-pocket mean, so their pooled pair is read from
    ours_qed_sa_div.json, which recompute_qed_sa.py derives per molecule from the stored
    SMILES (checked against the stored per-pocket means, drift 0.0000).

    Two columns stay per-pocket because that is their definition, not a data gap:
      diversity  -- mean pairwise Tanimoto WITHIN a pocket, so its mean and median are
                    over the 79 pocket values; there is no per-molecule diversity to pool
      high_aff   -- share of a pocket's molecules beating that pocket's own reference;
                    per-pocket by construction, then averaged (the baselines match)
    """
    f = f"{root}/eval_docking_results_full79.json"
    if not os.path.exists(f):
        return None
    per = json.load(open(f))["per_target"]
    out = {"pockets": f"{len(per)}/79"}
    nmol = 0
    for key, field in (("score", "vina_score"), ("min", "vina_min"), ("dock", "vina_dock")):
        pooled = [m[field] for e in per for m in (e.get("per_mol") or [])
                  if m.get(field) is not None]
        out[f"{key}_mean"] = round(st.mean(pooled), 2) if pooled else None
        out[f"{key}_med"] = round(st.median(pooled), 2) if pooled else None
        nmol = max(nmol, len(pooled))
    ha = [e["high_affinity"] for e in per if e.get("high_affinity") is not None]
    out["high_aff"] = round(100 * st.mean(ha), 1) if ha else None
    # QED/SA are stored only as a per-pocket mean, so the pooled median comes from
    # recompute_qed_sa.py, which re-derives them per molecule from the stored SMILES
    # (verified to reproduce the stored per-pocket means exactly, drift 0.0000).
    # Diversity is mean pairwise Tanimoto WITHIN a pocket, so its mean and median are
    # over the 79 pocket values -- there is no per-molecule diversity to pool.
    extra = QED_SA_DIV[name]
    for key in ("qed", "sa", "div"):
        out[f"{key}_mean"] = extra[f"{key}_mean"]
        out[f"{key}_med"] = extra[f"{key}_med"]
    na = [m["n_atoms"] for e in per for m in (e.get("per_mol") or [])
          if m.get("n_atoms") is not None]                 # per molecule, like the Vina columns
    out["atoms"] = round(st.mean(na), 1) if na else None
    out["mols"] = nmol
    # the calibration row: our own reference-ligand dock under this protocol
    rd = [e["ref_vina_dock"] for e in per if e.get("ref_vina_dock") is not None]
    out["_ref_dock_mean"] = round(st.mean(rd), 2) if rd else None
    return out


rows, missing = {}, []
for name, root in OURS.items():
    r = ours_row(name, root)
    if r is None:
        missing.append(name)
    else:
        rows[name] = r

print(f"baseline rows available : {len(BASE)} ({', '.join(BASE_ORDER)})")
print(f"our rows available      : {len(rows)}"
      + (f"   STILL RUNNING: {', '.join(missing)}" if missing else ""))

if rows:
    print("\n=== calibration: reference-ligand Vina Dock under this protocol ===")
    print(f"  baseline.html @79        : {BASE['Reference']['dock_mean']}")
    for name, r in rows.items():
        d = r["_ref_dock_mean"]
        delta = None if d is None else round(d - BASE["Reference"]["dock_mean"], 3)
        flag = "" if delta is None or abs(delta) < 0.15 else "   <-- protocols still differ"
        print(f"  our re-run via {name:16s}: {d}   Δ {delta}{flag}")

    print("\n=== merged table preview (79 pockets, full receptor, exh 32) ===")
    hdr = f"  {'Model':16s} {'pockets':>8s} {'mols':>6s} {'Score':>14s} {'Min':>14s} {'Dock':>14s} {'HA%':>6s} {'QED':>6s} {'SA':>6s}"
    print(hdr)
    for name in ["Reference"] + BASE_ORDER:
        b = BASE[name]
        print(f"  {name:16s} {b['pockets']:>8s} {str(b['mols']):>6s} "
              f"{str(b['score_mean']):>6s}/{str(b['score_med']):<7s} "
              f"{str(b['min_mean']):>6s}/{str(b['min_med']):<7s} "
              f"{str(b['dock_mean']):>6s}/{str(b['dock_med']):<7s} "
              f"{str(b['high_aff']):>6s} {str(b['qed_mean']):>6s} {str(b['sa_mean']):>6s}")
    for name, r in rows.items():
        print(f"  {name:16s} {r['pockets']:>8s} {r['mols']:>6d} "
              f"{str(r['score_mean']):>6s}/{str(r['score_med']):<7s} "
              f"{str(r['min_mean']):>6s}/{str(r['min_med']):<7s} "
              f"{str(r['dock_mean']):>6s}/{str(r['dock_med']):<7s} "
              f"{str(r['high_aff']):>6s} {str(r['qed_mean']):>6s} {str(r['sa_mean']):>6s}")

json.dump({"baseline_79": BASE, "ours": rows, "missing": missing},
          open(f"{HERE}/merged_baseline_protocol.json", "w"), indent=1)
print(f"\nwrote merged_baseline_protocol.json")
