"""build_results_baseline_protocol.py — assemble the baseline-protocol tables for
results_drug_design.html, and write the LaTeX table 260827/table_drug_design.tex from them.

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

The LaTeX table is generated here rather than typed, so it cannot drift from the numbers
the HTML reads. Its row order and row names are the canonical ones the 260910 figure code
follows (figures/draw.py ORDER); keep them when editing TEX_ROWS.

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_results_baseline_protocol.py
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
E = "/home1/irteam/VoxBind/voxbind/exps"
TEX_OUT = os.path.join(HERE, "..", "260827", "table_drug_design.tex")

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
# baseline_vina.json carries only the high-affinity MEAN; the median over pockets is in the
# svr12 summary the results bundle stages, keyed by the generator's own name.
BASE_SUMMARY = os.path.join(REPO, "results", "task2-drugdesign", "_shared", "baselines_eval",
                            "summary_density79.json")
BASE_SUMMARY_KEY = {"DecompDiff": "DecompDiff_ref_prior"}
# TargetDiff's compute_high_affinity (notebooks/summary.ipynb) skips pockets with fewer than
# 50 docked molecules; the baselines' high-affinity columns were computed the same way.
HA_MIN_DOCKED = 50


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
      high_aff   -- share of a pocket's molecules with dock <= that pocket's own reference;
                    per-pocket by construction, then mean / median over the pockets with
                    >= HA_MIN_DOCKED docked molecules (TargetDiff; the baselines match)
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
    ha = [e["high_affinity"] for e in per if e.get("high_affinity") is not None
          and sum(m.get("vina_dock") is not None
                  for m in (e.get("per_mol") or [])) >= HA_MIN_DOCKED]
    out["high_aff"] = round(100 * st.mean(ha), 1) if ha else None
    out["high_aff_med"] = round(100 * st.median(ha), 1) if ha else None
    out["high_aff_pockets"] = len(ha)
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


def add_baseline_ha_median(base):
    """Copy the per-pocket high-affinity median onto each baseline row, checking the mean."""
    if not os.path.exists(BASE_SUMMARY):
        print(f"WARNING: {BASE_SUMMARY} missing -- baseline high-affinity medians left empty")
        return
    summary = json.load(open(BASE_SUMMARY))
    for name in BASE_ORDER:
        key = BASE_SUMMARY_KEY.get(name, name)
        s = next((r for r in summary if r.get("baseline") == key), None)
        if s is None:
            print(f"WARNING: no {key} row in {os.path.basename(BASE_SUMMARY)}")
            continue
        if base[name]["high_aff"] is not None and \
                abs(round(s["high_affinity"], 1) - base[name]["high_aff"]) > 0.05:
            raise SystemExit(f"{name}: high-affinity mean {s['high_affinity']:.2f} in "
                             f"{os.path.basename(BASE_SUMMARY)} != {base[name]['high_aff']} "
                             f"in baseline_vina.json -- the two sources are not the same run")
        base[name]["high_aff_med"] = round(s["high_affinity_median"], 1)
        base[name]["high_aff_pockets"] = s.get("high_affinity_pockets")


rows, missing = {}, []
for name, root in OURS.items():
    r = ours_row(name, root)
    if r is None:
        missing.append(name)
    else:
        rows[name] = r
add_baseline_ha_median(BASE)

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
    hdr = f"  {'Model':16s} {'pockets':>8s} {'mols':>6s} {'Score':>14s} {'Min':>14s} {'Dock':>14s} {'HA%':>12s} {'QED':>6s} {'SA':>6s}"
    print(hdr)
    for name in ["Reference"] + BASE_ORDER:
        b = BASE[name]
        print(f"  {name:16s} {b['pockets']:>8s} {str(b['mols']):>6s} "
              f"{str(b['score_mean']):>6s}/{str(b['score_med']):<7s} "
              f"{str(b['min_mean']):>6s}/{str(b['min_med']):<7s} "
              f"{str(b['dock_mean']):>6s}/{str(b['dock_med']):<7s} "
              f"{str(b['high_aff']):>5s}/{str(b.get('high_aff_med')):<6s} "
              f"{str(b['qed_mean']):>6s} {str(b['sa_mean']):>6s}")
    for name, r in rows.items():
        print(f"  {name:16s} {r['pockets']:>8s} {r['mols']:>6d} "
              f"{str(r['score_mean']):>6s}/{str(r['score_med']):<7s} "
              f"{str(r['min_mean']):>6s}/{str(r['min_med']):<7s} "
              f"{str(r['dock_mean']):>6s}/{str(r['dock_med']):<7s} "
              f"{str(r['high_aff']):>5s}/{str(r['high_aff_med']):<6s} "
              f"{str(r['qed_mean']):>6s} {str(r['sa_mean']):>6s}")

json.dump({"baseline_79": BASE, "ours": rows, "missing": missing},
          open(f"{HERE}/merged_baseline_protocol.json", "w"), indent=1)
print(f"\nwrote merged_baseline_protocol.json")


# ------------------------------------------------------------------ LaTeX table
# (label, source, key): source "base" reads BASE, "ours" reads rows, None has no numbers
# under this protocol yet. None as a whole entry is a \midrule.
TEX_ROWS = [
    ("Reference", "base", "Reference"),
    None,
    ("AR", "base", "AR"),
    ("Pocket2Mol", "base", "Pocket2Mol"),
    ("DiffSBDD", "base", "DiffSBDD"),
    ("TargetDiff", "ours", "TargetDiff"),
    (r"DecompDiff\textsubscript{\scriptsize ref-informed}", "base", "DecompDiff"),
    (r"VoxBind\textsubscript{\scriptsize $\sigma$=0.9}", "ours", "VoxBind σ=0.9"),
    ("FuncBind", "base", "FuncBind"),
    None,
    (r"\textbf{VoxBind + C}", None, None),
    (r"\textbf{VoxBind + CDG}", "ours", "Ours · v1"),
]
# (field, decimals, better): each field is bolded separately where it is best among the
# generated-molecule rows (the reference ligand competes for nothing).
TEX_FIELDS = [("score_mean", 2, min), ("score_med", 2, min), ("min_mean", 2, min),
              ("min_med", 2, min), ("dock_mean", 2, min), ("dock_med", 2, min),
              ("high_aff", 1, max), ("high_aff_med", 1, max), ("qed_mean", 2, max),
              ("sa_mean", 2, max), ("div_mean", 2, max), ("atoms", 1, None)]


def tex_table():
    data = {}
    for entry in TEX_ROWS:
        if entry and entry[1]:
            src = BASE if entry[1] == "base" else rows
            if entry[2] not in src:
                raise SystemExit(f"LaTeX row {entry[0]!r}: no numbers for {entry[2]!r}")
            data[entry[2]] = src[entry[2]]
    best = {}
    for field, nd, better in TEX_FIELDS:
        vals = [round(r[field], nd) for k, r in data.items()
                if k != "Reference" and r.get(field) is not None]
        if better and vals:
            best[field] = better(vals)

    def cell(r, field):
        nd = dict((f, d) for f, d, _ in TEX_FIELDS)[field]
        v = r.get(field)
        if v is None:
            return "---"
        s = f"{v:.{nd}f}"
        return rf"\textbf{{{s}}}" if field in best and round(v, nd) == best[field] and \
            r is not data.get("Reference") else s

    lines = []
    for entry in TEX_ROWS:
        if entry is None:
            lines.append(r"            \midrule")
            continue
        label, src, key = entry
        if src is None:
            lines.append(rf"            {label} & \multicolumn{{8}}{{c}}{{?}} \\")
            continue
        r = data[key]
        pair = lambda a, b: f"{cell(r, a)} / {cell(r, b)}"
        lines.append(
            f"            {label} & {pair('score_mean', 'score_med')} & "
            f"{pair('min_mean', 'min_med')} & {pair('dock_mean', 'dock_med')} & "
            f"{pair('high_aff', 'high_aff_med')} & {cell(r, 'qed_mean')} & "
            f"{cell(r, 'sa_mean')} & {cell(r, 'div_mean')} & {cell(r, 'atoms')} \\\\")
    body = "\n".join(lines)
    return rf"""% GENERATED by notebook/html/260903/build_results_baseline_protocol.py -- edit that, not this.
\begin{{table}}[!t]
    \centering
    \caption{{
        \textbf{{Structure-based drug design results}} on 79 CrossDocked2020 benchmark test pockets with
        experimental density data available. Every row is scored with one protocol: AutoDock Vina 1.2.2
        docking into the whole receptor, exhaustiveness 32. Vina cells report mean / median over all
        generated molecules pooled across pockets; high-affinity cells report mean / median over pockets
        of the share of molecules docking at least as well as the pocket's reference ligand (pockets with
        at least {HA_MIN_DOCKED} docked molecules). QED, SA and \# atoms are means over molecules; diversity is the
        mean over pockets. Arrows indicate the preferred direction.
    }}
    \label{{tab:result-drug-design}}
    \resizebox{{.98\textwidth}}{{!}}{{%
        \begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
            \toprule
            \multirow{{2}}{{*}}{{\textbf{{Method}}}} & \multicolumn{{4}}{{c}}{{\textbf{{Vina evaluation}}}} & \multicolumn{{4}}{{c}}{{\textbf{{Sample quality}}}} \\
            \cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
            & \textbf{{Score $\downarrow$}} & \textbf{{Min $\downarrow$}} & \textbf{{Dock $\downarrow$}} & \textbf{{High aff. $\uparrow$}} & \textbf{{QED $\uparrow$}} & \textbf{{SA $\uparrow$}} & \textbf{{Diversity $\uparrow$}} & \textbf{{\# Atoms / Mol}} \\
            \midrule
{body}
            \bottomrule
        \end{{tabular}}
    }}
\end{{table}}
"""


if not missing:
    with open(TEX_OUT, "w") as fh:
        fh.write(tex_table())
    print(f"wrote {os.path.relpath(TEX_OUT, REPO)}")
else:
    print(f"LaTeX table NOT written: {', '.join(missing)} still missing")
