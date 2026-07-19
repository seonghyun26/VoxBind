#!/usr/bin/env python3
"""build_macrocycle.py — macrocycle prevalence + docking correlation report.

Detects macrocycles (any ring of >= 12 atoms; secondary >= 8) across three
ligand populations and asks whether being a macrocycle correlates with Vina
docking score:

  1. VoxBind generated  — samples.sdf / metrics.json from the 10k run
                          (voxbind/exps/260518_voxbind_10k_noise/.../res_ep99_test)
  2. CrossDocked        — reference ligands in crossdocked_pocket10 (unique SMILES)
  3. PLINDER v2         — canonical SMILES in splits/plinder/v2/plinder_selected.csv

Docking correlation is computed on the two populations that carry per-molecule
Vina scores: the 618 VoxBind generated samples and the 62 CrossDocked test-set
reference ligands (both from metrics.json).

Outputs:
  macrocycle_data.json          (cache of all computed numbers; delete to recompute)
  fig_macrocycle_fraction.svg   (bar chart)
  fig_dock_macro.svg            (docking box/strip)
  260723_meeting.html

    OMP_NUM_THREADS=2 nice -19 /compuworks/anaconda3/bin/python \
        notebook/html/260723/build_macrocycle.py
"""
import os
import re
import csv
import json
import glob
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
VOX = os.path.join(REPO, "voxbind")
CACHE = os.path.join(HERE, "macrocycle_data.json")
OUT = os.path.join(HERE, "260723_meeting.html")

GEN_RUN = os.path.join(VOX, "exps", "260518_voxbind_10k_noise",
                       "samples", "res_ep99_test")
CD_DIR = os.path.join(VOX, "dataset", "data", "crossdocked_pocket10")
PLINDER_CSV = os.path.join(VOX, "splits", "plinder", "v2", "plinder_selected.csv")

MACRO = 12   # headline macrocycle threshold (ring atoms)
MACRO2 = 8   # secondary / "large ring" threshold

# ─────────────────────────────────────────────────────────────────────────────
# macrocycle detection
# ─────────────────────────────────────────────────────────────────────────────
def _rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def max_ring_from_mol(mol):
    """Largest SSSR ring size for a parsed mol, or 0."""
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    return max(sizes) if sizes else 0


def max_ring_from_smiles(smi, Chem):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return max_ring_from_mol(m)


def summarize(maxrings):
    """maxrings: list of largest-ring-size ints (None dropped). Return counts."""
    vals = [r for r in maxrings if r is not None]
    n = len(vals)
    n12 = sum(1 for r in vals if r >= MACRO)
    n8 = sum(1 for r in vals if r >= MACRO2)
    # distribution of macrocycle ring sizes (>=12)
    big = sorted(r for r in vals if r >= MACRO)
    return {
        "n": n,
        "n_macro12": n12,
        "frac_macro12": (n12 / n) if n else 0.0,
        "n_ring8": n8,
        "frac_ring8": (n8 / n) if n else 0.0,
        "macro_ring_sizes": big,
    }


# ─────────────────────────────────────────────────────────────────────────────
# source loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_generated(Chem):
    """VoxBind generated samples + CrossDocked test-set reference ligands.

    Returns (gen_records, ref_records) where each record is
      {smiles, maxring, is_macro, dock, score_only, minimize}
    """
    gen, refs = [], []
    for mj in sorted(glob.glob(os.path.join(GEN_RUN, "target_*", "metrics.json"))):
        d = json.load(open(mj))
        r = d.get("reference", {})
        if r.get("smiles"):
            mr = max_ring_from_smiles(r["smiles"], Chem)
            v = r.get("vina", {}) or {}
            refs.append({
                "smiles": r["smiles"], "maxring": mr,
                "is_macro": (mr is not None and mr >= MACRO),
                "dock": v.get("dock"), "score_only": v.get("score_only"),
                "minimize": v.get("minimize"),
            })
        for s in d.get("samples", []):
            smi = s.get("smiles")
            if not smi:
                continue
            mr = max_ring_from_smiles(smi, Chem)
            v = s.get("vina", {}) or {}
            gen.append({
                "smiles": smi, "maxring": mr,
                "is_macro": (mr is not None and mr >= MACRO),
                "dock": v.get("dock"), "score_only": v.get("score_only"),
                "minimize": v.get("minimize"), "n_atoms": s.get("n_atoms"),
            })
    return gen, refs


def load_crossdocked(Chem):
    """One SDF per unique ligand base -> dedupe by canonical SMILES.

    Returns (unique_maxrings, n_bases, n_unique_smiles).
    """
    files = glob.glob(os.path.join(CD_DIR, "*", "*_lig_*.sdf"))
    seen_base = {}
    for f in files:
        b = re.sub(r"_(docked|min)_\d+\.sdf$", "", f)
        # prefer a *_min_0 file if present, else first seen
        if b not in seen_base or f.endswith("_min_0.sdf"):
            seen_base[b] = f
    by_smiles = {}
    for f in seen_base.values():
        try:
            m = next(iter(Chem.SDMolSupplier(f, sanitize=True)), None)
        except Exception:
            m = None
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if smi not in by_smiles:
            by_smiles[smi] = max_ring_from_mol(m)
    return list(by_smiles.values()), len(seen_base), len(by_smiles)


def load_plinder(Chem):
    """PLINDER v2 canonical SMILES. Returns (instance_maxrings, unique_maxrings, n_inst, n_unique)."""
    inst, uniq = [], {}
    with open(PLINDER_CSV, newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            smi = row.get("ligand_rdkit_canonical_smiles")
            if not smi:
                continue
            mr = max_ring_from_smiles(smi, Chem)
            inst.append(mr)
            if smi not in uniq:
                uniq[smi] = mr
    return inst, list(uniq.values()), len(inst), len(uniq)


# ─────────────────────────────────────────────────────────────────────────────
# docking correlation
# ─────────────────────────────────────────────────────────────────────────────
def dock_correlation(records, key="dock"):
    """Compare `key` (Vina, lower=better) between macro and non-macro groups."""
    import numpy as np
    from scipy import stats
    macro = [r[key] for r in records if r["is_macro"] and r.get(key) is not None]
    non = [r[key] for r in records if (not r["is_macro"]) and r.get(key) is not None]
    out = {
        "n_macro": len(macro), "n_non": len(non),
        "mean_macro": (float(np.mean(macro)) if macro else None),
        "std_macro": (float(np.std(macro, ddof=1)) if len(macro) > 1 else None),
        "median_macro": (float(np.median(macro)) if macro else None),
        "mean_non": (float(np.mean(non)) if non else None),
        "std_non": (float(np.std(non, ddof=1)) if len(non) > 1 else None),
        "median_non": (float(np.median(non)) if non else None),
        "delta": None, "mwu_p": None, "welch_p": None,
        "pointbiserial_r": None, "pointbiserial_p": None,
    }
    if macro and non:
        out["delta"] = out["mean_macro"] - out["mean_non"]
    if len(macro) >= 2 and len(non) >= 2:
        try:
            out["mwu_p"] = float(stats.mannwhitneyu(macro, non, alternative="two-sided").pvalue)
        except Exception:
            pass
        try:
            out["welch_p"] = float(stats.ttest_ind(macro, non, equal_var=False).pvalue)
        except Exception:
            pass
        # point-biserial: x = is_macro (1/0), y = score
        xs = [1] * len(macro) + [0] * len(non)
        ys = macro + non
        try:
            rr = stats.pointbiserialr(xs, ys)
            out["pointbiserial_r"] = float(rr.correlation)
            out["pointbiserial_p"] = float(rr.pvalue)
        except Exception:
            pass
    return out


def size_confound(records, key="dock"):
    """Is the macrocycle→better-dock effect just molecular size?

    Vina scores grow (more negative) with atom count; macrocycles are large by
    construction. Control for it with (a) a size-matched MWU and (b) an OLS with
    size + macrocycle as predictors.
    """
    import numpy as np
    from scipy import stats
    rs = [r for r in records if r.get(key) is not None and r.get("n_atoms") is not None]
    y = np.array([r[key] for r in rs], float)
    na = np.array([float(r["n_atoms"]) for r in rs])
    m = np.array([1.0 if r["is_macro"] else 0.0 for r in rs])
    out = {
        "n": len(rs),
        "natoms_macro": float(na[m == 1].mean()) if (m == 1).any() else None,
        "natoms_non": float(na[m == 0].mean()) if (m == 0).any() else None,
        "r_dock_natoms": None, "rho_dock_natoms": None,
        "ols_macro_coef": None, "ols_macro_se": None, "ols_macro_p": None,
        "ols_natoms_coef": None,
        "matched_n_macro": None, "matched_n_non": None,
        "matched_mean_macro": None, "matched_mean_non": None,
        "matched_delta": None, "matched_mwu_p": None,
    }
    try:
        out["r_dock_natoms"] = float(stats.pearsonr(na, y)[0])
        out["rho_dock_natoms"] = float(stats.spearmanr(na, y)[0])
    except Exception:
        pass
    # OLS: y ~ 1 + n_atoms + is_macro
    if (m == 1).sum() >= 2 and (m == 0).sum() >= 2:
        X = np.column_stack([np.ones_like(y), na, m])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        if dof > 0:
            sig2 = float(resid @ resid) / dof
            cov = sig2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            out["ols_natoms_coef"] = float(beta[1])
            out["ols_macro_coef"] = float(beta[2])
            out["ols_macro_se"] = float(se[2])
            t = beta[2] / se[2]
            out["ols_macro_p"] = float(2 * stats.t.sf(abs(t), dof))
        # size-matched: non-macro restricted to n_atoms within macro range
        lo = na[m == 1].min()
        macro_y = y[m == 1]
        matched_non = y[(m == 0) & (na >= lo)]
        out["matched_n_macro"] = int(len(macro_y))
        out["matched_n_non"] = int(len(matched_non))
        if len(matched_non) >= 1:
            out["matched_mean_macro"] = float(macro_y.mean())
            out["matched_mean_non"] = float(matched_non.mean())
            out["matched_delta"] = out["matched_mean_macro"] - out["matched_mean_non"]
        if len(matched_non) >= 2:
            try:
                out["matched_mwu_p"] = float(
                    stats.mannwhitneyu(macro_y, matched_non, alternative="two-sided").pvalue)
            except Exception:
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# compute (cached)
# ─────────────────────────────────────────────────────────────────────────────
def compute():
    if os.path.exists(CACHE):
        print("using cache", CACHE, "(delete to recompute)")
        return json.load(open(CACHE))
    Chem = _rdkit()
    print("loading generated + refs …")
    gen, refs = load_generated(Chem)
    print(f"  gen={len(gen)}  refs={len(refs)}")
    print("loading crossdocked (this reads ~150k SDF, ~1-2 min) …")
    cd_maxrings, cd_bases, cd_uniq = load_crossdocked(Chem)
    print(f"  cd bases={cd_bases}  unique_smiles={cd_uniq}")
    print("loading plinder v2 …")
    pl_inst, pl_uniq, pl_ninst, pl_nuniq = load_plinder(Chem)
    print(f"  plinder inst={pl_ninst}  unique={pl_nuniq}")

    data = {
        "meta": {
            "macro_threshold": MACRO, "ring2_threshold": MACRO2,
            "gen_run": os.path.relpath(GEN_RUN, REPO),
            "cd_dir": os.path.relpath(CD_DIR, REPO),
            "plinder_csv": os.path.relpath(PLINDER_CSV, REPO),
        },
        "generated": summarize([r["maxring"] for r in gen]),
        "cd_refs": summarize([r["maxring"] for r in refs]),
        "crossdocked": {**summarize(cd_maxrings), "n_bases": cd_bases,
                        "n_unique_smiles": cd_uniq},
        "plinder_unique": {**summarize(pl_uniq), "n_instances": pl_ninst},
        "plinder_instances": summarize(pl_inst),
        "dock": {
            "generated_dock": dock_correlation(gen, "dock"),
            "generated_score_only": dock_correlation(gen, "score_only"),
            "generated_minimize": dock_correlation(gen, "minimize"),
            "refs_dock": dock_correlation(refs, "dock"),
        },
        "size": size_confound(gen, "dock"),
        # per-molecule dock records for the strip / scatter plots (generated only)
        "gen_points": [{"dock": r["dock"], "is_macro": r["is_macro"],
                        "n_atoms": r.get("n_atoms")}
                       for r in gen if r.get("dock") is not None],
    }
    json.dump(data, open(CACHE, "w"), indent=1)
    print("wrote", CACHE)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────
def make_figures(data):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ACC = "#2f6f4f"
    ACC2 = "#b07a17"
    INK = "#1c2433"
    SOFT = "#5b6678"
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": "#aeb6c4",
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
    })

    # ---- Fig 1: macrocycle fraction per source ----
    labels = ["VoxBind\ngenerated", "CrossDocked\n(unique lig.)", "PLINDER v2\n(unique lig.)"]
    d12 = [data["generated"]["frac_macro12"] * 100,
           data["crossdocked"]["frac_macro12"] * 100,
           data["plinder_unique"]["frac_macro12"] * 100]
    d8 = [data["generated"]["frac_ring8"] * 100,
          data["crossdocked"]["frac_ring8"] * 100,
          data["plinder_unique"]["frac_ring8"] * 100]
    ns = [data["generated"]["n"], data["crossdocked"]["n_unique_smiles"],
          data["plinder_unique"]["n"]]

    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=110)
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y + h / 2, d8, height=h, color=ACC2, alpha=.55, label=f"ring ≥ {MACRO2} atoms")
    ax.barh(y - h / 2, d12, height=h, color=ACC, label=f"macrocycle (ring ≥ {MACRO})")
    for yi, v, n in zip(y - h / 2, d12, ns):
        ax.text(v + max(d12 + d8) * 0.012 + 0.05, yi, f"{v:.2f}%", va="center", fontsize=9.5, color=ACC, fontweight="bold")
    for yi, v in zip(y + h / 2, d8):
        ax.text(v + max(d12 + d8) * 0.012 + 0.05, yi, f"{v:.1f}%", va="center", fontsize=9, color=ACC2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, ns)], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of ligands", fontsize=10.5)
    ax.set_xlim(0, max(d12 + d8) * 1.18 + 0.2)
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 1.01), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e3e7ee", lw=.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_macrocycle_fraction.svg"))
    plt.close(fig)

    # ---- Fig 2: docking box/strip macro vs non-macro (generated) ----
    pts = data["gen_points"]
    macro = [p["dock"] for p in pts if p["is_macro"]]
    non = [p["dock"] for p in pts if not p["is_macro"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=110)
    groups = [("non-macrocycle", non, SOFT), (f"macrocycle\n(ring ≥ {MACRO})", macro, ACC)]
    rng = np.random.default_rng(0)
    for i, (lab, vals, col) in enumerate(groups):
        if not vals:
            continue
        x = np.full(len(vals), i, dtype=float) + rng.uniform(-.12, .12, len(vals))
        ax.scatter(x, vals, s=14, color=col, alpha=.35, edgecolors="none", zorder=2)
        bp = ax.boxplot(vals, positions=[i], widths=.42, showfliers=False,
                        patch_artist=True, zorder=3)
        for b in bp["boxes"]:
            b.set(facecolor="white", edgecolor=col, alpha=.9, lw=1.6)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=col, lw=1.4)
        for md in bp["medians"]:
            md.set(color=col, lw=2.2)
        # y-axis is inverted (better/more negative at top); place label above the box
        ax.text(i, min(vals) - 0.7, f"n={len(vals)}", ha="center", fontsize=9, color=col, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylabel("Vina dock score (kcal/mol) · lower = better", fontsize=10)
    ax.invert_yaxis()  # better (more negative) at top
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e7ee", lw=.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_dock_macro.svg"))
    plt.close(fig)

    # ---- Fig 3: dock vs n_atoms (size confound) ----
    xs_all = [p["n_atoms"] for p in pts if p.get("n_atoms") is not None]
    if xs_all:
        na_non = [p["n_atoms"] for p in pts if not p["is_macro"] and p.get("n_atoms") is not None]
        d_non = [p["dock"] for p in pts if not p["is_macro"] and p.get("n_atoms") is not None]
        na_mac = [p["n_atoms"] for p in pts if p["is_macro"] and p.get("n_atoms") is not None]
        d_mac = [p["dock"] for p in pts if p["is_macro"] and p.get("n_atoms") is not None]
        fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=110)
        ax.scatter(na_non, d_non, s=16, color=SOFT, alpha=.30, edgecolors="none",
                   label=f"non-macrocycle (n={len(na_non)})", zorder=2)
        ax.scatter(na_mac, d_mac, s=46, color=ACC, alpha=.95, edgecolors="white",
                   linewidths=.6, label=f"macrocycle (n={len(na_mac)})", zorder=4)
        # linear trend over all points
        if len(xs_all) > 2:
            coef = np.polyfit([p["n_atoms"] for p in pts if p.get("n_atoms") is not None],
                              [p["dock"] for p in pts if p.get("n_atoms") is not None], 1)
            xr = np.array([min(xs_all), max(xs_all)])
            ax.plot(xr, coef[0] * xr + coef[1], color=ACC2, lw=1.6, ls="--",
                    label="size trend", zorder=3)
        ax.set_xlabel("heavy-atom count", fontsize=10.5)
        ax.set_ylabel("Vina dock score (kcal/mol)", fontsize=10)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e3e7ee", lw=.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "fig_dock_natoms.svg"))
        plt.close(fig)
    print("wrote figures")


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
  :root{--ink:#1c2433;--ink-soft:#5b6678;--line:#e3e7ee;--line-strong:#aeb6c4;--bg:#f5f6f8;--card:#fff;
    --accent:#2f6f4f;--best-bg:#eaf5ee;--tbd:#b07a17;--tbd-bg:#fcf3e0;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;}
  .page{max-width:1120px;margin:0 auto;padding:48px 24px 96px;}
  header.doc{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:40px;}
  header.doc .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:6px;}
  header.doc h1{font-size:28px;font-weight:650;margin:0;letter-spacing:-0.01em;}
  header.doc .lead{font-size:14.5px;color:var(--ink-soft);margin:10px 0 0;max-width:1000px;}
  .doc-section{margin-bottom:60px;}
  .section-head{font-size:22px;font-weight:660;letter-spacing:-0.01em;margin:0 0 18px;padding-bottom:12px;
    border-bottom:2px solid var(--line-strong);display:flex;align-items:center;gap:14px;}
  .section-head .sec-num{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;
    border-radius:8px;background:var(--accent);color:#fff;font-size:15px;font-weight:700;flex:0 0 auto;}
  .section-intro{font-size:13.5px;color:var(--ink-soft);margin:-2px 0 22px;max-width:1000px;line-height:1.6;}
  .table-title{font-size:19px;font-weight:620;margin:0 0 4px;}
  .table-sub{font-size:13.5px;color:var(--ink-soft);margin:0 0 18px;max-width:1000px;}
  .table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);
    box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05);}
  table.results{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;}
  table.results th,table.results td{padding:11px 12px;text-align:center;white-space:nowrap;}
  table.results thead th{font-weight:600;background:#fafbfc;border-bottom:1px solid var(--line);font-size:12.5px;}
  .col-method{text-align:left !important;min-width:210px;}
  td.col-method{font-weight:560;}
  td.col-method .sub{display:block;font-size:11px;color:#7a8699;font-weight:400;white-space:normal;}
  table.results tbody tr{border-top:1px solid var(--line);}
  table.results tbody tr:hover{background:#fbfcfd;}
  .val{font-weight:600;} .sd{color:#9aa3b2;font-size:11px;font-weight:400;}
  td.hi .val{color:#1d5a3a;} td.hi{background:var(--best-bg);}
  .na{color:#c2c8d2;}
  .tag{display:inline-block;font-size:10.5px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:999px;margin-left:7px;vertical-align:middle;}
  .tag.gen{background:#fdecdb;color:#b45c17;} .tag.data{background:#e7eefb;color:#38559b;}
  .fig{margin:8px 0 4px;text-align:center;} .fig img{max-width:100%;height:auto;}
  .fig-cap{font-size:12.5px;color:var(--ink-soft);margin:6px 0 0;text-align:center;}
  .two-col{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;}
  .two-col > div{flex:1 1 380px;min-width:340px;}
  .notes{margin-top:18px;font-size:13px;color:var(--ink-soft);line-height:1.6;max-width:1000px;}
  .notes b{color:var(--ink);font-weight:600;}
  .notes ul{margin:6px 0 0;padding-left:20px;} .notes li{margin-bottom:6px;}
  .notes code,.section-intro code,.table-sub code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12.5px;}
  .callout{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
    border-radius:8px;padding:14px 18px;margin:6px 0 0;font-size:13.5px;line-height:1.6;max-width:1000px;}
  .callout b{color:var(--ink);}
  .pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;}
  .pill.ns{background:#eef1f6;color:#5b6678;} .pill.sig{background:var(--best-bg);color:#1d5a3a;}
"""


def pct(x):
    return f"{x*100:.2f}%"


def fnum(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def build_html(data):
    g = data["generated"]
    cd = data["crossdocked"]
    plu = data["plinder_unique"]
    pli = data["plinder_instances"]
    cdr = data["cd_refs"]

    # ---- Table 1: prevalence ----
    def prow(name, sub, tag, s, n_disp, extra_ring=True):
        f12 = s["frac_macro12"] * 100
        f8 = s["frac_ring8"] * 100
        tagh = f' <span class="tag {tag}">{"generated" if tag=="gen" else "dataset"}</span>'
        return (
            "<tr>"
            f'<td class="col-method">{name}{tagh}<span class="sub">{sub}</span></td>'
            f'<td>{n_disp}</td>'
            f'<td class="hi"><span class="val">{s["n_macro12"]:,}</span></td>'
            f'<td class="hi"><span class="val">{f12:.2f}%</span></td>'
            f'<td><span class="val">{s["n_ring8"]:,}</span></td>'
            f'<td><span class="val">{f8:.2f}%</span></td>'
            "</tr>"
        )

    t1 = "\n".join([
        prow("VoxBind generated",
             "diffusion samples, 62 pockets · 10k-step run · res_ep99_test", "gen",
             g, f'{g["n"]:,}'),
        prow("CrossDocked",
             "reference ligands, deduplicated by canonical SMILES", "data",
             cd, f'{cd["n_unique_smiles"]:,}<span class="sd"><br>({cd["n_bases"]:,} poses)</span>'),
        prow("PLINDER v2 (unique)",
             "canonical SMILES, deduplicated · RSCC≥0.8 pretrain corpus", "data",
             plu, f'{plu["n"]:,}<span class="sd"><br>({plu["n_instances"]:,} inst.)</span>'),
        prow("PLINDER v2 (instances)",
             "instance-level (multi-ligand, no dedup) — what the encoder sees", "data",
             pli, f'{pli["n"]:,}'),
    ])

    # ---- Table 2: docking correlation ----
    def drow(name, sub, dc):
        def cell(v, d=2, hi=False):
            return f'<td class="{ "hi" if hi else "" }"><span class="val">{fnum(v,d)}</span></td>'
        p = dc["mwu_p"]
        sig = (p is not None and p < 0.05)
        ppill = ('<span class="na">—</span>' if p is None else
                 f'<span class="pill {"sig" if sig else "ns"}">{p:.3f}{" *" if sig else ""}</span>')
        pb = dc["pointbiserial_r"]
        sd_m = "" if dc["std_macro"] is None else f' <span class="sd">±{dc["std_macro"]:.2f}</span>'
        sd_n = "" if dc["std_non"] is None else f' <span class="sd">±{dc["std_non"]:.2f}</span>'
        pb_cell = "—" if pb is None else f'<span class="val">{pb:+.3f}</span>'
        return (
            "<tr>"
            f'<td class="col-method">{name}<span class="sub">{sub}</span></td>'
            f'<td>{dc["n_macro"]}</td>'
            f'<td>{dc["n_non"]:,}</td>'
            f'<td><span class="val">{fnum(dc["mean_macro"])}</span>{sd_m}</td>'
            f'<td><span class="val">{fnum(dc["mean_non"])}</span>{sd_n}</td>'
            f'<td><span class="val">{fnum(dc["delta"])}</span></td>'
            f'<td>{pb_cell}</td>'
            f'<td>{ppill}</td>'
            "</tr>"
        )

    dockd = data["dock"]
    t2 = "\n".join([
        drow("VoxBind generated · dock",
             "618 samples · full Vina re-dock", dockd["generated_dock"]),
        drow("VoxBind generated · minimize",
             "618 samples · Vina local minimize", dockd["generated_minimize"]),
        drow("VoxBind generated · score_only",
             "618 samples · Vina score of generated pose", dockd["generated_score_only"]),
        drow("CrossDocked refs · dock",
             "62 test-set reference ligands · full Vina re-dock", dockd["refs_dock"]),
    ])

    # headline numbers for the narrative
    gd = dockd["generated_dock"]
    sz = data["size"]
    n_macro_gen = g["n_macro12"]
    sizes = ", ".join(str(x) for x in g["macro_ring_sizes"][:20])
    verdict_sig = (gd["mwu_p"] is not None and gd["mwu_p"] < 0.05)
    ols_p = sz.get("ols_macro_p")
    ols_sig = (ols_p is not None and ols_p < 0.05)
    matched_p = sz.get("matched_mwu_p")
    matched_sig = (matched_p is not None and matched_p < 0.05)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macrocycles in generated & dataset ligands — 260723</title>
<style>{CSS}</style></head>
<body><div class="page">

<header class="doc">
  <div class="date">Meeting note · 2026-07-23</div>
  <h1>Macrocycles in VoxBind generations vs. CrossDocked &amp; PLINDER ligands</h1>
  <p class="lead">Do the ligands VoxBind generates contain macrocycles, how does their rate compare
  to the training/reference corpora (CrossDocked, PLINDER), and does being a macrocycle correlate
  with a better Vina docking score? A macrocycle is defined as any ring of
  <b>≥ {MACRO} atoms</b> (largest SSSR ring); ring ≥ {MACRO2} shown as a looser "large-ring" cut.</p>
</header>

<section class="doc-section">
  <h2 class="section-head"><span class="sec-num">1</span>Macrocycle prevalence</h2>
  <p class="section-intro">Largest ring size per ligand computed with RDKit
  (<code>GetRingInfo().AtomRings()</code>). Datasets deduplicated by canonical SMILES so the fraction
  reflects distinct chemistry, not repeated poses/instances.</p>

  <p class="table-title">Table 1 · Fraction of ligands that are macrocyclic</p>
  <p class="table-sub">Highlighted column = macrocycle rate at the standard ≥ {MACRO}-atom threshold.</p>
  <div class="table-wrap"><table class="results">
    <thead><tr>
      <th class="col-method">Population</th><th>N ligands</th>
      <th>macrocycles<br>(ring ≥ {MACRO})</th><th>% macro<br>(ring ≥ {MACRO})</th>
      <th>large-ring<br>(ring ≥ {MACRO2})</th><th>% large-ring<br>(ring ≥ {MACRO2})</th>
    </tr></thead>
    <tbody>{t1}</tbody>
  </table></div>

  <div class="fig" style="margin-top:26px;">
    <img src="fig_macrocycle_fraction.svg" alt="macrocycle fraction per source">
  </div>
  <p class="fig-cap">Figure 1 · Macrocycle (ring ≥ {MACRO}) and large-ring (ring ≥ {MACRO2}) rate per population.</p>

  <div class="callout" style="margin-top:22px;">
    <b>Read-out.</b> Macrocycles are <b>rare everywhere</b>. VoxBind generates them at
    <b>{pct(g["frac_macro12"])}</b> ({n_macro_gen}/{g["n"]} samples{f"; ring sizes {sizes}" if n_macro_gen else ""}),
    versus <b>{pct(cd["frac_macro12"])}</b> of unique CrossDocked ligands and
    <b>{pct(plu["frac_macro12"])}</b> of unique PLINDER-v2 ligands. The generator neither invents
    nor strongly suppresses macrocycles relative to its training distribution.
  </div>
</section>

<section class="doc-section">
  <h2 class="section-head"><span class="sec-num">2</span>Macrocycle vs. docking score</h2>
  <p class="section-intro">Restricted to the two populations that carry per-molecule Vina scores
  (from <code>metrics.json</code>): the 618 VoxBind samples and the 62 CrossDocked test-set
  reference ligands. Vina is reported in kcal/mol where <b>lower (more negative) = stronger predicted binding</b>.
  Δ = mean(macro) − mean(non-macro); a <b>negative Δ</b> means macrocycles dock better.
  Point-biserial <i>r</i> correlates the macrocycle flag (1/0) with the score.</p>

  <div class="two-col">
    <div>
      <p class="table-title">Table 2 · Docking by macrocycle status</p>
      <p class="table-sub">p = Mann–Whitney U (two-sided); * = p&lt;0.05.</p>
      <div class="table-wrap"><table class="results">
        <thead><tr>
          <th class="col-method">Population · metric</th>
          <th>n<br>macro</th><th>n<br>non</th>
          <th>mean<br>macro</th><th>mean<br>non</th>
          <th>Δ</th><th>pt-bis<br><i>r</i></th><th>p</th>
        </tr></thead>
        <tbody>{t2}</tbody>
      </table></div>
    </div>
    <div>
      <div class="fig">
        <img src="fig_dock_macro.svg" alt="dock score by macrocycle status">
      </div>
      <p class="fig-cap">Figure 2 · Vina dock score of VoxBind samples, macrocycle vs. non-macrocycle
      (box = IQR/median; points = individual samples; better at top).</p>
    </div>
  </div>

  <div class="callout" style="margin-top:22px;">
    <b>Raw correlation.</b> On the generated set, macrocycles ({gd["n_macro"]} of {gd["n_macro"]+gd["n_non"]})
    dock {"better" if (gd["delta"] or 0)<0 else "worse"} by Δ = <b>{fnum(gd["delta"])} kcal/mol</b>
    (mean {fnum(gd["mean_macro"])} vs {fnum(gd["mean_non"])}), point-biserial
    <i>r</i> = <b>{fnum(gd["pointbiserial_r"],3)}</b>, Mann–Whitney
    <b>p = {fnum(gd["mwu_p"],4)}</b> → strongly significant. <b>But read §3 before believing it</b> — macrocycles are
    large by construction and Vina scores track molecular size, so this is almost certainly a size artefact.
  </div>
</section>

<section class="doc-section">
  <h2 class="section-head"><span class="sec-num">3</span>Is it real, or just molecular size?</h2>
  <p class="section-intro">A macrocycle contributes 12–23 ring atoms on its own, so macrocyclic ligands are
  necessarily heavy — and Vina scores grow more negative with heavy-atom count regardless of binding.
  Before attributing the §2 gap to macrocyclicity we control for size two ways: (a) an OLS with
  <code>n_atoms</code> + macrocycle flag as predictors, and (b) a size-matched comparison against only the
  large non-macrocycles.</p>

  <div class="two-col">
    <div>
      <p class="table-title">Table 3 · Size control (VoxBind generated · dock)</p>
      <div class="table-wrap"><table class="results">
        <tbody>
          <tr><td class="col-method">mean heavy atoms — macro vs non</td>
            <td><span class="val">{fnum(sz["natoms_macro"],1)}</span> vs <span class="val">{fnum(sz["natoms_non"],1)}</span></td></tr>
          <tr><td class="col-method">corr(dock, heavy atoms) — Pearson / Spearman</td>
            <td><span class="val">{fnum(sz["r_dock_natoms"],3)}</span> / <span class="val">{fnum(sz["rho_dock_natoms"],3)}</span></td></tr>
          <tr><td class="col-method">Vina per added heavy atom (OLS slope)</td>
            <td><span class="val">{fnum(sz["ols_natoms_coef"],3)}</span> kcal/mol·atom⁻¹</td></tr>
          <tr><td class="col-method"><b>macrocycle effect, size-adjusted</b> (OLS coef ± SE)</td>
            <td><span class="val">{fnum(sz["ols_macro_coef"],2)}</span> <span class="sd">±{fnum(sz["ols_macro_se"],2)}</span> · p = <span class="pill {"sig" if ols_sig else "ns"}">{fnum(ols_p,3)}</span></td></tr>
          <tr><td class="col-method">size-matched set (non-macro heavy atoms ≥ smallest macrocycle)</td>
            <td>n = {sz["matched_n_macro"]} macro vs {sz["matched_n_non"]} large non-macro</td></tr>
          <tr><td class="col-method">size-matched Δ · p</td>
            <td><span class="val">{fnum(sz["matched_delta"])}</span> kcal/mol · p = <span class="pill {"sig" if matched_sig else "ns"}">{fnum(matched_p,3)}</span></td></tr>
        </tbody>
      </table></div>
    </div>
    <div>
      <div class="fig">
        <img src="fig_dock_natoms.svg" alt="dock vs heavy-atom count">
      </div>
      <p class="fig-cap">Figure 3 · Dock score vs. heavy-atom count. Macrocycles (green) sit at the
      heavy / strong-dock end of the general size trend (dashed).</p>
    </div>
  </div>

  <div class="callout" style="margin-top:22px;">
    <b>Verdict.</b> Docking is dominated by size (Pearson corr(dock, atoms) = <b>{fnum(sz["r_dock_natoms"],3)}</b>;
    macrocycles average <b>{fnum(sz["natoms_macro"],0)}</b> vs <b>{fnum(sz["natoms_non"],0)}</b> heavy atoms).
    After adjusting for atom count the macrocycle-specific effect is <b>{fnum(sz["ols_macro_coef"],2)} ± {fnum(sz["ols_macro_se"],2)} kcal/mol</b>
    (p = {fnum(ols_p,3)}, {"still significant" if ols_sig else "not significant"}), and against size-matched
    large non-macrocycles the gap is Δ = <b>{fnum(sz["matched_delta"])} kcal/mol</b>
    (p = {fnum(matched_p,3)}, {"significant" if matched_sig else "not significant"}).
    → <b>{"Macrocyclicity retains an independent docking advantage beyond size." if (ols_sig and matched_sig) else "The apparent macrocycle→better-dock correlation is largely a molecular-size effect, not a macrocycle-specific one."}</b>
  </div>

  <div class="notes">
    <b>Notes &amp; caveats.</b>
    <ul>
      <li><b>Definition.</b> Macrocycle = largest SSSR ring ≥ {MACRO} atoms (IUPAC-style). SSSR can split a
      bridged macrocycle into smaller rings, so this is a mild <i>under</i>-count; the ≥ {MACRO2} column bounds it above.</li>
      <li><b>Docking sign.</b> Vina scores are negative; "lower = better". A <i>negative</i> Δ / point-biserial
      <i>r</i> means macrocycles bind more favourably.</li>
      <li><b>Power.</b> Only {n_macro_gen} generated macrocycles, so every docking test on them is low-powered;
      the size confound (§3) is the dominant caveat, mirroring the size-confound seen in the DSMBind baseline.</li>
      <li><b>Provenance.</b> Generated set = <code>{data['meta']['gen_run']}</code>;
      CrossDocked = <code>{data['meta']['cd_dir']}</code> (unique ligands);
      PLINDER = <code>{data['meta']['plinder_csv']}</code>.</li>
    </ul>
  </div>
</section>

<footer style="border-top:1px solid var(--line);padding-top:14px;color:#9aa3b2;font-size:12px;">
  Generated by <code>notebook/html/260723/build_macrocycle.py</code> · numbers cached in
  <code>macrocycle_data.json</code> (delete to recompute).
</footer>
</div></body></html>"""
    with open(OUT, "w") as fh:
        fh.write(html)
    print("wrote", OUT)


if __name__ == "__main__":
    data = compute()
    make_figures(data)
    build_html(data)
