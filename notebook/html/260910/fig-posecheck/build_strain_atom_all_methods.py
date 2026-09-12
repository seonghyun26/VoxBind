#!/usr/bin/env python3
"""build_strain_atom_all_methods.py — strain against ligand SIZE, ALL methods.

    strain_per_atom_all_methods_{median,mean}.*   lines, every method, per heavy-atom count
    strain_per_atom_all_methods.{json,csv}        the numbers behind them

`build_posecheck_per_atom.py` next door draws this metric on this axis for the three arms
we run locally (`strain_per_atom_{median,mean}_{core,all}`), and it cannot draw the
published baselines: pose_common.ARMS now names them, but the trees it points at
(`exps/baselines_pose/<m>`) carry PoseBusters only, so `arms_for("s", ...)` drops them.
This is the eight-method version of the same figure, and it is the SIZE sibling of
`build_strain_rotbond_baselines.py`, which does the same job on the rotatable-bond axis.

WHERE THE BASELINES COME FROM. Straight out of `posecheck_<Method>.json` -- the per-molecule
exports `export_posecheck_json.py` wrote from the published runs, which carry `n` (heavy
atoms) and `s` (strain) for every molecule. No join is needed here at all: unlike the
rotatable-bond axis, which had to go back to the results bundle for SMILES, the quantity
this figure resolves against is already in the export. Only the p79 electron-density
pockets are kept, by the export's own pocket index.

MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN -- the baselines were
scored against the whole `*_rec.pdb`, the local arms against the pocket10 crop. Strain is
UFF relaxation of the ligand's own conformer under a position constraint, so the receptor
cannot enter it, and measuring the same molecules both ways confirms it: median |relative
difference| 0.6-1.0 %. Do NOT extend the mixing to clashes or interactions.

DecompDiff PUTS EXACTLY ONE SIZE IN EACH POCKET, AND IT IS THE CRYSTAL LIGAND'S. Checked
here, not assumed: in all 79 pockets every DecompDiff molecule has the same heavy-atom
count, and in all 79 that count equals the reference ligand's. Its reference prior fixes
the atom budget per pocket, so on this axis DecompDiff is not a distribution but a comb
over the 79 reference sizes, with exact zeros at 7, 24, 30, 34, 36, 39-41, 43-44 atoms --
counts no crystal ligand in the set happens to occupy. Two consequences, both live:

  * EVERY MODEL CURVE POOLS A +-1 ATOM WINDOW. Drawn per exact count, DecompDiff's line
    would break ten times inside the axis on molecules that were never going to exist. The
    window is one atom wide, the same treatment for all eight methods, and it moves a
    median of a continuous quantity almost nowhere at these counts (100-400 molecules each).
    The one hole it does not fill -- 40 atoms, where no reference sits within +-1 -- is left
    as a break rather than widened away.
  * DecompDiff IS SIZE-MATCHED TO THE REFERENCE BY CONSTRUCTION, which is exactly what this
    figure controls for. Its pooled numbers elsewhere carry that advantage; here it is
    neutralised, so this axis is the fair place to read it.

THE AXIS RUNS 5-44 ATOMS, the range `build_posecheck_per_atom.py` already draws, and each
method is drawn only where its window holds at least pose_common.MIN_N molecules, so a
line stops where the method stopped making that size instead of running on noise. The
share of each method's ligands beyond 44 atoms is printed and exported, never dropped
silently.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posecheck/build_strain_atom_all_methods.py
    ... --stat median        # only the median figure
"""
import argparse
import collections
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

# The five published baselines, by the label their export file carries. DecompDiff's export
# is the reference-prior run, which is the one every 260910 figure reports.
BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind"]

# THE IDENTITY CONSTANTS BELOW ARE THE SIBLING'S, kept in step with
# build_strain_rotbond_baselines.py by hand -- the two figures are read side by side and a
# method must not change dash, order or spelling between them. Importing them from that
# module would drag in torch and its import-time `pc.REF_WIN` override, which is set for an
# axis a third as long as this one.
LOCAL_KEYS = ("targetdiff", "vanilla", "ours_v1")
REF_KEY = "ours_v1"          # any arm carries the same crystal ligand per pocket
LOCAL_LABELS = []            # filled by load_all(), in LOCAL_KEYS order
RELABEL = {"ours_v1": "CoDE"}
ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff", "FuncBind", "VoxBind",
         "CoDE"]
BASE_LW = 1.6
BASE_DASH = {"AR": (0, (5, 2)), "Pocket2Mol": (0, (1, 1.6)),
             "DiffSBDD": (0, (6, 2, 1, 2)), "DecompDiff": (0, (9, 3)),
             "FuncBind": (0, (3, 1.4, 1, 1.4))}

X_LO, X_HI = 5, 44
WIN = 1                      # +-1 atom, see the docstring: DecompDiff's comb
KEY_H = 0.92                 # inches of key strip under the panel; see lines()
MEAN_CLIP = 1e6              # the mean panel's top, as in build_posecheck_per_atom.py
STATS = {"median": lambda v: float(np.median(v)), "mean": lambda v: float(np.mean(v))}
TAIL = 1e4


# ── load ─────────────────────────────────────────────────────────────────────────
def baseline_rows(label, keep):
    """The export's p79 molecules, as pose_common-shaped rows."""
    export = json.load(open(os.path.join(HERE, f"posecheck_{label}.json")))
    rows = [{"n": m["n"], "s": m["s"], "c": m["c"]}
            for m in export["molecules"] if m["p"] in keep]
    return rows, len(export["molecules"])


def load_all():
    by_key = {key: (lab, root) for lab, key, root in pc.ARMS}
    series, notes, local = {}, {}, []
    for key in LOCAL_KEYS:
        if key not in by_key:
            raise SystemExit(f"pose_common.ARMS no longer defines {key!r}")
        lab, root = by_key[key]
        lab = RELABEL.get(key, lab)
        series[lab] = [r for t in pc.P79 for r in pc.rows_of(os.path.join(root, t))]
        notes[lab] = {"source": f"{root} (pocket10 crop)", "n_rows": len(series[lab])}
        local.append(lab)
    LOCAL_LABELS[:] = local

    refrows = [r for t in pc.P79
               for r in pc.rows_of(os.path.join(by_key[REF_KEY][1], t), reference=True)]

    keep = {int(t.split("_")[1]) for t in pc.P79}
    for lab in BASELINES:
        rows, n_all = baseline_rows(lab, keep)
        series[lab] = rows
        notes[lab] = {"source": f"posecheck_{lab}.json (whole receptor)",
                      "n_rows": len(rows), "n_all_pockets": n_all}
    return series, refrows, notes


def check_decompdiff():
    """The docstring's claim, re-checked on every run: one size per pocket, and it is the
    crystal ligand's. If DecompDiff's export is ever replaced by the non-reference-prior
    run this stops holding, and the +-1 window stops being something this figure needs."""
    export = json.load(open(os.path.join(HERE, "posecheck_DecompDiff.json")))
    ref = {}
    for t in pc.P79:
        r = pc.rows_of(os.path.join(pc.REF_ROOT, t), reference=True)
        if r:
            ref[int(t.split("_")[1])] = r[0]["n"]
    sizes = collections.defaultdict(set)
    for m in export["molecules"]:
        if m["p"] in ref:
            sizes[m["p"]].add(m["n"])
    one = sum(len(v) == 1 for v in sizes.values())
    same = sum(len(v) == 1 and next(iter(v)) == ref[p] for p, v in sizes.items())
    return one, same, len(sizes)


def order(labels):
    unknown = [l for l in labels if l not in ORDER]
    if unknown:
        raise SystemExit(f"not in table_drug_design.tex's order: {unknown}")
    return [l for l in ORDER if l in labels]


def style_of(label):
    return (pc.MODEL_LW, "-") if label in LOCAL_LABELS else (BASE_LW, BASE_DASH[label])


# ── figures ──────────────────────────────────────────────────────────────────────
def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), facecolor="white")
    plt.close(fig)


def handles(labels):
    h = [Line2D([], [], color=pc.REF_COLOR, lw=pc.REF_LW, ls=pc.DASH, label=pc.REF_LABEL)]
    for lab in labels:
        lw, ls = style_of(lab)
        h.append(Line2D([], [], color=color(lab), lw=lw, ls=ls, label=lab))
    return h


def pool(per_lab, a):
    """The molecules behind the point at `a`: its +-WIN neighbours, CLIPPED TO THE AXIS.

    The clip is not cosmetic. DecompDiff has no molecule at 43 or 44 heavy atoms and 125
    at 45+, so an unclipped window put a point at 44 computed entirely from molecules the
    same file reports as beyond the axis -- a plotted value representing nothing near where
    it was plotted. Inside the axis the window still interpolates across its comb, which is
    what it is for; at the edges it no longer extrapolates from outside."""
    return [v for n, vals in per_lab.items()
            if abs(n - a) <= WIN and X_LO <= n <= X_HI for v in vals]


def curve(per_lab, xs, f):
    """The statistic over that window, None where it is too thin to mean anything -- which
    leaves a gap rather than an invented value, the same rule pose_common.reference_curve
    applies to the crystal ligands."""
    out = []
    for a in xs:
        p = pool(per_lab, a)
        out.append(f(p) if len(p) >= pc.MIN_N else None)
    return out


def lines(xs, per, ref_per, labels, stat):
    """One statistic, one panel, and the key in a strip of its own beneath it.

    NINE SERIES DO NOT LEAVE A CORNER FREE. Inside the axes this key is either three
    columns across the top, where it covers precisely the high-strain curves the figure is
    about, or three rows deep in a corner the crystal-ligand line runs through. Under the
    panel it covers nothing, and it is the same 3x3 block the box figure's key is.

    THE MEAN PANEL IS CLIPPED TO THE BULK and the points that leaves off are NAMED in the
    run log, exactly as build_posecheck_per_atom.py does it: 4-13 % of molecules fail UFF
    relaxation and land between 1e4 and 1e13, so one of them at a thin count carries that
    count's mean four decades up and an unclipped axis spends fourteen decades on it."""
    f = STATS[stat]
    clip = MEAN_CLIP if stat == "mean" else None
    fig, (ax, key) = plt.subplots(2, 1, figsize=(pc.FIG_W, pc.PANEL_H + KEY_H), dpi=220,
                                  gridspec_kw={"height_ratios": [pc.PANEL_H, KEY_H]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key.axis("off")
    ax.plot(xs, pc.reference_curve(ref_per, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
            ls=pc.DASH, zorder=4, dash_capstyle="round")
    dropped = []
    for lab in labels:
        y = curve(per[lab], xs, f)
        if clip:
            over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
            if over:
                dropped.append((lab, over))
        lw, ls = style_of(lab)
        ax.plot(xs, y, color=color(lab), lw=lw, ls=ls, zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    if clip:
        ax.set_ylim(top=clip)
    pc.furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=pc.X_LABEL,
                 xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    pc.legend(key, handles(labels), loc="center", ncol=3, fontsize=11)
    pc.fit(fig, pad=0.5)
    save(fig, f"strain_per_atom_all_methods_{stat}")
    return dropped


# ── exports ──────────────────────────────────────────────────────────────────────
def tail_share(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > TAIL for x in v) / len(v), 2) if v else None


def beyond_share(rows):
    return round(100 * sum(r["n"] > X_HI for r in rows) / max(len(rows), 1), 2)


def exports(xs, series, per, ref_per, refrows, labels, notes, dd):
    out = {
        "built": "2026-09-10",
        "metric": "PoseCheck strain against generated-ligand heavy-atom count",
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79)},
        "window": f"model curves pool +-{WIN} atom, clipped to the axis; null below "
                  f"{pc.MIN_N} molecules",
        "n_columns": ("`n` is the molecules at that exact heavy-atom count; `n_window` is "
                      "the sample the value on the row was actually computed from"),
        "window_reason": ("DecompDiff's reference prior fixes one heavy-atom count per "
                          "pocket, so per exact count its line is a comb; checked on this "
                          f"run: {dd[0]}/{dd[2]} pockets hold a single size and "
                          f"{dd[1]}/{dd[2]} of those equal the crystal ligand's"),
        "reference_window": f"+-{pc.REF_WIN} atoms, >= {pc.MIN_REF} ligands",
        "scope_note": ("baselines scored whole-receptor, local arms on the pocket10 crop; "
                       "measured median |rel diff| 0.6-1.0 % on strain, which is "
                       "receptor-independent by construction"),
        "heavy_atoms": xs, "arms": {},
    }
    for lab in labels:
        rows = series[lab]
        out["arms"][lab] = {
            **notes[lab],
            "n_scored": sum(len(v) for v in per[lab].values()),
            "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2),
            "atoms_median": float(np.median([r["n"] for r in rows])),
            "strain_gt_1e4": tail_share(rows),
            f"beyond_{X_HI}_atoms_pct": beyond_share(rows),
            "n": [len(per[lab].get(a, ())) for a in xs],
            "n_window": [len(pool(per[lab], a)) for a in xs],
            **{f"strain_{s}": curve(per[lab], xs, STATS[s]) for s in STATS},
        }
    out["arms"][pc.REF_LABEL] = {
        "source": "crystal ligand of each of the 79 pockets",
        "n_scored": sum(len(v) for v in ref_per.values()),
        "atoms_mean": round(float(np.mean([r["n"] for r in refrows])), 2),
        "atoms_median": float(np.median([r["n"] for r in refrows])),
        "strain_gt_1e4": tail_share(refrows),
        f"beyond_{X_HI}_atoms_pct": beyond_share(refrows),
        "n": [len(ref_per.get(a, ())) for a in xs],
        "n_window": [sum(len(v) for x, v in ref_per.items() if abs(x - a) <= pc.REF_WIN)
                     for a in xs],
        **{f"strain_{s}": pc.reference_curve(ref_per, xs, STATS[s]) for s in STATS},
    }
    json.dump(out, open(os.path.join(HERE, "strain_per_atom_all_methods.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "strain_per_atom_all_methods.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "heavy_atoms", "n", "n_window", "strain_median", "strain_mean"])
        for lab in labels + [pc.REF_LABEL]:
            d = out["arms"][lab]
            for i, a in enumerate(xs):
                w.writerow([lab, a, d["n"][i], d["n_window"][i]]
                           + ["" if d[f"strain_{s}"][i] is None
                              else round(d[f"strain_{s}"][i], 3) for s in STATS])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stat", choices=("median", "mean", "both"), default="both")
    args = ap.parse_args()
    stats = ("median", "mean") if args.stat == "both" else (args.stat,)

    pc.use_style()
    series, refrows, notes = load_all()
    labels = order(series)
    dd = check_decompdiff()
    per = {lab: pc.by_size(series[lab], "s") for lab in labels}
    ref_per = pc.by_size(refrows, "s")
    xs = list(range(X_LO, X_HI + 1))
    drops = {s: lines(xs, per, ref_per, labels, s) for s in stats}
    out = exports(xs, series, per, ref_per, refrows, labels, notes, dd)

    print(f"{len(pc.P79)} pockets · {len(labels)} methods + reference · "
          f"x = {X_LO}-{X_HI} heavy atoms · ±{WIN}-atom window, "
          f"drawn where it pools ≥{pc.MIN_N} molecules\n")
    print(f"DecompDiff: {dd[0]}/{dd[2]} pockets hold a single heavy-atom count and "
          f"{dd[1]}/{dd[2]} of them equal the crystal ligand's — it is size-matched to the "
          f"reference by construction")
    print(f"\n{'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>7d}" for a in (10, 15, 20, 25, 30, 35, 40))
          + f" {'drawn':>9s} {'>44 at':>7s} {'>1e4':>7s}   (strain median at n atoms)")
    for lab in labels + [pc.REF_LABEL]:
        d = out["arms"][lab]
        med = d["strain_median"]
        cells = [(f"{med[xs.index(a)]:7.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>7s}") for a in (10, 15, 20, 25, 30, 35, 40)]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        gaps = [a for a in range(drawn[0], drawn[-1] + 1) if a not in drawn] if drawn else []
        span = (f"{drawn[0]}-{drawn[-1]}" + (f"*{len(gaps)}" if gaps else "")) if drawn \
            else "—"
        print(f"{lab:16s} {d['n_scored']:6d} {d['atoms_mean']:6.1f} " + " ".join(cells)
              + f" {span:>9s} {d[f'beyond_{X_HI}_atoms_pct']:6.1f}% "
              + (f"{d['strain_gt_1e4']:6.2f}%" if d["strain_gt_1e4"] is not None else
                 f"{'—':>7s}"))
    for lab, over in drops.get("mean", []):
        pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
        print(f"  strain_per_atom_all_methods_mean · off-panel above {MEAN_CLIP:.0e}: "
              f"{lab:12s} {len(over):2d} of {len(xs)}: {pts}")
    holes = {lab: [a for a in range(*(lambda d: (d[0], d[-1] + 1))(
        [a for a, v in zip(xs, out["arms"][lab]["strain_median"]) if v is not None]))
        if out["arms"][lab]["strain_median"][xs.index(a)] is None]
        for lab in labels + [pc.REF_LABEL]}
    for lab, hs in holes.items():
        if hs:
            print(f"  * {lab}'s line breaks inside its span at {hs} heavy atoms — too few "
                  f"molecules in the window there, not a drawing error")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
