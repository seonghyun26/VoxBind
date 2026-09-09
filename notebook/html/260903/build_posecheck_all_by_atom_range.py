"""build_posecheck_all_by_atom_range.py — every method on one axis: the five published
baselines PLUS the three we ran here (TargetDiff, vanilla VoxBind, Ours v1), stratified
by generated-molecule heavy-atom count.

This is the merge `build_posecheck_baselines_by_atom_range.py` says it would do "on that
machine". It cannot actually be run that way here, for two reasons:

 1. prj-denovo/baselines is not on this box, so the five baselines are read from the
    per-molecule exports the output folder ships (`260910/fig-posecheck/posecheck_<Method>.json`),
    exactly as that folder's README documents. The loader is checked on every run: the
    per-bin numbers it reproduces from the exports must match the stored
    `posecheck_baselines_by_atom_range.json` on all five methods x five bins x three
    statistics, or the script aborts instead of drawing.

 2. THE ROOTS THAT SCRIPT DECLARES FOR OUR THREE ARE THE WRONG PROTOCOL. Its `REMOTE`
    entries point at each run's main samples directory, where PoseCheck was scored against
    the *pocket10 crop*; the five baselines were scored against the whole `*_rec.pdb`.
    Clashes are receptor-dependent and the crop hides some of them -- pooled clash mean
    5.25 crop vs 6.27 full for vanilla, 6.49 vs 7.55 for Ours v1, 10.39 vs 11.11 for
    TargetDiff (see posecheck_crop_vs_full.json). Mixing the two would have handed our
    three a ~1-clash head start. This script therefore reads our three from
    `frozenenc_probes/posecheck_full/`, which is the whole-receptor re-scoring, and its
    pooled numbers reproduce `ours_posecheck_full.json` exactly.

POCKET SUBSET. The baselines cover all 79 electron-density pockets; the whole-receptor
re-scoring of our three now covers all 79 as well: target_71 was missing only because
scripts 70/71/72 default to `p78_targets.json`, a list that exists for a *docking* failure
(pdb2pqr30 rejects that pocket's 10 A crop) which PoseCheck never had. Re-running 72 on
that one target filled it in. Every series is still restricted to the pockets both sides
share, computed at run time, so no method is scored on a pocket another one never saw; the
console prints the shared count and anything dropped.

THE REFERENCE LIGAND is drawn on the strain figure only. PoseCheck's strain
(`calculate_strain_energy(mol, num_confs=50)`) never touches the receptor, so the crystal
ligand's strain is protocol-independent and can be read from our local run. Its *clashes*
are not: the only local reference scoring is against the crop, so it would sit ~0.3-1
clash low next to eight whole-receptor series, and it is left off the violin rather than
quietly compared. The whole-receptor reference clash means from the baselines' own run are
carried through into the JSON so the number is not lost.

STRAIN IS NOISY BY CONSTRUCTION. `num_confs=50` random conformers means re-scoring the
same pose gives a slightly different answer: over 7,349 molecules scored twice here the
median relative difference is 1.0%, p90 8.9%. Fine for an ECDF, not for ranking two
methods a few percent apart.

COLOURS. VoxBind and Ours take the blue/green pair the CDG charts use for "C" and
"C+D+G" (#4363d8 / #3cb44b), so the key carries across figures, and are drawn thick; the five baselines are context, so
they are thin and each carries its own dash pattern -- with nine series on one axis hue
alone cannot be the identity channel. Pocket2Mol moves off orange (#e67e22 -> #e87ba4)
because that orange is a dead ringer for Ours; every other baseline keeps the colour the
baseline-only figures used.

Writes into 260910/fig-posecheck/ (the previous baseline-only figures were copied to
_baselines_only/ first):
  * strain_ecdf_<bin>.{png,svg,pdf}    5 size bins + `all`; no footnote -- the protocol
    notes the grey caption used to carry live in posecheck_all_by_atom_range.json
  * clash_violin_<bin>.{png,svg,pdf}   5 size bins + `all`

STYLE follows build_vina_3line.py so the two families sit together in one document:
white ground, black spines and short outward ticks, a dotted mid-grey grid, an opaque
white legend box with a grey rule, and a PDF alongside the png/svg with TrueType (not
Type 3) fonts. The panel titles stay -- unlike the vina pair, where mean vs median moved
into the y-axis name, the size bin has nowhere else to live inside the frame.
  * posecheck_all_by_atom_range.json / .csv

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_posecheck_all_by_atom_range.py
"""
import csv
import glob
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "260910", "fig-posecheck")
E = "/home1/irteam/VoxBind/voxbind/exps"

# key -> (label, colour, linewidth, linestyle)
BASELINES = [
    ("AR",         "AR",         "#9b59b6", 1.6, "-"),
    ("Pocket2Mol", "Pocket2Mol", "#e87ba4", 1.6, (0, (5, 2))),
    ("DiffSBDD",   "DiffSBDD",   "#e34948", 1.6, (0, (1, 1.6))),
    ("DecompDiff", "DecompDiff", "#34495e", 1.6, (0, (6, 2, 1, 2))),
    ("FuncBind",   "FuncBind",   "#a9744f", 1.6, (0, (9, 3))),
]
# Blue and green are the CDG chart's own pair (build_appendixB_bar.PALETTE[9] / [10],
# "C" and "C+D+G"), so VoxBind and Ours keep the same key a reader already learned there:
# blue = the model without our addition, green = with it. TargetDiff takes that palette's
# orange (PALETTE[5]) and is dashed -- against the green it sits in the CVD 6-8 band, so it
# needs the second channel.
LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", "#f58231", 2.4, (0, (6, 2))),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    "#4363d8", 3.0, "-"),
    ("Ours · v1",     f"{E}/frozenenc_probes/posecheck_full/ours_v1",    "#3cb44b", 3.4, "-"),
]
# strain only -- see the module docstring
REF_ROOT = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
REF_LABEL, REF_COLOR = "Reference ligand", "#4b5563"

EDGES = [0, 16, 21, 26, 31, 10 ** 6]
LABELS = ["≤15", "16–20", "21–25", "26–30", ">30", "all sizes"]
SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30", "all"]
POOLED = len(LABELS) - 1          # the extra bin every molecule also lands in

# Shared with build_vina_3line.py so the two figure families read as one set: a white
# ground, black axis furniture with short outward ticks, a dotted mid-grey grid, and an
# opaque white legend box carrying a quiet grey rule.
BG, INK, GRID, AXIS = "#ffffff", "#000000", "#c2c6cd", "#000000"
LEGEND_EDGE = "#b6bbc3"


def bin_of(n):
    return min(int(np.searchsorted(EDGES, n, side="right")) - 1, len(EDGES) - 2)


def empty():
    return {b: {"strain": [], "clash": []} for b in range(len(LABELS))}


def add(out, n, s, c):
    for b in (bin_of(n), POOLED):
        if s is not None and np.isfinite(s):
            out[b]["strain"].append(float(s))
        if c is not None and np.isfinite(c):
            out[b]["clash"].append(float(c))


def load_baseline(method, keep):
    """One of the five, from its per-molecule export. `keep` is a set of pocket indices."""
    d = json.load(open(f"{OUT}/posecheck_{method}.json", encoding="utf-8"))
    out = empty()
    for m in d["molecules"]:
        if m["p"] in keep:
            add(out, m["n"], m["s"], m["c"])
    return out, set(d["density79_pockets"])


def load_local(root, keep, reference=False):
    """One of ours, from per-target metrics.json."""
    out = empty()
    seen = set()
    for path in sorted(glob.glob(f"{root}/target_*/metrics.json")):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        seen.add(idx)
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            pc = m.get("posecheck") or {}
            add(out, m["n_atoms"], pc.get("strain"), pc.get("clashes"))
    return out, seen


def pockets_of(root):
    return {int(os.path.basename(os.path.dirname(p)).split("_")[1])
            for p in glob.glob(f"{root}/target_*/metrics.json")}


def stats(cell):
    s_, c_ = cell["strain"], cell["clash"]
    return {
        "n_strain": len(s_), "n_clash": len(c_),
        "strain_median": round(st.median(s_), 1) if s_ else None,
        "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
        "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
        "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
        "clash_median": round(float(np.median(c_)), 1) if c_ else None,
    }


def main():
    # ── the pocket subsets ────────────────────────────────────────────────────
    probe = json.load(open(f"{OUT}/posecheck_AR.json", encoding="utf-8"))
    density79 = set(probe["density79_pockets"])
    shared = set(density79)
    for _, root, *_ in LOCAL:
        shared &= pockets_of(root)
    dropped = sorted(density79 - shared)
    print(f"pockets: baselines {len(density79)} · shared with our three {len(shared)}"
          f" · dropped {['target_%02d' % i for i in dropped]}")

    # ── verification: the exports must reproduce the stored per-bin numbers ───
    stored = json.load(open(f"{OUT}/posecheck_baselines_by_atom_range.json",
                            encoding="utf-8"))
    mismatch = 0
    for key, label, *_ in BASELINES:
        data, _ = load_baseline(key, density79)
        for b in range(len(EDGES) - 1):
            want, got = stored["methods"][label][b], stats(data[b])
            for a, c in (("n", "n_strain"), ("strain_median", "strain_median"),
                         ("clash_mean", "clash_mean")):
                if want.get(a) != got.get(c):
                    mismatch += 1
                    print(f"  MISMATCH {label} bin {LABELS[b]} {a}: "
                          f"stored {want.get(a)} != rebuilt {got.get(c)}")
    if mismatch:
        raise SystemExit(f"loader disagrees with the stored baseline JSON on {mismatch} "
                         f"values -- refusing to plot")
    print("verification: exports reproduce posecheck_baselines_by_atom_range.json "
          "on 5 methods x 5 bins x 3 statistics, 0 mismatches")

    # ── the data actually plotted, all on the shared pockets ──────────────────
    DATA, SERIES = {}, []
    for key, label, colour, lw, ls in BASELINES:
        d79, _ = load_baseline(key, density79)
        dsh, _ = load_baseline(key, shared)
        DATA[label] = dsh
        SERIES.append((label, colour, lw, ls))
        a, b = stats(d79[POOLED]), stats(dsh[POOLED])
        print(f"  {label:12s} 79 pockets n={a['n_clash']:5d} clash {a['clash_mean']:5.2f} "
              f"strain {a['strain_median']:7.1f}   ->  {len(shared)} pockets "
              f"n={b['n_clash']:5d} clash {b['clash_mean']:5.2f} strain {b['strain_median']:7.1f}")
    for label, root, colour, lw, ls in LOCAL:
        DATA[label], _ = load_local(root, shared)
        SERIES.append((label, colour, lw, ls))
    REF, _ = load_local(REF_ROOT, shared, reference=True)

    # ── figures ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "svg.fonttype": "none",          # keep SVG text editable in Illustrator
        "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
    })
    XFLOOR, XTOP = 1e-2, 3e3

    def style(ax, *, grid_axis="both"):
        """The vina figures' axis furniture: black spines and outward ticks, dotted grid.

        `grid_axis` is "y" for the violin -- its x is categorical, so a vertical rule
        through every category centre is a picket fence, not a reading aid."""
        ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(1.1)
        ax.tick_params(labelsize=11, direction="out", length=3.5, width=1.1, pad=4,
                       colors=AXIS)
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.9, ls=(0, (1, 2.6)))
        ax.set_axisbelow(True)

    for b, (lab, slug) in enumerate(zip(LABELS, SLUGS)):
        # strain ECDF -----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        fig.patch.set_facecolor(BG)
        style(ax)
        for label, colour, lw, ls in SERIES:
            v = np.asarray(DATA[label][b]["strain"], dtype=float)
            if v.size < 20:
                continue
            x = np.sort(np.clip(v, XFLOOR, None))
            ax.plot(x, np.arange(1, x.size + 1) / x.size, color=colour, lw=lw, ls=ls,
                    label=f"{label}  ·  median {np.median(v):.0f}", solid_capstyle="round")
        r = np.asarray(REF[b]["strain"], dtype=float)
        if r.size >= 3:
            x = np.sort(np.clip(r, XFLOOR, None))
            ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR, lw=2.0,
                    ls=(0, (3, 2)),
                    label=f"{REF_LABEL}  ·  median {np.median(r):.0f}  (n={r.size})")
        ax.set_xscale("log")
        ax.set_xlim(XFLOOR, XTOP)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("UFF strain energy (kcal mol⁻¹) · log scale", fontsize=11.5)
        ax.set_ylabel("cumulative probability", fontsize=11.5)
        ax.set_title(f"Strain energy — {lab} heavy atoms", fontsize=14,
                     fontweight="620", loc="left", pad=10)
        # Nine series with medians in the labels, so the box is wide; it stays opaque
        # white with a grey rule so the dotted grid does not run through the text. The
        # font stays at 9.5 rather than the vina figures' 10.5 -- three short labels
        # there, nine long ones here.
        leg = ax.legend(loc="upper left", frameon=True, fontsize=9.5,
                        handlelength=1.6, handletextpad=0.5, labelspacing=0.32,
                        borderpad=0.4, borderaxespad=0.5, facecolor="white",
                        edgecolor=LEGEND_EDGE, framealpha=1.0)
        leg.get_frame().set_boxstyle("square", pad=0.16)
        leg.get_frame().set_linewidth(0.7)
        leg.set_zorder(6)
        for text in leg.get_texts():      # identity rides the swatch, not the ink
            text.set_color(INK)
        fig.tight_layout(pad=0.5)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{OUT}/strain_ecdf_{slug}.{ext}", dpi=170, facecolor=BG,
                        bbox_inches="tight")
        plt.close(fig)

        # clash violin ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10.6, 5.6))
        fig.patch.set_facecolor(BG)
        style(ax, grid_axis="y")
        vals, cols, ticks = [], [], []
        for label, colour, lw, ls in SERIES:
            v = DATA[label][b]["clash"]
            if len(v) < 20:
                continue
            vals.append(v)
            cols.append(colour)
            ticks.append(label.replace(" · ", "\n").replace(" σ", "\nσ"))
        # KDE fitted on log10(x+1) and the ticks relabelled with real counts: a plain log
        # axis is impossible because a few percent of poses have zero clashes, and symlog
        # would warp only the display while the KDE stayed in data space.
        tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
        parts = ax.violinplot([tf(v) for v in vals], showextrema=False, widths=0.82)
        for body, colour in zip(parts["bodies"], cols):
            body.set_facecolor(colour)
            body.set_alpha(0.55)
            body.set_edgecolor(colour)
            body.set_linewidth(1.2)
        for i, v in enumerate(vals, start=1):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
            ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
        hi = max(max(v) for v in vals)
        TICKS = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= hi * 1.6]
        ax.set_yticks(tf(TICKS))
        ax.set_yticklabels([str(t) for t in TICKS])
        ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.20)
        for i, v in enumerate(vals, start=1):
            ax.text(i, ax.get_ylim()[1], f"mean {np.mean(v):.2f}\nmed {np.median(v):.0f}",
                    ha="center", va="top", fontsize=8.5, color="#3a4352", linespacing=1.35)
        ax.set_xticks(range(1, len(vals) + 1))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_ylabel("steric clashes per pose  ·  log-spaced", fontsize=11.5)
        ax.set_title(f"Steric clashes — {lab} heavy atoms", fontsize=14,
                     fontweight="620", loc="left", pad=10)
        fig.tight_layout(pad=0.5)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{OUT}/clash_violin_{slug}.{ext}", dpi=170, facecolor=BG,
                        bbox_inches="tight")
        plt.close(fig)

    # ── numbers ──────────────────────────────────────────────────────────────
    summary = {
        "bins": LABELS,
        "edges": EDGES[:-1] + ["inf"],
        "n_pockets": len(shared),
        "pockets": sorted(shared),
        "dropped_from_density79": [f"target_{i:02d}" for i in dropped],
        "protocol": "whole *_rec.pdb receptor, PoseCheck on the pose as generated",
        "sources": {
            "baselines": "260910/fig-posecheck/posecheck_<Method>.json (prj-denovo/baselines)",
            "ours": "voxbind/exps/frozenenc_probes/posecheck_full/<run>",
            "reference": f"{REF_ROOT} (strain only; its clashes are crop-receptor)",
        },
        "methods": {label: [stats(DATA[label][b]) for b in range(len(LABELS))]
                    for label, *_ in SERIES},
    }
    summary["methods"][REF_LABEL] = [
        {**stats(REF[b]), "clash_mean": None, "clash_median": None,
         "note": "strain only"} for b in range(len(LABELS))]
    # the whole-receptor reference clash numbers, carried over so they are not lost
    summary["reference_clash_whole_receptor"] = [
        stored["methods"]["Reference ligand"][b].get("clash_mean")
        for b in range(len(EDGES) - 1)] + [None]
    json.dump(summary, open(f"{OUT}/posecheck_all_by_atom_range.json", "w",
                            encoding="utf-8"), indent=1, ensure_ascii=False)

    with open(f"{OUT}/posecheck_all_by_atom_range.csv", "w", newline="",
              encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["bin", "method", "n_strain", "strain_median", "strain_q25",
                    "strain_q75", "n_clash", "clash_mean", "clash_median"])
        for b, lab in enumerate(LABELS):
            for label in summary["methods"]:
                r = summary["methods"][label][b]
                w.writerow([lab, label, r["n_strain"], r["strain_median"],
                            r["strain_q25"], r["strain_q75"], r["n_clash"],
                            r["clash_mean"], r["clash_median"]])

    print(f"\n{'bin':10s} {'method':14s} {'n':>6s} {'strain med':>11s} {'IQR':>19s} "
          f"{'clash mean':>11s} {'med':>5s}")
    for b, lab in enumerate(LABELS):
        for label in summary["methods"]:
            r = summary["methods"][label][b]
            if not r["n_strain"]:
                continue
            iqr = (f"{r['strain_q25']:7.1f}–{r['strain_q75']:<10.1f}"
                   if r["strain_q25"] is not None else " " * 18)
            cm = f"{r['clash_mean']:11.2f}" if r["clash_mean"] is not None else f"{'—':>11s}"
            cd = f"{r['clash_median']:5.1f}" if r["clash_median"] is not None else f"{'—':>5s}"
            print(f"{lab:10s} {label:14s} {r['n_strain']:6d} {r['strain_median']:11.1f} "
                  f"{iqr} {cm} {cd}")
        print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
