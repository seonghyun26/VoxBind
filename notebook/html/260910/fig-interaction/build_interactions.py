#!/usr/bin/env python3
"""build_interactions.py — protein–ligand interaction fingerprints, as violins.

    interaction_violin_{core,all}.*   one panel per interaction type, one violin per arm
    interactions.{json,csv}           the numbers behind them, plus the pooled totals

The VoxBind paper's Fig. 14 form, and the fourth pose-quality measure of the 260910
section: strain and clashes are `../fig-posecheck`, PoseBusters validity is
`../fig-posebusters`, rigid-fragment consistency is `../fig-consistency`, and this is what
the pose actually DOES against the pocket. Same 79 pockets, same arms, same colours — this
folder shares `../pose_common.py` and `../method_colors.py` with those three.

THE METRIC. An interaction fingerprint records, per pose, which contacts it makes with the
receptor. PoseCheck runs ProLIF over the protonated pocket and the pose as generated; see
Harris et al., 2023 (the PoseCheck paper) for the definitions. Four types appear in this
data:

    VdWContact    van der Waals contact — by far the most common, and mostly a size proxy
    Hydrophobic   apolar contact
    HBAcceptor    the ligand accepts a hydrogen bond from the protein
    HBDonor       the ligand donates one to the protein

MORE IS NOT AUTOMATICALLY BETTER, and this figure is not a leaderboard. A bigger ligand
makes more of every type, so an arm that generates larger molecules scores higher on all
four without binding better; the crystal ligands are the reference precisely because they
are the distribution a real binder draws from, and the useful reading is which arm sits
CLOSEST to them, not which sits highest. The per-molecule counts are exported so a
size-matched comparison can be made — the same trap the Vina numbers have already been
bitten by on this project, on a different metric.

A TYPE A MOLECULE DOES NOT MAKE IS ABSENT FROM ITS FINGERPRINT, NOT ZERO IN IT. Every
molecule scored by PoseCheck therefore contributes a 0 to every type it did not make, and
the violins are over all scored molecules, not over the ones that happened to make that
type. Counting only the latter would turn "Ours makes H-bond donors less often" into "Ours
makes more of them when it does", which is a different and much smaller claim.

THE POCKET IS THE pocket10 CROP that the target `metrics.json` files carry, the same source
`../fig-posecheck/build_posecheck_per_atom.py` uses — NOT the whole-receptor scoring in
`frozenenc_probes/posecheck_full/`. That scoring exists and covers all 79 pockets for all
three arms, but it has `reference: null`, so it cannot draw the crystal ligands, which are
the whole point of the comparison. The cost of using the crop was measured over the 21,865
molecules the two scorings share rather than assumed: per molecule the crop finds 0.16-0.29
fewer VdWContacts (1.8-3.2 %) and within 0.03 of the same count on all three other types.
A 10 Å crop around the crystal ligand contains essentially everything within ProLIF's
interaction cutoffs. Do not, however, read a number here against one from
`../fig-posecheck/build_posecheck_all_by_atom_range.py`, which IS whole-receptor.

STYLE. `../pose_common.py`'s furniture, and the violin construction of
`../fig-posecheck/build_posecheck_all_by_atom_range.py`: KDE body, a thick inter-quartile
bar, a white median dot. The type has nowhere to live inside the frame, so it names the y
axis rather than becoming a panel title.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-interaction/build_interactions.py
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

# key -> panel name. Ordered by how common the type is, so the panel that is mostly a size
# proxy is read first and the specific ones after it.
TYPES = [
    ("VdWContact",  "van der Waals contacts"),
    ("Hydrophobic", "Hydrophobic contacts"),
    ("HBAcceptor",  "H-bonds accepted by the ligand"),
    ("HBDonor",     "H-bonds donated by the ligand"),
]
TOTAL = "All interactions"
Y_LABEL = "Interactions per pose"

# THE TYPE IS A PANEL TITLE HERE, which the 260910 house style otherwise does not use. All
# four panels plot the same quantity in the same unit, so the y axis cannot carry the type
# as well -- and the names are too long to sit rotated beside a half-height panel without
# being clipped. This is the exception the binned PoseCheck figures already make, for the
# same reason: the label has nowhere else to live inside the frame.
TITLE_SIZE = 13.5

# GEOMETRY. Two by two off the shared panel size: four types is one too many for a row at
# this width, and a 2x2 keeps every violin wide enough to read its quartile bar.
FIG_W = pc.FIG_W * 1.24
FIG_H = pc.PANEL_H * 1.86
VIOLIN_W = 0.78
BODY_ALPHA = 0.55
IQR_LW, MED_MS = 5.0, 5.5
# Head-room above the tallest violin for the per-violin mean/median caption.
HEAD = 1.34
# ONE ABSOLUTE KDE BANDWIDTH FOR EVERY VIOLIN, in counts, rather than matplotlib's default.
# The default (Scott) scales the bandwidth by n^(-1/5), and the crystal ligands are 79
# molecules against each arm's ~7,800 -- so the reference came out visibly smoother than
# every arm beside it for no reason but its sample size, which on a shape comparison is
# exactly the thing that must not differ. Fixed at 0.45 of a count, each integer still
# reads as its own bump.
KDE_BW = 0.45


def series(arms):
    """(label, colour, values-getter key) with the crystal ligands FIRST. Reference-first,
    not reference-last: every other figure in 260910 draws it as the baseline the models
    are read against, and on a categorical axis that means the leftmost slot."""
    return [(pc.REF_LABEL, pc.REF_COLOR, None)] + \
           [(lab, color(lab), key) for lab, key, _ in arms]


def counts(rows, kind):
    """Per-molecule count of one interaction type over every molecule with a fingerprint.
    A type the molecule did not make is absent from the dict and counts as 0 — see the
    module docstring; this is the line that decides it."""
    if kind == TOTAL:
        return [sum(r["ifp"].values()) for r in rows if r["ifp"] is not None]
    return [r["ifp"].get(kind, 0) for r in rows if r["ifp"] is not None]


def tick_label(label):
    """`VoxBind + Ours` is too wide for a 2x2 panel's slot; break it rather than rotate,
    which would cost a line of figure height for every panel."""
    return {"VoxBind + Ours": "VoxBind\n+ Ours", "Reference ligand": "Reference\nligand"}\
        .get(label, label)


def violin_panel(ax, vals, cols, labels, title, ylabel):
    ax.set_facecolor("white")
    parts = ax.violinplot(vals, showextrema=False, widths=VIOLIN_W,
                          bw_method=lambda k: KDE_BW / np.std(k.dataset))
    for body, col in zip(parts["bodies"], cols):
        body.set(facecolor=col, alpha=BODY_ALPHA, edgecolor=col, linewidth=1.2)
    for i, v in enumerate(vals, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color=pc.INK, lw=IQR_LW, zorder=3)
        ax.plot(i, med, "o", color="white", ms=MED_MS, zorder=4)

    # The top is the largest count anyone actually made, plus head-room for the caption.
    # NOT a percentile: cutting at p99.5 sliced the KDE tails off mid-body and left a row
    # of hard vertical edges along the top of the panel that read as real structure.
    # A count cannot be negative either, but a KDE fitted to a zero-heavy distribution puts
    # a tail below zero anyway; the axis cuts it rather than implying a count of -1 exists.
    top = max(max(v) for v in vals) * HEAD
    ax.set_ylim(bottom=-0.02 * top, top=top)
    pc.furniture(ax, ylabel=ylabel, xlim=(0.4, len(vals) + 0.6), xloc=None)
    ax.set_title(title, fontsize=TITLE_SIZE, color=pc.INK, loc="left", pad=8)
    ax.set_xticks(range(1, len(vals) + 1))
    ax.set_xticklabels([tick_label(x) for x in labels], fontsize=11.5, linespacing=1.25)
    ax.grid(False, axis="x")          # the x is categorical: a rule per slot is a fence
    for i, v in enumerate(vals, start=1):
        ax.text(i, top, f"med {np.median(v):.0f}\nmean {np.mean(v):.2f}", ha="center",
                va="top", fontsize=10, color=pc.AXIS, linespacing=1.3, zorder=6)


def figure(rows_by_key, refrows, arms, variant):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    ser = series(arms)
    for i, (ax, (kind, title)) in enumerate(zip(axes.ravel(), TYPES)):
        vals = [counts(refrows if key is None else rows_by_key[key], kind)
                for _, _, key in ser]
        # The y name goes on the left column only: all four panels share the quantity and
        # the unit, so repeating it on the right merely narrows those panels.
        violin_panel(ax, vals, [c for _, c, _ in ser], [l for l, _, _ in ser], title,
                     Y_LABEL if i % 2 == 0 else None)
    pc.fit(fig, pad=0.5, h_pad=1.4)
    pc.save(fig, HERE, "interaction_violin", variant)


# ── exports ──────────────────────────────────────────────────────────────────────
def block(rows):
    out = {"n_molecules": len(rows),
           "n_scored": sum(r["ifp"] is not None for r in rows),
           "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None}
    for kind, _ in TYPES + [(TOTAL, None)]:
        v = counts(rows, kind)
        out[kind] = {"mean": round(float(np.mean(v)), 3), "median": float(np.median(v)),
                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                     "share_nonzero": round(100 * float(np.mean(np.asarray(v) > 0)), 2)}
    return out


def exports(data, p79_rows, refrows):
    summary = {
        "built": "2026-09-09",
        "metric": "PoseCheck / ProLIF interaction fingerprint (Harris et al., 2023)",
        "receptor_scope": ("pocket10 crop, as the target metrics.json carries it — the same "
                           "source as ../fig-posecheck/build_posecheck_per_atom.py"),
        "zero_handling": ("a type absent from a molecule's fingerprint counts as 0, over "
                          "every scored molecule"),
        "caution": ("more is not better — every type scales with ligand size, so read each "
                    "arm against the crystal ligands, not against the arm above it"),
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79)},
        "types": [k for k, _ in TYPES],
        "arms": {}, "reference_ligand": block(refrows),
    }
    for lab, key, root in pc.ARMS:
        summary["arms"][lab] = {
            "root": root, "key": key,
            "p79": block(p79_rows[key]),
            "all_pockets": block([r for t in sorted(data[key]) for r in data[key][t]]),
        }
    json.dump(summary, open(os.path.join(HERE, "interactions.json"), "w"),
              indent=1, ensure_ascii=False)

    kinds = [k for k, _ in TYPES] + [TOTAL]
    with open(os.path.join(HERE, "interactions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "type", "mean", "median", "q25", "q75", "share_nonzero_pct",
                    "n_scored"])
        rows = [(lab, summary["arms"][lab]["p79"]) for lab, _, _ in pc.ARMS]
        rows.append((pc.REF_LABEL, summary["reference_ligand"]))
        for lab, b in rows:
            for k in kinds:
                w.writerow([lab, k, b[k]["mean"], b[k]["median"], b[k]["q25"], b[k]["q75"],
                            b[k]["share_nonzero"], b["n_scored"]])
    return summary


def main():
    pc.use_style()
    data, p79_rows, refrows = pc.load_arms()
    for variant, arms in pc.variants():
        figure(p79_rows, refrows, arms, variant)
    summary = exports(data, p79_rows, refrows)

    kinds = [k for k, _ in TYPES] + [TOTAL]
    print(f"{len(pc.P79)}-pocket set · interaction fingerprint, pocket10 crop · "
          f"mean per molecule\n")
    print(f"{'arm':16s} {'atoms':>6s} {'mols':>6s} " + " ".join(f"{k:>12s}" for k in kinds))
    rows = [(lab, summary["arms"][lab]["p79"]) for lab, _, _ in pc.ARMS]
    rows.append((pc.REF_LABEL, summary["reference_ligand"]))
    for lab, b in rows:
        print(f"{lab:16s} {b['atoms_mean']:6.1f} {b['n_scored']:6d} "
              + " ".join(f"{b[k]['mean']:12.2f}" for k in kinds))
    print(f"\n{'share of molecules making the type at all':>30s}")
    print(f"{'arm':16s} {'':6s} {'':6s} " + " ".join(f"{k:>12s}" for k in kinds))
    for lab, b in rows:
        print(f"{lab:16s} {'':6s} {'':6s} "
              + " ".join(f"{b[k]['share_nonzero']:11.1f}%" for k in kinds))
    print("\nEvery type scales with ligand size (see the atoms column): read each arm "
          "against\nthe crystal ligands, not against the arm above it.")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
