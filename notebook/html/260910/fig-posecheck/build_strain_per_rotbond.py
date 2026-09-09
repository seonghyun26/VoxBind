#!/usr/bin/env python3
"""build_strain_per_rotbond.py — PoseCheck strain against the number of ROTATABLE BONDS.

    strain_box_per_rotbond_{core,all}.*      THE FIGURE TO READ — the distribution at
                                             each rotatable-bond count, as boxes
    strain_per_rotbond_median_{core,all}.*   the medians alone, as a line
    strain_per_rotbond_mean_{core,all}.*     the same line, mean — a separate figure
    strain_per_rotbond.{json,csv}            the numbers behind all of them

The VoxBind paper's Fig. 13 form. `build_posecheck_per_atom.py` next door draws the same
metric, the same arms and the same 79 pockets against ligand SIZE; this swaps the x axis
for the count of rotatable bonds and changes nothing else — same loader, same statistics,
same house furniture, same colours. A point here and a point there are the same molecules
grouped two ways.

WHY THE AXIS IS WORTH SWAPPING. Strain is conformational: it is the energy a pose carries
because its torsions are not where the force field would put them, so the number of
torsions a molecule HAS is the more direct explanatory variable, and heavy-atom count is
its proxy. The two are not interchangeable — a fused polycyclic and a long-chain ligand of
equal size have very different torsional freedom — and the ranking of the arms is allowed
to differ between the two views.

ROTATABLE BONDS ARE COUNTED FROM THE SMILES that `metrics.json` already records, with
RDKit's default (strict) `CalcNumRotatableBonds`, which excludes amides, terminal bonds and
ring bonds. It is a topological descriptor, so reading it off the recorded SMILES rather
than the pose gives the same integer with no risk of a sample-to-SDF index slip. Molecules
whose SMILES will not parse are dropped and counted in the run log; there were none in the
three local arms when this was written.

THE CRYSTAL REFERENCE WINDOW IS ±1 HERE, NOT pose_common's ±4. There is one crystal ligand
per pocket, so a per-count reference curve over 79 ligands would be noise — that is why
pose_common windows it at all. But rotatable-bond counts run 0–14 where ligand sizes run
5–45, and a ±4 window there spans two thirds of the axis and would flatten the reference
into a near-constant line. ±1 pools 12–29 ligands per point and the line stops at 9, where
the window falls under MIN_REF, rather than being extended into an invented value. This
follows ../fig-consistency/build_rigid_fragment.py, which narrows the same window for the
same reason.

THE BOXES ARE THE PRIMARY FIGURE AND THE LINES ARE KEPT BESIDE THEM. A median line says
where an arm sits; it cannot say whether two arms a factor of 1.5 apart are actually
separated, and on a metric whose distribution spans four decades inside a single bond count
that is the question. The boxes answer it — and they show something the lines cannot: the
arms' inter-quartile ranges overlap almost completely at every count, so the median
ordering is a shift of a wide distribution, not a separation of two narrow ones. The lines
stay because they are what a reader compares against the per-atom figure next door, and
because the mean only exists there (see below).

WHISKERS ARE THE 5TH AND 95TH PERCENTILES AND FLIERS ARE NOT DRAWN. Not a cosmetic choice:
4-7% of molecules relax to 1e4-1e13 (`strain_gt_1e4` in the JSON), so Tukey whiskers with
fliers would put single points nine decades above the boxes and squash every box in the
figure into a line. The percentile whisker is stated on the y axis, and the tail it leaves
out is reported as a number rather than drawn.

NO MEAN MARKER ON THE BOXES. `showmeans` was tried and cannot work here: within one arm and
one bond count the mean sits at 1e4-1e11 while the box sits near 1e2, so every marker lands
far above its own box and drags the axis with it. The mean has its own line figure, where
being unreadable at least reads as the finding it is.

MEAN AND MEDIAN ARE SEPARATE LINE FIGURES, as everywhere in 260910 — see the per-atom builder's
docstring for why strain in particular cannot share a panel between the two. THE MEDIAN IS
THE FIGURE TO READ; the mean is here so that claim can be checked, not as an alternative.

AND THE MEAN PANEL IS NOT CLIPPED, where the per-atom one is. That figure clips to 1e6
because its curves have a bulk down there and only a few thin heavy-atom counts spike out
of it. This axis has no such bulk: 15 rotatable-bond counts pool 500-1,100 molecules each,
so nearly every point catches one of the conformers that relax to 1e8-1e13, and the same
clip left the curve as disconnected fragments with most of it above the panel. Drawn whole
it spans nine decades, sits 6-9 decades above the crystal ligands and has no ordering at
all — which is the honest picture of what a mean does to this metric, and it is the
argument for the median rather than something to hide behind an axis limit. The size of the
tail driving it is reported per arm in the run log and in the JSON, as `strain_gt_1e4`.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posecheck/build_strain_per_rotbond.py       # everything
    ... build_strain_per_rotbond.py --kind box         # only the boxes
    ... build_strain_per_rotbond.py --kind line        # only the median and mean lines
    ... build_strain_per_rotbond.py --stat median      # lines: only the median one
"""
import argparse
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

from rdkit import Chem, RDLogger                                     # noqa: E402
from rdkit.Chem import rdMolDescriptors                              # noqa: E402

RDLogger.DisableLog("rdApp.*")

# See the docstring: narrower than pose_common's ±4 because this axis is a third as long.
# Set before anything is drawn, because pc.reference_curve reads these off the module.
pc.REF_WIN, pc.MIN_REF = 1, 12

X_LABEL = "Number of rotatable bonds in ligand"
XTICK = 1
# Above this a UFF relaxation has effectively failed rather than found a lower conformer
# (the converged bulk sits under ~1e3). Nothing is filtered on it; it only names the tail
# that makes the mean unreadable, so the mean figure can be read for what it is.
TAIL = 1e4
# Boxes: the ECDF pair's canvas width, the share of each count's slot left as white space
# between groups, and the whisker percentiles (see the docstring — Tukey + fliers is not an
# option on a metric with a 1e13 tail).
BOX_WIDE, BOX_GAP, WHIS = 1.52, 0.28, (5, 95)
# Strain is floored before boxing, at the value strain_clash_ecdf_pair already floors it to.
# A rigid ligand can relax to ~0, and on a log axis a single 1e-11 at 0 rotatable bonds
# pulled the panel down through fifteen decades and flattened every box in it. Values are
# CLIPPED, not dropped, so the whisker rests on the floor and the count is unchanged.
STRAIN_FLOOR = 1e-2
STATS = {"median": lambda v: float(np.median(v)),
         "mean": lambda v: float(np.mean(v))}
Y_LABEL = "Strain {stat}\n(kcal mol⁻¹)"
# The mean curve sits high and rises; the median curve sits low and rises. Each legend goes
# in the corner its own curves leave empty.
LEGEND_LOC = {"median": "upper left", "mean": "lower right"}

_RB_CACHE = {}


def rot_bonds(smiles):
    """RDKit's strict rotatable-bond count, or None if the SMILES will not parse. Cached:
    the three arms carry ~24k molecules and many repeat."""
    if not smiles:
        return None
    if smiles not in _RB_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _RB_CACHE[smiles] = (None if mol is None
                             else int(rdMolDescriptors.CalcNumRotatableBonds(mol)))
    return _RB_CACHE[smiles]


def attach(rows):
    """Add `rb` to each row in place, and report how many rows could not get one."""
    bad = 0
    for r in rows:
        r["rb"] = rot_bonds(r.get("smi"))
        bad += r["rb"] is None
    return bad


# ── figure ───────────────────────────────────────────────────────────────────────
def strain_panel(xs, arms, variant, stat, per_arm, ref_per):
    """One statistic, one panel, one file — the per-atom builder's layout with this axis."""
    f = STATS[stat]
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, pc.reference_curve(ref_per, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
            ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = pc.model_curve(per_arm[key], xs, f)
        ax.plot(xs, y, color=color(lab), lw=pc.MODEL_LW, zorder=5, solid_capstyle="round")

    ax.set_yscale("log")
    pc.furniture(ax, ylabel=Y_LABEL.format(stat=stat), xlabel=X_LABEL,
                 xlim=(xs[0] - 0.35, xs[-1] + 0.35), xloc=XTICK)
    pc.legend(ax, pc.arm_handles(arms), loc=LEGEND_LOC[stat], fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, f"strain_per_rotbond_{stat}", variant)


def strain_boxes(xs, arms, variant, per_arm, ref_per):
    """The distribution at each rotatable-bond count, one box per arm, dodged within the
    count. Wider canvas than the line panels (the ECDF pair's width, already in the house)
    because this draws 15 counts x 2-3 arms of boxes on one axis."""
    fig, ax = plt.subplots(figsize=(pc.FIG_W * BOX_WIDE, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Dodge: n boxes centred on the count, leaving BOX_GAP of the slot as white space so
    # neighbouring counts stay visually separate groups.
    n = len(arms)
    slot = (1.0 - BOX_GAP) / n
    for i, (lab, key, _) in enumerate(arms):
        offs = (i - (n - 1) / 2) * slot
        vals = [np.clip(per_arm[key].get(x, []), STRAIN_FLOOR, None) for x in xs]
        col = color(lab)
        bp = ax.boxplot(vals, positions=[x + offs for x in xs], widths=slot * 0.86,
                        whis=WHIS, showfliers=False, patch_artist=True, zorder=5,
                        manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=1.05)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=col, linewidth=1.05)
        for med in bp["medians"]:
            med.set(color=pc.INK, linewidth=1.5, solid_capstyle="butt")

    ax.plot(xs, pc.reference_curve(ref_per, xs, STATS["median"]), color=pc.REF_COLOR,
            lw=pc.REF_LW, ls=pc.DASH, zorder=6, dash_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(bottom=STRAIN_FLOOR)
    pc.furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{WHIS[0]:g}–{WHIS[1]:g}th pct whiskers",
                 xlabel=X_LABEL, xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=XTICK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    pc.legend(ax, pc.arm_handles(arms), loc="upper left", fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "strain_box_per_rotbond", variant)


# ── exports ──────────────────────────────────────────────────────────────────────
def ref_n(per, xs):
    """The reference's n is the WINDOWED pool, not the exact-count one: the plotted value
    comes from ±REF_WIN, so an n beside it that counted only the exact bond count would
    read as 4 ligands supporting a point that 16 produced."""
    return [sum(len(v) for x2, v in per.items() if abs(x2 - x) <= pc.REF_WIN) for x in xs]


def tail_share(per):
    """% of an arm's molecules whose relaxation ran past TAIL — the tail that decides its
    mean curve, and the reason the mean curve is not a location statistic."""
    vals = [v for vs in per.values() for v in vs]
    return round(100 * sum(v > TAIL for v in vals) / len(vals), 2) if vals else None


def exports(xs, per_arm, ref_per, stats):
    """The curves themselves plus the per-count molecule counts, so a reader can check a
    figure without re-running it — and so the rotatable-bond DISTRIBUTION is on the record.
    It is not drawn as a strip here (the per-atom figure this mirrors has no strip either),
    but the arms differ in it, and any claim about a curve has to be read against `n`."""
    out = {
        "built": "2026-09-09",
        "metric": "PoseCheck strain (posecheck 1.3.1 definition) against rotatable bonds",
        "x": "RDKit CalcNumRotatableBonds, strict, from the SMILES in metrics.json",
        "reference_note": (f"crystal ligands over a centred ±{pc.REF_WIN}-bond window, "
                           f"≥{pc.MIN_REF} ligands; null where the window is thinner. Its "
                           f"`n` is that window's pool, not the exact-count one"),
        "min_molecules_per_point": pc.MIN_N,
        "n_pockets": len(pc.P79),
        "rotatable_bonds": xs,
        "arms": {},
    }
    for lab, key, _ in pc.ARMS:
        out["arms"][lab] = {"n": [len(per_arm[key].get(x, ())) for x in xs],
                            "n_scored": sum(len(v) for v in per_arm[key].values()),
                            "strain_gt_1e4": tail_share(per_arm[key]),
                            **{f"strain_{s}": pc.model_curve(per_arm[key], xs, STATS[s])
                               for s in stats}}
    out["arms"][pc.REF_LABEL] = {
        "n": ref_n(ref_per, xs),
        "n_scored": sum(len(v) for v in ref_per.values()),
        "strain_gt_1e4": tail_share(ref_per),
        **{f"strain_{s}": pc.reference_curve(ref_per, xs, STATS[s]) for s in stats}}
    json.dump(out, open(os.path.join(HERE, "strain_per_rotbond.json"), "w"),
              indent=1, ensure_ascii=False)

    fields = [f"strain_{s}" for s in stats]
    with open(os.path.join(HERE, "strain_per_rotbond.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "rotatable_bonds", "n"] + fields)
        for arm, d in out["arms"].items():
            for i, x in enumerate(xs):
                w.writerow([arm, x, d["n"][i]]
                           + [("" if d[f][i] is None else round(d[f][i], 3))
                              for f in fields])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stat", choices=("median", "mean", "both"), default="both",
                    help="which LINE figure to draw; the exports always carry both")
    ap.add_argument("--kind", choices=("box", "line", "both"), default="both",
                    help="boxes, the median/mean lines, or both (default)")
    args = ap.parse_args()
    stats = ("median", "mean") if args.stat == "both" else (args.stat,)
    kinds = ("box", "line") if args.kind == "both" else (args.kind,)

    pc.use_style()
    data, p79_rows, refrows = pc.load_arms()
    bad = sum(attach(rows) for rows in p79_rows.values()) + attach(refrows)

    per_arm = {key: pc.by_size(p79_rows[key], "s", key="rb") for _, key, _ in pc.ARMS}
    ref_per = pc.by_size(refrows, "s", key="rb")

    ranges = {}
    for variant, arms in pc.variants():
        xs = pc.x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        ranges[variant] = xs
        if "box" in kinds:
            strain_boxes(xs, arms, variant, per_arm, ref_per)
        if "line" in kinds:
            for stat in stats:
                strain_panel(xs, arms, variant, stat, per_arm, ref_per)
    summary = exports(ranges["all"], per_arm, ref_per, ("median", "mean"))

    print(f"{len(pc.P79)}-pocket set · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]} rotatable bonds" for v, xs in ranges.items())
          + f" (counts where every drawn arm has ≥{pc.MIN_N} molecules)")
    if bad:
        print(f"{bad} molecules dropped: SMILES would not parse")
    print()
    header = [0, 2, 4, 6, 8, 10, 12]
    xs_all = summary["rotatable_bonds"]
    print("strain MEDIAN at a given rotatable-bond count, and the tail behind the mean\n")
    print(f"{'arm':16s} {'mols':>7s} " + " ".join(f"{'rb=' + str(h):>7s}" for h in header)
          + f" {'>1e4':>7s}")
    for lab, d in summary["arms"].items():
        cells = []
        for h in header:
            v = summary["arms"][lab]["strain_median"][xs_all.index(h)] \
                if h in xs_all else None
            cells.append(f"{v:7.1f}" if v is not None else f"{'—':>7s}")
        print(f"{lab:16s} {d['n_scored']:7d} " + " ".join(cells)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print(f"\nThe mean figure is drawn unclipped and spans ~9 decades: the tail above is "
          f"what\nputs it there. Read the median.\n\nwrote {HERE}")


if __name__ == "__main__":
    main()
