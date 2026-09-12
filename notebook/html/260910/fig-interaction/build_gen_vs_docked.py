#!/usr/bin/env python3
"""build_gen_vs_docked.py — the same molecule as generated and as redocked, paired.

    interaction_pair_contacts_core.*  van der Waals and hydrophobic, as violins
    interaction_pair_hbonds_core.*    H-bonds accepted and donated, as letter-value blocks
    interactions_pair.csv             the numbers behind them, and the change between them

VoxBind's Fig. 14 draws every interaction type twice: the pose AS GENERATED and the same
molecule REDOCKED. `build_interactions.py` draws the first only, because nothing in the
pipeline keeps a docked conformer — see ./README.md. `dock_core_poses.py` produces the
second for the core three arms and `compute_baseline_interactions.py` fingerprints both,
and this draws the pair.

WHAT THE PAIR IS FOR. Redocking asks a different question from generating: not "what did
the model put in the pocket" but "what does this molecule do in the pocket when a docking
program is allowed to place it". A model whose generated poses already sit where Vina would
put them has little to gain from redocking; one whose poses are merely plausible-looking
gains a lot. The gap between a method's two columns is that quantity, and it is only
readable because THE TWO COLUMNS ARE THE SAME MOLECULES: `dock_core_poses.py` drops a
molecule from both series when its dock fails, so nothing separates the columns except the
docking. 2,105 molecules over 78 pockets — see ./README.md for the 28 that did not pair.

COLOUR STAYS THE METHOD'S, which is where this figure departs from Fig. 14 on purpose.
Fig. 14 gives each series its own colour ramp (green #72b6a1 -> #eaf4f1 for generated,
salmon #e99675 -> #fcefea for redocked) and puts the methods on the x axis. It can afford
that because the series is the only other thing in the panel. Here colour means the method
in every figure of the 260910 section — four folders of it — and a reader who has learned
that palette should not have to unlearn it for one panel. So the pair takes the two
channels Fig. 14 leaves free: ADJACENCY, which puts the comparison inside one eyeful, and
a HATCH on the redocked column.

AND THE METHOD NAMES COME BACK TO THE X AXIS. `build_interactions.py` drops them because
eight of them, rotated to fit, cost a third of its height; three pairs on a 4:1 frame have
room for them upright, and spending the legend on the method as well would leave the pose
condition — the whole point of this figure — as a footnote in a five-entry key. Each
channel gets its own place: the method on the axis, the condition in the legend.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-interaction/build_gen_vs_docked.py
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402
import build_interactions as bi                                      # noqa: E402

# The arms dock_core_poses.py staged, in the drug-design table's row order. These are the
# labels interactions_<Arm>-{gen,docked}.json is written under — the export spelling, not
# the figure's; bi.display() maps them at draw time exactly as the sibling figures do.
ARMS = ["Reference", "VoxBind", "CoDE"]
SERIES = [("gen", "Generated"), ("docked", "Redocked")]

# The reference's colour is keyed under pose_common's canonical label, not under the
# "Reference" the staged JSON is named with.
COLOR_KEY = {"Reference": pc.REF_LABEL}

# GEOMETRY. The sibling figures' frame exactly — same aspect, same panel height, same
# legend strip — so the three figures of this folder stack in a document without a size
# change between them. Only the x spacing is this figure's own.
PAIR_GAP = 0.30                  # between the two columns of one method
GROUP_GAP = 1.15                 # between one method and the next
HATCH = "////"
HATCH_LW = 0.6
# Neutral grey for the two condition swatches: the legend here names the POSE, not the
# method, and giving it one of the method colours would say the opposite.
SWATCH = "#9aa0a8"


def positions():
    """x for each (arm, series), pairs tight and methods apart."""
    xs, x = [], 1.0
    for _ in ARMS:
        xs += [x, x + PAIR_GAP + bi.VIOLIN_W]
        x = xs[-1] + GROUP_GAP
    return xs


def load():
    """{(arm, series): rows} from the staged scorings, or an explanatory exit."""
    out = {}
    for arm in ARMS:
        for key, _ in SERIES:
            path = os.path.join(HERE, f"interactions_{arm}-{key}.json")
            if not os.path.exists(path):
                raise SystemExit(
                    f"no {os.path.basename(path)} — run dock_core_poses.py to stage the "
                    f"poses, then compute_baseline_interactions.py over them (see "
                    f"./README.md), or pull them with `bash dropbox_sync.sh pull`")
            out[arm, key] = [{"n": m["n"], "ifp": m["ifp"]}
                             for m in json.load(open(path))["molecules"]]
    return out


def paired_check(rows):
    """The figure's one precondition: a method's two columns must be the same molecules.
    Rows are written in pocket-then-row order by one scorer over one staged layout, so
    equal length and an equal heavy-atom sequence is the pairing — the same check
    stage_baseline_poses.py makes against the export."""
    for arm in ARMS:
        g, d = rows[arm, "gen"], rows[arm, "docked"]
        if len(g) != len(d) or [r["n"] for r in g] != [r["n"] for r in d]:
            raise SystemExit(
                f"{arm}: the generated and redocked series are not the same molecules "
                f"({len(g)} vs {len(d)} rows) — the gap between the columns would be a "
                f"difference in population, not in pose. Re-stage with dock_core_poses.py")


def bodies(ax, vals, cols, xs):
    """Violins at `xs` rather than at 1..n, hatched on the redocked column."""
    parts = ax.violinplot(vals, positions=xs, showextrema=False, widths=bi.VIOLIN_W,
                          bw_method=lambda k: bi.KDE_BW / np.std(k.dataset))
    for i, (body, col) in enumerate(zip(parts["bodies"], cols)):
        body.set(facecolor=col, alpha=bi.BODY_ALPHA, edgecolor=col, linewidth=1.2)
        if i % 2:
            body.set_hatch(HATCH)
    for x, v in zip(xs, vals):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(x, q1, q3, color=pc.INK, lw=bi.IQR_LW, zorder=3)
        ax.plot(x, med, "o", color="white", ms=bi.MED_MS, zorder=4)


def blocks(ax, vals, cols, xs):
    """bi.boxen_panel's letter-value stack at `xs`, hatched on the redocked column."""
    reach = 0.0
    for i, (x, v, col) in enumerate(zip(xs, vals, cols)):
        for lo, hi, d in bi.letter_bands(v):
            w = bi.VIOLIN_W / 2 ** d
            face = bi.tint(col, bi.BOXEN_LIGHT * d / (bi.BOXEN_K - 1)) + (bi.BODY_ALPHA,)
            ax.add_patch(Rectangle((x - w / 2, lo), w, hi - lo, zorder=3 + d,
                                   facecolor=face, edgecolor=pc.AXIS,
                                   linewidth=bi.BOXEN_EDGE_LW,
                                   hatch=HATCH if i % 2 else None))
            reach = max(reach, hi)
        ax.plot(x, np.median(v), "o", color="white", ms=bi.MED_MS, zorder=3 + bi.BOXEN_K)
    return reach


def panel(ax, kind, rows, xs):
    ylabel, form = bi.TYPE[kind]
    vals = [bi.counts(rows[arm, key], kind) for arm in ARMS for key, _ in SERIES]
    cols = [color(COLOR_KEY.get(arm, arm)) for arm in ARMS for _ in SERIES]
    ax.set_facecolor("white")

    if form == "violin":
        bodies(ax, vals, cols, xs)
        reach = max(max(v) for v in vals)
        cells = None
    else:
        reach = blocks(ax, vals, cols, xs)
        cells = reach

    # bi.frame lays out the shared furniture, then the x is rebuilt: its limits come from
    # these positions rather than from a count of columns, and the category labels go back
    # on -- one per PAIR, centred between the two columns. See the module docstring.
    bi.frame(ax, vals, ylabel, reach * bi.HEAD_BARE, caption=False, cells=cells)
    ax.set_xlim(xs[0] - bi.VIOLIN_W, xs[-1] + bi.VIOLIN_W)
    ax.set_xticks([(xs[i] + xs[i + 1]) / 2 for i in range(0, len(xs), 2)])
    ax.set_xticklabels([bi.display(COLOR_KEY.get(a, a)) for a in ARMS],
                       fontsize=bi.LEGEND_SIZE, color=pc.INK)
    ax.tick_params(axis="x", length=0, pad=6)
    return form


def legend_beneath(fig, form):
    """Two swatches, and they name the POSE. The method is on the x axis (see the module
    docstring), so this key carries only the thing the pair is about, drawn the way the
    panel draws it: plain for generated, hatched for redocked, in a neutral grey that
    cannot be mistaken for one of the method colours."""
    edge = SWATCH if form == "violin" else pc.AXIS
    lw = 1.2 if form == "violin" else bi.BOXEN_EDGE_LW
    handles = [Patch(facecolor=to_rgb(SWATCH) + (bi.BODY_ALPHA,), edgecolor=edge, lw=lw,
                     hatch=(HATCH if key == "docked" else None), label=name)
               for key, name in SERIES]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                     ncol=len(handles), frameon=True, facecolor="white",
                     edgecolor=pc.LEGEND_EDGE, framealpha=1.0, borderpad=0.5,
                     fontsize=bi.LEGEND_SIZE, handlelength=1.5, handleheight=1.0,
                     handletextpad=0.6, columnspacing=1.9, labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(pc.AXIS_LW)
    for t in leg.get_texts():
        t.set_color(pc.INK)
    return leg


def figure(part, kinds, rows):
    xs = positions()
    fig, axes = plt.subplots(1, 2, figsize=(bi.FIG_W, bi.FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        form = panel(ax, kind, rows, xs)
    pc.fit(fig, pad=0.5, w_pad=2.2, rect=(0, bi.LEGEND_H / bi.FIG_H, 1, 1))
    legend_beneath(fig, form)
    # `core` is in the filename because it is the only variant this figure has and the
    # folder's rule is that the variant is never only in the content: the redocking covers
    # the core three arms, since the published baselines' poses would each need their own
    # ~700 docks and the comparison this section is making is between ours and VoxBind's.
    pc.save(fig, HERE, f"interaction_pair_{part}", "core")


def exports(rows):
    path = os.path.join(HERE, "interactions_pair.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "type", "n_molecules", "gen_mean", "docked_mean", "delta_mean",
                    "gen_median", "docked_median", "share_nonzero_gen_pct",
                    "share_nonzero_docked_pct"])
        for arm in ARMS:
            for kind in bi.KINDS:
                g = np.asarray(bi.counts(rows[arm, "gen"], kind), float)
                d = np.asarray(bi.counts(rows[arm, "docked"], kind), float)
                w.writerow([arm, kind, len(g), round(g.mean(), 3), round(d.mean(), 3),
                            round(d.mean() - g.mean(), 3), np.median(g), np.median(d),
                            round(100 * (g > 0).mean(), 2), round(100 * (d > 0).mean(), 2)])
    return path


def main():
    pc.use_style()
    rows = load()
    paired_check(rows)
    for part, kinds in bi.PARTS:
        figure(part, kinds, rows)
    path = exports(rows)

    print(f"paired generated / redocked poses · {len(rows['CoDE', 'gen'])} molecules for "
          f"the model arms, {len(rows['Reference', 'gen'])} crystal ligands\n")
    head = " ".join(f"{k:>21s}" for k in bi.KINDS)
    print(f"{'arm':12s} {head}")
    for arm in ARMS:
        cells = []
        for kind in bi.KINDS:
            g = np.mean(bi.counts(rows[arm, "gen"], kind))
            d = np.mean(bi.counts(rows[arm, "docked"], kind))
            cells.append(f"{g:7.2f} ->{d:6.2f} {d - g:+5.2f}")
        print(f"{bi.display(COLOR_KEY.get(arm, arm)):12s} " + " ".join(cells))
    print("\nmean per molecule, generated -> redocked and the change. The crystal ligands "
          "are\nthe control: they are already in a measured pose, so redocking should move "
          "them\nleast.")
    print(f"\nwrote {path}\nwrote {HERE}")


if __name__ == "__main__":
    main()
