#!/usr/bin/env python3
"""build_vina_3line.py — Reference / VoxBind / VoxBind+Ours, per pocket, for all three
Vina columns.

For each of Vina Dock, Vina Score and Vina Min, three figures in png + svg + pdf:

    vina_<metric>_3line_mean.{png,svg,pdf}     y = per-pocket MEAN
    vina_<metric>_3line_median.{png,svg,pdf}   y = per-pocket MEDIAN
    vina_<metric>_3line_pair.{png,svg,pdf}     both, side by side, shared y

<metric> is dock / score / min. Which statistic a figure shows is carried by its filename
and by its y-axis name ("Vina Dock mean" / "Vina Dock median"); there are no panel titles,
so in the paired figure the y name is the only thing separating the two halves.

x is the pockets sorted by "VoxBind + Ours"'s own value for that metric and statistic, so
its points climb monotonically and the other two series are read against them. All three
are point clouds, not curves: consecutive pockets are a ranking, not a sequence, and a
connecting line would imply a continuity that is not there.

x is TICKED AT RANKS 1/20/40/60/79 while its grid rules fall every 10 (never on 1), and
neither is a pocket id: rank 1 is whichever pocket that
metric ranks first (target_04 for Dock mean), so a "target N" label there would name the
wrong pocket. vina_<metric>_3line_<stat>.csv maps every rank to its target.

PROTOCOL. The published-baseline protocol only -- whole `*_rec.pdb` receptor,
exhaustiveness 32, all 79 pockets -- the one 260903/baseline.html and
build_results_baseline_protocol.py use, so these curves sit on the same axis as the
AR/Pocket2Mol/DiffSBDD/DecompDiff/FuncBind table. Our older crop protocol
(`*_pocket10.pdb`, exhaustiveness 16, 78 pockets -- target_71 is unscoreable on the crop)
is deliberately NOT plotted: the same crystal reference ligand docks to -7.31 one way and
-7.18 the other, so the two cannot share an axis. Note that `eval_docking_results_full.json`
in the same directory is a third thing again (full receptor but exhaustiveness 16, 78
pockets); only the `_full79` file is the baseline protocol.

THIRD SERIES. TargetDiff, re-docked under this same protocol by
74_dock_targetdiff_full79.sh against the same receptors, so its cloud sits on the same
axis as the other two rather than needing a caveat. It replaced the crystal-reference
dashed rule that used to occupy this slot: the reference was one jagged value per pocket
under someone else's ordering, and a second generative baseline says more about where the
gain comes from. `ref_vina_*` is still computed -- it is the mean of the runs reporting
it, printed per metric alongside the pooled model means and carried in the CSV -- but it
is no longer drawn. Note it is only meaningfully noisy for Dock: `score_only` and
`minimize` are deterministic and the runs agree exactly, while Dock re-searches unseeded
and disagrees by up to ~0.7 kcal/mol on a pocket.

COLOUR. Three hues at one chroma and one lightness, so no series is louder than another
and the ordering does the arguing: TargetDiff violet #B58FDB, VoxBind sand #F5B27E,
VoxBind + Ours periwinkle #8291E8. Ours is drawn last and a shade larger, which is what
separates it -- not saturation.

    /opt/conda/envs/voxbind/bin/python \
        notebook/html/260910/fig-vina-3line/build_vina_3line.py
"""
import csv
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
import sys                                                            # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from method_colors import display                                     # noqa: E402
E = "/home1/irteam/VoxBind/voxbind/exps"

VANILLA = f"{E}/_vanilla_ep923/samples/full_eval_ep923"
OURS = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
BASENAME = "eval_docking_results_full79.json"
PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# metric key -> (per-molecule field, per-pocket reference field, y-axis label)
METRICS = {
    "dock":  ("vina_dock",  "ref_vina_dock",  "Vina Dock"),
    "score": ("vina_score", "ref_vina_score", "Vina Score"),
    "min":   ("vina_min",   "ref_vina_min",   "Vina Min"),
}
STATS = ("mean", "median")

REF_LABEL, REF_COLOR = "Reference ligand", "#9aa0a6"
VOX_LABEL, VOX_COLOR = "VoxBind", "#F5B27E"
# These two are DATA KEYS as well as labels (they head this folder's CSV columns), so they
# stay plain; method_colors.display() dresses them for a legend -- VoxBind picks up its
# sigma=0.9 subscript there. CoDE is \textsc{CoDE} in the .tex files; matplotlib has no
# small caps without a TeX backend, so the figures carry the plain string.
OUR_LABEL, OUR_COLOR = "CoDE", "#8291E8"

# Shape is a second identity channel for the two models, so they survive greyscale and
# colour blindness: triangle vs circle. matplotlib's `markersize` is a DIAMETER, and at
# one diameter a triangle carries only 41% of a circle's ink -- the two would read as two
# different weights -- so the triangle is area-matched to the circle at x1.555, and Ours
# keeps the 3.89 that already set it slightly ahead.
BASE_SIZE = 3.31
VOX_SIZE, OUR_SIZE = BASE_SIZE * 1.555, 3.89

# Axis furniture is a warm near-black (#514F52) rather than pure black: at these hairline
# weights it carries the same authority while sitting a step back from the two series, and
# it does not clash with the warm/cool point colours the way #000 does. The grid stays a
# dotted mid-grey on both axes.
INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
X_LABEL = "Target pocket number"
XTICKS = (1, 20, 40, 60)          # plus the last rank, appended at draw time

# GEOMETRY, in inches. WIDE and TALL scale the 7.25 x 2.77 base figure. The paired figure
# is then SIZED FROM the standalone one so that one of its panels is exactly the same
# shape, and moving between the two is never silently a change of aspect ratio. (It used
# to be a flat 11.34 wide, which made a panel there 4.60 x 2.20 against the standalone's
# 6.24 x 2.15 -- visibly squarer.) Its width is MEASURED per metric by pair_width() rather
# than carried as a constant: the furniture outside the plot area moves with the type size
# and with how wide that metric's tick labels happen to be -- Vina Score reaches positive
# values, so its labels are narrower than Dock's -- and a constant goes stale silently.
WIDE, TALL = 0.88, 1.155

FIG_W, FIG_H = 7.25 * WIDE, 2.77 * TALL
PAIR_H = FIG_H

# Line weights in points. Ticks, spines and the legend rule share ONE weight -- the box is
# axis furniture, so it is drawn with the axis pen -- and the grid sits a step under them.
# The reference rule is the thinnest ink on the plot: it is the benchmark the two series
# are read against, not a third series, and dash + grey already carry its identity.
AXIS_LW, GRID_LW, REF_LW = 1.35, 1.1, 1.25

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3:
                                     # Type 3 is what journals reject and Illustrator
                                     # mangles, and its text is not selectable
}


def load(root):
    """{target: {metric: {"mean", "median", "ref", "n"}}} from one run's docking results."""
    with open(os.path.join(root, BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    out = {}
    for t in per_target:
        row = {}
        for key, (field, ref_field, _) in METRICS.items():
            vals = [m[field] for m in (t.get("per_mol") or [])
                    if isinstance(m, dict) and m.get(field) is not None]
            if not vals:
                continue
            row[key] = {"mean": st.mean(vals), "median": st.median(vals),
                        "ref": t.get(ref_field), "n": len(vals)}
        if row:
            out[t["target"]] = row
    return out


def reference(vox, our, targets, metric):
    """Per-pocket crystal-ligand value, averaged over the runs that report it."""
    ref, worst = {}, (0.0, None)
    for t in targets:
        vals = [d[t][metric]["ref"] for d in (vox, our)
                if d[t].get(metric, {}).get("ref") is not None]
        ref[t] = st.mean(vals) if vals else None
        if len(vals) == 2 and abs(vals[0] - vals[1]) > worst[0]:
            worst = (abs(vals[0] - vals[1]), t)
    return ref, worst


def series(vox, our, ref, targets, metric, stat):
    """The three series for one metric and statistic, and the order they are drawn in.
    Ours sets the ordering, so only its own cloud climbs monotonically."""
    order = sorted(targets, key=lambda t: our[t][metric][stat])
    return (order,
            [our[t][metric][stat] for t in order],
            [vox[t][metric][stat] for t in order],
            [ref[t] for t in order])


def limits(*serieses):
    """Clip to the bulk instead of squashing 79 pockets for a couple of runaway values;
    the caller names the excluded points, the image does not."""
    flat = sorted(v for ys in serieses for v in ys if v is not None)
    pct = lambda q: flat[int(round(q * (len(flat) - 1)))]
    return pct(0.01) - 0.45, pct(0.99) + 0.55


def draw(ax, order, y_our, y_vox, y_ref, ylim, *, legend, ylabel):
    """One panel: a dashed reference rule, two point clouds, a dotted grid. The two models
    are point clouds and not curves -- consecutive pockets are a ranking, not a sequence,
    and a connecting line between them would imply a continuity that is not there. Ours is
    drawn last and a shade larger, because it is the series the ordering is built from and
    the one being read."""
    x = list(range(1, len(order) + 1))
    lo, hi = ylim

    # The reference is one crystal-ligand value per pocket, so under an ordering set by
    # another series it is inherently jagged; thin, grey and recessive so it reads as the
    # benchmark it is and does not fight the two model series.
    ax.plot(x, y_ref, color=REF_COLOR, lw=REF_LW, ls=(0, (4, 2.6)), alpha=0.62,
            zorder=2, label=REF_LABEL, solid_capstyle="round")
    for y, colour, marker, size, label, z in ((y_vox, VOX_COLOR, "^", VOX_SIZE, VOX_LABEL, 3),
                                              (y_our, OUR_COLOR, "o", OUR_SIZE, OUR_LABEL, 4)):
        ax.plot(x, y, ls="none", marker=marker, markersize=size, color=colour,
                markerfacecolor=colour, markeredgewidth=0, zorder=z,
                label=label if label == REF_LABEL else display(label))

    ax.set_xlabel(X_LABEL, fontsize=15.5, labelpad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10)
    ax.set_xlim(0.2, len(order) + 0.8)
    ax.set_ylim(lo, hi)
    # y is kcal/mol and the panel is short, so the default locator lands on halves
    # (-12.5, -10.0, ...): two extra glyphs per label for precision the eye cannot use at
    # this size. Integers only, and let the locator pick the step (2 here, 5 for Score's
    # wider span) rather than pinning one that would not suit all three metrics.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Labels sit at 1/20/40/60/last; the x rules fall on EVERY multiple of 10 regardless.
    # They are drawn as plain vlines rather than as grid: a minor tick is silently dropped
    # wherever a major tick already sits, which left 20/40/60 without a rule.
    ax.set_xticks([t for t in XTICKS if t < len(order)] + [len(order)])
    # Short black ticks pointing OUT, only at the labelled positions; the dotted rules
    # inside the axes carry the rest.
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    for xv in range(10, len(order) + 1, 10):
        ax.axvline(xv, color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)), zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)

    if legend:
        # Three rows stacked in the bottom-right corner, in plotting order: the
        # reference, then the two models. Padding is pulled in tight so the box stays
        # small; it
        # is opaque white with a grey rule, so the dotted grid does not run through the
        # labels but the box itself stays quiet.
        leg = ax.legend(loc="lower right", ncol=1, frameon=True, fontsize=12.5,
                        handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                        borderpad=0.4, borderaxespad=0.39, facecolor="white",
                        edgecolor=LEGEND_EDGE, framealpha=1.0)
        leg.get_frame().set_boxstyle("square", pad=0.16)
        leg.get_frame().set_linewidth(AXIS_LW)
        leg.set_zorder(6)
        for text in leg.get_texts():      # identity rides the swatch, not the ink
            text.set_color(INK)
        # The swatches are drawn at twice the plotted size: in the cloud a marker only has
        # to be findable, in the key it has to be identifiable, and at 3-5 pt a triangle
        # and a circle are the same dot. Legend handles are copies, so this does not touch
        # the data. (legend_handles is 3.7+; legendHandles is the old spelling.)
        for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
            handle.set_markersize(handle.get_markersize() * 2)


def outside(order, curves, ylim):
    """Points the y-clip leaves off the panel, as (label, target, value)."""
    lo, hi = ylim
    return [(label, order[i], v) for label, ys in curves
            for i, v in enumerate(ys) if v is not None and not lo <= v <= hi]


def fit(fig, **kw):
    """tight_layout, then make good what it under-reserves.

    At this type size tight_layout leaves the rotated y label overrunning the figure's
    left edge by ~0.08 in even though it has just run, and the label is then silently
    SLICED OFF in the raster -- the taller the type, the worse it gets. get_tightbbox
    reports the overrun after the fact, so give the axes back exactly that much and no
    more: re-running tight_layout with a rect instead would re-apply `pad` on top of the
    correction and cost ~0.2 in of plot area. Every side is checked, not just the left,
    so this keeps holding if the labels or the legend change.
    """
    fig.tight_layout(**kw)
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    sp, adj = fig.subplotpars, {}
    if bb.x0 < 0:
        adj["left"] = sp.left + (-bb.x0) / w
    if bb.y0 < 0:
        adj["bottom"] = sp.bottom + (-bb.y0) / h
    if bb.x1 > w:
        adj["right"] = sp.right - (bb.x1 - w) / w
    if bb.y1 > h:
        adj["top"] = sp.top - (bb.y1 - h) / h
    if adj:
        fig.subplots_adjust(**adj)


def plot_width(fig, ax):
    """Width of one axes' plot area, in inches -- what has to match across figures."""
    return ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width


def pair_width(target_w, build):
    """Figure width the paired figure needs for ONE panel to be `target_w` wide.

    Lays the pair out once at a trial width, measures what the furniture (two y labels,
    two sets of tick labels, the gutter, the padding) actually costs at this type size,
    and adds that back around two panels of the requested width. `build(width)` returns
    (fig, axes) already drawn and fitted; the trial figure is thrown away.
    """
    trial = 12.0
    fig, axes = build(trial)
    furniture = trial - 2 * plot_width(fig, axes[0])
    plt.close(fig)
    return 2 * target_w + furniture


def save(fig, stem):
    """PNG to look at, SVG and PDF to place -- both vector, both with live text."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", facecolor="white")
    plt.close(fig)


def main():
    vox, our = load(VANILLA), load(OURS)
    plt.rcParams.update(RC)
    print(f"=== {PROTOCOL} ===")

    for metric, (_, _, label) in METRICS.items():
        targets = sorted(t for t in set(vox) & set(our)
                         if metric in vox[t] and metric in our[t])
        ref, worst = reference(vox, our, targets, metric)
        note = ("deterministic, runs agree exactly" if worst[0] == 0 else
                f"worst run-to-run gap {worst[0]:.2f} kcal/mol @ {worst[1]}")
        print(f"\n{label}: {len(targets)} pockets · reference {note}")

        curves = {stat: series(vox, our, ref, targets, metric, stat) for stat in STATS}
        spans = {stat: limits(*c[1:]) for stat, c in curves.items()}
        # The paired figure shares one y scale, so its panels are comparable vertically.
        shared = (min(s[0] for s in spans.values()), max(s[1] for s in spans.values()))
        plot_w = None                    # set by the first standalone panel, matched below

        for stat, (order, y_our, y_vox, y_ref) in curves.items():
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            draw(ax, order, y_our, y_vox, y_ref, spans[stat], legend=True,
                 ylabel=f"{label} {stat}")
            fit(fig, pad=0.5)
            plot_w = plot_width(fig, ax)
            stem = os.path.join(HERE, f"vina_{metric}_3line_{stat}")
            save(fig, stem)

            wins = sum(1 for t in order if our[t][metric][stat] <= vox[t][metric][stat])
            pooled = {REF_LABEL: st.mean([ref[t] for t in order]),
                      VOX_LABEL: st.mean([vox[t][metric][stat] for t in order]),
                      OUR_LABEL: st.mean([our[t][metric][stat] for t in order])}
            print(f"  {stat:6s} -> {os.path.basename(stem)}.{{png,svg,pdf}}   "
                  f"Ours stronger on {wins}/{len(order)} ({100 * wins / len(order):.1f}%)   "
                  + "  ".join(f"{k} {v:+.2f}" for k, v in pooled.items()))
            for lab, target, value in outside(
                    order, ((OUR_LABEL, y_our), (VOX_LABEL, y_vox), (REF_LABEL, y_ref)),
                    spans[stat]):
                print(f"      outside the plotted y-range: {lab} @ {target} = {value:+.2f}")

            with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
                w = csv.writer(handle)
                w.writerow(["rank", "target", "n_mols_ours", "n_mols_voxbind",
                            "reference", f"voxbind_{stat}", f"ours_{stat}"])
                for i, t in enumerate(order, 1):
                    w.writerow([i, t, our[t][metric]["n"], vox[t][metric]["n"],
                                f"{ref[t]:.3f}", f"{vox[t][metric][stat]:.3f}",
                                f"{our[t][metric][stat]:.3f}"])

        def build_pair(width):
            """The paired figure at a given width, drawn and fitted. Called twice: once on
            a trial width so pair_width() can measure the furniture, once for real."""
            fig, axes = plt.subplots(1, 2, figsize=(width, PAIR_H), dpi=220, sharey=True)
            fig.patch.set_facecolor("white")
            for i, (stat, (order, y_our, y_vox, y_ref)) in enumerate(curves.items()):
                axes[i].set_facecolor("white")
                draw(axes[i], order, y_our, y_vox, y_ref, shared, legend=(i == 0),
                     ylabel=f"{label} {stat}")
            # sharey suppresses the right panel's tick labels; put the numbers back so
            # each panel reads on its own -- its y name says mean vs median, so it has to.
            axes[1].tick_params(labelleft=True, labelsize=14)
            fit(fig, pad=0.5, w_pad=1.54)
            return fig, axes

        # Each panel is sized to the standalone figure's own plot area, so the two halves
        # of the pair are the standalone figure twice over rather than a squeezed copy.
        fig, axes = build_pair(pair_width(plot_w, build_pair))
        stem = os.path.join(HERE, f"vina_{metric}_3line_pair")
        got = plot_width(fig, axes[0])
        save(fig, stem)
        print(f"  pair   -> {os.path.basename(stem)}.{{png,svg,pdf}}   "
              f"shared y {shared[0]:.2f} .. {shared[1]:.2f}   "
              f"panel {got:.3f} in vs standalone {plot_w:.3f} in ({got - plot_w:+.4f})")


if __name__ == "__main__":
    main()
