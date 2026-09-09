#!/usr/bin/env python3
"""build_vina_per_atom.py — Vina Dock against ligand size, in the 260903 3-line style.

    vina_dock_per_atom.{png,svg,pdf}    the figure
    vina_dock_per_atom.csv              every plotted number, per heavy-atom count

WHAT IT SHOWS. Two stacked panels over one x axis, the ligand's heavy-atom count.

  top     Vina Dock MEDIAN at each exact heavy-atom count, for VoxBind, VoxBind + Ours
          and the crystal reference ligand.
  bottom  How many molecules each set puts at each size, as a share of its own molecules
          -- a share and not a count, because 79 crystal ligands and ~7,900 generated ones
          do not share a count axis. Per exact heavy-atom count for all three, unsmoothed:
          the reference is 79 molecules, so its steps are 1.3% tall each and reach 7.6%
          where six pockets happen to share a size, and that spike is what sets the
          panel's y scale. That is the honest shape of it, and it is the same shape the
          top panel's grey is rolled out of
          -- which is what makes the top panel's tails trustworthy or not, and is itself
          the finding this figure exists to guard against: the arms differ in the size
          distribution they generate at least as much as in per-atom binding quality
          (see the size-confound note; raw pooled Vina Dock is ~80% a size statistic).
          The reference is on the same panel, so "does the model generate ligands the
          size of the real one" is readable without a second figure.

MEDIAN ONLY, ON PURPOSE. Both statistics were drawn at first, dash against solid, and the
figure said nothing extra for the ink: at every size the two run within ~0.2 kcal/mol of
each other, so six curves were three curves drawn twice and the pair merely thickened and
blurred each series. That agreement IS a result -- the Ours/VoxBind gap is the whole
distribution shifting, not a tail dragging the mean -- but it is a sentence, not a panel.
Both statistics stay in the CSV, and the run log prints the per-size win count and mean
gap under each, so the mean is one column away whenever it is wanted.

STYLE. 260903/build_vina_3line.py: no panel titles, warm near-black furniture (#514F52),
left+bottom spines only, dotted mid-grey rules, live text in the SVG and TrueType in the
PDF. Colour is the identity of the series -- reference grey #9aa0a6, VoxBind sand #F5B27E,
VoxBind + Ours periwinkle #8291E8 -- and the reference is additionally DASHED and thinner,
exactly as it is there: it is the benchmark the two models are read against, not a third
competitor, so it gets the least ink and a line the eye does not follow.

PROTOCOL. The published-baseline protocol, exactly as in build_vina_3line.py: whole
`*_rec.pdb` receptor, exhaustiveness 32, all 79 pockets, from `eval_docking_results_
full79.json`. The same two runs, so this figure and the 3-line figures sit on one axis.

WHY NO CONFIDENCE BANDS. The thing a band would guard against here is already drawn: x is
clipped to the counts where BOTH models have at least MIN_N molecules, so no curve has a
tail the other cannot answer, and the bottom panel shows how thin each end actually is.
The per-count sample sizes are in the CSV.

THE REFERENCE IS ROLLED, AND FADES WHERE IT THINS. There is exactly ONE crystal ligand per
pocket -- 79 in total, 1-6 at any exact heavy-atom count -- so a per-count reference curve
would be noise. Its VINA curve -- the top panel only -- is a centred rolling window of
+-REF_WIN atoms. The models are NOT rolled: they have hundreds of molecules per count, and
neither is the distribution panel, where a share is a share at any n.

Above 36 atoms the window holds fewer than REF_FIRM ligands and by 43 it is down to three,
so that end of the grey is drawn at REF_THIN_ALPHA rather than gated away. Read it as an
anecdote, not a curve: the plunge to -14.7 at 42-45 is two real crystal ligands, TNKS1
(target_72, -15.99) and AKT1 (target_80, -14.66), and nothing else. It costs the panel
about 4 kcal/mol of y range, which compresses the models where they are actually being
compared -- the price of the grey ending where the distribution panel below it ends
instead of five atoms short, which read as missing data. `n_reference_window` in the CSV
is the pool behind every reference point.

    /opt/conda/envs/voxbind/bin/python notebook/html/260910/fig-vina-per-atom/build_vina_per_atom.py
"""
import collections
import csv
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"

VANILLA = f"{E}/_vanilla_ep923/samples/full_eval_ep923"
OURS = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
BASENAME = "eval_docking_results_full79.json"
PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

FIELD, REF_FIELD = "vina_dock", "ref_vina_dock"
STATS = ("mean", "median")
DRAWN = "median"                 # the one the top panel plots; both go to the CSV

Y_LABEL = f"Vina Dock {DRAWN}"   # the y name is what says which statistic, as in 3line
X_LABEL = "Number of heavy atoms in ligand"
D_LABEL = "% of ligands"

REF_LABEL, REF_COLOR = "Reference ligand", "#9aa0a6"
VOX_LABEL, VOX_COLOR = "VoxBind", "#F5B27E"
OUR_LABEL, OUR_COLOR = "VoxBind + Ours", "#8291E8"

# A heavy-atom count is plotted only where BOTH models have this many molecules, so the
# two curves start and stop together and neither carries a tail the other cannot answer.
MIN_N = 25
# The reference window: +-REF_WIN atoms. MIN_REF is the floor for drawing a point at all;
# REF_FIRM is where the window is populated enough to be read as a curve. Between the two
# the line is drawn at REF_THIN_ALPHA, so the sparse end is present but visibly not
# load-bearing. The gate used to be a hard 12, which stopped the grey at 36 while the
# distribution panel below it ran to 45 -- the same series ending at two different places,
# which reads as missing data rather than as thin data.
REF_WIN, MIN_REF, REF_FIRM = 4, 3, 12
REF_THIN_ALPHA = 0.42

# The reference dash. It stays legible at hairline weights and does not shimmer where it
# runs close to a model curve.
DASH = (0, (4, 2.6))

INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"

# Line weights in points. Model curves carry the argument; the reference is the benchmark
# they are read against, so it is the thinnest ink on the panel.
MODEL_LW, REF_LW = 2.35, 1.5
AXIS_LW, GRID_LW = 1.35, 1.1
DIST_LW, DIST_FILL = 1.7, 0.16

# GEOMETRY, in inches. The 3-line base figure is 7.25 x 2.77; WIDE/TALL scale it. This one
# is a shade wider than the 3-line standalone, and its height is the top panel at roughly
# the 3-line panel's shape plus the distribution strip underneath.
WIDE, TALL = 1.05, 1.155
FIG_W = 7.25 * WIDE
FIG_H = 2.77 * TALL + 1.62
HEIGHT_RATIOS = (2.62, 1.05)
# The gap between the panels is set through tight_layout's h_pad (in font-size units),
# NOT through gridspec_kw's hspace: an explicit hspace makes tight_layout declare the axes
# "not compatible", warn, and then leave ~12% of the figure as dead white at the top.
H_PAD = 0.2

XTICK_STEP = 5                   # ticks and vertical rules both, unlike the 3-line figure
                                 # where the rules fall between the labelled ranks

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
}


def load(root):
    """{heavy atoms: [vina dock, ...]} pooled over the 79 pockets of one run."""
    with open(os.path.join(root, BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    by_size = collections.defaultdict(list)
    for t in per_target:
        for m in t.get("per_mol") or []:
            if isinstance(m, dict) and m.get(FIELD) is not None and m.get("n_atoms"):
                by_size[m["n_atoms"]].append(m[FIELD])
    return by_size, len(per_target)


def load_reference(root):
    """{heavy atoms: [crystal-ligand vina dock, ...]}, one entry per pocket.

    The docked value is in the results file, but the ligand's own atom count is not -- it
    lives in that pocket's metrics.json, under `reference`. Pockets missing either are
    dropped and named by the caller."""
    with open(os.path.join(root, BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    by_size, dropped = collections.defaultdict(list), []
    for t in per_target:
        path = os.path.join(root, t["target"], "metrics.json")
        n_atoms = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                n_atoms = (json.load(handle).get("reference") or {}).get("n_atoms")
        value = t.get(REF_FIELD)
        if n_atoms and value is not None:
            by_size[n_atoms].append(value)
        else:
            dropped.append(t["target"])
    return by_size, dropped


def model_curve(by_size, xs, stat):
    """mean/median at each exact heavy-atom count."""
    f = st.mean if stat == "mean" else st.median
    return [f(by_size[a]) for a in xs]


def reference_curve(by_size, xs, stat):
    """Same, over a centred +-REF_WIN window; None where the window is too thin to mean
    anything, which leaves a gap in the line rather than an invented value."""
    f = st.mean if stat == "mean" else st.median
    out = []
    for a in xs:
        pool = [v for n, vals in by_size.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(pool) if len(pool) >= MIN_REF else None)
    return out


def window_counts(by_size, xs):
    """How many crystal ligands each plotted point's rolling window pools."""
    return [sum(len(v) for n, v in by_size.items() if abs(n - a) <= REF_WIN) for a in xs]


def split_firm(values, counts):
    """The reference curve cut into the part its window can carry and the part it cannot.

    Returns (firm, thin): both are full-length, each holding None wherever the other holds
    the value. A point on the boundary belongs to BOTH, so the solid and the faded stretch
    meet rather than leaving a gap -- a gap would say "no data here", which is the
    opposite of what the fade is for."""
    firm_at = [c >= REF_FIRM for c in counts]
    n = len(firm_at)
    edge = [firm_at[i] and any(not firm_at[j] for j in (i - 1, i + 1) if 0 <= j < n)
            for i in range(n)]
    firm = [v if f else None for v, f in zip(values, firm_at)]
    thin = [v if (not f) or e else None for v, f, e in zip(values, firm_at, edge)]
    return firm, thin


def share(by_size, xs):
    """Percent of a set's molecules at each plotted count. A share, not a count: 79
    reference ligands and ~7,900 generated ones do not share a count axis, and the
    question the panel answers -- where does this set put its mass -- is a share."""
    total = sum(len(v) for v in by_size.values())
    return [100.0 * len(by_size.get(a, ())) / total for a in xs], total


def furniture(ax, *, ylabel, xlabel, xs):
    """The 3-line figure's axes: dotted rules, two spines, ticks pointing out."""
    ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15.5, labelpad=9)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)


def legend(ax):
    """One key, in the empty upper right, drawn in plotting order: the reference, then the
    two models. Opaque white with a rule in the axis pen, so the dotted grid does not run
    through the labels but the box itself stays quiet. Its handles carry the dash, so the
    reference's identity is readable from the key alone."""
    handles = [Line2D([], [], color=c, lw=lw, ls=ls, label=lab)
               for lab, c, lw, ls in ((REF_LABEL, REF_COLOR, REF_LW, DASH),
                                      (VOX_LABEL, VOX_COLOR, MODEL_LW, "-"),
                                      (OUR_LABEL, OUR_COLOR, MODEL_LW, "-"))]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=12.5,
                    handlelength=1.9, handletextpad=0.55, labelspacing=0.3,
                    borderpad=0.4, borderaxespad=0.39, facecolor="white",
                    edgecolor=LEGEND_EDGE, framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)


def fit(fig, **kw):
    """tight_layout, then give back whatever it under-reserved.

    At this type size tight_layout can leave a rotated y label overrunning the figure edge
    even though it has just run, and the label is then silently SLICED OFF in the raster.
    get_tightbbox reports the overrun after the fact, so hand back exactly that much:
    re-running tight_layout with a rect would re-apply `pad` on top of the correction and
    cost plot area. Every side is checked, so this keeps holding if the labels change."""
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


def save(fig, stem):
    """PNG to look at, SVG and PDF to place -- both vector, both with live text."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", facecolor="white")
    plt.close(fig)


def main():
    vox, n_vox_pockets = load(VANILLA)
    our, n_our_pockets = load(OURS)
    ref, dropped = load_reference(VANILLA)

    xs = [a for a in sorted(set(vox) & set(our))
          if len(vox[a]) >= MIN_N and len(our[a]) >= MIN_N]
    # A gap would break both curves at the same place, but nothing guarantees the gated
    # set is contiguous, so say so rather than letting matplotlib bridge it silently.
    gaps = [a for a in range(xs[0], xs[-1] + 1) if a not in xs]

    # Both statistics are computed for the CSV and the run log; only DRAWN is plotted.
    curves = {}
    for stat in STATS:
        curves[(REF_LABEL, stat)] = reference_curve(ref, xs, stat)
        curves[(VOX_LABEL, stat)] = model_curve(vox, xs, stat)
        curves[(OUR_LABEL, stat)] = model_curve(our, xs, stat)

    shares, totals = {}, {}
    for label, by_size in ((REF_LABEL, ref), (VOX_LABEL, vox), (OUR_LABEL, our)):
        shares[label], totals[label] = share(by_size, xs)

    plt.rcParams.update(RC)
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), dpi=220, sharex=True,
        gridspec_kw=dict(height_ratios=list(HEIGHT_RATIOS)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bx.set_facecolor("white")

    # Reference first and Ours last, so the series being read sits on top -- the same
    # ordering the 3-line figures use.
    order = ((REF_LABEL, REF_COLOR, REF_LW, DASH, 2),
             (VOX_LABEL, VOX_COLOR, MODEL_LW, "-", 3),
             (OUR_LABEL, OUR_COLOR, MODEL_LW, "-", 4))
    ref_counts = window_counts(ref, xs)
    for label, colour, lw, style, z in order:
        if label is REF_LABEL:
            firm, thin = split_firm(curves[(label, DRAWN)], ref_counts)
            for ys, alpha in ((thin, REF_THIN_ALPHA), (firm, 1.0)):
                ax.plot(xs, ys, color=colour, lw=lw, ls=style, zorder=z, alpha=alpha,
                        dash_capstyle="round")
        else:
            ax.plot(xs, curves[(label, DRAWN)], color=colour, lw=lw, ls=style, zorder=z,
                    solid_capstyle="round")

    # Same three identities below: colour for the series, dash for the reference. All
    # three are per-count steps -- the reference is NOT smoothed here, so its shape is the
    # 79 crystal ligands themselves. It is the only one left unfilled: at 79 molecules its
    # steps are tall and jagged, and a third fill over the two model areas would darken
    # exactly the region being compared without adding a reading.
    for label, colour, lw, style, z in order:
        if label is not REF_LABEL:
            bx.fill_between(xs, shares[label], step="mid", color=colour, alpha=DIST_FILL,
                            linewidth=0, zorder=z)
        bx.step(xs, shares[label], where="mid", color=colour, ls=style, zorder=z + 3,
                lw=DIST_LW if label is not REF_LABEL else lw,
                solid_capstyle="round", dash_capstyle="round")

    furniture(ax, ylabel=Y_LABEL, xlabel=None, xs=xs)
    furniture(bx, ylabel=D_LABEL, xlabel=X_LABEL, xs=xs)
    ax.margins(y=0.075)              # the default 5% puts Ours on the bottom spine
    # Every 2 kcal/mol. The locator, left to itself over a ~9 kcal/mol span, picks 3 and
    # labels -3/-6/-9, which reads as a coarser axis than the 3-line figures' step of 2.
    ax.yaxis.set_major_locator(MultipleLocator(2))
    bx.set_ylim(0, max(v for s in shares.values() for v in s) * 1.13)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    # Each y label is otherwise placed against ITS OWN tick labels, so "Vina Dock median"
    # (widest tick "-10") and "% of ligands" (widest tick "4") sit at two different x and
    # the pair reads as a ragged left edge. align_ylabels puts both at the outer of the
    # two, which is the only way they line up without hard-coding a coordinate.
    fig.align_ylabels((ax, bx))
    legend(ax)
    fit(fig, pad=0.5, h_pad=H_PAD)

    stem = os.path.join(HERE, "vina_dock_per_atom")
    save(fig, stem)

    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["heavy_atoms",
                    "n_voxbind", "voxbind_mean", "voxbind_median", "voxbind_pct",
                    "n_ours", "ours_mean", "ours_median", "ours_pct",
                    "n_reference_exact", "n_reference_window",
                    "reference_mean", "reference_median",
                    "reference_pct"])
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        for i, a in enumerate(xs):
            pool = [v for n, vals in ref.items() if abs(n - a) <= REF_WIN for v in vals]
            w.writerow([a,
                        len(vox[a]), fmt(curves[(VOX_LABEL, "mean")][i]),
                        fmt(curves[(VOX_LABEL, "median")][i]), f"{shares[VOX_LABEL][i]:.3f}",
                        len(our[a]), fmt(curves[(OUR_LABEL, "mean")][i]),
                        fmt(curves[(OUR_LABEL, "median")][i]), f"{shares[OUR_LABEL][i]:.3f}",
                        len(ref.get(a, ())), len(pool),
                        fmt(curves[(REF_LABEL, "mean")][i]),
                        fmt(curves[(REF_LABEL, "median")][i]),
                        f"{shares[REF_LABEL][i]:.3f}"])

    print(f"=== {PROTOCOL} ===")
    print(f"{os.path.basename(stem)}.{{png,svg,pdf,csv}}   drawn statistic: {DRAWN}")
    print(f"  pockets: VoxBind {n_vox_pockets} · Ours {n_our_pockets} · "
          f"reference {sum(len(v) for v in ref.values())}"
          + (f" (dropped {', '.join(dropped)})" if dropped else ""))
    print(f"  x = {xs[0]}..{xs[-1]} heavy atoms, both models >= {MIN_N} molecules"
          + (f"; GAPS at {gaps}" if gaps else " (contiguous)"))
    firm_xs = [a for a, c in zip(xs, ref_counts) if c >= REF_FIRM]
    print(f"  reference window >= {REF_FIRM} ligands at {firm_xs[0]}..{firm_xs[-1]} atoms "
          f"(drawn solid); {xs[0]}..{firm_xs[0] - 1} and {firm_xs[-1] + 1}..{xs[-1]} are "
          f"drawn at alpha {REF_THIN_ALPHA} on {min(ref_counts)}-{REF_FIRM - 1} ligands")
    for label, *_ in order:
        drawn = sum(1 for v in curves[(label, DRAWN)] if v is not None)
        print(f"  {label:16s} n={totals[label]:5d}  "
              f"Vina curve drawn at {drawn}/{len(xs)} counts  "
              f"{sum(shares[label]):.1f}% of its molecules inside the plotted x range")
    for stat in STATS:
        wins = [a for i, a in enumerate(xs)
                if curves[(OUR_LABEL, stat)][i] <= curves[(VOX_LABEL, stat)][i]]
        gap = st.mean([curves[(OUR_LABEL, stat)][i] - curves[(VOX_LABEL, stat)][i]
                       for i in range(len(xs))])
        mark = "  <- drawn" if stat == DRAWN else ""
        print(f"  {stat:6s}: Ours lower at {len(wins)}/{len(xs)} sizes, "
              f"mean per-size gap {gap:+.3f} kcal/mol{mark}")
    for label, by in ((REF_LABEL, ref), (VOX_LABEL, vox), (OUR_LABEL, our)):
        vals = [n for n, v in by.items() for _ in v]
        print(f"  size of {label:16s} mean {st.mean(vals):5.2f}  "
              f"median {st.median(vals):5.1f} heavy atoms")


if __name__ == "__main__":
    main()
