#!/usr/bin/env python3
"""build_vina_per_atom.py — Vina Dock against ligand size, in the fig-vina-3line style.

    vina_dock_per_atom_v1_mean.{png,svg,pdf}    vina_dock_per_atom_v1_median.{png,svg,pdf}
    vina_dock_per_atom_v2_mean.{png,svg,pdf}    vina_dock_per_atom_v2_median.{png,svg,pdf}
    vina_dock_per_atom_v3_mean.{png,svg,pdf}    vina_dock_per_atom_v3_median.{png,svg,pdf}
    vina_dock_per_atom.csv                      every plotted number, all six

SIX FIGURES: three variants x two statistics. Which statistic a figure shows is carried by
its filename and by its y-axis name ("Vina Dock / mean" / "Vina Dock / median", set over two
lines so the pair of panels keeps one narrow left margin), exactly as in build_vina_3line.py
-- there are no panel titles, so the y name is the only thing telling two otherwise
identical figures apart. One CSV serves all six: only which reference points get drawn
depends on the variant, and `reference_firm` is that column.

THREE VARIANTS, ONE SCRIPT. They differ in four knobs, all of them about the crystal
reference and all of them listed in VARIANTS below. They are kept in one file on purpose:
split into three scripts they would drift, and then the "same figure, one choice
different" claim would quietly stop being true.

  v1  The reference's Vina curve is GATED: drawn only where its rolling window holds at
      least REF_FIRM ligands, so the grey stops at 36 atoms. Its distribution below is
      smoothed and filled like the models'.
  v2  The reference's Vina curve runs the full x range, FADED where the window thins
      (below REF_FIRM, from 37 atoms up). Its distribution is smoothed but UNFILLED, so
      the two model areas underneath stay readable.

      v2's cost is real and visible: the reference median at 42-44 is two crystal ligands,
      TNKS1 (target_72, -15.99) and AKT1 (target_80, -14.66), and drawing them stretches y
      to -15.5, which compresses the -3..-11 band where the arms are actually being
      compared. v1 spends nothing on y but ends its grey eight atoms short of where the
      panel below it ends, which reads as missing data rather than as thin data. Neither
      is free; that is why both are built.
  v3  v2 with the reference DROPPED ENTIRELY -- no grey curve on top, no grey
      distribution below, no key entry. Every other choice is v2's. What it buys is the y
      axis: nothing on the panel then reaches past the models, so the -3..-11 band they
      are actually compared in fills the height instead of being squashed by two crystal
      outliers at 42-44 atoms. What it costs is the benchmark -- read alone, "CoDE is
      0.4 kcal/mol below VoxBind at every size" has no scale against a real ligand, and
      the size-distribution panel loses the "do these models generate ligands the size of
      the real one" reading that was the reason the reference sits on it at all. It is
      the variant for a slide where the reference is stated in words next to it.

WHAT THEY SHOW. Two stacked panels over one x axis, the ligand's heavy-atom count.

  top     Vina Dock mean or median at each exact heavy-atom count, for every arm that
          carries per-molecule Vina under this protocol -- the TargetDiff baseline,
          VoxBind, CoDE -- and the crystal reference ligand.

          THE OTHER FIVE PUBLISHED BASELINES ARE NOT HERE, and not by choice. AR,
          Pocket2Mol, DiffSBDD, DecompDiff and FuncBind are staged on this box as sample
          directories (exps/baselines_pose/<key>/target_XX/) and scored for PoseBusters
          and PoseCheck, but their metrics.json carries `docking: "none"`: the Vina run
          behind results/task2-drugdesign/_shared/baselines_eval/summary_density79.json
          happened on svr12 and only its AGGREGATES came back, so there is no
          (heavy atoms, vina_dock) pair per molecule to bin. Docking them here under this
          protocol is ~89 CPU-hours per method (79 pockets x ~100 molecules,
          exhaustiveness 32, whole receptor). Add the root to ARMS the day that exists;
          nothing else in this file needs to change.
  bottom  How many molecules each set puts at each size, as a share of its own molecules
          -- a share and not a count, because 79 crystal ligands and ~7,900 generated ones
          do not share a count axis. This is what makes the top panel's tails trustworthy
          or not, and is itself the finding the figure exists to guard against: the arms
          differ in the size distribution they generate at least as much as in per-atom
          binding quality (see the size-confound note; raw pooled Vina Dock is ~80% a size
          statistic). The reference is on the same panel, so "does the model generate
          ligands the size of the real one" is readable without a second figure.

EVERY ARM IS GATED THE SAME WAY, and the gate is now per arm rather than per figure. A
heavy-atom count enters x only if EVERY drawn arm has MIN_N molecules there, which sets
the two ends; inside that span an arm that dips below MIN_N (TargetDiff does, at 38 and
41 atoms) has its curve BROKEN there rather than interpolated, because a bridge over a
count where one method generated 15 molecules is a drawn claim about data that is not
there. The distribution panel underneath keeps drawing at those counts -- being thin is
exactly what it is for.

ONE STATISTIC PER FIGURE. Both were drawn in one panel at first, dash against solid, and
it said nothing extra for the ink: at every size the two run within ~0.2 kcal/mol of each
other, so six curves were three curves drawn twice and the pair merely thickened and
blurred each series. They are separate figures instead. That near-agreement IS a result --
the Ours/VoxBind gap is the whole distribution shifting, not a tail dragging the mean --
and it is what makes the two figures nearly interchangeable; the mean is the slightly
kinder one to CoDE (below VoxBind at 35/40 sizes against the median's 31/40, mean
per-size gap -0.441 against -0.388), because CoDE's size distribution reaches further
right.

STYLE. 260910/fig-vina-3line/build_vina_3line.py: no panel titles, warm near-black furniture (#514F52),
left+bottom spines only, dotted mid-grey rules, live text in the SVG and TrueType in the
PDF. Colour is the identity of the series and comes from ../method_colors.py, so a reader
moving between the Vina, PoseBusters and PoseCheck figures never re-learns the key --
reference grey #9AA0A6, TargetDiff violet #B58FDB, VoxBind sand #F5B27E, CoDE periwinkle
#8291E8 (the lighter tint of our blue this Vina family carries, as method_colors records).
INK IS THE SECOND CHANNEL, and it says what a series is FOR: the two arms being compared
are the thickest, solid, and the only ones with an area under their distribution; the
reference and the baseline are thinner and DASHED, because they are what the pair is read
against rather than a third and fourth competitor, and four filled areas on the lower
panel would be a stack no one can read through.

PROTOCOL. The published-baseline protocol, exactly as in build_vina_3line.py: whole
`*_rec.pdb` receptor, exhaustiveness 32, all 79 pockets, from `eval_docking_results_
full79.json`. The same runs, so this figure and the 3-line figures sit on one axis --
TargetDiff included, which 74_dock_targetdiff_full79.sh re-docked here for exactly that
reason.

WHY NO CONFIDENCE BANDS. The thing a band would guard against here is already drawn: x is
clipped to the counts where EVERY arm has at least MIN_N molecules, so no curve has a
tail the others cannot answer, and the bottom panel shows how thin each end actually is.
The per-count sample sizes are in the CSV.

THE REFERENCE IS ROLLED. There is exactly ONE crystal ligand per pocket -- 79 in total,
1-6 at any exact heavy-atom count -- so anything per-count off it is noise: raw, its
distribution is a picket fence of 1.3%-tall steps reaching 7.6% where six pockets happen
to share a size, and that spike, not the models, would set the lower panel's y scale. Both
its curves are therefore a centred rolling window of +-REF_WIN atoms. The models are NOT
rolled, in either panel: they have hundreds of molecules per count.

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
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
import sys                                                            # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from method_colors import display                                     # noqa: E402
E = "/home1/irteam/VoxBind/voxbind/exps"

BASENAME = "eval_docking_results_full79.json"
PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# The whole difference between the three figures. `ref_draw` is v3's one knob: no
# reference at all, on either panel or in the key. `ref_gate` "hard" stops the reference's
# Vina curve where its window drops below REF_FIRM; "fade" keeps drawing down to REF_FLOOR
# at REF_THIN_ALPHA. `ref_dist_smooth` rolls the reference's distribution; `ref_dist_fill`
# gives it an area under it like the two compared arms have.
VARIANTS = {
    "v1": dict(ref_draw=True, ref_gate="hard", ref_dist_smooth=True, ref_dist_fill=True),
    "v2": dict(ref_draw=True, ref_gate="fade", ref_dist_smooth=True, ref_dist_fill=False),
    "v3": dict(ref_draw=False, ref_gate="fade", ref_dist_smooth=True, ref_dist_fill=False),
}

FIELD, REF_FIELD = "vina_dock", "ref_vina_dock"
STATS = ("mean", "median")       # one figure each, per variant

# The y names are set over TWO LINES. One line each ("Vina Dock median", "% of ligands")
# put the widest label's length into the left margin of a 7.6-inch figure; broken after
# the metric they are the same words in half the width, and align_ylabels below keeps the
# two panels' labels on one left edge. The statistic still lives in the y name and
# nowhere else, exactly as in the 3-line figures.
Y_LABEL = "Vina Dock\n{stat}"
X_LABEL = "Number of heavy atoms in ligand"
D_LABEL = "Generated\nfraction"

REF_LABEL, REF_COLOR = "Reference ligand", "#9aa0a6"
VOX_LABEL, VOX_COLOR = "VoxBind", "#F5B27E"
# These are DATA KEYS as well as labels (they head this folder's CSV columns), so they
# stay plain; method_colors.display() dresses them for a legend -- VoxBind picks up its
# sigma=0.9 subscript there. CoDE is \textsc{CoDE} in the .tex files; matplotlib has no
# small caps without a TeX backend, so the figures carry the plain string.
OUR_LABEL, OUR_COLOR = "CoDE", "#8291E8"

# Line weights in points. MODEL_BOOST widens ONLY the two arms the figure exists to
# compare -- both their Vina curves and their distribution steps -- and is applied as a
# factor on the base weights rather than baked into them, so the gap it opens against the
# context series' thinner ink stays visible as a decision.
MODEL_BOOST = 1.2
MODEL_LW, REF_LW, BASE_LW = 2.35 * MODEL_BOOST, 1.5, 1.9
AXIS_LW, GRID_LW = 1.35, 1.1
DIST_LW, DIST_FILL = 1.7 * MODEL_BOOST, 0.16

# The reference dash. It stays legible at hairline weights and does not shimmer where it
# runs close to a model curve. BASE_DASH is the baseline's, longer so the two dashed
# series are told apart by rhythm as well as by colour.
DASH, BASE_DASH = (0, (4, 2.6)), (0, (6, 2))

# label, run root, line width, dash -- DRAWN IN THIS ORDER, so ours lands on top and the
# context series sit under it. Every entry must carry per-molecule `vina_dock` and
# `n_atoms` in its BASENAME file; the five other published baselines do not on this box
# (see the module docstring), which is why this list is three long and not eight.
ARMS = [
    ("TargetDiff", "/home1/irteam/base_drug/eval/targetdiff", BASE_LW, BASE_DASH),
    (VOX_LABEL, f"{E}/_vanilla_ep923/samples/full_eval_ep923", MODEL_LW, "-"),
    (OUR_LABEL, f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
     MODEL_LW, "-"),
]
# The two arms this figure exists to compare: the only ones drawn solid, and the only ones
# given an area under their distribution. Everything else on the panel is context.
FOCUS = (VOX_LABEL, OUR_LABEL)
COLORS = {REF_LABEL: REF_COLOR, VOX_LABEL: VOX_COLOR, OUR_LABEL: OUR_COLOR,
          "TargetDiff": "#B58FDB"}
# The crystal ligand is the same molecule in every run that carries one, but it is
# resolved BY LABEL, never by position in ARMS -- the staged baselines have no reference
# block at all, and a positional pick would silently become one of them.
REF_ROOT = next(root for lab, root, _, _ in ARMS if lab == VOX_LABEL)

# A heavy-atom count enters x only where EVERY drawn arm has this many molecules, which is
# what sets the two ends; inside the span an arm below it has its curve broken there
# rather than interpolated.
MIN_N = 25
# The reference window is +-REF_WIN atoms. REF_FIRM is where it holds enough crystal
# ligands to be read as a curve (v1 gates there, v2 fades below it); REF_FLOOR is v2's
# floor for drawing a point at all -- at 43-45 atoms the window is down to three ligands.
REF_WIN, REF_FIRM, REF_FLOOR = 4, 12, 3
REF_THIN_ALPHA = 0.42

INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"

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


def slug(label):
    """The CSV's column prefix for an arm: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


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
    """mean/median at each exact heavy-atom count, None below MIN_N.

    None and not the value: a count an arm put fewer than MIN_N molecules at is a number
    the other arms cannot be asked to answer, and plotting it would also let matplotlib
    draw a straight segment over it as though the arm had been measured there."""
    f = st.mean if stat == "mean" else st.median
    return [f(by_size[a]) if len(by_size.get(a, ())) >= MIN_N else None for a in xs]


def window_counts(by_size, xs):
    """How many crystal ligands each plotted point's rolling window pools."""
    return [sum(len(v) for n, v in by_size.items() if abs(n - a) <= REF_WIN) for a in xs]


def reference_curve(by_size, xs, stat, floor):
    """The reference over a centred +-REF_WIN window; None where the window holds fewer
    than `floor` ligands, which leaves a gap in the line rather than an invented value."""
    f = st.mean if stat == "mean" else st.median
    out = []
    for a in xs:
        pool = [v for n, vals in by_size.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(pool) if len(pool) >= floor else None)
    return out


def split_firm(values, counts):
    """The reference curve cut into the part its window can carry and the part it cannot.

    Returns (firm, thin): both full-length, each holding None wherever the other holds the
    value. A point on the boundary belongs to BOTH, so the solid and the faded stretch meet
    rather than leaving a gap -- a gap would say "no data here", which is the opposite of
    what the fade is for."""
    firm_at = [v is not None and c >= REF_FIRM for v, c in zip(values, counts)]
    n = len(firm_at)
    edge = [firm_at[i] and any(not firm_at[j] for j in (i - 1, i + 1) if 0 <= j < n)
            for i in range(n)]
    firm = [v if f else None for v, f in zip(values, firm_at)]
    thin = [v if (not f) or e else None for v, f, e in zip(values, firm_at, edge)]
    return firm, thin


def share(by_size, xs):
    """Percent of a set's molecules at each plotted count."""
    total = sum(len(v) for v in by_size.values())
    return [100.0 * len(by_size.get(a, ())) / total for a in xs], total


def rolled(values, xs):
    """The same shares under the reference's +-REF_WIN window.

    Averaging the shares over the window -- not summing them -- keeps the height on the
    same scale as the arms' per-count shares, so all of them can be read against one axis.
    It is the same window the reference's Vina curve uses, and for the same reason. At the
    two ends of x the window is truncated to the plotted counts, as any moving average's
    is, so the first and last few points sit on fewer sizes; `reference_pct_exact` in the
    CSV is the unsmoothed share if that matters."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= REF_WIN]) for a in xs]


def furniture(ax, *, ylabel, xlabel, xs):
    """The 3-line figure's axes: dotted rules, two spines, ticks pointing out."""
    ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10, multialignment="center")
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


def legend(ax, series):
    """One key, in the empty upper right, drawn in plotting order. Opaque white with a
    rule in the axis pen, so the dotted grid does not run through the labels but the box
    itself stays quiet. Its handles carry the dash, so a series' identity is readable from
    the key alone -- and CoDE is what our arm is called there, as everywhere else."""
    handles = [Line2D([], [], color=colour, lw=lw, ls=style,
                      label=lab if lab == REF_LABEL else display(lab))
               for lab, colour, lw, style, _ in series]
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


def build(name, cfg, stat, xs, curves, exact, counts):
    """One variant at one statistic: the figure, and the line about it for the run log."""
    stem = os.path.join(HERE, f"vina_dock_per_atom_{name}_{stat}")

    # Reference first and ours last, so the series being read sits on top -- the same
    # ordering the 3-line figures use. v3 simply never puts the reference in the list, so
    # it is absent from both panels and from the key by construction rather than by three
    # separate `if` statements downstream.
    series = [(lab, COLORS[lab], lw, style, z)
              for z, (lab, _, lw, style) in enumerate(ARMS, start=3)]
    if cfg["ref_draw"]:
        series.insert(0, (REF_LABEL, REF_COLOR, REF_LW, DASH, 2))

    shares = dict(exact)
    if cfg["ref_dist_smooth"]:
        shares[REF_LABEL] = rolled(exact[REF_LABEL], xs)
    ref_values = curves[(REF_LABEL, stat)] if cfg["ref_gate"] == "fade" else [
        v if c >= REF_FIRM else None for v, c in zip(curves[(REF_LABEL, stat)], counts)]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), dpi=220, sharex=True,
        gridspec_kw=dict(height_ratios=list(HEIGHT_RATIOS)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bx.set_facecolor("white")

    for label, colour, lw, style, z in series:
        if label != REF_LABEL:
            ax.plot(xs, curves[(label, stat)], color=colour, lw=lw, ls=style, zorder=z,
                    solid_capstyle="round", dash_capstyle="round")
        elif cfg["ref_gate"] == "fade":
            firm, thin = split_firm(ref_values, counts)
            for ys, alpha in ((thin, REF_THIN_ALPHA), (firm, 1.0)):
                ax.plot(xs, ys, color=colour, lw=lw, ls=style, zorder=z, alpha=alpha,
                        dash_capstyle="round")
        else:
            ax.plot(xs, ref_values, color=colour, lw=lw, ls=style, zorder=z,
                    dash_capstyle="round")

    # The same identities below: colour for the series, dash for the two it is read
    # against. The arms are per-count steps; the reference is a curve when it is smoothed,
    # because drawing a rolling average stepped would claim a precision it lost. Only the
    # two FOCUS arms get an area -- four filled bands would be a stack, not a comparison.
    for label, colour, lw, style, z in series:
        stepped = label != REF_LABEL or not cfg["ref_dist_smooth"]
        filled = label in FOCUS or (label == REF_LABEL and cfg["ref_dist_fill"])
        if filled:
            bx.fill_between(xs, shares[label], step="mid" if stepped else None,
                            color=colour, alpha=DIST_FILL, linewidth=0, zorder=z)
        draw = bx.step if stepped else bx.plot
        draw(xs, shares[label], color=colour, ls=style, zorder=z + 5,
             lw=DIST_LW if label in FOCUS else lw,
             solid_capstyle="round", dash_capstyle="round",
             **(dict(where="mid") if stepped else {}))

    furniture(ax, ylabel=Y_LABEL.format(stat=stat), xlabel=None, xs=xs)
    furniture(bx, ylabel=D_LABEL, xlabel=X_LABEL, xs=xs)
    ax.margins(y=0.075)              # the default 5% puts ours on the bottom spine
    # Every 2 kcal/mol. The locator, left to itself over a ~9 kcal/mol span, picks 3 and
    # labels -3/-6/-9, which reads as a coarser axis than the 3-line figures' step of 2.
    ax.yaxis.set_major_locator(MultipleLocator(2))
    bx.set_ylim(0, max(v for lab, *_ in series for v in shares[lab]) * 1.13)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    # "Generated fraction" is the name; the per-cent sign on the ticks is where the unit
    # is stated, so the label does not have to carry a parenthesis to stay honest.
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    # Each y label is otherwise placed against ITS OWN tick labels, so "Vina Dock median"
    # (widest tick "-10") and "Generated fraction" (widest tick "4%") sit at two different
    # x and the pair reads as a ragged left edge. align_ylabels puts both at the outer of
    # the two, which is the only way they line up without hard-coding a coordinate.
    fig.align_ylabels((ax, bx))
    legend(ax, series)
    fit(fig, pad=0.5, h_pad=H_PAD)
    lo, hi = ax.get_ylim()
    save(fig, stem)

    drawn = [a for a, v in zip(xs, ref_values) if v is not None]
    where = (f"reference {drawn[0]}..{drawn[-1]} atoms" if cfg["ref_draw"]
             else "no reference")
    return (f"  {os.path.basename(stem)}.{{png,svg,pdf}}   "
            f"{where}   y {lo:.1f} .. {hi:.1f}")


def write_csv(xs, data, ref, curves, exact, counts):
    """One table behind all six figures.

    Only WHICH reference points a figure draws depends on the variant, and that is
    `reference_firm` -- yes wherever the rolling window holds at least REF_FIRM crystal
    ligands. v1 draws the reference exactly there, v2 draws everywhere and fades the rest,
    v3 draws none of it. So a per-figure CSV would be six copies of one table plus one
    boolean."""
    path = os.path.join(HERE, "vina_dock_per_atom.csv")
    rolled_ref = rolled(exact[REF_LABEL], xs)
    header = ["heavy_atoms"]
    for label, *_ in ARMS:
        k = slug(label)
        header += [f"n_{k}", f"{k}_mean", f"{k}_median", f"{k}_pct"]
    header += ["n_reference_exact", "n_reference_window", "reference_firm",
               "reference_mean", "reference_median",
               "reference_pct_exact", "reference_pct_rolled"]
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        for i, a in enumerate(xs):
            row = [a]
            for label, *_ in ARMS:
                row += [len(data[label].get(a, ())),
                        fmt(curves[(label, "mean")][i]),
                        fmt(curves[(label, "median")][i]),
                        f"{exact[label][i]:.3f}"]
            row += [len(ref.get(a, ())), counts[i],
                    "yes" if counts[i] >= REF_FIRM else "no",
                    fmt(curves[(REF_LABEL, "mean")][i]),
                    fmt(curves[(REF_LABEL, "median")][i]),
                    f"{exact[REF_LABEL][i]:.3f}", f"{rolled_ref[i]:.3f}"]
            w.writerow(row)
    return path


def main():
    data, pockets = {}, {}
    for label, root, _, _ in ARMS:
        data[label], pockets[label] = load(root)
    ref, dropped = load_reference(REF_ROOT)

    # The two ends of x are where EVERY arm clears MIN_N. Inside them x stays contiguous
    # and an arm that dips below is broken by model_curve() instead, so one thin patch in
    # one arm no longer truncates the figure for all of them.
    firm = [a for a in sorted(set.intersection(*(set(d) for d in data.values())))
            if all(len(d.get(a, ())) >= MIN_N for d in data.values())]
    xs = list(range(firm[0], firm[-1] + 1))
    thin = {label: [a for a in xs if len(by.get(a, ())) < MIN_N]
            for label, by in data.items()}

    # The reference is computed at REF_FLOOR and narrowed per variant, so one array serves
    # all three gates.
    curves = {}
    for stat in STATS:
        curves[(REF_LABEL, stat)] = reference_curve(ref, xs, stat, REF_FLOOR)
        for label in data:
            curves[(label, stat)] = model_curve(data[label], xs, stat)
    counts = window_counts(ref, xs)

    exact, totals = {}, {}
    for label, by_size in list(data.items()) + [(REF_LABEL, ref)]:
        exact[label], totals[label] = share(by_size, xs)

    plt.rcParams.update(RC)
    print(f"=== {PROTOCOL} ===")
    print("  pockets: "
          + " · ".join(f"{lab} {pockets[lab]}" for lab, *_ in ARMS)
          + f" · reference {sum(len(v) for v in ref.values())}"
          + (f" (dropped {', '.join(dropped)})" if dropped else ""))
    print(f"  x = {xs[0]}..{xs[-1]} heavy atoms, all arms >= {MIN_N} molecules at both "
          f"ends; broken inside where an arm is not: "
          + (", ".join(f"{lab} at {v}" for lab, v in thin.items() if v) or "nowhere"))
    for label in [REF_LABEL] + [lab for lab, *_ in ARMS]:
        print(f"  {label:16s} n={totals[label]:5d}  "
              f"{sum(exact[label]):.1f}% of its molecules inside the plotted x range")
    for stat in STATS:
        both = [i for i in range(len(xs))
                if curves[(VOX_LABEL, stat)][i] is not None
                and curves[(OUR_LABEL, stat)][i] is not None]
        wins = [i for i in both
                if curves[(OUR_LABEL, stat)][i] <= curves[(VOX_LABEL, stat)][i]]
        gap = st.mean([curves[(OUR_LABEL, stat)][i] - curves[(VOX_LABEL, stat)][i]
                       for i in both])
        print(f"  {stat:6s}: CoDE lower than VoxBind at {len(wins)}/{len(both)} sizes, "
              f"mean per-size gap {gap:+.3f} kcal/mol")
    for label, by in [(REF_LABEL, ref)] + [(lab, data[lab]) for lab, *_ in ARMS]:
        vals = [n for n, v in by.items() for _ in v]
        print(f"  size of {label:16s} mean {st.mean(vals):5.2f}  "
              f"median {st.median(vals):5.1f} heavy atoms")
    for name, cfg in VARIANTS.items():
        firm_xs = [a for a, c in zip(xs, counts) if c >= REF_FIRM]
        if not cfg["ref_draw"]:
            print(f"\n{name}: no reference on either panel")
        else:
            tail = ("gated off beyond it" if cfg["ref_gate"] == "hard" else
                    f"faded to alpha {REF_THIN_ALPHA} outside it, down to "
                    f"{min(counts)} ligands")
            print(f"\n{name}: reference window >= {REF_FIRM} ligands at "
                  f"{firm_xs[0]}..{firm_xs[-1]} atoms, {tail}; distribution "
                  + ("rolled" if cfg["ref_dist_smooth"] else "per-count")
                  + (", filled" if cfg["ref_dist_fill"] else ", unfilled"))
        for stat in STATS:
            print(build(name, cfg, stat, xs, curves, exact, counts))
    print(f"\n{os.path.basename(write_csv(xs, data, ref, curves, exact, counts))}"
          f"  -- one table behind all six")


if __name__ == "__main__":
    main()
