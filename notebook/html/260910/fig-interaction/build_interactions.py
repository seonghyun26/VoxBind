#!/usr/bin/env python3
"""build_interactions.py — protein–ligand interaction fingerprints, four panels.

    interaction_core.*                one panel per interaction type; our arms only
    interaction_all.*                 the same four panels with every published baseline
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

TWO PANEL FORMS, AND THE SPLIT IS NOT COSMETIC. VdW and hydrophobic contacts run to 30 and
12 per pose, so their distribution has a body a KDE can describe and they stay violins. THE
TWO H-BOND TYPES DO NOT: they run 0-10 and 0-6, roughly half of every arm's molecules sit
at exactly zero, and a KDE over that invents a smooth hump through counts that do not exist
and a tail below zero. Those two are drawn as LETTER-VALUE PLOTS instead -- VoxBind's
Fig. 14(a),(b). Its SVG measures as five nested boxes per column at exactly halving widths
(21.39, 10.69, 5.35, 2.67, 1.34 pt), each box one quantile depth further out, all of them
sitting on the axis because their lower letter values are 0. The BOX WIDTH carries the
depth, not the box colour, which leaves the colour free to stay the method's own.

STYLE. `../pose_common.py`'s furniture, and the violin construction of
`../fig-posecheck/build_posecheck_all_by_atom_range.py`: KDE body, a thick inter-quartile
bar, a white median dot. The letter-value panels keep that same white median dot rather
than boxenplot's median line, so the mark that says "here is the middle" does not change
between panels of one figure.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-interaction/build_interactions.py
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import ALIASES, color                            # noqa: E402

# key -> panel name -> the form it is drawn in. Ordered by how common the type is, so the
# panel that is mostly a size proxy is read first and the specific ones after it. The two
# H-bond types are letter-value plots and the two contact types are violins; see the module
# docstring for why the same figure carries both.
TYPES = [
    ("VdWContact",  "van der Waals contacts",         "violin"),
    ("Hydrophobic", "Hydrophobic contacts",           "violin"),
    ("HBAcceptor",  "H-bonds accepted by the ligand", "boxen"),
    ("HBDonor",     "H-bonds donated by the ligand",  "boxen"),
]
TOTAL = "All interactions"
KINDS = [k for k, _, _ in TYPES] + [TOTAL]
TYPE = {k: (name, form) for k, name, form in TYPES}

# TWO FIGURES, ONE ROW OF TWO PANELS EACH, and the split is the one the forms already make:
# the contacts are violins and the H-bonds are letter-value blocks. Four panels under one
# caption meant a reader comparing arms on H-bonds had two violins in the same eyeful,
# arguing for a different reading of the same data at a different scale. They are separate
# measurements and they are now separate figures.
PARTS = [
    ("contacts", ("VdWContact", "Hydrophobic")),
    ("hbonds",   ("HBAcceptor", "HBDonor")),
]

# DISPLAY NAMES, for this folder's figures only. pose_common.REF_LABEL and ARMS carry the
# canonical spellings the whole 260910 section shares; renaming them there would move every
# other figure in it, so they are mapped here at draw time instead. Our arm is labelled
# \textsc{CoDE} -- the LaTeX form, verbatim, because these figures are placed in the paper
# and typeset there, and method_colors.ALIASES already resolves that spelling so neither
# the colour nor the drug-design row order depends on it. IT RENDERS LITERALLY in the PNG
# and the PDF; the SVG keeps live text (svg.fonttype: none) and is the copy that is
# typeset.
DISPLAY = {pc.REF_LABEL: "Reference", "Ours": r"\textsc{CoDE}", "CoDE": r"\textsc{CoDE}"}

# The published baselines, when compute_baseline_interactions.py has written them. Their
# fingerprints are not in posecheck_<Method>.json -- that export carries only strain and
# clashes -- so they are recomputed here from the poses in the results bundle, against the
# same pocket10 crop, by an interpreter checked to reproduce the local arms' stored
# fingerprints exactly. See ./README.md.
BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff"]

# THE ROW ORDER OF ../../260827/table_drug_design.tex, top to bottom. Every figure, the CSV,
# the JSON and the printed table in this folder use it, and none of them use a ranking or an
# order that falls out of how the data happened to load. The reader moves between that table
# and these panels; re-finding a method in a different order each time is a cost paid on
# every glance. The crystal ligands lead, exactly as the table's Reference row does. FuncBind
# is carried even though it has no fingerprint here (see ./README.md), so adding it later
# moves nothing.
ORDER = ("Reference ligand", "AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff",
         "VoxBind", "FuncBind", "CoDE")

# THE TYPE NAMES THE Y AXIS, and there are no panel titles -- which is the 260910 house
# style, and which this folder had to break while it was one 2x2 figure: all four panels
# plotted the same quantity in the same unit, so the y axis could not carry the type as
# well. Split into two figures of two panels, each panel's y axis is free again and the
# titles are gone. The names are long for a rotated label, so they take their own size
# rather than pc.furniture's 15.5.
Y_SIZE = 13.0

# GEOMETRY. One row of two panels off the shared panel height, plus a strip beneath the
# axes for the legend, and the WIDTH FALLS OUT OF A 4:1 FIGURE ASPECT rather than out of
# pc.FIG_W. Two panels side by side want a landscape frame: at the 3.4:2.4 the shared width
# gave them, eight columns were pressed together while the panel had height to spare above
# every violin.
#
# BOTH VARIANTS ARE THE SAME SIZE, which is what the single-row legend buys. The `all`
# variant's eight names needed two rows at the old width and made its figure 0.3 in taller
# than core's; at this width they fit on one, so core and all are the same frame and sit
# together in a document without a size change between them.
ASPECT = 4.0
ROW_H = pc.PANEL_H * 1.06
LEGEND_H = 0.46                                  # inches reserved under the axes
FIG_H = ROW_H + LEGEND_H
FIG_W = ASPECT * FIG_H
LEGEND_SIZE = 12.5
VIOLIN_W = 0.78
BODY_ALPHA = 0.55
IQR_LW, MED_MS = 5.0, 5.5
# Head-room above the tallest mark for the per-category mean/median caption. Two lines of
# 10pt over a panel of this height need about 12 % of it, so 1.17 clears the caption with a
# little air and no more: at the 1.34 the old 2x2 figure used, a full-height panel left the
# caption floating a third of the axes above the data it describes.
HEAD = 1.17
HEAD_BARE = 1.06                  # no caption to clear, so barely any head-room needed
# ONE ABSOLUTE KDE BANDWIDTH FOR EVERY VIOLIN, in counts, rather than matplotlib's default.
# The default (Scott) scales the bandwidth by n^(-1/5), and the crystal ligands are 79
# molecules against each arm's ~7,800 -- so the reference came out visibly smoother than
# every arm beside it for no reason but its sample size, which on a shape comparison is
# exactly the thing that must not differ. Fixed at 0.45 of a count, each integer still
# reads as its own bump.
KDE_BW = 0.45
# LETTER-VALUE GEOMETRY, off VoxBind's Fig. 14 SVG: five nested boxes whose width halves at
# each step, so the outermost is 1/16 of the innermost. Depth d spans the middle
# 1 - 2^-(d+1) of the distribution -- the quartiles, then the eighths, and so on out to the
# 1.6th and 98.4th percentiles. Five is Fig. 14's count, and it is also all this data can
# carry: the counts are integers running 0-10, so deeper boxes would repeat a bound already
# drawn. The shade ramp goes dark at the middle to pale at the tails, which is boxenplot's,
# but here it is a tint of the METHOD's colour -- the depth is already in the width.
BOXEN_K = 5
# HOW FAR TOWARD WHITE THE OUTERMOST BOX IS MIXED. Fig. 14 goes to 0.85 (#72b6a1 ->
# #eaf4f1), which on this palette left the two narrowest tiers invisible against the white
# ground -- and those tiers ARE the tail this figure is read for, since every arm's median
# is 0 or 1 and the whole difference between them lives above the quartiles. Held short of
# that, and shorter again since the blocks took the violins' alpha: BODY_ALPHA already
# lightens every tier, so a wide ramp on top of it washed the pale end out entirely.
BOXEN_LIGHT = 0.46
BOXEN_EDGE_LW = 0.8               # every block is outlined, in the axis pen


def ordered(labels):
    """`labels` in ORDER. Raises on a label the table does not name rather than appending
    it, for the same reason method_colors.color() raises: a method that silently drifts to
    the end of every figure is worse than a failed build.

    Spellings resolve through method_colors.ALIASES first, so the table's order survives a
    rename -- our method answers to "CoDE", "VoxBind + Ours", "Ours v1" and "ours_v1"
    alike, exactly as its colour does."""
    rank = {lab: i for i, lab in enumerate(ORDER)}
    canon = {l: ALIASES.get(l, l) for l in labels}
    unknown = [l for l, c in canon.items() if c not in rank]
    if unknown:
        raise KeyError(f"not in the drug-design table's order: {', '.join(unknown)}")
    return sorted(labels, key=lambda l: rank[canon[l]])


def series(arms):
    """(label, colour, values-getter key) in the drug-design table's row order, which puts
    the crystal ligands first -- and every other figure in 260910 draws them as the baseline
    the models are read against, so leftmost is where they belong anyway. `None` as the key
    marks the reference, which is not an arm."""
    key_of = {lab: key for lab, key, _ in arms}
    return [(lab, color(lab), key_of.get(lab))
            for lab in ordered([pc.REF_LABEL] + [lab for lab, _, _ in arms])]


def baseline_rows(label):
    """Rows shaped like pose_common's, from compute_baseline_interactions.py, or None if
    that step has not been run. Only `n` and `ifp` are used by this figure."""
    path = os.path.join(HERE, f"interactions_{label}.json")
    if not os.path.exists(path):
        return None
    doc = json.load(open(path))
    return [{"n": m["n"], "ifp": m["ifp"]} for m in doc["molecules"]]


def counts(rows, kind):
    """Per-molecule count of one interaction type over every molecule with a fingerprint.
    A type the molecule did not make is absent from the dict and counts as 0 — see the
    module docstring; this is the line that decides it."""
    if kind == TOTAL:
        return [sum(r["ifp"].values()) for r in rows if r["ifp"] is not None]
    return [r["ifp"].get(kind, 0) for r in rows if r["ifp"] is not None]


def display(label):
    """The name this folder's figures print for an arm. See DISPLAY."""
    return DISPLAY.get(label, label)


def cell_axis(ax, reach):
    """A count axis whose NUMBER SITS IN THE MIDDLE OF ITS BLOCK rather than on the rule
    between two of them. Fig. 14's does the same -- its SVG puts the tick stubs at +0.5,
    +1.5, +2.5 ... of a count off the baseline, never on the integers.

    THE CELLS ARE NUMBERED FROM 0, because 0 is a real count here and the commonest one:
    more than half of most arms' poses donate no hydrogen bond at all. Cell k is the band
    [k, k+1] and its number sits at k+0.5. The dotted rules stay on the integers, where the
    block edges are: a rule through the middle of a block would cut the thing the reader is
    counting. So THE RULES ARE THE COUNTS and a number names its own cell's LOWER edge —
    the topmost number in a column is one less than the count that column reaches.

    Labels stop at the highest count anyone reached, NOT at the panel top: the top is
    head-room for the caption, and numbering empty air both wastes the reader's attention
    and makes the captioned and uncaptioned variants of the same panel carry different
    axes."""
    ks = range(0, int(np.ceil(reach)))
    ax.yaxis.set_major_locator(FixedLocator([k + 0.5 for k in ks]))
    ax.yaxis.set_major_formatter(FixedFormatter([str(k) for k in ks]))
    ax.yaxis.set_minor_locator(FixedLocator([float(k) for k in range(1, ks.stop + 1)]))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(False, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", color=pc.GRID, lw=pc.GRID_LW, ls=(0, (1, 2.6)))


def frame(ax, vals, ylabel, top, caption, cells=None):
    """Everything both panel forms share: the limits, the house furniture, the y name and
    the per-category caption."""
    ax.set_ylim(bottom=-0.02 * top, top=top)
    pc.furniture(ax, ylabel=None, xlim=(0.4, len(vals) + 0.6), xloc=None)
    ax.set_ylabel(ylabel, fontsize=Y_SIZE, labelpad=10)
    if cells:
        cell_axis(ax, cells)
    else:
        # The violins are counts too, so their ticks are whole ones. The default locator
        # offered 0.0 / 2.5 / 5.0 on the hydrophobic panel the moment the panels grew to
        # full height -- a tick at two and a half contacts, beside a companion panel
        # stepping by whole ones.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # NO CATEGORY TICK LABELS. The legend beneath the figure is the key, and printing the
    # same eight names under each of two panels as well is the same information three
    # times -- which, rotated to fit, cost a third of the figure's height.
    ax.set_xticks([])
    ax.grid(False, axis="x")          # the x is categorical: a rule per slot is a fence
    # The per-category caption is dropped once there are eight of them: at that width
    # "mean 8.41" is wider than its own slot and the captions overprint each other. The
    # median dot still carries the location, and the numbers are in interactions.csv --
    # an unreadable overlap is worse than a table lookup.
    if caption:
        for i, v in enumerate(vals, start=1):
            ax.text(i, top, f"med {np.median(v):.0f}\nmean {np.mean(v):.2f}", ha="center",
                    va="top", fontsize=10, color=pc.AXIS, linespacing=1.3, zorder=6)


def violin_panel(ax, vals, cols, ylabel, caption=True):
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
    top = max(max(v) for v in vals) * (HEAD if caption else HEAD_BARE)
    frame(ax, vals, ylabel, top, caption)


def letter_values(v, k=BOXEN_K):
    """The k nested (lower, upper) bounds of a letter-value plot: the quartiles, then the
    eighths, and so on, each depth covering half the remaining tail. Depth d is the
    (100/2^(d+2))th and its mirror -- 25/75, 12.5/87.5, 6.25/93.75, 3.1/96.9, 1.6/98.4.

    EVERY BOUND IS AN OBSERVED COUNT (method="nearest"), not numpy's default interpolation
    between the two values a quantile falls between. These are counts: no pose accepts 1.5
    hydrogen bonds, and the default put the crystal ligands' quartiles at 0.5 and 2.5 purely
    because there are 79 of them and the quantile landed between two neighbours. On an axis
    of unit cells that also cut blocks half a count off the rules, so a block edge sat
    exactly where a cell's number sits. Snapped, every edge falls on a count and every block
    is a whole cell.

    The q25/q75 in interactions.csv and interactions.json are NOT snapped -- they stay the
    conventional interpolated quantiles, being a summary statistic rather than something
    drawn. Read the figure against the figure and the table against the table."""
    return [tuple(np.percentile(v, [100 / 2 ** (d + 2), 100 - 100 / 2 ** (d + 2)],
                                method="nearest"))
            for d in range(k)]


def letter_bands(v, k=BOXEN_K):
    """The same letter values as (lower, upper, depth) bands that DO NOT OVERLAP: depth 0
    is the quartile box, and each deeper level contributes only the step it adds above and
    below the one inside it.

    Drawn as nested boxes instead -- which is what boxenplot does, and what Fig. 14's SVG
    shows -- every level is a full rectangle from its lower to its upper bound, so the
    narrow pale ones overprint the middle of the wide dark ones. A horizontal slice through
    the column then runs dark at the flanks to pale at the centre, and that gradient encodes
    nothing: the depth is already in the width. Cut into bands, ONE HEIGHT OF THE COLUMN IS
    ONE WIDTH AND ONE SHADE, and the column reads as what it is -- a stack of blocks.

    A band of zero height is dropped rather than drawn, which is why a width can be skipped:
    where two letter values are equal, no height of the column belongs to the shallower of
    them."""
    lv = letter_values(v, k)
    bands = [(lv[0][0], lv[0][1], 0)]
    for d in range(1, k):
        bands.append((lv[d - 1][1], lv[d][1], d))     # the step this depth adds above
        bands.append((lv[d][0], lv[d - 1][0], d))     # and below -- empty when both are 0
    return cell_split(b for b in bands if b[1] > b[0])


def cell_split(bands):
    """Every band cut again at each integer it crosses, so ONE CELL OF THE AXIS IS ONE
    BLOCK — always, including inside a band that spans several counts.

    Without this a depth whose two letter values are three counts apart is drawn as one tall
    rectangle running straight through the rules at the counts between them, and the column
    stops being something you can count: two arms with the same top edge look different
    depending on where their quantiles happened to fall. Cut, every column is the same ladder
    of unit blocks and only the width and the shade of each rung differ. Nothing here rounds:
    the bounds arrive already sitting on counts (see letter_values), and this only cuts the
    bands between them."""
    out = []
    for lo, hi, d in bands:
        cuts = [lo] + [float(c) for c in range(int(np.floor(lo)) + 1, int(np.ceil(hi)))
                       if lo < c < hi] + [hi]
        out += [(a, b, d) for a, b in zip(cuts, cuts[1:]) if b > a]
    return out


def tint(col, t):
    """`col` mixed t of the way to white."""
    return tuple(c + (1.0 - c) * t for c in to_rgb(col))


def boxen_panel(ax, vals, cols, ylabel, caption=True):
    """VoxBind's Fig. 14(a),(b): nested boxes, one per quantile depth, each half the width
    of the one inside it and all centred on the category — but cut into non-overlapping
    bands, so one height of the column carries one width and one shade and nothing else
    (see letter_bands). WHERE THE LOWER LETTER VALUE IS
    ZERO the box sits on the axis — and on both H-bond types it is zero at every depth for
    every arm but one, because at least a quarter of every arm's molecules make none. So the
    column reads as a stack of blocks narrowing upward, each block one step further into the
    tail, which is exactly what a count distribution piled on zero is. (The one exception is
    the crystal ligands' quartile box on HBAcceptor, which starts at 1 rather than 0: only
    25.3 % of them accept none, so its p25 lands on the first molecule that does.)

    A LETTER-VALUE PLOT AND NOT A VIOLIN because these counts are small integers. A KDE has
    to smooth 0-10 into a continuum, which draws a body between the counts rather than on
    them, and the arms then differ in the shape of an artefact. Nested boxes are quantiles
    of the data and nothing else: no bandwidth, no interpolation between counts a molecule
    could not have made, and the zero-heavy left edge stays an edge."""
    ax.set_facecolor("white")
    reach = 0.0
    for i, (v, col) in enumerate(zip(vals, cols), start=1):
        for lo, hi, d in letter_bands(v):
            w = VIOLIN_W / 2 ** d     # the innermost band is as wide as a violin body
            # THE VIOLINS' SATURATION, so the two figures of this folder read as one set:
            # the alpha goes on the FACE only, as a fourth channel, rather than on the
            # patch -- a patch alpha would take the outline down with it, and the outline
            # is what makes a block a block.
            face = tint(col, BOXEN_LIGHT * d / (BOXEN_K - 1)) + (BODY_ALPHA,)
            ax.add_patch(Rectangle((i - w / 2, lo), w, hi - lo, zorder=3 + d,
                                   facecolor=face,
                                   edgecolor=pc.AXIS, linewidth=BOXEN_EDGE_LW))
            reach = max(reach, hi)
        # The violins' white median dot, not boxenplot's median line: the two forms sit in
        # one figure and the mark for "here is the middle" must not change between panels.
        # On these types the dot is often ON the axis, which is the finding — more than
        # half of most arms' poses donate no hydrogen bond at all.
        ax.plot(i, np.median(v), "o", color="white", ms=MED_MS, zorder=3 + BOXEN_K)
    top = reach * (HEAD if caption else HEAD_BARE)
    frame(ax, vals, ylabel, top, caption, cells=reach)


PANELS = {"violin": violin_panel, "boxen": boxen_panel}


def _row_major(n, ncol):
    """Indices that make matplotlib's column-major legend fill read left-to-right."""
    nrow = -(-n // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < n]
    return order


def legend_beneath(fig, names, cols, form, bottom):
    """The method key, under the panels instead of inside one of them — which is what lets
    the category tick labels go (see frame). The swatch is drawn the way that figure's own
    marks are: a translucent violin body for the contacts, a solid block in the axis pen
    for the letter values. A legend that does not look like the thing it names is a second
    key to learn."""
    if form == "violin":
        handles = [Patch(facecolor=c, alpha=BODY_ALPHA, edgecolor=c, lw=1.2,
                         label=display(n)) for n, c in zip(names, cols)]
    else:
        handles = [Patch(facecolor=to_rgb(c) + (BODY_ALPHA,), edgecolor=pc.AXIS,
                         lw=BOXEN_EDGE_LW, label=display(n))
                   for n, c in zip(names, cols)]
    # ONE ROW, which the 4:1 frame is wide enough for even at eight names. The reordering
    # below is what makes any other count safe: MATPLOTLIB FILLS A MULTI-ROW LEGEND
    # COLUMN-MAJOR, and on the two rows of four this used to need, that turned the
    # drug-design order into Reference, Pocket2Mol, TargetDiff, VoxBind across the top --
    # an order that is no order at all. Re-sequencing the handles so a column-major fill
    # lands them row-major is the only way back; there is no rcParam for it.
    ncol = len(names)
    handles = [handles[i] for i in _row_major(len(handles), ncol)]
    # The house legend box, as pc.legend() draws it on an axes: opaque white with a rule
    # in the axis pen, square corners. Quiet enough not to compete with the panels, and it
    # gives the key an edge of its own now that it sits in open figure margin.
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.012), ncol=ncol, frameon=True,
                     facecolor="white", edgecolor=pc.LEGEND_EDGE, framealpha=1.0,
                     borderpad=0.5, fontsize=LEGEND_SIZE, handlelength=1.5,
                     handleheight=1.0, handletextpad=0.6, columnspacing=1.9,
                     labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(pc.AXIS_LW)
    for t in leg.get_texts():
        t.set_color(pc.INK)
    return leg


def figure(part, kinds, rows_by_label, names, variant, caption):
    """One figure: a row of two panels of one form, with the key beneath them.

    `part` is in the filename beside the variant, for the reason the variant is: two
    figures that differ in what they draw must not be able to sit in a folder under one
    name."""
    cols = [color(n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        ylabel, form = TYPE[kind]
        vals = [counts(rows_by_label[n], kind) for n in names]
        PANELS[form](ax, vals, cols, ylabel, caption)
    # The legend strip is RESERVED with tight_layout's rect and the legend drawn into it
    # afterwards: tight_layout does not measure a figure-level legend, so laying the axes
    # out first and adding it second would print it over the x spine.
    pc.fit(fig, pad=0.5, w_pad=2.2, rect=(0, LEGEND_H / FIG_H, 1, 1))
    legend_beneath(fig, names, cols, form, LEGEND_H / FIG_H)
    pc.save(fig, HERE, f"interaction_{part}", variant)


# ── exports ──────────────────────────────────────────────────────────────────────
def block(rows):
    out = {"n_molecules": len(rows),
           "n_scored": sum(r["ifp"] is not None for r in rows),
           "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None}
    for kind in KINDS:
        v = counts(rows, kind)
        out[kind] = {"mean": round(float(np.mean(v)), 3), "median": float(np.median(v)),
                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                     "share_nonzero": round(100 * float(np.mean(np.asarray(v) > 0)), 2)}
    return out


def ordered_arms(arms):
    """(label, key, root) triples in the drug-design table's order."""
    by_label = {a[0]: a for a in arms}
    return [by_label[l] for l in ordered(list(by_label))]


def table_rows(summary):
    """Every method's p79 block, in the drug-design table's row order — the order the
    figures draw, so the CSV, the printed table and the panels can be read against each
    other without re-sorting any of them."""
    blocks = {lab: b["p79"] for lab, b in summary["arms"].items()}
    blocks.update(summary["baselines"])
    blocks[pc.REF_LABEL] = summary["reference_ligand"]
    return [(lab, blocks[lab]) for lab in ordered(list(blocks))]


def exports(data, p79_rows, refrows, arms, base=None):
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
        "types": KINDS,
        "arms": {}, "baselines": {}, "reference_ligand": block(refrows),
    }
    for lab in ordered(list(base or {})):
        summary["baselines"][lab] = {
            "source": f"interactions_{lab}.json — recomputed from the results bundle",
            **block(base[lab])}
    # `arms` is what pose_common says carries a fingerprint over the whole pocket set --
    # NOT every entry of pc.ARMS. Since 2026-09-10 the five published baselines are ARMS
    # too, staged under exps/baselines_pose/, but they hold PoseBusters and not PoseCheck;
    # blocking one here would export a row of nulls that looks measured and lost.
    for lab, key, root in ordered_arms(arms):
        summary["arms"][lab] = {
            "root": root, "key": key,
            "p79": block(p79_rows[key]),
            "all_pockets": block([r for t in sorted(data[key]) for r in data[key][t]]),
        }
    json.dump(summary, open(os.path.join(HERE, "interactions.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "interactions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "type", "mean", "median", "q25", "q75", "share_nonzero_pct",
                    "n_scored"])
        for lab, b in table_rows(summary):
            for k in KINDS:
                w.writerow([lab, k, b[k]["mean"], b[k]["median"], b[k]["q25"], b[k]["q75"],
                            b[k]["share_nonzero"], b["n_scored"]])
    return summary


def main():
    pc.use_style()
    data, p79_rows, refrows = pc.load_arms()
    # ARMS carries every arm the 260910 section knows; only some of them have been scored
    # with PoseCheck. `arms_for` keeps an arm only where all 79 pockets have a scored
    # molecule, so an arm mid-scoring cannot enter a figure as a complete-looking column.
    scored = pc.arms_for("ifp", data)

    # The published baselines come from THIS folder's recompute, not from pc.ARMS: their
    # staged sample dirs hold PoseBusters and not PoseCheck (see ./README.md). A name that
    # has become a scored arm is taken from the arm, so no method can enter twice.
    live = {lab for lab, _, _ in scored}
    base = {l: r for l in BASELINES if l not in live
            for r in [baseline_rows(l)] if r is not None}
    by_label = {lab: p79_rows[key] for lab, key, _ in scored}
    by_label.update(base)
    by_label[pc.REF_LABEL] = refrows

    core = ordered([pc.REF_LABEL] + [lab for lab, key, _ in scored if key in pc.CORE])
    every = ordered(list(by_label))
    for part, kinds in PARTS:
        figure(part, kinds, by_label, core, "core", caption=True)
        figure(part, kinds, by_label, every, "all", caption=False)
    missing = [l for l in BASELINES if l not in base and l not in live]

    summary = exports(data, p79_rows, refrows, scored, base)

    rows = table_rows(summary)
    print(f"{len(pc.P79)}-pocket set · interaction fingerprint, pocket10 crop · "
          f"mean per molecule\n")
    print(f"{'arm':16s} {'atoms':>6s} {'mols':>6s} " + " ".join(f"{k:>12s}" for k in KINDS))
    for lab, b in rows:
        print(f"{lab:16s} {b['atoms_mean']:6.1f} {b['n_scored']:6d} "
              + " ".join(f"{b[k]['mean']:12.2f}" for k in KINDS))
    print(f"\n{'share of molecules making the type at all':>30s}")
    print(f"{'arm':16s} {'':6s} {'':6s} " + " ".join(f"{k:>12s}" for k in KINDS))
    for lab, b in rows:
        print(f"{lab:16s} {'':6s} {'':6s} "
              + " ".join(f"{b[k]['share_nonzero']:11.1f}%" for k in KINDS))
    print("\nEvery type scales with ligand size (see the atoms column): read each arm "
          "against\nthe crystal ligands, not against the arm above it.")
    if missing:
        print(f"\nno interactions_<M>.json for: {', '.join(missing)} — run "
              f"stage_baseline_poses.py then compute_baseline_interactions.py")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
