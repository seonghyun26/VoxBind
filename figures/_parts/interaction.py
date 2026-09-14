

# ════════════════════════════════════════════════════════════════════════════════
# fig-interaction-{contacts,hbonds,pair} — ProLIF interaction fingerprints
# ════════════════════════════════════════════════════════════════════════════════
# The VoxBind paper's Fig. 14 form, and the fourth pose-quality measure of this section:
# strain and clashes are fig-posecheck-*, PoseBusters validity is fig-posebusters-*,
# rigid-fragment consistency is fig-consistency-*, and this is what the pose actually DOES
# against the pocket. Same 79 pockets, same arms, same colours.
#
# THE METRIC. An interaction fingerprint records, per pose, which contacts it makes with the
# receptor. PoseCheck runs ProLIF over the protonated pocket and the pose as generated; see
# Harris et al., 2023 (the PoseCheck paper) for the definitions. Four types appear in this
# data:
#
#     VdWContact    van der Waals contact — by far the most common, and mostly a size proxy
#     Hydrophobic   apolar contact
#     HBAcceptor    the ligand accepts a hydrogen bond from the protein
#     HBDonor       the ligand donates one to the protein
#
# MORE IS NOT AUTOMATICALLY BETTER, and this figure is not a leaderboard. A bigger ligand
# makes more of every type, so an arm that generates larger molecules scores higher on all
# four without binding better; the crystal ligands are the reference precisely because they
# are the distribution a real binder draws from, and the useful reading is which arm sits
# CLOSEST to them, not which sits highest. The per-molecule counts are exported so a
# size-matched comparison can be made — the same trap the Vina numbers have already been
# bitten by on this project, on a different metric.
#
# A TYPE A MOLECULE DOES NOT MAKE IS ABSENT FROM ITS FINGERPRINT, NOT ZERO IN IT. Every
# molecule scored by PoseCheck therefore contributes a 0 to every type it did not make, and
# the violins are over all scored molecules, not over the ones that happened to make that
# type. Counting only the latter would turn "Ours makes H-bond donors less often" into "Ours
# makes more of them when it does", which is a different and much smaller claim.
#
# THE POCKET IS THE pocket10 CROP that the target `metrics.json` files carry, the same
# source fig-posecheck-per-atom uses — NOT the whole-receptor scoring in
# `frozenenc_probes/posecheck_full/`. That scoring exists and covers all 79 pockets for all
# three arms, but it has `reference: null`, so it cannot draw the crystal ligands, which are
# the whole point of the comparison. The cost of using the crop was measured over the 21,865
# molecules the two scorings share rather than assumed: per molecule the crop finds 0.16-0.29
# fewer VdWContacts (1.8-3.2 %) and within 0.03 of the same count on all three other types.
# A 10 Å crop around the crystal ligand contains essentially everything within ProLIF's
# interaction cutoffs. Do not, however, read a number here against one from
# fig-posecheck-by-size, which IS whole-receptor.
#
# TWO PANEL FORMS, AND THE SPLIT IS NOT COSMETIC. VdW and hydrophobic contacts run to 30 and
# 12 per pose, so their distribution has a body a KDE can describe and they stay violins. THE
# TWO H-BOND TYPES DO NOT: they run 0-10 and 0-6, roughly half of every arm's molecules sit
# at exactly zero, and a KDE over that invents a smooth hump through counts that do not exist
# and a tail below zero. Those two are drawn as LETTER-VALUE PLOTS instead -- VoxBind's
# Fig. 14(a),(b). Its SVG measures as five nested boxes per column at exactly halving widths
# (21.39, 10.69, 5.35, 2.67, 1.34 pt), each box one quantile depth further out, all of them
# sitting on the axis because their lower letter values are 0. The BOX WIDTH carries the
# depth, not the box colour, which leaves the colour free to stay the method's own.
#
# STYLE. The shared furniture, and the violin construction of fig-posecheck-by-size: KDE
# body, a thick inter-quartile bar, a white median dot. The letter-value panels keep that
# same white median dot rather than boxenplot's median line, so the mark that says "here is
# the middle" does not change between panels of one figure.

# Names this family needs that the shared header does not import. Reached through the
# already-imported `matplotlib` package rather than re-imported, because a part file adds no
# import statements of its own.
IX_RECTANGLE = matplotlib.patches.Rectangle
IX_TO_RGB = matplotlib.colors.to_rgb
IX_FIXED_LOCATOR = matplotlib.ticker.FixedLocator
IX_FIXED_FORMATTER = matplotlib.ticker.FixedFormatter

# key -> panel name -> the form it is drawn in. Ordered by how common the type is, so the
# panel that is mostly a size proxy is read first and the specific ones after it. The two
# H-bond types are letter-value plots and the two contact types are violins; see the banner
# above for why the same family carries both.
IX_TYPES = [
    ("VdWContact",  "van der Waals contacts",         "violin"),
    ("Hydrophobic", "Hydrophobic contacts",           "violin"),
    # SHORT NAMES (2026-09-14). "H-bonds accepted by the ligand" did not survive the axis
    # name going to the ECDF's 16.1 pt: this family draws to a FIXED frame at dpi 220 with no
    # crop to the ink, so the tail of a long name falls off the canvas rather than pushing it
    # wider -- the rendered panels read "...by the liga" and "...by the ligar". The contacts
    # names were short enough to survive, which is why only these two were cut.
    # ONLY THE DISPLAY NAME CHANGES. The keys still drive IX_KINDS, the summary table and the
    # `type` column of interactions.csv, so nothing downstream moves.
    ("HBAcceptor",  "H-bonds acceptor",               "boxen"),
    ("HBDonor",     "H-bonds donor",                  "boxen"),
]
IX_TOTAL = "All interactions"
IX_KINDS = [k for k, _, _ in IX_TYPES] + [IX_TOTAL]
IX_TYPE = {k: (name, form) for k, name, form in IX_TYPES}

# TWO FIGURES, ONE ROW OF TWO PANELS EACH, and the split is the one the forms already make:
# the contacts are violins and the H-bonds are letter-value blocks. Four panels under one
# caption meant a reader comparing arms on H-bonds had two violins in the same eyeful,
# arguing for a different reading of the same data at a different scale. They are separate
# measurements and they are now separate figures.
IX_PARTS = [
    ("contacts", ("VdWContact", "Hydrophobic")),
    ("hbonds",   ("HBAcceptor", "HBDonor")),
]

# DISPLAY NAMES, for this family's figures only. REF_LABEL and ARMS carry the canonical
# spellings the whole section shares; renaming them there would move every other figure in
# it, so they are mapped here at draw time instead. Our arm is labelled \textsc{CoDE} -- the
# LaTeX form, verbatim, because these figures are placed in the paper and typeset there, and
# ALIASES already resolves that spelling so neither the colour nor the drug-design row order
# depends on it. IT RENDERS LITERALLY in the PNG and the PDF; the SVG keeps live text
# (svg.fonttype: none) and is the copy that is typeset.
# Only the crystal ligands get a name of this family's own; everything else goes through the
# shared display(), so a method is spelled here exactly as the ECDF and the clash box spell
# it. The entries that used to live here were r"\textsc{CoDE}" -- LaTeX that nothing in this
# pipeline renders, so the legend printed the six characters \textsc literally. It was
# invisible while the names lived only in a legend nobody re-read; it would have gone
# straight onto the x axis the moment the names moved there.
IX_DISPLAY = {REF_LABEL: "Reference"}
# The method names on the x axis. Smaller than the ECDF's axis-name size: eight of them share
# one panel here, where that figure spends its 16.1 on a single line of axis title.
IX_NAME_FS = 12.5
# WHICH FIGURE CARRIES THE NAMES (2026-09-14). Both figures draw the same methods in the same
# order, so stacked -- contacts over hbonds -- one set of names underneath serves both, and
# printing them twice is the same information twice. Only the bottom figure is named.
# THIS MAKES contacts DEPENDENT ON ITS NEIGHBOUR: on its own, nothing in it says which
# violin is which. If it is ever published alone, put "contacts" back in this tuple.
IX_NAMED_PARTS = ("hbonds",)

# The published baselines, when compute_baseline_interactions.py has written them. Their
# fingerprints are not in posecheck_<Method>.json -- that export carries only strain and
# clashes -- so they are recomputed from the poses in the results bundle, against the same
# pocket10 crop, by an interpreter checked to reproduce the local arms' stored fingerprints
# exactly. See notebook/html/260910/fig-interaction/README.md.
IX_BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff"]

# THE ROW ORDER OF 260827/table_drug_design.tex, top to bottom. Every figure and the CSV of
# this family use it, and none of them use a ranking or an order that falls out of how the
# data happened to load. The reader moves between that table and these panels; re-finding a
# method in a different order each time is a cost paid on every glance. The crystal ligands
# lead, exactly as the table's Reference row does. FuncBind is carried even though it has no
# fingerprint here, so adding it later moves nothing.
IX_ORDER = ("Reference ligand", "AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff",
            "VoxBind", "FuncBind", "CoDE")

# THE TYPE NAMES THE Y AXIS, and there are no panel titles -- which is the house style, and
# which this family had to break while it was one 2x2 figure: all four panels plotted the
# same quantity in the same unit, so the y axis could not carry the type as well. Split into
# two figures of two panels, each panel's y axis is free again and the titles are gone. The
# names are long for a rotated label, so they take their own size rather than furniture()'s
# 15.5.
IX_Y_SIZE = 13.0

# GEOMETRY. One row of two panels off the shared panel height, plus a strip beneath the
# axes for the legend, and the WIDTH FALLS OUT OF A 4:1 FIGURE ASPECT rather than out of
# FIG_W. Two panels side by side want a landscape frame: at the 3.4:2.4 the shared width
# gave them, eight columns were pressed together while the panel had height to spare above
# every violin.
#
# BOTH VARIANTS ARE THE SAME SIZE, which is what the single-row legend buys. The `all`
# variant's eight names needed two rows at the old width and made its figure 0.3 in taller
# than core's; at this width they fit on one, so core and all are the same frame and sit
# together in a document without a size change between them.
IX_ASPECT = 4.0
IX_ROW_H = PANEL_H * 1.06
IX_LEGEND_H = 0.46                               # inches reserved under the axes
IX_FIG_H = IX_ROW_H + IX_LEGEND_H
IX_FIG_W = IX_ASPECT * IX_FIG_H
IX_LEGEND_SIZE = 12.5
IX_VIOLIN_W = 0.78
IX_BODY_ALPHA = 0.55
IX_IQR_LW, IX_MED_MS = 5.0, 5.5
# Head-room above the tallest mark for the per-category mean/median caption. Two lines of
# 10pt over a panel of this height need about 12 % of it, so 1.17 clears the caption with a
# little air and no more: at the 1.34 the old 2x2 figure used, a full-height panel left the
# caption floating a third of the axes above the data it describes.
IX_HEAD = 1.17
IX_HEAD_BARE = 1.06               # no caption to clear, so barely any head-room needed
# ONE ABSOLUTE KDE BANDWIDTH FOR EVERY VIOLIN, in counts, rather than matplotlib's default.
# The default (Scott) scales the bandwidth by n^(-1/5), and the crystal ligands are 79
# molecules against each arm's ~7,800 -- so the reference came out visibly smoother than
# every arm beside it for no reason but its sample size, which on a shape comparison is
# exactly the thing that must not differ. Fixed at 0.45 of a count, each integer still
# reads as its own bump.
IX_KDE_BW = 0.45
# LETTER-VALUE GEOMETRY, off VoxBind's Fig. 14 SVG: five nested boxes whose width halves at
# each step, so the outermost is 1/16 of the innermost. Depth d spans the middle
# 1 - 2^-(d+1) of the distribution -- the quartiles, then the eighths, and so on out to the
# 1.6th and 98.4th percentiles. Five is Fig. 14's count, and it is also all this data can
# carry: the counts are integers running 0-10, so deeper boxes would repeat a bound already
# drawn. The shade ramp goes dark at the middle to pale at the tails, which is boxenplot's,
# but here it is a tint of the METHOD's colour -- the depth is already in the width.
IX_BOXEN_K = 5
# HOW FAR TOWARD WHITE THE OUTERMOST BOX IS MIXED. Fig. 14 goes to 0.85 (#72b6a1 ->
# #eaf4f1), which on this palette left the two narrowest tiers invisible against the white
# ground -- and those tiers ARE the tail this figure is read for, since every arm's median
# is 0 or 1 and the whole difference between them lives above the quartiles. Held short of
# that, and shorter again since the blocks took the violins' alpha: IX_BODY_ALPHA already
# lightens every tier, so a wide ramp on top of it washed the pale end out entirely.
IX_BOXEN_LIGHT = 0.46
IX_BOXEN_EDGE_LW = 0.8            # every block is outlined, in the axis pen


def _interaction_ordered(labels):
    """`labels` in IX_ORDER. Raises on a label the table does not name rather than appending
    it, for the same reason color() raises: a method that silently drifts to the end of
    every figure is worse than a failed build.

    Spellings resolve through ALIASES first, so the table's order survives a rename -- our
    method answers to "CoDE", "VoxBind + Ours", "Ours v1" and "ours_v1" alike, exactly as
    its colour does."""
    rank = {lab: i for i, lab in enumerate(IX_ORDER)}
    canon = {l: ALIASES.get(l, l) for l in labels}
    unknown = [l for l, c in canon.items() if c not in rank]
    if unknown:
        raise KeyError(f"not in the drug-design table's order: {', '.join(unknown)}")
    return sorted(labels, key=lambda l: rank[canon[l]])


def _interaction_baseline_rows(label):
    """Rows shaped like the pose loader's, from compute_baseline_interactions.py, or None if
    that step has not been run. Only `n` and `ifp` are used by this figure.

    Read off LEGACY directly rather than through legacy(): this input is OPTIONAL -- a
    baseline without a recompute is reported as a missing name at the end of the run, not as
    a failed build, because the figure it would join is complete without it."""
    path = LEGACY / "fig-interaction" / f"interactions_{label}.json"
    if not path.exists():
        return None
    doc = json.load(open(path))
    return [{"n": m["n"], "ifp": m["ifp"]} for m in doc["molecules"]]


def _interaction_counts(rows, kind):
    """Per-molecule count of one interaction type over every molecule with a fingerprint.
    A type the molecule did not make is absent from the dict and counts as 0 — see the
    banner above; this is the line that decides it."""
    if kind == IX_TOTAL:
        return [sum(r["ifp"].values()) for r in rows if r["ifp"] is not None]
    return [r["ifp"].get(kind, 0) for r in rows if r["ifp"] is not None]


def _interaction_display(label):
    """The name this family's figures print for an arm. See IX_DISPLAY."""
    return IX_DISPLAY.get(label) or display(label)


def _interaction_cell_axis(ax, reach):
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
    ax.yaxis.set_major_locator(IX_FIXED_LOCATOR([k + 0.5 for k in ks]))
    ax.yaxis.set_major_formatter(IX_FIXED_FORMATTER([str(k) for k in ks]))
    ax.yaxis.set_minor_locator(IX_FIXED_LOCATOR([float(k) for k in range(1, ks.stop + 1)]))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(False, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", color=GRID, lw=GRID_LW, ls=DOT)


def _interaction_frame(ax, vals, ylabel, top, caption, cells=None):
    """Everything both panel forms share: the limits, the house furniture, the y name and
    the per-category caption."""
    ax.set_ylim(bottom=-0.02 * top, top=top)
    furniture(ax, ylabel=None, xlim=(0.4, len(vals) + 0.6), xloc=None)
    ax.set_ylabel(ylabel, fontsize=IX_Y_SIZE, labelpad=10)
    if cells:
        _interaction_cell_axis(ax, cells)
    else:
        # The violins are counts too, so their ticks are whole ones. The default locator
        # offered 0.0 / 2.5 / 5.0 on the hydrophobic panel the moment the panels grew to
        # full height -- a tick at two and a half contacts, beside a companion panel
        # stepping by whole ones.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Cleared here and PUT BACK BY THE CALLER (2026-09-14). This used to be the end of it:
    # the legend under the figure was the key, so printing the names under the panels too
    # was the same information twice. With the legend gone the names are the only way to
    # tell one violin from another, so _interaction_ecdf_furniture sets them.
    ax.set_xticks([])
    ax.grid(False, axis="x")          # the x is categorical: a rule per slot is a fence
    # The per-category caption is dropped once there are eight of them: at that width
    # "mean 8.41" is wider than its own slot and the captions overprint each other. The
    # median dot still carries the location, and the numbers are in interactions.csv --
    # an unreadable overlap is worse than a table lookup.
    if caption:
        for i, v in enumerate(vals, start=1):
            ax.text(i, top, f"med {np.median(v):.0f}\nmean {np.mean(v):.2f}", ha="center",
                    va="top", fontsize=10, color=AXIS, linespacing=1.3, zorder=6)


def _interaction_violin_panel(ax, vals, cols, ylabel, caption=True):
    ax.set_facecolor("white")
    parts = ax.violinplot(vals, showextrema=False, widths=IX_VIOLIN_W,
                          bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
    for body, col in zip(parts["bodies"], cols):
        # THE PAIR FIGURE'S PAINTING (2026-09-14): the outline is the body's OWN colour, which
        # is what _interaction_pair_bodies has always done. Briefly removed earlier the same
        # day and put back so that all four interaction panels are painted by one rule --
        # face a translucent tint, stroke the saturated hue of the same method.
        body.set(facecolor=col, alpha=IX_BODY_ALPHA, edgecolor=col, linewidth=1.2)
    for i, v in enumerate(vals, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=IX_IQR_LW, zorder=3)
        ax.plot(i, med, "o", color="white", ms=IX_MED_MS, zorder=4)

    # The top is the largest count anyone actually made, plus head-room for the caption.
    # NOT a percentile: cutting at p99.5 sliced the KDE tails off mid-body and left a row
    # of hard vertical edges along the top of the panel that read as real structure.
    # A count cannot be negative either, but a KDE fitted to a zero-heavy distribution puts
    # a tail below zero anyway; the axis cuts it rather than implying a count of -1 exists.
    top = max(max(v) for v in vals) * (IX_HEAD if caption else IX_HEAD_BARE)
    _interaction_frame(ax, vals, ylabel, top, caption)


def _interaction_letter_values(v, k=IX_BOXEN_K):
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

    The q25/q75 in interactions.csv are NOT snapped -- they stay the conventional
    interpolated quantiles, being a summary statistic rather than something drawn. Read the
    figure against the figure and the table against the table."""
    return [tuple(np.percentile(v, [100 / 2 ** (d + 2), 100 - 100 / 2 ** (d + 2)],
                                method="nearest"))
            for d in range(k)]


def _interaction_letter_bands(v, k=IX_BOXEN_K):
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
    lv = _interaction_letter_values(v, k)
    bands = [(lv[0][0], lv[0][1], 0)]
    for d in range(1, k):
        bands.append((lv[d - 1][1], lv[d][1], d))     # the step this depth adds above
        bands.append((lv[d][0], lv[d - 1][0], d))     # and below -- empty when both are 0
    return _interaction_cell_split(b for b in bands if b[1] > b[0])


def _interaction_cell_split(bands):
    """Every band cut again at each integer it crosses, so ONE CELL OF THE AXIS IS ONE
    BLOCK — always, including inside a band that spans several counts.

    Without this a depth whose two letter values are three counts apart is drawn as one tall
    rectangle running straight through the rules at the counts between them, and the column
    stops being something you can count: two arms with the same top edge look different
    depending on where their quantiles happened to fall. Cut, every column is the same ladder
    of unit blocks and only the width and the shade of each rung differ. Nothing here rounds:
    the bounds arrive already sitting on counts (see _interaction_letter_values), and this
    only cuts the bands between them."""
    out = []
    for lo, hi, d in bands:
        cuts = [lo] + [float(c) for c in range(int(np.floor(lo)) + 1, int(np.ceil(hi)))
                       if lo < c < hi] + [hi]
        out += [(a, b, d) for a, b in zip(cuts, cuts[1:]) if b > a]
    return out


def _interaction_tint(col, t):
    """`col` mixed t of the way to white."""
    return tuple(c + (1.0 - c) * t for c in IX_TO_RGB(col))


def _interaction_boxen_panel(ax, vals, cols, ylabel, caption=True):
    """VoxBind's Fig. 14(a),(b): nested boxes, one per quantile depth, each half the width
    of the one inside it and all centred on the category — but cut into non-overlapping
    bands, so one height of the column carries one width and one shade and nothing else
    (see _interaction_letter_bands). WHERE THE LOWER LETTER VALUE IS
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
        for lo, hi, d in _interaction_letter_bands(v):
            w = IX_VIOLIN_W / 2 ** d  # the innermost band is as wide as a violin body
            # THE VIOLINS' SATURATION, so the two figures of this family read as one set:
            # the alpha goes on the FACE only, as a fourth channel, rather than on the patch.
            #
            # THE OUTLINE STAYS, and it is not decoration: the outline is what makes a block
            # a block. Taken off on 2026-09-14 and PUT BACK the same day, because the column
            # collapsed into one stepped silhouette -- the bands are NON-OVERLAPPING and depth
            # is carried by width and tint, but IX_BOXEN_LIGHT spreads only 0.46 over five
            # depths, so without a rule between them the quartile box and the tail bands
            # stopped being separable.
            # IN THE METHOD'S OWN COLOUR, not the neutral AXIS grey it used to take: that is
            # the rule the pair figure paints by, and all four panels of this family now
            # share it -- face a translucent tint, stroke the saturated hue of the same
            # method.
            #
            # The alpha rides on the FACE as a fourth channel rather than on the patch: a
            # patch alpha would fade the outline with it.
            face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                + (IX_BODY_ALPHA,)
            ax.add_patch(IX_RECTANGLE((i - w / 2, lo), w, hi - lo, zorder=3 + d,
                                      facecolor=face,
                                      edgecolor=col, linewidth=IX_BOXEN_EDGE_LW))
            reach = max(reach, hi)
        # The violins' white median dot, not boxenplot's median line: the two forms sit in
        # one figure and the mark for "here is the middle" must not change between panels.
        # On these types the dot is often ON the axis, which is the finding — more than
        # half of most arms' poses donate no hydrogen bond at all.
        ax.plot(i, np.median(v), "o", color="white", ms=IX_MED_MS, zorder=3 + IX_BOXEN_K)
    top = reach * (IX_HEAD if caption else IX_HEAD_BARE)
    _interaction_frame(ax, vals, ylabel, top, caption, cells=reach)


IX_PANELS = {"violin": _interaction_violin_panel, "boxen": _interaction_boxen_panel}


def _interaction_row_major(n, ncol):
    """Indices that make matplotlib's column-major legend fill read left-to-right."""
    nrow = -(-n // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < n]
    return order


def _interaction_legend(fig, names, cols, form):
    """The method key, under the panels instead of inside one of them — which is what lets
    the category tick labels go (see _interaction_frame). The swatch is drawn the way that
    figure's own marks are: a translucent violin body for the contacts, a solid block in the
    axis pen for the letter values. A legend that does not look like the thing it names is a
    second key to learn."""
    if form == "violin":
        handles = [Patch(facecolor=c, alpha=IX_BODY_ALPHA, edgecolor=c, lw=1.2,
                         label=_interaction_display(n)) for n, c in zip(names, cols)]
    else:
        handles = [Patch(facecolor=IX_TO_RGB(c) + (IX_BODY_ALPHA,), edgecolor=AXIS,
                         lw=IX_BOXEN_EDGE_LW, label=_interaction_display(n))
                   for n, c in zip(names, cols)]
    # ONE ROW, which the 4:1 frame is wide enough for even at eight names. The reordering
    # below is what makes any other count safe: MATPLOTLIB FILLS A MULTI-ROW LEGEND
    # COLUMN-MAJOR, and on the two rows of four this used to need, that turned the
    # drug-design order into Reference, Pocket2Mol, TargetDiff, VoxBind across the top --
    # an order that is no order at all. Re-sequencing the handles so a column-major fill
    # lands them row-major is the only way back; there is no rcParam for it.
    ncol = len(names)
    handles = [handles[i] for i in _interaction_row_major(len(handles), ncol)]
    # The house legend box, as legend() draws it on an axes: opaque white with a rule in the
    # axis pen, square corners. Quiet enough not to compete with the panels, and it gives
    # the key an edge of its own now that it sits in open figure margin.
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.012), ncol=ncol, frameon=True,
                     facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0,
                     borderpad=0.5, fontsize=IX_LEGEND_SIZE, handlelength=1.5,
                     handleheight=1.0, handletextpad=0.6, columnspacing=1.9,
                     labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _interaction_ecdf_furniture(ax, names, show_names=True):
    """The strain ECDF's axis weight and type, plus the method names back on the x axis.

    `show_names` is False for the figure that sits ABOVE another carrying the same
    categories -- see IX_NAMED_PARTS.

    ONLY THE SPINES, TICKS AND AXIS NAME MOVE. Not _pcsz_style() wholesale: that turns the
    major y grid back on, and the H-bond panel runs a deliberate minor-grid arrangement
    (_interaction_cell_axis) that would lose. Not PCSZ_RC either -- it carries savefig.dpi
    170 and a crop to the ink, and this family draws at 220 to its own frame.

    PCSZ_* are read at CALL time: they live in a part assembled after this one."""
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(labelsize=11, direction="out", length=3.5, width=1.1, pad=4,
                   colors=PCSZ_AXIS)
    ax.yaxis.label.set_fontsize(PCSZ_LABEL_FS)
    ax.yaxis.label.set_color(PCSZ_AXIS)
    # The violins sit at 1..n, which is where _interaction_frame's xlim is built from.
    ax.set_xticks(range(1, len(names) + 1))
    labels = ax.set_xticklabels([_interaction_display(n) for n in names],
                                fontsize=IX_NAME_FS, rotation=30, ha="right",
                                rotation_mode="anchor")
    if not show_names:
        # DRAWN BUT INVISIBLE, rather than absent. tight_layout measures a text's extent
        # whatever its colour, so colouring the names "none" reserves the exact band of
        # height they would have taken -- which is what keeps this figure's axes the same
        # height as the named one it sits above. Dropping the labels instead let fit() hand
        # that band to the axes, and the two panels came out with their bottom spines 174 px
        # apart inside a canvas they share, so stacked they would have carried different
        # vertical scales for the same categories.
        # The tick MARKS stay: they say where a category is without naming it.
        for t in labels:
            t.set_color("none")


def _interaction_panels(out, part, kinds, rows_by_label, names, variant, caption):
    """One figure: a row of two panels of one form, with the key beneath them.

    `part` is in the filename beside the variant, for the reason the variant is: two
    figures that differ in what they draw must not be able to sit in a folder under one
    name."""
    cols = [color(n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(IX_FIG_W, IX_FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        ylabel, form = IX_TYPE[kind]
        vals = [_interaction_counts(rows_by_label[n], kind) for n in names]
        IX_PANELS[form](ax, vals, cols, ylabel, caption)
        _interaction_ecdf_furniture(ax, names, part in IX_NAMED_PARTS)
    # NO LEGEND, and so no reserved strip: the names are on the x axis now, and the rect that
    # used to hold the key back would leave an empty band under the panels.
    fit(fig, pad=0.5, w_pad=2.2)
    save(fig, out, f"interaction_{part}_{variant}")


# ── the numbers behind the panels ────────────────────────────────────────────────
def _interaction_block(rows):
    out = {"n_molecules": len(rows),
           "n_scored": sum(r["ifp"] is not None for r in rows),
           "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None}
    for kind in IX_KINDS:
        v = _interaction_counts(rows, kind)
        out[kind] = {"mean": round(float(np.mean(v)), 3), "median": float(np.median(v)),
                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                     "share_nonzero": round(100 * float(np.mean(np.asarray(v) > 0)), 2)}
    return out


def _interaction_table(p79_rows, refrows, arms, base):
    """Every method's p79 block, in the drug-design table's row order — the order the
    figures draw, so the CSV, the printed table and the panels can be read against each
    other without re-sorting any of them.

    `arms` is what the pose loader says carries a fingerprint over the whole pocket set --
    NOT every entry of ARMS. Since 2026-09-10 the five published baselines are ARMS too,
    staged under exps/baselines_pose/, but they hold PoseBusters and not PoseCheck; blocking
    one here would export a row of nulls that looks measured and lost."""
    blocks = {lab: _interaction_block(p79_rows[key]) for lab, key, _ in arms}
    blocks.update({lab: _interaction_block(base[lab]) for lab in base})
    blocks[REF_LABEL] = _interaction_block(refrows)
    return [(lab, blocks[lab]) for lab in _interaction_ordered(list(blocks))]


def _interaction_part(out, part):
    """One of the two panel figures — core and all — plus the shared summary table.

    THE CSV IS THE WHOLE TABLE, not this part's two types, and it is written into both
    figure folders. It carries the pooled "All interactions" row and the atom counts that
    say why an arm sits high, and a reader checking one panel against the numbers needs
    those whichever half of the split they are looking at."""
    use_style()
    data, p79_rows, refrows = pose_data()
    # ARMS carries every arm this section knows; only some of them have been scored with
    # PoseCheck. arms_for() keeps an arm only where all 79 pockets have a scored molecule,
    # so an arm mid-scoring cannot enter a figure as a complete-looking column.
    scored = arms_for("ifp", data)

    # The published baselines come from the recompute beside the original builder, not from
    # ARMS: their staged sample dirs hold PoseBusters and not PoseCheck. A name that has
    # become a scored arm is taken from the arm, so no method can enter twice.
    live = {lab for lab, _, _ in scored}
    base = {l: r for l in IX_BASELINES if l not in live
            for r in [_interaction_baseline_rows(l)] if r is not None}
    by_label = {lab: p79_rows[key] for lab, key, _ in scored}
    by_label.update(base)
    by_label[REF_LABEL] = refrows

    core = _interaction_ordered([REF_LABEL] + [lab for lab, key, _ in scored
                                               if key in CORE])
    every = _interaction_ordered(list(by_label))
    kinds = dict(IX_PARTS)[part]
    _interaction_panels(out, part, kinds, by_label, core, "core", caption=True)
    _interaction_panels(out, part, kinds, by_label, every, "all", caption=False)
    missing = [l for l in IX_BASELINES if l not in base and l not in live]

    rows = _interaction_table(p79_rows, refrows, scored, base)
    write_csv(out, "interactions",
              ["arm", "type", "mean", "median", "q25", "q75", "share_nonzero_pct",
               "n_scored"],
              [[lab, k, b[k]["mean"], b[k]["median"], b[k]["q25"], b[k]["q75"],
                b[k]["share_nonzero"], b["n_scored"]]
               for lab, b in rows for k in IX_KINDS])

    print(f"  {len(P79)}-pocket set · interaction fingerprint, pocket10 crop · "
          f"mean per molecule\n")
    print(f"  {'arm':16s} {'atoms':>6s} {'mols':>6s} "
          + " ".join(f"{k:>12s}" for k in IX_KINDS))
    for lab, b in rows:
        print(f"  {lab:16s} {b['atoms_mean']:6.1f} {b['n_scored']:6d} "
              + " ".join(f"{b[k]['mean']:12.2f}" for k in IX_KINDS))
    print(f"\n  {'share of molecules making the type at all':>30s}")
    print(f"  {'arm':16s} {'':6s} {'':6s} " + " ".join(f"{k:>12s}" for k in IX_KINDS))
    for lab, b in rows:
        print(f"  {lab:16s} {'':6s} {'':6s} "
              + " ".join(f"{b[k]['share_nonzero']:11.1f}%" for k in IX_KINDS))
    print("\n  Every type scales with ligand size (see the atoms column): read each arm "
          "against\n  the crystal ligands, not against the arm above it.")
    if missing:
        print(f"\n  no interactions_<M>.json for: {', '.join(missing)} — run "
              f"stage_baseline_poses.py then compute_baseline_interactions.py")


@figure("fig-interaction-contacts", folder="fig-posecheck/interaction",
        needs=("metrics.json", "interactions_<Baseline>.json"))
def draw_interaction_contacts(out):
    """van der Waals and hydrophobic contacts per pose, as violins — core and all."""
    _interaction_part(out, "contacts")


@figure("fig-interaction-hbonds", folder="fig-posecheck/interaction",
        needs=("metrics.json", "interactions_<Baseline>.json"))
def draw_interaction_hbonds(out):
    """H-bonds accepted and donated per pose, as letter-value blocks — core and all."""
    _interaction_part(out, "hbonds")


# ════════════════════════════════════════════════════════════════════════════════
# fig-interaction-pair — the same molecule as generated and as redocked
# ════════════════════════════════════════════════════════════════════════════════
# VoxBind's Fig. 14 draws every interaction type twice: the pose AS GENERATED and the same
# molecule REDOCKED. The two figures above draw the first only, because nothing in the
# pipeline keeps a docked conformer. dock_core_poses.py produces the second for the core
# three arms and compute_baseline_interactions.py fingerprints both; this draws the pair.
#
# WHAT THE PAIR IS FOR. Redocking asks a different question from generating: not "what did
# the model put in the pocket" but "what does this molecule do in the pocket when a docking
# program is allowed to place it". A model whose generated poses already sit where Vina would
# put them has little to gain from redocking; one whose poses are merely plausible-looking
# gains a lot. The gap between a method's two columns is that quantity, and it is only
# readable because THE TWO COLUMNS ARE THE SAME MOLECULES: dock_core_poses.py drops a
# molecule from both series when its dock fails, so nothing separates the columns except the
# docking. 2,105 molecules over 78 pockets — see the original folder's README for the 28
# that did not pair.
#
# COLOUR STAYS THE METHOD'S, which is where this figure departs from Fig. 14 on purpose.
# Fig. 14 gives each series its own colour ramp (green #72b6a1 -> #eaf4f1 for generated,
# salmon #e99675 -> #fcefea for redocked) and puts the methods on the x axis. It can afford
# that because the series is the only other thing in the panel. Here colour means the method
# in every figure of this section, and a reader who has learned that palette should not have
# to unlearn it for one panel. So the pair takes the two channels Fig. 14 leaves free:
# ADJACENCY, which puts the comparison inside one eyeful, and a HATCH on the redocked column.
#
# AND THE METHOD NAMES COME BACK TO THE X AXIS. The two figures above drop them because
# eight of them, rotated to fit, cost a third of the height; three pairs on a 4:1 frame have
# room for them upright, and spending the legend on the method as well would leave the pose
# condition — the whole point of this figure — as a footnote in a five-entry key. Each
# channel gets its own place: the method on the axis, the condition in the legend.

# The arms dock_core_poses.py staged, in the drug-design table's row order. These are the
# labels interactions_<Arm>-{gen,docked}.json is written under — the export spelling, not
# the figure's; _interaction_display() maps them at draw time exactly as the siblings do.
IX_PAIR_ARMS = ["Reference", "VoxBind", "CoDE"]
IX_SERIES = [("gen", "Generated"), ("docked", "Redocked")]

# The reference's colour is keyed under the canonical label, not under the "Reference" the
# staged JSON is named with.
IX_COLOR_KEY = {"Reference": REF_LABEL}

# GEOMETRY. The sibling figures' frame exactly — same aspect, same panel height, same
# legend strip — so the three figures of this family stack in a document without a size
# change between them. Only the x spacing is this figure's own.
IX_PAIR_GAP = 0.30               # between the two columns of one method
IX_GROUP_GAP = 1.15              # between one method and the next
IX_HATCH = "////"
# Neutral grey for the two condition swatches: the legend here names the POSE, not the
# method, and giving it one of the method colours would say the opposite.
IX_SWATCH = "#9aa0a8"


def _interaction_pair_positions():
    """x for each (arm, series), pairs tight and methods apart."""
    xs, x = [], 1.0
    for _ in IX_PAIR_ARMS:
        xs += [x, x + IX_PAIR_GAP + IX_VIOLIN_W]
        x = xs[-1] + IX_GROUP_GAP
    return xs


def _interaction_pair_load():
    """{(arm, series): rows} from the staged scorings."""
    out = {}
    for arm in IX_PAIR_ARMS:
        for key, _ in IX_SERIES:
            path = legacy("fig-interaction", f"interactions_{arm}-{key}.json")
            out[arm, key] = [{"n": m["n"], "ifp": m["ifp"]}
                             for m in json.load(open(path))["molecules"]]
    return out


def _interaction_pair_check(rows):
    """The figure's one precondition: a method's two columns must be the same molecules.
    Rows are written in pocket-then-row order by one scorer over one staged layout, so
    equal length and an equal heavy-atom sequence is the pairing — the same check
    stage_baseline_poses.py makes against the export."""
    for arm in IX_PAIR_ARMS:
        g, d = rows[arm, "gen"], rows[arm, "docked"]
        if len(g) != len(d) or [r["n"] for r in g] != [r["n"] for r in d]:
            raise SystemExit(
                f"{arm}: the generated and redocked series are not the same molecules "
                f"({len(g)} vs {len(d)} rows) — the gap between the columns would be a "
                f"difference in population, not in pose. Re-stage with dock_core_poses.py")


def _interaction_pair_bodies(ax, vals, cols, xs):
    """Violins at `xs` rather than at 1..n, hatched on the redocked column."""
    parts = ax.violinplot(vals, positions=xs, showextrema=False, widths=IX_VIOLIN_W,
                          bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
    for i, (body, col) in enumerate(zip(parts["bodies"], cols)):
        body.set(facecolor=col, alpha=IX_BODY_ALPHA, edgecolor=col, linewidth=1.2)
        if i % 2:
            body.set_hatch(IX_HATCH)
    for x, v in zip(xs, vals):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(x, q1, q3, color=INK, lw=IX_IQR_LW, zorder=3)
        ax.plot(x, med, "o", color="white", ms=IX_MED_MS, zorder=4)


def _interaction_pair_blocks(ax, vals, cols, xs):
    """The boxen panel's letter-value stack at `xs`, hatched on the redocked column."""
    reach = 0.0
    for i, (x, v, col) in enumerate(zip(xs, vals, cols)):
        for lo, hi, d in _interaction_letter_bands(v):
            w = IX_VIOLIN_W / 2 ** d
            face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                + (IX_BODY_ALPHA,)
            # THE METHOD'S OWN COLOUR, as the violins beside it already use (2026-09-14):
            # _interaction_pair_bodies strokes each body in `col`, so the blocks stroking in
            # the neutral AXIS grey made the two panels of one figure look like two
            # conventions. The face is a tint of the same hue, so the stroke reads as the
            # saturated edge of its own block rather than as a foreign rule.
            # IT ALSO FIXES THE HATCH. matplotlib draws a hatch in the patch's EDGE colour,
            # so the redocked column's bars were grey here and method-coloured in the violin
            # panel -- the same encoding drawn two ways in one figure.
            ax.add_patch(IX_RECTANGLE((x - w / 2, lo), w, hi - lo, zorder=3 + d,
                                      facecolor=face, edgecolor=col,
                                      linewidth=IX_BOXEN_EDGE_LW,
                                      hatch=IX_HATCH if i % 2 else None))
            reach = max(reach, hi)
        ax.plot(x, np.median(v), "o", color="white", ms=IX_MED_MS,
                zorder=3 + IX_BOXEN_K)
    return reach


def _interaction_pair_panel(ax, kind, rows, xs):
    ylabel, form = IX_TYPE[kind]
    vals = [_interaction_counts(rows[arm, key], kind)
            for arm in IX_PAIR_ARMS for key, _ in IX_SERIES]
    cols = [color(IX_COLOR_KEY.get(arm, arm)) for arm in IX_PAIR_ARMS for _ in IX_SERIES]
    ax.set_facecolor("white")

    if form == "violin":
        _interaction_pair_bodies(ax, vals, cols, xs)
        reach = max(max(v) for v in vals)
        cells = None
    else:
        reach = _interaction_pair_blocks(ax, vals, cols, xs)
        cells = reach

    # _interaction_frame lays out the shared furniture, then the x is rebuilt: its limits
    # come from these positions rather than from a count of columns, and the category labels
    # go back on -- one per PAIR, centred between the two columns. See the banner above.
    _interaction_frame(ax, vals, ylabel, reach * IX_HEAD_BARE, caption=False, cells=cells)
    ax.set_xlim(xs[0] - IX_VIOLIN_W, xs[-1] + IX_VIOLIN_W)
    ax.set_xticks([(xs[i] + xs[i + 1]) / 2 for i in range(0, len(xs), 2)])
    ax.set_xticklabels([_interaction_display(IX_COLOR_KEY.get(a, a))
                        for a in IX_PAIR_ARMS],
                       fontsize=IX_LEGEND_SIZE, color=INK)
    ax.tick_params(axis="x", length=0, pad=6)
    return form


def _interaction_pair_legend(fig, form):
    """Two swatches, and they name the POSE. The method is on the x axis (see the banner
    above), so this key carries only the thing the pair is about, drawn the way the panel
    draws it: plain for generated, hatched for redocked, in a neutral grey that cannot be
    mistaken for one of the method colours."""
    edge = IX_SWATCH if form == "violin" else AXIS
    lw = 1.2 if form == "violin" else IX_BOXEN_EDGE_LW
    handles = [Patch(facecolor=IX_TO_RGB(IX_SWATCH) + (IX_BODY_ALPHA,), edgecolor=edge,
                     lw=lw, hatch=(IX_HATCH if key == "docked" else None), label=name)
               for key, name in IX_SERIES]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                     ncol=len(handles), frameon=True, facecolor="white",
                     edgecolor=LEGEND_EDGE, framealpha=1.0, borderpad=0.5,
                     fontsize=IX_LEGEND_SIZE, handlelength=1.5, handleheight=1.0,
                     handletextpad=0.6, columnspacing=1.9, labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _interaction_pair_panels(out, part, kinds, rows):
    xs = _interaction_pair_positions()
    fig, axes = plt.subplots(1, 2, figsize=(IX_FIG_W, IX_FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        form = _interaction_pair_panel(ax, kind, rows, xs)
    # NO LEGEND, and so no reserved strip (2026-09-14): the condition key moved out to
    # fig-interaction-legend, which names the whole family in one image rather than each
    # figure naming a part of it. The rect that used to hold the key back would leave an
    # empty band under the panels.
    fit(fig, pad=0.5, w_pad=2.2)
    # `core` is in the filename because it is the only variant this figure has and the
    # family's rule is that the variant is never only in the content: the redocking covers
    # the core three arms, since the published baselines' poses would each need their own
    # ~700 docks and the comparison this section is making is between ours and VoxBind's.
    # interaction_<part>_pair_core, not interaction_pair_<part>_core (2026-09-14): the part
    # is what the figure draws and the pair is how it draws it, so the files of one part sort
    # together -- contacts-all, contacts-core, contacts-pair-core, then the hbonds three.
    save(fig, out, f"interaction_{part}_pair_core")


# ══════════════════════════════════════════════════════════════════════════════════
# fig-interaction-legend — the family's key, as its own image
# ══════════════════════════════════════════════════════════════════════════════════
# Every figure in this family has now given up its own key: the methods went onto the x axis
# of the H-bond panel, the contacts panel carries them invisibly so it can stack under it,
# and the pair figure's condition key came off. So nothing names the encoding any more, and
# this draws it once to sit beside the panels.
#
# THREE SECTIONS, SEPARATED BY RULES, because the key carries three kinds of thing and a
# reader should not have to work out which is which: the crystal ligands, which are the
# benchmark and not a method; the eight methods; and the two POSE CONDITIONS, which are not
# methods either and are drawn in a neutral grey for exactly that reason -- giving them a
# method colour would say the opposite of what they mean.
#
# PAINTED BY THE PANELS' OWN RULE: a translucent face under a stroke of the same saturated
# hue, and the redocked swatch hatched as its column is. A key that does not look like the
# thing it names is a second key to learn.
IX_LEG_SW_W = 0.30                # swatch width, as a share of the key's width
IX_LEG_SW_H = 0.52                # swatch height, in row units
IX_LEG_ROW_H = 0.33               # inches per row
IX_LEG_W = 2.7                    # inches


@figure("fig-interaction-legend", folder="fig-posecheck/interaction")
def draw_interaction_legend(out):
    """Reference, a rule, the eight methods, a rule, then generated and redocked."""
    use_style()
    methods = [m for m in IX_ORDER if m != REF_LABEL]
    # THE TEXT IS RESOLVED HERE, not in the drawing loop. _interaction_display() is strict --
    # it defers to the shared display(), which RAISES on a label the palette does not know --
    # and that is right for a method name, where a typo should fail loudly rather than print
    # itself. But the last two rows are POSE CONDITIONS, not methods, so they carry their own
    # text and never reach it.
    rows = ([(_interaction_display(REF_LABEL), color(REF_LABEL), None)]
            + [(_interaction_display(m), color(m), None) for m in methods]
            + [("Generated", IX_SWATCH, None), ("Redocked", IX_SWATCH, IX_HATCH)])
    # A rule under the reference and under the last method: the two places the KIND of
    # entry changes, which is the only thing a rule should ever mark here.
    rules = {0, len(methods)}
    with plt.rc_context({"hatch.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(IX_LEG_W, IX_LEG_ROW_H * len(rows) + 0.3), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-len(rows) + 0.5, 0.5)
        ax.axis("off")
        for i, (lab, col, hatch) in enumerate(rows):
            y = -i
            ax.add_patch(IX_RECTANGLE((0.05, y - IX_LEG_SW_H / 2), IX_LEG_SW_W,
                                      IX_LEG_SW_H,
                                      facecolor=IX_TO_RGB(col) + (IX_BODY_ALPHA,),
                                      edgecolor=col, linewidth=1.2, hatch=hatch,
                                      zorder=3))
            ax.text(0.05 + IX_LEG_SW_W + 0.07, y, lab,
                    ha="left", va="center", fontsize=IX_LEGEND_SIZE, color=INK)
            if i in rules:
                ax.plot([0.03, 0.97], [y - 0.5, y - 0.5], color=LEGEND_EDGE,
                        lw=AXIS_LW, zorder=2)
        # One frame round the whole key, as the house legend box draws it.
        ax.add_patch(IX_RECTANGLE((0.0, -len(rows) + 0.5), 1.0, len(rows),
                                  facecolor="none", edgecolor=LEGEND_EDGE,
                                  linewidth=AXIS_LW, zorder=1))
        fit(fig, pad=0.3)
        save(fig, out, "interaction_legend")
    print(f"  key: reference · {len(methods)} methods · {len(IX_SERIES)} pose conditions")
    print(f"  wrote {out}")


def _interaction_pair_row(arm, kind, rows):
    """One CSV line: the type's mean and median under both conditions, and the change."""
    g = np.asarray(_interaction_counts(rows[arm, "gen"], kind), float)
    d = np.asarray(_interaction_counts(rows[arm, "docked"], kind), float)
    return [arm, kind, len(g), round(g.mean(), 3), round(d.mean(), 3),
            round(d.mean() - g.mean(), 3), np.median(g), np.median(d),
            round(100 * (g > 0).mean(), 2), round(100 * (d > 0).mean(), 2)]


@figure("fig-interaction-pair", folder="fig-posecheck/interaction",
        needs=("interactions_<Arm>-gen.json", "interactions_<Arm>-docked.json"))
def draw_interaction_pair(out):
    """Generated vs redocked, paired, for the core three arms — contacts and H-bonds."""
    use_style()
    rows = _interaction_pair_load()
    _interaction_pair_check(rows)
    for part, kinds in IX_PARTS:
        _interaction_pair_panels(out, part, kinds, rows)

    write_csv(out, "interactions_pair",
              ["arm", "type", "n_molecules", "gen_mean", "docked_mean", "delta_mean",
               "gen_median", "docked_median", "share_nonzero_gen_pct",
               "share_nonzero_docked_pct"],
              [_interaction_pair_row(arm, kind, rows)
               for arm in IX_PAIR_ARMS for kind in IX_KINDS])

    print(f"  paired generated / redocked poses · {len(rows['CoDE', 'gen'])} molecules for "
          f"the model arms, {len(rows['Reference', 'gen'])} crystal ligands\n")
    head = " ".join(f"{k:>21s}" for k in IX_KINDS)
    print(f"  {'arm':12s} {head}")
    for arm in IX_PAIR_ARMS:
        cells = []
        for kind in IX_KINDS:
            g = np.mean(_interaction_counts(rows[arm, "gen"], kind))
            d = np.mean(_interaction_counts(rows[arm, "docked"], kind))
            cells.append(f"{g:7.2f} ->{d:6.2f} {d - g:+5.2f}")
        print(f"  {_interaction_display(IX_COLOR_KEY.get(arm, arm)):12s} "
              + " ".join(cells))
    print("\n  mean per molecule, generated -> redocked and the change. The crystal ligands "
          "are\n  the control: they are already in a measured pose, so redocking should "
          "move them\n  least.")
