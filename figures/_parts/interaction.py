

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
# FuncBind BEFORE VoxBind (2026-09-15), which is RB_ORDER's sequence and not the drug-design
# table's. The table puts VoxBind first and this family used to follow it, but rotbond and
# the shared posecheck legend both run RB_ORDER, so following the table here made the
# interaction figures the odd ones out in a document that stacks them together. Consistency
# between the FIGURES was chosen over consistency with the table; if the table is ever the
# thing to match, swap these two back and RB_ORDER with them, not this alone.
IX_ORDER = ("Reference ligand", "AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff",
            "FuncBind", "VoxBind", "CoDE")

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


def _interaction_axis_weight(ax, lw=None, title_fs=None, tick_fs=None):
    """The ecdf-by-size-all axis: spine weight, tick type and the y name's size.

    SPLIT OUT OF _interaction_ecdf_furniture (2026-09-15) so the PAIR figure can take the
    same look. That function cannot simply be called there: it also rebuilds the x axis at
    1..n with the names rotated 30 degrees, and the pair figure deliberately keeps its
    names UPRIGHT on each split body's own spine -- calling it would undo the widening that
    was chosen to make upright names fit.

    ONLY THE SPINES, TICKS AND AXIS NAME MOVE. Not _pcsz_style() wholesale: that turns the
    major y grid back on, and the H-bond panels run a deliberate minor-grid arrangement
    (_interaction_cell_axis) that would lose to it.

    PCSZ_* are read at CALL time: they live in a part assembled after this one."""
    # DEFAULTS ARE THE ECDF'S OWN NUMBERS -- _pcsz_style sets spine and tick 1.1, tick labels
    # at 11 and the y name at PCSZ_LABEL_FS, and contacts/hbonds must keep exactly those,
    # since being indistinguishable from ecdf-by-size-all is what this function is for. The
    # three overrides let ONE caller ask for more; only the pair panel does.
    #
    # `tick_fs` REACHES THE Y AXIS ONLY IN PRACTICE, even though tick_params sets both: the
    # pair panel rebuilds its x axis afterwards with set_xticklabels(fontsize=...), so the
    # method names keep IX_PAIR_NAME_FS and this number lands on the y numbers alone.
    lw = 1.1 if lw is None else lw
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(labelsize=11 if tick_fs is None else tick_fs,
                   direction="out", length=3.5, width=lw, pad=4, colors=PCSZ_AXIS)
    ax.yaxis.label.set_fontsize(PCSZ_LABEL_FS if title_fs is None else title_fs)
    ax.yaxis.label.set_color(PCSZ_AXIS)


def _interaction_ecdf_furniture(ax, names, show_names=True):
    """The strain ECDF's axis weight and type, plus the method names back on the x axis.

    `show_names` is False for the figure that sits ABOVE another carrying the same
    categories -- see IX_NAMED_PARTS.

    ONLY THE SPINES, TICKS AND AXIS NAME MOVE. Not _pcsz_style() wholesale: that turns the
    major y grid back on, and the H-bond panel runs a deliberate minor-grid arrangement
    (_interaction_cell_axis) that would lose. Not PCSZ_RC either -- it carries savefig.dpi
    170 and a crop to the ink, and this family draws at 220 to its own frame.

    PCSZ_* are read at CALL time: they live in a part assembled after this one."""
    _interaction_axis_weight(ax)
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
# docking. 8,180 molecules over 78 pockets, from 8,286 docked. The 106 that did not pair
# were the two failures dock_core_poses.py's docstring predicts and nothing else: 105 are
# target_71, whose pocket crop pdb2pqr30 refuses for a chain gap at GLU 866, so it loses
# every arm; the last is VoxBind's `CNNO` at target_96, four heavy atoms whose extent gives
# Vina a box with a zero dimension.
#
# COLOUR STAYS THE METHOD'S, which is where this figure departs from Fig. 14 on purpose.
# Fig. 14 gives each series its own colour ramp (green #72b6a1 -> #eaf4f1 for generated,
# salmon #e99675 -> #fcefea for redocked) and puts the methods on the x axis. It can afford
# that because the series is the only other thing in the panel. Here colour means the method
# in every figure of this section, and a reader who has learned that palette should not have
# to unlearn it for one panel. So the pair takes the two channels Fig. 14 leaves free:
# THE TWO HALVES OF ONE BODY, which puts the comparison inside a single shape, and a HATCH
# on the redocked half.
#
# SPLIT, NOT ADJACENT (2026-09-15). This drew two whole bodies side by side until the split
# violin replaced them. Two bodies made the reader compare across a gap and cost twice the
# ink per method; one body cut down the middle puts the two distributions against a shared
# spine, which is the comparison this figure exists to make. It also doubles the width each
# method's ink gets, because nine slots now hold one body instead of two.
#
# AND THE METHOD NAMES COME BACK TO THE X AXIS. The two figures above drop them because
# eight of them, rotated to fit, cost a third of the height; here they stay upright and the
# FRAME widens to make room, rather than the names tilting to fit the frame. Spending the
# legend on the method as well would leave the pose condition — the whole point of this
# figure — as a footnote in a five-entry key. Each channel gets its own place: the method
# on the axis, the condition in the legend.
#
# Upright names were free when this was three pairs on the siblings' 4:1 frame; at nine they
# cost the width IX_PAIR_ASPECT carries, and that was chosen deliberately over tilting them.
# See the note on IX_PAIR_ASPECT for what the alternative would have been.

# The arms dock_core_poses.py staged, in the drug-design table's row order. These are the
# labels interactions_<Arm>-{gen,docked}.json is written under — the export spelling, not
# the figure's; _interaction_display() maps them at draw time exactly as the siblings do.
# ALL NINE (2026-09-15). This was the core three until the redocking was extended to every
# published baseline: 8,286 docks over 79 pockets, 8,180 of which paired. The five staged
# baselines reach dock_core_poses.py through --staged and TargetDiff through
# --with-targetdiff, so every arm in IX_ORDER now has both a -gen and a -docked scoring.
IX_PAIR_ARMS = ["Reference", "AR", "Pocket2Mol", "DiffSBDD", "TargetDiff",
                "DecompDiff", "FuncBind", "VoxBind", "CoDE"]
IX_SERIES = [("gen", "Generated"), ("docked", "Redocked")]

# The reference's colour is keyed under the canonical label, not under the "Reference" the
# staged JSON is named with.
IX_COLOR_KEY = {"Reference": REF_LABEL}

# GEOMETRY. The panel height and the legend strip are the siblings' exactly, so the figures
# of this family still stack without a vertical size change. THE WIDTH IS NO LONGER THEIRS
# (2026-09-15): nine pairs of upright names do not fit a 4:1 frame.
#
# WHY THIS NUMBER, AND WHY THE NAME FONT MOVES WITH IT. The binding constraint is that one
# method's slot must
# hold its own name AND a gap to its neighbour's, and the slot is the group pitch — so the
# requirement is essentially `n_arms x (widest_label + margin)` and TIGHTENING IX_GROUP_GAP
# BUYS NOTHING, because the span and the pitch shrink together. Measured under this style at
# IX_LEGEND_SIZE, the widest label is VoxBind$_{\sigma=0.9}$ at 1.139 in.
#
# THE MARGIN IS NOT OPTIONAL, which is what a first pass at 5.85 got wrong: sizing the slot
# to the widest label ALONE leaves zero room between neighbours, and the two widest that
# happen to be adjacent -- DecompDiff and VoxBind -- came out 0.034 in apart while every
# other pair had 0.15-0.56 in.
#
# AND THE PITCH DOES NOT GET THE WHOLE WIDENING, which is what the SECOND pass got wrong.
# fit() spends part of every added inch on the margins and the y names, so only about 86 %
# of it reaches the panel: 6.12 was computed to land exactly on a 0.10 in floor and measured
# 0.090. Do not predict this from one render -- sweep the aspect and read the gap back. On
# this frame the worst gap moves ~0.21 in per unit of aspect, and the measured ladder is
# 5.85 -> 0.034, 6.12 -> 0.090, 6.20 -> 0.107, 6.30 -> 0.128. 6.20 is the smallest that
# clears the floor, which is why the figure is no wider than it is.
#
# Measure with the tick labels' own bounding boxes after fit(), never by eye: at this size
# 0.034 in reads as a collision and is not one.
#
# NARROWED TO 0.8x ON REQUEST (2026-09-15): 6.20 -> 4.96, 23.88 in -> 19.10 in. At the font
# of the time that OVERLAPPED -- measured -0.068 in, the labels actually colliding, because
# width is the only lever once the font is fixed and the names are upright. So the font took
# the same 0.8x: 12.5 x 0.8 = 10.0, and the worst gap came back to 0.127 in.
#
# THAT WHOLE CONSTRAINT IS GONE NOW, and the paragraph above is kept only as the record of
# how this number was arrived at. The names were rotated 30 degrees shortly afterwards and
# IX_PAIR_NAME_FS went to 14.0: rotated labels run off their own tick instead of competing
# for a method's pitch, so width no longer gates the font, and widening only ever helps.
#
# THEN 1.03x ON REQUEST: 4.96 -> 5.11, 19.10 in -> 19.68 in. THEN 0.98x: 5.11 -> 5.01,
# 19.68 in -> 19.30 in. That narrowing alone costs the stacked axes 9.009 -> 8.816 in of
# width, before the type sitting on it is counted.
#
# THE COST, recorded so nobody re-derives it: each half violin goes 0.438 in -> 0.342 in,
# under the 0.411 in of the single violin the old adjacent layout drew. That is inherent to
# narrowing the frame, not a sizing mistake like the IX_GROUP_GAP one below it.
# THEN 1.2x ON REQUEST: 5.01 -> 6.01, 19.30 in -> 23.15 in. Widening is the safe direction --
# the names are rotated, so it only ever adds room between them, and it leaves the vertical
# dimension (where the y title's clipping lives) alone.
# THEN 1.03x ON REQUEST (2026-09-15): 6.01 -> 6.1903, 23.15 in -> 23.84 in. Height is
# deliberately NOT touched -- IX_PAIR_TALL stays 1.20 -- so this is width alone.
IX_PAIR_ASPECT = 6.1903
# HEIGHT ONLY, ON TOP OF THE WIDTH ABOVE (2026-09-15). The pair figures are taller than the
# siblings by this much; the WIDTH stays IX_PAIR_ASPECT * IX_FIG_H so the 0.98x narrowing is
# untouched. IX_FIG_H itself must not move: IX_FIG_W is derived from it, so raising it would
# silently widen contacts-all, hbonds-all, both core variants and the legend as well.
#
# WHY 1.20 AND NOT "SLIGHTLY". The y name is rotated, so its LENGTH runs up the figure --
# "van der Waals contacts" is 3.16 in at 19.5 pt against a 2.44 in axes, and it was running
# off the canvas. Growing the frame grows the axes at roughly HALF the rate, so the fix is
# expensive: measured contacts margin at 19.5 pt is 1.00 -> -0.257 in (clipped), 1.10 ->
# -0.064 (still clipped), 1.15 -> +0.032 (seven pixels, breaks on any rounding), 1.20 ->
# +0.128, 1.25 -> +0.224. 1.20 is the smallest that holds.
#
# THE CHEAPER TRADE, if this frame is ever too tall: lower the title with it. At 17.5 pt a
# 1.10 multiplier already gives +0.155 in, and a two-line title clears +0.364 in at no extra
# height at all.
# 1.20 -> 1.35 -> BACK TO 1.20 (2026-09-15). The 1.35 was bought for ONE REASON ONLY: a
# one-line "van der Waals contacts" at 22 pt ran off the canvas at 1.20, measured -0.145 in.
# Wrapping the pair titles to two lines (IX_PAIR_YLABEL) halves the length that runs up the
# figure and repays that debt outright, so the extra height is no longer owed and the frame
# returns to the height it had before.
#
# MEASURED WITH THE WRAP IN PLACE, vertical margin on the contacts standalone -- the panel
# that gates this, since it carries the longest y name: 1.00 -> +0.170, 1.10 -> +0.362,
# 1.20 -> +0.555, 1.35 -> +0.844. Even 1.00 clears now. 1.20 is kept because it is the
# height that was settled on earlier, not because anything below it fails.
#
# WIDTH CANNOT SUBSTITUTE FOR THIS. The title is rotated 90 degrees, so its length runs UP
# the figure; a wider frame buys the y name nothing at all.
IX_PAIR_TALL = 1.20
# The method names' size, PAIR-ONLY and deliberately not IX_LEGEND_SIZE: that constant is
# shared with fig-interaction-legend and _interaction_pair_legend, and retyping it for this
# figure's geometry would silently retype the standalone key as well.
#
# THE SAME SIZE AS THE Y AXIS TITLE, on request (2026-09-15): this must equal
# IX_PAIR_TITLE_FS. It is written as a literal rather than as that name because TITLE_FS is
# defined BELOW this line, and referring to it here would raise at import. CHANGE THE TWO
# TOGETHER -- a title moved on its own silently leaves the method names behind.
#
# The road here: 10.0 while the names were upright, because upright names had to fit side by
# side inside one method's pitch and the frame had been narrowed to 0.8x. Rotating them 30
# degrees removed that constraint entirely -- a diagonal label runs off its own tick instead
# of competing for its neighbour's space -- so the size became free to follow the title.
#
# IT COSTS AXES HEIGHT, and the standalone pays about twice as much because one band of
# names sits on half the height: at the 19.30 in frame, 18.5 gives stacked 8.816 x 3.169 and
# standalone 8.697 x 2.498, 21.0 gives 8.784 x 3.098 and 8.587 x 2.357.
#
# BUT THE CEILING IS THE Y TITLE, NOT THE AXES. The y name is rotated 90 degrees, so its
# LENGTH runs up the figure: "van der Waals contacts" is 3.4 in at 21 pt against a 3.1 in
# axes, and matplotlib centres it on the axes and lets the overflow run off the canvas.
# Measured top margin on the stacked frame: 18.5 -> +0.146 in, 19.5 -> +0.049, 20.0 ->
# +0.004, 20.5 -> -0.043 CLIPPED, 21.0 -> -0.092 CLIPPED. 19.5 is the largest size with a
# margin that survives rounding; 20.0's four thousandths of an inch is one pixel.
#
# CHECK THIS ON THE RENDERED PIXELS. fig.get_tightbbox() reported "no clipping" at 21.0 AND
# at 23.0, both of which cut the title off at row 0 of the raster. Measure the y label's own
# window extent against the figure height, or read the PNG.
#
# DO NOT check these for collisions with get_window_extent: it returns the AXIS-ALIGNED box
# of a rotated label, and neighbouring diagonal labels overlap those boxes while their ink
# passes cleanly between. It reports collisions at sizes that are demonstrably fine.
# 19.5 -> 22.0 (2026-09-15). IT MOVES WITH THE TITLE, per the rule above: the two were
# already EQUAL at 19.5 -- measured on the rendered canvas, not merely equal in the source --
# so raising the title on its own would have left the names behind and broken that rule.
IX_PAIR_NAME_FS = 22.0
# THE PAIR FIGURES' OWN AXIS WEIGHT AND Y NAME SIZE (2026-09-15), heavier than the ecdf's.
# _interaction_axis_weight defaults to the ecdf's exact numbers -- spine and tick 1.1, y name
# PCSZ_LABEL_FS -- and contacts/hbonds MUST keep them, since being indistinguishable from
# ecdf-by-size-all is the whole point of that function. These are passed by the pair panel
# alone so the siblings are untouched.
#
# 2.2 IS THE JSD SUMMARY'S, TAKEN DELIBERATELY (2026-09-15). Weight has to be read against
# the canvas it sits on: this figure is 19.10 in wide, the widest in the set, so at 1.6 it
# was the THINNEST of the three by the measure that matters -- 0.084 pt/in against the JSD
# summary's 0.165 and the clash box's. Matching JSD's absolute 2.2 brings it to 0.115 pt/in,
# most of the way there without going to the full column-scaled 3.15.
#
# The ladder this repo has, for whoever tunes it next: ecdf 1.1, the house default AXIS_LW
# 1.35, the clash box's PCSZ_CBOX_AXIS_LW 1.0 (a narrow canvas, so a small number), the JSD
# summary's 2.2 and the rotatable-bond grid's 2.0. Compare pt/in, not pt.
#
# 18.5 IS SAFE, MEASURED. Raising this family's y name from 13 to 16.1 once truncated the
# H-bond titles to "...by the liga" on the fixed dpi-220 frame, so it was swept before being
# changed: at 16.1/17.5/18.5/19.5/21.0 nothing leaves the frame in either the standalone or
# the stacked figure, and the axes lose 0.06 in of width across that whole range.
IX_PAIR_AXIS_LW = 2.2
# 19.5 -> 22.0 (2026-09-15), ON REQUEST, AND IX_PAIR_NAME_FS MOVES WITH IT. The ceiling is
# the rotated y name's clearance, which is bought either with IX_PAIR_TALL or by wrapping the
# name -- see IX_PAIR_YLABEL, which is what actually paid for this size.
#
# THE OLD WARNING HERE SAID "22.0 clips at the 1.20 frame by 0.145 in". That was measured on
# a ONE-LINE title and is no longer what this figure draws: with the two-line names the same
# frame clears by +0.555 in. Re-run the height sweep before raising this again, and do not
# trust fig.get_tightbbox() for it -- it reported no clipping at sizes that were visibly cut.
IX_PAIR_TITLE_FS = 22.0
# The y numbers. 14.0 is the house default -- furniture()'s labelsize -- rather than a number
# picked by eye, and it replaces the ecdf's 11, which read small once the names and the title
# both went to 18.5. Swept before changing: 11 -> 15 costs the axes 0.6 % of their width and
# nothing at all in height or row gap, so this size is essentially free and can go further.
# 14.0 -> 16.5 (2026-09-15), on request. The sweep recorded above still holds -- the y
# numbers cost axes WIDTH and nothing in height -- so this one does not touch the title's
# vertical clearance, which is the only thing that clips in this figure.
IX_PAIR_TICK_FS = 16.5
# THE GAP BETWEEN THE Y NAME AND ITS AXIS, DOUBLED ON REQUEST (2026-09-15): 10 -> 20 points.
# PAIR-ONLY, and it has to be. _interaction_frame sets labelpad=10 and is called by BOTH
# sibling panel forms as well as this one, so doubling it there would push the y name out on
# contacts-all, hbonds-all and both cores -- six figures that were not asked to move.
#
# IT COSTS AXES WIDTH, NOT CLEARANCE. tight_layout measures the label with its pad, so the
# extra 10 points (0.139 in) is taken out of the panel rather than pushing the title off the
# left edge; the title's own horizontal margin is unchanged. Measure both after changing it.
IX_PAIR_LABELPAD = 20
# A FIXED Y RANGE FOR ONE PANEL, PAIR-ONLY (2026-09-15): kind -> (top, tick step).
# _interaction_frame sizes every panel to its own data (reach * IX_HEAD_BARE), which lets the
# van der Waals violins run to 32 on the strength of a few tails while the mass sits under
# 15. Cutting that panel at 25 spends the height on the part of the distribution a reader is
# comparing. THE TAILS ARE CLIPPED, NOT DROPPED -- the bodies are drawn before this runs, so
# the data is unchanged and only the view is cropped.
#
# ONLY VdWContact IS LISTED, DELIBERATELY. Hydrophobic tops out near 13, so the same 25 would
# leave its upper half empty and squeeze a median of 1-2 into a sliver; it keeps its own
# scale. A kind absent from this map is sized by the frame exactly as before.
# kind -> (ylim top, topmost tick, tick step). THE LIMIT AND THE TOP TICK ARE SEPARATE
# (2026-09-15) and both exist to fix the same thing. Cutting at exactly 25 put the last tick
# ON the top spine, so half of the "25" label sat outside the axes, tight_layout reserved the
# overhang, and the contacts panel came out 0.086 in SHORTER than the H-bond panel it is
# placed next to -- a size difference created purely by a tick label. Raising the limit to 26
# pulls the label inside, and THAT is what fixed it -- the two panels measure the same
# 3.0671 in again. The label is kept; only the limit does the work. If the top tick is ever
# moved back onto the limit, expect the panel to lose height to the overhang again.
IX_PAIR_YCUT = {"VdWContact": (26, 25, 5)}
IX_SPLIT_W = 1.60                # the full split body: left half generated, right redocked
# 0.50, NOT THE 1.15 THE ADJACENT LAYOUT USED (2026-09-15). The split is only worth having
# if the body gets the room the two old columns had between them, and a gap sized for
# separating two whole violins wastes it: at 1.15 each half measured 0.347 in, THINNER than
# the 0.411 in single violin it replaced, because 1.60 of ink in a 2.75 pitch is 58 % of the
# slot where the old pair filled 83 %. At 0.50 each half is 0.437 in and the worst label gap
# is still 0.181 in, comfortably over the 0.10 floor.
#
# THE SLACK GOES INTO THE BODY, NOT INTO A NARROWER FIGURE. Pulling IX_PAIR_ASPECT down
# instead spends it on nothing a reader can see and collapses the labels quickly -- 5.60
# measures a 0.064 in gap, already under the floor.
IX_GROUP_GAP = 0.50              # between one method and the next
# Where each half's median dot and IQR bar sit, as a share of IX_SPLIT_W off the centre
# spine. Far enough in to read as belonging to its own half, not so far that it leaves the
# body at a method whose distribution is narrow.
IX_SPLIT_STAT_OFF = 0.18
IX_HATCH = "////"
# Neutral grey for the two condition swatches: the legend here names the POSE, not the
# method, and giving it one of the method colours would say the opposite.
IX_SWATCH = "#9aa0a8"


# THE PAIR FIGURES' Y NAMES, WRAPPED TO TWO LINES (2026-09-15), AND PAIR-ONLY.
# IX_TYPE is read by _interaction_panels as well -- the sibling contacts-all and hbonds-all
# figures and both of their cores -- so wrapping the strings where they are DEFINED would
# re-break the titles on six figures that were never asked to change. This maps the same
# keys to a two-line spelling and is consulted only by _interaction_pair_panel. A kind
# missing here falls back to IX_TYPE's own name, so IX_TYPES stays the single source of
# which kinds exist and adding one cannot raise here.
#
# ALL FOUR BREAK, AND EACH BREAKS BEFORE ITS LAST WORD. The stacked figure puts contacts
# above H-bonds, so titles with unequal line counts would hang at different heights against
# axes aligned to 0.0000 in -- exactly the ragged edge fig.align_ylabels was added to remove.
# Keeping every title two lines is what preserves that alignment.
IX_PAIR_YLABEL = {
    "VdWContact":  "van der Waals\ncontacts",
    "Hydrophobic": "Hydrophobic\ncontacts",
    "HBAcceptor":  "H-bonds\nacceptor",
    "HBDonor":     "H-bonds\ndonor",
}


def _interaction_pair_positions():
    """x for each ARM -- one slot per method, the two conditions split inside it.

    One position per arm, not two (2026-09-15): the conditions are halves of a single body
    now, so they share a centre rather than sitting at their own x."""
    pitch = IX_SPLIT_W + IX_GROUP_GAP
    return [1.0 + i * pitch for i in range(len(IX_PAIR_ARMS))]


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
    """One SPLIT violin per method: left half the generated pose, right half the redocked.

    `vals` is still flat and interleaved -- gen then docked for each arm -- so vals[2i] and
    vals[2i+1] are the two halves of the body at xs[i]. `cols` is one colour per ARM.

    HOW THE HALF IS MADE. violinplot has no split, so each half is drawn as a whole violin
    at the same x and then its path is CLIPPED to one side of that x. Clipping the vertices
    rather than re-deriving the KDE keeps both halves on exactly the shared spine, and keeps
    the bandwidth rule identical to the one every other violin in this family uses."""
    for i, (x, col) in enumerate(zip(xs, cols)):
        for half, v in enumerate((vals[2 * i], vals[2 * i + 1])):
            part = ax.violinplot([v], positions=[x], showextrema=False, widths=IX_SPLIT_W,
                                 bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
            body = part["bodies"][0]
            vtx = body.get_paths()[0].vertices
            lo, hi = (-np.inf, x) if half == 0 else (x, np.inf)
            vtx[:, 0] = np.clip(vtx[:, 0], lo, hi)
            # THE ALPHA IS BAKED INTO THE FACE, NOT SET ON THE ARTIST (2026-09-15). An
            # artist-level alpha applies to the EDGE too, and matplotlib draws a hatch in the
            # EDGE colour -- so the redocked half's slashes were being drawn in the body's own
            # colour at the body's own 0.55 opacity, on top of the fill they sat on, and were
            # effectively invisible. Measured before the fix: face and edge were the identical
            # RGBA (0.604, 0.627, 0.651, 0.55).
            #
            # _interaction_pair_blocks NEVER HAD THIS BUG and is the pattern copied here: it
            # passes an RGBA face and an opaque edgecolor, which is exactly why the H-bond
            # panels showed their slashes while the violins beside them did not.
            body.set(facecolor=IX_TO_RGB(col) + (IX_BODY_ALPHA,), edgecolor=col,
                     linewidth=1.2)
            if half:
                body.set_hatch(IX_HATCH)
            # The stats go INSIDE their own half, offset off the spine -- drawn on the
            # centre they would be one mark for two distributions.
            off = (-1 if half == 0 else 1) * IX_SPLIT_STAT_OFF * IX_SPLIT_W
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(x + off, q1, q3, color=INK, lw=IX_IQR_LW, zorder=3)
            ax.plot(x + off, med, "o", color="white", ms=IX_MED_MS, zorder=4)


def _interaction_pair_blocks(ax, vals, cols, xs):
    """The boxen panel's letter-value stack, SPLIT the way the violins are: each band is
    half its full width, generated on the left of the spine and redocked on the right.

    `vals` is flat and interleaved as in _interaction_pair_bodies; `cols` is one per ARM."""
    reach = 0.0
    for i, (x, col) in enumerate(zip(xs, cols)):
        for half, v in enumerate((vals[2 * i], vals[2 * i + 1])):
            for lo, hi, d in _interaction_letter_bands(v):
                # Half of the full band width, so the two conditions together occupy the
                # same envelope one undivided block used to.
                hw = IX_SPLIT_W / 2 ** d / 2
                x0 = x - hw if half == 0 else x
                face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                    + (IX_BODY_ALPHA,)
                # THE METHOD'S OWN COLOUR, as the violins beside it already use (2026-09-14):
                # _interaction_pair_bodies strokes each body in `col`, so the blocks stroking
                # in the neutral AXIS grey made the two panels of one figure look like two
                # conventions. The face is a tint of the same hue, so the stroke reads as the
                # saturated edge of its own block rather than as a foreign rule.
                # IT ALSO FIXES THE HATCH. matplotlib draws a hatch in the patch's EDGE
                # colour, so the redocked bars were grey here and method-coloured in the
                # violin panel -- the same encoding drawn two ways in one figure.
                ax.add_patch(IX_RECTANGLE((x0, lo), hw, hi - lo, zorder=3 + d,
                                          facecolor=face, edgecolor=col,
                                          linewidth=IX_BOXEN_EDGE_LW,
                                          hatch=IX_HATCH if half else None))
                reach = max(reach, hi)
            off = (-1 if half == 0 else 1) * IX_SPLIT_STAT_OFF * IX_SPLIT_W
            ax.plot(x + off, np.median(v), "o", color="white", ms=IX_MED_MS,
                    zorder=3 + IX_BOXEN_K)
    return reach


def _interaction_pair_panel(ax, kind, rows, xs, part=None):
    # The form and the fallback name come from the shared map; the WRAPPED name is this
    # figure's own, because IX_TYPE is the siblings' too. See IX_PAIR_YLABEL.
    ylabel, form = IX_TYPE[kind]
    ylabel = IX_PAIR_YLABEL.get(kind, ylabel)
    # Flat and interleaved: gen then docked for each arm, so the drawing helpers read
    # vals[2i] and vals[2i+1] as the two halves of the body at xs[i]. ONE colour per arm --
    # the halves of a split body are the same method and so the same hue; what tells them
    # apart is the side and the hatch.
    vals = [_interaction_counts(rows[arm, key], kind)
            for arm in IX_PAIR_ARMS for key, _ in IX_SERIES]
    cols = [color(IX_COLOR_KEY.get(arm, arm)) for arm in IX_PAIR_ARMS]
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
    # go back on -- one per METHOD, on the split body's own spine. See the banner above.
    _interaction_frame(ax, vals, ylabel, reach * IX_HEAD_BARE, caption=False, cells=cells)
    # The siblings' axis weight and type (2026-09-15), so the three interaction figures
    # carry one look. Applied BEFORE the x axis is rebuilt: it sets a tick labelsize and
    # colour for both axes, and the method names below want their own size and INK.
    _interaction_axis_weight(ax, IX_PAIR_AXIS_LW, IX_PAIR_TITLE_FS, IX_PAIR_TICK_FS)
    # BOTH LINES CENTRED (2026-09-15). The y name is rotated 90 degrees, so a two-line
    # title's lines stack ACROSS the axis rather than along it; on the default the shorter
    # line hangs off one end and the block reads as leaning away from the spine.
    ax.yaxis.label.set_multialignment("center")
    # Set AFTER _interaction_frame, which applies the shared labelpad=10. See IX_PAIR_LABELPAD.
    ax.yaxis.labelpad = IX_PAIR_LABELPAD
    # The fixed range, for the kinds that ask for one. AFTER _interaction_frame, which has
    # already set a data-derived ylim, and keeping that function's -0.02 * top baseline so the
    # zero line sits where it does on every other panel. See IX_PAIR_YCUT.
    cut = IX_PAIR_YCUT.get(kind)
    if cut is not None:
        cut_top, tick_top, cut_step = cut
        ax.set_ylim(-0.02 * cut_top, cut_top)
        # EVERY TICK KEEPS ITS LABEL, 25 INCLUDED. It was blanked briefly to stop it
        # overhanging the top spine; the 26 limit already pulls it inside, so the blanking
        # was redundant and the number is back. See IX_PAIR_YCUT.
        ax.set_yticks(list(range(0, tick_top + 1, cut_step)))
    ax.set_xlim(xs[0] - IX_SPLIT_W, xs[-1] + IX_SPLIT_W)
    ax.set_xticks(xs)
    # ROTATED, AND ONLY ONE PART NAMES THEM (2026-09-15) -- the siblings' rule, IX_NAMED_PARTS,
    # brought over here so the two pair figures can stack. `part` None keeps the names, for a
    # caller drawing one panel on its own.
    labels = ax.set_xticklabels([_interaction_display(IX_COLOR_KEY.get(a, a))
                                 for a in IX_PAIR_ARMS],
                                fontsize=IX_PAIR_NAME_FS, color=INK,
                                rotation=30, ha="right", rotation_mode="anchor")
    if part is not None and part not in IX_NAMED_PARTS:
        # DRAWN BUT INVISIBLE, not absent. tight_layout measures a text's extent whatever its
        # colour, so colouring them "none" reserves the exact band of height they would have
        # taken -- which is what keeps this panel's axes the same height as the named one it
        # sits above. Dropping the labels instead hands that band to the axes and the two
        # rows come out carrying different vertical scales for the same categories.
        for t in labels:
            t.set_color("none")
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
    fig, axes = plt.subplots(1, 2, dpi=220,
                             figsize=(IX_PAIR_ASPECT * IX_FIG_H,
                                      IX_FIG_H * IX_PAIR_TALL))
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        form = _interaction_pair_panel(ax, kind, rows, xs, part)
    # NO LEGEND, and so no reserved strip (2026-09-14): the condition key moved out to
    # fig-interaction-legend, which names the whole family in one image rather than each
    # figure naming a part of it. The rect that used to hold the key back would leave an
    # empty band under the panels.
    fit(fig, pad=0.5, w_pad=2.2)
    # `all`, not `core` (2026-09-15): the redocking now covers every arm, and the family's
    # rule is that the variant is never only in the content. It was `core` while the
    # published baselines' poses would each have needed their own ~1,000 docks; they have
    # since been docked, so the name would otherwise be a lie about what is in the figure.
    # interaction_<part>_pair_all, not interaction_pair_<part>_all (2026-09-14): the part
    # is what the figure draws and the pair is how it draws it, so the files of one part sort
    # together -- contacts-all, contacts-core, contacts-pair-all, then the hbonds three.
    save(fig, out, f"interaction_{part}_pair_all")


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
    """Generated vs redocked, paired, for every arm — contacts and H-bonds."""
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


# ══════════════════════════════════════════════════════════════════════════════════
# fig-interaction-pair-stacked — both parts of the pair figure in one frame
# ══════════════════════════════════════════════════════════════════════════════════
# The two pair figures were always meant to stack: contacts draws its method names in
# colour "none" and H-bonds draws them for both, which is only worth doing if something
# puts one above the other. This is that something -- the same four panels, in one image,
# so the whole comparison is a single figure rather than two that a document has to be
# trusted to keep together and in order.
#
# NOTHING ABOUT THE PANELS CHANGES. It reuses _interaction_pair_panel, so the split bodies,
# the colours, the hatch and the axis weight are the ones the standalone figures draw; only
# the frame is this figure's own. A panel that looked different here would be a second
# convention for the same measurement.
IX_STACK_PAD = 0.06              # inches of air outside the ink, all four sides
IX_STACK_ROW_GAP = 0.10          # inches between the two rows -- "almost none", as asked


def _interaction_stack_layout(fig, axes):
    """Place the stacked grid BY HAND rather than by tight_layout.

    WHY NOT fit(). tight_layout sizes a row to its own content, so the row carrying the
    method names would come out shorter than the one that does not -- and h_pad barely
    touches it: measured 1.18 in of row gap at h_pad 1.4 and still 0.89 in at h_pad 0.0,
    because the gap IS the name band and not padding. Setting the margins directly gives
    both rows exactly the gridspec's height and puts the gap where it was asked for.

    THE INSETS ARE MEASURED, NOT ASSUMED. `left` is the widest y name plus its tick labels
    over all four panels (they differ: the contacts row needed 0.657 in and the H-bond row
    0.559 in, and using the smaller would clip the wider one). `bottom` is what the rotated
    names of the named row need. An inset is how far a panel's decorations overhang its own
    box, so it does not depend on where that panel currently sits -- which is what makes it
    safe to measure before the final placement.

    wspace and hspace are fractions of the AXES size, and the axes size is the thing they
    change, so the fraction that yields a wanted absolute gap has to be solved for:
    gap = s*plot/(n + s)  =>  s = n*gap/(plot - gap)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    flat = [ax for row in axes for ax in row]
    left_in = max(ax.get_position().x0 * W - ax.get_tightbbox(r).x0 / fig.dpi
                  for ax in flat)
    bot_in = max(ax.get_position().y0 * H - ax.get_tightbbox(r).y0 / fig.dpi
                 for ax in axes[-1])
    left, bottom = (left_in + IX_STACK_PAD) / W, (bot_in + IX_STACK_PAD) / H
    right, top = 1.0 - IX_STACK_PAD / W, 1.0 - IX_STACK_PAD / H
    plot_w, plot_h = (right - left) * W, (top - bottom) * H
    # The columns are separated by exactly the room the right column's own y name needs.
    col_gap = left_in + IX_STACK_PAD
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom,
        wspace=len(axes[0]) * col_gap / (plot_w - col_gap),
        hspace=len(axes) * IX_STACK_ROW_GAP / (plot_h - IX_STACK_ROW_GAP),
    )


@figure("fig-interaction-pair-stacked", folder="fig-posecheck/interaction",
        needs=("interactions_<Arm>-gen.json", "interactions_<Arm>-docked.json"))
def draw_interaction_pair_stacked(out):
    """Generated vs redocked, paired, every arm — contacts above H-bonds in one frame."""
    use_style()
    rows = _interaction_pair_load()
    _interaction_pair_check(rows)
    xs = _interaction_pair_positions()
    # Two rows of the single figure's frame. The height doubles and the width does not: the
    # rows share one x axis of methods, which is the point of stacking them.
    fig, axes = plt.subplots(len(IX_PARTS), 2,
                             figsize=(IX_PAIR_ASPECT * IX_FIG_H,
                                      len(IX_PARTS) * IX_FIG_H * IX_PAIR_TALL),
                             dpi=220)
    fig.patch.set_facecolor("white")
    for row, (part, kinds) in enumerate(IX_PARTS):
        for ax, kind in zip(axes[row], kinds):
            _interaction_pair_panel(ax, kind, rows, xs, part)
    # THE UNNAMED ROW DROPS ITS LABELS OUTRIGHT, which the STANDALONE figures must never do.
    # There the names are drawn in colour "none" so tight_layout still measures them and the
    # two separate figures keep the same axes height. Inside ONE figure the row heights come
    # from the gridspec instead, so that reservation buys nothing and costs the entire band
    # -- 0.889 in of empty space between the rows, measured.
    for ax in axes[0]:
        ax.set_xticklabels([])
    # THE Y NAMES LINE UP ACROSS THE ROWS (2026-09-15). Each one is placed off its own tick
    # labels, and the rows carry different numbers -- two digits on the contacts row, one on
    # the H-bond row -- so left to itself the upper title sits further out than the lower and
    # the two read as a ragged edge even though the AXES are aligned to 0.0000 in. This pins
    # them to a common x. It runs BEFORE the layout because _interaction_stack_layout sizes
    # the left margin from the measured label overhang, which this changes.
    fig.align_ylabels([ax for row in axes for ax in row])
    _interaction_stack_layout(fig, axes)
    save(fig, out, "interaction_pair_all")
    print(f"  {len(IX_PARTS)} x 2 panels · {len(IX_PAIR_ARMS)} arms · "
          f"names on {', '.join(IX_NAMED_PARTS)} only")
