

# ════════════════════════════════════════════════════════════════════════════════
# fig-posebusters-{valid-per-atom,valid-heatmap,group-bars,check-failures,sucos-ecdf,
#                  sucos-per-atom}
# ════════════════════════════════════════════════════════════════════════════════
# SIZE IS THE CONFOUND, SO SIZE IS THE X AXIS. Every PoseBusters check gets harder as the
# molecule grows -- more rings to pucker, more angles to strain, more atoms to reach the
# protein -- and the arms draw different size mixes (24.9 heavy atoms for CoDE against
# TargetDiff's 22.2). A pooled validity rate is therefore partly a report of the size mix,
# the same trap the Vina numbers have, which is why the headline figure is per-atom with
# the size distribution drawn underneath it rather than a single pooled bar.
#
# SuCOS is the one check PoseBusters' `gen` config adds over `dock`: shape overlap x
# pharmacophore-feature overlap against the pocket's crystal ligand. It is NOT folded into
# `valid` -- every other column asks "is this pose physically possible", SuCOS asks "does it
# sit where the crystal ligand sits", and a de novo model is not trying to reproduce the
# crystal ligand. Its per-molecule values are recomputed by
# notebook/html/260910/fig-posebusters/build_sucos.py, which needs RDKit and the posebusters
# package (the `moleval` env); this file only draws what that script exported.
#
# THE FOUR CHECKED-IN sucos_*.png WERE RASTERIZED IN THAT `moleval` ENV, and it carries
# FreeType 2.14.3 against `voxbind`'s 2.6.1. Same matplotlib, same data, same code -- but a
# different glyph rasterizer, and tight_layout measures the axes off the text it lays out,
# so a redraw here is sub-pixel-shifted against them over the whole figure. Drawing them
# from `voxbind` is correct and identical to what build_sucos.py itself produces there; it
# just cannot be byte-compared against PNGs made by the other env's FreeType.

# Validity is a proportion, so its per-count curve is smoothed over +-PB_VALID_WIN atoms --
# see model_curve for why the strain and clash curves are not.
PB_VALID_WIN = 2
# A check no method fails above this often is a row of white space in the breakdown: it
# says only that PoseBusters ran it. The full counts stay in the JSON the recompute script
# writes. The filter stays a RATE even though the bars are counts -- it asks "is this row
# informative", and the arms hold different numbers of molecules.
PB_MIN_FAIL_PCT = 0.5
# The check-failure breakdown, split into stacked panels by how often a check fails, each on
# a linear axis scaled to itself (2026-09-14, on request; it was one symlog axis). The median
# rate over the eight arms puts them in three decades:
#   ~10%   bond angles 9.7, min distance 7.0, steric clash 5.7, ring non-flatness 5.4,
#          bond lengths 3.2
#   ~1%    internal energy 1.6, volume overlap 0.36
#   <=0.1% double bond flatness 0.26, aromatic ring flatness 0.02 -- zero for most sets
# Fixed rather than recomputed, so `core` and `all` split the same way; inside a panel the
# checks keep the figure's order (worst first). A check shown but in no panel is an error.
PB_FAIL_PANELS = (
    ("bond_lengths", "bond_angles", "non-aromatic_ring_non-flatness",
     "minimum_distance_to_protein", "internal_steric_clash"),
    ("volume_overlap_with_protein", "internal_energy"),
    ("aromatic_ring_flatness", "double_bond_flatness"),
)
# THE HEAT MAP'S GROUPS. The 20 scored dock-mode checks partitioned into what each one is
# actually asking about, so `valid` can be read as "passes all five" instead of as one
# number. EVERY CHECK IS IN EXACTLY ONE GROUP and the partition is asserted against the
# data: a PoseBusters release that adds a check breaks the build rather than quietly
# dropping it out of the picture, and no check can be double-counted into two groups.
#
# Six of the nine in `Protein clash & overlap` are VACUOUS HERE and pass by construction:
# the receptor every arm is scored against is the pocket10 crop, which carries no HETATM
# and no HOH, so the cofactor and water checks have nothing to measure. They stay in the
# group because the group is a partition of `valid` and dropping them would make the
# columns stop multiplying out to it -- not because they carry information.
#
# `Planarity`, `Ring pucker` and `Self-clash & energy` are not among the four groups this was
# first asked for; they exist because the partition has to be complete, and they are cut this
# way BY WHAT THE CHECK WANTS rather than by where PoseBusters lists it (2026-09-15, on
# request). `aromatic_ring_flatness` and `double_bond_flatness` both fail when something that
# should be FLAT is not, so they are one column; `non-aromatic_ring_non-flatness` is the exact
# opposite test -- `check_nonflat: True` flips the comparison, and it fails when a saturated
# 6-ring comes out PLANAR -- so it stands alone. It is the single worst check for the voxel
# arms (21-23% fail), and it was previously buried in a four-check column called `Strain.`,
# where its 78.9 read as 74.5 and collided in name with the PoseCheck strain-energy figures,
# which measure something else entirely. What is left, `internal_steric_clash` and
# `internal_energy`, is the ligand against ITSELF rather than against the protein -- hence
# `Self-clash & energy`, beside `Protein clash & overlap`.
PB_GROUPS = (
    ("Bond geometry", ("bond_lengths", "bond_angles")),
    ("Planarity", ("aromatic_ring_flatness", "double_bond_flatness")),
    ("Ring pucker", ("non-aromatic_ring_non-flatness",)),
    ("Valence & connectivity", ("sanitization", "all_atoms_connected", "inchi_convertible",
                                "no_radicals")),
    ("Protein clash & overlap", ("minimum_distance_to_protein", "volume_overlap_with_protein",
                                 "protein-ligand_maximum_distance",
                                 "minimum_distance_to_organic_cofactors",
                                 "minimum_distance_to_inorganic_cofactors",
                                 "minimum_distance_to_waters",
                                 "volume_overlap_with_organic_cofactors",
                                 "volume_overlap_with_inorganic_cofactors",
                                 "volume_overlap_with_waters")),
    ("Self-clash & energy", ("internal_steric_clash", "internal_energy")),
)
PB_HEAT_COL0 = "PoseBusters valid"
# THE RAMP IS THE DATA'S OWN RANGE: it ends at 100, a rate's ceiling, and starts at the
# WORST CELL DRAWN (2026-09-14, on request) -- FuncBind's 50.1% valid. A 0-100 ramp would
# spend half its range on ground no method stands on and paint the whole map one shade,
# which is the failure mode of a heat map whose numbers all sit near the top.
#
# Both variants take the range of the ALL-ARMS map, so a cell is the same colour in `core`
# as in `all`. Scaling `core` to its own worst cell (CoDE's 67.5) would make our two arms
# look as far apart as the eight are.
PB_HEAT_VMAX = 100.0
# THE COLOURBAR AND THE RAMP ARE THE HOUSE ONES (2026-09-14/15, on request): heat_colorbar()
# and HEAT_CMAP from 00_core, the same pair the per-rotbond strain grid draws. Both used to
# live in the rotbond family and be read across from here at call time; they moved up to the
# shared section so neither figure owns the other's colours. One hue means the ramp needs no
# reversal between the two -- darker is simply more of whatever the bar is named after, more
# strain there and a higher pass rate here.
# Luminance below which a cell's number is set in white instead of ink, so the text follows
# the background (2026-09-15, on request) rather than being one colour throughout. 0.55 is
# where the two swap places on this ramp: on #5F75E2 white is 4.1:1 against ink's 2.1:1,
# while on #B4BEF0 it is ink 4.5:1 against white's 1.2:1. Around #8291E8 they are equal, and
# that is where the threshold sits. It reads the FILL, never the value, so changing the ramp
# moves the flip point with it.
PB_HEAT_WHITE_TEXT_BELOW = 0.55
# Point sizes are NOT copied from the rotbond grid, which is three rows of nine panels on a
# 14.45 in canvas: same bar, this figure's scale. 13/11.5 x 1.3 with everything else.
PB_HEAT_CBAR_FS, PB_HEAT_CBAR_TICK_FS = 20.5, 18
# THE CELL SHAPE, 3 WIDE TO 2 TALL (2026-09-15, on request: square, then "2:3 정도"). It was
# neither before -- at a fixed 8.83 in width a cell came out 1.07 x 0.92 on `core` and
# 1.07 x 0.49 on `all`, so the same number sat in a near-square box in one variant and a
# 2.2:1 letterbox in the other. Now the figure is built FROM the cell, so both variants draw
# the same shape and `all` is simply the taller figure.
# THE WIDTH IS THE HEADING'S (2026-09-15, on request), not a round number, and what sets it
# is the widest ADJACENT PAIR rather than the widest heading: two neighbours each spend half
# their width towards the gutter between them, so "Bond len." (1.116 in at 17 pt) beside
# "Bond ang." (1.201) needs 1.159 plus a gutter, where "Bond ang." alone would only ask for
# 1.20. 1.29 leaves that worst pair 0.13 in of white and every other pair more -- the
# smallest column that still holds all eight headings on ONE line, which is what the short
# forms were for. EVERY MEASUREMENT HERE MOVED WITH THE TYPE when the map's text went up 30%
# (2026-09-15, on request): the column was 0.99 at 13 pt headings, and the gutter scaled with
# it, 0.10 -> 0.13, so the map keeps the same proportions at the larger size rather than
# getting tighter. The height follows from the ratio; "100.0" at 18 pt is 0.715 in, so a
# number still keeps a clear margin in its cell.
PB_HEAT_CELL_W = 1.29
PB_HEAT_CELL_ASPECT = 2 / 3      # cell height / cell width -- what set_aspect() takes
PB_HEAT_CELL_H = PB_HEAT_CELL_W * PB_HEAT_CELL_ASPECT
# The numbers in the cells. 12.5 -> 14 when the cell came in (the number is what the figure is
# read for, so it went the other way), then 14 -> 18 -> 21.5 with the 30% and 20% raises.
PB_HEAT_NUM_FS = 21.5
# The row names, which are not a tick label size any more: they match the headings, and both
# are 13 x 1.3 x 1.2. Held in a constant rather than written at the call because the two have to
# move together -- a map whose columns are named larger than its rows reads as two figures.
PB_HEAT_LABEL_FS = 20.5
# A tick per row and per column (2026-09-15, on request), where the block had none: the
# spines are hidden and the cells are separated by white, so a long row had nothing tying its
# name to it. They sit at the CELL CENTRES, with the labels, not on the cell edges. 4.5 x 1.3
# with the type, so a tick stays the same fraction of the name it points at.
PB_HEAT_TICK_LEN = 7.1
# What the axes does NOT cover, in inches: the row names to its left plus the colourbar and
# its name to the right, and the column headings above it. They size the canvas; set_aspect()
# is what actually holds the cell shape, so an allowance being off costs a margin, never the
# shape. Both moved with the type: 2.50 -> 3.75 over two raises, and the heading strip
# 0.35 -> 0.75 while the headings were stacked, back to 0.55 now they are one line again.
# THE SIDE ALLOWANCE IS DELIBERATELY OVER-GENEROUS: constrained layout under-reserves next to
# a FIXED-ASPECT axes, and the symptom is the longest row name, VoxBind(sigma=0.9), starting
# OFF the left edge of the canvas and losing its V. It has done that twice -- at 2.16 with
# 13 pt names, and again at 2.50 once the map went to eight columns. The spare width cannot
# change the cells: the block's width is what binds, so it becomes margin on the left, which
# is where the names are. Checked by reading the tick labels' own bboxes back off the drawn
# canvas rather than by eye: a clipped label is the one failure that survives a glance at the
# PNG, because the eye reads "VoxBind" from the letters that are left.
PB_HEAT_SIDE, PB_HEAT_HEAD = 3.75, 0.55
# A little air under the bottom row (2026-09-15). The WIDTH is what binds under a fixed
# aspect -- the side allowance above is deliberately generous -- so this does not change the
# cells at all: it lengthens the canvas, and constrained layout centres the block in what is
# left, which puts half of it under the last row. 0.24 buys about 26 px at 220 dpi. The top
# margin looks bigger in the numbers only because the column headings are reserved inside it;
# the white above the headings is the same 12 px the block has below it.
PB_HEAT_FOOT = 0.24
# The blank strip after the first column and the first row, in cell widths (2026-09-15, on
# request: the ink rule alone did not read as a division). A rule marks a boundary; a gap
# makes the block on either side of it a separate object, which is what the first column and
# the first row are.
PB_HEAT_GAP = 0.22
PB_HEAT_WRAP = 12                # characters per line of a column heading
# The headings sit at the row names' size, and the column width above is solved FOR this
# number -- raise one and the other has to move. 13 x 1.3.
PB_HEAT_HEAD_FS = PB_HEAT_LABEL_FS
# Every heading now fits on one line at this width, which is the point of the short forms;
# the wrap is the backstop that keeps a longer one from running into its neighbour instead of
# silently overlapping it.
# What a column is CALLED on the heat map (2026-09-15, on request). The keys are the group
# names, which stay in full everywhere a number is read back -- the CSV headers, the bar
# charts' axis labels, the README table. A heading here only has to say which column you are
# in, and at full length six of them took three lines each and half the figure's height.
PB_HEAT_SHORT = {
    "PoseBusters valid": "Valid.",
    "Bond geometry": "Bond.",
    # The split pair. `Bond` IS DROPPED RATHER THAN THE HEADING STACKED (2026-09-15, on
    # request: one line each). At 20.5 pt "Bond len." and "Bond ang." are 1.35 and 1.45 in,
    # and side by side they ask for a 1.58 in column -- the cell would have to grow right back
    # to the size the type raise was meant to shrink it against. Two lines fixed that and were
    # asked to go; what is left is to drop the word both share. These are the only two columns
    # on the map that could be a length or an angle OF anything else, they sit next to each
    # other, and both are named in full in the CSV and in the bar chart. A newline in a short
    # form would still be taken literally; everything without one goes through PB_HEAT_WRAP.
    "Bond lengths": "Length.",
    "Bond angles": "Angle.",
    # The two ring columns are the pair a reader can mix up, and they are kept apart by the
    # word that distinguishes them: `Planarity.` is what must be flat, `Pucker.` what must
    # not, and they sit side by side so the opposition is visible. `Ring pucker.` in full is
    # 1.118 in -- the widest heading on the map by a quarter inch, and next to `Planarity.`
    # it alone would have pushed every column from 0.99 to 1.08. Named in full in the CSV and
    # in the bar chart that draws the same group.
    "Planarity": "Planarity.",
    "Ring pucker": "Pucker.",
    "Valence & connectivity": "Valence.",
    "Protein clash & overlap": "Clash.",
    # 1.451 in, wider than the 1.29 cell -- deliberately. What has to clear is the GAP to the
    # heading beside it, and `Clash.` is 0.871, so the pair leaves 0.129 in of white, the same
    # gutter every other pair gets. It overhangs its own column by 0.08 in each side, which is
    # empty on the right (the colourbar's own margin) and is the narrow `Clash.` on the left.
    "Self-clash & energy": "Self-clash.",
}
# The groups that get a bar chart of their own (2026-09-15, on request), one file each. The
# others are left out because a bar chart of them says nothing a reader cannot already see:
# `Valence & connectivity` is 100.0 for every set, `Planarity` is 99.5+ for everything except
# AR, and `Self-clash & energy` moves over eight points across the whole field. `Ring pucker`
# takes the slot the four-check `Ring pucker & internal strain` used to hold, and is the same
# bar with the three passengers removed -- it was always the check that moved it. Order is
# the heat map's, so a reader moving between them meets the groups in the same order.
PB_GROUP_BARS = ("Bond geometry", "Protein clash & overlap", "Ring pucker")
PB_GROUP_BAR_H = 0.42            # inches per method row
# The axis runs the full 0-100 even though nothing is below 68. A bar's LENGTH is its value,
# so it has to start at a true zero -- floored at 60 the same numbers would read as a
# fourfold spread. At this range the differences are legible anyway: 70.4 against 100 is a
# third of the axis.
PB_GROUP_BAR_XLIM = (0, 109)
PB_SUCOS_THRESHOLD = 0.4         # gen.yml's own value
PB_BIN_COLS = ["n_molecules", "atoms_mean", "n_posebusters", "pb_valid_rate",
               "pb_valid_rate_size_standardized"]


def _posebusters_rate(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def _posebusters_std_weights(data, p79_rows):
    """The common size distribution for direct standardization -- every arm pooled.

    The weights come from the arms that are actually IN the comparison, not from every key
    the loader returned: a scoring run in progress leaves partial rows in p79_rows, and
    folding those into the standard population moves every arm's standardized rate."""
    return collections.Counter(
        r["n"] for _, key, _ in arms_for("v", data) for r in p79_rows[key]
        if r["v"] is not None)


def _posebusters_std_rate(rows, weights):
    """Each arm's rate WITHIN every 1-heavy-atom stratum, re-weighted by one common size
    distribution, so what is left is validity at matched size. Strata where the arm holds
    <10 molecules are dropped, not extrapolated. This is the number to compare arms with;
    the crude rate is what the arm actually produced, and the two answer different
    questions."""
    per = by_size(rows, "v")
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * float(np.mean(per[n])); den += w
    return 100 * num / den if den else float("nan")


def _posebusters_block(rows, weights):
    v = [r["v"] for r in rows if r["v"] is not None]
    std = _posebusters_std_rate(rows, weights)
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_posebusters": len(v),
        "pb_valid_rate": round(_posebusters_rate(v) / 100, 4) if v else None,
        "pb_valid_rate_size_standardized": round(std / 100, 4) if np.isfinite(std) else None,
    }


# ── figure 1: validity per heavy-atom count, over each arm's size distribution ───
def _posebusters_valid_panel(out, arms, variant, p79_rows, refrows):
    per = {key: by_size(p79_rows[key], "v") for _, key, _ in arms}
    ref_per = by_size(refrows, "v")
    xs = x_range(per, arms)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    rate100 = lambda v: 100 * float(np.mean(v))
    top.plot(xs, reference_curve(ref_per, xs, rate100), color=REF_COLOR,
             lw=REF_LW, ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, model_curve(per[key], xs, rate100, win=PB_VALID_WIN),
                 color=soft(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    legend(top, arm_handles(arms, paint=soft), loc="lower left", fontsize=11.5)

    size_distribution(bot, xs, per, ref_per, arms, paint=soft)
    furniture(bot, ylabel="% of ligands", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    fig.align_ylabels((top, bot))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, out, f"pb_valid_per_atom_{variant}")
    return xs


# ── figure 2: which checks fail ─────────────────────────────────────────────────
def _posebusters_check_failures(data, p79_rows, refrows):
    """{key: {n_mols, counts, rates}} over every arm, plus the crystal ligands."""
    out = {}
    for lab, key, _ in arms_for("v", data):
        rows = [r for r in p79_rows[key] if r["v"] is not None]
        cnt = collections.Counter(c for r in rows for c in r["f"])
        out[key] = {"n_mols": len(rows), "counts": dict(cnt),
                    "rates": {k: 100 * v / len(rows) for k, v in cnt.items()}}
    rrows = [r for r in refrows if r["v"] is not None]
    cnt = collections.Counter(c for r in rrows for c in r["f"])
    out["reference"] = {"n_mols": len(rrows), "counts": dict(cnt),
                        "rates": {k: 100 * v / max(1, len(rrows)) for k, v in cnt.items()}}
    return out


def _posebusters_wrap_check(name):
    """The PoseBusters check name, wrapped instead of abbreviated.

    `non-aromatic_ring_non-flatness` on one line is 30 characters and was eating 2.9 in of
    a 7.6 in figure -- nearly half the width -- as a tick label. Abbreviating it is the
    wrong fix: this check passes when a non-aromatic ring is sufficiently NON-flat (it is
    `check_nonflat: True` in dock.yml, threshold 0.1 A), so failing it means a saturated
    ring came out planar, and every shortening of that name I tried either flipped its
    sense or read as the aromatic check next to it. Wrapping is free and exact.

    textwrap is not one of draw.py's shared imports and this is the only figure that needs
    it, so it is fetched here rather than at the top of the file. Its exact wrapping --
    including breaking on the hyphens -- is what the tick labels are, so this must stay
    textwrap and not a hand-rolled splitter."""
    return "\n".join(__import__("textwrap").wrap(name.replace("_", " "), 18))


def _posebusters_failures_panel(out, arms, variant, fails):
    """Rows are checks, bars are the SHARE of each set's molecules that fail them.

    Rates, not counts, and that is what lets the crystal ligands be an ordinary bar here:
    on a count axis 79 of them against ~7,900 generated molecules put their worst row at 2
    molecules, invisible beside a bar of 1,822, and they had to be drawn as a rate-matched
    marker instead. The price is that a rate hides its denominator -- the reference's 2.5%
    IS those 2 molecules, and it carries about +-1.8 points of binomial noise against the
    arms' +-0.2 -- so its bar alone keeps its n in the key. Counts for every arm and every
    check stay in posebusters_check_failures.json.
    """
    # The reference is one more series, first in every group and first in the key.
    series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    shown = [fails[key] for _, key in series]
    names = sorted({k for g in shown for k in g["counts"]
                    if max(h["rates"].get(k, 0) for h in shown) >= PB_MIN_FAIL_PCT},
                   key=lambda k: -max(g["rates"].get(k, 0) for g in shown))
    # One row of the y axis is 1.0 apart, so the bars of a group must fit inside that:
    # a fixed height works for three series and silently overlaps the neighbouring groups
    # at nine (9 x 0.19 = 1.71), which reads as bars detached from their labels. Derive it.
    h = 0.86 / len(series)
    # THE KEY GOES OUTSIDE THE AXES, centred along the bottom, 3 x 3. There is no empty
    # corner inside:
    # the long bars fill the top and the right, and the in-axes box this used to carry
    # reached far enough left to bury the bottom rows' bars -- `double bond flatness`
    # looked empty while AR, DecompDiff and Pocket2Mol were failing it 3.1, 1.8 and 1.4% of
    # the time. Outside it covers nothing. Three columns only fit because the bars are
    # rates now: an arm's own n no longer has to be in the key for its bar to be readable.
    ncol = 3
    leg_rows = -(-len(series) // ncol)
    # Row pitch: enough that nine thin bars stay readable, without turning a 7.6 in wide
    # figure into a 10 in tall one. Plus the strip the key needs at the bottom.
    # STACKED PANELS, LINEAR X (2026-09-14, on request; it was one symlog axis). The rates run
    # 0.01% to 23%, and on one linear axis everything under ~2% was a stub against FuncBind's
    # 23%; symlog fixed that but made every bar's length a log reading. Three panels -- see
    # PB_FAIL_PANELS -- each give a decade of checks its own linear scale, so a length is a
    # rate again, and a zero is still a bar of nothing. Panel height = its check count, so
    # the bars are the same thickness in every panel; the x scales differ, so each keeps
    # its own tick labels.
    panels = [[k for k in names if k in p] for p in PB_FAIL_PANELS]
    leftover = [k for k in names if not any(k in p for p in PB_FAIL_PANELS)]
    if leftover:
        raise ValueError(f"PoseBusters checks in no PB_FAIL_PANELS panel: {leftover}")
    panels = [p for p in panels if p]
    fig = plt.figure(figsize=(FIG_W, (0.40 if len(arms) <= 3 else 0.62) * len(names)
                              + 1.6 + 0.45 * (len(panels) - 1) + 0.30 * leg_rows + 0.2),
                     dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[len(p) for p in panels])
    for j, panel in enumerate(panels):
        ax = fig.add_subplot(gs[j, 0])
        ax.set_facecolor("white")
        ys = np.arange(len(panel))[::-1]
        for i, (lab, key) in enumerate(series):
            # top of the group downwards, so the key's order IS the order of the bars
            ax.barh(ys + ((len(series) - 1) / 2 - i) * h,
                    [fails[key]["rates"].get(k, 0.0) for k in panel],
                    height=h, color=soft(lab), edgecolor=soft(lab), lw=0.8, zorder=3)
        top = max(g["rates"].get(k, 0.0) for g in shown for k in panel)
        furniture(ax, ylabel=None, xlim=(0, top * 1.08), xloc=None,
                  xlabel="Molecules failing the check (%)" if j == len(panels) - 1 else None)
        ax.grid(False, axis="y")
        ax.set_yticks(ys)
        ax.set_yticklabels([_posebusters_wrap_check(k) for k in panel], fontsize=12)
        ax.set_ylim(-0.6, len(panel) - 0.4)
    # soft() tints, the eight-method figures' colours (2026-09-14, on request)
    handles = [Patch(facecolor=soft(lab), edgecolor=soft(lab),
                     label=display(lab) + ("  (n=79)" if key == "reference" else ""))
               for lab, key in series]
    fit(fig, pad=0.5)
    # tight_layout does not see a FIGURE legend, so it has just laid the axes out over the
    # strip the key occupies. Place the key, measure what it actually took, and RAISE the
    # axes by that much -- reserving the measured height keeps the tick labels and the x
    # label, which sit BELOW the axes box and would otherwise end up behind an opaque
    # legend; reserving up to its top edge instead would bury them. Columns are dropped
    # if the key overruns the figure, which is cheaper than trusting a width estimate.
    fs = 12.5 if len(arms) <= 3 else 11
    for c in range(ncol, 0, -1):
        leg = legend(fig, handles, loc="lower center", ncol=c, fontsize=fs,
                     bbox_to_anchor=(0.5, 0.008))
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        if (bb.x0 >= 0.003 and bb.x1 <= 0.997) or c == 1:
            break
        leg.remove()
    fig.subplots_adjust(bottom=min(0.6, fig.subplotpars.bottom
                                  + (bb.y1 - bb.y0) + 0.016))
    save(fig, out, f"pb_check_failures_{variant}")


@figure("fig-posebusters-valid-per-atom", needs=("metrics.json (posebusters block)",))
def draw_posebusters_valid_per_atom(out):
    """PoseBusters dock-mode validity against ligand size, over each arm's size mix."""
    data, p79_rows, refrows = pose_data()
    use_style()
    weights = _posebusters_std_weights(data, p79_rows)
    ranges = {}
    for variant, arms in variants("v", data):
        ranges[variant] = _posebusters_valid_panel(out, arms, variant, p79_rows, refrows)

    scored = arms_for("v", data)
    pooled = {lab: _posebusters_block(p79_rows[key], weights) for lab, key, _ in scored}
    ref_block = _posebusters_block(refrows, weights)
    by_bin = {lab: [_posebusters_block([r for r in p79_rows[key] if bin_of(r["n"]) == b],
                                       weights)
                    for b in range(len(BIN_LABELS))] for lab, key, _ in scored}
    by_bin[REF_LABEL] = [_posebusters_block([r for r in refrows if bin_of(r["n"]) == b],
                                            weights)
                         for b in range(len(BIN_LABELS))]

    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    print(f"  {'arm':16s} {'atoms':>6s} {'PB-valid':>9s} {'size-std':>9s} {'mols':>7s}")
    for lab, key, _ in scored:
        r = pooled[lab]
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {100 * r['pb_valid_rate']:8.1f}% "
              f"{100 * r['pb_valid_rate_size_standardized']:8.1f}% {r['n_molecules']:7d}")
    print(f"  {REF_LABEL:16s} {ref_block['atoms_mean']:6.1f} "
          f"{100 * ref_block['pb_valid_rate']:8.1f}% {'—':>9s} "
          f"{ref_block['n_molecules']:7d}")
    print(f"\n  PB-valid by bin ({' · '.join(BIN_LABELS)}):")
    for lab, key, _ in scored:
        print(f"    {lab:16s} " + "  ".join(
            f"{100 * b['pb_valid_rate']:5.1f}%" if b["pb_valid_rate"] is not None else "    -"
            for b in by_bin[lab]))

    write_csv(out, "posebusters_by_atom_range", ["arm", "bin"] + PB_BIN_COLS,
              [[arm, lab] + [r[c] for c in PB_BIN_COLS]
               for arm, rowset in by_bin.items()
               for lab, r in zip(BIN_LABELS, rowset)])


@figure("fig-posebusters-check-failures", needs=("metrics.json (posebusters block)",))
def draw_posebusters_check_failures(out):
    """Which PoseBusters checks fail, per method, as a share of its own molecules."""
    data, p79_rows, refrows = pose_data()
    use_style()
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    for variant, arms in variants("v", data):
        _posebusters_failures_panel(out, arms, variant, fails)


# ── validity heat map ───────────────────────────────────────────────────────────
def _posebusters_group_rates(rows, checked, columns=()):
    """(% passing each group, % valid, n scored) for one set of molecules.

    A GROUP IS PASSED WHEN THE MOLECULE FAILS NOTHING IN IT, which is per-molecule and not
    recoverable from the per-check rates: two checks each failing 5% of molecules are one
    column at 90% if they fail different molecules and at 95% if they fail the same ones.
    That is also why the group columns do not multiply out to the `valid` column.

    `columns` adds further (name, checks) pairs to rate the same way -- the heat map passes
    its split of Bond geometry, which is a narrower cut of the same partition and not a group
    in its own right."""
    scored = [r for r in rows if r["v"] is not None]
    n = len(scored)
    if not n:
        return {}, float("nan"), 0
    seen = {c for r in scored for c in r["f"]}
    if not seen <= checked:
        raise KeyError(f"PoseBusters check outside PB_GROUPS: {sorted(seen - checked)} -- "
                       "add it to a group, or the heat map stops being a partition of `valid`")
    out = {}
    for name, checks in tuple(PB_GROUPS) + tuple(columns or ()):
        want = set(checks)
        out[name] = 100 * sum(not (want & set(r["f"])) for r in scored) / n
    return out, 100 * sum(bool(r["v"]) for r in scored) / n, n


# BOND GEOMETRY IS TWO COLUMNS ON THE MAP (2026-09-15, on request), one per check, because
# the two come apart: TargetDiff passes 98.9% on lengths and 77.5% on angles, Pocket2Mol is
# the other way round (89.6 / 98.0), and the single column averaged that away. PB_GROUPS is
# left alone -- it is the partition `valid` is graded on and what the bar charts and the CSV's
# `group` field mean -- so this is a cut WITHIN one group, listed here and nowhere else.
PB_HEAT_SPLIT = {"Bond geometry": (("Bond lengths", ("bond_lengths",)),
                                   ("Bond angles", ("bond_angles",)))}
PB_HEAT_COLS = tuple(c for name, checks in PB_GROUPS
                     for c in PB_HEAT_SPLIT.get(name, ((name, checks),)))


def _posebusters_heat_norm(grid):
    """One ramp for every PoseBusters figure: floored at the worst cell of the ALL-arms map.

    Both map variants and all three bar charts take it, so a value is the same colour
    wherever it is drawn -- which is the whole reason the bars gave up the method palette."""
    return matplotlib.colors.Normalize(float(grid.min()), PB_HEAT_VMAX)


def _posebusters_heatmap_grid(arms, p79_rows, refrows):
    """(column names, row labels, the rates, n per row) -- the map, before it is drawn."""
    series = [(REF_LABEL, refrows)] + [(lab, p79_rows[key]) for lab, key, _ in arms]
    checked = {c for _, checks in PB_GROUPS for c in checks}
    cols = [PB_HEAT_COL0] + [name for name, _ in PB_HEAT_COLS]
    grid, ns, bars = [], [], {name: [] for name, _ in PB_GROUPS}
    for lab, rows in series:
        groups, valid, n = _posebusters_group_rates(rows, checked, PB_HEAT_COLS)
        grid.append([valid] + [groups[name] for name, _ in PB_HEAT_COLS])
        ns.append(n)
        # The GROUP rates travel alongside the map's columns: Bond geometry is two cells here
        # but stays one bar, and a group's rate is per-molecule, so it cannot be recovered
        # from the two halves after the fact.
        for name, _ in PB_GROUPS:
            bars[name].append(groups[name])
    return (cols, [lab for lab, _ in series], np.array(grid, float), ns,
            {k: np.array(v, float) for k, v in bars.items()})


def _posebusters_heatmap_panel(out, variant, built, norm):
    cols, labels, grid, ns, bars = built
    series = list(zip(labels, ns))

    # The canvas is the CELL BLOCK plus its margins, so both variants draw the same square
    # cell and `all` is simply the taller figure. The block is measured in cell widths, the
    # spacer strip included, exactly as the mesh edges below are built.
    # CONSTRAINED LAYOUT, as the rotbond grid uses -- tight_layout does not see a colourbar's
    # label, and fit()'s overrun correction moves the main axes, not the bar's, so the name
    # came out sliced off the right edge of the core map.
    nrow, ncol = grid.shape
    gap = PB_HEAT_GAP
    fig, ax = plt.subplots(
        figsize=(PB_HEAT_CELL_W * (ncol + gap) + PB_HEAT_SIDE,
                 PB_HEAT_CELL_H * (nrow + gap) + PB_HEAT_HEAD + PB_HEAT_FOOT),
        dpi=220, layout="constrained")
    # Constrained layout's own padding, 3 pt by default, which is what the canvas edge leaves
    # round the whole figure. At 3 pt the longest row name ended 9 px off the edge -- drawn in
    # full, but reading as if it had been cut -- and NO amount of extra canvas fixes it: the
    # layout reserves exactly the label's width plus this pad, so the slack lands somewhere
    # else. 0.10 in is the same 3 pt grown with the type.
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.10)
    fig.patch.set_facecolor("white")
    cmap = HEAT_CMAP
    # A MESH, NOT AN IMAGE, so the gap can be a fraction of a cell. imshow lays cells on one
    # uniform grid, where the only available spacer is a whole empty column; pcolormesh takes
    # the edge coordinates, so PB_HEAT_GAP is inserted into them and the blank strip is
    # exactly as wide as it should be. The spacer cells are NaN and masked, which leaves the
    # figure's own white showing through rather than a painted rectangle.
    xe = np.array([0.0, 1.0] + [1.0 + gap + k for k in range(ncol)])
    ye = np.array([0.0, 1.0] + [1.0 + gap + k for k in range(nrow)])
    xc = [0.5] + [1.5 + gap + k for k in range(ncol - 1)]
    yc = [0.5] + [1.5 + gap + k for k in range(nrow - 1)]
    mesh = np.insert(np.insert(grid, 1, np.nan, axis=0), 1, np.nan, axis=1)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(mesh), cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=1.4)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[-1], ye[0])              # first row at the top, as imshow had it
    # A cell is 1 x 1 in DATA units, so the axes aspect is the cell's shape -- and it holds
    # even if the canvas arithmetic above is ever off, which a hand-set figure height would not.
    ax.set_aspect(PB_HEAT_CELL_ASPECT)
    for i in range(nrow):
        for j in range(ncol):
            v = grid[i, j]
            # Ink or white off the FILL's own luminance, never off the value: which cells
            # are dark is a property of the ramp, and the ramp is shared with another figure.
            r, g, b, _ = cmap(norm(v))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < PB_HEAT_WHITE_TEXT_BELOW
            ax.text(xc[j], yc[i], f"{v:.1f}", ha="center", va="center",
                    fontsize=PB_HEAT_NUM_FS, color="white" if dark else INK, zorder=3)
    # A rule after the first column and after the first row (2026-09-15, on request). Both
    # mark a change of KIND, not of degree: column 0 is the aggregate rather than a sixth
    # category -- and the groups do not multiply out to it, see _posebusters_group_rates --
    # while row 0 is the crystal ligands, the benchmark the arms are read against. The rule
    # runs down the middle of the gap, so the two together read as one division.
    ax.axvline(1.0 + gap / 2, color=INK, lw=2.0, zorder=5)
    ax.axhline(1.0 + gap / 2, color=INK, lw=2.0, zorder=5)
    ax.set_xticks(xc)
    # Wrapped at PB_HEAT_WRAP, not at the 18 the check-failure tick labels use: a column here
    # is ~1.2 in wide, and at 18 the group names ran into each other across the header.
    # A short form that already carries a newline is used as written -- textwrap would treat
    # it as whitespace and put the heading back on one line, which is the opposite of the
    # point. The wrap stays as the backstop for everything else.
    heads = [PB_HEAT_SHORT.get(c, c) for c in cols]
    ax.set_xticklabels([h if "\n" in h else "\n".join(__import__("textwrap").wrap(h, PB_HEAT_WRAP))
                        for h in heads], fontsize=PB_HEAT_HEAD_FS)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(yc)
    # The method name alone (2026-09-15, on request). Every cell is a RATE, so a reader does
    # not need the denominator to compare two of them, and the counts crowded nine already
    # long labels. They are still in posebusters-valid-heatmap.csv, one column per row.
    # "Reference", not the shared REF_LABEL, exactly as the rotbond grid titles its own first
    # panel: with the units gone from the row there is nothing left for "ligand" to
    # disambiguate, and it is the longest label in the column.
    ax.set_yticklabels(["Reference" if lab == REF_LABEL else display(lab)
                        for lab, _ in series], fontsize=PB_HEAT_LABEL_FS)
    ax.tick_params(length=PB_HEAT_TICK_LEN, width=AXIS_LW, colors=INK, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    # The bar's name wraps when the bar is shorter than the name. One line measures 2.13 in
    # at 13 pt and the bar is HEAT_CBAR_SHRINK of the block, so `core`'s three rows give it
    # 1.84 in and `all`'s nine give 5.6: the short map takes two lines ("% of molecules" is
    # 1.38 in), the tall one stays on one. Scaling the type down instead would have put the
    # bar's name below its own tick numbers.
    bar_in = HEAT_CBAR_SHRINK * PB_HEAT_CELL_H * (nrow + gap)
    label = ("% of molecules passing" if bar_in >= 2.13 * PB_HEAT_CBAR_FS / 13
             else "% of molecules\npassing")
    heat_colorbar(fig, ax, norm, label,
                  label_fs=PB_HEAT_CBAR_FS, tick_fs=PB_HEAT_CBAR_TICK_FS)
    save(fig, out, f"pb_valid_heatmap_{variant}")


def _posebusters_group_bar(out, variant, built, name, norm):
    """One group, one bar per method, as the share of its molecules that pass the group.

    PASS, not fail, so a bar reads the same way round as the heat map cell it comes from --
    both come out of one call to _posebusters_heatmap_grid, so a bar and a cell cannot
    disagree. A bar is always the whole GROUP, which for Bond geometry is now two cells. The crystal ligands are the first bar rather than a rule: this is
    a like-for-like quantity here, measured on the same 79 pockets by the same code."""
    cols, labels, grid, ns, bars = built
    vals = bars[name]
    fig, ax = plt.subplots(figsize=(FIG_W * 0.86, PB_GROUP_BAR_H * len(labels) + 1.35),
                           dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(len(labels))[::-1]
    # THE HOUSE HEAT RAMP, NOT THE METHOD PALETTE (2026-09-15, on request). A bar and the heat
    # map cell it comes from are now the same colour as well as the same number, because both
    # go through HEAT_CMAP under the SAME norm -- the one floored at the worst cell of the
    # whole map, passed in rather than recomputed here so a bar cannot key itself to a
    # different scale than the map does. The method's own colour is given up for that: on a
    # chart with one bar per method the name at the end of every bar already says which method
    # it is, so identity colour was spending the figure's only free channel on a label it
    # already has, while LENGTH and colour now say the same thing twice.
    ax.barh(ys, vals, height=0.66, color=[HEAT_CMAP(norm(v)) for v in vals], zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 1.4, y, f"{v:.1f}", va="center", ha="left", fontsize=11.5, color=INK,
                zorder=4)
    # The group's name and the quantity on two lines. On one, "Protein clash & overlap - % of
    # molecules passing" is wider than the figure, and a centred x label that overruns cannot
    # be rescued by fit(): it just loses its right-hand end off the canvas.
    furniture(ax, ylabel=None, xlabel=f"{name}\n% of molecules passing", xloc=None,
              xlim=PB_GROUP_BAR_XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.grid(False, axis="y")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{display(lab)}  ({n:,})" for lab, n in zip(labels, ns)], fontsize=13)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    fit(fig, pad=0.5)
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    save(fig, out, f"pb_group_bar_{slug}_{variant}")


@figure("fig-posebusters-group-bars", needs=("metrics.json (posebusters block)",))
def draw_posebusters_group_bars(out):
    """Each check group on its own, one bar per method -- the heat map's columns, unrolled."""
    data, p79_rows, refrows = pose_data()
    use_style()
    drawn = {variant: _posebusters_heatmap_grid(arms, p79_rows, refrows)
             for variant, arms in variants("v", data)}
    norm = _posebusters_heat_norm(drawn["all"][2])
    for variant, built in drawn.items():
        for name in PB_GROUP_BARS:
            _posebusters_group_bar(out, variant, built, name, norm)
    _, labels, _, _, bars = _posebusters_heatmap_grid(arms_for("v", data), p79_rows, refrows)
    for name in PB_GROUP_BARS:
        vals = bars[name]
        order = np.argsort(vals)[::-1]
        print(f"  {name}: " + ", ".join(f"{labels[i]} {vals[i]:.1f}" for i in order[:3])
              + f"  ...  worst {labels[order[-1]]} {vals[order[-1]]:.1f}")


@figure("fig-posebusters-valid-heatmap", needs=("metrics.json (posebusters block)",))
def draw_posebusters_valid_heatmap(out):
    """PoseBusters validity as a method x check-group heat map, with the per-check table."""
    data, p79_rows, refrows = pose_data()
    use_style()
    drawn = {variant: _posebusters_heatmap_grid(arms, p79_rows, refrows)
             for variant, arms in variants("v", data)}
    norm = _posebusters_heat_norm(drawn["all"][2])
    for variant, built in drawn.items():
        _posebusters_heatmap_panel(out, variant, built, norm)

    cols, labels, grid, ns, _ = drawn["all"]
    worst = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"  colour ramp {norm.vmin:.1f}-{norm.vmax:.0f}%, floored at the worst cell: "
          f"{labels[worst[0]]} / {cols[worst[1]]}")
    # The console table heads its columns with the MAP's short names: cut to nine characters
    # the two halves of Bond geometry both came out "Bond".
    print(f"  {'method':16s} {'n':>7s} "
          + " ".join(f"{PB_HEAT_SHORT.get(c, c).replace(chr(10), ' ')[:9]:>9s}" for c in cols))
    for lab, row, n in zip(labels, grid, ns):
        print(f"  {lab:16s} {n:7,d} " + " ".join(f"{v:8.1f}%" for v in row))
    write_csv(out, "posebusters_valid_heatmap",
              ["method", "n_molecules"] + [f"pct_pass_{c}" for c in cols],
              [[lab, n] + [round(float(v), 2) for v in row]
               for lab, row, n in zip(labels, grid, ns)])

    # The per-check table the groups are built from -- a group at 99.9% can be one check at
    # 99.9% or four at 100 and one at 99.9, and only this says which.
    checks = [c for _, cs in PB_GROUPS for c in cs]
    group_of = {c: name for name, cs in PB_GROUPS for c in cs}
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    rows = []
    for lab, key, _ in [(REF_LABEL, "reference", None)] + list(arms_for("v", data)):
        g = fails[key]
        rows += [[lab, g["n_mols"], group_of[c], c, g["counts"].get(c, 0),
                  round(100 - g["rates"].get(c, 0.0), 3)] for c in checks]
    write_csv(out, "posebusters_check_pass_rates",
              ["method", "n_molecules", "group", "check", "n_failed", "pct_pass"], rows)


# ── SuCOS ───────────────────────────────────────────────────────────────────────
_PB_SUCOS_CACHE = {}


def _posebusters_sucos_rows():
    """{key: [{t, n, sucos}]}, from the per-molecule export.

    build_sucos.py computes these with `check_sucos(..., sucos_threshold=0.4)` -- exactly
    what gen.yml configures -- and caches them beside itself; it needs RDKit, posebusters
    and the `moleval` env, none of which drawing does. Reading its export is the same code
    path that script itself takes on a warm cache."""
    if not _PB_SUCOS_CACHE:
        for _, key, _ in ARMS:
            path = legacy("fig-posebusters", f"sucos_per_molecule_{key}.json")
            _PB_SUCOS_CACHE[key] = json.load(open(path))["molecules"]
    return _PB_SUCOS_CACHE


def _posebusters_sucos_p79():
    """The same, restricted to the 79-pocket like-for-like set."""
    p79 = set(P79)
    return {key: [r for r in rows if r["t"] in p79]
            for key, rows in _posebusters_sucos_rows().items()}


def _posebusters_sucos_standardized(rows, weights, f):
    """The arm's SuCOS WITHIN each 1-heavy-atom stratum, re-weighted by `weights`. SuCOS
    turns out to be nearly flat in size, so this barely moves -- which is itself worth
    recording, because every other metric in this section is size-confounded."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["n"]].append(r["sucos"])
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * f(per[n]); den += w
    return num / den if den else float("nan")


def _posebusters_sucos_paired(a_rows, b_rows):
    """Per-pocket mean SuCOS, a - b, over the pockets both cover. Paired because the
    pockets differ from each other far more than the arms do: an unpaired comparison of
    two arms is mostly a comparison of which pockets each happened to cover."""
    def by_pocket(rows):
        d = collections.defaultdict(list)
        for r in rows:
            d[r["t"]].append(r["sucos"])
        return {t: float(np.mean(v)) for t, v in d.items()}
    a, b = by_pocket(a_rows), by_pocket(b_rows)
    ts = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in ts]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    half = 1.96 * sd / max(1, len(diff)) ** 0.5
    return {"n_pockets": len(ts), "mean_diff": round(float(np.mean(diff)), 4),
            "median_diff": round(st.median(diff), 4),
            "ci95": [round(float(np.mean(diff)) - half, 4),
                     round(float(np.mean(diff)) + half, 4)],
            "a_higher_in": sum(d > 0 for d in diff)}


@figure("fig-posebusters-sucos-ecdf", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_ecdf(out):
    """SuCOS against the pocket's crystal ligand, as a distribution, with gen's 0.4 mark."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    for variant, arms in variants():
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for lab, key, _ in arms:
            v = np.sort([r["sucos"] for r in p79_rows[key]])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=soft(lab),
                    lw=MODEL_LW, zorder=5, solid_capstyle="round")
        # gen.yml would call everything left of this line invalid.
        ax.axvline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
        # Horizontal and high: rotated against the line it ran straight through the curves.
        ax.text(PB_SUCOS_THRESHOLD + 0.015, 0.97, f"gen threshold {PB_SUCOS_THRESHOLD}",
                ha="left", va="top", fontsize=11, color=INK)
        furniture(ax, ylabel="Cumulative share",
                  xlabel="SuCOS vs the pocket's crystal ligand", xloc=None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        legend(ax, arm_handles(arms, include_ref=False, paint=soft), loc="upper left",
               fontsize=11.5)
        fit(fig, pad=0.5)
        save(fig, out, f"sucos_ecdf_{variant}")

    # One common size distribution -- every arm pooled -- for direct standardization.
    weights = collections.Counter(r["n"] for rows in p79_rows.values() for r in rows)
    print(f"\n  {'arm':16s} {'mean':>7s} {'median':>7s} {'q25':>7s} {'q75':>7s} "
          f"{'std mean':>9s} {'>=0.4':>8s} {'mols':>7s}")
    rates = {}
    for lab, key, _ in ARMS:
        v = [r["sucos"] for r in p79_rows[key]]
        over = 100 * sum(x >= PB_SUCOS_THRESHOLD for x in v) / len(v)
        rates[lab] = round(over / 100, 4)
        std = round(_posebusters_sucos_standardized(p79_rows[key], weights, np.mean), 4)
        print(f"  {lab:16s} {np.mean(v):7.3f} {st.median(v):7.3f} "
              f"{np.percentile(v, 25):7.3f} {np.percentile(v, 75):7.3f} "
              f"{std:9.3f} {over:7.1f}% {len(v):7d}")

    # The claim this figure exists to support, stated as a paired test rather than a
    # difference of two pooled means.
    print()
    for name, (a, b) in (("CoDE - VoxBind", ("ours_v1", "vanilla")),
                         ("CoDE - TargetDiff", ("ours_v1", "targetdiff")),
                         ("CoDE - DecompDiff", ("ours_v1", "decompdiff"))):
        d = _posebusters_sucos_paired(p79_rows[a], p79_rows[b])
        print(f"    {name:30s} {d['mean_diff']:+.4f} "
              f"(95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}), "
              f"higher in {d['a_higher_in']}/{d['n_pockets']} pockets")

    # What running config="gen" instead of "dock" would have cost, stated rather than
    # implied: `valid` there is all-must-pass INCLUDING sucos_within_threshold.
    try:
        pb = json.load(open(legacy("fig-posebusters", "posebusters_summary.json")))
        print(f"\n  {'arm':16s} {'dock valid':>11s} {'gen valid (upper bound)':>24s}")
        for lab, key, _ in ARMS:
            # An arm whose PoseBusters run has not finished is simply absent from that
            # summary; skip its row rather than dropping the whole table.
            if lab not in pb.get("arms", {}):
                continue
            dock = pb["arms"][lab]["p79"]["pb_valid_rate"]
            print(f"  {lab:16s} {100 * dock:10.1f}% {100 * dock * rates[lab]:23.1f}%")
        print("    (upper bound: the product assumes the two are independent; the true gen\n"
              "     rate is the share passing BOTH, which cannot exceed either factor)")
    except (OSError, KeyError):
        pass


@figure("fig-posebusters-sucos-per-atom", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_per_atom(out):
    """SuCOS against ligand size — mean and median, core and all."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    ranges = {}
    for variant, arms in variants():
        per = {key: by_size(p79_rows[key], "sucos") for _, key, _ in arms}
        xs = x_range(per, arms)
        for stat, f in (("mean", lambda v: float(np.mean(v))),
                        ("median", lambda v: float(np.median(v)))):
            fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            for lab, key, _ in arms:
                ax.plot(xs, model_curve(per[key], xs, f), color=soft(lab),
                        lw=MODEL_LW, zorder=5, solid_capstyle="round")
            ax.axhline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
            furniture(ax, ylabel=f"SuCOS {stat}", xlabel=X_LABEL,
                      xlim=(xs[0] - 0.6, xs[-1] + 0.6))
            ax.set_ylim(0, 1)
            legend(ax, arm_handles(arms, include_ref=False, paint=soft), loc="upper left",
                   fontsize=11.5)
            fit(fig, pad=0.5)
            save(fig, out, f"sucos_per_atom_{stat}_{variant}")
        ranges[variant] = xs
    print("  per-atom x range: "
          + " · ".join(f"{v} {xs[0]}-{xs[-1]}" for v, xs in ranges.items()))
