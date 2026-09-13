

# ════════════════════════════════════════════════════════════════════════════════
# fig-similarity-{dumbbell,bars,scatter,table,novelty} — likeness to the crystal ligand
# ════════════════════════════════════════════════════════════════════════════════
# Four candidate treatments of the SAME numbers, so one can be picked and the rest
# deleted, plus the novelty companion that reads in the same grammar:
#
#     similarity_a_dumbbell   ranked rows; ECFP4 mean and median as a barbell
#     similarity_b_bars       ranked rows; two bar panels from zero
#     similarity_c_scatter    ECFP4 against scaffold match, one point each
#     similarity_d_table      the table itself, with a bar behind every number
#     similarity_novelty      training-set novelty barbell beside SNN stems
#
# WHAT THE NUMBERS ARE. Per pocket, every generated molecule is compared to that pocket's
# crystal ligand; the per-pocket mean is then macro-averaged over the 79 pockets that all
# nine methods share. ECFP4 is Morgan r=2 / 2048-bit Tanimoto, scaffold match the fraction
# of molecules whose Bemis-Murcko scaffold equals the reference's. These figures only DRAW
# reference_similarity.csv -- build_reference_similarity.py computes it -- so the figure
# cannot drift from the table.
#
# DIRECTION. Low is good here and an axis does not say so, so the subtitle does, once.
# DecompDiff at 0.152 / 2.17% is the one method visibly rediscovering the ligand it was
# shown; the other eight sit in a 0.089-0.118 band that is, for practical purposes, one
# band, and the figures are built not to manufacture a gap inside it.
#
# COLOUR. Three tiers, the same three the Vina charts use: ours is periwinkle #8291E8, the
# VoxBind runs it is built on are sand #F5B27E, and the published baselines wear the
# neutral grey the crystal reference wears there -- they are the backdrop the comparison is
# read against, not nine separate categories. The tiers are NOT taken from the shared
# COLORS map: nine rows of one colour each would be nine categories, which is the reading
# this family is built to avoid.
#
# LEGEND. Figure-level and above the panels in the bars/scatter/table variants. Nine ranked
# rows leave no in-panel corner that is reliably empty in all three metrics, and a key that
# lands on the data in one variant and not another is worse than one that is never in the
# way.

# Row labels. The noise level is a setting of one model, not part of its name, so it
# rides as a subscript exactly as the paper table writes it -- and, unlike the two-line
# form, it keeps every row one line tall so the rows stay evenly spaced.
# "Ours v1" is the key the CSV and every other table use; CoDE is what the paper calls
# the model, so the rename lives here, at the display edge, and no data key moves.
SIM_PRETTY = {"Ours v1": "CoDE",
              "VoxBind σ=0.9": r"VoxBind$_{\sigma=0.9}$",
              "VoxBind σ=1.0": r"VoxBind$_{\sigma=1.0}$"}
SIM_OURS = "Ours v1"
SIM_VOXBIND = ("VoxBind σ=0.9", "VoxBind σ=1.0")

# The order of results_drug_design.html, not the ranking: this figure sits beside that
# table and a reader moving between them should not have to re-find the rows. σ=1.0 is not
# in that table, so it follows σ=0.9; ours stays last, where the table puts it.
SIM_ORDER = ("AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind", "TargetDiff",
             "VoxBind σ=0.9", "VoxBind σ=1.0", "Ours v1")

SIM_OUR_COLOR, SIM_VOX_COLOR, SIM_BASE_COLOR = soft("CoDE"), color("VoxBind"), "#9aa0a6"
# The two VoxBind rows are one family at two noise levels, so they share a hue and
# separate by tint rather than by taking a third colour: sigma=0.9 is the run ours is
# built on and keeps the full-strength sand, sigma=1.0 the lighter one.
SIM_VOX_TINTS = {"VoxBind σ=1.0": pale("VoxBind")}
SIM_BASE_LABEL, SIM_VOX_LABEL, SIM_OUR_LABEL = "Published baselines", "VoxBind", "CoDE"

# Heavier than the Vina charts: this figure carries nine rows of two dots each, not a
# 79-point cloud, so the marks have to hold their own at slide size rather than stay out
# of each other's way. (The grid dash is the shared DOT, (0, (1, 2.6)).)
SIM_AXIS_LW, SIM_GRID_LW = 1.6, 1.3

SIM_ECFP_NAME = "ECFP4 Tanimoto to reference ligand"
SIM_SCAF_NAME = "Bemis-Murcko scaffold match"
SIM_LOWER_IS_NOVEL = "lower = less like the crystal ligand"
# "%g" so 0.005 prints as 0.5% and 0.01 as 1% -- a fixed 0 decimals turned every tick on
# the scaffold axis into "1%", "1%", "2%", and a fixed 1 gave "1.0%" where it adds nothing.
SIM_PCT = plt.FuncFormatter(lambda v, _: f"{100 * v:g}%")

_SIM_CACHE = {}


def _similarity_style():
    """The house style, plus the one rc this family needs on top of it.

    `mathtext.default` is "regular" because the sigma subscript in a row label is TEXT,
    not maths: same face, upright. It is handed back as a context manager rather than left
    in rcParams, because the shared DISPLAY puts VoxBind's sigma through mathtext too, and
    leaking "regular" would silently un-italicise every later figure in an --all run."""
    use_style()
    return plt.rc_context({"mathtext.default": "regular"})


def _similarity_num(text):
    """A CSV cell as a float, or None where the run had nothing to write."""
    return float(text) if text not in (None, "") else None


def _similarity_rows():
    """The table as rows, in the drug-design table's order. Cached for the life of the
    process -- five figures draw the same nine numbers."""
    if "rows" not in _SIM_CACHE:
        with open(legacy("fig-ref-ligand-similarity", "reference_similarity.csv"),
                  encoding="utf-8") as handle:
            rows = [{"method": r["method"],
                     "label": SIM_PRETTY.get(r["method"], r["method"]),
                     "mean": float(r["ecfp4_mean"]),
                     "median": float(r["ecfp4_median"]),
                     "scaffold": float(r["scaffold_match"]),
                     # Blank for the four methods sampled on the other box: they have no
                     # per-molecule SMILES here, so novelty is unmeasured, not zero.
                     "novelty": _similarity_num(r.get("novelty")),
                     "scaffold_novelty": _similarity_num(r.get("scaffold_novelty")),
                     "snn": _similarity_num(r.get("snn"))}
                    for r in csv.DictReader(handle)]
        rank = {m: i for i, m in enumerate(SIM_ORDER)}
        missing = [r["method"] for r in rows if r["method"] not in rank]
        if missing:                   # a new row in the CSV must be placed deliberately
            raise SystemExit(f"not in SIM_ORDER, add it: {', '.join(missing)}")
        rows = sorted(rows, key=lambda r: rank[r["method"]])
        _SIM_CACHE["rows"] = rows
        print(f"  === reference-ligand similarity, {len(rows)} methods, "
              f"79 shared pockets ===")
        print("    ECFP4 mean, low = most novel: "
              + " < ".join(f"{r['label']} {r['mean']:.3f}" for r in rows))
    return _SIM_CACHE["rows"]


def _similarity_colour(row):
    if row["method"] == SIM_OURS:
        return SIM_OUR_COLOR
    if row["method"] in SIM_VOXBIND:
        return SIM_VOX_TINTS.get(row["method"], SIM_VOX_COLOR)
    return SIM_BASE_COLOR


# ── shared parts ─────────────────────────────────────────────────────────────────
def _similarity_spines_and_ticks(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
        if sp in keep:
            ax.spines[sp].set_color(AXIS)
            ax.spines[sp].set_linewidth(SIM_AXIS_LW)
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=SIM_AXIS_LW, pad=4,
                   colors=AXIS)


def _similarity_value_axis(ax, name, *, pct=False, nbins=5):
    ax.set_xlabel(name, fontsize=15, labelpad=9)
    ax.grid(True, axis="x", color=GRID, lw=SIM_GRID_LW, ls=DOT)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins))
    if pct:
        ax.xaxis.set_major_formatter(SIM_PCT)


def _similarity_method_axis(ax, rows):
    """Method names down the y axis. They are the row identity, so no tick marks."""
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=14)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["method"] == SIM_OURS:      # the one row a reader is looking for
            tick.set_fontweight("bold")


def _similarity_tier_handles(marker="o", size=5.0):
    return [Line2D([], [], ls="none", marker=marker, markersize=size, color=c,
                   markeredgewidth=0, label=lab)
            for c, lab in ((SIM_BASE_COLOR, SIM_BASE_LABEL),
                           (SIM_VOX_COLOR, SIM_VOX_LABEL),
                           (SIM_OUR_COLOR, SIM_OUR_LABEL))]


def _similarity_top_key(fig, handles, *, y=0.995, fontsize=12.5):
    """One horizontal key along the top of the figure, clear of every panel."""
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=len(handles), frameon=True, fontsize=fontsize,
                     handlelength=1.4, handletextpad=0.4, columnspacing=1.5,
                     borderpad=0.4, facecolor="white", edgecolor=LEGEND_EDGE,
                     framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(SIM_AXIS_LW)
    for text in leg.get_texts():
        text.set_color(INK)
    for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if isinstance(handle, Line2D) and handle.get_markersize():
            handle.set_markersize(handle.get_markersize() * 2)
    return leg


def _similarity_panel_key(ax, handles, *, loc="upper right", fontsize=12.5, scale=1.0):
    """The same key as the Vina charts, inside a panel rather than above the figure.

    `scale` shrinks the whole key at once -- type, swatches and the ring weight together --
    so a panel whose data reaches into its corner can make room without the key losing its
    proportions or drifting from the one the other figures carry.
    """
    fontsize *= scale
    leg = ax.legend(handles=handles, loc=loc, ncol=1, frameon=True, fontsize=fontsize,
                    handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                    borderpad=0.4, borderaxespad=0.39, facecolor="white",
                    edgecolor=LEGEND_EDGE, framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(SIM_AXIS_LW)
    leg.set_zorder(6)
    for text in leg.get_texts():
        text.set_color(INK)
    for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if isinstance(handle, Line2D) and handle.get_markersize():
            handle.set_markersize(handle.get_markersize() * 2 * scale)
            handle.set_markeredgewidth(handle.get_markeredgewidth() * scale)
    return leg


def _similarity_header(fig, handles, note):
    """Key across the top, one grey line of direction under it, and the y the panels must
    stay below. Both are MEASURED rather than placed at a guessed fraction: the key's
    height depends on how many entries a variant has, and a fixed y put the subtitle
    through the middle of the legend box in the four-entry variants."""
    leg = _similarity_top_key(fig, handles)
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    under_key = to_fig.transform(leg.get_window_extent().p0)[1]
    text = fig.text(0.5, under_key - 0.018, note, ha="center", va="top", fontsize=12.5,
                    color=SIM_BASE_COLOR)
    fig.canvas.draw()
    return to_fig.transform(text.get_window_extent().p0)[1] - 0.022


def _similarity_fit(fig, *, rect=None, **kw):
    """tight_layout, then hand back what it under-reserves.

    Iterated, unlike the single pass the shared fit() makes: an x label is CENTRED on its
    axes, so pulling the axes in by d only moves the label by d/2 and one correction
    leaves half the overrun behind -- which is exactly how "Bemis-Murcko scaffold match"
    kept losing its last two characters off the right edge.
    """
    for _ in range(4):
        fig.tight_layout(rect=rect, **kw)
        fig.canvas.draw()
        w, h = fig.get_size_inches()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        sp, adj = fig.subplotpars, {}
        if bb.x0 < -1e-3:
            adj["left"] = sp.left + (-bb.x0) / w
        if bb.y0 < -1e-3:
            adj["bottom"] = sp.bottom + (-bb.y0) / h
        if bb.x1 > w + 1e-3:
            adj["right"] = sp.right - (bb.x1 - w) / w
        if bb.y1 > h + 1e-3:
            adj["top"] = sp.top - (bb.y1 - h) / h
        if not adj:
            return
        fig.subplots_adjust(**adj)
        rect = None                        # the correction owns the margins from here


def _similarity_save(fig, out, stem, note):
    save(fig, out, stem)
    print(f"    {stem}.{{png,svg,pdf}}   {note}")


def _similarity_place_labels(ax, items, *, fontsize):
    """Label every point without collisions, by trying offsets until one is clear.

    Nine labels around nine points, eight of them inside a 0.03-wide clump: fixed offsets
    put "VoxBind σ=0.9" straight through "Ours". Each label takes the first candidate
    offset whose rendered box misses every box already placed and every plotted point, so
    the crowded middle spreads itself and the isolated points keep the natural position.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = [ax.transData.transform(xy) for xy, _, _ in items]
    panel = ax.get_window_extent().expanded(0.985, 0.985)   # labels stay off the axes
    candidates = ((10, 0), (-10, 0), (0, 12), (0, -12), (9, 9), (-9, 9), (9, -9),
                  (-9, -9), (16, 5), (-16, 5), (16, -5), (-16, -5))
    taken = []
    for xy, text, c in items:
        for k, (dx, dy) in enumerate(candidates):
            # A hairline leader in the point's own colour: once a label has been pushed
            # off its natural side to dodge a neighbour, proximity alone stops saying
            # which dot it belongs to. It collapses to nothing where nothing moved.
            ann = ax.annotate(
                text, xy=xy, textcoords="offset points", xytext=(dx, dy),
                ha="left" if dx > 0 else "right" if dx < 0 else "center",
                va="center" if dy == 0 else "bottom" if dy > 0 else "top",
                fontsize=fontsize, color=INK, zorder=5,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.9, alpha=0.6,
                                shrinkA=1.5, shrinkB=6.0))
            fig.canvas.draw()
            box = ann.get_window_extent(renderer).expanded(1.08, 1.25)
            clear = (panel.containsx(box.x0) and panel.containsx(box.x1)
                     and panel.containsy(box.y0) and panel.containsy(box.y1)
                     and not any(box.overlaps(b) for b in taken)
                     and not any(box.contains(*p) for p in points))
            if clear or k == len(candidates) - 1:
                taken.append(box)
                break
            ann.remove()


# ── the variants ─────────────────────────────────────────────────────────────────
@figure("fig-similarity-dumbbell", needs=("reference_similarity.csv",))
def draw_similarity_dumbbell(out):
    """A: methods down the left in the drug-design table's order, value across.

    ECFP4 as a barbell -- filled dot the mean, hollow the median, the stem between them
    the gap. Scaffold match beside it as a stem from zero, because it is a single value
    per method and a barbell would imply a second one. Two panels sharing one method axis,
    so the row names are written once and a row is read straight across.

    Rows run top-down in table order rather than bottom-up, so the figure and
    results_drug_design.html list the methods in the same direction.
    """
    rows = _similarity_rows()
    with _similarity_style():
        # Sized so the PLOT AREA is 0.8x what the 8.67 in draft had, with the type
        # untouched. Shrinking the figure by 0.8 would not do that: the furniture outside
        # the axes -- row names, tick labels, the gutter -- is type, so it keeps its
        # physical size, and the plot area would absorb the whole cut and land near 0.73x.
        # Measured at 8.67 in the two panels were 4.115 + 2.352 = 6.467 in of plot area
        # against 2.203 in of furniture, so 8.67 - 0.2 * 6.467 leaves the furniture alone
        # and takes the fifth off the axes.
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.377, 4.05), dpi=220,
                                      gridspec_kw={"width_ratios": [1.75, 1]}, sharey=True)
        fig.patch.set_facecolor("white")
        for i, row in enumerate(rows):
            c = _similarity_colour(row)
            ax.plot([row["median"], row["mean"]], [i, i], color=c, lw=3.6, alpha=0.5,
                    solid_capstyle="round", zorder=2)
            ax.plot(row["median"], i, marker="o", markersize=7.0, color=c, mfc="white",
                    markeredgewidth=2.7, markeredgecolor=c, zorder=3)
            ax.plot(row["mean"], i, marker="o", markersize=10.2, color=c,
                    markeredgewidth=0, zorder=4)
            ax2.plot([0, row["scaffold"]], [i, i], color=c, lw=3.6, alpha=0.5,
                     solid_capstyle="round", zorder=2)
            ax2.plot(row["scaffold"], i, marker="o", markersize=10.2, color=c,
                     markeredgewidth=0, zorder=3)

        # The scaffold axis is labelled in bare whole percents -- 0 1 2 3 -- so the unit is
        # stated once in the axis name instead of five times along the ticks; at 0.5% steps
        # the row of "0.5% 1% 1.5%" was more punctuation than number.
        bare_pct = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}")
        for axis, name, step, fmt in (
                (ax, "ECFP4 Tanimoto similarity", 0.02, plt.FormatStrFormatter("%.2f")),
                (ax2, "Scaffold match (%)", 0.01, bare_pct)):
            _similarity_spines_and_ticks(axis)
            axis.set_xlabel(name, fontsize=15, labelpad=9)
            axis.grid(True, axis="x", color=GRID, lw=SIM_GRID_LW, ls=DOT)
            axis.grid(False, axis="y")
            axis.set_axisbelow(True)
            # A fixed step, not a bin count: at two decimals an auto-chosen 0.005 step
            # would print 0.09 twice, which is how an earlier draft ended up with
            # "1%, 1%, 2%".
            axis.xaxis.set_major_locator(MultipleLocator(step))
            axis.xaxis.set_major_formatter(fmt)

        _similarity_method_axis(ax, rows)
        ax.invert_yaxis()              # SIM_ORDER[0] at the top, as the table reads
        ax.set_xlim(0.079, 0.161)
        ax2.set_xlim(0, 0.0315)
        ax2.tick_params(axis="y", length=0)
        ax2.spines["left"].set_visible(False)

        # Only the convention the picture cannot state: filled is the mean, hollow the
        # median. The three colours are named where they stand, on the row labels.
        shape = [Line2D([], [], ls="none", marker="o", markersize=6.4, color=INK,
                        markeredgewidth=0, label="mean"),
                 Line2D([], [], ls="none", marker="o", markersize=5.2, color=INK,
                        mfc="white", markeredgewidth=2.2, markeredgecolor=INK,
                        label="median")]
        _similarity_panel_key(ax, shape, loc="lower right")
        _similarity_fit(fig, pad=0.5, w_pad=1.3)
        _similarity_save(fig, out, "similarity_a_dumbbell",
                         "methods down, metric across; ECFP4 barbell beside scaffold stems")


@figure("fig-similarity-bars", needs=("reference_similarity.csv",))
def draw_similarity_bars(out):
    """B: the table's own shape kept, encoded as length from zero -- the reading that
    needs no explanation and the one a sceptical reader will trust. The cost is honest and
    visible: from zero, eight of the nine bars are nearly the same length, because they
    nearly are."""
    rows = _similarity_rows()
    with _similarity_style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.05), dpi=220,
                                      gridspec_kw={"width_ratios": [1.75, 1]}, sharey=True)
        fig.patch.set_facecolor("white")
        for i, row in enumerate(rows):
            c = _similarity_colour(row)
            ax.barh(i, row["mean"], height=0.58, color=c, edgecolor="none", zorder=2)
            # The median is the same quantity as the bar, so it rides inside it as a notch
            # -- a second bar per row would read as a second method.
            ax.plot([row["median"]] * 2, [i - 0.29, i + 0.29], color="white", lw=2.0,
                    zorder=3, solid_capstyle="butt")
            ax2.barh(i, row["scaffold"], height=0.58, color=c, edgecolor="none", zorder=2)

        _similarity_spines_and_ticks(ax)
        _similarity_spines_and_ticks(ax2)
        _similarity_value_axis(ax, SIM_ECFP_NAME)
        _similarity_value_axis(ax2, SIM_SCAF_NAME, pct=True, nbins=4)
        _similarity_method_axis(ax, rows)
        ax2.tick_params(axis="y", length=0)
        ax2.spines["left"].set_visible(False)

        notch = Line2D([], [], color=INK, lw=2.0, label="median")
        top = _similarity_header(fig, _similarity_tier_handles(marker="s", size=4.8)
                                 + [notch], SIM_LOWER_IS_NOVEL)
        _similarity_fit(fig, pad=0.5, w_pad=1.3, rect=(0, 0, 1, top))
        _similarity_save(fig, out, "similarity_b_bars",
                         "ranked rows; bars from zero, median as an inset notch")


@figure("fig-similarity-scatter", needs=("reference_similarity.csv",))
def draw_similarity_scatter(out):
    """C: the two metrics against each other. They are not independent -- a method that
    reproduces the reference's atoms tends to reproduce its scaffold -- and this is the
    only treatment that shows the correlation, and that DecompDiff sits alone off the end
    of it while everything else is one cluster."""
    rows = _similarity_rows()
    with _similarity_style():
        fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=220)
        fig.patch.set_facecolor("white")
        for row in rows:
            big = row["method"] == SIM_OURS
            ax.plot(row["mean"], row["scaffold"], marker="o",
                    markersize=12.0 if big else 8.8, color=_similarity_colour(row),
                    markeredgewidth=0, zorder=4 if big else 3)

        _similarity_spines_and_ticks(ax)
        _similarity_value_axis(ax, f"{SIM_ECFP_NAME}  (mean)")
        ax.set_ylabel(SIM_SCAF_NAME, fontsize=15, labelpad=10)
        ax.grid(True, axis="y", color=GRID, lw=SIM_GRID_LW, ls=DOT)
        ax.yaxis.set_major_formatter(SIM_PCT)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlim(0.082, 0.163)
        ax.set_ylim(0.0, 0.0245)
        top = _similarity_header(fig, _similarity_tier_handles(size=4.6),
                                 "lower-left = less like the crystal ligand on both axes")
        _similarity_fit(fig, pad=0.5, rect=(0, 0, 1, top))
        _similarity_place_labels(
            ax, [((r["mean"], r["scaffold"]), r["label"], _similarity_colour(r))
                 for r in rows], fontsize=12.5)
        _similarity_save(fig, out, "similarity_c_scatter",
                         "ECFP4 against scaffold match, one point per method")


@figure("fig-similarity-table", needs=("reference_similarity.csv",))
def draw_similarity_table(out):
    """D: the table, still a table, with a bar behind each number.

    For the reader who wants the digits -- a reviewer checking a claim, a slide that has
    to survive being quoted -- and the ranking at a glance. Bars are scaled per column to
    that column's largest value, which is a WITHIN-column comparison only: the three
    columns are three different quantities and their bar lengths are not comparable
    across, which is why each column carries its own number.
    """
    rows = _similarity_rows()
    with _similarity_style():
        cols = [("ECFP4 avg.", "mean", lambda v: f"{v:.3f}"),
                ("ECFP4 med.", "median", lambda v: f"{v:.3f}"),
                (SIM_SCAF_NAME, "scaffold", lambda v: f"{100 * v:.2f}%")]
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.2), dpi=220, sharey=True)
        fig.patch.set_facecolor("white")

        for ax, (name, field, fmt) in zip(axes, cols):
            top = max(r[field] for r in rows)
            for i, row in enumerate(rows):
                c = _similarity_colour(row)
                if row["method"] == SIM_OURS:   # a quiet band, so the row is findable
                    ax.add_patch(plt.Rectangle((0, i - 0.46), 1, 0.92,
                                               transform=ax.get_yaxis_transform(),
                                               facecolor=SIM_OUR_COLOR, alpha=0.09,
                                               edgecolor="none", zorder=0))
                ax.barh(i, row[field], height=0.42, color=c, edgecolor="none", zorder=2)
                ax.text(row[field] + 0.035 * top, i, fmt(row[field]), va="center",
                        ha="left", fontsize=12.5, color=INK, zorder=3,
                        fontweight="bold" if row["method"] == SIM_OURS else "normal")
            ax.set_xlim(0, top * 1.42)     # room for the number at the end of the bar
            ax.set_xlabel(name, fontsize=13.5, labelpad=8)
            ax.set_xticks([])
            ax.grid(False)
            _similarity_spines_and_ticks(ax, keep=())
            ax.tick_params(axis="y", length=0)   # sharey leaves stubs on columns 2 and 3

        _similarity_method_axis(axes[0], rows)
        axes[0].tick_params(axis="y", length=0)
        top = _similarity_header(
            fig, _similarity_tier_handles(marker="s", size=4.8),
            f"{SIM_LOWER_IS_NOVEL} · bars scale within a column, not across")
        _similarity_fit(fig, pad=0.5, w_pad=0.8, rect=(0, 0, 1, top))
        _similarity_save(fig, out, "similarity_d_table",
                         "the table, with a bar behind every number")


@figure("fig-similarity-novelty", needs=("reference_similarity.csv",))
def draw_similarity_novelty(out):
    """The novelty companion, in the same grammar as the reference-similarity figure.

    Same rows in the same order and the same geometry, so the two can sit one above the
    other and a method stays on its own line across both. Left is a barbell again, but
    between two LEVELS of the same question rather than two statistics: the filled dot is
    the share of generated molecules whose SMILES never appeared in training, the hollow
    one the share whose Bemis-Murcko scaffold never did. Scaffold is always the harder
    test, so it always sits left, and the stem is the gap between "a new molecule" and
    "a new skeleton". Right is SNN, one value per method, as a stem from zero.

    DIRECTION IS NOT SHARED between the panels: on the left more is newer, on the right
    LESS is -- SNN is how close the nearest training molecule got. That reverse is the one
    thing a reader can get backwards from the marks alone, so each axis name carries the
    arrow for its own good direction. A method whose CSV row has no novelty (none, since
    the four remote baselines were re-measured here on 2026-09-09) draws an italic note
    rather than a zero, which would read as "memorised everything".

    The novelty axis starts at 50, not at zero: the rates live in a 70-99 band and a
    zero-based axis spent half its width on empty space. It is cropped only to half, and
    to a round number, so the truncation is legible rather than a subtle exaggeration --
    and the panel is narrowed to match, so half the range does not occupy the width the
    full one did. SNN, a similarity, keeps its true zero.
    """
    rows = _similarity_rows()
    with _similarity_style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.42, 4.05), dpi=220,
                                      gridspec_kw={"width_ratios": [1.22, 1]}, sharey=True)
        fig.patch.set_facecolor("white")
        for i, row in enumerate(rows):
            c = _similarity_colour(row)
            if row["novelty"] is None:
                for axis, at in ((ax, 0.5), (ax2, 0.5)):
                    axis.text(at, i, "not measured here",
                              transform=axis.get_yaxis_transform(), ha="center",
                              va="center", fontsize=11, color=SIM_BASE_COLOR,
                              style="italic")
                continue
            lo, hi = row["scaffold_novelty"], row["novelty"]
            ax.plot([lo, hi], [i, i], color=c, lw=3.6, alpha=0.5, solid_capstyle="round",
                    zorder=2)
            ax.plot(lo, i, marker="o", markersize=7.0, color=c, mfc="white",
                    markeredgewidth=2.7, markeredgecolor=c, zorder=3)
            ax.plot(hi, i, marker="o", markersize=10.2, color=c, markeredgewidth=0,
                    zorder=4)
            ax2.plot([0, row["snn"]], [i, i], color=c, lw=3.6, alpha=0.5,
                     solid_capstyle="round", zorder=2)
            ax2.plot(row["snn"], i, marker="o", markersize=10.2, color=c,
                     markeredgewidth=0, zorder=3)

        bare_pct = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}")
        for axis, name, step, fmt in (
                # the arrow in the name is the direction of BETTER, the usual convention
                # Both names lead with the shared qualifier so the parallel is visible, and
                # both now fit their own panel: at this width the old "Novelty vs. training
                # set (%)" was 3.09 in against a 2.53 in panel and ran into its neighbour.
                (ax, "Training-set novelty (%) ↑", 0.1, bare_pct),
                (ax2, "Training-set SNN ↓", 0.1, plt.FormatStrFormatter("%.1f"))):
            _similarity_spines_and_ticks(axis)
            axis.set_xlabel(name, fontsize=15, labelpad=9)
            axis.grid(True, axis="x", color=GRID, lw=SIM_GRID_LW, ls=DOT)
            axis.grid(False, axis="y")
            axis.set_axisbelow(True)
            axis.xaxis.set_major_locator(MultipleLocator(step))
            axis.xaxis.set_major_formatter(fmt)

        _similarity_method_axis(ax, rows)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.02)
        ax2.set_xlim(0, 0.41)
        ax2.tick_params(axis="y", length=0)
        ax2.spines["left"].set_visible(False)

        shape = [Line2D([], [], ls="none", marker="o", markersize=6.4, color=INK,
                        markeredgewidth=0, label="molecule"),
                 Line2D([], [], ls="none", marker="o", markersize=5.2, color=INK,
                        mfc="white", markeredgewidth=2.2, markeredgecolor=INK,
                        label="scaffold")]
        # 80%: at full size the key reached the sigma=1.0 scaffold dot at 78%.
        _similarity_panel_key(ax, shape, loc="lower left", scale=0.8)
        _similarity_fit(fig, pad=0.5, w_pad=1.3)
        _similarity_save(fig, out, "similarity_novelty",
                         "novelty barbell (molecule vs scaffold) beside SNN stems")
