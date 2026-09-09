#!/usr/bin/env python3
"""build_similarity_figure.py — the reference-ligand similarity table as a figure, in the
visual language of the Vina 3-line charts next to it.

Four candidate treatments of the SAME numbers, so one can be picked and the rest deleted:

    similarity_a_dumbbell.{png,svg,pdf}   ranked rows; ECFP4 mean and median as a barbell
    similarity_b_bars.{png,svg,pdf}       ranked rows; two bar panels from zero
    similarity_c_scatter.{png,svg,pdf}    ECFP4 against scaffold match, one point each
    similarity_d_table.{png,svg,pdf}      the table itself, with a bar behind every number

WHAT THE NUMBERS ARE. Per pocket, every generated molecule is compared to that pocket's
crystal ligand; the per-pocket mean is then macro-averaged over the 79 pockets that all
nine methods share. ECFP4 is Morgan r=2 / 2048-bit Tanimoto, scaffold match the fraction
of molecules whose Bemis-Murcko scaffold equals the reference's. This script only DRAWS
reference_similarity.csv -- build_reference_similarity.py computes it -- so the figure
cannot drift from the table.

DIRECTION. Low is good here and an axis does not say so, so the subtitle does, once.
DecompDiff at 0.152 / 2.17% is the one method visibly rediscovering the ligand it was
shown; the other eight sit in a 0.089-0.118 band that is, for practical purposes, one
band, and the figures are built not to manufacture a gap inside it.

COLOUR. Three tiers, the same three the Vina charts use: ours is periwinkle #8291E8, the
VoxBind runs it is built on are sand #F5B27E, and the published baselines wear the neutral
grey the crystal reference wears there -- they are the backdrop the comparison is read
against, not nine separate categories.

LEGEND. Figure-level and above the panels in every variant. Nine ranked rows leave no
in-panel corner that is reliably empty in all three metrics, and a key that lands on the
data in one variant and not another is worse than one that is simply never in the way.

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_similarity_figure.py
"""
import csv
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import (FormatStrFormatter, FuncFormatter,
                               MaxNLocator, MultipleLocator)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "reference_similarity.csv")
OUT = os.path.join(os.path.dirname(HERE), "260910", "fig-ref-ligand-similarity")

# Row labels run along the y axis, where a long name costs width and nothing else, so
# they stay on one line; only "Ours v1" is shortened, to the name the deck uses.
PRETTY = {"Ours v1": "Ours"}
OURS = "Ours v1"
VOXBIND = ("VoxBind σ=0.9", "VoxBind σ=1.0")

# The order of results_drug_design.html, not the ranking: this figure sits beside that
# table and a reader moving between them should not have to re-find the rows. σ=1.0 is not
# in that table, so it follows σ=0.9; ours stays last, where the table puts it.
ORDER = ("AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind", "TargetDiff",
         "VoxBind σ=0.9", "VoxBind σ=1.0", "Ours v1")

INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
OUR_COLOR, VOX_COLOR, BASE_COLOR = "#8291E8", "#F5B27E", "#9aa0a6"
BASE_LABEL, VOX_LABEL, OUR_LABEL = "Published baselines", "VoxBind", "Ours"

# Heavier than the Vina charts: this figure carries nine rows of two dots each, not a
# 79-point cloud, so the marks have to hold their own at slide size rather than stay out
# of each other's way.
AXIS_LW, GRID_LW = 1.6, 1.3
GRID_DASH = (0, (1, 2.6))

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

ECFP_NAME = "ECFP4 Tanimoto to reference ligand"
SCAF_NAME = "Bemis-Murcko scaffold match"
LOWER_IS_NOVEL = "lower = less like the crystal ligand"
# "%g" so 0.005 prints as 0.5% and 0.01 as 1% -- a fixed 0 decimals turned every tick on
# the scaffold axis into "1%", "1%", "2%", and a fixed 1 gave "1.0%" where it adds nothing.
PCT = FuncFormatter(lambda v, _: f"{100 * v:g}%")


def load():
    """The table as rows, ordered by ECFP4 mean (lowest = most novel first)."""
    with open(CSV, encoding="utf-8") as handle:
        rows = [{"method": r["method"],
                 "label": PRETTY.get(r["method"], r["method"]),
                 "mean": float(r["ecfp4_mean"]),
                 "median": float(r["ecfp4_median"]),
                 "scaffold": float(r["scaffold_match"])}
                for r in csv.DictReader(handle)]
    rank = {m: i for i, m in enumerate(ORDER)}
    missing = [r["method"] for r in rows if r["method"] not in rank]
    if missing:                       # a new row in the CSV must be placed deliberately
        raise SystemExit(f"not in ORDER, add it: {', '.join(missing)}")
    return sorted(rows, key=lambda r: rank[r["method"]])


def colour(row):
    if row["method"] == OURS:
        return OUR_COLOR
    return VOX_COLOR if row["method"] in VOXBIND else BASE_COLOR


# ------------------------------------------------------------------------ shared parts

def spines_and_ticks(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
        if sp in keep:
            ax.spines[sp].set_color(AXIS)
            ax.spines[sp].set_linewidth(AXIS_LW)
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)


def value_axis(ax, name, *, pct=False, nbins=5):
    ax.set_xlabel(name, fontsize=15, labelpad=9)
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=GRID_DASH)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins))
    if pct:
        ax.xaxis.set_major_formatter(PCT)


def method_axis(ax, rows):
    """Method names down the y axis. They are the row identity, so no tick marks."""
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=14)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["method"] == OURS:          # the one row a reader is looking for
            tick.set_fontweight("bold")


def tier_handles(marker="o", size=5.0):
    return [Line2D([], [], ls="none", marker=marker, markersize=size, color=c,
                   markeredgewidth=0, label=lab)
            for c, lab in ((BASE_COLOR, BASE_LABEL), (VOX_COLOR, VOX_LABEL),
                           (OUR_COLOR, OUR_LABEL))]


def top_key(fig, handles, *, y=0.995, fontsize=12.5):
    """One horizontal key along the top of the figure, clear of every panel."""
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=len(handles), frameon=True, fontsize=fontsize,
                     handlelength=1.4, handletextpad=0.4, columnspacing=1.5,
                     borderpad=0.4, facecolor="white", edgecolor=LEGEND_EDGE,
                     framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    for text in leg.get_texts():
        text.set_color(INK)
    for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if isinstance(handle, Line2D) and handle.get_markersize():
            handle.set_markersize(handle.get_markersize() * 2)
    return leg


def panel_key(ax, handles, *, loc="upper right", fontsize=12.5):
    """The same key as the Vina charts, inside a panel rather than above the figure."""
    leg = ax.legend(handles=handles, loc=loc, ncol=1, frameon=True, fontsize=fontsize,
                    handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                    borderpad=0.4, borderaxespad=0.39, facecolor="white",
                    edgecolor=LEGEND_EDGE, framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(6)
    for text in leg.get_texts():
        text.set_color(INK)
    for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if isinstance(handle, Line2D) and handle.get_markersize():
            handle.set_markersize(handle.get_markersize() * 2)
    return leg


def header(fig, handles, note):
    """Key across the top, one grey line of direction under it, and the y the panels must
    stay below. Both are MEASURED rather than placed at a guessed fraction: the key's
    height depends on how many entries a variant has, and a fixed y put the subtitle
    through the middle of the legend box in the four-entry variants."""
    leg = top_key(fig, handles)
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    under_key = to_fig.transform(leg.get_window_extent().p0)[1]
    text = fig.text(0.5, under_key - 0.018, note, ha="center", va="top", fontsize=12.5,
                    color=BASE_COLOR)
    fig.canvas.draw()
    return to_fig.transform(text.get_window_extent().p0)[1] - 0.022


def fit(fig, *, rect=None, **kw):
    """tight_layout, then hand back what it under-reserves.

    Iterated, unlike the single pass the Vina script needs: an x label is CENTRED on its
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


def save(fig, stem, note):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), facecolor="white")
    plt.close(fig)
    print(f"  {stem}.{{png,svg,pdf}}   {note}")


# ---------------------------------------------------------------------------- variants

def variant_dumbbell(rows):
    """A: methods down the left in the drug-design table's order, value across.

    ECFP4 as a barbell -- filled dot the mean, hollow the median, the stem between them
    the gap. Scaffold match beside it as a stem from zero, because it is a single value
    per method and a barbell would imply a second one. Two panels sharing one method axis,
    so the row names are written once and a row is read straight across.

    Rows run top-down in table order rather than bottom-up, so the figure and
    results_drug_design.html list the methods in the same direction.
    """
    # 0.85x the 10.2 in draft: nine rows do not need a full text width, and the tighter
    # box sits the two panels closer to the row names they belong to.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.67, 4.05), dpi=220,
                                  gridspec_kw={"width_ratios": [1.75, 1]}, sharey=True)
    fig.patch.set_facecolor("white")
    for i, row in enumerate(rows):
        c = colour(row)
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
    bare_pct = FuncFormatter(lambda v, _: f"{100 * v:.0f}")
    for axis, name, step, fmt in (
            (ax, "ECFP4 Tanimoto similarity", 0.02, FormatStrFormatter("%.2f")),
            (ax2, "Scaffold match (%)", 0.01, bare_pct)):
        spines_and_ticks(axis)
        axis.set_xlabel(name, fontsize=15, labelpad=9)
        axis.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=GRID_DASH)
        axis.grid(False, axis="y")
        axis.set_axisbelow(True)
        # A fixed step, not a bin count: at two decimals an auto-chosen 0.005 step would
        # print 0.09 twice, which is how an earlier draft ended up with "1%, 1%, 2%".
        axis.xaxis.set_major_locator(MultipleLocator(step))
        axis.xaxis.set_major_formatter(fmt)

    method_axis(ax, rows)
    ax.invert_yaxis()                  # ORDER[0] at the top, as the table reads
    ax.set_xlim(0.079, 0.161)
    ax2.set_xlim(0, 0.0315)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)

    # Only the convention the picture cannot state: filled is the mean, hollow the median.
    # The three colours are named where they stand, on the row labels.
    shape = [Line2D([], [], ls="none", marker="o", markersize=6.4, color=INK,
                    markeredgewidth=0, label="mean"),
             Line2D([], [], ls="none", marker="o", markersize=5.2, color=INK, mfc="white",
                    markeredgewidth=2.2, markeredgecolor=INK, label="median")]
    panel_key(ax, shape, loc="lower right")
    fit(fig, pad=0.5, w_pad=1.3)
    save(fig, "similarity_a_dumbbell",
         "methods down, metric across; ECFP4 barbell beside scaffold stems")


def variant_bars(rows):
    """B: the table's own shape kept, encoded as length from zero -- the reading that
    needs no explanation and the one a sceptical reader will trust. The cost is honest and
    visible: from zero, eight of the nine bars are nearly the same length, because they
    nearly are."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.05), dpi=220,
                                  gridspec_kw={"width_ratios": [1.75, 1]}, sharey=True)
    fig.patch.set_facecolor("white")
    for i, row in enumerate(rows):
        c = colour(row)
        ax.barh(i, row["mean"], height=0.58, color=c, edgecolor="none", zorder=2)
        # The median is the same quantity as the bar, so it rides inside it as a notch --
        # a second bar per row would read as a second method.
        ax.plot([row["median"]] * 2, [i - 0.29, i + 0.29], color="white", lw=2.0,
                zorder=3, solid_capstyle="butt")
        ax2.barh(i, row["scaffold"], height=0.58, color=c, edgecolor="none", zorder=2)

    spines_and_ticks(ax)
    spines_and_ticks(ax2)
    value_axis(ax, ECFP_NAME)
    value_axis(ax2, SCAF_NAME, pct=True, nbins=4)
    method_axis(ax, rows)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)

    notch = Line2D([], [], color=INK, lw=2.0, label="median")
    top = header(fig, tier_handles(marker="s", size=4.8) + [notch], LOWER_IS_NOVEL)
    fit(fig, pad=0.5, w_pad=1.3, rect=(0, 0, 1, top))
    save(fig, "similarity_b_bars", "ranked rows; bars from zero, median as an inset notch")


def place_labels(ax, items, *, fontsize):
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


def variant_scatter(rows):
    """C: the two metrics against each other. They are not independent -- a method that
    reproduces the reference's atoms tends to reproduce its scaffold -- and this is the
    only treatment that shows the correlation, and that DecompDiff sits alone off the end
    of it while everything else is one cluster."""
    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=220)
    fig.patch.set_facecolor("white")
    for row in rows:
        big = row["method"] == OURS
        ax.plot(row["mean"], row["scaffold"], marker="o",
                markersize=12.0 if big else 8.8, color=colour(row),
                markeredgewidth=0, zorder=4 if big else 3)

    spines_and_ticks(ax)
    value_axis(ax, f"{ECFP_NAME}  (mean)")
    ax.set_ylabel(SCAF_NAME, fontsize=15, labelpad=10)
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=GRID_DASH)
    ax.yaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlim(0.082, 0.163)
    ax.set_ylim(0.0, 0.0245)
    top = header(fig, tier_handles(size=4.6),
                 "lower-left = less like the crystal ligand on both axes")
    fit(fig, pad=0.5, rect=(0, 0, 1, top))
    place_labels(ax, [((r["mean"], r["scaffold"]), r["label"], colour(r)) for r in rows],
                 fontsize=12.5)
    save(fig, "similarity_c_scatter", "ECFP4 against scaffold match, one point per method")


def variant_table(rows):
    """D: the table, still a table, with a bar behind each number.

    For the reader who wants the digits -- a reviewer checking a claim, a slide that has
    to survive being quoted -- and the ranking at a glance. Bars are scaled per column to
    that column's largest value, which is a WITHIN-column comparison only: the three
    columns are three different quantities and their bar lengths are not comparable
    across, which is why each column carries its own number.
    """
    cols = [("ECFP4 avg.", "mean", lambda v: f"{v:.3f}"),
            ("ECFP4 med.", "median", lambda v: f"{v:.3f}"),
            (SCAF_NAME, "scaffold", lambda v: f"{100 * v:.2f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.2), dpi=220, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (name, field, fmt) in zip(axes, cols):
        top = max(r[field] for r in rows)
        for i, row in enumerate(rows):
            c = colour(row)
            if row["method"] == OURS:      # a quiet band, so the row is findable
                ax.add_patch(Rectangle((0, i - 0.46), 1, 0.92, transform=ax.get_yaxis_transform(),
                                       facecolor=OUR_COLOR, alpha=0.09, edgecolor="none",
                                       zorder=0))
            ax.barh(i, row[field], height=0.42, color=c, edgecolor="none", zorder=2)
            ax.text(row[field] + 0.035 * top, i, fmt(row[field]), va="center", ha="left",
                    fontsize=12.5, color=INK, zorder=3,
                    fontweight="bold" if row["method"] == OURS else "normal")
        ax.set_xlim(0, top * 1.42)         # room for the number at the end of the bar
        ax.set_xlabel(name, fontsize=13.5, labelpad=8)
        ax.set_xticks([])
        ax.grid(False)
        spines_and_ticks(ax, keep=())
        ax.tick_params(axis="y", length=0)   # sharey leaves stubs on columns 2 and 3

    method_axis(axes[0], rows)
    axes[0].tick_params(axis="y", length=0)
    top = header(fig, tier_handles(marker="s", size=4.8),
                 f"{LOWER_IS_NOVEL} · bars scale within a column, not across")
    fit(fig, pad=0.5, w_pad=0.8, rect=(0, 0, 1, top))
    save(fig, "similarity_d_table", "the table, with a bar behind every number")


def main():
    rows = load()
    plt.rcParams.update(RC)
    print(f"=== reference-ligand similarity, {len(rows)} methods, 79 shared pockets ===")
    print("  ECFP4 mean, low = most novel: "
          + " < ".join(f"{r['label']} {r['mean']:.3f}" for r in rows))
    variant_dumbbell(rows)
    if "--all" in sys.argv:      # the three treatments that were not chosen
        variant_bars(rows)
        variant_scatter(rows)
        variant_table(rows)


if __name__ == "__main__":
    main()
