#!/usr/bin/env python3
"""draw.py — every 260910 figure, one function each, from one command.

    python figures/draw.py --list
    python figures/draw.py fig-vina-per-pocket-dock fig-posebusters-check-failures
    python figures/draw.py --all
    python figures/draw.py --all --formats png          # skip the vector pair

Each figure is ONE function registered under the id that also names its output folder:

    figures/260910/fig-{eval}-{figname}/{stem}.{png,svg,pdf}

`{eval}` is the evaluation the figure reads (vina, posebusters, posecheck, consistency,
interaction, molweight, similarity, mcp, ensemble) and `{figname}` is the figure within it.
One id may write several files -- the `core`/`all` variants of a panel, or mean and median
of one statistic, are the same figure drawn twice and belong together in one folder.

WHY ONE FILE. The drawing code used to live in ten `build_*.py` scripts under
notebook/html/260910/, each with its own copy of the palette, the loaders and the axis
furniture, and each rebuilding its own data on import. Everything shared now sits once at
the top of this file, the per-figure code is the function body and nothing else, and the
expensive per-molecule load is done once per RUN no matter how many figures ask for it.

WHAT THIS FILE DOES NOT DO. It draws; it does not re-score. The per-molecule inputs
(`metrics.json` under each run, the docking result JSONs, the PoseCheck/PoseBusters
exports) are produced by voxbind/scripts/73_evaluate_samples.sh and the staging scripts,
and the original builders under notebook/html/260910/ still hold the recompute paths that
made the exported CSV/JSON artifacts. Those scripts remain the source of truth for DATA;
this file is the source of truth for PICTURES.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402
from matplotlib.patches import Patch                                  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator            # noqa: E402

HERE = Path(__file__).resolve().parent                 # VoxBind/figures
REPO = HERE.parent                                     # VoxBind
# Everything below is resolved FROM those two, never written out absolutely: this tree gets
# copied between boxes, and a "/home1/irteam/..." baked into a constant is the thing that
# breaks first. The two sibling checkouts this needs -- base_drug (TargetDiff's own eval
# tree) and funcbind -- sit NEXT TO the repo, so they are REPO.parent-relative; both take
# an environment override for a box that arranges them differently.
SIBLING = Path(os.environ.get("VOXBIND_SIBLING_ROOT", REPO.parent))
BASEDRUG = Path(os.environ.get("VOXBIND_BASEDRUG", SIBLING / "base_drug"))
FUNCBIND = Path(os.environ.get("VOXBIND_FUNCBIND", SIBLING / "funcbind"))
OUT_ROOT = HERE                                        # figures/fig-<eval>/

# ════════════════════════════════════════════════════════════════════════════════
# the layout — one folder per EVALUATION, named by it, files named by the figure
# ════════════════════════════════════════════════════════════════════════════════
# A figure id is `fig-<eval>-<name>` and selects one figure on the command line; the
# folder it lands in is its EVALUATION, so everything that answers the same question sits
# together and the folder name carries the part of the identity every file in it shares.
EVAL_FOLDER = {
    "vina":        "fig-vina",
    "posebusters": "fig-posebusters",
    "posecheck":   "fig-posecheck",
    "consistency": "fig-rigid-consistency",
    # The interaction fingerprint IS a PoseCheck output -- ProLIF, run by PoseCheck over
    # the same pocket10 crop and read out of the same `posecheck` block of the same
    # metrics.json that strain and clashes come from (compute_baseline_interactions.py
    # imports PoseCheck directly to produce it for the baselines, which carry no posecheck
    # block of their own). So it shares PoseCheck's folder rather than taking one of its
    # own; its files keep an `interaction-` prefix to stay findable among the strain and
    # clash ones. `draw.py interaction` still selects exactly these six.
    "interaction": "fig-posecheck",
    "molweight":   "fig-molweight",
    "similarity":  "fig-similarity",
    "mcp":         "fig-mcp",
    "ensemble":    "fig-ensemble",
}

# WHAT A FILE IS CALLED, once its folder already says which evaluation it belongs to.
# The part files still call save() with the name the original builder used, and the rename
# happens here, in one table, at the writing edge. So this table is also the map between
# this tree and notebook/html/260910/ -- which is how the port was verified figure by
# figure -- and nothing in a family's code has to know about it.
#
# The rule is: drop whatever the folder already says (`rigid_`, `pb_`, `mw_`, `vina_`,
# `interaction_`, `similarity_`, `mcp_`), keep what distinguishes the figure -- including
# the metric, since `strain` and `clash` are not the evaluation's name -- and use hyphens.
STEMS = {
    # fig-vina
    "vina_dock_per_pocket_mean_core": "dock-per-pocket-mean-core",
    "vina_dock_per_pocket_mean_all": "dock-per-pocket-mean-all",
    "vina_dock_per_pocket_median_core": "dock-per-pocket-median-core",
    "vina_dock_per_pocket_median_all": "dock-per-pocket-median-all",
    "vina_dock_per_pocket_pair_core": "dock-per-pocket-pair-core",
    "vina_dock_per_pocket_pair_all": "dock-per-pocket-pair-all",
    "vina_score_per_pocket_mean_core": "score-per-pocket-mean-core",
    "vina_score_per_pocket_mean_all": "score-per-pocket-mean-all",
    "vina_score_per_pocket_median_core": "score-per-pocket-median-core",
    "vina_score_per_pocket_median_all": "score-per-pocket-median-all",
    "vina_score_per_pocket_pair_core": "score-per-pocket-pair-core",
    "vina_score_per_pocket_pair_all": "score-per-pocket-pair-all",
    "vina_min_per_pocket_mean_core": "min-per-pocket-mean-core",
    "vina_min_per_pocket_mean_all": "min-per-pocket-mean-all",
    "vina_min_per_pocket_median_core": "min-per-pocket-median-core",
    "vina_min_per_pocket_median_all": "min-per-pocket-median-all",
    "vina_min_per_pocket_pair_core": "min-per-pocket-pair-core",
    "vina_min_per_pocket_pair_all": "min-per-pocket-pair-all",
    "vina_dock_per_atom": "dock-per-atom",
    "vina_dock_per_atom_v1_mean_core": "dock-per-atom-v1-mean-core",
    "vina_dock_per_atom_v1_mean_all": "dock-per-atom-v1-mean-all",
    "vina_dock_per_atom_v1_median_core": "dock-per-atom-v1-median-core",
    "vina_dock_per_atom_v1_median_all": "dock-per-atom-v1-median-all",
    "vina_dock_per_atom_v2_mean_core": "dock-per-atom-v2-mean-core",
    "vina_dock_per_atom_v2_mean_all": "dock-per-atom-v2-mean-all",
    "vina_dock_per_atom_v2_median_core": "dock-per-atom-v2-median-core",
    "vina_dock_per_atom_v2_median_all": "dock-per-atom-v2-median-all",
    "vina_dock_per_atom_v3_mean_core": "dock-per-atom-v3-mean-core",
    "vina_dock_per_atom_v3_mean_all": "dock-per-atom-v3-mean-all",
    "vina_dock_per_atom_v3_median_core": "dock-per-atom-v3-median-core",
    "vina_dock_per_atom_v3_median_all": "dock-per-atom-v3-median-all",
    # fig-posebusters — `sucos` stays: it is a second metric, not the evaluation's name
    "pb_check_failures_all": "check-failures-all",
    "pb_check_failures_core": "check-failures-core",
    "pb_valid_per_atom_all": "valid-per-atom-all",
    "pb_valid_per_atom_core": "valid-per-atom-core",
    "posebusters_by_atom_range": "by-atom-range",
    "sucos_ecdf_all": "sucos-ecdf-all",
    "sucos_ecdf_core": "sucos-ecdf-core",
    "sucos_per_atom_mean_all": "sucos-per-atom-mean-all",
    "sucos_per_atom_mean_core": "sucos-per-atom-mean-core",
    "sucos_per_atom_median_all": "sucos-per-atom-median-all",
    "sucos_per_atom_median_core": "sucos-per-atom-median-core",
    # fig-posecheck
    "strain_per_atom_mean_all": "per-atom-mean-all",
    "strain_per_atom_mean_core": "per-atom-mean-core",
    "strain_per_atom_median_all": "per-atom-median-all",
    "strain_per_atom_median_core": "per-atom-median-core",
    "clash_per_atom_mean_all": "clash-per-atom-mean-all",
    "clash_per_atom_mean_core": "clash-per-atom-mean-core",
    "clash_per_atom_median_all": "clash-per-atom-median-all",
    "clash_per_atom_median_core": "clash-per-atom-median-core",
    "posecheck_per_atom": "per-atom",
    # the by-size set: the bin is the variant, so it reads as one family in a listing
    "strain_ecdf_le15": "ecdf-by-size-le15",
    "strain_ecdf_16_20": "ecdf-by-size-16-20",
    "strain_ecdf_21_25": "ecdf-by-size-21-25",
    "strain_ecdf_26_30": "ecdf-by-size-26-30",
    "strain_ecdf_gt30": "ecdf-by-size-gt30",
    "strain_ecdf_all": "ecdf-by-size-all",
    "clash_violin_le15": "clash-violin-by-size-le15",
    "clash_violin_16_20": "clash-violin-by-size-16-20",
    "clash_violin_21_25": "clash-violin-by-size-21-25",
    "clash_violin_26_30": "clash-violin-by-size-26-30",
    "clash_violin_gt30": "clash-violin-by-size-gt30",
    "clash_violin_all": "clash-violin-by-size-all",
    "posecheck_all_by_atom_range": "all-by-atom-range",
    "strain_per_rotbond_mean_all": "per-rotbond-mean-all",
    "strain_per_rotbond_mean_core": "per-rotbond-mean-core",
    "strain_per_rotbond_median_all": "per-rotbond-median-all",
    "strain_per_rotbond_median_core": "per-rotbond-median-core",
    "strain_box_per_rotbond_all": "per-rotbond-box-all",
    "strain_box_per_rotbond_core": "per-rotbond-box-core",
    "strain_per_rotbond": "per-rotbond",
    "strain_rotbond_all_methods_mean": "per-rotbond-all-methods-mean",
    "strain_rotbond_all_methods_median": "per-rotbond-all-methods-median",
    "strain_box_per_rotbond_all_methods": "per-rotbond-box-all-methods",
    "strain_box_per_rotbond_grid_generated": "per-rotbond-grid-generated",
    "rotbond_distribution_all_methods": "rotbond-distribution-all-methods",
    "strain_rotbond_all_methods": "per-rotbond-all-methods",
    "strain_per_atom_all_methods_mean": "per-atom-all-methods-mean",
    "strain_per_atom_all_methods_median": "per-atom-all-methods-median",
    "strain_per_atom_all_methods": "per-atom-all-methods",
    # fig-rigid-consistency
    "rigid_ecdf_pair_all": "ecdf-pair-all",
    "rigid_ecdf_pair_core": "ecdf-pair-core",
    "rigid_per_size_mean_all": "per-size-mean-all",
    "rigid_per_size_mean_core": "per-size-mean-core",
    "rigid_per_size_median_all": "per-size-median-all",
    "rigid_per_size_median_core": "per-size-median-core",
    "rigid_violin_all": "violin-all",
    "rigid_violin_core": "violin-core",
    "rigid_violin_paperbins_all": "violin-paperbins-all",
    "rigid_violin_paperbins_core": "violin-paperbins-core",
    "rigid_fragment_per_size": "fragment-per-size",
    "rigid_fragment_by_bin": "fragment-by-bin",
    # fig-posecheck, interaction fingerprints (ProLIF, via PoseCheck)
    "interaction_contacts_all": "interaction-contacts-all",
    "interaction_contacts_core": "interaction-contacts-core",
    "interaction_hbonds_all": "interaction-hbonds-all",
    "interaction_hbonds_core": "interaction-hbonds-core",
    "interaction_pair_contacts_core": "interaction-pair-contacts-core",
    "interaction_pair_hbonds_core": "interaction-pair-hbonds-core",
    "interactions": "interactions",
    "interactions_pair": "interactions-pair",
    # fig-molweight
    "mw_distribution_all": "distribution-all",
    "mw_distribution_core": "distribution-core",
    "mw_ecdf_all": "ecdf-all",
    "mw_ecdf_core": "ecdf-core",
    "mw_histogram": "histogram",
    # fig-similarity — a/b/c/d is the paper's panel order, so it stays
    "similarity_a_dumbbell": "a-dumbbell",
    "similarity_b_bars": "b-bars",
    "similarity_c_scatter": "c-scatter",
    "similarity_d_table": "d-table",
    "similarity_novelty": "novelty",
    # fig-mcp
    "mcp_finetune_size": "finetune-size",
    # fig-ensemble
    "fig1_ensemble_6set_3metric": "6set-3metric",
    "fig2_ensemble_vs_alone": "vs-alone",
}


def filename(stem):
    """What a figure is called on disk: hyphens, minus whatever the folder already says.

    STEMS spells every current name out rather than deriving it, because the token to drop
    is different in each family (`rigid_`, `pb_`, `mw_`, ...) and a clever rule would have
    to special-case `sucos_`, `strain_` and `clash_`, which are metrics and must stay. The
    mechanical part -- underscores to hyphens -- is applied to anything not listed, so a
    newly added figure is named consistently from its first run; add it to STEMS, which is
    where the layout is documented."""
    return STEMS.get(stem, stem.replace("_", "-"))
LEGACY = REPO / "notebook" / "html" / "260910"         # the exported CSV/JSON artifacts
E = REPO / "voxbind" / "exps"                          # the per-molecule source trees

# ════════════════════════════════════════════════════════════════════════════════
# palette — one colour per method, every figure, every spelling
# ════════════════════════════════════════════════════════════════════════════════
# A reader moving between the Vina, PoseBusters and PoseCheck figures should never have to
# re-learn the key, so the assignment lives here and nothing defines a colour locally.
# Fixed points (2026-09-09): CoDE is OUR blue #4363D8; VoxBind keeps its sand; DecompDiff
# green; TargetDiff violet (it used to be orange, a few degrees of hue from VoxBind's sand,
# which made two BASELINES read as one family). AR moved off purple onto teal-cyan because
# violet now sits where it was. Ours v2 is a DARKER shade of the CoDE blue rather than a
# hue of its own -- it is the same family -- and dark blue / mid blue / light violet is a
# lightness ladder that reads even where the hues are neighbours.
COLORS = {
    "Reference ligand": "#9AA0A6",   # grey, and dashed wherever it is drawn
    "AR":               "#17A2B8",   # teal-cyan
    "Pocket2Mol":       "#E87BA4",   # pink
    "DiffSBDD":         "#E34948",   # red
    "DecompDiff":       "#3CB44B",   # green
    "FuncBind":         "#A9744F",   # brown
    "TargetDiff":       "#B58FDB",   # violet
    "VoxBind":          "#F5B27E",   # sand
    "CoDE":             "#4363D8",   # blue — ours (LaTeX: \textsc{CoDE})
    "Ours v2":          "#2B3A8C",   # deep indigo — same family as CoDE
}

# The MCP fine-tune arms are not four independent methods -- they are one model at four
# amounts of receptor-ED fine-tuning -- so they take an ordinal ramp off the FuncBind brown
# (they ARE FuncBind), darkening with training, rather than four categorical hues.
MCP_RAMP = {
    "FuncBind vanilla":   "#A9744F",
    "FuncBind ft 3.17M":  "#86593A",
    "FuncBind ft 8.21M":  "#684226",
    "FuncBind ft 26.1M":  "#4A2C12",
}
COLORS.update(MCP_RAMP)

ALIASES = {
    "Reference": "Reference ligand", "reference": "Reference ligand",
    "DecompDiff_ref_prior": "DecompDiff", "DecompDiff (ref-informed)": "DecompDiff",
    "targetdiff": "TargetDiff",
    "vanilla": "VoxBind", "VoxBind σ=0.9": "VoxBind", "VoxBind σ0.9": "VoxBind",
    "VoxBind sigma=0.9": "VoxBind", r"VoxBind$_{\sigma=0.9}$": "VoxBind",
    "ours_v1": "CoDE", "Ours": "CoDE", "Ours · v1": "CoDE", "Ours &middot; v1": "CoDE",
    "Ours v1": "CoDE", "VoxBind + Ours": "CoDE", "\\textsc{CoDE}": "CoDE", "CODE": "CoDE",
    "ours_v2": "Ours v2", "Ours · v2": "Ours v2", "Ours &middot; v2": "Ours v2",
    "vanilla_mcp": "FuncBind vanilla", "fb_unified": "FuncBind vanilla",
    "ft_3.17M": "FuncBind ft 3.17M", "ft_8.21M": "FuncBind ft 8.21M",
    "ft_26.1M": "FuncBind ft 26.1M",
}

# What a LEGEND shows, where that differs from the key the data is stored under. The key
# heads every exported CSV column and JSON object, and a column header containing
# matplotlib mathtext is not something a reader can join on -- so the exports carry plain
# "VoxBind" and only the drawn label picks up the sigma subscript.
DISPLAY = {"VoxBind": r"VoxBind$_{\sigma=0.9}$"}


def color(label):
    """Colour for a method, by any of its spellings. Raises rather than defaulting: a
    silently grey method in a figure is worse than a failed build."""
    key = ALIASES.get(label, label)
    if key not in COLORS:
        raise KeyError(f"no colour for method {label!r} (known: {', '.join(sorted(COLORS))})")
    return COLORS[key]


def display(label):
    """The string to put in a legend for a method, by any of its spellings."""
    key = ALIASES.get(label, label)
    if key not in COLORS:
        raise KeyError(f"no method named {label!r}")
    return DISPLAY.get(key, key)


# LIGHTER STEPS OF THE PALETTE HUES, shared like COLORS so a pastel tone is defined once. They
# used to be hex literals repeated per family (CoDE's soft blue in three of them), which is how
# one figure's tint drifts from another's. Keyed like COLORS, reached by any spelling.
#   SOFT  one step lighter -- CoDE in the Vina per-atom family, the similarity dumbbell and the
#         eight-method strain figures, where the full #4363D8 was the one saturated hue.
#   PALE  two steps lighter -- VoxBind sigma=1.0's tint in the similarity figure, and the two
#         ends of the strain-grid colormap.
SOFT = {"CoDE": "#8291E8"}
PALE = {"CoDE": "#B4BEF0", "VoxBind": "#F8D3B0"}


def soft(label):
    """A method's soft tint where it has one, otherwise its palette colour."""
    return SOFT.get(ALIASES.get(label, label)) or color(label)


def pale(label):
    """A method's pale tint. Raises rather than falling back: a pale step is chosen on purpose,
    and silently getting the full-strength colour instead would not look like an error."""
    key = ALIASES.get(label, label)
    if key not in PALE:
        raise KeyError(f"no pale tint for {label!r} (known: {', '.join(sorted(PALE))})")
    return PALE[key]


# ════════════════════════════════════════════════════════════════════════════════
# house style — no panel titles, warm near-black furniture, left+bottom spines only
# ════════════════════════════════════════════════════════════════════════════════
INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
SOLID, DASH = (0, ()), (0, (4, 2.6))
DOT = (0, (1, 2.6))
MODEL_LW, REF_LW = 2.35, 1.5
AXIS_LW, GRID_LW = 1.35, 1.1
DIST_LW, DIST_FILL = 1.7, 0.16

# GEOMETRY, in inches, off the 3-line base figure of 7.25 x 2.77.
WIDE, TALL = 1.05, 1.155
FIG_W = 7.25 * WIDE
PANEL_H = 2.77 * TALL
STACK_H = PANEL_H + 1.62              # a full panel plus a distribution strip
HEIGHT_RATIOS = (2.62, 1.05)
# The gap between stacked panels goes through tight_layout's h_pad, NOT gridspec_kw's
# hspace: an explicit hspace makes tight_layout call the axes "not compatible", warn, and
# leave ~12% of the figure as dead white at the top.
H_PAD = 0.2
XTICK_STEP = 5
X_LABEL = "Number of heavy atoms in ligand"

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3, which
                                     # journals reject and Illustrator mangles
}


def use_style():
    plt.rcParams.update(RC)


def furniture(ax, *, ylabel=None, xlabel=None, xlim=None, xloc=XTICK_STEP):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15.5, labelpad=9)
    if xlim:
        ax.set_xlim(*xlim)
    if xloc:
        ax.xaxis.set_major_locator(MultipleLocator(xloc))
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=DOT)
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=DOT)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)


def legend(target, handles, loc="upper right", ncol=1, fontsize=12.5, **kw):
    """Opaque white with a rule in the axis pen, so the dotted grid does not run through
    the labels but the box itself stays quiet. `target` is an Axes OR a Figure -- a key
    that has to sit outside the data area belongs to the figure, not to the axes it would
    otherwise cover."""
    leg = target.legend(handles=handles, loc=loc, ncol=ncol, frameon=True,
                        fontsize=fontsize, handlelength=1.9, handletextpad=0.55,
                        labelspacing=0.3, borderpad=0.4, borderaxespad=0.39,
                        facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def arm_handles(arms, include_ref=True):
    h = [Line2D([], [], color=color(REF_LABEL), lw=REF_LW, ls=DASH, label=REF_LABEL)] \
        if include_ref else []
    return h + [Line2D([], [], color=color(lab), lw=MODEL_LW, ls="-", label=display(lab))
                for lab, _, _ in arms]


def fit(fig, **kw):
    """tight_layout, then give back whatever it under-reserved.

    At this type size it can leave a rotated y label overrunning the figure edge even
    though it has just run, and the label is then silently SLICED OFF in the raster.
    get_tightbbox reports the overrun after the fact, so hand back exactly that much --
    re-running tight_layout with a rect would re-apply `pad` on top of the correction and
    cost plot area."""
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


# ════════════════════════════════════════════════════════════════════════════════
# the pose data — one load per run, shared by every figure that asks for it
# ════════════════════════════════════════════════════════════════════════════════
# label, key, run root — drawn in this order, so our method lands on top. The labels are
# DATA KEYS as well as labels (they head the exported CSV columns), so they stay plain;
# display() dresses them for a legend.
ARMS = [
    ("AR",         "ar",         f"{E}/baselines_pose/ar"),
    ("Pocket2Mol", "pocket2mol", f"{E}/baselines_pose/pocket2mol"),
    ("DiffSBDD",   "diffsbdd",   f"{E}/baselines_pose/diffsbdd"),
    ("DecompDiff", "decompdiff", f"{E}/baselines_pose/decompdiff"),
    ("FuncBind",   "funcbind",   f"{E}/baselines_pose/funcbind"),
    ("TargetDiff", "targetdiff", str(BASEDRUG / "eval" / "targetdiff")),
    ("VoxBind",    "vanilla",    f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       "ours_v1",    f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]
# `core` is the comparison this section makes: our method against the model it modifies,
# with the crystal ligand as the benchmark. `all` is every arm the figure has data for.
CORE = ("vanilla", "ours_v1")
REF_LABEL = "Reference ligand"
REF_COLOR = COLORS[REF_LABEL]
# The crystal ligand is the same molecule in every run that carries one, so any of OUR
# roots will do -- but it is resolved BY KEY, never by position: ARMS[2][2] silently became
# DiffSBDD the moment the baselines were prepended, and the staged baseline dirs carry no
# reference block at all.
REF_ROOT = next(root for _, key, root in ARMS if key == "ours_v1")

# THE POCKET SET IS THE 79, NOT EACH ARM'S OWN COVERAGE. TargetDiff holds 100 pockets, but
# CoDE only sampled the 79 with usable deposited electron density, and every arm covers
# those -- so every figure is like-for-like. target_71 is in: it is unscoreable on the
# DOCKING side only, both pose metrics work on it.
P79 = json.load(open(f"{E}/frozenenc_probes/p79_targets.json"))
EDGES = [0, 16, 21, 26, 31, 10 ** 6]
BIN_LABELS = ["≤15", "16–20", "21–25", "26–30", ">30"]

# A heavy-atom count is plotted only where EVERY drawn arm has this many molecules, so the
# curves start and stop together and none carries a tail the others cannot answer.
MIN_N = 25
# The reference window: +-REF_WIN atoms, kept where it pools at least MIN_REF ligands.
# There is one crystal ligand per pocket -- 79 over ~40 sizes -- so a per-count reference
# curve would be noise, and its share would be a picket fence of 5%-tall spikes.
REF_WIN, MIN_REF = 4, 12


def rows_of(target_dir, reference=False):
    """One record per molecule: heavy atoms, PoseCheck strain and clashes, its interaction
    fingerprint, PoseBusters validity and the checks it failed. `s=None` means the UFF
    relaxation did not converge -- kept as None rather than 0, because dropping a value is
    not the same as scoring it.

    `ifp` IS THE POSECHECK FINGERPRINT, NOT `m["interactions"]`. A molecule record carries
    two unrelated things under that name: `m["interactions"]` is the docking side's
    contact/clash geometry, `m["posecheck"]["interactions"]` is the ProLIF fingerprint. A
    type the molecule does not make is ABSENT from the dict, not zero in it, so read it
    with .get(type, 0) and never with len() or a sum over its keys."""
    path = os.path.join(target_dir, "metrics.json")
    if not os.path.exists(path):
        return []
    try:
        j = json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items = [j.get("reference")] if reference else (j.get("samples") or [])
    out = []
    for m in items:
        if not m or not m.get("n_atoms"):
            continue
        pc = m.get("posecheck") if isinstance(m.get("posecheck"), dict) else {}
        pb = m.get("posebusters") if isinstance(m.get("posebusters"), dict) else {}
        s = pc.get("strain")
        out.append({
            "n": int(m["n_atoms"]),
            "smi": m.get("smiles"),
            "s": float(s) if isinstance(s, (int, float)) and np.isfinite(s) else None,
            "c": float(pc["clashes"]) if pc.get("clashes") is not None else None,
            "ifp": pc["interactions"] if isinstance(pc.get("interactions"), dict) else None,
            "v": pb["valid"] if isinstance(pb.get("valid"), bool) else None,
            "f": sorted(k for k, ok in (pb.get("checks") or {}).items() if ok is False),
        })
    return out


_POSE_CACHE = {}


def pose_data():
    """(per-arm {target: rows}, p79 rows per arm, the 79 crystal-ligand rows).

    Cached for the life of the process. This is the expensive part of every pose figure --
    ~60k molecules across eight run trees -- and drawing six of them in one command used to
    mean reading it six times, once per builder."""
    if "d" not in _POSE_CACHE:
        t0 = time.time()
        data = {}
        for _, key, root in ARMS:
            names = sorted(d for d in os.listdir(root) if d.startswith("target_")
                           and os.path.isdir(os.path.join(root, d)))
            data[key] = {t: rows_of(os.path.join(root, t)) for t in names}
        p79 = {key: [r for t in P79 for r in data[key].get(t, [])] for _, key, _ in ARMS}
        ref = [r for t in P79 for r in rows_of(os.path.join(REF_ROOT, t), reference=True)]
        _POSE_CACHE["d"] = (data, p79, ref)
        n = sum(len(v) for v in p79.values())
        print(f"  [loaded {n:,} molecules over {len(ARMS)} arms in {time.time()-t0:.1f}s]")
    return _POSE_CACHE["d"]


def arms_for(field, data, arms=None):
    """The arms that carry `field` across the WHOLE 79-pocket set, in ARMS order.

    An arm with no data for a metric must not become an empty curve and a legend row that
    says it was measured and lost -- it was never measured. The baselines hold PoseBusters
    and not PoseCheck, so the two families end up with different arm lists from one ARMS.
    The test is per POCKET, not "has any value at all": a scoring run in progress leaves an
    arm with a handful of finished pockets, and pooling those would draw a complete-looking
    curve over 4% of the data."""
    arms = ARMS if arms is None else arms
    out = []
    for a in arms:
        pockets = [t for t in P79 if t in data.get(a[1], {})]
        if not pockets:
            continue
        scored = sum(any(r[field] is not None for r in data[a[1]][t]) for t in pockets)
        if scored == len(pockets) == len(P79):
            out.append(a)
    return out


def variants(field=None, data=None):
    """(name, arms) for each figure variant, core first."""
    arms = ARMS if field is None else arms_for(field, data)
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


def core_only(pairs):
    """Just the `core` entry of a variants()-shaped sequence.

    For PoseCheck STRAIN, `all` is not eight methods -- only TargetDiff, VoxBind and CoDE
    carry a strain value at all, because the five published baselines were scored for
    PoseBusters here and their PoseCheck came back from svr12 as exports rather than as
    per-molecule rows. So `all` was `core` plus TargetDiff, a third figure that is neither
    the two-arm comparison nor the whole field, and the whole field already has its own
    figures: strain_per_atom_all_methods, strain_rotbond_all_methods and
    strain_box_per_rotbond_all_methods read those exports and draw all eight. Dropping the
    near-duplicate leaves one figure per question instead of three per question."""
    return [p for p in pairs if p[0] == "core"]


# ── statistics ───────────────────────────────────────────────────────────────────
def by_size(rows, field, key="n"):
    """{x: [values]}, where x is the row's `key` -- heavy atoms by default. `key` exists so
    a builder can resolve the same metric against another per-molecule integer (rotatable
    bonds) through the same statistics and the same x_range/model_curve/reference_curve."""
    out = collections.defaultdict(list)
    for r in rows:
        if r[field] is not None and r.get(key) is not None:
            out[r[key]].append(r[field])
    return out


def bin_of(n):
    return min(int(np.searchsorted(EDGES, n, side="right")) - 1, len(BIN_LABELS) - 1)


def model_curve(per, xs, f, win=0):
    """Value at each exact heavy-atom count, or over a centred +-win window.

    A median of a continuous quantity (strain, clashes) is stable at one count -- each arm
    has 100-300 molecules there -- so those curves are drawn unrolled. A PROPORTION IS NOT:
    at n=150 a per-count validity rate carries about +-4 pp of sampling noise, enough that
    arms separated by 2-10 pp cross each other repeatedly on nothing, so those curves pass
    win=2 (~750 molecules per point, about +-1.8 pp). The window is the only smoothing
    anywhere; nothing is interpolated and no point is invented."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= win for v in vals] if win \
            else per.get(a, [])
        out.append(f(p) if p else None)
    return out


def reference_curve(per, xs, f):
    """The same over a centred +-REF_WIN window; None where the window is too thin to mean
    anything, which leaves a gap in the line rather than an invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(p) if len(p) >= MIN_REF else None)
    return out


def x_range(per_arm, arms):
    """Counts where every drawn arm has at least MIN_N molecules."""
    common = set.intersection(*(set(per_arm[key]) for _, key, _ in arms))
    return [a for a in sorted(common) if all(len(per_arm[key][a]) >= MIN_N
                                             for _, key, _ in arms)]


def share(per, xs):
    total = sum(len(v) for v in per.values())
    return ([100.0 * len(per.get(a, ())) / total for a in xs] if total else []), total


def rolled(values, xs):
    """The reference's shares under the same +-REF_WIN window -- averaged, not summed, so
    it stays on the same scale as the models' per-count shares and all read on one axis."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= REF_WIN]) for a in xs]


def size_distribution(ax, xs, per_arm, ref_per, arms):
    """Where each set puts its molecules, as a share of its own -- which is what makes the
    panel above it trustworthy, and is itself a finding, since the arms differ in the sizes
    they generate as much as in per-size pose quality. It counts the SAME molecules the
    panel above plots."""
    for lab, key, _ in arms:
        pct, _ = share(per_arm[key], xs)
        col = color(lab)
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, step="mid", color=col, alpha=DIST_FILL, lw=0, zorder=2)
    raw, _ = share(ref_per, xs)
    ax.plot(xs, rolled(raw, xs), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    ax.set_ylim(bottom=0)


# ════════════════════════════════════════════════════════════════════════════════
# per-molecule Vina, from the results bundle — the one source every method shares
# ════════════════════════════════════════════════════════════════════════════════
# Until 2026-09-13 the Vina figures read each run's own eval_docking_results_full79.json
# out of voxbind/exps/, which only OUR arms have: the five published baselines were docked
# on svr12 and only their aggregates came back, so they could not be drawn at all. The
# bundle now carries <Method>/eval/vina_docking/per_molecule.csv for every method --
# target, n_atoms, vina_score, vina_min, vina_dock, and an in_density79 flag -- which is
# one uniform table per method instead of a per-run file format. Both Vina families read
# it, so "which methods are in this figure" is a question about which CSVs exist.
BUNDLE = REPO / "results" / "task2-drugdesign"
VINA_PER_MOLECULE = "eval/vina_docking/per_molecule.csv"

# (palette label, bundle folder), in the section's arm order so the drawn order, the zorder
# and the legend all agree with every other `all` figure here. VoxBind is the published
# sigma=0.9 reproduction and CoDE is ours; the Fusion arms and the base-ep350 variants are
# an ablation, a different comparison, and are deliberately not part of "all methods".
VINA_BUNDLE_ARMS = [
    ("AR",         "AR",              None),
    ("Pocket2Mol", "Pocket2Mol",      None),
    ("DiffSBDD",   "DiffSBDD",        None),
    ("DecompDiff", "DecompDiff",      None),
    ("FuncBind",   "FuncBind",        None),
    # THIS ARM'S BUNDLE CSV IS THE WRONG PROTOCOL. `VoxBind-vanilla` is
    # exps/reproduction/samples/res_test_100, Vina-scored by metrics.py against the
    # pocket10 CROP (its results.json records dock_receptor_scope: null), and a crop and a
    # whole receptor do not share an axis -- the same crystal ligand docks to -7.18 one way
    # and -7.31 the other. The third field names a run tree to read INSTEAD when the
    # bundle's scope is not `full`: _vanilla_ep923 is the same published model under the
    # baseline protocol, and is what the per-atom family already draws, so the two Vina
    # families now show the same VoxBind. Delete it once res_test_100's re-dock lands in
    # the bundle and the CSV itself is full/exh32.
    ("VoxBind",    "VoxBind-vanilla", f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       "VoxBind-Ours",    None),
]
VINA_WANT_SCOPE = "full"


def vina_rows(folder, *, density79=True):
    """Per-molecule Vina for one bundle method, or None if that method has no CSV.

    `density79` keeps only the 79 pockets with usable deposited density -- the set every
    figure in this section is drawn over. The flag is in the file, so the subset is the
    bundle's own definition and not a pocket list re-derived here."""
    path = BUNDLE / folder / VINA_PER_MOLECULE
    if not path.exists():
        return None
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if density79 and r.get("in_density79") != "1":
                continue
            try:
                row = {"target": r["target"], "n": int(r["n_atoms"])}
            except (KeyError, ValueError):
                continue
            for f in ("vina_score", "vina_min", "vina_dock"):
                v = r.get(f)
                row[f] = float(v) if v not in (None, "", "None") else None
            out.append(row)
    return out


def vina_engine(folder):
    """The engine string the bundle records for one method's Vina, for the run log.

    THE FIGURES MIX ENGINES AND MUST SAY SO. The five published baselines and CoDE are
    TargetDiff's VinaDockingTask over the whole receptor; the Fusion and base arms went
    through notebook/webapp/metrics.py; VoxBind-vanilla's older numbers were scored against
    the pocket10 CROP, which is not the same axis at all -- the same crystal ligand docks
    to -7.31 one way and -7.18 the other. A figure that pools them is only honest if the
    census is printed beside it."""
    path = BUNDLE / folder / "eval/vina_docking/results.json"
    if not path.exists():
        return None
    try:
        pr = json.load(open(path, encoding="utf-8")).get("protocol", {})
    except (json.JSONDecodeError, OSError):
        return None
    return f"{pr.get('engine')} / scope={pr.get('dock_receptor_scope')}"


def vina_run_rows(root, field="vina_dock"):
    """Per-molecule rows from a run's own eval_docking_results_full79.json, shaped like
    vina_rows(). Used where the bundle CSV is not the protocol this figure draws."""
    path = os.path.join(root, "eval_docking_results_full79.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            per_target = json.load(handle)["per_target"]
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    out = []
    for t in per_target:
        for m in t.get("per_mol") or []:
            if not isinstance(m, dict) or not m.get("n_atoms"):
                continue
            row = {"target": t["target"], "n": int(m["n_atoms"])}
            for f in ("vina_score", "vina_min", "vina_dock"):
                v = m.get(f)
                row[f] = float(v) if v is not None else None
            out.append(row)
    return out or None


def vina_bundle(*, density79=True):
    """[(label, folder, rows)] for every arm this figure can draw, in section order, with
    the engine census printed. An arm that has no data is named, so "not drawn" never looks
    like "not measured"; an arm whose bundle CSV is the wrong protocol is read from the run
    tree its entry names, and the substitution is printed."""
    out, engines = [], collections.Counter()
    for label, folder, run_root in VINA_BUNDLE_ARMS:
        rows = vina_rows(folder, density79=density79)
        eng = vina_engine(folder) or "unrecorded"
        if run_root and f"scope={VINA_WANT_SCOPE}" not in eng:
            run_rows = vina_run_rows(run_root)
            if run_rows:
                print(f"    {label}: bundle CSV is {eng.split('scope=')[-1]} scope; "
                      f"reading {os.path.relpath(run_root, REPO)} instead "
                      f"(baseline protocol)")
                rows = run_rows
                eng = f"run_docking_eval.py baseline protocol / scope={VINA_WANT_SCOPE}"
            else:
                print(f"    {label}: bundle CSV is the wrong protocol and "
                      f"{os.path.relpath(run_root, REPO)} has no full79 results")
        if rows is None:
            print(f"    {label} skipped: no {folder}/{VINA_PER_MOLECULE}")
            continue
        if not rows:
            print(f"    {label} skipped: {folder}/{VINA_PER_MOLECULE} has no scored rows")
            continue
        engines[eng] += 1
        out.append((label, folder, rows))
    for eng, n in engines.most_common():
        print(f"    engine: {eng}  ({n} arm{'s' if n > 1 else ''})")
    # SCOPE is what breaks the axis, not the implementation. A crop and a whole receptor
    # put the same crystal ligand 0.13 kcal/mol apart, so mixing them is a real error;
    # TargetDiff's VinaDockingTask and run_docking_eval.py at the same scope and the same
    # exhaustiveness are the same protocol by construction, which is the whole point of
    # 73_dock_baseline_protocol_79.sh. So the warning fires on scope, and the engine census
    # above is printed either way for the reader to judge.
    scopes = {e.split("scope=")[-1] for e in engines}
    if len(scopes) > 1:
        print(f"    NOTE: more than one receptor scope in this figure ({', '.join(sorted(scopes))})"
              " — the arms are NOT on one axis; see each method's eval/vina_docking/results.json")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# the registry — id -> function, and the folder that id names
# ════════════════════════════════════════════════════════════════════════════════
FIGURES = {}          # "fig-{eval}-{figname}" -> function(out_dir) -> None
_FORMATS = ("png", "svg", "pdf")


def figure(fig_id, *, needs=(), folder=None):
    """Register one figure.

    `fig_id` is the CLI name; the output folder is its EVALUATION (see EVAL_FOLDER), so
    everything answering the same question lands together. `folder` overrides that with a
    path under OUT_ROOT, which is how an evaluation that grows past a readable folder --
    PoseCheck has forty figures over three different measurements -- splits into
    sub-folders without the ids or the CLI changing.

    `needs` names the evaluation inputs the function reads, and is printed by --list so a
    missing input is a readable message rather than a traceback three call levels down."""
    def deco(fn):
        if fig_id in FIGURES:
            raise KeyError(f"duplicate figure id {fig_id!r}")
        if not fig_id.startswith("fig-"):
            raise ValueError(f"figure id must be fig-<eval>-<name>: {fig_id!r}")
        fn.fig_id, fn.needs, fn.folder = fig_id, tuple(needs), folder
        FIGURES[fig_id] = fn
        return fn
    return deco


def figure_folder(fig_id):
    """Where a figure's files go, relative to OUT_ROOT."""
    override = getattr(FIGURES[fig_id], "folder", None)
    return override or EVAL_FOLDER[fig_id.split("-")[1]]


def save(fig, out, stem):
    """PNG to look at, SVG and PDF to place -- both vector, both with live text.

    `stem` is the name the ORIGINAL builder used; filename() turns it into the name this
    tree uses. Families pass the original name so STEMS stays a usable map back to
    notebook/html/260910/, which is what the port was verified against."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = filename(stem)
    written = []
    for ext in _FORMATS:
        p = out / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        written.append(p)
    plt.close(fig)
    return written


def write_csv(out, stem, header, rows):
    """The numbers behind a figure, beside it. A figure whose values cannot be read back
    out is not reviewable."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{filename(stem)}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


def legacy(*parts):
    """A data artifact under notebook/html/260910/, which is where the recompute scripts
    still write. Raises with the path if it is not there -- an empty figure that silently
    dropped a method is the failure mode this exists to prevent."""
    p = LEGACY.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(
            f"missing input {p}\n"
            f"  rebuild it with the matching notebook/html/260910/*/build_*.py")
    return p


# ════════════════════════════════════════════════════════════════════════════════
# fig-vina-per-pocket-{dock,score,min} — every method's per-pocket Vina, ranked
# ════════════════════════════════════════════════════════════════════════════════
# x is the pockets sorted by the FOCUS arm's own value for that metric and statistic, so
# its points climb monotonically and everything else is read against them. Every series is
# a point cloud, not a curve: consecutive pockets are a RANKING, not a sequence, and a
# connecting line would imply a continuity that is not there. x is ticked at ranks
# 1/20/40/60/last while its rules fall every 10, and neither is a pocket id -- rank 1 is
# whichever pocket that metric ranks first -- so the exported CSV maps every rank to its
# target.
#
# THE ARM LIST IS DATA. It was three hardcoded series ("3line") until 2026-09-13; it is now
# VPP_ARMS, and an arm whose per-pocket file is absent is SKIPPED WITH ITS PATH NAMED
# rather than silently missing. So the day a method's per-pocket results land, it appears
# here with no code change -- which is the whole reason the figure stopped being called
# "3line". What each one needs is `<root>/<VPP_BASENAME>` with a `per_target` list whose
# entries carry `target` and a `per_mol` list of {vina_score, vina_min, vina_dock}.
#
# PROTOCOL. The published-baseline protocol only -- whole `*_rec.pdb` receptor,
# exhaustiveness 32, the 79 density pockets -- so these clouds sit on the same axis as the
# AR/Pocket2Mol/DiffSBDD/DecompDiff/FuncBind table. The older crop protocol
# (`*_pocket10.pdb`, exhaustiveness 16, 78 pockets) is deliberately NOT plotted: the same
# crystal reference docks to -7.31 one way and -7.18 the other, so the two cannot share an
# axis. `eval_docking_results_full.json` is a third thing again (full receptor,
# exhaustiveness 16); only the `_full79` file is the baseline protocol.
VPP_BASENAME = "eval_docking_results_full79.json"
VPP_PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# label -> run root. Order is the drawing order, so the focus arm lands on top; an entry
# with no VPP_BASENAME under it is reported and dropped. The five published baselines and
# the two svr12 arms are listed so the skip message names the exact file that has to
# appear -- "not plotted" and "not measured" must never look the same.
VPP_ARMS = [
    ("AR",         f"{E}/baselines_pose/ar"),
    ("Pocket2Mol", f"{E}/baselines_pose/pocket2mol"),
    ("DiffSBDD",   f"{E}/baselines_pose/diffsbdd"),
    ("DecompDiff", f"{E}/baselines_pose/decompdiff"),
    ("FuncBind",   f"{E}/baselines_pose/funcbind"),
    ("TargetDiff", str(BASEDRUG / "eval" / "targetdiff")),
    ("VoxBind",    f"{E}/reproduction/samples/res_test_100"),
    ("VoxBind",    f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]
# The arm the ranking is built from, and the one drawn last and largest.
VPP_FOCUS = "CoDE"
# `core` is the comparison this section makes -- our method against the model it modifies,
# with the crystal ligand as the benchmark -- and `all` adds the published baselines. The
# same split every other figure here carries, and for the same reason: two figures that
# differ in which arms they draw must not be able to share a name.
VPP_CORE = ("VoxBind", "CoDE")

# metric key -> (per-molecule field, per-pocket reference field, y-axis label)
VPP_METRICS = {
    "dock":  ("vina_dock",  "ref_vina_dock",  "Vina Dock"),
    "score": ("vina_score", "ref_vina_score", "Vina Score"),
    "min":   ("vina_min",   "ref_vina_min",   "Vina Min"),
}
VPP_STATS = ("mean", "median")

# Ours is drawn in its SOFT tint (the shared table above), not the flat #4363D8: against the
# sand at 79 small markers the saturated blue reads as the louder series, and the ordering --
# not saturation -- is what is supposed to be doing the arguing.
# Shape is a second identity channel, so the series survive greyscale and colour blindness.
# matplotlib's `markersize` is a DIAMETER, and at one diameter a triangle carries only 41%
# of a circle's ink -- two series would read as two different weights -- so every non-round
# marker is area-matched by VPP_SHAPE_SCALE and the focus arm keeps the 3.89 that sets it
# slightly ahead. Anything not named here takes the next marker in VPP_MARKER_CYCLE, which
# is what lets a new method arrive without a palette decision.
VPP_BASE_SIZE = 3.31
VPP_FOCUS_SIZE = 3.89
VPP_MARKERS = {"VoxBind": "^", "CoDE": "o"}
VPP_MARKER_CYCLE = ("s", "D", "v", "P", "X", "<", ">", "*")
VPP_SHAPE_SCALE = {"o": 1.0, "^": 1.555, "v": 1.555, "s": 1.13, "D": 1.35,
                   "P": 1.2, "X": 1.2, "<": 1.555, ">": 1.555, "*": 1.7}
VPP_REF_LW = 1.25
VPP_X_LABEL = "Target pocket number"
VPP_XTICKS = (1, 20, 40, 60)          # plus the last rank, appended at draw time
# WIDE/TALL scale the 7.25 x 2.77 base figure. The paired figure is SIZED FROM the
# standalone one so one of its panels is exactly the same shape -- moving between the two
# is never silently a change of aspect ratio.
VPP_FIG_W, VPP_FIG_H = 7.25 * 0.88, 2.77 * 1.155


def _vpp_reference_root():
    """A run that records the crystal ligand's own docked value per pocket.

    The bundle's per-molecule CSVs are GENERATED molecules only -- there is no reference
    row in them -- so the grey benchmark still comes from a run's own
    eval_docking_results_full79.json, which carries ref_vina_* beside each target."""
    for _, root in (("CoDE", f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
                    ("VoxBind", f"{E}/_vanilla_ep923/samples/full_eval_ep923")):
        if os.path.exists(os.path.join(root, VPP_BASENAME)):
            return root
    return None


def _vpp_per_target(rows, metric_field):
    """{target: {"mean", "median", "n"}} from per-molecule rows."""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(metric_field) is not None:
            by[r["target"]].append(r[metric_field])
    return {t: {"mean": st.mean(v), "median": st.median(v), "n": len(v)}
            for t, v in by.items() if v}


def _vpp_slug(label):
    """A label as a CSV column prefix: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


def _vpp_marker(label, index):
    m = VPP_MARKERS.get(label) or VPP_MARKER_CYCLE[index % len(VPP_MARKER_CYCLE)]
    base = VPP_FOCUS_SIZE if label == VPP_FOCUS else VPP_BASE_SIZE
    return m, base * VPP_SHAPE_SCALE.get(m, 1.0)


def _vpp_load(root):
    """{target: {metric: {"mean", "median", "ref", "n"}}} from one run's docking results."""
    with open(os.path.join(root, VPP_BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    out = {}
    for t in per_target:
        row = {}
        for key, (field, ref_field, _) in VPP_METRICS.items():
            vals = [m[field] for m in (t.get("per_mol") or [])
                    if isinstance(m, dict) and m.get(field) is not None]
            if not vals:
                continue
            row[key] = {"mean": st.mean(vals), "median": st.median(vals),
                        "ref": t.get(ref_field), "n": len(vals)}
        if row:
            out[t["target"]] = row
    return out


def _vpp_reference(data, targets, metric):
    """Per-pocket crystal-ligand value, averaged over the arms that report it. Only Dock is
    meaningfully noisy: score_only and minimize are deterministic and the runs agree
    exactly, while Dock re-searches unseeded and can disagree by ~0.7 kcal/mol on a pocket."""
    ref, worst = {}, (0.0, None)
    for t in targets:
        vals = [d[t][metric]["ref"] for d in data.values()
                if t in d and d[t].get(metric, {}).get("ref") is not None]
        ref[t] = st.mean(vals) if vals else None
        if len(vals) > 1 and max(vals) - min(vals) > worst[0]:
            worst = (max(vals) - min(vals), t)
    return ref, worst


def _vpp_limits(*serieses):
    """Clip to the bulk instead of squashing 79 pockets for a couple of runaway values; the
    caller names the excluded points, the image does not."""
    flat = sorted(v for ys in serieses for v in ys if v is not None)
    pct = lambda q: flat[int(round(q * (len(flat) - 1)))]
    return pct(0.01) - 0.45, pct(0.99) + 0.55


def _vpp_draw(ax, order, clouds, y_ref, ylim, *, legend_on, ylabel):
    """One panel: a dashed reference rule, one point cloud per arm, a dotted grid. The focus
    arm is drawn last and a shade larger, because it is the series the ordering is built
    from and the one being read."""
    x = list(range(1, len(order) + 1))
    lo, hi = ylim
    # The reference is one crystal-ligand value per pocket, so under an ordering set by
    # another series it is inherently jagged; thin, grey and recessive so it reads as the
    # benchmark it is and does not fight the model series.
    ax.plot(x, y_ref, color=REF_COLOR, lw=VPP_REF_LW, ls=DASH, alpha=0.62, zorder=2,
            label=REF_LABEL, solid_capstyle="round")
    for i, (label, ys) in enumerate(clouds):
        marker, size = _vpp_marker(label, i)
        colour = soft(label)
        focus = label == VPP_FOCUS
        ax.plot(x, ys, ls="none", marker=marker, markersize=size, color=colour,
                markerfacecolor=colour, markeredgewidth=0, zorder=4 if focus else 3,
                label=display(label))

    ax.set_xlabel(VPP_X_LABEL, fontsize=15.5, labelpad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10)
    ax.set_xlim(0.2, len(order) + 0.8)
    ax.set_ylim(lo, hi)
    # y is kcal/mol and the panel is short, so the default locator lands on halves: two
    # extra glyphs per label for precision the eye cannot use at this size. Integers only,
    # and let the locator pick the step rather than pinning one that suits only one metric.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([t for t in VPP_XTICKS if t < len(order)] + [len(order)])
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=DOT)
    # The x rules are plain vlines rather than a minor grid: a minor tick is silently
    # dropped wherever a major tick already sits, which left 20/40/60 without a rule.
    for xv in range(10, len(order) + 1, 10):
        ax.axvline(xv, color=GRID, lw=GRID_LW, ls=DOT, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)

    if legend_on:
        # One column while the key is short, two once it would run past the panel -- a
        # nine-row key in the corner of a 2.2 in panel covers the data it explains.
        ncol = 1 if len(clouds) <= 3 else 2
        leg = ax.legend(loc="lower right", ncol=ncol, frameon=True, fontsize=12.5,
                        handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                        columnspacing=1.0, borderpad=0.4, borderaxespad=0.39,
                        facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0)
        leg.get_frame().set_boxstyle("square", pad=0.16)
        leg.get_frame().set_linewidth(AXIS_LW)
        leg.set_zorder(6)
        for text in leg.get_texts():      # identity rides the swatch, not the ink
            text.set_color(INK)
        # Swatches at twice the plotted size: in the cloud a marker only has to be
        # findable, in the key it has to be identifiable, and at 3-5 pt a triangle and a
        # circle are the same dot. Legend handles are copies, so this does not touch data.
        for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
            if handle.get_markersize():
                handle.set_markersize(handle.get_markersize() * 2)


def _vpp_outside(order, curves, ylim):
    """Points the y-clip leaves off the panel, as (label, target, value)."""
    lo, hi = ylim
    return [(label, order[i], v) for label, ys in curves
            for i, v in enumerate(ys) if v is not None and not lo <= v <= hi]


def _vpp_pair_width(target_w, build):
    """Figure width the paired figure needs for ONE panel to be `target_w` wide. Lays the
    pair out once at a trial width, measures what the furniture actually costs at this type
    size, and adds that back around two panels of the requested width -- a constant here
    goes stale silently when the type size or the tick-label width moves."""
    trial = 12.0
    fig, axes = build(trial)
    furn = trial - 2 * plot_width(fig, axes[0])
    plt.close(fig)
    return 2 * target_w + furn


def _vina_per_pocket(metric, out):
    use_style()
    field, ref_field, label = VPP_METRICS[metric]
    print(f"  {VPP_PROTOCOL}")
    available = vina_bundle(density79=True)
    if not available:
        raise SystemExit("no method has a per-molecule Vina CSV in the results bundle")
    data = {lab: _vpp_per_target(rows, field) for lab, _, rows in available}
    if VPP_FOCUS not in data:
        raise SystemExit(f"the focus arm {VPP_FOCUS} has no {field}; the ranking this "
                         "figure is built on cannot be formed")

    ref, ref_root = {}, _vpp_reference_root()
    if ref_root:
        with open(os.path.join(ref_root, VPP_BASENAME), encoding="utf-8") as handle:
            per_target = json.load(handle)["per_target"]
        ref = {t["target"]: t.get(ref_field) for t in per_target}
        print(f"    reference: {os.path.relpath(ref_root, REPO)}/{VPP_BASENAME}")
    else:
        print("    reference: none available — the grey benchmark is omitted")
    y_ref_of = lambda order: [ref.get(t) for t in order]

    for variant, labels in (("core", [l for l, _, _ in available if l in VPP_CORE]),
                            ("all", [l for l, _, _ in available])):
        missing = [l for l in VPP_CORE if l not in labels]
        if missing:
            raise SystemExit(f"`core` is {' + '.join(VPP_CORE)}; {', '.join(missing)} "
                             "has no per-molecule Vina CSV, so the comparison this figure "
                             "exists to make cannot be drawn")
        _vpp_variant(metric, out, label, field, data, labels, ref, y_ref_of, variant)


def _vpp_variant(metric, out, label, field, data, labels, ref, y_ref_of, variant):
    """One arm set of one metric: mean, median and the pair."""
    targets = sorted(set.intersection(*(set(data[l]) for l in labels)))
    print(f"  {label} · {variant}: {' + '.join(labels)} · {len(targets)} pockets")

    curves = {}
    for stat in VPP_STATS:
        order = sorted(targets, key=lambda t: data[VPP_FOCUS][t][stat])
        clouds = [(lab, [data[lab][t][stat] for t in order]) for lab in labels]
        curves[stat] = (order, clouds, y_ref_of(order))

    spans = {s2: _vpp_limits(*[ys for _, ys in c[1]], c[2]) for s2, c in curves.items()}
    # The paired figure shares one y scale, so its panels are comparable vertically.
    shared = (min(v[0] for v in spans.values()), max(v[1] for v in spans.values()))
    plot_w = None                    # set by the first standalone panel, matched below

    for stat, (order, clouds, y_ref) in curves.items():
        fig, ax = plt.subplots(figsize=(VPP_FIG_W, VPP_FIG_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        _vpp_draw(ax, order, clouds, y_ref, spans[stat], legend_on=True,
                  ylabel=f"{label} {stat}")
        fit(fig, pad=0.5)
        plot_w = plot_width(fig, ax)
        stem = f"vina_{metric}_per_pocket_{stat}_{variant}"
        save(fig, out, stem)

        pooled = {}
        if any(v is not None for v in y_ref):
            pooled[REF_LABEL] = st.mean([v for v in y_ref if v is not None])
        pooled.update({lab: st.mean(ys) for lab, ys in clouds})
        print(f"    {stat:6s} " + "  ".join(f"{k} {v:+.2f}" for k, v in pooled.items()))
        for lab, target, value in _vpp_outside(
                order, clouds + [(REF_LABEL, y_ref)], spans[stat]):
            print(f"      outside the plotted y-range: {lab} @ {target} = {value:+.2f}")

        header = ["rank", "target", "reference"]
        for lab in labels:
            header += [f"n_{_vpp_slug(lab)}", f"{_vpp_slug(lab)}_{stat}"]
        rows = []
        for i, t in enumerate(order, 1):
            r = ref.get(t)
            row = [i, t, f"{r:.3f}" if r is not None else ""]
            for lab in labels:
                row += [data[lab][t]["n"], f"{data[lab][t][stat]:.3f}"]
            rows.append(row)
        write_csv(out, stem, header, rows)

    def build_pair(width):
        fig, axes = plt.subplots(1, 2, figsize=(width, VPP_FIG_H), dpi=220, sharey=True)
        fig.patch.set_facecolor("white")
        for i, (stat, (order, clouds, y_ref)) in enumerate(curves.items()):
            axes[i].set_facecolor("white")
            _vpp_draw(axes[i], order, clouds, y_ref, shared, legend_on=(i == 0),
                      ylabel=f"{label} {stat}")
        # sharey suppresses the right panel's tick labels; put the numbers back so each
        # panel reads on its own -- its y name says mean vs median, so it has to.
        axes[1].tick_params(labelleft=True, labelsize=14)
        fit(fig, pad=0.5, w_pad=1.54)
        return fig, axes

    fig, axes = build_pair(_vpp_pair_width(plot_w, build_pair))
    got = plot_width(fig, axes[0])
    save(fig, out, f"vina_{metric}_per_pocket_pair_{variant}")
    print(f"    pair   shared y {shared[0]:.2f} .. {shared[1]:.2f}   "
          f"panel {got:.3f} in vs standalone {plot_w:.3f} in ({got - plot_w:+.4f})")


@figure("fig-vina-per-pocket-dock", needs=(VPP_BASENAME,))
def draw_vina_per_pocket_dock(out):
    """Vina Dock per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("dock", out)


@figure("fig-vina-per-pocket-score", needs=(VPP_BASENAME,))
def draw_vina_per_pocket_score(out):
    """Vina Score per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("score", out)


@figure("fig-vina-per-pocket-min", needs=(VPP_BASENAME,))
def draw_vina_per_pocket_min(out):
    """Vina Min per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("min", out)

# ════════════════════════════════════════════════════════════════════════════════
# fig-consistency-rigid-{per-size,violin,ecdf} — rigid-fragment RMSD by fragment size
# ════════════════════════════════════════════════════════════════════════════════
# THE METRIC. A rigid fragment has no internal degrees of freedom -- a benzene ring, an
# amide, a fused bicycle each has exactly one correct shape, so a force field has no reason
# to move it. Optimise the molecule with MMFF, cut every rotatable bond, and RMSD each
# fragment against its own optimised self after Kabsch superposition. If it moved, the
# generated geometry was wrong to begin with. Lower is better; the unit is Angstrom.
#
# It is the complement of the two metrics next door. Strain (fig-posecheck) relaxes under a
# 0.1 A position constraint precisely so that local errors wash out and only GLOBAL
# conformational strain is reported -- exactly what this measures instead. PoseBusters
# (fig-posebusters) asks the same local question but only through pass/fail thresholds;
# this is continuous. A model can be good at one and bad at another, and here it is.
#
# WHAT IS ON THE X AXIS IS THE FRAGMENT, NOT THE LIGAND. Every other size-resolved figure
# here plots against heavy atoms in the ligand. This one cannot: the metric has no
# per-molecule value, only per-fragment ones, and fragment size is what drives it (a 2-atom
# fragment is a bond length, a 14-atom fragment is a fused ring system). So the axis, the
# distribution strip and MIN_N all count FRAGMENTS. Do not read a point here against a
# point in fig-posecheck at the same x.
#
# The evaluation itself is voxbind/exps/frozenenc_probes/eval_rigid_fragments.py, which
# writes one `rigid_fragment_results.json` per run root and the crystal ligands'
# `rigid_fragment_reference.json` beside ours; this only draws what that wrote.
RF_RESULTS, RF_REFERENCE = "rigid_fragment_results.json", "rigid_fragment_reference.json"

# THE REFERENCE WINDOW IS NARROWER HERE. The shared `rolled` smooths the crystal ligands'
# share over +-REF_WIN=4 atoms because ligand sizes run 5-45; fragment sizes run 2-25 and
# the distribution is spiky (a third of all fragments have exactly 2 atoms, and 6 --
# benzene -- is the next spike), so a +-4 window would pool a bond length with a fused
# bicycle and smear a strip whose whole point is where the spikes are. The reference
# STATISTIC does not use a window at all -- see _rf_ref_curve.
RF_REF_WIN = 1
# The crystal reference is drawn per exact fragment size, where at least this many crystal
# fragments have that size.
RF_MIN_REF_EXACT = 8

RF_X_LABEL = "Number of heavy atoms in rigid fragment"
RF_XTICK = 2
RF_STATS = (("median", lambda v: float(np.median(v))),
            ("mean", lambda v: float(np.mean(v))))
# TWO BINNINGS, AND THE FINE ONE IS THE DEFAULT READ.
#
# `fine` is one bin per exact fragment size up to 14, then a 15+ tail -- the tail starts
# where the paper's own last bin does. Every model arm holds hundreds to thousands of
# fragments at each of those sizes, so nothing needs pooling, and pooling costs
# something real: the `paper` binning puts benzene (6 atoms, where both VoxBind arms sit at
# crystal quality, 21% of all their fragments) in one bin with the thin and much worse
# 5-atom fragments, which hides the finding.
#
# `paper` is what eval_rigid_fragments.py writes and what 260827 reported -- roughly
# log-spaced, chosen there to match the form the VoxBind paper reports. It is kept so the
# numbers stay directly comparable to that write-up, under its own filename. It is NOT
# kept because the data asks for it: only the 79 crystal ligands are thin enough to need
# pooling (9-15 fragments at sizes 4, 5, 8 and 9), and see RF_MIN_BODY for how that is
# handled instead.
RF_BINNINGS = {
    "fine": ([(n, n + 1) for n in range(2, 15)] + [(15, 10 ** 6)],
             [str(n) for n in range(2, 15)] + ["15+"]),
    "paper": ([(2, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, 10 ** 6)],
              ["2", "3–4", "5–6", "7–9", "10–13", "14+"]),
}
# The stem each binning is saved under. The fine one keeps the plain name: two figures that
# differ in how they group the data must not be able to sit in a folder under one name.
RF_BINNING_STEM = {"fine": "rigid_violin", "paper": "rigid_violin_paperbins"}
# The tables and the run log stay on the paper bins, so they keep lining up with 260827.
RF_PAPER_BINS = RF_BINNINGS["paper"][0]
# A KDE fitted to fewer than this many values is a shape invented from noise. Below it the
# violin is drawn as its glyphs alone -- range, IQR, median, mean -- which are honest at any
# n. In practice this only ever touches the crystal reference.
RF_MIN_BODY = 20
# Below this, nothing is drawn at all. A min-max whisker over three values is not a range,
# it is two points and a line between them, and at a glance it reads exactly like a range
# built from thousands. The 79 crystal ligands hit this at fragment sizes 11-14, where they
# hold two to four fragments; every bin dropped this way is named in the run log.
RF_MIN_DRAW = 5
# The right-hand ECDF panel: where both models leave the crystal ligands behind.
RF_BIG_FRAG = 7
# Below this many labelled decades on the axis, the 2/3/5 minor ticks get labels too.
RF_MIN_DECADE_LABELS = 3


def _rf_log_y_ticks(ax):
    """A log y axis holding one or two decade labels leaves the reader almost no scale to
    read the curve against -- the mean panel spans 0.02 to 0.7 and shows `10^-1`, alone.
    Where that happens, label the 2/3/5 minor ticks as plain decimals as well.

    The test counts the labels the axis will actually SHOW, not the span of the data: a log
    axis autoscales out to the enclosing decades, so a 0.025-0.72 curve reports a span of
    ~1.7 decades while still carrying two labels."""
    lo, hi = ax.get_ylim()
    if len([t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]) >= RF_MIN_DECADE_LABELS:
        return
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10, subs=(2, 3, 5), numticks=20))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="minor", labelsize=12.5, colors=AXIS,
                   length=2.5, width=AXIS_LW)


def _rf_ref_curve(per, xs, f):
    """The crystal ligands at each EXACT fragment size, from xs[0] up to the first size
    with fewer than RF_MIN_REF_EXACT of them.

    NOT the shared reference_curve, and the difference matters. That windows the crystal
    ligands over +-REF_WIN because one ligand per pocket is too thin per LIGAND size. The
    same fix applied to fragment sizes was actively misleading here: the distribution
    spikes at 6 atoms (benzene), where the 33 crystal fragments sit at 0.027 A, and a +-1
    window pools that spike with the thin, worse-scoring 5s and 7s and lifts the plotted
    point to 0.041 -- ABOVE both models, reversing the comparison the line exists to make.

    79 crystal ligands give 294 fragments, which cannot support a per-size curve past ~10
    atoms. So the line stops there rather than being smoothed into reaching further, and
    the pooled 7+ comparison is made in the ECDF figure instead, where every fragment
    counts once.
    """
    out = []
    for a in xs:
        v = per.get(a, ())
        if len(v) < RF_MIN_REF_EXACT:
            break
        out.append(f(v))
    return out + [None] * (len(xs) - len(out))


def _rf_rolled(values, xs):
    """The shared `rolled`, over RF_REF_WIN instead of REF_WIN -- see the note on that
    constant. Written out rather than setting the global, because every other figure in
    this file is drawn in the same process and reads ligand sizes, not fragment sizes."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= RF_REF_WIN])
            for a in xs]


def _rf_contiguous(xs):
    """The leading run of consecutive sizes. x_range keeps every count where all arms clear
    MIN_N, which for LIGAND sizes is a dense range; fragment sizes are not dense out in the
    tail -- 25, 26 and 27 fall under the floor while 28 (a common fused system) clears it --
    and a line drawn from 24 to 28 would cross three counts that were dropped for being too
    thin. Truncating is the same rule the rest of the file follows: no point is invented,
    and nothing is interpolated over a gap."""
    for i, (a, b) in enumerate(zip(xs, xs[1:])):
        if b != a + 1:
            return xs[:i + 1]
    return xs


# ── load ─────────────────────────────────────────────────────────────────────────
def _rf_load(path):
    """{target: {n_mols, n_mmff_failed, fragments:[(size, rmsd), ...]}} for every target
    the run scored. Filtering to the 79 happens in the caller, so the pooled all-pockets
    numbers stay available."""
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    return {t["target"]: t for t in j["per_target"]}


def _rf_pooled(per_target, targets=None):
    """(fragment rows, molecule count, MMFF failure count) over a set of targets."""
    names = sorted(per_target) if targets is None else [t for t in targets
                                                        if t in per_target]
    rows = [tuple(f) for t in names for f in per_target[t]["fragments"]]
    return (rows,
            sum(per_target[t]["n_mols"] for t in names),
            sum(per_target[t]["n_mmff_failed"] for t in names))


def _rf_by_frag_size(rows):
    out = collections.defaultdict(list)
    for n, v in rows:
        out[n].append(v)
    return out


def _rf_scored(root):
    """(per-target rows, connected_only) for a run root, or None if it has not been scored
    over all 79 pockets. An arm mid-evaluation must not become a curve drawn over part of
    the data -- the same rule arms_for applies to the metrics tree."""
    path = os.path.join(root, RF_RESULTS)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    per = {t["target"]: t for t in j["per_target"]}
    if any(t not in per for t in P79):
        return None
    return per, bool(j["summary"].get("connected_only", False))


_RF_CACHE = {}


def _rf_data():
    """(arms scored, {key: {size: [rmsd]}}, the reference's, {key: pooled}, ref pooled).

    Cached for the life of the process, like pose_data -- the three rigid-fragment figures
    read the same ~200k fragments over eight run trees."""
    if "d" not in _RF_CACHE:
        t0 = time.time()
        arm_data, mode, arms = {}, {}, []
        for a in ARMS:
            got = _rf_scored(a[2])
            if got is None:
                continue
            arm_data[a[1]], mode[a[0]] = got
            arms.append(a)
        if not arms:
            raise SystemExit(f"no arm has {RF_RESULTS} over all {len(P79)} pockets — run "
                             f"voxbind/exps/frozenenc_probes/eval_rigid_fragments.py first")
        # Every arm must have been scored the same way. metrics.py drops molecules that are
        # not one connected component, so an arm scored WITH --connected-only and one
        # scored without are counting different molecules, and a figure drawn across both
        # would compare two different questions without saying so.
        if len(set(mode.values())) > 1:
            raise SystemExit("arms disagree on --connected-only: "
                             + ", ".join(f"{k}={v}" for k, v in mode.items())
                             + " — re-run eval_rigid_fragments.py so they match")
        ref_frags = _rf_pooled(_rf_load(os.path.join(REF_ROOT, RF_REFERENCE)), P79)
        p79_frags = {key: _rf_pooled(arm_data[key], P79) for key in arm_data}
        per = {key: _rf_by_frag_size(p79_frags[key][0]) for key in arm_data}
        ref_per = _rf_by_frag_size(ref_frags[0])
        _RF_CACHE["d"] = (arms, per, ref_per, p79_frags, ref_frags)
        n = sum(len(v[0]) for v in p79_frags.values())
        print(f"  [loaded {n:,} rigid fragments over {len(arms)} arms "
              f"in {time.time() - t0:.1f}s]")
    return _RF_CACHE["d"]


def _rf_variants(arms):
    """(name, arms), core first -- the shared `variants` over the arms that were scored."""
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


# ── figure 1: RMSD per fragment size, over the fragment-size distribution ────────
def _rf_size_distribution(ax, xs, arms, per, ref_per):
    """Where each arm puts its fragments, as a share of its own -- the same panel the
    shared size_distribution draws, with the fill dropped once the figure carries more than
    the core arms. Filled steps read well for two or three series; at nine they stack into
    one opaque mass and the arm you are looking for is the one you cannot see. The
    reference keeps a rolled share, over RF_REF_WIN."""
    fill = len(arms) <= len(CORE)
    for lab, key, _ in arms:
        pct, _ = share(per[key], xs)
        col = color(lab)
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        if fill:
            ax.fill_between(xs, pct, step="mid", color=col, alpha=DIST_FILL, lw=0,
                            zorder=2)
    raw, _ = share(ref_per, xs)
    ax.plot(xs, _rf_rolled(raw, xs), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    ax.set_ylim(bottom=0)


def _rf_per_size_fig(out, xs, arms, variant, stat, f, per, ref_per):
    """One statistic, one panel, one file -- the rule the 3-line figures set. The median is
    the one to read: the RMSD distribution inside a size is heavy-tailed (a handful of
    fragments the force field rebuilds outright sit an order of magnitude above the bulk),
    so the mean tracks that tail rather than the typical fragment. Both are drawn, apart,
    and the filename says which."""
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    top.plot(xs, _rf_ref_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
             ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, model_curve(per[key], xs, f), color=color(lab), lw=MODEL_LW,
                 zorder=5, solid_capstyle="round")
    top.set_yscale("log")
    # Two lines, as the strain figures do it: on one line the label is taller than this
    # panel and runs into the distribution strip's own label.
    furniture(top, ylabel=f"Fragment RMSD {stat}\n(Å)",
              xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=RF_XTICK)
    _rf_log_y_ticks(top)
    # Lower right, not the house default upper left: every curve climbs to the right, so
    # upper left is where TargetDiff's rise is and the opaque legend box would cover it.
    legend(top, arm_handles(arms), loc="lower right",
           ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)

    _rf_size_distribution(bot, xs, arms, per, ref_per)
    furniture(bot, ylabel="% of fragments", xlabel=RF_X_LABEL,
              xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=RF_XTICK)
    fig.align_ylabels((top, bot))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, out, f"rigid_per_size_{stat}_{variant}")


# ── the run log's pooled numbers ────────────────────────────────────────────────
# An arm gets a standardised number only if the sizes it clears MIN_N at account for at
# least this much of the common mix. Otherwise the re-weighting is not a correction, it is
# a different question answered on whatever sizes happened to survive.
RF_STD_COVER = 0.90


def _rf_std_coverage(per, std_w):
    """The share of the common size mix this arm can answer at MIN_N."""
    total = sum(std_w.values())
    return sum(w for n, w in std_w.items()
               if len(per.get(n, ())) >= MIN_N) / total if total else 0.0


def _rf_standardised(per, std_w):
    """The arm's median at each fragment size, averaged over the COMMON size mix.

    The pooled median cannot be compared across arms and the numbers here are the reason
    why, not a hypothetical: DecompDiff's pooled median (0.043) sits below Ours (0.047)
    while being worse at every fragment size from 5 atoms up -- it simply draws a different
    mix of sizes, and small fragments score better. This re-weights every arm onto one mix,
    so the summary number answers "how good is this arm's geometry" instead of "what sizes
    does this arm happen to make". It is an average of medians, not a median: a median does
    not re-weight."""
    num = den = 0.0
    for n, w in std_w.items():
        v = per.get(n, ())
        if len(v) >= MIN_N:
            num += w * float(np.median(v))
            den += w
    total = sum(std_w.values())
    # The 79 crystal ligands are the case this guard exists for: they clear MIN_N at four
    # fragment sizes out of nine, and those four carry most of the common mix's weight and
    # are the sizes crystal geometry is best at. A number built from them would read as a
    # like-for-like summary and would not be one.
    return num / den if den and den / total >= RF_STD_COVER else None


def _rf_block(rows, n_mols, n_failed, std_w):
    """One arm's pooled numbers for the run log, over the paper bins."""
    v = [r[1] for r in rows]
    per = _rf_by_frag_size(rows)
    std = _rf_standardised(per, std_w) if rows else None
    binned = {}
    for lo, hi in RF_PAPER_BINS:
        sel = [r[1] for r in rows if lo <= r[0] < hi]
        if sel:
            name = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
            binned[name] = {"n": len(sel), "median": round(st.median(sel), 4),
                            "mean": round(float(np.mean(sel)), 4)}
    return {
        "n_molecules": n_mols,
        "mmff_failure_rate_pct": round(100 * n_failed / n_mols, 2) if n_mols else None,
        "n_fragments": len(rows),
        "rmsd_median": round(st.median(v), 4) if v else None,
        "rmsd_median_size_standardised": None if std is None else round(std, 4),
        "size_standardised_coverage": round(_rf_std_coverage(per, std_w), 3)
        if rows else None,
        "rmsd_mean": round(float(np.mean(v)), 4) if v else None,
        "by_fragment_size": binned,
    }


@figure("fig-consistency-rigid-per-size",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_per_size(out):
    """Rigid-fragment RMSD against fragment size, mean and median, each over the arms'
    fragment-size distribution."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    ranges = {}
    for variant, vs in _rf_variants(arms):
        xs = _rf_contiguous(x_range(per, vs))
        ranges[variant] = xs
        for stat, f in RF_STATS:
            _rf_per_size_fig(out, xs, vs, variant, stat, f, per, ref_per)

    xs = ranges["all"]
    fields = [f"rmsd_{stat}" for stat, _ in RF_STATS]
    curves = {lab: {"n": [len(per[key].get(a, ())) for a in xs],
                    **{f"rmsd_{stat}": model_curve(per[key], xs, f)
                       for stat, f in RF_STATS}}
              for lab, key, _ in arms}
    # The reference is per exact fragment size, null past the first size holding fewer than
    # RF_MIN_REF_EXACT crystal fragments; nothing is windowed.
    curves[REF_LABEL] = {"n": [len(ref_per.get(a, ())) for a in xs],
                         **{f"rmsd_{stat}": _rf_ref_curve(ref_per, xs, f)
                            for stat, f in RF_STATS}}
    write_csv(out, "rigid_fragment_per_size",
              ["arm", "fragment_atoms", "n"] + fields,
              [[arm, a, d["n"][i]]
               + ["" if d[f][i] is None else round(d[f][i], 4) for f in fields]
               for arm, d in curves.items() for i, a in enumerate(xs)])

    # One common fragment-size distribution, every arm pooled, for the size-standardised
    # summary. Sizes where an arm holds fewer than MIN_N fragments are dropped from ITS
    # average rather than extrapolated.
    std_w = collections.Counter(n for key in per for n, v in per[key].items() for _ in v)
    blocks = {lab: _rf_block(*p79_frags[key], std_w) for lab, key, _ in arms}
    blocks[REF_LABEL] = _rf_block(*ref_frags, std_w)

    print(f"  {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs_[0]}-{xs_[-1]}" for v, xs_ in ranges.items())
          + f" (fragment sizes where every drawn arm has >={MIN_N} fragments,\n"
          + "    truncated at the first gap: "
          + ", ".join(f"{v} drops {sorted(set(x_range(per, a)) - set(ranges[v]))}"
                      for v, a in _rf_variants(arms)) + ")\n")
    print(f"  {'arm':16s} {'frags':>7s} {'med':>7s} {'std med':>8s} {'mean':>7s} "
          f"{'mols':>6s} {'MMFF fail':>10s}")
    for lab, r in blocks.items():
        std = r["rmsd_median_size_standardised"]
        print(f"  {lab:16s} {r['n_fragments']:7d} {r['rmsd_median']:7.4f} "
              f"{('%8.4f' % std) if std is not None else '   n/a  '} "
              f"{r['rmsd_mean']:7.4f} {r['n_molecules']:6d} "
              f"{r['mmff_failure_rate_pct']:9.1f}%"
              + ("" if std is not None else
                 f"   (covers {100 * r['size_standardised_coverage']:.0f}% of the mix)"))

    print(f"\n  {'fragment size':>14s} " + "".join(f"{l:>18s}" for l in blocks))
    for name in blocks[REF_LABEL]["by_fragment_size"]:
        cells = ""
        for r in blocks.values():
            b = r["by_fragment_size"].get(name)
            cells += f"{b['median']:10.4f} ({b['n']:>5d})" if b else f"{'':>18s}"
        print(f"  {name:>14s} {cells}")


# ── figure 2: the paper's own form ──────────────────────────────────────────────
# The violin y axis is floored here. The 2-atom bin reaches 8.6e-08 -- a bond MMFF did not
# move at all -- which is seven decades below that bin's median, and an axis that reached it
# would squash every violin in the figure into a line. Values below the floor are drawn AT
# it and the share that hits it is reported in the run log and the README.
# The floor sits where the low tail stops carrying anything: 0.97% of the 201,568 fragments
# across all nine series fall below 1e-3, and they are almost all 2-atom fragments MMFF did
# not move at all. Dropping to 1e-4 to reach them costs a whole decade of panel height and
# buys 0.09% more data. The share drawn AT the floor is printed on every build.
RF_VIOLIN_FLOOR = 1e-3
RF_VIOLIN_TICKS = [1e-3, 1e-2, 1e-1, 1e0]
# Inches per violin where there is room, and the floor below which one stops being a shape.
RF_PER_VIOLIN_IN, RF_MIN_VIOLIN_IN = 0.40, 0.24
# The figure never grows past this, in inches, and never past RF_MAX_ROWS rows. Past the
# width it WRAPS instead of growing sideways: nine arms over fourteen sizes is 126 violins,
# and in one row at a readable width that is a figure fifty inches across. TWO ROWS IS THE
# CAP -- a third makes the reader hunt for a size across three bands, and the width needed
# to keep a violin readable in two rows is the price of that.
RF_MAX_FIG_W, RF_MAX_ROWS = 18.0, 2


def _rf_bin_values(rows, binning="paper"):
    """{bin label: np.array of RMSDs}."""
    bins, labels = RF_BINNINGS[binning]
    return {lab: np.array([r[1] for r in rows if lo <= r[0] < hi])
            for (lo, hi), lab in zip(bins, labels)}


def _rf_violin(out, arms, variant, binning, p79_frags, ref_frags):
    """Fragment-size bins drawn as violins rather than bars.

    A bar shows one number per bin. The distribution inside a bin is the thing that decides
    whether that number means anything, and here it is wide and right-skewed -- so the
    violin carries the KDE plus, per arm and bin, the full min-max range, the interquartile
    box, the median and the mean. Mean sits above median everywhere in this data, which is
    the skew made visible and the reason the median is the number reported.

    KDE IN LOG SPACE, AXIS RELABELLED. Fragment RMSD spans four decades and is strongly
    right-skewed, so a linear violin is a spike on the floor. Fitting the KDE to log10(RMSD)
    and relabelling the ticks with real values keeps the density in the space the data
    actually lives in -- matplotlib's violinplot fits in data space, so setting a log scale
    afterwards would warp the drawn shape instead. Same trick, same reason, as the clash
    violins in fig-posecheck.
    """
    labels_all = RF_BINNINGS[binning][1]
    series = [(REF_LABEL, REF_COLOR, ref_frags[0])] + \
             [(lab, color(lab), p79_frags[key][0]) for lab, key, _ in arms]
    per = [(lab, col, _rf_bin_values(rows, binning)) for lab, col, rows in series]
    labels = [l for l in labels_all if any(len(v[l]) for _, _, v in per)]

    # As many rows as it takes for a violin to stay at least RF_MIN_VIOLIN_IN wide inside
    # RF_MAX_FIG_W. The rows split the SIZE BINS, never the arms: every arm has to stay
    # comparable within a bin, and a reader comparing two arms must never have to look at
    # two panels to do it.
    nrow = 1
    while (RF_MAX_FIG_W - 2.3) / (len(series) * -(-len(labels) // nrow)) < RF_MIN_VIOLIN_IN \
            and nrow < min(RF_MAX_ROWS, len(labels)):
        nrow += 1
    step = -(-len(labels) // nrow)
    rows_of_labels = [labels[i:i + step] for i in range(0, len(labels), step)]
    slots = max(len(r) for r in rows_of_labels)
    slot = 0.86 / len(series)
    t = lambda v: np.log10(np.clip(v, RF_VIOLIN_FLOOR, None))

    # Width is set by the widest ROW, so violins are the same size in one row or several.
    width = min(RF_MAX_FIG_W, max(FIG_W * 1.52,
                                  2.3 + RF_PER_VIOLIN_IN * len(series) * slots))
    # EACH ROW IS CROPPED TO THE DECADES ITS OWN BINS REACH, AND PAYS FOR THEM IN HEIGHT.
    # The rows split the size bins and RMSD grows with fragment size, so the 9+ atom row
    # never comes within a decade of the 1e-3 floor the 2-atom fragments sit on: a shared
    # bottom spends a third of that panel on a band holding nothing. Its floor is instead
    # the enclosing decade of its own smallest drawn fragment -- 1e-2 for sizes 9-15+ --
    # and the panel is shortened by exactly the decade it gave up, so A DECADE IS THE SAME
    # PHYSICAL HEIGHT IN EVERY ROW. That, not a shared range, is what keeps the rows
    # comparable, and it is what stops a cropped row from silently stretching its violins.
    # The TOP is shared and uncropped, so nothing is ever cut off the tall end.
    top = max(t(v).max() for _, _, vals in per for v in vals.values() if len(v)) + 0.30
    bottoms = [max(np.log10(RF_VIOLIN_FLOOR),
                   np.floor(min((t(v).min() for _, _, vals in per for l, v in vals.items()
                                 if l in labs and len(v) >= RF_MIN_DRAW),
                                default=np.log10(RF_VIOLIN_FLOOR)))) - 0.12
               for labs in rows_of_labels]
    spans = [top - b for b in bottoms]
    row_h = PANEL_H * (1.30 if nrow == 1 else 1.02)
    fig, axes = plt.subplots(nrow, 1, dpi=220, squeeze=False,
                             figsize=(width, row_h * sum(spans) / max(spans)),
                             gridspec_kw={"height_ratios": spans})
    axes = [a for row in axes for a in row]
    fig.patch.set_facecolor("white")

    clipped = {lab: 0 for lab, _, _ in per}
    thin = {lab: [] for lab, _, _ in per}
    skipped = {lab: [] for lab, _, _ in per}
    row_ticks = []
    for ax, labs, bottom in zip(axes, rows_of_labels, bottoms):
        ax.set_facecolor("white")
        idx = np.arange(len(labs))
        for i, (lab, col, vals) in enumerate(per):
            pos = idx - 0.43 + slot * (i + 0.5)
            keep = [(l, x, vals[l]) for x, l in zip(pos, labs)
                    if len(vals[l]) >= RF_MIN_DRAW]
            skipped[lab] += [l for l in labs if 0 < len(vals[l]) < RF_MIN_DRAW]
            clipped[lab] += sum(int((v < RF_VIOLIN_FLOOR).sum()) for _, _, v in keep)
            body_at = [(x, v) for _, x, v in keep if len(v) >= RF_MIN_BODY]
            thin[lab] += [l for l, _, v in keep if len(v) < RF_MIN_BODY]
            parts = ax.violinplot([t(v) for _, v in body_at],
                                  positions=[x for x, _ in body_at],
                                  widths=slot * 0.92, showextrema=False) if body_at \
                else {"bodies": []}
            for body in parts["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.55)
                body.set_edgecolor(col)
                body.set_linewidth(1.1)
                body.set_zorder(2)
                if lab == REF_LABEL:      # the benchmark, as in every other figure here
                    body.set_hatch("///")
            for _, x, v in keep:
                lo, q1, med, q3, hi = np.percentile(v, [0, 25, 50, 75, 100])
                ax.vlines(x, t(lo), t(hi), color=INK, lw=0.9, zorder=3)
                ax.hlines([t(lo), t(hi)], x - slot * 0.16, x + slot * 0.16, color=INK,
                          lw=0.9, zorder=3)
                ax.vlines(x, t(q1), t(q3), color=INK, lw=4.0, zorder=4)
                ax.plot(x, t(med), "o", mfc="white", mec=INK, mew=0.7, ms=4.4, zorder=6)
                ax.plot(x, t(v.mean()), "D", mfc=INK, mec="white", mew=0.8, ms=3.8,
                        zorder=6)

        last = ax is axes[-1]
        furniture(ax, ylabel="Fragment RMSD (Å)",
                  xlabel="Rigid fragment size (heavy atoms)" if last else None,
                  xloc=None)
        ax.grid(False, axis="x")   # the groups are the categories; an x rule is only ink
        # The axis is linear in log10(RMSD) -- see the docstring -- so the ticks are placed
        # by hand and labelled as powers of ten, which is what the axis actually is and
        # what the other log-scaled figures here show.
        ticks = [v for v in RF_VIOLIN_TICKS if bottom <= np.log10(v) <= top]
        row_ticks.append(ticks[0])
        ax.set_yticks([np.log10(v) for v in ticks])
        ax.set_yticklabels([f"$10^{{{int(round(np.log10(v)))}}}$" for v in ticks])
        ax.set_ylim(bottom, top)
        ax.set_xticks(idx)
        ax.set_xticklabels(labs)
        # Every row spans the same number of SLOTS even when its last one is empty, so a
        # violin is the same width in both and the two rows read as one axis.
        ax.set_xlim(-0.58, slots - 0.42)

    swatches = [plt.Rectangle((0, 0), 1, 1, facecolor=col, alpha=0.62, edgecolor=col,
                              hatch="///" if lab == REF_LABEL else None,
                              label=lab if lab == REF_LABEL else display(lab))
                for lab, col, _ in per]
    glyphs = [
        Line2D([], [], color=INK, lw=0.9, label="min–max"),
        Line2D([], [], color=INK, lw=4.0, label="IQR"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, mew=0.7, ms=4.4,
               label="median"),
        Line2D([], [], ls="none", marker="D", mfc=INK, mec="white", mew=0.8, ms=3.8,
               label="mean"),
    ]
    # BOTH KEYS SIT ALONG THE BOTTOM OF THE LAST ROW -- methods right, glyphs left. The
    # violins climb to the right and upward with fragment size, so the band under the last
    # row is the emptiest strip in the figure; splitting the two keys to opposite ends of
    # it keeps each clear of the other and of the data. The methods go in a square block
    # (three columns for nine series) rather than a tall single column, which would run up
    # into the violins above it. Cropping that row to its own decades takes most of the
    # depth out of the strip, so the glyph key goes in ONE ROW of four there -- it is the
    # key that can afford to be wide, since the figure is eighteen inches across and the
    # left of the strip is the emptiest part of it.
    ax = axes[-1]
    ncol = 3 if len(series) > 5 else 1
    meth = legend(ax, swatches, loc="lower right", ncol=ncol,
                  fontsize=11.5 if len(series) <= 5 else 10.0)
    ax.add_artist(meth)
    keys = legend(ax, glyphs, loc="lower left", ncol=4 if nrow > 1 else 2, fontsize=10.5)
    if nrow > 1:
        fig.align_ylabels(axes)
    relayout = lambda: fit(fig, pad=0.5, **({"h_pad": H_PAD} if nrow > 1 else {}))
    relayout()

    # NEITHER KEY MAY COVER A WHISKER, and cropping the last row took most of the depth out
    # of the strip they sit in. So the strip is MEASURED, not assumed: lay the figure out
    # once, ask each key how tall it actually came out, convert that to decades through the
    # panel's own scale, and hand the row back exactly the shortfall -- over the half of
    # the row each key sits in, since the two halves bottom out at different places. What
    # is added is blank axis BELOW the lowest tick, so 1e-2 stays the bottom label; and
    # nothing is added when the data already leaves room, which is the single-row case.
    labs = rows_of_labels[-1]
    half = len(labs) // 2
    short = 0.0
    for leg, sel in ((meth, labs[half:]), (keys, labs[:len(labs) - half])):
        per_in = (top - bottoms[-1]) / (ax.get_window_extent().height / fig.dpi)
        lo = min((t(v).min() for _, _, vals in per for l, v in vals.items()
                  if l in sel and len(v) >= RF_MIN_DRAW), default=top)
        short = max(short, (leg.get_window_extent().height / fig.dpi + 0.07) * per_in
                    - (lo - bottoms[-1]))
    if short > 0.01:
        bottoms[-1] -= short
        spans[-1] = top - bottoms[-1]
        fig.set_size_inches(width, row_h * sum(spans) / max(spans))
        ax.get_subplotspec().get_gridspec().set_height_ratios(spans)
        ax.set_ylim(bottoms[-1], top)
        relayout()
    if nrow > 1:
        # A row that does not start at the same place as the one above it has to say so.
        print(f"  {RF_BINNING_STEM[binning]}_{variant}: rows carry their own floor — "
              + "; ".join(f"sizes {labs_[0]}-{labs_[-1]} from {tick:g} A"
                          for labs_, tick in zip(rows_of_labels, row_ticks))
              + " (a decade is the same height in each)")
    save(fig, out, f"{RF_BINNING_STEM[binning]}_{variant}")
    return clipped, thin, skipped


@figure("fig-consistency-rigid-violin",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_violin(out):
    """The RMSD distribution inside each fragment-size bin, per arm — over the fine
    per-size binning and over the paper's coarser one."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    clipped, thin, skipped = {}, {}, {}
    for variant, vs in _rf_variants(arms):
        for binning in RF_BINNINGS:
            clipped[variant], thin[(variant, binning)], skipped[(variant, binning)] \
                = _rf_violin(out, vs, variant, binning, p79_frags, ref_frags)

    # the violin figure's numbers, flat -- every glyph it draws, per arm and bin
    rows = []
    for binning in RF_BINNINGS:
        for lab, key, _ in list(arms) + [(REF_LABEL, None, None)]:
            frags = ref_frags[0] if key is None else p79_frags[key][0]
            for name, v in _rf_bin_values(frags, binning).items():
                if not len(v):
                    continue
                lo, q1, med, q3, hi = np.percentile(v, [0, 25, 50, 75, 100])
                rows.append([binning, lab, name, len(v)]
                            + [round(float(x), 4) for x in (lo, q1, med, v.mean(), q3, hi)])
    write_csv(out, "rigid_fragment_by_bin",
              ["binning", "arm", "fragment_size_bin", "n", "min", "q25", "median",
               "mean", "q75", "max"], rows)

    hit = clipped["all"]
    if any(hit.values()):
        tot = {lab: len(ref_frags[0] if lab == REF_LABEL
                        else p79_frags[dict((l, k) for l, k, _ in arms)[lab]][0])
               for lab in hit}
        print(f"\n  rigid_violin: fragments drawn AT the {RF_VIOLIN_FLOOR:g} A axis floor "
              f"(all in the 2-atom bin): "
              + ", ".join(f"{lab} {n} of {tot[lab]} ({100 * n / tot[lab]:.2f}%)"
                          for lab, n in hit.items() if n))
    for binning in RF_BINNINGS:
        gone = {lab: b for lab, b in skipped[("all", binning)].items() if b}
        if gone:
            print(f"  {RF_BINNING_STEM[binning]}: fewer than {RF_MIN_DRAW} fragments, so "
                  f"not drawn at all: " + "; ".join(f"{lab} at {', '.join(b)}"
                                                    for lab, b in gone.items()))
        lean = {lab: b for lab, b in thin[("all", binning)].items() if b}
        if lean:
            print(f"  {RF_BINNING_STEM[binning]}: fewer than {RF_MIN_BODY} fragments, so "
                  f"drawn as glyphs with no KDE body: "
                  + "; ".join(f"{lab} at {', '.join(b)}" for lab, b in lean.items()))


# ── figure 3: the distribution itself ───────────────────────────────────────────
@figure("fig-consistency-rigid-ecdf",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_ecdf(out):
    """Two panels on one shared y, the pose ECDF-pair layout. Left, every fragment; right,
    only the 7+ atom fragments. The split is there because the left panel is dominated by
    the 2- and 3-atom fragments -- more than half of every arm's total -- whose RMSD is a
    bond length and where all four lines sit on top of each other. The right panel is the
    same curve over the fragments that actually carry ring and conjugation geometry."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    for variant, vs in _rf_variants(arms):
        fig, (left, right) = plt.subplots(1, 2, figsize=(FIG_W * 1.52, PANEL_H),
                                          dpi=220, sharey=True)
        fig.patch.set_facecolor("white")
        for ax in (left, right):
            ax.set_facecolor("white")

        def ecdf(ax, keep):
            def draw(rows, col, lw, ls):
                v = np.sort(np.clip([r[1] for r in rows if keep(r[0])], 1e-4, None))
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=lw, ls=ls,
                        zorder=4 if ls != "-" else 5)
            for lab, key, _ in vs:
                draw(p79_frags[key][0], color(lab), MODEL_LW, "-")
            draw(ref_frags[0], REF_COLOR, REF_LW, DASH)

        ecdf(left, lambda n: True)
        left.set_xscale("log")
        left.set_xlim(1e-3, 3e0)
        furniture(left, ylabel="Cumulative share", xlabel="Fragment RMSD (Å)", xloc=None)
        left.set_ylim(0, 1.0)
        legend(left, arm_handles(vs), loc="upper left",
               ncol=2 if len(vs) > 4 else 1, fontsize=11.5 if len(vs) <= 4 else 10.0)

        ecdf(right, lambda n: n >= RF_BIG_FRAG)
        right.set_xscale("log")
        right.set_xlim(1e-3, 3e0)
        furniture(right, ylabel=None,
                  xlabel=f"Fragment RMSD (Å), {RF_BIG_FRAG}+ atom fragments", xloc=None)
        fit(fig, pad=0.5)
        save(fig, out, f"rigid_ecdf_pair_{variant}")


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
    ("HBAcceptor",  "H-bonds accepted by the ligand", "boxen"),
    ("HBDonor",     "H-bonds donated by the ligand",  "boxen"),
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
IX_DISPLAY = {REF_LABEL: "Reference", "Ours": r"\textsc{CoDE}", "CoDE": r"\textsc{CoDE}"}

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
    return IX_DISPLAY.get(label, label)


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
                    va="top", fontsize=10, color=AXIS, linespacing=1.3, zorder=6)


def _interaction_violin_panel(ax, vals, cols, ylabel, caption=True):
    ax.set_facecolor("white")
    parts = ax.violinplot(vals, showextrema=False, widths=IX_VIOLIN_W,
                          bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
    for body, col in zip(parts["bodies"], cols):
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
            # the alpha goes on the FACE only, as a fourth channel, rather than on the
            # patch -- a patch alpha would take the outline down with it, and the outline
            # is what makes a block a block.
            face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                + (IX_BODY_ALPHA,)
            ax.add_patch(IX_RECTANGLE((i - w / 2, lo), w, hi - lo, zorder=3 + d,
                                      facecolor=face,
                                      edgecolor=AXIS, linewidth=IX_BOXEN_EDGE_LW))
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
    # The legend strip is RESERVED with tight_layout's rect and the legend drawn into it
    # afterwards: tight_layout does not measure a figure-level legend, so laying the axes
    # out first and adding it second would print it over the x spine.
    fit(fig, pad=0.5, w_pad=2.2, rect=(0, IX_LEGEND_H / IX_FIG_H, 1, 1))
    _interaction_legend(fig, names, cols, form)
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
            ax.add_patch(IX_RECTANGLE((x - w / 2, lo), w, hi - lo, zorder=3 + d,
                                      facecolor=face, edgecolor=AXIS,
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
    fit(fig, pad=0.5, w_pad=2.2, rect=(0, IX_LEGEND_H / IX_FIG_H, 1, 1))
    _interaction_pair_legend(fig, form)
    # `core` is in the filename because it is the only variant this figure has and the
    # family's rule is that the variant is never only in the content: the redocking covers
    # the core three arms, since the published baselines' poses would each need their own
    # ~700 docks and the comparison this section is making is between ours and VoxBind's.
    save(fig, out, f"interaction_pair_{part}_core")


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


# ════════════════════════════════════════════════════════════════════════════════
# fig-molweight-{distribution,ecdf} — the size prior every other figure is read against
# ════════════════════════════════════════════════════════════════════════════════
# NOT a quality metric -- nothing here says an arm is better -- it is the size prior every
# quality metric in 260910 is read against. Vina score grows with molecule size, strain
# grows with size, rigid-fragment RMSD grows with fragment size. So "which arm wins" is
# only meaningful once you know whether the arms are drawing from the same weight
# distribution, and they are not.
#
# MOLECULAR WEIGHT IS READ STRAIGHT FROM `target_*/samples.sdf`, through RDKit's
# `Descriptors.MolWt` (average mass, implicit hydrogens included).
#
# It used to come from the SMILES in each target's `metrics.json`, and that had to change
# to cover the five published baselines: their sample dirs are staged and complete, but
# their `metrics.json` files are written by the PoseBusters scoring run and appear as it
# finishes each pocket. Keying a size prior to the progress of a scoring run is the wrong
# dependency -- the molecules exist either way -- so this reads the molecules. The two
# agree: over the three arms that have both, SDF and SMILES give the same weight for every
# molecule (the load prints the check), because the recorded SMILES was derived from the
# same SDF entry.
#
# THE ARMS AND THE POCKET SET ARE THE SHARED ONES -- the 79 electron-density pockets. Arms
# hold different numbers of molecules (7,287 for TargetDiff against 7,888 for VoxBind)
# because that is what each sampled; nothing is filtered here. An arm is drawn only if it
# has a samples.sdf for all 79 pockets, which is the same rule arms_for() applies to
# metrics -- an arm mid-stage must not become a curve over part of the data.
MW_X_LABEL = "Molecular weight (Da)"
MW_BIN = 20                  # Da; ~52 bins over the occupied range
MW_XMAX = 800                # the plotted range; the share above it is named in the log
MW_XTICK = 100
# The crystal reference is 79 molecules over ~35 occupied bins -- about two per bin, so its
# raw histogram is a picket fence of 2.5%-tall spikes that says nothing. It is rolled over
# +-MW_REF_ROLL bins and AVERAGED, which keeps it on the same per-bin scale as the models'
# shares, exactly as size_distribution() rolls the reference over ligand size.
MW_REF_ROLL = 2
MW_EDGES = np.arange(0, MW_XMAX + MW_BIN, MW_BIN)
MW_CENTRES = MW_EDGES[:-1] + MW_BIN / 2


def _mw_sdf_weights(paths):
    """(molecular weights, unsanitisable entries, disconnected entries dropped).

    DISCONNECTED MOLECULES ARE DROPPED, because notebook/webapp/metrics.py drops them
    before scoring anything else -- so keeping them would put this figure on a different
    molecule set than the strain, clash and PoseBusters figures for the same run. It
    matters for exactly one arm: TargetDiff emits 511 of 7,798 (6.6%) over the 79 pockets
    and every other arm emits none. Their weight is also not a ligand's weight -- it is
    the sum of two or more pieces that never bonded.

    RDKit is imported HERE, not at the top of draw.py: it is the only figure-specific
    dependency in the file, and a top-level import would make every other figure -- and
    --list -- refuse to run on a box that has no rdkit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    out, bad, disc = [], 0, 0
    for path in paths:
        for m in Chem.SDMolSupplier(path):
            if m is None:
                bad += 1
                continue
            if len(Chem.GetMolFrags(m)) > 1:
                disc += 1
                continue
            out.append(Descriptors.MolWt(m))
    return np.asarray(out), bad, disc


def _mw_samples(root, targets=None):
    """The samples.sdf under `root`, over `targets` or over every target it holds."""
    names = sorted(d for d in os.listdir(root) if d.startswith("target_")) \
        if targets is None else targets
    paths = [os.path.join(root, t, "samples.sdf") for t in names]
    return [p for p in paths if os.path.exists(p)]


def _mw_reference_paths():
    """The deposited ligand in each of the 79 pockets: the one SDF in the target dir that
    is not samples.sdf. Every arm that has one carries the same molecule, so this reads
    ours -- the staged baseline dirs hold no reference at all."""
    out = []
    for t in P79:
        d = os.path.join(REF_ROOT, t)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".sdf") and f != "samples.sdf"]
    return out


def _mw_share(v):
    """Percent of a set's molecules in each bin. Molecules beyond MW_XMAX are NOT folded
    into the last bin -- that would put a spike where the data has a tail -- so the shares
    sum to slightly under 100 and the remainder is reported."""
    counts, _ = np.histogram(v, bins=MW_EDGES)
    return 100 * counts / len(v), counts


def _mw_rolled(pct):
    """The reference's shares under a centred +-MW_REF_ROLL-bin window, averaged."""
    k = 2 * MW_REF_ROLL + 1
    pad = np.pad(pct, MW_REF_ROLL, mode="constant")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _mw_block(v, bad=0, disc=0):
    """n, centre, spread and tails for one set -- what the load summary prints."""
    return {
        "n_molecules": int(len(v)), "n_sdf_unsanitisable": int(bad),
        "n_disconnected_dropped": int(disc),
        "mw_mean": round(float(v.mean()), 1),
        "mw_median": round(float(np.median(v)), 1),
        "mw_q25": round(float(np.percentile(v, 25)), 1),
        "mw_q75": round(float(np.percentile(v, 75)), 1),
        "mw_p1": round(float(np.percentile(v, 1)), 1),
        "mw_p99": round(float(np.percentile(v, 99)), 1),
        "mw_min": round(float(v.min()), 1), "mw_max": round(float(v.max()), 1),
        "pct_over_500": round(100 * float((v > 500).mean()), 2),
        f"pct_over_{MW_XMAX}": round(100 * float((v > MW_XMAX).mean()), 2),
    }


def _mw_smiles_crosscheck(arms, mw):
    """Molecules whose recorded SMILES gives a different weight than their SDF entry, over
    the arms that have metrics.json. Reported, never silently trusted: this builder changed
    source and the check is what says the change was free."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _, p79_rows, _ = pose_data()
    out = {}
    for lab, key, root in arms:
        smi = [r["smi"] for r in p79_rows[key] if r.get("smi")]
        if not smi:
            continue
        sdf = mw[key]
        if len(smi) != len(sdf):
            out[lab] = (f"{len(smi)} SMILES vs {len(sdf)} SDF — not comparable; "
                        f"metrics.json is still being written for this arm")
            continue
        mism = sum(1 for a, b in zip(smi, sdf)
                   if abs(Descriptors.MolWt(Chem.MolFromSmiles(a)) - b) > 0.05)
        out[lab] = f"{mism} of {len(sdf)} differ"
    return out


_MW_CACHE = {}


def _mw_data():
    """(arms that have samples over all 79 pockets, {key: weights}, reference weights).

    Cached for the life of the process: the distribution and the ECDF are the same ~60k
    molecules read out of ~630 SDF files, and drawing both used to mean reading them twice.
    The per-arm summary and the SMILES cross-check print from HERE rather than from a draw
    function, so they appear once per run however many of the two figures are asked for.

    The original also built a weights-over-EVERY-pocket set; it fed only mw_summary.json,
    which the recompute script still owns, so it is not built here."""
    if "d" not in _MW_CACHE:
        t0 = time.time()
        # An arm is in only if it holds a samples.sdf for every one of the 79 pockets.
        arms = [a for a in ARMS if len(_mw_samples(a[2], P79)) == len(P79)]
        missing = [a[0] for a in ARMS if a not in arms]
        mw, bad, disc = {}, {}, {}
        for _, key, root in arms:
            mw[key], bad[key], disc[key] = _mw_sdf_weights(_mw_samples(root, P79))
        mw_ref, bad_ref, _ = _mw_sdf_weights(_mw_reference_paths())
        _MW_CACHE["d"] = (arms, mw, mw_ref)

        n = sum(len(v) for v in mw.values())
        print(f"  [read {n:,} molecules from {len(arms)} arms' samples.sdf "
              f"in {time.time()-t0:.1f}s]")
        print(f"  {len(P79)} pockets · molecular weight from target_*/samples.sdf, "
              f"single-component molecules only · {MW_BIN} Da bins to {MW_XMAX} Da\n")
        print(f"  {'arm':16s} {'mols':>6s} {'median':>8s} {'mean':>7s} {'IQR':>15s} "
              f"{'>500 Da':>8s} {f'>{MW_XMAX} Da':>9s} {'disconn.':>9s}")
        blocks = {lab: _mw_block(mw[key], bad[key], disc[key]) for lab, key, _ in arms}
        blocks[REF_LABEL] = _mw_block(mw_ref, bad_ref)
        for lab in [l for l, _, _ in arms] + [REF_LABEL]:
            b = blocks[lab]
            print(f"  {lab:16s} {b['n_molecules']:6d} {b['mw_median']:8.1f} "
                  f"{b['mw_mean']:7.1f} {b['mw_q25']:7.1f}–{b['mw_q75']:<7.1f} "
                  f"{b['pct_over_500']:7.2f}% {b[f'pct_over_{MW_XMAX}']:8.2f}% "
                  f"{b['n_disconnected_dropped']:9d}")
        if missing:
            print(f"\n  not drawn — no samples.sdf over all {len(P79)} pockets: "
                  + ", ".join(missing))
        print("\n  SDF vs recorded SMILES, molecules whose weight differs by >0.05 Da:")
        for lab, msg in _mw_smiles_crosscheck(arms, mw).items():
            print(f"    {lab:16s} {msg}")
    return _MW_CACHE["d"]


def _mw_variants(arms):
    """(name, arms) for each figure variant, core first -- variants() over the arms that
    have samples, which is a different list from the arms that have a given metric."""
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


@figure("fig-molweight-distribution", needs=("target_*/samples.sdf",))
def draw_molweight_distribution(out):
    """Share of an arm's molecules per 20 Da bin, core and all arms."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Filled steps read well for the two or three arms of `core`; at nine they stack
        # into one opaque mass and the arm you are looking for is the one you cannot see.
        # So the fill is dropped once the figure carries more than the core arms, and the
        # step lines alone do the work -- the same reason the posecheck family draws its
        # eight-method figures as unfilled curves.
        fill = len(arms) <= len(CORE)
        for lab, key, _ in arms:
            pct, _ = _mw_share(mw[key])
            col = color(lab)
            ax.step(MW_CENTRES, pct, where="mid", color=col, lw=DIST_LW + 0.35, zorder=3)
            if fill:
                ax.fill_between(MW_CENTRES, pct, step="mid", color=col, alpha=DIST_FILL,
                                lw=0, zorder=2)
        pct, _ = _mw_share(mw_ref)
        ax.plot(MW_CENTRES, _mw_rolled(pct), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4,
                dash_capstyle="round")

        # "% of molecules" is per ARM -- each curve is normalised by its own total, which
        # is the only way a 79-molecule reference and a 7,888-molecule arm share an axis
        furniture(ax, ylabel=f"% of molecules\nper {MW_BIN} Da", xlabel=MW_X_LABEL,
                  xlim=(0, MW_XMAX), xloc=MW_XTICK)
        ax.set_ylim(bottom=0)
        legend(ax, arm_handles(arms), loc="upper right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_distribution_{variant}")

    rows = []
    for lab, key, _ in all_arms:
        pct, counts = _mw_share(mw[key])
        rows += [[lab, lo, lo + MW_BIN, int(n), round(float(q), 3)]
                 for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    pct, counts = _mw_share(mw_ref)
    rows += [[REF_LABEL, lo, lo + MW_BIN, int(n), round(float(q), 3)]
             for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    write_csv(out, "mw_histogram", ["arm", "mw_bin_lo", "mw_bin_hi", "n", "pct"], rows)


@figure("fig-molweight-ecdf", needs=("target_*/samples.sdf",))
def draw_molweight_ecdf(out):
    """The distribution figure answers "where does this arm put its molecules"; this one
    answers "how much of an arm sits below any given weight", which is the form to read a
    shift between two arms off. No binning and no rolling -- the reference's 79 molecules
    are 79 honest steps."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        def ecdf(v, col, lw, ls):
            x = np.sort(v)
            ax.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=col, lw=lw, ls=ls,
                    zorder=4 if ls != "-" else 5, solid_capstyle="round")

        for lab, key, _ in arms:
            ecdf(mw[key], color(lab), MODEL_LW, "-")
        ecdf(mw_ref, REF_COLOR, REF_LW, DASH)

        furniture(ax, ylabel="Cumulative share (%)", xlabel=MW_X_LABEL, xlim=(0, MW_XMAX),
                  xloc=MW_XTICK)
        ax.set_ylim(0, 100)
        legend(ax, arm_handles(arms), loc="lower right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_ecdf_{variant}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size and fig-ensemble-* — drawn under matplotlib's own defaults
# ════════════════════════════════════════════════════════════════════════════════
# These three figures predate 260910's house style and never applied it: each ran as its own
# process, under stock rcParams. draw.py is ONE process, so a figure drawn after a styled
# one would silently inherit 15 pt DejaVu in warm near-black, and one drawn BEFORE a styled
# one would leave the style stripped for it. MISC_PLAIN_RC is rcParams as they stood when
# draw.py was imported -- before any use_style() can have run -- which is exactly the
# fresh-process state the originals drew in, and it is applied through rc_context so the
# reset lasts only as long as the figure. `backend` is left out: restoring it through
# rcParams is not how a backend is set, and rc_context drops it for the same reason.
MISC_PLAIN_RC = {k: v for k, v in plt.rcParams.copy().items() if k != "backend"}


def _misc_savefig(fig, out, stems, **kw):
    """save() writes png+svg+pdf at the figure's own dpi and no bbox; these families set
    their own dpi and a tight bbox and ship a different pair of formats, and changing that
    changes the image. So they keep their original savefig call."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for path in stems:
        fig.savefig(out / path, **kw)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size — does more MCP fine-tuning buy binding, or just size?
# ════════════════════════════════════════════════════════════════════════════════
# THE ARGUMENT. Vina rewards size, so an arm that draws smaller peptides is penalised and
# an arm that draws larger ones is flattered, regardless of whether it binds better. Binning
# by heavy-atom count removes that: inside a bin every arm draws the same-sized peptides, so
# a gap that survives the bin is a binding difference. This is §4 of the 260827 note applied
# to the MCP family, and it is the reason the pooled numbers in the tables cannot be read on
# their own.
#
# Colours come from the shared palette, where the four arms are registered as an ordinal
# ramp -- they are one model at four amounts of fine-tuning, not four methods.
MCP_FB = str(FUNCBIND / "artifacts" / "reproduction" / "mcpp")
# Four steps of one brown ramp separate cleanly in the bars, but the top panel's lines
# cross, and at a crossing two adjacent steps of a lightness ladder are genuinely hard to
# tell apart. So each arm carries a second identity channel -- line style -- the same way
# this note's published baselines do. The two solid lines are the lightest and the darkest
# step, which is the pair that never needed the help.
MCP_ARMS = [
    ("vanilla",        "FuncBind vanilla",  f"{MCP_FB}/cmp10/_eval/vanilla/eval_docking_results.json",      "-"),
    ("fine-tune 3.17M", "ft_3.17M",         f"{MCP_FB}/cmp10/_eval/finetuned/eval_docking_results.json",    (0, (1, 1.6))),
    ("fine-tune 8.21M", "ft_8.21M",         f"{MCP_FB}/cmp10_r10/_eval/finetuned/eval_docking_results.json", (0, (5, 2.2))),
    ("fine-tune 26.1M", "ft_26.1M",         f"{MCP_FB}/cmp10_r14/_eval/finetuned/eval_docking_results.json", "-"),
]
MCP_BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 89)]
MCP_LBL = ["20–29", "30–39", "40–49", "50–59", "60+"]
MCP_MIN_N = 5      # fewer molecules than this in a bin is a data point, not a trend


@figure("fig-mcp-finetune-size", needs=("mcpp/*/_eval/*/eval_docking_results.json",))
def draw_mcp_finetune_size(out):
    """Median Vina Dock per heavy-atom bin (top) over the molecule count each arm puts in
    that bin (bottom)."""
    d = {lab: json.load(open(p))["per_target"] for lab, _, p, _ in MCP_ARMS}
    # Only targets every arm produced molecules for: comparing an arm against a target where
    # another arm returned nothing is not a comparison.
    common = sorted(set.intersection(*[
        {e["target"] for e in per if e.get("vina_dock") is not None} for per in d.values()]))

    def mols(lab):
        return [(m["n_atoms"], m["vina_dock"])
                for e in d[lab] if e["target"] in common
                for m in (e.get("per_mol") or [])
                if m.get("vina_dock") is not None and m.get("n_atoms")]

    m = {lab: mols(lab) for lab, _, _, _ in MCP_ARMS}
    binned = {lab: [[v for a, v in vals if lo <= a <= hi] for lo, hi in MCP_BINS]
              for lab, vals in m.items()}

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 7.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.13))
        x = np.arange(len(MCP_BINS))

        for lab, key, _, style in MCP_ARMS:
            col = color(key)
            ys = [np.median(b) if len(b) >= MCP_MIN_N else np.nan for b in binned[lab]]
            ax.plot(x, ys, color=col, lw=2.0, ls=style, marker="o", ms=8.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.6, label=lab)
            # An arm that runs out of molecules early gets its label above the point; to the
            # right, the arms that continue would run through it.
            last = max(i for i, y in enumerate(ys) if not np.isnan(y))
            kw = dict(xytext=(10, -2), ha="left") if last == len(MCP_BINS) - 1 \
                else dict(xytext=(0, 23), ha="center")
            ax.annotate(lab, (last, ys[last]), textcoords="offset points",
                        fontsize=10.2, color=col, fontweight="600", va="center", **kw)

        ax.set_ylabel("Vina Dock, median\n(kcal/mol)", fontsize=11.3)
        ax.grid(color="#e6e9ef", lw=.85)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10.0, ncols=4, loc="lower center",
                  bbox_to_anchor=(.5, 1.01))
        ax.set_xlim(-.45, len(MCP_BINS) - .55)
        ax.margins(y=.20)

        w = .21
        for i, (lab, key, _, _s) in enumerate(MCP_ARMS):
            counts = [len(b) for b in binned[lab]]
            bx.bar(x + (i - 1.5) * w, counts, width=w * .86, color=color(key), linewidth=0)
            for xi, c in enumerate(counts):
                if c:
                    bx.annotate(str(c), (xi + (i - 1.5) * w, c), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=7.8, color="#7a8699")

        bx.set_ylabel("molecules", fontsize=11.3)
        bx.set_xlabel("heavy atoms", fontsize=11.3)
        bx.set_xticks(x)
        bx.set_xticklabels(MCP_LBL)
        bx.grid(axis="y", color="#e6e9ef", lw=.85)
        bx.set_axisbelow(True)

        for a in (ax, bx):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color("#c4cad4")
            a.tick_params(labelsize=10.2, colors="#5b6678")

        _misc_savefig(fig, out, [f"mcp_finetune_size.{ext}" for ext in ("png", "svg")],
                      dpi=170, bbox_inches="tight", facecolor="white")

    write_csv(out, "mcp_finetune_size", ["arm", "bin", "n", "dock_median"],
              [[lab, l, len(b), f"{np.median(b):.3f}" if len(b) >= MCP_MIN_N else ""]
               for lab, _, _, _ in MCP_ARMS for l, b in zip(MCP_LBL, binned[lab])])

    print("  common targets:", len(common), common)
    for lab, _, _, _ in MCP_ARMS:
        ha = [a for a, _ in m[lab]]
        print(f"  {lab:17s} n={len(m[lab]):4d} median heavy={np.median(ha):5.1f} "
              f"bins={[len(b) for b in binned[lab]]} "
              f"medians={[round(float(np.median(b)), 2) if len(b) >= MCP_MIN_N else None for b in binned[lab]]}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-ensemble-{6set-3metric,vs-alone} — probe-side feature ensembles
# ════════════════════════════════════════════════════════════════════════════════
# Does bolting a frozen baseline encoder onto the champion's features buy anything the
# champion does not already have? `fig1` is six cohorts x three metrics for the champion
# and for champion+partner; `fig2` asks the sharper question -- is the ensemble better than
# the PARTNER alone, or is the partner simply carrying it?
#
# ensemble_results.json is produced by notebook/html/260910/fig-ensemble/gen_ensemble_data.py,
# which reads feature .pt files that live on the box the probe ran on; it is a recompute
# script, not a drawing script, and it stays there.
#
# These are exploratory panels: bold suptitles, per-panel titles, matplotlib's own tab
# colours. They are deliberately NOT restyled into the house style -- they are not §1/§2
# figures and were never drawn as such.
ENS_COH = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
ENS_COHL = ["FULL", "CL3", "CL3\nID60", "CL3\nID30", "CASF\nnontrain", "CASF\nclean"]
ENS_PARTNERS = ["GET", "EGNN", "CheapNet", "ProFSA"]
ENS_COLORS = {"GET": "#1f77b4", "EGNN": "#2ca02c", "CheapNet": "#ff7f0e", "ProFSA": "#d62728"}
ENS_METRICS = [("r", "Pearson r", False), ("rho", "Spearman ρ", False), ("rmse", "RMSE", True)]


def _ens_results():
    return json.load(open(legacy("fig-ensemble", "ensemble_results.json")))


def _ens_series(d, metric):
    return [d.get(c, {}).get(metric, np.nan) for c in ENS_COH]


@figure("fig-ensemble-6set-3metric", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_6set_3metric(out):
    """Six cohorts x three metrics: champion baseline + champion⊕{GET,EGNN,CheapNet,ProFSA}."""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, (mk, ml, lower_better) in zip(axes, ENS_METRICS):
            ax.plot(x, _ens_series(r["champion"], mk), "-o", color="black", lw=2.6, ms=7,
                    label="champion (C+D+G)", zorder=5)
            for p in ENS_PARTNERS:
                ax.plot(x, _ens_series(r["ensembles"][p], mk), "--o", color=ENS_COLORS[p],
                        lw=1.8, ms=5, label=f"⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=9)
            ax.set_title(ml + ("  (↓ better)" if lower_better else "  (↑ better)"),
                         fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.axvspan(3.5, 5.5, color="grey", alpha=0.06)      # shade CASF holdout region
            if mk == "r":
                ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
        fig.suptitle("Probe-side feature ensemble: champion ⊕ baseline encoder (PCA-64), "
                     f"5-seed, n_shared={r['n_shared']}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _misc_savefig(fig, out,
                      [f"fig1_ensemble_6set_3metric.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig1")

    print("\n  Δρ (ensemble − champion) per cohort:")
    print(f"  {'cohort':<15}" + "".join(f"{p:>10}" for p in ENS_PARTNERS))
    for c in ENS_COH:
        row = f"  {c:<15}"
        for p in ENS_PARTNERS:
            d = r["ensembles"][p].get(c, {}).get("rho", np.nan) \
                - r["champion"].get(c, {}).get("rho", np.nan)
            row += f"{d:>+10.3f}"
        print(row)


@figure("fig-ensemble-vs-alone", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_vs_alone(out):
    """Spearman rho, champion vs partner-ALONE vs ensemble per cohort — does the ensemble
    beat the partner alone?"""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
        for ax, p in zip(axes2, ENS_PARTNERS):
            w = 0.27
            ax.bar(x - w, _ens_series(r["champion"], "rho"), w, color="black", label="champion")
            ax.bar(x, _ens_series(r["alone"][p], "rho"), w, color=ENS_COLORS[p], alpha=0.45,
                   label=f"{p} alone")
            ax.bar(x + w, _ens_series(r["ensembles"][p], "rho"), w, color=ENS_COLORS[p],
                   label=f"champion ⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=8)
            ax.set_title(p, fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(0.40, 0.75)
            ax.legend(fontsize=8, loc="lower left")
        axes2[0].set_ylabel("Spearman ρ", fontsize=11)
        fig2.suptitle("Is the ensemble better than the partner ALONE?  (Spearman ρ per cohort)",
                      fontsize=13, fontweight="bold")
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        _misc_savefig(fig2, out, [f"fig2_ensemble_vs_alone.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig2")


# ════════════════════════════════════════════════════════════════════════════════
# fig-posebusters-{valid-per-atom,check-failures,sucos-ecdf,sucos-per-atom}
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
# Where the symlog x axis stops being linear and starts being logarithmic. 0.1% is ~8 of
# the ~7,900 molecules an arm holds: below it the difference between two arms is a handful
# of molecules and belongs in the JSON, above it the decades do the work.
PB_LINTHRESH = 0.1
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
                 color=color(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    legend(top, arm_handles(arms), loc="lower left", fontsize=11.5)

    size_distribution(bot, xs, per, ref_per, arms)
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
    fig, ax = plt.subplots(
        figsize=(FIG_W, (0.40 if len(arms) <= 3 else 0.62) * len(names)
                 + 1.6 + 0.30 * leg_rows + 0.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(len(names))[::-1]
    for i, (lab, key) in enumerate(series):
        # top of the group downwards, so the key's order IS the order of the bars
        ax.barh(ys + ((len(series) - 1) / 2 - i) * h,
                [fails[key]["rates"].get(k, 0.0) for k in names],
                height=h, color=color(lab), edgecolor=color(lab), lw=0.8, zorder=3)

    top = max(g["rates"].get(k, 0.0) for g in shown for k in names)
    # LOG X, AND SYMLOG RATHER THAN LOG. The rates that matter run 0.09% to 23%, and on a
    # linear axis everything under ~2% -- volume overlap, internal energy, the reference's
    # own two rows -- was a stub against FuncBind's 23%. But 13 of the 81 cells here are an
    # exact zero and 5 more are a single-digit molecule count, and a plain log axis cannot
    # draw a bar that starts at zero: it would clip them all to whatever floor the axis was
    # given, making "never fails this" and "fails it 5 times" the same picture. symlog is
    # linear below PB_LINTHRESH and logarithmic above, so the bars still start at a true
    # zero, a 1-molecule cell still looks like 1 molecule, and nothing is hidden or invented.
    ax.set_xscale("symlog", linthresh=PB_LINTHRESH, linscale=0.35)
    furniture(ax, ylabel=None, xlabel="Molecules failing the check (%, log scale)",
              xlim=(0, top * 1.25), xloc=None)
    symlog = matplotlib.ticker.SymmetricalLogLocator
    ax.xaxis.set_major_locator(symlog(base=10, linthresh=PB_LINTHRESH))
    # "0.1" and "10", not matplotlib's 10^-1 and 10^1: two decades of percentages read as
    # numbers, and the zero tick has to be a zero.
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(symlog(base=10, linthresh=PB_LINTHRESH,
                                      subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x", which="minor", color=GRID, lw=GRID_LW * 0.7,
            ls=(0, (1, 4)), alpha=0.6)
    ax.grid(False, axis="y")
    ax.set_yticks(ys)
    ax.set_yticklabels([_posebusters_wrap_check(k) for k in names], fontsize=12)
    ax.set_ylim(-0.6, len(names) - 0.4)
    handles = [Patch(facecolor=color(lab), edgecolor=color(lab),
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
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color(lab),
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
        legend(ax, arm_handles(arms, include_ref=False), loc="upper left", fontsize=11.5)
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
                ax.plot(xs, model_curve(per[key], xs, f), color=color(lab),
                        lw=MODEL_LW, zorder=5, solid_capstyle="round")
            ax.axhline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
            furniture(ax, ylabel=f"SuCOS {stat}", xlabel=X_LABEL,
                      xlim=(xs[0] - 0.6, xs[-1] + 0.6))
            ax.set_ylim(0, 1)
            legend(ax, arm_handles(arms, include_ref=False), loc="upper left",
                   fontsize=11.5)
            fit(fig, pad=0.5)
            save(fig, out, f"sucos_per_atom_{stat}_{variant}")
        ranges[variant] = xs
    print("  per-atom x range: "
          + " · ".join(f"{v} {xs[0]}-{xs[-1]}" for v, xs in ranges.items()))


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-ecdf-by-size / fig-posecheck-clash-violin-by-size
# ════════════════════════════════════════════════════════════════════════════════
# Every method on one axis -- the five published baselines PLUS the three we ran here
# (TargetDiff, vanilla VoxBind, Ours v1) -- stratified by generated-molecule heavy-atom
# count. Six panels each: the five size bins plus `all sizes`.
#
# WHY THIS FAMILY DOES NOT USE pose_data(). It reads a different scoring of the same
# molecules, and a different pocket set, for two reasons:
#
#  1. prj-denovo/baselines is not on this box, so the five baselines are read from the
#     per-molecule exports the legacy folder ships (`fig-posecheck/posecheck_<Method>.json`).
#     The loader is checked on every run: the per-bin numbers it reproduces from the
#     exports must match the stored `posecheck_baselines_by_atom_range.json` on all five
#     methods x five bins x three statistics, or the build aborts instead of drawing.
#
#  2. THE RUNS' MAIN SAMPLES DIRECTORIES ARE THE WRONG PROTOCOL. There PoseCheck was
#     scored against the *pocket10 crop*; the five baselines were scored against the whole
#     `*_rec.pdb`. Clashes are receptor-dependent and the crop hides some of them --
#     pooled clash mean 5.25 crop vs 6.27 full for vanilla, 6.49 vs 7.55 for Ours v1,
#     10.39 vs 11.11 for TargetDiff (see posecheck_crop_vs_full.json). Mixing the two
#     would have handed our three a ~1-clash head start. Our three are therefore read from
#     `frozenenc_probes/posecheck_full/`, the whole-receptor re-scoring, whose pooled
#     numbers reproduce `ours_posecheck_full.json` exactly.
#
# POCKET SUBSET. The baselines cover all 79 electron-density pockets; the whole-receptor
# re-scoring of our three now covers all 79 as well: target_71 was missing only because
# scripts 70/71/72 default to `p78_targets.json`, a list that exists for a *docking*
# failure (pdb2pqr30 rejects that pocket's 10 A crop) which PoseCheck never had. Every
# series is still restricted to the pockets both sides share, computed at run time, so no
# method is scored on a pocket another one never saw; the console prints the shared count
# and anything dropped.
#
# THE REFERENCE LIGAND is drawn on the strain figure only. PoseCheck's strain
# (`calculate_strain_energy(mol, num_confs=50)`) never touches the receptor, so the
# crystal ligand's strain is protocol-independent and can be read from our local run. Its
# *clashes* are not: the only local reference scoring is against the crop, so it would sit
# ~0.3-1 clash low next to eight whole-receptor series, and it is left off the violin
# rather than quietly compared. (The whole-receptor reference clash means from the
# baselines' own run live on in the legacy posecheck_all_by_atom_range.json.)
#
# STRAIN IS NOISY BY CONSTRUCTION. `num_confs=50` random conformers means re-scoring the
# same pose gives a slightly different answer: over 7,349 molecules scored twice here the
# median relative difference is 1.0%, p90 8.9%. Fine for an ECDF, not for ranking two
# methods a few percent apart.
#
# COLOURS come from the shared palette, so the key carries across figures; only line
# weight and dash are decided here. Our two are drawn thick and solid; the five published
# baselines are context, so they are thin and each carries its own dash pattern -- with
# nine series on one axis hue alone cannot be the identity channel.
#
# STYLE. This family predates the house style at the top of this file and keeps its own:
# 12 pt type, BLACK spines and tick labels. The violin keeps its panel title -- the size bin
# has nowhere else to live inside the frame -- but the strain ECDF dropped its title on
# request (2026-09-13), so there the bin is named by the file only.

# key -> (label, linewidth, linestyle). Dash is a second identity channel for the five
# published baselines: they are drawn thin, five at a time, and colour alone is not enough
# at that weight.
PCSZ_BASELINES = [
    ("AR",         "AR",         1.6, "-"),
    ("Pocket2Mol", "Pocket2Mol", 1.6, (0, (5, 2))),
    ("DiffSBDD",   "DiffSBDD",   1.6, (0, (1, 1.6))),
    ("DecompDiff", "DecompDiff", 1.6, (0, (6, 2, 1, 2))),
    ("FuncBind",   "FuncBind",   1.6, (0, (9, 3))),
]
# TargetDiff is violet and still dashed: it used to be orange, which put two BASELINES in
# the same hue family as each other, and the dash costs nothing now that the collision is
# gone. These labels are the drawn labels, not the ARMS keys -- they reach color() through
# the alias table.
PCSZ_LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", 2.4, (0, (6, 2))),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    3.0, "-"),
    ("CoDE",          f"{E}/frozenenc_probes/posecheck_full/ours_v1",    3.4, "-"),
]

# The five shared size bins plus the pooled one every molecule ALSO lands in.
PCSZ_LABELS = BIN_LABELS + ["all sizes"]
PCSZ_SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30", "all"]
PCSZ_POOLED = len(PCSZ_LABELS) - 1

# This family's own furniture: a white ground and BLACK spines/ticks (the house INK/AXIS
# are the warm near-black of the newer figures), with the shared GRID and LEGEND_EDGE.
PCSZ_BG, PCSZ_INK, PCSZ_AXIS = "#ffffff", "#000000", "#000000"
# The axis starts at 10 kcal/mol. Below that a pose is effectively unstrained, and spending
# decades of width on it pushed the region where the methods separate into the right half.
# Values under the floor are clipped onto it, so the curve enters the axis at its true share.
PCSZ_XFLOOR, PCSZ_XTOP = 1e1, 3e3
# 80 % of the original 8.6 x 5.6 in canvas both ways, then 1.2x wider again so the lower-right
# key leaves room for the curves' long right tails, then 0.9x both ways (2026-09-13). The
# per-atom all-methods figure reads its aspect from this, so keep scaling both sides together
# unless that one should change too.
PCSZ_ECDF_SIZE = (8.6 * 0.8 * 1.2 * 0.9, 5.6 * 0.8 * 0.9)
# Axis-name size for the ECDF, and for the per-atom all-methods figure drawn in its style.
# 1.4x the family's original 11.5 pt (2026-09-13).
PCSZ_LABEL_FS = 11.5 * 1.4
PCSZ_RC = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": PCSZ_INK, "axes.labelcolor": PCSZ_INK,
    "xtick.color": PCSZ_AXIS, "ytick.color": PCSZ_AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
    # save() takes no dpi/bbox arguments, and this family is drawn at 170 dpi and cropped
    # to its own ink rather than sized to the figure. Both ride in on the savefig.* rc
    # keys, which Figure.savefig and print_figure fall back to when the kwargs are absent.
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
}


def _pcsz_empty():
    return {b: {"strain": [], "clash": []} for b in range(len(PCSZ_LABELS))}


def _pcsz_add(out, n, s, c):
    for b in (bin_of(n), PCSZ_POOLED):
        if s is not None and np.isfinite(s):
            out[b]["strain"].append(float(s))
        if c is not None and np.isfinite(c):
            out[b]["clash"].append(float(c))


def _pcsz_load_baseline(method, keep):
    """One of the five, from its per-molecule export. `keep` is a set of pocket indices."""
    d = json.load(open(legacy("fig-posecheck", f"posecheck_{method}.json"),
                       encoding="utf-8"))
    out = _pcsz_empty()
    for m in d["molecules"]:
        if m["p"] in keep:
            _pcsz_add(out, m["n"], m["s"], m["c"])
    return out, set(d["density79_pockets"])


def _pcsz_metrics(root):
    """Every per-target metrics.json under one run root, in path order."""
    return sorted(str(p) for p in Path(root).glob("target_*/metrics.json"))


def _pcsz_load_local(root, keep, reference=False):
    """One of ours, from per-target metrics.json."""
    out = _pcsz_empty()
    seen = set()
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        seen.add(idx)
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            pc = m.get("posecheck") or {}
            _pcsz_add(out, m["n_atoms"], pc.get("strain"), pc.get("clashes"))
    return out, seen


def _pcsz_pockets_of(root):
    return {int(os.path.basename(os.path.dirname(p)).split("_")[1])
            for p in _pcsz_metrics(root)}


def _pcsz_stats(cell):
    s_, c_ = cell["strain"], cell["clash"]
    return {
        "n_strain": len(s_), "n_clash": len(c_),
        "strain_median": round(st.median(s_), 1) if s_ else None,
        "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
        "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
        "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
        "clash_median": round(float(np.median(c_)), 1) if c_ else None,
    }


_PCSZ_CACHE = {}


def _pcsz_data():
    """(per-method {bin: {strain, clash}}, the drawn series, the reference, the summary).

    Cached for the life of the process: the two figures below are the same nine series
    read once and drawn twice, and the verification pass alone re-bins every baseline
    molecule five times."""
    if "d" in _PCSZ_CACHE:
        return _PCSZ_CACHE["d"]

    # ── the pocket subsets ────────────────────────────────────────────────────
    probe = json.load(open(legacy("fig-posecheck", "posecheck_AR.json"), encoding="utf-8"))
    density79 = set(probe["density79_pockets"])
    shared = set(density79)
    for _, root, *_ in PCSZ_LOCAL:
        shared &= _pcsz_pockets_of(root)
    dropped = sorted(density79 - shared)
    print(f"  pockets: baselines {len(density79)} · shared with our three {len(shared)}"
          f" · dropped {['target_%02d' % i for i in dropped]}")

    # ── verification: the exports must reproduce the stored per-bin numbers ───
    stored = json.load(open(legacy("fig-posecheck",
                                   "posecheck_baselines_by_atom_range.json"),
                            encoding="utf-8"))
    mismatch = 0
    for key, label, *_ in PCSZ_BASELINES:
        data, _ = _pcsz_load_baseline(key, density79)
        for b in range(len(EDGES) - 1):
            want, got = stored["methods"][label][b], _pcsz_stats(data[b])
            for a, c in (("n", "n_strain"), ("strain_median", "strain_median"),
                         ("clash_mean", "clash_mean")):
                if want.get(a) != got.get(c):
                    mismatch += 1
                    print(f"    MISMATCH {label} bin {PCSZ_LABELS[b]} {a}: "
                          f"stored {want.get(a)} != rebuilt {got.get(c)}")
    if mismatch:
        raise SystemExit(f"loader disagrees with the stored baseline JSON on {mismatch} "
                         f"values -- refusing to plot")
    print("  verification: exports reproduce posecheck_baselines_by_atom_range.json "
          "on 5 methods x 5 bins x 3 statistics, 0 mismatches")

    # ── the data actually plotted, all on the shared pockets ──────────────────
    data_by_label, series = {}, []
    for key, label, lw, ls in PCSZ_BASELINES:
        d79, _ = _pcsz_load_baseline(key, density79)
        dsh, _ = _pcsz_load_baseline(key, shared)
        data_by_label[label] = dsh
        series.append((label, color(label), lw, ls))
        a, b = _pcsz_stats(d79[PCSZ_POOLED]), _pcsz_stats(dsh[PCSZ_POOLED])
        print(f"    {label:12s} 79 pockets n={a['n_clash']:5d} clash {a['clash_mean']:5.2f} "
              f"strain {a['strain_median']:7.1f}   ->  {len(shared)} pockets "
              f"n={b['n_clash']:5d} clash {b['clash_mean']:5.2f} strain {b['strain_median']:7.1f}")
    for label, root, lw, ls in PCSZ_LOCAL:
        data_by_label[label], _ = _pcsz_load_local(root, shared)
        series.append((label, color(label), lw, ls))
    ref, _ = _pcsz_load_local(REF_ROOT, shared, reference=True)

    # ── the numbers behind both figures ──────────────────────────────────────
    methods = {label: [_pcsz_stats(data_by_label[label][b])
                       for b in range(len(PCSZ_LABELS))] for label, *_ in series}
    methods[REF_LABEL] = [
        {**_pcsz_stats(ref[b]), "clash_mean": None, "clash_median": None,
         "note": "strain only"} for b in range(len(PCSZ_LABELS))]

    print(f"\n  {'bin':10s} {'method':14s} {'n':>6s} {'strain med':>11s} {'IQR':>19s} "
          f"{'clash mean':>11s} {'med':>5s}")
    for b, lab in enumerate(PCSZ_LABELS):
        for label in methods:
            r = methods[label][b]
            if not r["n_strain"]:
                continue
            iqr = (f"{r['strain_q25']:7.1f}–{r['strain_q75']:<10.1f}"
                   if r["strain_q25"] is not None else " " * 18)
            cm = f"{r['clash_mean']:11.2f}" if r["clash_mean"] is not None else f"{'—':>11s}"
            cd = f"{r['clash_median']:5.1f}" if r["clash_median"] is not None else f"{'—':>5s}"
            print(f"  {lab:10s} {label:14s} {r['n_strain']:6d} {r['strain_median']:11.1f} "
                  f"{iqr} {cm} {cd}")
        print()

    _PCSZ_CACHE["d"] = (data_by_label, series, ref, methods)
    return _PCSZ_CACHE["d"]


def _pcsz_style(ax, *, grid_axis="both"):
    """The vina figures' axis furniture: black spines and outward ticks, dotted grid.

    `grid_axis` is "y" for the violin -- its x is categorical, so a vertical rule through
    every category centre is a picket fence, not a reading aid."""
    ax.set_facecolor(PCSZ_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(labelsize=11, direction="out", length=3.5, width=1.1, pad=4,
                   colors=PCSZ_AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.9, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)


PCSZ_KEY_NCOL = 2
PCSZ_KEY_KW = dict(fontsize=9.5, handlelength=1.6, handletextpad=0.5, labelspacing=0.32,
                   borderpad=0.4, columnspacing=1.2)


def _pcsz_key(fig, ax):
    """The ECDF key: the crystal ligands on a centred line of their own, over the eight methods
    in 4 rows x 2 columns reading across, in one frame in the lower right.

    Built like the Vina per-atom key (_vpa_legend): TWO legends inside ONE rectangle. A legend
    column is as wide as its widest entry and matplotlib cannot span a cell, so the reference
    cannot sit centred over the grid as a cell of it. The grid is placed first, the reference
    legend is centred on it and stacked on top, both frames come off, and a single rectangle is
    drawn round their union. Call AFTER the layout is final: positions are axes fractions of
    the axes as laid out when this runs."""
    pairs = list(zip(*ax.get_legend_handles_labels()))
    ref = [p for p in pairs if p[1] == REF_LABEL]
    methods = [p for p in pairs if p[1] != REF_LABEL]
    rows = -(-len(methods) // PCSZ_KEY_NCOL)
    # matplotlib fills a legend column by column; interleave so the grid reads across.
    methods = [methods[r * PCSZ_KEY_NCOL + c] for c in range(PCSZ_KEY_NCOL)
               for r in range(rows) if r * PCSZ_KEY_NCOL + c < len(methods)]
    grid = ax.legend(*zip(*methods), loc="lower right", ncol=PCSZ_KEY_NCOL, borderaxespad=0.5,
                     frameon=False, **PCSZ_KEY_KW)
    legs = [grid]
    render = fig.canvas.get_renderer
    if ref:
        ax.add_artist(grid)          # the next ax.legend() would otherwise replace it
        fig.canvas.draw()
        bb = grid.get_window_extent(render()).transformed(ax.transAxes.inverted())
        legs.append(ax.legend(*zip(*ref), loc="lower center", borderaxespad=0, frameon=False,
                              bbox_to_anchor=((bb.x0 + bb.x1) / 2, bb.y1), **PCSZ_KEY_KW))
    fig.canvas.draw()
    boxes = [l.get_window_extent(render()).transformed(ax.transAxes.inverted()) for l in legs]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    # An AXES artist, not a figure one: a figure-level patch is drawn after the whole axes and
    # would cover the legend text. Opaque white under the text, over the curves.
    ax.add_artist(matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes, facecolor="white",
        edgecolor=LEGEND_EDGE, linewidth=0.7, zorder=5, clip_on=False))
    for l in legs:
        l.set_zorder(6)
        for text in l.get_texts():   # identity rides the swatch, not the ink
            text.set_color(PCSZ_INK)


def _pcsz_csv(out, methods):
    write_csv(out, "posecheck_all_by_atom_range",
              ["bin", "method", "n_strain", "strain_median", "strain_q25", "strain_q75",
               "n_clash", "clash_mean", "clash_median"],
              [[lab, label, r["n_strain"], r["strain_median"], r["strain_q25"],
                r["strain_q75"], r["n_clash"], r["clash_mean"], r["clash_median"]]
               for b, lab in enumerate(PCSZ_LABELS)
               for label, r in ((k, v[b]) for k, v in methods.items())])


@figure("fig-posecheck-strain-ecdf-by-size", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_strain_ecdf_by_size(out):
    """UFF strain ECDF, nine methods on one log axis, per heavy-atom bin."""
    data_by_label, series, ref, methods = _pcsz_data()
    with plt.rc_context(PCSZ_RC):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            for label, colour, lw, ls in series:
                v = np.asarray(data_by_label[label][b]["strain"], dtype=float)
                if v.size < 20:
                    continue
                x = np.sort(np.clip(v, PCSZ_XFLOOR, None))
                # soft(): CoDE in its lighter tint, as in the eight-method strain line figures
                # (2026-09-13). The clash violin below still draws the palette colour.
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=soft(label), lw=lw, ls=ls,
                        label=label, solid_capstyle="round")
            r = np.asarray(ref[b]["strain"], dtype=float)
            if r.size >= 3:
                x = np.sort(np.clip(r, PCSZ_XFLOOR, None))
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR, lw=2.0,
                        ls=(0, (3, 2)),
                        label=REF_LABEL)
            ax.set_xscale("log")
            ax.set_xlim(PCSZ_XFLOOR, PCSZ_XTOP)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("UFF strain energy (kcal mol⁻¹) · log scale", fontsize=PCSZ_LABEL_FS)
            ax.set_ylabel("Cumulative probability", fontsize=PCSZ_LABEL_FS)
            # No title (removed on request 2026-09-13): the size bin is carried by the file
            # name (`ecdf-by-size-<bin>`) and by whatever caption places the figure.
            # Nine series; the legend names them only -- the medians and n are in the CSV and
            # the run log, not the key. The crystal ligands head it on a centred line of their
            # own, over the eight methods in 4 x 2 (see _pcsz_key); lower right, opaque white
            # with a grey rule so the dotted grid does not run through the text. The key is
            # placed after tight_layout, because its frame is measured in axes fractions.
            fig.tight_layout(pad=0.5)
            _pcsz_key(fig, ax)
            save(fig, out, f"strain_ecdf_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


@figure("fig-posecheck-clash-violin-by-size", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_violin_by_size(out):
    """Steric clashes per pose, eight methods as violins, per heavy-atom bin."""
    data_by_label, series, _ref, methods = _pcsz_data()
    with plt.rc_context(PCSZ_RC):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            fig, ax = plt.subplots(figsize=(10.6, 5.6))
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax, grid_axis="y")
            vals, cols, ticks = [], [], []
            for label, colour, lw, ls in series:
                v = data_by_label[label][b]["clash"]
                if len(v) < 20:
                    continue
                vals.append(v)
                cols.append(colour)
                ticks.append(label.replace(" · ", "\n").replace(" σ", "\nσ"))
            # KDE fitted on log10(x+1) and the ticks relabelled with real counts: a plain
            # log axis is impossible because a few percent of poses have zero clashes, and
            # symlog would warp only the display while the KDE stayed in data space.
            tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
            parts = ax.violinplot([tf(v) for v in vals], showextrema=False, widths=0.82)
            for body, colour in zip(parts["bodies"], cols):
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)
                body.set_linewidth(1.2)
            for i, v in enumerate(vals, start=1):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
                ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
            hi = max(max(v) for v in vals)
            ticks_at = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= hi * 1.6]
            ax.set_yticks(tf(ticks_at))
            ax.set_yticklabels([str(t) for t in ticks_at])
            ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.20)
            for i, v in enumerate(vals, start=1):
                ax.text(i, ax.get_ylim()[1],
                        f"mean {np.mean(v):.2f}\nmed {np.median(v):.0f}",
                        ha="center", va="top", fontsize=8.5, color="#3a4352",
                        linespacing=1.35)
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(ticks, fontsize=9)
            ax.set_ylabel("steric clashes per pose  ·  log-spaced", fontsize=11.5)
            ax.set_title(f"Steric clashes — {lab} heavy atoms", fontsize=14,
                         fontweight="620", loc="left", pad=10)
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_violin_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-atom, fig-posecheck-clash-per-atom, fig-posecheck-ecdf-pair
# ════════════════════════════════════════════════════════════════════════════════
# The size-resolved view of PoseCheck strain and steric clashes: what each arm's poses cost
# at a given ligand size, against the crystal ligand at that size.
#
# MEAN AND MEDIAN ARE SEPARATE FIGURES, which is what the 3-line figures do
# (vina_dock_3line_mean / _median) and here it is not optional. Strain's mean is not a
# location statistic: 6.6% of molecules fail UFF relaxation and land between 1e4 and 1e13,
# and one of those at a thin heavy-atom count moves that count's mean by four decades. The
# two sit three to four decades apart, so overlaid, the mean's spikes cross the whole panel
# and bury the medians -- which are tight, ordered, and the thing worth reading. Drawn
# apart, the contrast is itself the argument for reporting the median: in the mean figure
# the arms are tangled with no order at all, in the median figure they separate cleanly.
#
# The mean figure is still clipped to the bulk -- the 3-line figure's own answer (see its
# `_v3_limits`) -- and the excluded points are NAMED in the run log rather than squashing
# everything else into two decades. Clashes have no such problem and are drawn the same way
# for symmetry.
PC_STATS = (("mean", lambda v: float(np.mean(v))),
            ("median", lambda v: float(np.median(v))))
PC_STRAIN_CLIP = 1e6
PC_STRAIN_UNIT = "\n(kcal mol⁻¹)"
# The ECDF's x floors: strain is drawn on a log axis, so a value at or below zero has no
# place on it and is clipped onto the first decade rather than dropped; clashes are counts
# and floor at 0.
PC_ECDF_FLOOR = {"s": 1e-2, "c": 0}
PC_ECDF_W = 1.52                       # two panels side by side, off the one-panel width


def _pc_ranges():
    """(variant, arms, xs) per variant, plus the line saying what was drawn.

    THE X RANGE IS THE STRAIN RANGE, for the clash figure too. The builder computed it once
    from field "s" and handed the same xs to both, so the two panels register against each
    other count for count. Resolving it against "c" instead would silently shift the clash
    figure: a UFF relaxation that did not converge drops a molecule from the strain counts
    and not from the clash ones, so the two fields do not reach MIN_N at the same sizes."""
    data, p79_rows, _ = pose_data()
    out = []
    for variant, arms in variants("s", data):
        per = {key: by_size(p79_rows[key], "s") for _, key, _ in arms}
        out.append((variant, arms, x_range(per, arms)))
    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, _, xs in out)
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    return out


def _pc_per_atom_stat(out, field, xs, arms, variant, name, unit, stat, f, *, log,
                      clip=None, stem, legend_loc="upper left"):
    """One statistic, one panel, one file. Which statistic you are looking at is carried by
    the filename and by the y-axis name, exactly as the 3-line figures carry it."""
    _, p79_rows, refrows = pose_data()
    per = {key: by_size(p79_rows[key], field) for _, key, _ in arms}
    ref_per = by_size(refrows, field)
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dropped = []
    ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per[key], xs, f)
        if clip:
            over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
            if over:
                dropped.append((lab, over))
        ax.plot(xs, y, color=color(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if clip:
        ax.set_ylim(top=clip)
    furniture(ax, ylabel=f"{name} {stat}{unit}", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    legend(ax, arm_handles(arms), loc=legend_loc, fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"{stem}_{variant}")
    return dropped


def _pc_block(rows):
    """The pooled numbers the run log reports for one set of molecules. These used to fill
    posecheck_summary.json too; that export belongs to the recompute script, so only what
    the table prints survives here."""
    s = [r["s"] for r in rows if r["s"] is not None]
    c = [r["c"] for r in rows if r["c"] is not None]
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "strain_mean": round(float(np.mean(s)), 1) if s else None,
        "strain_median": round(st.median(s), 1) if s else None,
        "clash_median": round(float(np.median(c)), 1) if c else None,
    }


def _pc_table():
    """Pooled strain and clashes per arm over the 79-pocket set, and the crystal ligand."""
    data, p79_rows, refrows = pose_data()
    print(f"  {'arm':16s} {'atoms':>6s} {'strain med':>11s} {'strain mean':>12s} "
          f"{'clash med':>10s} {'mols':>7s}")
    for lab, key, _ in arms_for("s", data):
        r = _pc_block(p79_rows[key])
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
              f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")
    r = _pc_block(refrows)
    print(f"  {REF_LABEL:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
          f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")


def _pc_curve_csv(out, xs):
    """The curves themselves, beside the figure, so a reader can check one without
    re-running it. Strain and clashes go in ONE table at ONE xs -- four curves over the
    same molecules at the same heavy-atom counts -- so the strain folder and the clash
    folder each get the whole thing rather than half of it. xs is the `all` variant's
    range, the wider of the two, and `core` is a subset of these rows."""
    data, p79_rows, refrows = pose_data()
    fields = ["strain_mean", "strain_median", "clash_mean", "clash_median"]
    cells = lambda curves, i: [("" if curves[f][i] is None else round(curves[f][i], 3))
                               for f in fields]
    rows = []
    for lab, key, _ in arms_for("s", data):
        s_per, c_per = by_size(p79_rows[key], "s"), by_size(p79_rows[key], "c")
        curves = {"strain_mean": model_curve(s_per, xs, PC_STATS[0][1]),
                  "strain_median": model_curve(s_per, xs, PC_STATS[1][1]),
                  "clash_mean": model_curve(c_per, xs, PC_STATS[0][1]),
                  "clash_median": model_curve(c_per, xs, PC_STATS[1][1])}
        rows += [[lab, a, len(s_per.get(a, ()))] + cells(curves, i)
                 for i, a in enumerate(xs)]
    # The reference's own curves come from the centred +-REF_WIN window, not from a count,
    # so it has no per-count n to report.
    s_ref, c_ref = by_size(refrows, "s"), by_size(refrows, "c")
    curves = {"strain_mean": reference_curve(s_ref, xs, PC_STATS[0][1]),
              "strain_median": reference_curve(s_ref, xs, PC_STATS[1][1]),
              "clash_mean": reference_curve(c_ref, xs, PC_STATS[0][1]),
              "clash_median": reference_curve(c_ref, xs, PC_STATS[1][1])}
    rows += [[REF_LABEL, a, ""] + cells(curves, i) for i, a in enumerate(xs)]
    write_csv(out, "posecheck_per_atom", ["arm", "heavy_atoms", "n"] + fields, rows)


@figure("fig-posecheck-strain-per-atom", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain)",))
def draw_posecheck_strain_per_atom(out):
    """PoseCheck strain against heavy-atom count — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_per_atom_all_methods."""
    use_style()
    drops, all_xs = {}, None
    for variant, arms, xs in core_only(_pc_ranges()):
        drops[variant] = _pc_per_atom_stat(
            out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT, *PC_STATS[0],
            log=True, clip=PC_STRAIN_CLIP, stem="strain_per_atom_mean",
            legend_loc="lower right")
        _pc_per_atom_stat(out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT,
                          *PC_STATS[1], log=True, stem="strain_per_atom_median",
                          legend_loc="upper left")
        all_xs = xs
    _pc_table()
    # Named, not hidden: the mean figure clips to the bulk, so say which points that leaves
    # off the panel and how far above they went.
    if drops["core"]:
        print(f"\n  strain_per_atom_mean_core: above the {PC_STRAIN_CLIP:.0e} clip, "
              f"off-panel (a failed UFF relaxation at a thin count moves that count's "
              f"mean):")
        for lab, over in drops["core"]:
            pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
            print(f"    {lab:16s} {len(over):2d} of {len(all_xs):2d}: {pts}")
    _pc_curve_csv(out, all_xs)


@figure("fig-posecheck-clash-per-atom", folder="fig-posecheck/clash", needs=("metrics.json (posecheck.clashes)",))
def draw_posecheck_clash_per_atom(out):
    """PoseCheck steric clashes against heavy-atom count — mean and median, core only.

    No `all`: it was core plus TargetDiff, because only those three carry a per-molecule
    clash count here. The whole field is clash_violin_by_size, which reads the svr12
    PoseCheck exports and draws all eight methods. See core_only()."""
    use_style()
    all_xs = None
    for variant, arms, xs in _pc_ranges():
        for stat, f in PC_STATS:
            _pc_per_atom_stat(out, "c", xs, arms, variant, "Clashes", "", stat, f,
                              log=False, stem=f"clash_per_atom_{stat}",
                              legend_loc="upper left")
        all_xs = xs
    _pc_curve_csv(out, all_xs)




# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-rotbond, fig-posecheck-strain-box-per-rotbond,
# fig-posecheck-strain-rotbond-all-methods, fig-posecheck-strain-per-atom-all-methods
# ════════════════════════════════════════════════════════════════════════════════
# STRAIN AGAINST THE NUMBER OF ROTATABLE BONDS, and the eight-method version of the same
# metric on both axes. The VoxBind paper's Fig. 12/13 form.
#
# WHY THE AXIS IS WORTH SWAPPING. Strain is conformational: it is the energy a pose carries
# because its torsions are not where the force field would put them, so the number of
# torsions a molecule HAS is the more direct explanatory variable, and heavy-atom count is
# its proxy. The two are not interchangeable -- a fused polycyclic and a long-chain ligand
# of equal size have very different torsional freedom -- and the ranking of the arms is
# allowed to differ between the two views.
#
# ROTATABLE BONDS ARE COUNTED FROM THE SMILES that `metrics.json` already records, with
# RDKit's default (strict) `CalcNumRotatableBonds`, which excludes amides, terminal bonds
# and ring bonds. It is a topological descriptor, so reading it off the recorded SMILES
# rather than the pose gives the same integer with no risk of a sample-to-SDF index slip.
# Molecules whose SMILES will not parse are dropped and counted in the run log.
#
# THE CRYSTAL REFERENCE WINDOW IS +-1 HERE, NOT the header's +-4. There is one crystal
# ligand per pocket, so a per-count reference curve over 79 ligands would be noise -- that
# is why the window exists at all. But rotatable-bond counts run 0-14 where ligand sizes run
# 5-45, and a +-4 window there spans two thirds of the axis and would flatten the reference
# into a near-constant line. +-1 pools 12-29 ligands per point and the line stops at 9,
# where the window falls under the minimum, rather than being extended into an invented
# value. The originals got this by MUTATING pose_common.REF_WIN at import; in one process
# that draws every figure a module-level override would silently re-window the per-atom
# figures next door, so the narrow window is a local constant and a local curve function.
#
# THE BOXES ARE THE PRIMARY FIGURE AND THE LINES ARE KEPT BESIDE THEM. A median line says
# where an arm sits; it cannot say whether two arms a factor of 1.5 apart are actually
# separated, and on a metric whose distribution spans four decades inside a single bond
# count that is the question. The boxes answer it -- and they show something the lines
# cannot: the arms' inter-quartile ranges overlap almost completely at every count, so the
# median ordering is a shift of a wide distribution, not a separation of two narrow ones.
#
# WHISKERS ARE THE 5TH AND 95TH PERCENTILES AND FLIERS ARE NOT DRAWN. Not a cosmetic
# choice: 4-7% of molecules relax to 1e4-1e13, so Tukey whiskers with fliers would put
# single points nine decades above the boxes and squash every box in the figure into a
# line. The percentile whisker is stated on the y axis, and the tail it leaves out is
# reported as a number rather than drawn. NO MEAN MARKER ON THE BOXES either: within one arm
# and one bond count the mean sits at 1e4-1e11 while the box sits near 1e2, so every marker
# would land far above its own box and drag the axis with it. The mean has its own line
# figure, where being unreadable at least reads as the finding it is.
#
# MEAN AND MEDIAN ARE SEPARATE LINE FIGURES, as everywhere in 260910. THE MEDIAN IS THE
# FIGURE TO READ; the mean is here so that claim can be checked, not as an alternative. AND
# THE PER-ROTBOND MEAN PANEL IS NOT CLIPPED, where the per-atom one is: 15 rotatable-bond
# counts pool 500-1,100 molecules each, so nearly every point catches one of the conformers
# that relax to 1e8-1e13, and a 1e6 clip left the curve as disconnected fragments with most
# of it above the panel. Drawn whole it spans nine decades, sits 6-9 decades above the
# crystal ligands and has no ordering at all -- which is the honest picture of what a mean
# does to this metric, and it is the argument for the median rather than something to hide
# behind an axis limit.
RB_REF_WIN, RB_MIN_REF = 1, 12
RB_X_LABEL = "Number of rotatable bonds in ligand"
RB_XTICK = 1
# Above this a UFF relaxation has effectively failed rather than found a lower conformer
# (the converged bulk sits under ~1e3). Nothing is filtered on it; it only names the tail
# that makes the mean unreadable, so the mean figure can be read for what it is.
RB_TAIL = 1e4
RB_WHIS = (5, 95)
# Strain is floored before boxing, at the value strain_clash_ecdf_pair already floors it to.
# A rigid ligand can relax to ~0, and on a log axis a single 1e-11 at 0 rotatable bonds
# pulled the panel down through fifteen decades and flattened every box in it. Values are
# CLIPPED, not dropped, so the whisker rests on the floor and the count is unchanged.
RB_STRAIN_FLOOR = 1e-2
RB_STATS = {"median": lambda v: float(np.median(v)),
            "mean": lambda v: float(np.mean(v))}
RB_Y_LABEL = "Strain {stat}\n(kcal mol⁻¹)"
# The mean curve sits high and rises; the median curve sits low and rises. Each legend goes
# in the corner its own curves leave empty.
RB_LEGEND_LOC = {"median": "upper left", "mean": "lower right"}
# The three-arm boxes: the ECDF pair's canvas width, and the share of each count's slot left
# as white space between groups.
RB_BOX_WIDE, RB_BOX_GAP = 1.52, 0.28

# ── the eight-method versions ────────────────────────────────────────────────────
# `posecheck_<Method>.json` holds the published baselines' per-molecule strain, and has all
# along -- but those exports carry no SMILES, so rotatable bonds could not be counted from
# them. The molecules themselves are in the results bundle, so the count is recovered by
# joining the two.
#
# WHERE THE ROTATABLE BONDS COME FROM, AND WHY THE JOIN IS SAFE. The baselines' molecules
# live in `results/task2-drugdesign/<M>/samples/meta/` as the TargetDiff meta format: one
# list per test pocket of {mol, smiles, ligand_filename, pred_pos}. `export_posecheck_json.py`
# built `posecheck_<Method>.json` from THE SAME meta, walking each pocket's entries in order
# and skipping the ones whose `mol` is None -- so the two are the same molecules in the same
# order, and `_rb_load_meta` reproduces its loader exactly (base + `_part2` concatenated per
# pocket; `_gap.pt` is present in the bundle but that loader does not use it, so neither
# does this one).
#
# That is an argument, not evidence, so the join is CHECKED rather than trusted: for every
# pocket the heavy-atom sequence recomputed from the meta must equal the `n` sequence in the
# export, position by position, and a mismatch aborts. It matches 100/100 pockets for all
# five methods.
#
# MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN. The baselines were
# scored against the whole `*_rec.pdb` receptor; the local arms are read from the pocket10
# crop in their target `metrics.json`. Strain is a property of the ligand's own conformer --
# UFF relaxation under a position constraint -- so receptor scope cannot enter it, and
# measuring the same molecules both ways confirms it does not: median |relative difference|
# 0.6-1.0 %, which is the run-to-run noise `num_confs=50` already carries. Do NOT extend
# this to clashes or interactions: those are receptor-dependent and a crop cannot see an
# atom it does not contain.
#
# label, bundle folder, meta stem. DecompDiff's meta is the reference-prior run, which is
# the one export_posecheck_json.py scored. FuncBind's meta reached the bundle on 2026-09-10
# and joins cleanly (100/100 pockets, 9,992 molecules); its shard SDFs under
# `funcbind/artifacts/reproduction/crossdocked/paper_run` were tried as a substitute and
# FAILED this same check -- a different sampling run from the one PoseCheck scored -- so do
# not reach for them again if the meta ever goes missing.
RB_BASELINES = [
    ("AR",         "AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
    ("FuncBind",   "FuncBind",   "FuncBind"),
]
# Thin and dashed: with eight models on one axis hue alone is not enough, and the three
# local arms are the subject while these are context. Same channel split the nine-series
# by-atom-range figures already use.
RB_BASE_LW = 1.6
RB_BASE_DASH = {"AR": (0, (5, 2)), "Pocket2Mol": (0, (1, 1.6)),
                "DiffSBDD": (0, (6, 2, 1, 2)), "DecompDiff": (0, (9, 3)),
                "FuncBind": (0, (3, 1.4, 1, 1.4))}
# The eight-method figures draw every method through soft(): CoDE's lighter tint, the one the
# Vina per-atom family (dock-per-atom-v*) carries, and the palette colour for everyone else.
# The shared #4363D8 was the only saturated hue among nine series (changed 2026-09-13). The
# three-arm figures above keep the full colour: their key comes from arm_handles().
RB_ALL_BOX_GAP = 0.30        # of a bond count's width, left clear between neighbouring groups
# The eight-method box panel is wider and taller than the house figure -- eight boxes have
# to fit inside each of thirteen counts -- and the share panels are a 3x3 block of small ones.
RB_BOX_WIDE_1, RB_BOX_TALL = 1.72, 1.22
RB_DIST_COLS, RB_DIST_WIDE, RB_DIST_TALL = 3, 1.34, 0.66
RB_BOX_YLIM = (1e0, 1e5)
# THE AXIS STOPS AT 12 ROTATABLE BONDS, because each INDIVIDUAL count past it is thin and
# stretching the axis to the last molecule anyone made (25, one VoxBind ligand) spent two
# thirds of the width on a handful of boxes.
#
# THE CUMULATIVE TAIL IS NOT NEGLIGIBLE, THOUGH, and the figure must not imply it is: 5.0 %
# of VoxBind's ligands, 3.1 % of DecompDiff's and 2.8 % of DiffSBDD's have more than 12
# rotatable bonds (Pocket2Mol is the outlier at 0.0 %). So every share panel PRINTS its own
# excluded percentage rather than letting the cap pass silently. Raise RB_X_MAX to see them.
RB_X_MAX = 12
# Above this many counts the axis is split over two rows so the boxes stay wide enough to
# read; at RB_X_MAX = 12 it is one row. Raise RB_X_MAX and the split comes back on its own.
RB_SPLIT_ABOVE = 14
# Boxes are drawn further into the tail than the sibling figures' MIN_N=25 allows, so every
# method covers its own full range instead of being clipped to the narrowest one. Ten
# molecules is the floor for a box to carry quartiles at all; the share figure is what tells
# the reader which end of the axis is thin. The crystal ligands are a LINE in the box
# figure, not a ninth box series, so this floor never applies to them -- theirs is
# RB_MIN_REF over the +-RB_REF_WIN window, and it is why their line stops at 9.
RB_GRID_MIN_N = 10
# Floor for the log-scaled share panels; below this a bond count is empty, not rare.
RB_DIST_FLOOR = 0.05
RB_KEY_H = 0.92              # inches of key strip under the line panels; see _rb_lines()
# The summary table bins the axis; the figures keep every count. Half-open, so these are
# {0}, {1,2}, {3,4}, {5,6}, {7,8,9}, {10+}.
RB_BIN_EDGES = [0, 1, 3, 5, 7, 10, 10 ** 6]
RB_BIN_LABELS = ["0", "1–2", "3–4", "5–6", "7–9", "10+"]

# The three arms this section runs locally, addressed by their stable KEYS -- their LABELS
# moved ("VoxBind + Ours" -> "Ours" -> "CoDE") and may move again. They are picked BY KEY
# rather than by walking ARMS: that list grew from three arms to eight on 2026-09-10 when
# the five published baselines were added to it, pointing at `exps/baselines_pose/<m>`,
# which another job is still filling. Those trees carry PoseBusters but no `posecheck` block
# yet, so walking ARMS here would load three methods' worth of rows with `s=None` and
# silently empty every range these builders compute.
#
# WHEN `exps/baselines_pose/` IS COMPLETE the bundle join becomes unnecessary: those trees
# carry SMILES and will carry strain, so the three-arm figures above will cover every method
# through the shared loader alone and these two can be retired.
RB_LOCAL_KEYS = ("targetdiff", "vanilla", "ours_v1")
RB_LOCAL_LABELS = []         # filled by the loaders from ARMS, in RB_LOCAL_KEYS order
RB_RELABEL = {"ours_v1": "CoDE"}
# THE ROW ORDER OF 260827/table_drug_design.tex, which is canonical for this section, except
# that VoxBind is pulled down next to CoDE so the model our arm modifies sits immediately
# before it and the two read as a pair. Legends, panels and exports all read from this, so a
# method cannot sit in one order in the legend and another in the share panels.
RB_ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff", "FuncBind",
            "VoxBind", "CoDE"]

# The per-ATOM eight-method figure. No join is needed there at all: unlike the rotatable-bond
# axis, which had to go back to the results bundle for SMILES, heavy-atom count is already
# in `posecheck_<Method>.json`. Only the p79 pockets are kept, by the export's own index.
RB_ATOM_BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind"]
RB_X_LO, RB_X_HI = 5, 44
# DecompDiff PUTS EXACTLY ONE SIZE IN EACH POCKET, AND IT IS THE CRYSTAL LIGAND'S -- checked
# on every run, not assumed. Its reference prior fixes the atom budget per pocket, so on
# that axis DecompDiff is not a distribution but a comb over the 79 reference sizes, with
# exact zeros at 7, 24, 30, 34, 36, 39-41, 43-44 atoms. So EVERY model curve pools a +-1
# atom window: drawn per exact count DecompDiff's line would break ten times inside the axis
# on molecules that were never going to exist. The window is one atom wide, the same
# treatment for all eight methods, and it moves a median of a continuous quantity almost
# nowhere at these counts. The one hole it does not fill -- 40 atoms, where no reference
# sits within +-1 -- is left as a break rather than widened away.
RB_WIN = 1
RB_MEAN_CLIP = 1e6           # the per-atom mean panel's top, as in the three-arm figure

_RB_CHEM = None
_RB_CACHE = {}


def _rb_rdkit():
    """RDKit, imported on first use rather than at the top of draw.py. The chemistry toolkit
    costs seconds to import and is the only thing in this file the other figure families do
    not need, so a run that draws none of these four never pays for it."""
    global _RB_CHEM
    if _RB_CHEM is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
        _RB_CHEM = (Chem, rdMolDescriptors)
    return _RB_CHEM


def _rb_rot_bonds(smiles):
    """RDKit's strict rotatable-bond count, or None if the SMILES will not parse. Cached:
    the arms carry ~60k molecules and many repeat."""
    if not smiles:
        return None
    if smiles not in _RB_CACHE:
        chem, desc = _rb_rdkit()
        mol = chem.MolFromSmiles(smiles)
        _RB_CACHE[smiles] = None if mol is None else int(desc.CalcNumRotatableBonds(mol))
    return _RB_CACHE[smiles]


def _rb_attach(rows):
    """Add `rb` to each row in place, and report how many rows could not get one."""
    bad = 0
    for r in rows:
        r["rb"] = _rb_rot_bonds(r.get("smi"))
        bad += r["rb"] is None
    return bad


def _rb_reference_curve(per, xs, f):
    """reference_curve over the NARROW +-RB_REF_WIN window -- see the banner. None where the
    window is too thin to mean anything, which leaves a gap in the line rather than an
    invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= RB_REF_WIN for v in vals]
        out.append(f(p) if len(p) >= RB_MIN_REF else None)
    return out


def _rb_ref_window_n(per, xs):
    """The reference's n is the WINDOWED pool, not the exact-count one: the plotted value
    comes from +-RB_REF_WIN, so an n beside it that counted only the exact bond count would
    read as 4 ligands supporting a point that 16 produced."""
    return [sum(len(v) for x2, v in per.items() if abs(x2 - x) <= RB_REF_WIN) for x in xs]


def _rb_cell(v):
    return "" if v is None else round(v, 3)


def _rb_tail_share(per):
    """% of an arm's molecules whose relaxation ran past RB_TAIL -- the tail that decides its
    mean curve, and the reason the mean curve is not a location statistic."""
    vals = [v for vs in per.values() for v in vs]
    return round(100 * sum(v > RB_TAIL for v in vals) / len(vals), 2) if vals else None


def _rb_tail_share_rows(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > RB_TAIL for x in v) / len(v), 2) if v else None


# ── the three local arms, per rotatable bond ─────────────────────────────────────
def _rb_local():
    """(per-arm {rb: [strain]}, the crystal ligands' own, the (name, arms) variants, dropped).

    THE ARMS ARE THE ONES THAT CARRY STRAIN, not all of ARMS. That list grew to eight on
    2026-09-10 and the five staged baselines hold PoseBusters only, so `variants()` with no
    field would put five all-None curves in `all` and leave x_range intersecting an empty
    set. `variants("s", data)` is the test the per-atom sibling already applies."""
    data, p79_rows, refrows = pose_data()
    bad = sum(_rb_attach(rows) for rows in p79_rows.values()) + _rb_attach(refrows)
    per_arm = {key: by_size(p79_rows[key], "s", key="rb") for _, key, _ in ARMS}
    ref_per = by_size(refrows, "s", key="rb")
    return per_arm, ref_per, variants("s", data), bad


def _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per):
    """One statistic, one panel, one file — the per-atom builder's layout with this axis."""
    f = RB_STATS[stat]
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per_arm[key], xs, f)
        ax.plot(xs, y, color=color(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")

    ax.set_yscale("log")
    furniture(ax, ylabel=RB_Y_LABEL.format(stat=stat), xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.35, xs[-1] + 0.35), xloc=RB_XTICK)
    legend(ax, arm_handles(arms), loc=RB_LEGEND_LOC[stat], fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_per_rotbond_{stat}_{variant}")


def _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per):
    """The distribution at each rotatable-bond count, one box per arm, dodged within the
    count. Wider canvas than the line panels (the ECDF pair's width, already in the house)
    because this draws 15 counts x 2-3 arms of boxes on one axis."""
    fig, ax = plt.subplots(figsize=(FIG_W * RB_BOX_WIDE, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Dodge: n boxes centred on the count, leaving RB_BOX_GAP of the slot as white space so
    # neighbouring counts stay visually separate groups.
    n = len(arms)
    slot = (1.0 - RB_BOX_GAP) / n
    for i, (lab, key, _) in enumerate(arms):
        offs = (i - (n - 1) / 2) * slot
        vals = [np.clip(per_arm[key].get(x, []), RB_STRAIN_FLOOR, None) for x in xs]
        col = color(lab)
        bp = ax.boxplot(vals, positions=[x + offs for x in xs], widths=slot * 0.86,
                        whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                        manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=1.05)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=col, linewidth=1.05)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=1.5, solid_capstyle="butt")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]), color=REF_COLOR,
            lw=REF_LW, ls=DASH, zorder=6, dash_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(bottom=RB_STRAIN_FLOOR)
    furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers",
              xlabel=RB_X_LABEL, xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    legend(ax, arm_handles(arms), loc="upper left", fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_box_per_rotbond_{variant}")


def _rb_local_csv(out, xs, arms, per_arm, ref_per):
    """The curves themselves plus the per-count molecule counts, so a reader can check a
    figure without re-running it — and so the rotatable-bond DISTRIBUTION is on the record.
    It is not drawn as a strip (the per-atom figure this mirrors has no strip either), but
    the arms differ in it, and any claim about a curve has to be read against `n`."""
    rows, summary = [], {}
    for lab, key, _ in arms:
        med = model_curve(per_arm[key], xs, RB_STATS["median"])
        mean = model_curve(per_arm[key], xs, RB_STATS["mean"])
        summary[lab] = {"n_scored": sum(len(v) for v in per_arm[key].values()),
                        "strain_gt_1e4": _rb_tail_share(per_arm[key]), "median": med}
        for i, x in enumerate(xs):
            rows.append([lab, x, len(per_arm[key].get(x, ())),
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    summary[REF_LABEL] = {"n_scored": sum(len(v) for v in ref_per.values()),
                          "strain_gt_1e4": _rb_tail_share(ref_per), "median": med}
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, _rb_ref_window_n(ref_per, xs)[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_rotbond",
              ["arm", "rotatable_bonds", "n", "strain_median", "strain_mean"], rows)
    return summary


def _rb_local_log(xs, summary, ranges, bad):
    print(f"  {len(P79)}-pocket set · "
          + " · ".join(f"{v} x {r[0]}-{r[-1]} rotatable bonds" for v, r in ranges.items())
          + f" (counts where every drawn arm has ≥{MIN_N} molecules)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")
    header = [0, 2, 4, 6, 8, 10, 12]
    print("  strain MEDIAN at a given rotatable-bond count, and the tail behind the mean")
    print(f"  {'arm':16s} {'mols':>7s} " + " ".join(f"{'rb=' + str(h):>7s}" for h in header)
          + f" {'>1e4':>7s}")
    for lab, d in summary.items():
        cells = []
        for h in header:
            v = d["median"][xs.index(h)] if h in xs else None
            cells.append(f"{v:7.1f}" if v is not None else f"{'—':>7s}")
        print(f"  {lab:16s} {d['n_scored']:7d} " + " ".join(cells)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print("  The mean figure is drawn unclipped and spans ~9 decades: the tail above is "
          "what puts it there. Read the median.")


@figure("fig-posecheck-strain-per-rotbond", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_per_rotbond(out):
    """Strain per rotatable bond — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    ranges, all_arms = {}, []
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        ranges[variant] = xs
        all_arms = arms
        for stat in ("median", "mean"):
            _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per)
    summary = _rb_local_csv(out, ranges["core"], all_arms, per_arm, ref_per)
    _rb_local_log(ranges["core"], summary, ranges, bad)


@figure("fig-posecheck-strain-box-per-rotbond", folder="fig-posecheck/strain-energy",
        needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_box_per_rotbond(out):
    """The strain DISTRIBUTION at each rotatable-bond count, as boxes — core only.

    No `all`: see core_only(). The eight-method view is strain_box_per_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per)
        print(f"  {variant:4s} {len(arms)} arms x {xs[0]}-{xs[-1]} rotatable bonds "
              f"({RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers, no fliers)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")


# ── all eight methods ────────────────────────────────────────────────────────────
def _rb_bundle():
    """The published baselines' sample bundle, which is where their molecules are."""
    p = REPO / "results" / "task2-drugdesign"
    if not p.is_dir():
        raise FileNotFoundError(f"missing the baseline sample bundle {p}")
    return p


def _rb_load_meta(folder, stem):
    """export_posecheck_json.py's loader, reproduced exactly: base, then `_part2`
    concatenated PER POCKET. `_gap.pt` ships in the bundle and that loader ignores it, so
    including it here would shift every molecule after the split."""
    import torch                    # local, like RDKit above: nothing else here needs it
    d = _rb_bundle() / folder / "samples" / "meta"
    meta = torch.load(d / f"{stem}.pt", weights_only=False)
    part2 = d / f"{stem}_part2.pt"
    if part2.exists():
        meta = [a + b for a, b in zip(meta, torch.load(part2, weights_only=False))]
    return meta


def _rb_baseline_rows(label, folder, stem, keep):
    """Rows shaped like the shared loader's, for the p79 pockets, with the join checked.

    THE CHECK RUNS OVER EVERY POCKET THE EXPORT HOLDS -- all 100 -- while only the p79 ones
    become rows. Checking just the 79 that are drawn would leave the other 21 as evidence
    nobody looked at, and they cost nothing: the meta is already in memory and the export
    already carries them."""
    meta = _rb_load_meta(folder, stem)
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    by_pocket = collections.defaultdict(list)
    for m in export["molecules"]:
        by_pocket[m["p"]].append(m)

    rows, unparsed, checked = [], 0, 0
    for p in sorted(by_pocket):
        entries = [e for e in meta[p] if e.get("mol") is not None]
        scored = by_pocket[p]
        mine = [int(e["mol"].GetNumAtoms()) for e in entries[:len(scored)]]
        if mine != [m["n"] for m in scored]:
            raise SystemExit(f"{label}: meta/export heavy-atom sequence differs at pocket "
                             f"{p} -- the join is not valid, refusing to guess")
        checked += 1
        if p not in keep:
            continue
        for e, m in zip(entries, scored):
            rb = _rb_rot_bonds(e.get("smiles"))
            unparsed += rb is None
            rows.append({"n": m["n"], "s": m["s"], "c": m["c"], "rb": rb})
    return rows, unparsed, checked


def _rb_locals_from_arms():
    """(label -> rows) for the three arms we run locally, picked BY KEY out of ARMS, plus
    the crystal-ligand rows. See the banner for why it is by key and not a walk."""
    _, p79_rows, refrows = pose_data()
    by_key = {key: (lab, root) for lab, key, root in ARMS}
    series, local = {}, []
    for key in RB_LOCAL_KEYS:
        if key not in by_key:
            raise SystemExit(f"ARMS no longer defines {key!r}")
        lab = RB_RELABEL.get(key, by_key[key][0])
        series[lab] = p79_rows[key]
        local.append(lab)
    RB_LOCAL_LABELS[:] = local
    return series, refrows


def _rb_load_all():
    """(label -> rows) for every drawn method on the rotatable-bond axis, plus the crystal
    ligands and the join-check counts."""
    series, refrows = _rb_locals_from_arms()
    for rows in series.values():
        _rb_attach(rows)
    _rb_attach(refrows)

    keep = {int(t.split("_")[1]) for t in P79}
    checks = []
    for lab, folder, stem in RB_BASELINES:
        rows, bad, checked = _rb_baseline_rows(lab, folder, stem, keep)
        series[lab] = rows
        checks.append((lab, len(rows), bad, checked))
    return series, refrows, checks


def _rb_order(labels):
    """The drug-design table's row order, ours last. Anything the table does not name is a
    bug rather than something to append quietly, so it raises."""
    unknown = [l for l in labels if l not in RB_ORDER]
    if unknown:
        raise SystemExit(f"not in table_drug_design.tex's order: {unknown}")
    return [l for l in RB_ORDER if l in labels]


def _rb_style_of(label):
    return (MODEL_LW, "-") if label in RB_LOCAL_LABELS else (RB_BASE_LW, RB_BASE_DASH[label])

def _rb_panel_order(labels):
    """Same order as `_rb_order`, with the crystal ligands FIRST -- they are the table's
    first row and the thing every other panel is read against."""
    return [REF_LABEL] + _rb_order(labels)


def _rb_handles(labels, solid=False):
    """`solid` for the box figure: nothing in it is a dashed line, so a dashed swatch in the
    key advertises an encoding the panel does not use. The crystal ligands keep their dash
    either way -- there they really are a dashed line."""
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH, label=REF_LABEL)]
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        h.append(Line2D([], [], color=soft(lab), lw=MODEL_LW if solid else lw,
                        ls="-" if solid else ls, label=lab))
    return h


def _rb_line_curve(per_lab, xs, f):
    """The statistic at each exact bond count, null where that count holds fewer than MIN_N
    molecules. THE AXIS IS FIXED AT 0-RB_X_MAX rather than cut back to the counts every
    method can answer: past nine bonds the methods thin out at very different rates --
    Pocket2Mol has 40 ligands at nine and single figures at eleven, CoDE and VoxBind still
    have hundreds -- and the old rule let the emptiest method decide where everyone's line
    stopped. Now each line simply ends where its own method ran out, which is the more
    informative thing to show, and the crystal ligands' dashed line ends earlier still: 79
    of them cannot fill a window at twelve bonds."""
    return [f(per_lab[x]) if len(per_lab.get(x, ())) >= MIN_N else None for x in xs]


def _rb_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, and the key in a strip of its own beneath it.

    NINE SERIES DO NOT LEAVE A CORNER FREE. Inside the axes this key covered FuncBind's
    spike at six bonds and everything above ~500 kcal/mol on the left half -- the part of
    the figure that carries the finding. Under the panel it covers nothing, and it is the
    same 3x3 block the box figure's key is."""
    f = RB_STATS[stat]
    fig, (ax, key) = plt.subplots(2, 1, figsize=(FIG_W, PANEL_H + RB_KEY_H), dpi=220,
                                  gridspec_kw={"height_ratios": [PANEL_H, RB_KEY_H]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key.axis("off")
    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        ax.plot(xs, _rb_line_curve(per[lab], xs, f), color=soft(lab), lw=lw, ls=ls,
                zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    legend(key, _rb_handles(labels), loc="center", ncol=3, fontsize=11)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_rotbond_all_methods_{stat}")


def _rb_axis(labels, refrows, series):
    """The shared x range and per-method rows both figures are drawn over, so the boxes and
    the share panels cannot end up covering different molecules."""
    panels = _rb_panel_order(labels)
    rows_by_lab = {lab: series[lab] for lab in labels}
    rows_by_lab[REF_LABEL] = refrows
    per = {lab: by_size(rows_by_lab[lab], "s", key="rb") for lab in panels}
    dist = {lab: collections.Counter(r["rb"] for r in rows_by_lab[lab]
                                     if r["rb"] is not None) for lab in panels}
    # 0 to RB_X_MAX, the same span the line figures get, so the three are read against one
    # axis BY CONSTRUCTION rather than by today's data happening to agree. Boxes still
    # appear only where a method has RB_GRID_MIN_N molecules, so pinning the axis adds no
    # box; it only stops the axis shrinking when the tail thins.
    xs = list(range(0, RB_X_MAX + 1))
    beyond = {lab: 100 * sum(n for x, n in dist[lab].items() if x > RB_X_MAX)
              / max(sum(dist[lab].values()), 1) for lab in panels}
    return panels, per, dist, xs, beyond


def _rb_all_boxes(out, series, refrows, labels):
    """Strain against rotatable bonds, every method on one axis, dodged inside each count.

    The paper's Fig. 12 shape: one panel of boxplots per number of rotatable bonds, all
    methods together. (The paper packs 151 boxes into one row and colours each box by its
    median strain, spending colour on the value; here colour stays the method key it is in
    every other 260910 figure, and the legend carries it.)

    The crystal ligands stay the dashed grey line rather than becoming a ninth box series:
    79 of them over thirteen bond counts cannot fill a box at every count, and as a line
    they are the same ruler here that they are in every sibling figure."""
    panels, per, dist, xs, beyond = _rb_axis(labels, refrows, series)
    if len(xs) > RB_SPLIT_ABOVE:
        split = len(xs) - len(xs) // 2                 # low row takes the extra count
        bands = [xs[:split], xs[split:]]
    else:
        bands = [xs]
    ref_line = {x: v for x, v in
                zip(xs, _rb_reference_curve(per[REF_LABEL], xs, RB_STATS["median"]))}

    fig, axes = plt.subplots(len(bands), 1,
                             figsize=(FIG_W * RB_BOX_WIDE_1,
                                      PANEL_H * RB_BOX_TALL * len(bands)),
                             dpi=220, squeeze=False)
    fig.patch.set_facecolor("white")
    n = len(labels)
    slot = (1.0 - RB_ALL_BOX_GAP) / n
    for band, ax in zip(bands, axes.ravel()):
        ax.set_facecolor("white")
        for i, lab in enumerate(labels):
            offs = (i - (n - 1) / 2) * slot
            drawn = [x for x in band if x in per[lab] and len(per[lab][x]) >= RB_GRID_MIN_N]
            if not drawn:
                continue
            col = soft(lab)
            bp = ax.boxplot([np.clip(per[lab][x], RB_STRAIN_FLOOR, None) for x in drawn],
                            positions=[x + offs for x in drawn], widths=slot * 0.88,
                            whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                            manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=0.8)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=col, linewidth=0.8)
            for med in bp["medians"]:
                med.set(color=INK, linewidth=1.1, solid_capstyle="butt")
        ax.plot(band, [ref_line.get(x) for x in band], color=REF_COLOR, lw=REF_LW,
                ls=DASH, zorder=6, dash_capstyle="round")
        ax.set_yscale("log")
        # A FIXED FIVE DECADES, AND THE TAIL IS ALLOWED TO RUN OFF THE TOP. FuncBind's
        # 95th percentile at 6 rotatable bonds reaches ~1e11; autoscaling to it stretched
        # the panel over fourteen decades and pressed every box into the bottom fifth. The
        # window is pinned so the boxes -- which all sit between 1e0 and 1e5 -- stay legible
        # across every rebuild, and the whiskers that leave the top simply leave it. How
        # much tail each method carries is reported as `strain_gt_1e4`, not drawn.
        ax.set_ylim(*RB_BOX_YLIM)
        furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct "
                             "whiskers",
                  xlabel=RB_X_LABEL, xlim=(band[0] - 0.62, band[-1] + 0.62), xloc=1)
        ax.set_xticks(band)
    # Nine entries -- the crystal ligands plus eight methods -- in three columns, so the key
    # is a 3x3 block rather than one long strip across the top of the panel. Lower right:
    # the boxes climb left-to-right, so the empty corner is under the high bond counts,
    # where only the lower whiskers reach.
    legend(axes.ravel()[0], _rb_handles(labels, solid=True), loc="lower right",
           fontsize=10, ncol=3)
    fit(fig, pad=0.5)
    save(fig, out, "strain_box_per_rotbond_all_methods")


def _rb_distribution(out, series, refrows, labels):
    """Where each method puts its ligands on the same axis the boxes use -- its own share
    at each rotatable-bond count, one panel per method in a 3x3 block.

    ITS OWN FIGURE, not a strip under the boxes. The two answer different questions and are
    read at different times: the boxes compare methods at a bond count, this compares the
    bond counts a method produces. Sharing a canvas forced one to be a third the height of
    the other, and a nine-panel block does not fit under a box panel at any useful size.

    LOG y. The share spans two decades inside 0-12 bonds -- Pocket2Mol puts 33 % at one bond
    and 0.1 % at twelve -- and on a linear axis everything under ~2 % is a flat line on the
    floor."""
    panels, per, dist, xs, beyond = _rb_axis(labels, refrows, series)
    ncol = RB_DIST_COLS
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(FIG_W * RB_DIST_WIDE, PANEL_H * RB_DIST_TALL * nrow),
                             dpi=220, sharex=True, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for i, lab in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor("white")
        col = REF_COLOR if lab == REF_LABEL else soft(lab)
        total = sum(dist[lab].values())
        pct = [100 * dist[lab].get(x, 0) / total for x in xs]
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, RB_DIST_FLOOR, step="mid", color=col, alpha=DIST_FILL,
                        lw=0, zorder=2)
        ax.set_yscale("log")
        # The x name goes under the BOTTOM ROW only; repeated under all nine it is wider
        # than a panel and the copies overprint each other.
        furniture(ax, ylabel="% of ligands" if i % ncol == 0 else None,
                  xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=2)
        ax.set_ylim(bottom=RB_DIST_FLOOR)
        ax.set_title(lab, fontsize=12, color=INK, loc="left", pad=4)
        ax.text(0.97, 0.07, f">{RB_X_MAX} bonds: {beyond[lab]:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=AXIS, zorder=6)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    # ONE x name for the whole block, centred under the bottom row. sharex already leaves
    # the tick labels on that row alone; three copies of the name did not fit -- each is
    # wider than a panel -- and overprinted each other.
    fig.supxlabel(RB_X_LABEL, fontsize=15.5, color=INK)
    fit(fig, pad=0.5, h_pad=0.9)
    save(fig, out, "rotbond_distribution_all_methods")


def _rb_bin_of(rb):
    return min(int(np.searchsorted(RB_BIN_EDGES, rb, side="right")) - 1,
               len(RB_BIN_LABELS) - 1)


def _rb_all_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels:
        med = _rb_line_curve(per[lab], xs, RB_STATS["median"])
        mean = _rb_line_curve(per[lab], xs, RB_STATS["mean"])
        for i, x in enumerate(xs):
            n = len(per[lab].get(x, ()))
            # the models are drawn per exact count, so their window IS that count
            rows.append([lab, x, n, n, _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    win = _rb_ref_window_n(ref_per, xs)
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, len(ref_per.get(x, ())), win[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_rotbond_all_methods",
              ["arm", "rotatable_bonds", "n", "n_window", "strain_median", "strain_mean"],
              rows)


@figure("fig-posecheck-strain-rotbond-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_all_methods(out):
    """Strain against rotatable bonds, all eight methods: lines, boxes and the share block."""
    use_style()
    series, refrows, checks = _rb_load_all()
    labels = _rb_order(series)
    per = {lab: by_size(series[lab], "s", key="rb") for lab in labels}
    ref_per = by_size(refrows, "s", key="rb")

    # 0 to RB_X_MAX, always -- the same span the box and share figures cover, so the three
    # are read against one axis. Where a method (or the reference) is too thin at a count,
    # its own line stops; the axis does not.
    xs = list(range(0, RB_X_MAX + 1))
    for s in ("median", "mean"):
        _rb_lines(out, xs, per, ref_per, labels, s)
    _rb_all_boxes(out, series, refrows, labels)
    _rb_distribution(out, series, refrows, labels)
    _rb_all_csv(out, xs, per, ref_per, labels)

    for lab, n_rows, bad, checked in checks:
        print(f"  {lab:12s} join checked on {checked} pockets · {n_rows:,} p79 molecules"
              + (f" · {bad} SMILES would not parse" if bad else ""))
    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {xs[0]}-{xs[-1]} rotatable bonds "
          f"(each line drawn where its own method has ≥{MIN_N} molecules)")
    for lab in labels:
        drawn = [x for x in xs if len(per[lab].get(x, ())) >= MIN_N]
        # A break INSIDE a method's span is reported rather than collapsed into the
        # endpoints -- "0-12" over a line with a hole in it would be a false summary.
        gaps = [x for x in range(drawn[0], drawn[-1] + 1) if x not in drawn] if drawn else []
        print(f"    {lab:12s} line drawn {drawn[0]}-{drawn[-1]}"
              + (f", broken at {gaps}" if gaps else "") if drawn else
              f"    {lab:12s} nowhere thick enough to draw")
    ref_drawn = [x for x, v in zip(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]))
                 if v is not None]
    print(f"    {REF_LABEL:12s} line drawn {ref_drawn[0]}-{ref_drawn[-1]} "
          f"(±{RB_REF_WIN} window, ≥{RB_MIN_REF} ligands)")
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{b:>8s}" for b in RB_BIN_LABELS)
          + f" {'>1e4':>7s}   (strain median by bin)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        cells = [[r["s"] for r in rows if r["s"] is not None and r["rb"] is not None
                  and _rb_bin_of(r["rb"]) == i] for i in range(len(RB_BIN_LABELS))]
        floor = RB_MIN_REF if lab == REF_LABEL else 1
        vals = [f"{np.median(v):8.1f}" if len(v) >= floor else f"{'—':>8s}" for v in cells]
        n_scored = sum(len(v) for v in (ref_per if lab == REF_LABEL else per[lab]).values())
        print(f"  {lab:16s} {n_scored:6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(vals) + f" {_rb_tail_share_rows(rows):6.2f}%")


# ── all eight methods, against ligand SIZE ───────────────────────────────────────
def _rb_atom_baseline_rows(label, keep):
    """The export's p79 molecules, as shared-loader-shaped rows."""
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    rows = [{"n": m["n"], "s": m["s"], "c": m["c"]}
            for m in export["molecules"] if m["p"] in keep]
    return rows, len(export["molecules"])


def _rb_atom_load_all():
    series, refrows = _rb_locals_from_arms()
    keep = {int(t.split("_")[1]) for t in P79}
    for lab in RB_ATOM_BASELINES:
        series[lab], _ = _rb_atom_baseline_rows(lab, keep)
    return series, refrows


def _rb_check_decompdiff():
    """The banner's claim, re-checked on every run: one size per pocket, and it is the
    crystal ligand's. If DecompDiff's export is ever replaced by the non-reference-prior
    run this stops holding, and the +-1 window stops being something this figure needs."""
    export = json.load(open(legacy("fig-posecheck", "posecheck_DecompDiff.json")))
    ref = {}
    for t in P79:
        r = rows_of(os.path.join(REF_ROOT, t), reference=True)
        if r:
            ref[int(t.split("_")[1])] = r[0]["n"]
    sizes = collections.defaultdict(set)
    for m in export["molecules"]:
        if m["p"] in ref:
            sizes[m["p"]].add(m["n"])
    one = sum(len(v) == 1 for v in sizes.values())
    same = sum(len(v) == 1 and next(iter(v)) == ref[p] for p, v in sizes.items())
    return one, same, len(sizes)


def _rb_pool(per_lab, a):
    """The molecules behind the point at `a`: its +-RB_WIN neighbours, CLIPPED TO THE AXIS.

    The clip is not cosmetic. DecompDiff has no molecule at 43 or 44 heavy atoms and 125 at
    45+, so an unclipped window put a point at 44 computed entirely from molecules the same
    file reports as beyond the axis -- a plotted value representing nothing near where it
    was plotted. Inside the axis the window still interpolates across its comb, which is
    what it is for; at the edges it no longer extrapolates from outside."""
    return [v for n, vals in per_lab.items()
            if abs(n - a) <= RB_WIN and RB_X_LO <= n <= RB_X_HI for v in vals]


def _rb_curve(per_lab, xs, f):
    """The statistic over that window, None where it is too thin to mean anything -- which
    leaves a gap rather than an invented value, the same rule reference_curve applies to the
    crystal ligands."""
    return [f(p) if len(p) >= MIN_N else None
            for p in (_rb_pool(per_lab, a) for a in xs)]


def _rb_atom_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, NO KEY.

    The key was dropped on request (2026-09-13). It used to be a 3x3 block in a strip of its
    own beneath the panel -- nine series leave no corner of the axes free -- and the strip
    went with it, so the panel is the house single-panel size. The colours and dashes are the
    ones every sibling figure keys (strain_rotbond_all_methods_*), which is where to read them.

    THE MEAN PANEL IS CLIPPED TO THE BULK and the points that leaves off are NAMED in the
    run log: 4-13 % of molecules fail UFF relaxation and land between 1e4 and 1e13, so one
    of them at a thin count carries that count's mean four decades up and an unclipped axis
    spends fourteen decades on it.

    THE REFERENCE KEEPS THE HEADER'S +-4 WINDOW HERE, not the narrow one the rotatable-bond
    figures use: this axis runs 5-44 atoms, which is what that window was set for."""
    f = RB_STATS[stat]
    clip = RB_MEAN_CLIP if stat == "mean" else None
    # DRAWN IN THE STRAIN ECDF'S HOUSE (2026-09-13), so the two strain figures sit side by side
    # as a pair: its canvas (PCSZ_ECDF_SIZE), its rc (12 pt type, black ink, 170 dpi cropped to
    # the ink), its axis furniture and its label size. Read at call time -- the ECDF family is
    # another part of this file. The y name is one line, as the ECDF's are.
    with plt.rc_context(PCSZ_RC):
        fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
        fig.patch.set_facecolor(PCSZ_BG)
        _pcsz_style(ax)
        ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
                ls=DASH, zorder=4, dash_capstyle="round")
        dropped = []
        for lab in labels:
            y = _rb_curve(per[lab], xs, f)
            if clip:
                over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
                if over:
                    dropped.append((lab, over))
            lw, ls = _rb_style_of(lab)
            ax.plot(xs, y, color=soft(lab), lw=lw, ls=ls, zorder=5, solid_capstyle="round")
        ax.set_yscale("log")
        if clip:
            ax.set_ylim(top=clip)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
        ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
        ax.set_ylabel(f"Strain {stat} (kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS)
        fig.tight_layout(pad=0.5)
        save(fig, out, f"strain_per_atom_all_methods_{stat}")
    return dropped


def _rb_atom_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels + [REF_LABEL]:
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        mean = (reference_curve(p, xs, RB_STATS["mean"]) if lab == REF_LABEL
                else _rb_curve(p, xs, RB_STATS["mean"]))
        win = ([sum(len(v) for x, v in p.items() if abs(x - a) <= REF_WIN) for a in xs]
               if lab == REF_LABEL else [len(_rb_pool(p, a)) for a in xs])
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ())), win[i],
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "n_window", "strain_median", "strain_mean"], rows)
    return rows


@figure("fig-posecheck-strain-per-atom-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "metrics.json (posecheck.strain)"))
def draw_posecheck_strain_per_atom_all_methods(out):
    """Strain against ligand size, all eight methods — mean and median."""
    use_style()
    series, refrows = _rb_atom_load_all()
    labels = _rb_order(series)
    dd = _rb_check_decompdiff()
    per = {lab: by_size(series[lab], "s") for lab in labels}
    ref_per = by_size(refrows, "s")
    xs = list(range(RB_X_LO, RB_X_HI + 1))
    drops = {s: _rb_atom_lines(out, xs, per, ref_per, labels, s) for s in ("median", "mean")}
    _rb_atom_csv(out, xs, per, ref_per, labels)

    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, "
          f"drawn where it pools ≥{MIN_N} molecules")
    print(f"  DecompDiff: {dd[0]}/{dd[2]} pockets hold a single heavy-atom count and "
          f"{dd[1]}/{dd[2]} of them equal the crystal ligand's — it is size-matched to the "
          f"reference by construction")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>7d}" for a in heads)
          + f" {'drawn':>9s} {'>44 at':>7s} {'>1e4':>7s}   (strain median at n atoms)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:7.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>7s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        gaps = [a for a in range(drawn[0], drawn[-1] + 1) if a not in drawn] if drawn else []
        span = (f"{drawn[0]}-{drawn[-1]}" + (f"*{len(gaps)}" if gaps else "")) if drawn \
            else "—"
        beyond = round(100 * sum(r["n"] > RB_X_HI for r in rows) / max(len(rows), 1), 2)
        tail = _rb_tail_share_rows(rows)
        print(f"  {lab:16s} {sum(len(v) for v in p.values()):6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(cells) + f" {span:>9s} {beyond:6.1f}% "
              + (f"{tail:6.2f}%" if tail is not None else f"{'—':>7s}"))
        if gaps:
            print(f"    * {lab}'s line breaks inside its span at {gaps} heavy atoms — too "
                  f"few molecules in the window there, not a drawing error")
    for lab, over in drops.get("mean", []):
        pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
        print(f"    strain_per_atom_all_methods_mean · off-panel above {RB_MEAN_CLIP:.0e}: "
              f"{lab:12s} {len(over):2d} of {len(xs)}: {pts}")


# ── all eight methods, the paper's small-multiples form ──────────────────────────
# VoxBind Fig. 12 (arXiv 2405.03961, appendix): ONE PANEL PER METHOD, rotatable bonds on x,
# Tukey boxes WITH fliers, and each box FILLED BY ITS OWN MEDIAN on a diverging colormap. The
# dodged figure above answers "which method is lower at this count"; this one answers "how
# does each method's strain climb with torsional freedom", and the colour lets the reader
# compare levels across panels without lining up y values.
#
# WHAT IS KEPT FROM THE PAPER, WHAT IS NOT. Kept: the grid, x = 0-7, Tukey whiskers (1.5 IQR)
# with fliers, log y, a LINEAR colour norm = median. Changed: nine panels (the crystal ligands
# plus eight methods) in 3x3 rather than 2x4, the house fonts and furniture, and the colormap.
#
# THE COLORMAP IS BUILT FROM THE HOUSE PALETTE, not coolwarm: a light version of CoDE's blue at
# the low end running straight to a light version of VoxBind's sand at the high end, so the
# figure reads in the same two hues as every other 260910 figure.
#
# THE NORM IS A FIXED 0-800 kcal/mol, ticked every 200. Over the eight methods' 64 boxes the
# median is 136 and the 95th percentile 694; only FuncBind at seven and six bonds (982, 5,638)
# lie above 800, and they saturate onto the top colour with the colorbar's arrow saying so
# (the run log names every saturated box). The price of
# a linear norm is that medians under ~100 -- every crystal-ligand box, and VoxBind/CoDE at low
# counts -- share the darkest blues; their differences are read off the y axis, as in the paper.
#
# THE AXIS STOPS AT 7 AS THE PAPER'S DOES, which leaves a real share of every method off it
# (more than the dodged figure's cut at 12). That share is printed per method and written to
# the CSV rather than implied away; raise RB_GRID_X_MAX to see it.
#
# THE Y WINDOW IS THE DODGED FIGURE'S FIXED FIVE DECADES. Fliers that relaxed past 1e5 leave
# the top, as they do in the paper; how many per box is in the CSV (`n_above_ylim`).
#
# STAGES. The paper draws this twice, on the generated pose (Fig. 12) and after a local
# force-field minimisation (Fig. 13). Only `generated` exists in the data: no per-molecule
# minimised strain has been computed for any method yet. A stage is a field name on the rows,
# so adding the minimised one is a new entry here once the rows carry it.
RB_GRID_X_MAX = 7
RB_GRID_COLS = 3
# The crystal ligands are 79 molecules over eight counts; at the methods' floor of
# RB_GRID_MIN_N most of their boxes would vanish. Three is the least that gives a box a median
# and quartiles that are not the same point, and the CSV carries n for every box.
RB_GRID_REF_MIN_N = 3
# (position, colour) stops, all from the shared tables (2026-09-13): CoDE's SOFT blue at 0,
# through its PALE tint and VoxBind's PALE tint, to VoxBind's palette sand at the top -- the
# ends carry the colour, the middle stays light enough for the dark median bars and fliers.
RB_GRID_CMAP_STOPS = [(0.0, soft("CoDE")), (1 / 3, pale("CoDE")), (2 / 3, pale("VoxBind")),
                      (1.0, color("VoxBind"))]
RB_GRID_NORM, RB_GRID_CTICK = (0.0, 800.0), 200.0
RB_GRID_TALL = 0.9975        # per row, as a share of PANEL_H
# The outer names are a step above the house 15.5 pt, and the y name sits further off its
# tick labels -- on a 3x3 block the house sizes read as small.
RB_GRID_LABEL_FS, RB_GRID_CBAR_FS, RB_GRID_YPAD = 17, 15, 16
# Spines and MAJOR tick marks 1.2x the house weight: nine small panels read as washed out at
# 1.35, and 1.6x was too heavy. Minor ticks (the log decades' 2-9) keep their own weight.
RB_GRID_AXIS_LW = AXIS_LW * 1.2
# stage -> (row field, y-axis qualifier)
RB_GRID_STAGES = {"generated": ("s", "generated pose")}


def _rb_grid_boxes(rows, field, floor):
    """[(bond count, clipped values)] for every count on the axis that holds `floor` rows."""
    per = by_size(rows, field, key="rb")
    return [(x, np.clip(per[x], RB_STRAIN_FLOOR, None))
            for x in range(0, RB_GRID_X_MAX + 1) if len(per.get(x, ())) >= floor]


def _rb_grid(out, series, refrows, labels, stage):
    field, qualifier = RB_GRID_STAGES[stage]
    panels = _rb_panel_order(labels)
    rows_by_lab = {**{lab: series[lab] for lab in labels}, REF_LABEL: refrows}
    boxes = {lab: _rb_grid_boxes(rows_by_lab[lab], field,
                                 RB_GRID_REF_MIN_N if lab == REF_LABEL else RB_GRID_MIN_N)
             for lab in panels}

    all_meds = [float(np.median(v)) for lab in panels for _, v in boxes[lab]]
    norm = matplotlib.colors.Normalize(*RB_GRID_NORM)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rb_grid", RB_GRID_CMAP_STOPS)
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(min(all_meds) < norm.vmin, max(all_meds) > norm.vmax)]

    nrow = -(-len(panels) // RB_GRID_COLS)
    fig, axes = plt.subplots(nrow, RB_GRID_COLS, sharex=True, sharey=True,
                             figsize=(FIG_W * RB_BOX_WIDE_1, PANEL_H * RB_GRID_TALL * nrow),
                             dpi=220, squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for ax, lab in zip(flat, panels):
        ax.set_facecolor("white")
        drawn = boxes[lab]
        if drawn:
            bp = ax.boxplot([v for _, v in drawn], positions=[x for x, _ in drawn],
                            widths=0.72, whis=1.5, showfliers=True, patch_artist=True,
                            manage_ticks=False, zorder=5,
                            flierprops=dict(marker="d", markersize=2.6, markerfacecolor=INK,
                                            markeredgecolor="none", alpha=0.5))
            for box, (_, v) in zip(bp["boxes"], drawn):
                box.set(facecolor=cmap(norm(float(np.median(v)))), edgecolor=INK,
                        linewidth=0.9)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=INK, linewidth=0.9)
            for med in bp["medians"]:
                med.set(color=INK, linewidth=1.3, solid_capstyle="butt")
            # Tens of thousands of flier markers as vector paths make the PDF/SVG unplaceable;
            # the points are rasterised inside an otherwise vector figure.
            for fl in bp["fliers"]:
                fl.set_rasterized(True)
        ax.set_yscale("log")
        ax.set_ylim(*RB_BOX_YLIM)
        xlo = -0.6
        furniture(ax, xlim=(xlo, RB_GRID_X_MAX + 0.6), xloc=1)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(RB_GRID_AXIS_LW)
        ax.tick_params(which="major", width=RB_GRID_AXIS_LW)
        # An unlabelled tick where the axes meet, matching the y axis's 10^0 in that corner:
        # the first bond count sits 0.6 in, so without it the x axis looks cut short.
        counts = list(range(0, RB_GRID_X_MAX + 1))
        ax.set_xticks([xlo] + counts, [""] + [str(x) for x in counts])
        ax.set_title(display(lab), fontsize=14, color=INK, pad=5)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    fig.supxlabel(RB_X_LABEL, fontsize=RB_GRID_LABEL_FS, color=INK)
    # The y name goes on the MIDDLE row's left axes rather than fig.supylabel, because only an
    # axes label takes a labelpad -- supylabel sits flush against the tick labels. With three
    # rows the middle axes' centre is the block's centre.
    axes[nrow // 2, 0].set_ylabel(f"UFF strain energy (kcal mol⁻¹), {qualifier}",
                                  fontsize=RB_GRID_LABEL_FS, color=INK,
                                  labelpad=RB_GRID_YPAD)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, extend=extend,
                      shrink=0.92, aspect=38, pad=0.012,
                      ticks=np.arange(RB_GRID_NORM[0], RB_GRID_NORM[1] + RB_GRID_CTICK,
                                      RB_GRID_CTICK))
    cb.set_label("Median strain energy (kcal mol⁻¹)", fontsize=RB_GRID_CBAR_FS, color=INK)
    cb.ax.tick_params(labelsize=12, colors=AXIS, width=AXIS_LW)
    cb.outline.set_edgecolor(AXIS)
    cb.outline.set_linewidth(AXIS_LW)
    save(fig, out, f"strain_box_per_rotbond_grid_{stage}")

    csv_rows = []
    for lab in panels:
        for x, v in boxes[lab]:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            csv_rows.append([lab, x, len(v), round(float(med), 3), round(float(q1), 3),
                             round(float(q3), 3), int(np.sum(v > RB_BOX_YLIM[1]))])
    write_csv(out, f"strain_box_per_rotbond_grid_{stage}",
              ["method", "rotatable_bonds", "n", "strain_median", "strain_q25", "strain_q75",
               "n_above_ylim"], csv_rows)

    sat = [(lab, x, round(float(np.median(v)))) for lab in panels for x, v in boxes[lab]
           if not norm.vmin <= float(np.median(v)) <= norm.vmax]
    print(f"  {stage}: {len(panels)} panels · x = 0-{RB_GRID_X_MAX} rotatable bonds · "
          f"colour linear {norm.vmin:g}-{norm.vmax:g} kcal/mol (extend={extend}) · "
          f"saturated boxes {sat}")
    for lab in panels:
        rows = [r for r in rows_by_lab[lab] if r[field] is not None and r["rb"] is not None]
        beyond = 100 * sum(r["rb"] > RB_GRID_X_MAX for r in rows) / max(len(rows), 1)
        top = sum(int(np.sum(v > RB_BOX_YLIM[1])) for _, v in boxes[lab])
        print(f"    {lab:16s} {len(boxes[lab])} boxes · {len(rows):6d} scored · "
              f">{RB_GRID_X_MAX} bonds {beyond:5.1f}% (not drawn) · {top} fliers above "
              f"{RB_BOX_YLIM[1]:.0e}")


@figure("fig-posecheck-strain-rotbond-grid", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_grid(out):
    """Strain per rotatable bond, one panel per method, boxes coloured by median (VoxBind
    Fig. 12 form)."""
    use_style()
    series, refrows, _ = _rb_load_all()
    labels = _rb_order(series)
    for stage in RB_GRID_STAGES:
        _rb_grid(out, series, refrows, labels, stage)


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


# ════════════════════════════════════════════════════════════════════════════════
# fig-vina-per-atom — Vina Dock against ligand size, in the fig-vina-3line style
# ════════════════════════════════════════════════════════════════════════════════
# TWELVE FILES: three reference variants x two statistics x two arm sets, plus one CSV
# behind all of them. Which statistic a figure shows is carried by its filename and by its
# y-axis name ("Vina Dock / mean" / "Vina Dock / median", set over two lines so the pair of
# panels keeps one narrow left margin), exactly as in the 3-line figures -- there are no
# panel titles, so the y name is the only thing telling two otherwise identical figures
# apart. One CSV serves all twelve: the per-arm columns do not depend on which arms are
# drawn, and `plotted_in`, `reference_source` and `reference_firm` are the columns that
# carry what does.
#
# ── WHICH LIGANDS THE GREY CURVE IS ──────────────────────────────────────────────
# There are two possible reference populations. Which one is drawn is stated on EVERY run,
# in the progress line and in the CSV's `reference_source` and `n_reference_total` columns.
# The KEY names it only when it is the CrossDocked set -- "the 79" quietly passed off as
# "CrossDocked" is the mistake this guards against, and that mistake only runs one way.
#
# WHY THE KEY IS ASYMMETRIC, since a matching "(79 pockets)" looks more careful and is not.
# Every other figure in this section draws "Reference ligand" and means these same 79
# crystal ligands. Annotating it here alone would put one key at odds with its neighbours
# while describing an identical population, and a reader moving between them would
# reasonably infer that THIS reference is some other set. It is not. The ambiguity an
# annotation prevents does not exist until a second population does, and at that point the
# CrossDocked label carries the distinction exactly where it is needed. The bare label also
# keeps v1 and v2 byte-identical to the figures this port was verified against, which is
# the check the whole family rests on: between an annotation that is redundant in every
# figure already meaning the 79, and losing that check, the annotation goes. Provenance is
# not lost either way -- the log and the CSV carry it unconditionally, which is where a
# reader who needs it will look.
#
#   p79          The 79 crystal ligands of the test set, one per pocket: the docked value
#                out of each run's eval_docking_results_full79.json, the ligand's own
#                heavy-atom count out of that pocket's metrics.json `reference` block.
#                Drawn as plain "Reference ligand", the same key every other figure in the
#                section uses for these same molecules. This is the fallback.
#   crossdocked  The whole CrossDocked set, docked per ligand on another box and dropped
#                into the Dropbox-backed bundle at
#                results/task2-drugdesign/Reference/crossdocked_reference_dock.json,
#                which arrives with `bash results/dropbox_pull.sh Reference`. Drawn as
#                "Reference ligand (CrossDocked, n=...)". Schema, so the upload matches:
#
#                    {"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1},
#                                                    ...]}
#
#                `ligands` must be a non-empty list and every entry must carry both
#                `n_atoms` and `vina_dock`; anything else in the file is ignored, and
#                `protocol` is echoed into the run log so the figure states what it was
#                docked under. A file that does not parse fails LOUDLY with its path and
#                this schema rather than falling back to the 79 behind the reader's back.
#
# The file decides: crossdocked when it is there, p79 when it is not.
# VOXBIND_VINA_REFERENCE=p79|crossdocked forces one, and errors if the forced one is not
# available. The choice is printed either way.
#
# THIS IS WHY THE THINNESS LOGIC STAYS even though CrossDocked will mostly retire it. With
# tens of thousands of ligands behind it the rolling window is never thin and the fade
# never fires, which is the whole point of the upload; with the 79 it is thin at both ends
# and the fade is the honesty cue. The code does not need to know which -- it fades what is
# thin.
#
# ── TWO ARM SETS ─────────────────────────────────────────────────────────────────
# The same split every other figure in this section makes:
#
#   core  VoxBind and CoDE -- the two arms the figure exists to compare, and the only two
#         with an area under their distribution.
#   all   every arm that ACTUALLY HAS per-molecule Vina on this box, which today is those
#         two plus TargetDiff.
#
# WHAT AN ARM NEEDS TO BE IN `all`: `<its root>/eval_docking_results_full79.json`, carrying
# `per_target[].per_mol[]` entries with both `n_atoms` and `vina_dock`. An arm whose file is
# absent is SKIPPED WITH A PRINTED NOTE naming the path, not a crash and not a silent drop,
# so the day a file lands `all` simply grows -- adding a method is data, not surgery.
#
# The five published baselines are listed in VPA_ARMS and are skipped today. AR,
# Pocket2Mol, DiffSBDD, DecompDiff and FuncBind are staged here as sample directories
# (exps/baselines_pose/<key>/target_XX/) and scored for PoseBusters and PoseCheck, but
# their metrics.json carries `docking: "none"`: the Vina run behind
# results/task2-drugdesign/_shared/baselines_eval/summary_density79.json happened on svr12
# and only its AGGREGATES came back, so there is no (heavy atoms, vina_dock) pair per
# molecule to bin. Docking them here under this protocol is ~89 CPU-hours per method
# (79 pockets x ~100 molecules, exhaustiveness 32, whole receptor). So `all` DOES NOT MEAN
# "all eight published methods" today, and the run log says which arms it did mean.
#
# The two sets do not share an x span, and that is the point of drawing both: x runs only
# where EVERY drawn arm clears MIN_N, so dropping TargetDiff buys a count back at the right
# end (core reaches 45, all stops at 44) and removes the two interior counts where
# TargetDiff alone is thin.
#
# ── THREE VARIANTS ───────────────────────────────────────────────────────────────
# ONE FUNCTION. They differ only in what the crystal reference does, over the three knobs
# in VPA_VARIANTS below: whether it is on the Vina panel, whether it is on the distribution
# strip, and how thin its rolling window may get before it stops being drawn at full
# strength. They are kept together on purpose: split apart they would drift, and then the
# "same figure, one choice different" claim would quietly stop being true.
#
#   v1  The reference's Vina curve is GATED: drawn only where its rolling window holds at
#       least VPA_REF_FIRM ligands, so on the 79 the grey stops at 36 atoms. Its
#       distribution below is smoothed and filled like the models'.
#   v2  The reference's Vina curve runs the full x range, FADED where the window thins
#       (below VPA_REF_FIRM, from 37 atoms up on the 79). Its distribution is smoothed but
#       UNFILLED, so the two model areas underneath stay readable.
#
#       v2's cost is real and visible: on the 79 the reference median at 42-44 is two
#       crystal ligands, TNKS1 (target_72, -15.99) and AKT1 (target_80, -14.66), and
#       drawing them stretches y to -15.5, which compresses the -3..-11 band where the arms
#       are actually being compared. v1 spends nothing on y but ends its grey eight atoms
#       short of where the panel below it ends, which reads as missing data rather than as
#       thin data. Neither is free; that is why both are built.
#   v3  v2's Vina panel with the gate opened all the way, and NO reference on the
#       distribution strip. The grey curve is drawn at every count whose +-REF_WIN window
#       holds so much as one ligand -- nothing is dropped for being thin, only faded below
#       VPA_REF_FIRM, which is the honesty cue doing that job -- and the strip below carries
#       the model arms alone. The reference keeps its key entry, because it is still drawn.
#
#       WHAT V3 GIVES UP. It used to drop the reference from BOTH panels, and its selling
#       point was the y axis: nothing on the panel reached past the models, so the -3..-11
#       band they are compared in filled the height. Drawing the grey again hands that
#       back -- on the 79, TNKS1 and AKT1 sit under 42-44 atoms and stretch y to -15.5, the
#       same squeeze v2 pays. What it keeps is the benchmark on the panel where the
#       comparison is actually made, and what it buys is the strip: with no grey step or
#       fill there, the model areas are read without a fourth series over them.
#
#       ON THE 79 THE OPEN GATE CHANGES NOTHING. Over the plotted span that window never
#       falls below three ligands -- its minimum, at 43-45 atoms -- which v2's floor already
#       admits, so v3's grey curve and v2's are the same line and the two variants differ
#       only in the strip below. The open gate is what the figure does when x reaches
#       further right: the 79 crystal ligands span 6 to 57 heavy atoms but only five sit
#       past 40, and from 46 up the window is down to one or two. Against the CrossDocked
#       reference it is moot in the other direction -- nothing is thin.
#
# ── WHAT THEY SHOW ───────────────────────────────────────────────────────────────
# Two stacked panels over one x axis, the ligand's heavy-atom count.
#
#   top     Vina Dock mean or median at each exact heavy-atom count, for every arm in the
#           set, and the reference ligands.
#   bottom  How many molecules each set puts at each size, as a share of its own molecules
#           -- a share and not a count, because the reference set and ~7,900 generated
#           molecules do not share a count axis. This is what makes the top panel's tails
#           trustworthy or not, and is itself the finding the figure exists to guard
#           against: the arms differ in the size distribution they generate at least as
#           much as in per-atom binding quality (see the size-confound note; raw pooled
#           Vina Dock is ~80% a size statistic). In v1 and v2 the reference is on this
#           panel too, so "does the model generate ligands the size of the real one" is
#           readable without a second figure; v3 is the variant that gives that up.
#
# EVERY ARM IS GATED THE SAME WAY, per arm rather than per figure. A heavy-atom count
# enters x only if EVERY drawn arm has MIN_N molecules there, which sets the two ends;
# inside that span an arm that dips below MIN_N (TargetDiff does, at 38 and 41 atoms, which
# is why only `all` is broken there) has its curve BROKEN rather than interpolated, because
# a bridge over a count where one method generated 15 molecules is a drawn claim about data
# that is not there. The distribution panel underneath keeps drawing at those counts --
# being thin is exactly what it is for.
#
# ONE STATISTIC PER FIGURE. Both were drawn in one panel at first, dash against solid, and
# it said nothing extra for the ink: at every size the two run within ~0.2 kcal/mol of each
# other, so six curves were three curves drawn twice and the pair merely thickened and
# blurred each series. They are separate figures instead. That near-agreement IS a result
# -- the Ours/VoxBind gap is the whole distribution shifting, not a tail dragging the mean
# -- and it is what makes the two figures nearly interchangeable; the mean is the slightly
# kinder one to CoDE (below VoxBind at 35/40 sizes against the median's 31/40, mean
# per-size gap -0.441 against -0.388 on the `all` span), because CoDE's size distribution
# reaches further right.
#
# INK IS THE SECOND CHANNEL, and it says what a series is FOR: the two arms being compared
# are the thickest, solid, and the only ones with an area under their distribution;
# everything else is thinner and DASHED, because it is what the pair is read against rather
# than a further competitor, and four filled areas on the lower panel would be a stack no
# one can read through. That is also the rule any newly docked baseline arrives under: it
# joins as context ink, not as a third solid curve. CoDE is periwinkle #8291E8 here, the
# lighter tint of our blue this Vina family carries, as the palette records.
#
# PROTOCOL. The published-baseline protocol, exactly as in the 3-line figures: whole
# `*_rec.pdb` receptor, exhaustiveness 32, all 79 pockets, from
# `eval_docking_results_full79.json`. The same runs, so this figure and the 3-line figures
# sit on one axis -- TargetDiff included, which 74_dock_targetdiff_full79.sh re-docked here
# for exactly that reason. The CrossDocked reference is docked elsewhere, so its own
# `protocol` block is echoed into the run log rather than assumed to match.
#
# WHY NO CONFIDENCE BANDS. The thing a band would guard against here is already drawn: x is
# clipped to the counts where EVERY arm has at least MIN_N molecules, so no curve has a
# tail the others cannot answer, and the bottom panel shows how thin each end actually is.
# The per-count sample sizes are in the CSV.
#
# THE REFERENCE IS ROLLED. On the 79 there is exactly ONE crystal ligand per pocket, 1-6 at
# any exact heavy-atom count, so anything per-count off it is noise: raw, its distribution
# is a picket fence of 1.3%-tall steps reaching 7.6% where six pockets happen to share a
# size, and that spike, not the models, would set the lower panel's y scale. Both its
# curves are therefore a centred rolling window of +-REF_WIN atoms, and the window stays
# for the CrossDocked set so the two sources are read the same way. The models are NOT
# rolled, in either panel: they have hundreds of molecules per count.
VPA_BASENAME = "eval_docking_results_full79.json"
VPA_PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# The CrossDocked reference drop, REPO-relative because this tree gets copied between
# boxes. The folder is the Dropbox-backed bundle, so the upload lands with
# `bash results/dropbox_pull.sh Reference` and needs no sync path of its own.
VPA_REF_DOCK = ("results", "task2-drugdesign", "Reference",
                "crossdocked_reference_dock.json")
VPA_REF_SOURCES = ("p79", "crossdocked")
VPA_REF_ENV = "VOXBIND_VINA_REFERENCE"
VPA_REF_SCHEMA = ('{"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1}, '
                  '...]}  -- extra keys ignored')

# The whole difference between the three figures, and all of it about the reference.
# `ref_vina` puts it on the top panel, `ref_dist` on the strip below; it earns its key
# entry by being on either. `ref_gate` names the floor its rolling window has to clear
# before a point is drawn at all -- VPA_REF_GATE maps the three settings to that number --
# and everything except "hard" fades the stretch below VPA_REF_FIRM rather than cutting it.
# `ref_dist_smooth` rolls the reference's distribution; `ref_dist_fill` gives it an area
# under it like the two compared arms have.
VPA_VARIANTS = {
    "v1": dict(ref_vina=True, ref_dist=True, ref_gate="hard",
               ref_dist_smooth=True, ref_dist_fill=True),
    "v2": dict(ref_vina=True, ref_dist=True, ref_gate="fade",
               ref_dist_smooth=True, ref_dist_fill=False),
    "v3": dict(ref_vina=True, ref_dist=False, ref_gate="full",
               ref_dist_smooth=True, ref_dist_fill=False),
}

VPA_FIELD, VPA_REF_FIELD = "vina_dock", "ref_vina_dock"
VPA_STATS = ("mean", "median")       # one figure each, per variant and per arm set

# The y names are set over TWO LINES. One line each ("Vina Dock median", "% of ligands")
# put the widest label's length into the left margin of a 7.6-inch figure; broken after
# the metric they are the same words in half the width, and align_ylabels below keeps the
# two panels' labels on one left edge. The statistic still lives in the y name and
# nowhere else, exactly as in the 3-line figures.
VPA_Y_LABEL = "Vina Dock\n{stat}"
VPA_D_LABEL = "Generated\nfraction"

# These are DATA KEYS as well as labels (they head this folder's CSV columns), so they
# stay plain; display() dresses them for a legend -- VoxBind picks up its sigma=0.9
# subscript there. CoDE is \textsc{CoDE} in the .tex files; matplotlib has no small caps
# without a TeX backend, so the figures carry the plain string.
VPA_VOX_LABEL, VPA_OUR_LABEL = "VoxBind", "CoDE"
VPA_OUR_COLOR = soft(VPA_OUR_LABEL)

# Line weights in points. VPA_MODEL_BOOST widens ONLY the two arms the figure exists to
# compare -- both their Vina curves and their distribution steps -- and is applied as a
# factor on the shared base weights rather than baked into them, so the gap it opens
# against the context series' thinner ink stays visible as a decision.
VPA_MODEL_BOOST = 1.2
VPA_MODEL_LW, VPA_BASE_LW = MODEL_LW * VPA_MODEL_BOOST, 1.9
VPA_DIST_LW = DIST_LW * VPA_MODEL_BOOST

# The reference dash is the shared DASH: legible at hairline weights, and it does not
# shimmer where it runs close to a model curve. VPA_BASE_DASH is the context arms', longer
# so the two kinds of dashed series are told apart by rhythm as well as by colour.
VPA_BASE_DASH = (0, (6, 2))

# label, run root, line width, dash -- DRAWN IN THIS ORDER, so ours lands on top and the
# context series sit under it. Roots are REPO-relative (E) or resolved off the sibling
# checkout (BASEDRUG); nothing here is an absolute path. The five baselines are listed and
# will be SKIPPED until their eval_docking_results_full79.json exists, which is how a newly
# docked method joins `all` without a code change.
VPA_ARMS = [
    ("AR",          f"{E}/baselines_pose/ar",         VPA_BASE_LW, VPA_BASE_DASH),
    ("Pocket2Mol",  f"{E}/baselines_pose/pocket2mol", VPA_BASE_LW, VPA_BASE_DASH),
    ("DiffSBDD",    f"{E}/baselines_pose/diffsbdd",   VPA_BASE_LW, VPA_BASE_DASH),
    ("DecompDiff",  f"{E}/baselines_pose/decompdiff", VPA_BASE_LW, VPA_BASE_DASH),
    ("FuncBind",    f"{E}/baselines_pose/funcbind",   VPA_BASE_LW, VPA_BASE_DASH),
    ("TargetDiff",  str(BASEDRUG / "eval" / "targetdiff"), VPA_BASE_LW, VPA_BASE_DASH),
    (VPA_VOX_LABEL, f"{E}/_vanilla_ep923/samples/full_eval_ep923", VPA_MODEL_LW, "-"),
    (VPA_OUR_LABEL, f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
     VPA_MODEL_LW, "-"),
]
# The two arms this figure exists to compare: the only ones drawn solid, and the only ones
# given an area under their distribution. Everything else on the panel is context -- and
# `core` is exactly this pair with the context removed, which is why one tuple names both.
VPA_FOCUS = (VPA_VOX_LABEL, VPA_OUR_LABEL)
# Every arm takes its colour from the shared palette, so a method is the same colour here
# as in the PoseBusters and PoseCheck figures. CoDE is the one override: this Vina family
# carries the lighter tint of our blue.
VPA_COLORS = {lab: (VPA_OUR_COLOR if lab == VPA_OUR_LABEL else color(lab))
              for lab, *_ in VPA_ARMS}
VPA_COLORS[REF_LABEL] = REF_COLOR
# The p79 crystal ligand is the same molecule in every run that carries one, but it is
# resolved BY LABEL, never by position in VPA_ARMS -- the staged baselines have no
# reference block at all, and a positional pick would silently become one of them.
VPA_REF_ROOT = next(root for lab, root, _, _ in VPA_ARMS if lab == VPA_VOX_LABEL)

# The reference window is +-REF_WIN atoms, and these are the three floors it can be held
# to. VPA_REF_FIRM is where it holds enough ligands to be read as a curve (v1 gates there,
# v2 and v3 fade below it); VPA_REF_FLOOR is v2's floor for drawing a point at all;
# VPA_REF_ANY is v3's, which refuses only a window holding nothing. The curve is computed
# once at VPA_REF_ANY and masked down, so one array serves all three gates and the CSV
# reports every value any of them can draw.
VPA_REF_FIRM, VPA_REF_FLOOR, VPA_REF_ANY = 12, 3, 1
VPA_REF_GATE = {"hard": VPA_REF_FIRM, "fade": VPA_REF_FLOOR, "full": VPA_REF_ANY}
VPA_REF_THIN_ALPHA = 0.42


def _vpa_slug(label):
    """The CSV's column prefix for an arm: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


# Bundle folder for an arm that has no run tree here. The five published baselines were
# docked on svr12 and only their aggregates came back until 2026-09-13, when the bundle
# gained <Method>/eval/vina_docking/per_molecule.csv -- target, n_atoms and the three Vina
# columns per molecule, which is exactly what this figure bins. They are read from THERE.
# Our own arms keep their run tree: the bundle's `VoxBind-vanilla` is res_test_100, a
# DIFFERENT run from the _vanilla_ep923 this figure has always drawn, and swapping the
# source underneath an arm would silently redraw it.
VPA_BUNDLE = {"AR": "AR", "Pocket2Mol": "Pocket2Mol", "DiffSBDD": "DiffSBDD",
              "DecompDiff": "DecompDiff", "FuncBind": "FuncBind"}


def _vpa_load_bundle(label):
    """{heavy atoms: [vina dock, ...]} over the 79 density pockets, from the bundle CSV."""
    folder = VPA_BUNDLE.get(label)
    rows = vina_rows(folder, density79=True) if folder else None
    if not rows:
        return None, 0
    sizes = collections.defaultdict(list)
    for r in rows:
        if r.get(VPA_FIELD) is not None and r.get("n"):
            sizes[r["n"]].append(r[VPA_FIELD])
    return (sizes, len({r["target"] for r in rows})) if sizes else (None, 0)


def _vpa_load(root, label=None):
    """{heavy atoms: [vina dock, ...]} pooled over the 79 pockets of one run, or None if
    this arm has not been docked here -- an absent file is a method waiting for its Vina
    run, not an error, and the caller says so in the log instead of drawing an empty
    curve. An arm with no run tree falls to its bundle CSV (see VPA_BUNDLE)."""
    path = os.path.join(root, VPA_BASENAME)
    if not os.path.exists(path):
        return _vpa_load_bundle(label) if label else (None, 0)
    with open(path, encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes = collections.defaultdict(list)
    for t in per_target:
        for m in t.get("per_mol") or []:
            if isinstance(m, dict) and m.get(VPA_FIELD) is not None and m.get("n_atoms"):
                sizes[m["n_atoms"]].append(m[VPA_FIELD])
    return sizes, len(per_target)


def _vpa_load_reference(root):
    """The p79 reference: {heavy atoms: [crystal-ligand vina dock, ...]}, one entry per
    pocket.

    The docked value is in the results file, but the ligand's own atom count is not -- it
    lives in that pocket's metrics.json, under `reference`. Pockets missing either are
    dropped and named by the caller."""
    with open(os.path.join(root, VPA_BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes, dropped = collections.defaultdict(list), []
    for t in per_target:
        path = os.path.join(root, t["target"], "metrics.json")
        n_atoms = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                n_atoms = (json.load(handle).get("reference") or {}).get("n_atoms")
        value = t.get(VPA_REF_FIELD)
        if n_atoms and value is not None:
            sizes[n_atoms].append(value)
        else:
            dropped.append(t["target"])
    return sizes, dropped


def _vpa_load_crossdocked(path):
    """The CrossDocked reference drop: {heavy atoms: [vina dock, ...]} over every ligand in
    it, plus whatever `protocol` block it carries.

    Every failure raises with the path AND the schema. This file is written on another box
    and pulled in, so the realistic failure is a shape mismatch, and the one thing that
    must never happen is falling back to the 79 while the key still says CrossDocked."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    ligands = payload.get("ligands") if isinstance(payload, dict) else None
    if not isinstance(ligands, list) or not ligands:
        raise ValueError(f"{path}: needs a non-empty 'ligands' list\n"
                         f"  schema: {VPA_REF_SCHEMA}")
    sizes = collections.defaultdict(list)
    for i, m in enumerate(ligands):
        if not isinstance(m, dict) or m.get("n_atoms") is None \
                or m.get(VPA_FIELD) is None:
            raise ValueError(
                f"{path}: ligand {i} has no 'n_atoms' and/or '{VPA_FIELD}'\n"
                f"  schema: {VPA_REF_SCHEMA}")
        sizes[int(m["n_atoms"])].append(float(m[VPA_FIELD]))
    return sizes, payload.get("protocol")


def _vpa_reference_set():
    """(source, {heavy atoms: [vina dock]}, the key's label, the log line, ligand count).

    The file decides -- CrossDocked when the drop is there, the 79 when it is not -- and
    VPA_REF_ENV forces one either way, erroring rather than silently substituting the
    other. Whichever wins, its name goes in the key, in the log and in the CSV."""
    path = REPO.joinpath(*VPA_REF_DOCK)
    forced = os.environ.get(VPA_REF_ENV) or None
    if forced is not None and forced not in VPA_REF_SOURCES:
        raise ValueError(f"{VPA_REF_ENV}={forced!r}: expected one of "
                         f"{' or '.join(VPA_REF_SOURCES)}")
    if forced == "crossdocked" and not path.exists():
        raise FileNotFoundError(
            f"{VPA_REF_ENV}=crossdocked, but {path} is not there\n"
            f"  pull it with: bash results/dropbox_pull.sh Reference\n"
            f"  schema: {VPA_REF_SCHEMA}")
    if forced == "p79" or (forced is None and not path.exists()):
        sizes, dropped = _vpa_load_reference(VPA_REF_ROOT)
        n = sum(len(v) for v in sizes.values())
        note = (f"p79 · {n} crystal ligands, one per pocket, from {VPA_BASENAME}"
                + (f" (dropped {', '.join(dropped)})" if dropped else ""))
        return "p79", sizes, REF_LABEL, note, n
    sizes, protocol = _vpa_load_crossdocked(path)
    n = sum(len(v) for v in sizes.values())
    if isinstance(protocol, dict):
        protocol = ", ".join(f"{k}={v}" for k, v in protocol.items())
    note = (f"crossdocked · {n:,} ligands from {path.relative_to(REPO)}"
            + (f" · docked under {protocol}" if protocol else ""))
    # One line. It is roughly three times the width of a method label, which is exactly
    # why it cannot sit INSIDE the method grid: a legend column is as wide as its widest
    # entry, so dropping this into one would stretch that column and break the 2 x 4. It
    # gets its own centred key above the grid instead.
    #
    # The count alone carries the distinction the annotation exists for: the p79 set is 79
    # and this one is six figures, so "(n=100,090)" cannot be mistaken for the test
    # pockets' ligands, and the bare "Reference ligand" keeps meaning the 79 exactly as it
    # does in every other figure of the section. Which population it is, and under what
    # protocol, is printed in full on every run and carried in the CSV.
    return "crossdocked", sizes, f"{REF_LABEL} (n={n:,})", note, n


def _vpa_arm_sets(arms):
    """(name, arms) for the two sets, core first. `arms` is what actually has data."""
    return (("core", [a for a in arms if a[0] in VPA_FOCUS]), ("all", list(arms)))


def _vpa_model_curve(sizes, xs, stat):
    """mean/median at each exact heavy-atom count, None below MIN_N.

    None and not the value: a count an arm put fewer than MIN_N molecules at is a number
    the other arms cannot be asked to answer, and plotting it would also let matplotlib
    draw a straight segment over it as though the arm had been measured there."""
    f = st.mean if stat == "mean" else st.median
    return [f(sizes[a]) if len(sizes.get(a, ())) >= MIN_N else None for a in xs]


def _vpa_window_counts(sizes, xs):
    """How many reference ligands each plotted point's rolling window pools."""
    return [sum(len(v) for n, v in sizes.items() if abs(n - a) <= REF_WIN) for a in xs]


def _vpa_reference_curve(sizes, xs, stat, floor):
    """The reference over a centred +-REF_WIN window; None where the window holds fewer
    than `floor` ligands, which leaves a gap in the line rather than an invented value."""
    f = st.mean if stat == "mean" else st.median
    out = []
    for a in xs:
        pool = [v for n, vals in sizes.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(pool) if len(pool) >= floor else None)
    return out


def _vpa_gated(values, counts, gate):
    """The reference curve held to one variant's floor: the value where its window clears
    VPA_REF_GATE[gate], None below. The curve itself is computed once at VPA_REF_ANY, so
    the three gates are three masks over one array rather than three recomputations."""
    floor = VPA_REF_GATE[gate]
    return [v if c >= floor else None for v, c in zip(values, counts)]


def _vpa_split_firm(values, counts):
    """The reference curve cut into the part its window can carry and the part it cannot.

    Returns (firm, thin): both full-length, each holding None wherever the other holds the
    value. A point on the boundary belongs to BOTH, so the solid and the faded stretch meet
    rather than leaving a gap -- a gap would say "no data here", which is the opposite of
    what the fade is for."""
    firm_at = [v is not None and c >= VPA_REF_FIRM for v, c in zip(values, counts)]
    n = len(firm_at)
    edge = [firm_at[i] and any(not firm_at[j] for j in (i - 1, i + 1) if 0 <= j < n)
            for i in range(n)]
    firm = [v if f else None for v, f in zip(values, firm_at)]
    thin = [v if (not f) or e else None for v, f, e in zip(values, firm_at, edge)]
    return firm, thin


def _vpa_furniture(ax, *, ylabel, xlabel, xs):
    """The 3-line figure's axes, over the plotted heavy-atom span. The y names are two
    lines each, so they are centred on one another rather than left-ragged."""
    furniture(ax, ylabel=ylabel, xlabel=xlabel, xlim=(xs[0] - 0.6, xs[-1] + 0.6),
              xloc=XTICK_STEP)   # ticks and vertical rules both, unlike the 3-line figure
                                 # where the rules fall between the labelled ranks
    ax.yaxis.label.set_multialignment("center")


# Above this many entries the key stops fitting in the panel's empty corner and starts
# covering the curves it explains. `core` has three and keeps the corner; `all` has nine
# and takes a strip under the figure instead.
VPA_KEY_INSIDE_MAX = 4
# Inside that strip the reference is NOT one of the methods -- it is the benchmark they are
# read against -- so it takes its own centred line above a 2 rows x 4 columns grid of the
# eight methods, which is the shape that fits this figure's width without reaching the axes.
VPA_KEY_NCOL = 4
# The two lines are laid out as two legends and then wrapped in one frame; this is the gap
# left between them, in figure fractions, INSIDE that shared frame.
VPA_KEY_ROW_GAP = 0.002


def _vpa_row_major(handles, ncol):
    """Reorder for matplotlib's column-major legend fill, so the key reads ACROSS.

    A legend with ncol=4 lays its entries down column 1, then column 2 -- so the drawing
    order the eye expects along a row is not the order it gets. Interleaving here puts the
    entries back in reading order."""
    rows = -(-len(handles) // ncol)
    out = []
    for c in range(ncol):
        for r in range(rows):
            i = r * ncol + c
            if i < len(handles):
                out.append(handles[i])
    return out


def _vpa_legend(ax, series, ref_label, *, fig=None):
    """The key, drawn in plotting order, its handles carrying each series' dash so an
    identity is readable from the key alone.

    WHERE IT GOES DEPENDS ON HOW MANY THERE ARE. Three series fit in the empty upper right
    and belong there -- next to the data, costing no height. Nine do not: at this panel
    size a nine-row box reaches more than halfway down and across, and in the `all` variant
    it sat squarely over the x axis. Past VPA_KEY_INSIDE_MAX the key moves beneath the
    figure as a reference row over a 2x4 grid of methods.

    `ref_label` is bare "Reference ligand" for the 79 -- the section's own name for them --
    and names the size only for the CrossDocked set, which is the one that would otherwise
    be read as the 79 (see the banner)."""
    handle = lambda lab, colour, lw, style: Line2D(
        [], [], color=colour, lw=lw, ls=style,
        label=ref_label if lab == REF_LABEL else display(lab))
    handles = [handle(*a[:4]) for a in series]
    if len(handles) <= VPA_KEY_INSIDE_MAX or fig is None:
        legend(ax, handles, loc="upper right")
        return []

    ref = [h for a, h in zip(series, handles) if a[0] == REF_LABEL]
    methods = [h for a, h in zip(series, handles) if a[0] != REF_LABEL]
    ncol = VPA_KEY_NCOL

    # ONE box, built as TWO legends. A legend column is as wide as its widest entry, so
    # the reference -- one line, and about three times the width of a method name -- cannot
    # be a cell of the method grid without stretching whichever column it landed in, and
    # the 2 x 4 would stop being a grid. matplotlib cannot span a cell either. So the two
    # lines get their own layouts, stacked and each centred, and then the frames come off
    # and a single rectangle is drawn round the pair: the reader sees one key, while
    # matplotlib still lays out the grid and the reference independently, which is what
    # keeps the grid even and the reference genuinely centred over it.
    legs = [legend(fig, _vpa_row_major(methods, ncol), loc="lower center", ncol=ncol,
                   fontsize=11.5, bbox_to_anchor=(0.5, 0.008), columnspacing=1.5)]
    if not ref:
        return legs
    fig.canvas.draw()
    h = legs[0].get_window_extent(fig.canvas.get_renderer()).height / \
        fig.get_size_inches()[1] / fig.dpi
    legs.append(legend(fig, ref, loc="lower center", ncol=1, fontsize=11.5,
                       bbox_to_anchor=(0.5, 0.008 + h + VPA_KEY_ROW_GAP)))
    return legs + [_vpa_one_frame(fig, legs)]


def _vpa_one_frame(fig, legs):
    """Take the frames off stacked keys and draw ONE around their union, so they read as a
    single box. Returns the patch, which is then the outermost thing the axes must clear.

    A legend's window extent already includes its own borderpad, so the union sits exactly
    where the individual frames did -- no padding is added here, or the pair would gain a
    ring the single-legend figures do not have."""
    fig.canvas.draw()
    boxes = [l.get_window_extent(fig.canvas.get_renderer()) for l in legs]
    for l in legs:
        l.set_frame_on(False)
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(min(b.x0 for b in boxes), min(b.y0 for b in boxes)),
         (max(b.x1 for b in boxes), max(b.y1 for b in boxes))])
    patch = matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, figure=fig,
        facecolor="white", edgecolor=LEGEND_EDGE, linewidth=AXIS_LW, zorder=6)
    fig.add_artist(patch)
    return patch


def _vpa_clear_keys(fig, legs, axes):
    """Move the axes up until the LOWEST THING THEY DRAW clears the figure-level keys.

    tight_layout lays the axes out over the whole figure and cannot see a legend that
    belongs to the FIGURE, so it puts the x label underneath one. Raising `bottom` is not
    enough on its own either: the x label hangs BELOW the axes box and travels up with it,
    so the thing that has to clear the key is the axes' tight bounding box, not its frame.
    Measure both and close the gap -- the same after-the-fact correction fit() makes for an
    overrunning y label, and it keeps holding when the key gains a row or the type moves."""
    if not legs:
        return
    for _ in range(3):                      # a shift changes the layout; re-measure
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        h = fig.get_size_inches()[1] * fig.dpi
        key_top = max(l.get_window_extent(r).y1 for l in legs) / h
        drawn_bottom = min(a.get_tightbbox(r).y0 for a in axes) / h
        gap = key_top + 0.02 - drawn_bottom
        if gap <= 0.001:
            break
        sp = fig.subplotpars
        fig.subplots_adjust(bottom=min(sp.bottom + gap, sp.top - 0.1))


def _vpa_resolve(arms, data, ref):
    """Everything that depends on WHICH arms are drawn, for one arm set.

    x is the only thing the two sets really argue about, and it cascades: the per-count
    curves, the window counts and the shares are all evaluated over it, and the reference's
    rolled share is truncated by its two ends. Per-arm values themselves do not depend on
    the set -- an arm's mean at 30 atoms is its mean at 30 atoms -- which is what lets one
    CSV stand behind both."""
    firm = [a for a in sorted(set.intersection(*(set(data[lab]) for lab, *_ in arms)))
            if all(len(data[lab].get(a, ())) >= MIN_N for lab, *_ in arms)]
    xs = list(range(firm[0], firm[-1] + 1))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, xs, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], xs, stat)
    exact, totals = {}, {}
    for label, sizes in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]:
        exact[label], totals[label] = share(sizes, xs)
    return {"arms": arms, "xs": xs, "curves": curves, "exact": exact, "totals": totals,
            "counts": _vpa_window_counts(ref, xs),
            "thin": {lab: [a for a in xs if len(data[lab].get(a, ())) < MIN_N]
                     for lab, *_ in arms}}


def _vpa_build(out, name, cfg, stat, aset, S, ref_label):
    """One variant at one statistic on one arm set: the figure, and the line about it for
    the run log."""
    stem = f"vina_dock_per_atom_{name}_{stat}_{aset}"
    xs, curves, counts = S["xs"], S["curves"], S["counts"]

    # Reference first and ours last, so the series being read sits on top -- the same
    # ordering the 3-line figures use. The key is the union of the two panels: a series
    # drawn on either one belongs in it, which is how v3 keeps its reference entry while
    # taking the grey off the strip below.
    series = [(lab, VPA_COLORS[lab], lw, style, z)
              for z, (lab, _, lw, style) in enumerate(S["arms"], start=3)]
    if cfg["ref_vina"] or cfg["ref_dist"]:
        series.insert(0, (REF_LABEL, REF_COLOR, REF_LW, DASH, 2))
    dist_series = [s for s in series if s[0] != REF_LABEL or cfg["ref_dist"]]

    shares = dict(S["exact"])
    if cfg["ref_dist_smooth"]:
        shares[REF_LABEL] = rolled(S["exact"][REF_LABEL], xs)
    ref_values = _vpa_gated(curves[(REF_LABEL, stat)], counts, cfg["ref_gate"])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw=dict(height_ratios=list(HEIGHT_RATIOS)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bx.set_facecolor("white")

    for label, colour, lw, style, z in series:
        if label != REF_LABEL:
            ax.plot(xs, curves[(label, stat)], color=colour, lw=lw, ls=style, zorder=z,
                    solid_capstyle="round", dash_capstyle="round")
        elif not cfg["ref_vina"]:
            continue
        elif cfg["ref_gate"] == "hard":
            ax.plot(xs, ref_values, color=colour, lw=lw, ls=style, zorder=z,
                    dash_capstyle="round")
        else:
            firm, thin = _vpa_split_firm(ref_values, counts)
            for ys, alpha in ((thin, VPA_REF_THIN_ALPHA), (firm, 1.0)):
                ax.plot(xs, ys, color=colour, lw=lw, ls=style, zorder=z, alpha=alpha,
                        dash_capstyle="round")

    # The same identities below: colour for the series, dash for the ones it is read
    # against. The arms are per-count steps; the reference is a curve when it is smoothed,
    # because drawing a rolling average stepped would claim a precision it lost. Only the
    # two VPA_FOCUS arms get an area -- four filled bands would be a stack, not a
    # comparison -- and in `core` that is both of them.
    for label, colour, lw, style, z in dist_series:
        stepped = label != REF_LABEL or not cfg["ref_dist_smooth"]
        filled = label in VPA_FOCUS or (label == REF_LABEL and cfg["ref_dist_fill"])
        if filled:
            bx.fill_between(xs, shares[label], step="mid" if stepped else None,
                            color=colour, alpha=DIST_FILL, linewidth=0, zorder=z)
        draw = bx.step if stepped else bx.plot
        draw(xs, shares[label], color=colour, ls=style, zorder=z + 5,
             lw=VPA_DIST_LW if label in VPA_FOCUS else lw,
             solid_capstyle="round", dash_capstyle="round",
             **(dict(where="mid") if stepped else {}))

    _vpa_furniture(ax, ylabel=VPA_Y_LABEL.format(stat=stat), xlabel=None, xs=xs)
    _vpa_furniture(bx, ylabel=VPA_D_LABEL, xlabel=X_LABEL, xs=xs)
    ax.margins(y=0.075)              # the default 5% puts ours on the bottom spine
    # Every 2 kcal/mol. The locator, left to itself over a ~9 kcal/mol span, picks 3 and
    # labels -3/-6/-9, which reads as a coarser axis than the 3-line figures' step of 2.
    ax.yaxis.set_major_locator(MultipleLocator(2))
    # Headroom over what is ON THIS PANEL, which is why it reads dist_series: v3's grey is
    # on the panel above only, and letting its rolled share set this scale would leave the
    # strip short by the height of a series that is not in it.
    bx.set_ylim(0, max(v for lab, *_ in dist_series for v in shares[lab]) * 1.13)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if name == "v3":
        # v3's strip is ticked 0/4/8 % on request (2026-09-13) rather than the locator's
        # 0/3/6; the other variants keep the locator.
        bx.yaxis.set_major_locator(MultipleLocator(4))
    # "Generated fraction" is the name; the per-cent sign on the ticks is where the unit
    # is stated, so the label does not have to carry a parenthesis to stay honest.
    bx.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    # Each y label is otherwise placed against ITS OWN tick labels, so "Vina Dock median"
    # (widest tick "-10") and "Generated fraction" (widest tick "4%") sit at two different
    # x and the pair reads as a ragged left edge. align_ylabels puts both at the outer of
    # the two, which is the only way they line up without hard-coding a coordinate.
    fig.align_ylabels((ax, bx))
    legs = _vpa_legend(ax, series, ref_label, fig=fig)
    fit(fig, pad=0.5, h_pad=H_PAD)
    _vpa_clear_keys(fig, legs, (ax, bx))
    lo, hi = ax.get_ylim()
    save(fig, out, stem)

    drawn = [a for a, v in zip(xs, ref_values) if v is not None]
    where = (f"reference {drawn[0]}..{drawn[-1]} atoms" if cfg["ref_vina"]
             else "no reference")
    if not cfg["ref_dist"]:
        where += ", off the strip"
    return f"    {stem:40s} x {xs[0]}..{xs[-1]}   {where}   y {lo:.1f} .. {hi:.1f}"


def _vpa_write_csv(out, sets, arms, data, ref, source, n_ref):
    """One table behind all twelve figures.

    It is one table because almost nothing in it is per figure. An arm's count, mean,
    median and share at a heavy-atom count are properties of that arm, not of the picture
    it is drawn in, so the arm columns are evaluated once over the UNION of the two x spans
    -- which means a row may carry a value at a count only `core` plots, and `plotted_in`
    is the column that says so.

    Four columns carry what does differ. `plotted_in` names the arm sets whose x span
    contains the row. `reference_source` and `n_reference_total` name the POPULATION behind
    every reference column -- p79 or crossdocked, and how many ligands -- which is constant
    down the table and repeated anyway, because a flat CSV read apart from its figure has
    nowhere else to put it and a reference column whose population is unstated is the one
    thing this figure will not ship. `reference_firm` is yes wherever the rolling window
    holds at least VPA_REF_FIRM ligands, which is the whole of the variant difference on
    the Vina panel: v1 draws the reference exactly there, v2 and v3 draw beyond it and
    fade. And the reference's ROLLED share is the one number that genuinely depends on the
    span, because a moving average is truncated at its ends -- hence one column per arm
    set, empty where that set does not reach."""
    union = sorted(set().union(*(set(S["xs"]) for S in sets.values())))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, union, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], union, stat)
    counts = _vpa_window_counts(ref, union)
    exact = {label: share(sizes, union)[0] for label, sizes
             in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]}
    rolled_ref = {name: dict(zip(S["xs"], rolled(S["exact"][REF_LABEL], S["xs"])))
                  for name, S in sets.items()}
    spans = {name: set(S["xs"]) for name, S in sets.items()}

    header = ["heavy_atoms", "plotted_in"]
    for label, *_ in arms:
        k = _vpa_slug(label)
        header += [f"n_{k}", f"{k}_mean", f"{k}_median", f"{k}_pct"]
    header += ["reference_source", "n_reference_total", "n_reference_exact",
               "n_reference_window", "reference_firm", "reference_mean",
               "reference_median", "reference_pct_exact"]
    header += [f"reference_pct_rolled_{name}" for name in sets]
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    rows = []
    for i, a in enumerate(union):
        row = [a, "+".join(n for n, span in spans.items() if a in span)]
        for label, *_ in arms:
            row += [len(data[label].get(a, ())),
                    fmt(curves[(label, "mean")][i]),
                    fmt(curves[(label, "median")][i]),
                    f"{exact[label][i]:.3f}"]
        row += [source, n_ref, len(ref.get(a, ())), counts[i],
                "yes" if counts[i] >= VPA_REF_FIRM else "no",
                fmt(curves[(REF_LABEL, "mean")][i]),
                fmt(curves[(REF_LABEL, "median")][i]),
                f"{exact[REF_LABEL][i]:.3f}"]
        row += [fmt(rolled_ref[name].get(a)) for name in sets]
        rows.append(row)
    return write_csv(out, "vina_dock_per_atom", header, rows)


@figure("fig-vina-per-atom", needs=(VPA_BASENAME, "crossdocked_reference_dock.json"))
def draw_vina_per_atom(out):
    """Vina Dock against heavy-atom count, over a size-distribution strip — three
    reference variants (gated / faded / strip-free) x mean and median x core and all."""
    data, pockets, arms, missing = {}, {}, [], []
    for arm in VPA_ARMS:
        label, root = arm[0], arm[1]
        sizes, n_pockets = _vpa_load(root, label)
        if sizes is None:
            missing.append((label, os.path.join(root, VPA_BASENAME)))
            continue
        data[label], pockets[label] = sizes, n_pockets
        arms.append(arm)
    absent = [lab for lab in VPA_FOCUS if lab not in data]
    if absent:
        raise FileNotFoundError(
            f"the compared arms are the figure: {', '.join(absent)} has no {VPA_BASENAME}")
    source, ref, ref_label, ref_note, n_ref = _vpa_reference_set()

    use_style()
    print(f"  {VPA_PROTOCOL}")
    print("  arms: " + " · ".join(f"{lab} {pockets[lab]} pockets" for lab, *_ in arms))
    for label, path in missing:
        print(f"    {label} skipped: not docked here, no "
              f"{os.path.relpath(path, REPO)}")
    print(f"  reference: {ref_note}")
    print(f"    drawn as {ref_label!r}")
    for label, by in [(REF_LABEL, ref)] + [(lab, data[lab]) for lab, *_ in arms]:
        vals = [n for n, v in by.items() for _ in v]
        print(f"    size of {label:16s} mean {st.mean(vals):5.2f}  "
              f"median {st.median(vals):5.1f} heavy atoms")

    # The two ends of x are where EVERY arm IN THE SET clears MIN_N, so the sets do not
    # share a span. Inside them x stays contiguous and an arm that dips below is broken by
    # _vpa_model_curve() instead, so one thin patch in one arm no longer truncates the
    # figure for all of them.
    sets = {}
    for aset, subset in _vpa_arm_sets(arms):
        S = sets[aset] = _vpa_resolve(subset, data, ref)
        xs, curves = S["xs"], S["curves"]
        print(f"  {aset}: " + " + ".join(lab for lab, *_ in subset)
              + f" · x = {xs[0]}..{xs[-1]} heavy atoms, all drawn arms >= {MIN_N} "
              f"molecules at both ends; broken inside where an arm is not: "
              + (", ".join(f"{lab} at {v}" for lab, v in S["thin"].items() if v)
                 or "nowhere"))
        for label in [REF_LABEL] + [lab for lab, *_ in subset]:
            print(f"    {label:16s} n={S['totals'][label]:6d}  "
                  f"{sum(S['exact'][label]):.1f}% of its molecules inside the plotted "
                  f"x range")
        for stat in VPA_STATS:
            both = [i for i in range(len(xs))
                    if curves[(VPA_VOX_LABEL, stat)][i] is not None
                    and curves[(VPA_OUR_LABEL, stat)][i] is not None]
            wins = [i for i in both
                    if curves[(VPA_OUR_LABEL, stat)][i] <= curves[(VPA_VOX_LABEL, stat)][i]]
            gap = st.mean([curves[(VPA_OUR_LABEL, stat)][i]
                           - curves[(VPA_VOX_LABEL, stat)][i] for i in both])
            print(f"    {stat:6s}: CoDE lower than VoxBind at {len(wins)}/{len(both)} "
                  f"sizes, mean per-size gap {gap:+.3f} kcal/mol")

    for name, cfg in VPA_VARIANTS.items():
        panels = ("Vina panel and strip" if cfg["ref_dist"] else "Vina panel only")
        tail = ("gated off below it" if cfg["ref_gate"] == "hard" else
                f"faded to alpha {VPA_REF_THIN_ALPHA} below it, down to a window of "
                f"{VPA_REF_GATE[cfg['ref_gate']]}")
        print(f"  {name}: reference on the {panels}, firm at >= {VPA_REF_FIRM} ligands in "
              f"its ±{REF_WIN}-atom window and {tail}; distribution "
              + ("rolled" if cfg["ref_dist_smooth"] else "per-count")
              + (", filled" if cfg["ref_dist_fill"] else ", unfilled"))
        for stat in VPA_STATS:
            for aset in sets:
                print(_vpa_build(out, name, cfg, stat, aset, sets[aset], ref_label))
    _vpa_write_csv(out, sets, arms, data, ref, source, n_ref)
    print("  vina_dock_per_atom.csv -- one table behind all twelve")


# ════════════════════════════════════════════════════════════════════════════════
# the runner
# ════════════════════════════════════════════════════════════════════════════════
def _by_eval():
    """{eval: [fig_id, ...]} in registration order — `fig-<eval>-<name>` splits on the
    SECOND hyphen, so a figure name may itself contain hyphens."""
    out = collections.OrderedDict()
    for fid in FIGURES:
        ev = fid.split("-")[1]
        out.setdefault(ev, []).append(fid)
    return out


def _list():
    print(f"{len(FIGURES)} figures in {len(_by_eval())} evaluations "
          f"-> {OUT_ROOT.relative_to(REPO)}/fig-<eval>/\n")
    for ev, ids in _by_eval().items():
        folders = sorted({figure_folder(f) for f in ids})
        print(f"  {ev}  ->  {', '.join(f + '/' for f in folders)}")
        for fid in ids:
            doc = (FIGURES[fid].__doc__ or "").strip().splitlines()
            here = figure_folder(fid)
            tag = f"[{here.split('/')[-1]}] " if len(folders) > 1 else ""
            print(f"    {fid:44s} {tag}{doc[0] if doc else ''}")
        print()


def _resolve(names):
    """Figure ids for the names asked for. A name that is an EVALUATION expands to every
    figure in it, so `draw.py posecheck` is the whole PoseCheck set. An unknown name is a
    hard error listing the near misses -- silently drawing nothing is the failure mode
    this exists to prevent."""
    by_eval = _by_eval()
    out = []
    for n in names:
        if n in FIGURES:
            out.append(n)
        elif n in by_eval:
            out.extend(by_eval[n])
        else:
            near = [f for f in FIGURES if n in f] or [f"{e} (whole evaluation)"
                                                      for e in by_eval if n in e]
            raise SystemExit(f"unknown figure {n!r}"
                             + (f"\n  did you mean: {', '.join(near)}" if near else
                                f"\n  --list shows all {len(FIGURES)}"))
    seen, uniq = set(), []
    for f in out:                       # keep order, drop repeats
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="draw.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A name may be a figure id or a whole evaluation "
               "(vina, posebusters, posecheck, consistency, interaction, molweight, "
               "similarity, mcp, ensemble).")
    ap.add_argument("names", nargs="*", help="figure ids and/or evaluation names")
    ap.add_argument("-a", "--all", action="store_true", help="draw every figure")
    ap.add_argument("-l", "--list", action="store_true", help="list the figures and exit")
    ap.add_argument("-f", "--formats", default="png,svg,pdf",
                    help="comma-separated: png,svg,pdf (default all three)")
    ap.add_argument("-o", "--out", default=str(OUT_ROOT),
                    help="output root; each figure lands in <root>/fig-<eval>/ "
                         f"(default {OUT_ROOT})")
    ap.add_argument("-k", "--keep-going", action="store_true",
                    help="carry on after a figure fails, and report at the end")
    args = ap.parse_args(argv)

    if args.list or (not args.names and not args.all):
        _list()
        return 0

    global _FORMATS
    _FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    bad = [f for f in _FORMATS if f not in ("png", "svg", "pdf")]
    if bad:
        raise SystemExit(f"unsupported format(s): {', '.join(bad)}")

    ids = list(FIGURES) if args.all else _resolve(args.names)
    root = Path(args.out)
    use_style()
    print(f"drawing {len(ids)} figure(s) as {'+'.join(_FORMATS)} -> {root}/\n")

    failed, t_all = [], time.time()
    for i, fid in enumerate(ids, 1):
        # The folder is the EVALUATION, not the figure: everything answering the same
        # question sits together, and the shared CSVs two figures both export (the pose
        # per-atom table, the interaction table) land once instead of once per figure.
        out = root / figure_folder(fid)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(ids)}] {fid}  -> {out.name}/")
        t0 = time.time()
        try:
            FIGURES[fid](out)
        except Exception as exc:                                  # noqa: BLE001
            if not args.keep_going:
                raise
            failed.append((fid, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n{len(ids) - len(failed)}/{len(ids)} drawn in {time.time() - t_all:.1f}s")
    if failed:
        print(f"{len(failed)} failed:")
        for fid, why in failed:
            print(f"  {fid}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
