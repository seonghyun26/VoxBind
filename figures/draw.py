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
    "jsd":         "fig-jsd",
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
    # No `_all` stems: the clash per-atom figure is core-only (2026-09-14), and the whole
    # field is clash_per_atom_all_methods_*.
    "clash_per_atom_mean_core": "clash-per-atom-mean-core",
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
    "interaction_contacts_pair_all": "interaction-contacts-pair-all",
    "interaction_hbonds_all": "interaction-hbonds-all",
    "interaction_hbonds_core": "interaction-hbonds-core",
    "interaction_hbonds_pair_all": "interaction-hbonds-pair-all",
    "interaction_pair_all": "interaction-pair-all",
    "interaction_legend": "interaction-legend",
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
    "FuncBind":         "#9A8430",   # olive — the saturated step of SOFT's #C6B46A
    "TargetDiff":       "#B58FDB",   # violet
    "VoxBind":          "#F5B27E",   # sand
    "CoDE":             "#4363D8",   # blue — ours (LaTeX: \textsc{CoDE})
    "Ours v2":          "#2B3A8C",   # deep indigo — same family as CoDE
}

# The MCP fine-tune arms are not four independent methods -- they are one model at four
# amounts of receptor-ED fine-tuning -- so they take an ordinal ramp darkening with training
# rather than four categorical hues.
#
# THE RAMP NO LONGER STARTS AT FuncBind'S OWN COLOUR (2026-09-14). It used to: both were the
# brown #A9744F, and the ramp read as "this IS FuncBind, trained further". FuncBind the
# METHOD has since moved to olive so that COLORS and SOFT agree on its hue family -- the two
# tables disagreed, and the eight-method figures draw through soft() while the interaction
# family draws through color(), so the same method came out olive in one and brown in the
# other. The ramp is left brown because it belongs to the macrocycle section, which is not
# part of that set and was not re-coloured with it. If those figures should track FuncBind
# again, re-base this ramp on #9A8430.
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
# The eight-method figures draw EVERY series through soft(), so the five published baselines
# now have tints of their own (2026-09-14): at full saturation they sat badly beside CoDE's and
# VoxBind's tints and TargetDiff's violet, which are already pale. Each keeps its hue family --
# teal, pink, red, green, brown -- so a method is the same colour across the paper, lifted to
# the tints' lightness. COLORS still holds the saturated originals, which the three-arm, Vina
# and PoseBusters figures draw; only soft() sees these.
SOFT = {
    "CoDE":       "#8291E8",
    "AR":         "#7FC4D1",   # was #17A2B8 teal-cyan
    "Pocket2Mol": "#F0A6C0",   # was #E87BA4 pink
    "DiffSBDD":   "#EE9190",   # was #E34948 red
    "DecompDiff": "#8DCB92",   # was #3CB44B green
    # FuncBind LEAVES ITS HUE FAMILY here (2026-09-14), the one tint that does: as a tan it was
    # the same warm family as VoxBind's sand and the two could only be told apart by weight.
    # Olive-gold is the one family none of the other eight occupy -- teal, pink, coral, green,
    # violet, sand, periwinkle, grey. COLORS keeps the brown, which is what the MCP fine-tune
    # ramp darkens off, so FuncBind is brown in the macrocycle figures and olive in these.
    # COLORS now carries the saturated olive #9A8430, so this is an ordinary lighter step of
    # its own hue like every other entry here -- not a hue change of its own. It used to be
    # the one tint that left its family, which put FuncBind in two colours depending on
    # which table a figure read.
    "FuncBind":   "#C6B46A",   # was #A9744F brown, then #C4A083 tan
}
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

# ── the dash a method is drawn with, everywhere it is drawn ──────────────────────
# THE SECOND IDENTITY CHANNEL, and like the palette it is decided HERE and nowhere else.
# Nine series on one axis cannot be told apart by hue alone, so every eight-method figure
# carries dash as well -- and it only works if a method's dash is the same in all of them.
# It was not. Two tables had drifted out of step: the ECDF family's (in PCSZ_BASELINES /
# PCSZ_LOCAL) and the rotatable-bond family's (RB_BASE_DASH), and the rotatable-bond one was
# shifted by a whole method -- its Pocket2Mol carried the ECDF's DiffSBDD pattern, its
# DiffSBDD the ECDF's DecompDiff, and so on down the list. Worse, the standalone key
# (fig-posecheck-strain-legend) read the rotatable-bond table while the ECDF drew the other,
# so the legend was decoding the panel wrongly.
#
# AR AND POCKET2MOL USED TO SHARE (5, 2) in the ECDF, which left them separable by hue only
# -- exactly what the dash channel exists to prevent. Pocket2Mol takes the pattern that was
# spare, so all five baselines are now distinct.
#
# SOLID MEANS OURS. TargetDiff is dashed even though we run it locally: the rotatable-bond
# family gave it a solid line because its helper keyed on provenance (RB_LOCAL_LABELS, "we
# ran this") rather than on authorship, and solid reads as ours in every other figure.
METHOD_DASH = {
    "Reference ligand": SOLID,          # solid since 2026-09-14, as the ECDF draws it
    "AR":               (0, (5, 2)),
    "Pocket2Mol":       (0, (3, 1.4, 1, 1.4)),
    "DiffSBDD":         (0, (1, 1.6)),
    "DecompDiff":       (0, (6, 2, 1, 2)),
    "FuncBind":         (0, (9, 3)),
    "TargetDiff":       (0, (6, 2)),
    "VoxBind":          SOLID,
    "CoDE":             SOLID,
}


def dash(label):
    """The dash pattern for `label`, through the alias table.

    Raises rather than defaulting to solid: a method missing here is a rename to fix, and a
    silent solid would hand it the channel that means ours."""
    key = ALIASES.get(label, label)
    if key not in METHOD_DASH:
        raise KeyError(f"no dash for {label!r} (known: {', '.join(sorted(METHOD_DASH))})")
    return METHOD_DASH[key]
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


def arm_handles(arms, include_ref=True, paint=None):
    """`paint` maps a method to its colour: color() by default, soft() for the families
    drawn in the tints."""
    paint = paint or color
    h = [Line2D([], [], color=color(REF_LABEL), lw=REF_LW, ls=DASH, label=REF_LABEL)] \
        if include_ref else []
    return h + [Line2D([], [], color=paint(lab), lw=MODEL_LW, ls="-", label=display(lab))
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
# the heat ramp and its colourbar — shared by every figure keyed to a continuous scale
# ════════════════════════════════════════════════════════════════════════════════
# HERE, NOT IN A PART (2026-09-15, on request). The ramp was born in posecheck_rotbond and
# fig-posebusters-valid-heatmap reached across for it by name at call time, which worked
# only because parts are pasted into one module: it read as one family borrowing another's
# private constant, and the borrow was easy to miss when either end changed. A colour scale
# used by two families is house furniture, so it sits beside INK, legend() and furniture()
# under the same rule the palette follows -- decided in ONE place, read everywhere else.
# THE HOUSE HEAT RAMP: one hue, light to dark (2026-09-15, on request; it was matplotlib's
# diverging coolwarm, then a purple). "Darker is more of the quantity the bar is named
# after" -- more strain in the rotbond grid, a higher pass rate in the PoseBusters map.
# Diverging was the wrong family for either: its white middle marked 400 kcal/mol, a number
# nothing in the data means.
#
# TWO STOPS, AND THE ENDS ARE CHOSEN SO THE BLEND PASSES THROUGH THE TWO GIVEN COLOURS
# #B4BEF0 AND #8291E8 (2026-09-15, on request, after a five-stop version). matplotlib blends
# in sRGB along a straight line, so both ends lie on the line through those two, extended
# 1.4x their spacing each way (d = #8291E8 - #B4BEF0; head = #B4BEF0 - 1.4d, tail =
# #8291E8 + 1.4d). The given colours then fall at 37% and 63% of the bar, symmetric about its
# middle. 1.44 is as far as the head can go before its green channel leaves the cube; the
# tail could go to 2.6 (#001CD3), but symmetry is the point, so it stops at #3C52DD.
# WHAT THE FIVE-STOP VERSION WAS FOR, and why two is better: hand-placed stops (#F5F8FE,
# #B4BEF0, #8291E8, #4D65E0, then a tail that went #192D90 -> #1E36AC when it read too dark)
# let each end be tuned by eye, but the spacing between them is then a design choice made
# five times, and the ramp's rate of change jumps at every stop. Two ends on one line through
# the given pair gives constant rate and a single decision, and it costs only the deepest
# end: luminance now bottoms out at 0.12 rather than 0.06, so the very top of a bar separates
# a little less. Hex literals outside the palette tables, which the palette rule otherwise
# forbids: that rule is about the colour of a METHOD, and a continuous ramp keyed by its own
# colourbar carries no method identity. Neither end is #4363D8 (CoDE) or #2B3A8C (Ours v2).
HEAT_STOPS = ("#FAFDFB", "#3C52DD")
HEAT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("voxbind_heat", HEAT_STOPS)
# The bar's SHAPE, which is the same in both figures and is what a reader recognises before
# reading either: vertical on the right, a little short of the block it keys (0.92) and thin
# (aspect 38), so it reads as a key rather than as a tenth panel.
HEAT_CBAR_SHRINK, HEAT_CBAR_ASPECT = 0.92, 38
# Gap from the panel block to the bar, doubled from 0.012 (2026-09-13) -- a share of the
# figure width, which is what colorbar(pad=) takes.
HEAT_CBAR_PAD = 0.024
# The caption's gap off the bar's numbers, DOUBLED (2026-09-14): measured off the PNG it was
# 19 px at 220 dpi, i.e. 6.2 pt on matplotlib's default labelpad of 4, so 10.5 buys about 12.4.
HEAT_CBAR_LABELPAD = 10.5


def heat_colorbar(fig, ax, norm, label, *, label_fs, tick_fs, extend="neither", ticks=None,
                  pad=HEAT_CBAR_PAD, labelpad=HEAT_CBAR_LABELPAD, cmap=None):
    """The house heat colourbar: HEAT_CMAP under `norm`, drawn the same way every time.

    `ax` is whatever the bar should steal its space from -- one axes or an array of them.
    What is NOT shared is point size: the rotbond grid is nine panels on a 14.45 in canvas
    and carries 27/18 pt, the PoseBusters map is one 8.4 x 6.2 in block at 13/11.5, and a bar
    that copied one into the other would tower over or vanish beside the figure it keys. So
    the sizes are arguments and everything else -- ramp, shape, pad, the quarter-turned
    numbers, the missing outline -- is fixed here, which is the point of having the helper.

    The numbers are turned a quarter turn anticlockwise (2026-09-14) so they read along the
    bar like its name; anchored left and centred on the tick, which is where they sat upright.
    No frame round the bar (2026-09-14), as the boxes it keys lost theirs -- the extend arrow
    is part of that same outline path, so it loses its edge with it and reads as pure fill."""
    kw = {} if ticks is None else {"ticks": ticks}
    # `cmap` is for trying a ramp on one figure without moving the house one under the other.
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap or HEAT_CMAP), ax=ax,
                      extend=extend, shrink=HEAT_CBAR_SHRINK, aspect=HEAT_CBAR_ASPECT,
                      pad=pad, **kw)
    cb.set_label(label, fontsize=label_fs, color=INK, labelpad=labelpad)
    cb.ax.tick_params(labelsize=tick_fs, colors=AXIS, width=AXIS_LW)
    plt.setp(cb.ax.get_yticklabels(), rotation=90, va="center", ha="left")
    cb.outline.set_visible(False)
    return cb


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


def size_distribution(ax, xs, per_arm, ref_per, arms, paint=None):
    """Where each set puts its molecules, as a share of its own -- which is what makes the
    panel above it trustworthy, and is itself a finding, since the arms differ in the sizes
    they generate as much as in per-size pose quality. It counts the SAME molecules the
    panel above plots. `paint` as in arm_handles."""
    for lab, key, _ in arms:
        pct, _ = share(per_arm[key], xs)
        col = (paint or color)(lab)
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


# ════════════════════════════════════════════════════════════════════════════════
# fig-jsd-{bond,bond-distance,pair,atom-type,summary,ring-size,n-rings,aromatic,rings}
# ════════════════════════════════════════════════════════════════════════════════
# The distribution metrics of TargetDiff and VoxBind: how closely each arm's bond lengths,
# atom-pair distances, element mix and rings follow CrossDocked ligands.
#
#     bond         bond-distance JSD per bond type          VoxBind Table 2
#     bond-distance  the distance histograms those JSDs compare (was bond-length, 2026-09-15)
#     pair         all-atom (<12 Å) and C-C (<2 Å) pair distances   TargetDiff Fig. 2
#     atom-type    heavy-atom element shares
#     summary      the four headline JSDs side by side
#     ring-size    share of rings by size, 3-9               TargetDiff Table 2
#     n-rings      rings per molecule                        VoxBind Fig. 10, middle
#     aromatic     aromatic share of atoms and of rings      VoxBind Fig. 10, bottom
#     rings        all three, one column per method          VoxBind Fig. 10
#
# THESE ONLY DRAW. voxbind/scripts/89_eval_crossdocked_jsd.sh computes everything into one
# JSON; its tool's docstring carries the protocol and the self-check that pins it to the
# published TargetDiff row. What a reader of the figures needs from it:
#
# THE REFERENCE DRAWN IS EVERY CROSSDOCKED LIGAND of the train+test split, each distinct
# molecule weighing one (100,100 files are 8,829 molecules; one is cross-docked into 869
# pockets). The published papers score against the 100 test ligands instead, which are
# too few: a model sampling the training distribution exactly would score C=N JSD 0.52
# against them (16 bonds). Those numbers are still computed -- `jsd_test` in the JSON, the
# CSVs here and the appendix tables of jsd-tables.ipynb -- and so is a pose-weighted
# reference, as the check that the de-duplication does not move anything.
#
# THE ARMS stand on the 79 electron-density pockets (TargetDiff and VoxBind hold 100). JSD
# is scipy's JS DISTANCE, as both papers print it.
#
# EVERY METHOD IS DRAWN IN ITS soft() TINT (2026-09-14, on request), the colours of the
# eight-method strain figures (ecdf-by-size-all), not COLORS' saturated originals.
JSD_JSON = BUNDLE / "_shared" / "260913_crossdocked_jsd" / "crossdocked_jsd.json"
JSD_NEEDS = ("crossdocked_jsd.json (voxbind/scripts/89_eval_crossdocked_jsd.sh)",)
# Bond lengths are SCORED in 0.005 Å bins; they are DRAWN four bins at a time, so a
# histogram reads as a shape rather than a comb. The JSD columns still come from the
# scoring bins.
JSD_LEN_MERGE = 4
JSD_CC_XLIM = (1.0, 2.0)        # C-C pairs under 2 Å are bonds; nothing sits below 1.0
JSD_NRINGS_MAX = 8              # rings per molecule: the last bar pools 8 and more
JSD_AROM_MERGE = 2              # 20 scored aromatic-fraction bins drawn as 10
# Share panels (atom type, ring size) split into tiers: >= 50% gets a 0-100 axis, >= 5% a
# middle axis, the rest a small one -- each tier on a linear axis scaled to itself. What is
# measured against those thresholds is the TIER CRITERION, which is the tallest bar of the
# category by default and the reference ligand's own share where a panel passes `tier_by`.
JSD_TIER_PCT = (50.0, 5.0)
JSD_KEY_NCOL = 4                # method columns of the all-arms key: 8 arms = 2 rows x 4
# AXIS WEIGHT, AS RATIOS OF THE TICK-LABEL SIZE (2026-09-15, on request). Set on the summary
# first -- 11.5 pt ticks, 16.1 pt axis names (the strain ECDF's PCSZ_LABEL_FS = 11.5 x 1.4) and a
# 2.2 pt axis line -- then carried to every jsd figure as the same PROPORTIONS, so a grid with
# 9.5 pt ticks gets a lighter line and smaller names than a bar chart with 14 pt ones.
JSD_TITLE_PER_TICK = 1.4
JSD_AXIS_LW_PER_TICK = 2.2 / 11.5
# ...UP TO THE SUMMARY'S OWN 16.1 pt. The single-panel bar charts carry 14 pt ticks, and 1.4x
# that is a 19.6 pt name: the two-line "JSD to CrossDocked / ligands" and the ring-size figure's
# then "% of rings, sizes 3-9" (now "% of rings", 2026-09-15, on request) ran past both ends of a
# PANEL_H axis and were cut off at the figure edge.
JSD_TITLE_MAX = 11.5 * 1.4
# The line is capped the same way (2026-09-15, on request: similar proportions, not identical
# weights) -- at 14 pt ticks the ratio gave 2.7 pt, which on a single 7.6 in panel read heavier
# than the summary's 2.2 pt does across its four.
JSD_AXIS_LW_MAX = 2.2
# Ring-size panels, fixed (2026-09-14, on request): 6 | 3, 5, 7 | 4, 8, 9. The 5-ring shares
# its panel with the 3- and 7-ring, which the arms push to 12-30%; 4/8/9 stay under 4% for
# every set, so on a panel of their own they are not stubs under AR's 3-rings. Sizes run in
# ascending order inside a panel, whatever order a tuple lists them in.
JSD_RING_PANELS = (("6",), ("3", "5", "7"), ("4", "8", "9"))
# HATCHING, for colour-blind readers (2026-09-14, trial on request): every grouped-bar chart
# and its key give a method a pattern as well as its soft() tint, drawn in a darker step of
# that tint. The reference stays plain grey and CoDE plain -- ours, and the one solid method
# bar. Summary and rings are not hatched: each bar there already carries the method's name.
# OFF (2026-09-14): tried and set aside on request -- the figures are drawn plain. Set True to
# bring the patterns back.
JSD_HATCH_ON = False
JSD_HATCH = {"AR": "///", "Pocket2Mol": "\\\\\\", "DiffSBDD": "xxx", "DecompDiff": "...",
             "FuncBind": "---", "TargetDiff": "|||", "VoxBind": "ooo"}
JSD_HATCH_LW = 0.6               # points; matplotlib's 1.0 fills a 0.1-inch bar
JSD_HATCH_INK = 0.55             # hatch = the bar's tint times this (toward black)
JSD_SUMMARY = (
    ("Mean bond-distance JSD ↓", lambda j: j["bond_mean"]),
    ("All-atom pair JSD ↓", lambda j: j["pair"]["All_12A"]),
    ("C–C pair JSD ↓", lambda j: j["pair"]["CC_2A"]),
    ("Atom-type JSD ↓", lambda j: j["atom_type"]),
)
# (CSV source label, arm key) for every reference the eval scores against; the first is
# the one the figures draw.
JSD_REFERENCES = (("vs CrossDocked train+test (distinct molecules)", "jsd"),
                  ("vs CrossDocked train+test (pose-weighted)", "jsd_pose_weighted"),
                  ("vs CrossDocked test (100)", "jsd_test"))
_JSD_CACHE = {}


def _jsd_data():
    """(the eval JSON, the ARMS it covers in ARMS order). An arm the JSON lacks is named
    and skipped rather than drawn empty."""
    if "d" not in _JSD_CACHE:
        if not JSD_JSON.exists():
            raise FileNotFoundError(f"missing input {JSD_JSON}\n"
                                    f"  rebuild it with bash voxbind/scripts/89_eval_crossdocked_jsd.sh")
        d = json.load(open(JSD_JSON))
        if "reference_test" not in d:
            raise ValueError(f"{JSD_JSON} predates the train+test reference; "
                             f"rerun bash voxbind/scripts/89_eval_crossdocked_jsd.sh")
        arms = [a for a in ARMS if a[0] in d["arms"]]
        if not arms:
            raise ValueError(f"{JSD_JSON} holds none of the ARMS labels")
        missing = [a[0] for a in ARMS if a[0] not in d["arms"]]
        pockets = {d["arms"][lab]["n_pockets"] for lab, _, _ in arms}
        ref = d["reference"]
        print(f"  [{os.path.relpath(JSD_JSON, REPO)}]")
        print(f"  reference: CrossDocked train+test, {ref['n_files']:,} ligand files = "
              f"{ref['n_unique_ligands']:,} distinct molecules, each weighing 1 · "
              f"arms over {'/'.join(map(str, sorted(pockets)))} pockets")
        sc = d.get("selfcheck")
        if sc:
            print(f"  selfcheck vs published TargetDiff (test reference): max |Δ| bond JSD "
                  f"{sc['max_abs_diff_bond_jsd']:.4f}, ring % {sc['max_abs_diff_ring_pct']:.2f}"
                  f" -> {'PASS' if sc['pass'] else 'FAIL'}")
        if missing:
            print("  not in the JSON, not drawn: " + ", ".join(missing))
        _JSD_CACHE["d"] = (d, arms)
    return _JSD_CACHE["d"]


def _jsd_ref_key():
    """The reference's legend entry, with its size -- distinct molecules, not files."""
    ref = _jsd_data()[0]["reference"]
    return f"{ref['label']}, {ref['n_unique_ligands']:,} molecules"


def _jsd_ref_sets(d):
    """(CSV label, summary) of every reference whose histograms a CSV should carry."""
    return [(f"{REF_LABEL} (train+test)", d["reference"]),
            (f"{REF_LABEL} (train+test, pose-weighted)", d["reference_pose_weighted"]),
            (f"{REF_LABEL} (test)", d["reference_test"])]


def _jsd_variants(arms):
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


def _jsd_new(w, h):
    fig = plt.figure(figsize=(w, h), dpi=220)
    fig.patch.set_facecolor("white")
    return fig


def _jsd_axis_weight(fig):
    """Axis names and axis lines scaled to each panel's tick labels -- JSD_TITLE_PER_TICK and
    JSD_AXIS_LW_PER_TICK. Call once, after every panel is drawn and BEFORE fit(), since the
    larger names change the layout. The smaller of a panel's two tick sizes sets its line:
    the numeric axis, not a column of method names. Tick LENGTH is left alone -- the bar
    charts set it to zero on their categorical axis. The spines go above the data: every bar
    starts at the baseline and painted over half of the heavier line."""
    for ax in fig.axes:
        size = {a: a.get_major_ticks()[0].label1.get_fontsize()
                for a in (ax.xaxis, ax.yaxis) if a.get_major_ticks()}
        if not size:
            continue
        lw = min(JSD_AXIS_LW_PER_TICK * min(size.values()), JSD_AXIS_LW_MAX)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(lw)
            ax.spines[side].set_zorder(5)
        ax.tick_params(width=lw)
        for a, fs in size.items():
            if a.label.get_text():
                a.label.set_size(min(JSD_TITLE_PER_TICK * fs, JSD_TITLE_MAX))


def _jsd_key_above(fig, handles, ncol, fontsize=11.5):
    """A figure-level key in a strip above every panel. Laid out AFTER fit(), because
    tight_layout does not know about figure legends: the panels are fitted first, then
    pushed down by exactly the overlap the key makes with the tallest one.

    THE REFERENCE TAKES A CENTRED LINE OF ITS OWN over the method grid, one frame round both
    -- the Vina per-atom v3 key's layout. As one more cell of a single-row grid, the
    "Reference ligand (CrossDocked train+test), 8,829 molecules" entry made the key wider
    than the figure and it was cut off at both edges (atom-type, 2026-09-14). The grid reads
    across rows, and it loses a column at a time until it fits the figure's width."""
    _jsd_axis_weight(fig)
    fit(fig, pad=0.5)
    # No handles, no key (2026-09-15: atom-type, bond-distance and ring-size `all` dropped
    # theirs on request) -- the axis weights and the fit above still apply.
    if not handles:
        return
    ref_key = _jsd_ref_key()
    ref = [h for h in handles if h.get_label() == ref_key]
    methods = [h for h in handles if h.get_label() != ref_key]
    inv = fig.transFigure.inverted()
    ncol = max(1, min(ncol, len(methods)))
    while True:
        legs, top = [], 0.995
        for group, cols in ((ref, 1), (methods, ncol)):
            if not group:
                continue
            legs.append(legend(fig, _vpa_row_major(group, cols), loc="upper center", ncol=cols,
                               fontsize=fontsize, bbox_to_anchor=(0.5, top)))
            fig.canvas.draw()
            top = legs[-1].get_window_extent(fig.canvas.get_renderer()).transformed(inv).y0 \
                - VPA_KEY_ROW_GAP
        rend = fig.canvas.get_renderer()
        widest = max(l.get_window_extent(rend).transformed(inv).width for l in legs)
        if widest <= 0.98 or ncol == 1:
            break
        for l in legs:
            l.remove()
        ncol -= 1
    if len(legs) > 1:
        _vpa_one_frame(fig, legs)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = min(l.get_window_extent(rend).transformed(inv).y0 for l in legs)
    y1 = max(ax.get_tightbbox(rend).transformed(inv).y1 for ax in fig.axes)
    if y1 > y0 - 0.012:
        fig.subplots_adjust(top=fig.subplotpars.top - (y1 - y0 + 0.012))


def _jsd_hatch(label, col):
    """(pattern, ink) for a method's bars, or (None, col) where it is drawn plain."""
    pattern = JSD_HATCH.get(ALIASES.get(label, label)) if JSD_HATCH_ON else None
    if not pattern:
        return None, col
    return pattern, tuple(JSD_HATCH_INK * c for c in matplotlib.colors.to_rgb(col))


def _jsd_handles(arms, *, ref=True, patch=False):
    def one(label, col, ls="-", lw=MODEL_LW, key=None):
        if patch:
            hatch, ink = _jsd_hatch(key, col)
            h = Patch(facecolor=col, edgecolor=ink, hatch=hatch, lw=0, label=label)
            h.set_hatch_linewidth(JSD_HATCH_LW)
            return h
        return Line2D([], [], color=col, lw=lw, ls=ls, label=label)
    hs = [one(_jsd_ref_key(), REF_COLOR, DASH, REF_LW, key=REF_LABEL)] if ref else []
    return hs + [one(display(lab), soft(lab), key=lab) for lab, _, _ in arms]


def _jsd_key_cols(n, wide):
    """Method columns of the key. The reference always takes its own line above, so this only
    shapes the methods: the eight arms read 2 x 4 (2026-09-14, on request; they were 5 + 3),
    the same grid as the Vina per-atom v3 and strain keys. _jsd_key_above caps it at the
    method count, so the core pair stays one row."""
    return n if n <= 3 else (JSD_KEY_NCOL if wide else 3)


def _jsd_grouped(ax, groups, series):
    """Grouped bars: one group per x category, one bar per (label, values, colour) in
    `series` order, so the key's order is the bars' order. Bar width is derived from the
    series count -- a fixed width overlaps neighbouring groups at nine series."""
    x = np.arange(len(groups))
    w = 0.84 / len(series)
    for i, (lab, vals, col) in enumerate(series):
        vals = [np.nan if v is None else v for v in vals]
        xs = x + (i - (len(series) - 1) / 2) * w
        # A patch's hatch is drawn in its EDGE colour, so a hatched bar is two passes: the fill
        # with the pattern in its ink and no outline, then the white spacer outline on top.
        hatch, ink = _jsd_hatch(lab, col)
        if hatch:
            for bar in ax.bar(xs, vals, width=w, color=col, edgecolor=ink, hatch=hatch, lw=0,
                              zorder=3):
                bar.set_hatch_linewidth(JSD_HATCH_LW)
            ax.bar(xs, vals, width=w, fill=False, edgecolor="white",
                   lw=0.4 if len(series) > 4 else 0.8, zorder=3)
        else:
            ax.bar(xs, vals, width=w, color=col, edgecolor="white",
                   lw=0.4 if len(series) > 4 else 0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, len(groups) - 0.5)


def _jsd_tiered(fig, cats, names, series, ylabel, tier_by=None, panels=None):
    """Grouped share bars (in %) on up to three side-by-side LINEAR axes, one per JSD_TIER_PCT
    tier of a category -- so a category at ~1% is not a stub under one at ~70%, and nothing
    needs a log axis. `series` is (label, {category: pct}, colour). The top tier is fixed at
    0-100; the others scale to their own tallest bar. Panel width = category count, so a bar
    is the same width in every panel; the y scales differ, so each keeps its ticks.
    Categories keep their given order inside a panel; panels run tallest tier first.

    WHICH TIER A CATEGORY LANDS IN is decided by `tier_by` -- {category: pct} -- and defaults
    to the category's tallest bar. Passing the reference ligand's own shares instead groups
    the panels by what a real ligand MAKES rather than by what the worst arm happens to do
    with it, which is a statement about the categories and stays put when an arm is added or
    dropped. The y limit is always the real tallest bar, so nothing is ever clipped: a tier
    chosen against the reference can still hold a bar many times the reference's height, and
    that is the finding, not a drawing error.

    `panels` -- a sequence of category tuples -- overrides the tiers outright; a panel whose
    every category reaches the top tier still gets the fixed 0-100 axis."""
    peak = {c: max((vals[c] or 0) for _, vals, _ in series) for c in cats}
    rank = peak if tier_by is None else {c: tier_by.get(c, 0) or 0 for c in cats}
    hi, mid = JSD_TIER_PCT
    if panels is not None:
        tiers = [[c for c in cats if c in p] for p in panels]
        leftover = [c for c in cats if not any(c in p for p in panels)]
        if leftover:
            raise ValueError(f"categories in no panel: {leftover}")
        tiers = [(0 if all(rank[c] >= hi for c in t) else 1, t) for t in tiers if t]
    else:
        tiers = [[c for c in cats if rank[c] >= hi], [c for c in cats if mid <= rank[c] < hi],
                 [c for c in cats if rank[c] < mid]]
        tiers = [(i, t) for i, t in enumerate(tiers) if t]
    gs = fig.add_gridspec(1, len(tiers), width_ratios=[len(t) for _, t in tiers])
    for k, (level, tier) in enumerate(tiers):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor("white")
        _jsd_grouped(ax, [names[c] for c in tier],
                     [(lab, [vals[c] for c in tier], col) for lab, vals, col in series])
        furniture(ax, ylabel=ylabel if k == 0 else None, xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        if level == 0:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(MultipleLocator(25))
        else:
            ax.set_ylim(0, max(peak[c] for c in tier) * 1.1)
    return [t for _, t in tiers]


def _jsd_round(v, nd=4):
    """A JSON number for a CSV cell. None is a JSD with nothing to compare (an arm with no
    bond of that type) and stays an empty cell rather than becoming 0."""
    return "" if v is None else round(v, nd)


def _jsd_pct(counts):
    counts = np.asarray(counts, float)
    return 100 * counts / counts.sum() if counts.sum() else counts


# ── bond distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-bond", needs=JSD_NEEDS)
def draw_jsd_bond(out):
    """Bond-distance JSD per bond type against the CrossDocked ligands (VoxBind Table 2)."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * 1.45, PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(lab, [d["arms"][lab]["jsd"]["bond"][t] for t in types]
                   + [d["arms"][lab]["jsd"]["bond_mean"]], soft(lab)) for lab, _, _ in arms]
        _jsd_grouped(ax, types + ["mean"], series)
        top = max(v for _, vals, _ in series for v in vals if v is not None)
        furniture(ax, ylabel="JSD to CrossDocked\nligands ↓", xloc=None)
        # AFTER furniture(), whose tick_params would reset the size
        ax.set_xticklabels(types + ["mean"], fontsize=13)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.axvline(len(types) - 0.5, color=AXIS, lw=GRID_LW, zorder=1)
        ax.set_ylim(0, top * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, ref=False, patch=True),
                       _jsd_key_cols(len(arms), wide))
        save(fig, out, f"bond_{variant}")

    rows = []
    for source, key in JSD_REFERENCES:
        rows += [[lab, source, d["arms"][lab]["n_mols"]]
                 + [_jsd_round(d["arms"][lab][key]["bond"][t]) for t in types]
                 + [_jsd_round(d["arms"][lab][key]["bond_mean"])] for lab, _, _ in all_arms]
    for lab, s in _jsd_ref_sets(d):
        rows.append([f"{lab}: bonds", "reference bond count (weighted where de-duplicated)",
                     s["n_mols"]] + [s["bond_n"][t] for t in types] + [""])
    for lab, vals in d["published"]["bond_jsd"].items():
        rows.append([lab, "published, vs CrossDocked test (100 pockets)", ""] + vals
                    + [round(float(np.mean(vals)), 4)])
    write_csv(out, "bond_jsd", ["set", "source", "n_mols"] + types + ["mean"], rows)


def _jsd_method_grid(out, d, arms, kinds, *, xlabel, stem, key=True):
    """Rows are methods, columns are what is measured. Every panel is ONE method's
    distribution over the reference's dashed outline, with that panel's JSD in its corner.

    The overlay puts nine curves in one axes, where the arm you are looking for is the one
    you cannot see; here each is read against the reference alone. y is shared down a
    column, so a method's peak height still compares with the other methods'.

    `kinds` is [(column title, set -> (pct, edges), arm -> JSD, xlim, x tick step)]."""
    n, m = len(arms), len(kinds)
    fig = _jsd_new(1.95 * m + 1.7, 1.12 * n + 1.5)
    axes = fig.subplots(n, m, sharex="col", sharey="col", squeeze=False)
    for j, (title, dist, jsd_of, xlim, xstep) in enumerate(kinds):
        ref_pct, _ = dist(d["reference"])
        top = float(ref_pct.max())
        for i, (lab, _, _) in enumerate(arms):
            ax = axes[i][j]
            ax.set_facecolor("white")
            pct, edges = dist(d["arms"][lab])
            top = max(top, float(pct.max()))
            ax.stairs(pct, edges, color=soft(lab), fill=True, alpha=DIST_FILL + 0.14, lw=0, zorder=2)
            ax.stairs(pct, edges, color=soft(lab), lw=DIST_LW - 0.4, zorder=3)
            ax.stairs(ref_pct, edges, color=REF_COLOR, lw=REF_LW - 0.2, ls=DASH, zorder=4)
            furniture(ax, xloc=xstep, xlim=xlim, xlabel=xlabel if i == n - 1 else None)
            ax.tick_params(labelsize=9.5, length=2.5)
            ax.xaxis.label.set_size(11.5)
            v = jsd_of(d["arms"][lab])
            ax.text(0.97, 0.93, "JSD —" if v is None else f"JSD {v:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
            if j == 0:
                ax.set_ylabel(display(lab), fontsize=12, labelpad=8)
            if i == 0:
                ax.set_title(title, fontsize=11, color=INK, pad=6)
        axes[0][j].set_ylim(0, top * 1.12)
    # The key names the reference only: the y unit is in each column head, and a longer key
    # outran the two-column pair grid and was cut off at both edges.
    _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                label=_jsd_ref_key())] if key else None, 1, fontsize=11)
    save(fig, out, stem)


# Renamed from fig-jsd-bond-length / bond-length-* (2026-09-15, on request), to match the
# "bond-distance JSD" of fig-jsd-bond that these histograms feed.
@figure("fig-jsd-bond-distance", needs=JSD_NEEDS)
def draw_jsd_bond_distance(out):
    """The bond-distance histograms behind fig-jsd-bond, one panel per bond type."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    bins = np.asarray(d["protocol"]["bond_bins"])
    step = float(bins[1] - bins[0])

    def drawn(counts):
        """Share of ALL of a set's bonds per merged bin. Index 0 and the last index are the
        under/overflow bins (searchsorted over the edges): they count in the denominator,
        so a set with many out-of-range bonds shows less mass, and are not drawn."""
        counts = np.asarray(counts, float)
        inner = counts[1:-1]
        inner = np.concatenate([inner, np.zeros((-len(inner)) % JSD_LEN_MERGE)])
        merged = inner.reshape(-1, JSD_LEN_MERGE).sum(1)
        edges = bins[0] + step * JSD_LEN_MERGE * np.arange(len(merged) + 1)
        # 118 inner bins pad to 120, but the last scoring edge is 1.69 Å: longer bonds are
        # overflow, so the last bar must not claim [1.69, 1.70) it holds nothing from
        edges[-1] = bins[-1]
        return (100 * merged / counts.sum() if counts.sum() else merged), edges

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x bond-type grid (see _jsd_method_grid); core keeps the
            # overlay, where two curves over the reference still read directly
            _jsd_method_grid(out, d, arms, [
                (f"{t}\n% per {step * JSD_LEN_MERGE:.2f} Å",
                 (lambda s, t=t: drawn(s["bond_counts"][t])),
                 (lambda a, t=t: a["jsd"]["bond"][t]), (bins[0], bins[-1]), 0.2) for t in types],
                # no key (2026-09-15, on request); pair-all keeps its reference key
                xlabel="Bond distance (Å)", stem=f"bond_distance_{variant}", key=False)
            continue
        fig = _jsd_new(FIG_W * 1.55, PANEL_H * 1.62)
        axes = fig.subplots(2, 4)
        for i, t in enumerate(types):
            ax = axes.flat[i]
            ax.set_facecolor("white")
            for lab, _, _ in arms:
                pct, edges = drawn(d["arms"][lab]["bond_counts"][t])
                ax.stairs(pct, edges, color=soft(lab), lw=DIST_LW, zorder=3)
                ax.stairs(pct, edges, color=soft(lab), fill=True, alpha=DIST_FILL, lw=0, zorder=2)
            pct, edges = drawn(d["reference"]["bond_counts"][t])
            ax.stairs(pct, edges, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xloc=0.2, xlim=(bins[0], bins[0] + step * 120),
                      ylabel=f"% of bonds\nper {step * JSD_LEN_MERGE:.2f} Å" if i % 4 == 0 else None,
                      xlabel="Bond distance (Å)" if i >= 4 else None)
            ax.tick_params(labelsize=11.5)
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.set_ylim(bottom=0)
            # a panel LABEL, not a title
            ax.set_title(t, loc="left", fontsize=12.5, color=INK, pad=5)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"bond_distance_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for t in types:
            c = s["bond_counts"][t]
            total = sum(c)
            lo = [float("-inf")] + bins.tolist()
            hi = bins.tolist() + [float("inf")]
            rows += [[lab, t, round(a, 3), round(b, 3), n, round(100 * n / total, 4) if total else 0]
                     for a, b, n in zip(lo, hi, c)]
    write_csv(out, "bond_distance_hist",["set", "bond_type", "len_lo", "len_hi", "n", "pct"], rows)


# ── pair distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-pair", needs=JSD_NEEDS)
def draw_jsd_pair(out):
    """All-atom (<12 Å) and C–C (<2 Å) pair-distance distributions (TargetDiff Fig. 2)."""
    d, all_arms = _jsd_data()
    use_style()
    panels = (("All_12A", "All heavy-atom pairs · distance (Å)", None),
              ("CC_2A", "C–C pairs under 2 Å · distance (Å)", JSD_CC_XLIM))

    def dist(key):
        edges = np.asarray(d["protocol"]["pair_bins"][key])
        return lambda s: (_jsd_pct(s["pair_counts"][key])[1:-1], edges)

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x pair-kind grid (see _jsd_method_grid); core keeps the overlay
            step = {k: float(np.diff(d["protocol"]["pair_bins"][k][:2])[0]) for k, _, _ in panels}
            _jsd_method_grid(out, d, arms, [
                (f"All heavy-atom pairs < 12 Å\n% per {step['All_12A']:.2f} Å", dist("All_12A"),
                 lambda a: a["jsd"]["pair"]["All_12A"], (0, 12), 4),
                (f"C–C pairs < 2 Å\n% per {step['CC_2A']:.2f} Å", dist("CC_2A"),
                 lambda a: a["jsd"]["pair"]["CC_2A"], JSD_CC_XLIM, 0.5)],
                xlabel="Distance (Å)", stem=f"pair_{variant}")
            continue
        fig = _jsd_new(FIG_W * 1.45, PANEL_H * 1.05)
        axes = fig.subplots(1, 2)
        for ax, (key, xlabel, xlim) in zip(axes, panels):
            ax.set_facecolor("white")
            edges = np.asarray(d["protocol"]["pair_bins"][key])
            for lab, _, _ in arms:
                ax.stairs(_jsd_pct(d["arms"][lab]["pair_counts"][key])[1:-1], edges,
                          color=soft(lab), lw=DIST_LW, zorder=3)
            ax.stairs(_jsd_pct(d["reference"]["pair_counts"][key])[1:-1], edges,
                      color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xlabel=xlabel, xloc=None, xlim=xlim or (edges[0], edges[-1]),
                      # two lines, as bond-distance's: at the axis-name size _jsd_axis_weight sets,
                      # one line ran past both ends of the panel and lost its unit
                      ylabel=f"% of pairs\nper {edges[1] - edges[0]:.2f} Å")
            ax.tick_params(labelsize=12.5)
            ax.xaxis.label.set_size(13.5)
            ax.yaxis.label.set_size(13.5)
            ax.set_ylim(bottom=0)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"pair_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for key, _, _ in panels:
            edges = d["protocol"]["pair_bins"][key]
            pct = _jsd_pct(s["pair_counts"][key])
            rows += [[lab, key, round(a, 4), round(b, 4), n, round(float(q), 4)]
                     for a, b, n, q in zip(edges[:-1], edges[1:], s["pair_counts"][key][1:-1], pct[1:-1])]
    write_csv(out, "pair_hist", ["set", "pairs", "dist_lo", "dist_hi", "n", "pct"], rows)


# ── atom types and the headline numbers ─────────────────────────────────────────
@figure("fig-jsd-atom-type", needs=JSD_NEEDS)
def draw_jsd_atom_type(out):
    """Heavy-atom element shares against the CrossDocked ligands: C | N, O | F, P, S, Cl."""
    d, all_arms = _jsd_data()
    use_style()
    elems = d["protocol"]["atom_types"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        # C | N, O | F, P, S, Cl on three linear axes (2026-09-14, on request; it was one log axis)
        series = [(REF_LABEL, {e: 100 * d["reference"]["atom_frac"][e] for e in elems}, REF_COLOR)]
        series += [(lab, {e: 100 * d["arms"][lab]["atom_frac"][e] for e in elems}, soft(lab))
                   for lab, _, _ in arms]
        _jsd_tiered(fig, elems, {e: e for e in elems}, series, "% of heavy atoms")
        # no key on `all` (2026-09-15, on request); `core` keeps its own
        _jsd_key_above(fig, None if wide else _jsd_handles(arms, patch=True),
                       _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"atom_type_{variant}")

    rows = [[lab, s["n_mols"]] + [round(100 * s["atom_frac"][e], 3) for e in elems]
            + [round(100 * s["atom_other_frac"], 3), "", "", ""] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, d["arms"][lab]["n_mols"]] + [round(100 * d["arms"][lab]["atom_frac"][e], 3) for e in elems]
             + [round(100 * d["arms"][lab]["atom_other_frac"], 3)]
             + [_jsd_round(d["arms"][lab][key]["atom_type"]) for _, key in JSD_REFERENCES]
             for lab, _, _ in all_arms]
    write_csv(out, "atom_type", ["set", "n_mols"] + [f"pct_{e}" for e in elems]
              + ["pct_other_not_scored"] + [f"atom_type_{key}" for _, key in JSD_REFERENCES], rows)


@figure("fig-jsd-summary", needs=JSD_NEEDS)
def draw_jsd_summary(out):
    """Mean bond, all-atom pair, C–C pair and atom-type JSD per arm, side by side."""
    d, all_arms = _jsd_data()
    use_style()
    for variant, arms in _jsd_variants(all_arms):
        fig = _jsd_new(FIG_W * 1.75, 0.5 * len(arms) + 1.55)
        axes = fig.subplots(1, len(JSD_SUMMARY), sharey=True)
        ys = np.arange(len(arms))[::-1]
        for ax, (name, get) in zip(axes, JSD_SUMMARY):
            ax.set_facecolor("white")
            vals = [get(d["arms"][lab]["jsd"]) for lab, _, _ in arms]
            ax.barh(ys, [np.nan if v is None else v for v in vals], height=0.66,
                    color=[soft(lab) for lab, _, _ in arms], zorder=3)
            top = max([v for v in vals if v is not None], default=1.0)
            # every value is labelled: this figure IS the table, drawn. A None is a JSD with
            # nothing to compare and says so, rather than drawing as a zero-length "best"
            for y, v in zip(ys, vals):
                ax.text((v or 0) + top * 0.03, y, "—" if v is None else f"{v:.3f}",
                        va="center", ha="left", fontsize=11.5, color=INK, zorder=4)
            furniture(ax, xlabel=name, xloc=None, xlim=(0, top * 1.38))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=11.5, length=5)
            ax.grid(False, axis="y")
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels([display(lab) for lab, _, _ in arms], fontsize=14.5)
        axes[0].set_ylim(-0.6, len(arms) - 0.4)
        # heavier axes, larger names (2026-09-15, on request): the figure the ratios come from
        _jsd_axis_weight(fig)
        fit(fig, pad=0.5)
        save(fig, out, f"summary_{variant}")
    head = ["set", "n_mols", "n_disconnected_dropped"]
    cols = ("bond_mean", "pair_all_12A", "pair_cc_2A", "atom_type")
    for _, key in JSD_REFERENCES:
        head += [f"{c}_{key}" for c in cols]
    write_csv(out, "summary", head,
              [[lab, d["arms"][lab]["n_mols"], d["arms"][lab]["n_disconnected"]]
               + [_jsd_round(get(d["arms"][lab][key])) for _, key in JSD_REFERENCES for _, get in JSD_SUMMARY]
               for lab, _, _ in all_arms])


# ── rings ───────────────────────────────────────────────────────────────────────
@figure("fig-jsd-ring-size", needs=JSD_NEEDS)
def draw_jsd_ring_size(out):
    """Share of rings by size 3–9 (TargetDiff Table 2), with the table itself as CSV."""
    d, all_arms = _jsd_data()
    use_style()
    sizes = d["published"]["ring_size_pct_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        series = [(REF_LABEL, {s: d["reference"]["ring_size_pct"][s] for s in sizes}, REF_COLOR)]
        series += [(lab, {s: d["arms"][lab]["ring_size_pct"][s] for s in sizes}, soft(lab))
                   for lab, _, _ in arms]
        # Tiered linear axes (2026-09-14, on request; it was one log axis over 0.01-100%).
        # TIERED BY THE CRYSTAL LIGANDS, NOT BY THE TALLEST BAR (2026-09-14, on request): the
        # panels then read 6-ring | 5-ring | the sizes a real ligand barely makes, and 3- and
        # 7-rings sit in that third panel where the reference puts them (1.5% and 0.8%)
        # instead of being lifted into the middle one by AR's 30% 3-rings and TargetDiff's 12%
        # 7-rings. Those two bars are still drawn at full height -- the panel scales to its
        # tallest bar, so the third panel says "sizes the reference avoids, and by how far the
        # arms overshoot them". The grouping no longer moves when an arm is added or dropped:
        # `core` and `all` now split the same way.
        # Then FIXED PANELS (2026-09-14, on request): 6 | 3, 5, 7 | 4, 8, 9 -- see
        # JSD_RING_PANELS. tier_by still decides which panel is the 0-100 one.
        _jsd_tiered(fig, sizes, {s: f"{s}-ring" for s in sizes}, series, "% of rings",
                    tier_by=d["reference"]["ring_size_pct"], panels=JSD_RING_PANELS)
        # no key on `all` (2026-09-15, on request); `core` keeps its own
        _jsd_key_above(fig, None if wide else _jsd_handles(arms, patch=True),
                       _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"ring_size_{variant}")

    head = ["set", "source"] + sizes + ["10+ of all rings"]
    rows = [[lab, "reference"] + [round(s["ring_size_pct"][z], 1) for z in sizes]
            + [round(s["ring_size_pct"]["10+ of all"], 1)] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, f"this eval ({d['arms'][lab]['n_pockets']} pockets)"]
             + [round(d["arms"][lab]["ring_size_pct"][s], 1) for s in sizes]
             + [round(d["arms"][lab]["ring_size_pct"]["10+ of all"], 1)] for lab, _, _ in all_arms]
    rows += [[lab, "published TargetDiff Table 2 (reference = test)"] + vals + [""]
             for lab, vals in d["published"]["ring_size_pct"].items()]
    write_csv(out, "ring_size_pct", head, rows)


@figure("fig-jsd-n-rings", needs=JSD_NEEDS)
def draw_jsd_n_rings(out):
    """Share of molecules by number of rings (VoxBind Fig. 10, middle row), every method in one axes."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(REF_LABEL, _jsd_ring_rows(d["reference"])[1], REF_COLOR)]
        series += [(lab, _jsd_ring_rows(d["arms"][lab])[1], soft(lab)) for lab, _, _ in arms]
        _jsd_grouped(ax, labels, series)
        furniture(ax, ylabel="% of molecules", xlabel="Rings per molecule", xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"n_rings_{variant}")

    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    write_csv(out, "n_rings", ["set", "mean_rings_per_molecule"] + [f"pct_{b}" for b in labels],
              [[lab, round(s["mean_n_rings"], 3)] + [round(float(q), 3) for q in _jsd_ring_rows(s)[1]]
               for lab, s in sets])


@figure("fig-jsd-aromatic", needs=JSD_NEEDS)
def draw_jsd_aromatic(out):
    """Aromatic share of each molecule's heavy atoms (VoxBind Fig. 10, bottom row) and of its rings."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [f"{k / 10:.1f}–{(k + 1) / 10:.1f}" for k in range(10)]
    # Two definitions, one panel each. VoxBind Fig. 10 plots the aromatic share of a
    # molecule's heavy ATOMS; the share of its RINGS that are aromatic (every ring bond
    # aromatic) says the same thing without the ring-free atoms diluting it, and is only
    # defined for a molecule that has a ring -- so its panel counts those molecules alone.
    panels = (
        ("Aromatic share of a molecule's heavy atoms",
         lambda s: _jsd_ring_rows(s)[2], "mean_arom_atom_frac", "% of molecules"),
        ("Aromatic share of a molecule's rings (molecules with ≥1 ring)",
         lambda s: _jsd_pct(s["arom_ring_frac_counts"]), "mean_arom_ring_frac", "% of molecules\nwith a ring"),
    )
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H * 1.9)
        axes = fig.subplots(2, 1)
        for ax, (xlabel, get, _, ylabel) in zip(axes, panels):
            ax.set_facecolor("white")
            series = [(REF_LABEL, get(d["reference"]), REF_COLOR)]
            series += [(lab, get(d["arms"][lab]), soft(lab)) for lab, _, _ in arms]
            _jsd_grouped(ax, labels, series)
            furniture(ax, ylabel=ylabel, xlabel=xlabel, xloc=None)
            ax.tick_params(axis="x", length=0, labelsize=11 if wide else 10)
            ax.grid(False, axis="x")
            ax.xaxis.label.set_size(13.5)
            ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"aromatic_{variant}")

    rows = []
    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    for name, (_, get, mean_key, _) in zip(("atoms", "rings"), panels):
        rows += [[lab, name, _jsd_round(s[mean_key])] + [round(float(q), 3) for q in get(s)]
                 for lab, s in sets]
    write_csv(out, "aromatic", ["set", "aromatic_share_of", "mean"] + [f"pct_{b}" for b in labels], rows)


JSD_RING_ROWS = (
    ([str(k) for k in range(3, 10)] + ["≥10"], "Ring size\n% of rings", (0, 2, 4, 6, 7)),
    ([str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"],
     "Rings per molecule\n% of molecules", (0, 2, 4, 6, 8)),
    (None, "Aromatic atom fraction\n% of molecules", None),
)


def _jsd_ring_rows(s):
    """The three Fig. 10 histograms of one set, in percent: ring sizes over ALL rings
    (10+ pooled), rings per molecule (8+ pooled), aromatic share of heavy atoms."""
    rs = {int(k): v for k, v in s["ring_size_counts"].items()}
    sizes = [rs.get(k, 0) for k in range(3, 10)] + [sum(v for k, v in rs.items() if k >= 10)]
    nr = {int(k): v for k, v in s["n_rings_counts"].items()}
    counts = [nr.get(k, 0) for k in range(JSD_NRINGS_MAX)] \
        + [sum(v for k, v in nr.items() if k >= JSD_NRINGS_MAX)]
    arom = np.asarray(s["arom_frac_counts"], float).reshape(-1, JSD_AROM_MERGE).sum(1)
    return [_jsd_pct(sizes), _jsd_pct(counts), _jsd_pct(arom)]


@figure("fig-jsd-rings", needs=JSD_NEEDS)
def draw_jsd_rings(out):
    """VoxBind Fig. 10: ring sizes, rings per molecule and aromatic fraction, one column per arm."""
    d, all_arms = _jsd_data()
    use_style()
    ref_rows = _jsd_ring_rows(d["reference"])
    for variant, arms in _jsd_variants(all_arms):
        n = len(arms)
        tick_fs = 10 if n > 2 else 11.5
        fig = _jsd_new(max(FIG_W, 2.05 * n + 1.5), 7.2)
        # The paper's layout: a column per method, the reference as the same dashed outline
        # in every panel, and y shared along a row so columns compare by eye. Column heads
        # name the method -- the one place this family puts text above an axes.
        axes = fig.subplots(3, n, sharey="row", squeeze=False)
        per = {lab: _jsd_ring_rows(d["arms"][lab]) for lab, _, _ in arms}
        for c, (lab, _, _) in enumerate(arms):
            rows = per[lab]
            for r, (labels, ylabel, ticks) in enumerate(JSD_RING_ROWS):
                ax = axes[r][c]
                ax.set_facecolor("white")
                if labels is None:
                    edges = np.linspace(0, 1, len(rows[r]) + 1)
                    x = (edges[:-1] + edges[1:]) / 2
                    width = (edges[1] - edges[0]) * 0.86
                else:
                    x = np.arange(len(rows[r]))
                    edges = np.arange(len(rows[r]) + 1) - 0.5
                    width = 0.78
                ax.bar(x, rows[r], width=width, color=soft(lab), lw=0, zorder=3)
                ax.stairs(ref_rows[r], edges, color=REF_COLOR, lw=REF_LW, ls=DASH,
                          baseline=None, zorder=4)
                furniture(ax, xloc=None, ylabel=ylabel if c == 0 else None)
                ax.grid(False, axis="x")
                ax.tick_params(labelsize=tick_fs)
                ax.yaxis.label.set_size(12)
                if labels is None:
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(["0", "0.5", "1"])
                else:
                    ax.set_xlim(edges[0], edges[-1])
                    ax.set_xticks(list(ticks))
                    ax.set_xticklabels([labels[i] for i in ticks])
            axes[0][c].set_title(display(lab), fontsize=13, color=INK, pad=7)
        # The row's y range is set ONCE, from every column and the reference. A per-panel
        # set_ylim(bottom=0) freezes the shared top at whatever the columns drawn so far
        # reached, and clipped FuncBind's 83% six-ring bar at AR's 70%.
        for r in range(len(JSD_RING_ROWS)):
            top = max(max(float(per[lab][r].max()) for lab in per), float(ref_rows[r].max()))
            axes[r][0].set_ylim(0, top * 1.08)
        _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                    label=_jsd_ref_key())], 1)
        save(fig, out, f"rings_{variant}")

    rows = []
    names = ("ring_size", "rings_per_molecule", "aromatic_fraction")
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for name, (labels, _, _), pct in zip(names, JSD_RING_ROWS, _jsd_ring_rows(s)):
            if labels is None:
                labels = [f"{i / len(pct):.1f}-{(i + 1) / len(pct):.1f}" for i in range(len(pct))]
            rows += [[lab, name, b, round(float(q), 3)] for b, q in zip(labels, pct)]
    write_csv(out, "rings_hist", ["set", "histogram", "bin", "pct"], rows)


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
# ══════════════════════════════════════════════════════════════════════════════════════
# fig-palette-methods — the shared method palette, drawn and exported
#
# THE PALETTE IS DEFINED ONCE, IN 00_core.py: COLORS (the saturated hue every method owns),
# SOFT (the lighter tint the eight-method figures draw through soft()) and PALE (the fill
# tints). This figure does not define anything -- it READS those three tables, so the sheet
# and the CSV cannot drift from what the figures actually draw. Regenerate it after any
# palette change and the CSV is the file to hand to a co-author, a slide deck or a notebook
# that has to match these figures without importing draw.py.
#
# WHY THREE TABLES AND NOT ONE. A method is one hue everywhere, but the same hue cannot do
# every job: nine series on one axis need the light end (soft), a box fill under a dark
# median needs lighter still (pale), and a three-line panel wants the saturated original.
# Reading the sheet left to right is reading those three jobs.
# ══════════════════════════════════════════════════════════════════════════════════════

# The nine series of the eight-method PoseCheck figures, in the drawn order, then everything
# else COLORS registers -- our v2 arm and the MCP fine-tune ramp, which are the same FuncBind
# brown darkening with training rather than four categorical hues.
PAL_ORDER = ["Reference ligand", "AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind",
             "TargetDiff", "VoxBind", "CoDE"]
PAL_EXTRA = ["Ours v2", "FuncBind vanilla", "FuncBind ft 3.17M", "FuncBind ft 8.21M",
             "FuncBind ft 26.1M"]
PAL_COLS = ["COLORS · color()", "SOFT · soft()", "PALE · pale()"]
PAL_SW_W, PAL_SW_H = 0.88, 0.66      # swatch, in the one-unit-per-column grid
PAL_ROW_H = 0.42                     # inches per row


def _pal_ink(hexv):
    """Black or white for text ON a swatch, by that swatch's luminance."""
    r, g, b = (int(hexv[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def _pal_rows():
    """(canonical key, palette, soft-or-None, pale-or-None, drawn label) per method.

    SOFT and PALE are read with .get: most methods have no entry, and that is information --
    soft() falls back to the palette colour for them, which is what the sheet shows."""
    rows = []
    for lab in PAL_ORDER + PAL_EXTRA:
        key = ALIASES.get(lab, lab)
        rows.append((key, COLORS[key], SOFT.get(key), PALE.get(key), DISPLAY.get(key, key)))
    return rows


@figure("fig-palette-methods", folder="palette")
def draw_palette_methods(out):
    """Every method's registered tints, as a swatch sheet and a CSV."""
    rows = _pal_rows()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8.4, PAL_ROW_H * len(rows) + 1.25), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_xlim(-2.35, len(PAL_COLS) + 0.05)
        # The bottom edge clears the last row AND the footnote under it.
        ax.set_ylim(-len(rows) - 0.3, 1.45)
        ax.axis("off")
        for j, name in enumerate(PAL_COLS):
            ax.text(j + PAL_SW_W / 2, 0.72, name, ha="center", va="center", fontsize=11.5,
                    color=INK)
        for i, (key, c, s, p, disp) in enumerate(rows):
            y = -i
            ax.text(-0.16, y, disp, ha="right", va="center", fontsize=12, color=INK)
            for j, hexv in enumerate((c, s, p)):
                if not hexv:
                    # No entry: soft()/pale() do not invent one, and neither does this sheet.
                    ax.text(j + PAL_SW_W / 2, y, "—", ha="center", va="center", fontsize=11,
                            color=AXIS)
                    continue
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j, y - PAL_SW_H / 2), PAL_SW_W, PAL_SW_H, facecolor=hexv,
                    edgecolor=LEGEND_EDGE, linewidth=0.6))
                ax.text(j + PAL_SW_W / 2, y, hexv.upper(), ha="center", va="center",
                        fontsize=9, color=_pal_ink(hexv))
        ax.text(-2.3, -len(rows) + 0.1,
                "soft() falls back to the palette colour where SOFT has no entry; "
                "pale() raises instead.",
                ha="left", va="center", fontsize=9.5, color=AXIS)
        fit(fig, pad=0.4)
        save(fig, out, "palette_methods")
    write_csv(out, "palette_methods",
              ["method", "palette_hex", "soft_hex", "pale_hex", "legend_label"],
              [[k, c, s or "", p or "", d] for k, c, s, p, d in rows])
    print(f"  {len(rows)} methods · {sum(1 for r in rows if r[2])} with a soft tint · "
          f"{sum(1 for r in rows if r[3])} with a pale tint")
    print(f"  wrote {out}")


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
# The dashes come from METHOD_DASH in 00_core.py now (2026-09-14), so this family and the
# rotatable-bond one cannot drift apart again. AR and Pocket2Mol both carried (5, 2) here,
# which left them separable by hue alone; dash() gives Pocket2Mol its own pattern.
PCSZ_BASELINES = [
    ("AR",         "AR",         1.6, dash("AR")),
    ("Pocket2Mol", "Pocket2Mol", 1.6, dash("Pocket2Mol")),
    ("DiffSBDD",   "DiffSBDD",   1.6, dash("DiffSBDD")),
    ("DecompDiff", "DecompDiff", 1.6, dash("DecompDiff")),
    ("FuncBind",   "FuncBind",   1.6, dash("FuncBind")),
]
# TargetDiff is violet and still dashed: it used to be orange, which put two BASELINES in
# the same hue family as each other, and the dash costs nothing now that the collision is
# gone. These labels are the drawn labels, not the ARMS keys -- they reach color() through
# the alias table.
PCSZ_LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", 1.6,
     dash("TargetDiff")),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    3.4,
     dash("VoxBind σ=0.9")),
    ("CoDE",          f"{E}/frozenenc_probes/posecheck_full/ours_v1",    3.8, dash("CoDE")),
]


def _pcsz_ordered(labels):
    """`labels` in RB_ORDER — the sequence the rotbond grid and the shared legend draw in.

    THE LIST ORDER IS A LOADING ORDER, NOT A DISPLAY ORDER (2026-09-15). PCSZ_BASELINES is
    the five published baselines, which come out of one export, and PCSZ_LOCAL is the three
    arms scored from run trees; concatenating them put DecompDiff and FuncBind AHEAD of
    TargetDiff, which is nobody's row order. Sorting here makes this figure read in the same
    sequence as the rotbond grid and the legend that names it.

    SPELLINGS CANONICALISE FIRST. PCSZ_LOCAL calls it "VoxBind σ=0.9" and RB_ORDER calls it
    "VoxBind", so a raw membership test would report it missing and take the arm out of the
    figure. Anything the table does not name raises, exactly as _rb_order does: a method
    that quietly vanishes from a figure is worse than a failed build.

    RB_ORDER and ALIASES are read at CALL time -- RB_ORDER lives in a part assembled after
    this one, so it cannot be referenced while this module is being defined."""
    rank = {lab: i for i, lab in enumerate(RB_ORDER)}
    canon = {l: ALIASES.get(l, l) for l in labels}
    unknown = [l for l, c in canon.items() if c not in rank]
    if unknown:
        raise SystemExit(f"not in RB_ORDER: {unknown}")
    return sorted(labels, key=lambda l: rank[canon[l]])
# Weight per METHOD, keyed by the canonical name so the other family can read it: the two
# figures are a pair and a series that is thicker in one of them reads as a different series
# (2026-09-13). NOT one flat weight -- our two arms are the subject and the five published
# baselines are context, which is the same reason they are dashed and these are solid.
# ONE WEIGHT FOR EVERY SERIES THAT IS NOT OURS (2026-09-14): the five published baselines,
# TargetDiff and the crystal ligands all draw at 1.6, and dash alone tells them apart. Only
# VoxBind (3.4) and CoDE (3.8) are heavier, which is the whole point of the weight channel.
PCSZ_REF_LW = 1.6
# What the crystal ligands are CALLED in this family's key (2026-09-14). The shared REF_LABEL
# ("Reference ligand") still names them everywhere else; only the drawn key is shortened, and
# _pcsz_key splits the key on this name.
PCSZ_REF_NAME = "Reference"
PCSZ_LINE_LW = {**{ALIASES.get(lab, lab): lw for _, lab, lw, _ in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): lw for lab, _, lw, _ in PCSZ_LOCAL}}

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
# The clash axis is log10(x+1), so it can hold the zero-clash poses; these are the labels put
# back on it in real counts (2026-09-14).
PCSZ_CLASH_TICKS = [0, 1, 10, 100]
# 80 % of the original 8.6 x 5.6 in canvas both ways, then 1.2x wider again so the lower-right
# key leaves room for the curves' long right tails, then 0.9x both ways (2026-09-13). The
# per-atom all-methods figure reads its aspect from this, so keep scaling both sides together
# unless that one should change too.
# PCSZ_SHRINK pulls the canvas in once more (2026-09-13) WITHOUT touching the type sizes or
# the dpi, so the same ink fills a smaller frame -- that, not a font change, is what makes the
# panel read fuller. PCSZ_LW_SCALE thickens the curves by the same argument: at this canvas
# the nine series were drawn for a frame a fifth wider. The trailing 1.1x is WIDTH ONLY
# (2026-09-13): the lower-right key and the curves' right tails were tight against each other
# once the canvas came in.
PCSZ_SHRINK, PCSZ_LW_SCALE = 0.85, 1.3
PCSZ_ECDF_SIZE = (8.6 * 0.8 * 1.2 * 0.9 * PCSZ_SHRINK * 1.1,
                  5.6 * 0.8 * 0.9 * PCSZ_SHRINK * 0.9)   # trailing 0.9 is HEIGHT only
# Axis-name size for the ECDF, and for the per-atom all-methods figure drawn in its style.
# 1.4x the family's original 11.5 pt (2026-09-13).
PCSZ_LABEL_FS = 11.5 * 1.4
# The y name's gap off its tick labels, DOUBLED (2026-09-14). Measured off the drawn PNGs the
# matplotlib default (labelpad 4.0) buys 5.5 pt here and 6.8 pt on the per-atom panel; a pad of
# 10 is twice that on both. Shared, so the pair keeps one left margin.
PCSZ_YPAD = 10
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
# handlelength 1.6 showed barely one repeat of a dash pattern, so the five baselines' second
# identity channel was unreadable in the key; 2.8 fits two to three repeats of every pattern.
PCSZ_KEY_KW = dict(fontsize=9.5, handlelength=2.8, handletextpad=0.5, labelspacing=0.32,
                   borderpad=0.4, columnspacing=1.2)
# The frame's face, translucent so a curve running under the key is still followable, still
# opaque enough to keep the dotted grid out of the text (2026-09-13). Set on the FACE, not as
# the artist's alpha, which would fade the grey rule with it.
PCSZ_KEY_FACE = (1.0, 1.0, 1.0, 0.82)
# The reference line is anchored on the grid's top edge, so the gap between them is the two
# legends' borderpad back to back. Pull it down by this much of the axes to close half of it.
PCSZ_KEY_GAP = 0.022


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
    ref = [p for p in pairs if p[1] == PCSZ_REF_NAME]
    methods = [p for p in pairs if p[1] != PCSZ_REF_NAME]
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
                              bbox_to_anchor=((bb.x0 + bb.x1) / 2, bb.y1 - PCSZ_KEY_GAP),
                              **PCSZ_KEY_KW))
    fig.canvas.draw()
    boxes = [l.get_window_extent(render()).transformed(ax.transAxes.inverted()) for l in legs]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    # An AXES artist, not a figure one: a figure-level patch is drawn after the whole axes and
    # would cover the legend text. Translucent white under the text, over the curves.
    ax.add_artist(matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes, facecolor=PCSZ_KEY_FACE,
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
                # display(): the key carries VoxBind's sigma as a subscript (2026-09-14). It is
                # the ONLY name DISPLAY overrides, so every other entry is unchanged, and the
                # series keeps its plain label everywhere else -- CSV, violin ticks, data keys.
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=soft(label),
                        lw=lw * PCSZ_LW_SCALE, ls=ls,
                        label=display(label), solid_capstyle="round")
            r = np.asarray(ref[b]["strain"], dtype=float)
            if r.size >= 3:
                x = np.sort(np.clip(r, PCSZ_XFLOOR, None))
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR,
                        # SOLID, and AR took its dash (2026-09-14): solid was the one style no
                        # baseline had, so it now marks the crystal ligands and our two arms --
                        # the series a reader returns to -- and every baseline carries a dash.
                        lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls="-",
                        label=PCSZ_REF_NAME, solid_capstyle="round")
            ax.set_xscale("log")
            ax.set_xlim(PCSZ_XFLOOR, PCSZ_XTOP)
            ax.set_ylim(0, 1.0)
            # "· log scale" dropped (2026-09-14); the decade ticks say it.
            ax.set_xlabel("UFF strain energy (kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS)
            # Two lines (2026-09-14), as on the per-atom panel it pairs with.
            ax.set_ylabel("Cumulative\nprobability", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
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
            # The strain ECDF's canvas (2026-09-14), so the two PoseCheck measurements are the
            # same shape on a page; its rc is already in force for the whole family.
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax, grid_axis="y")
            vals, cols, ticks = [], [], []
            for label, colour, lw, ls in series:
                v = data_by_label[label][b]["clash"]
                if len(v) < 20:
                    continue
                vals.append(v)
                # soft(), as every other eight-method figure draws (2026-09-14). `colour` off
                # the series table is the saturated palette entry, which now only the
                # three-arm and docking figures use.
                cols.append(soft(label))
                ticks.append(display(label))
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
            # DECADES ONLY (2026-09-14): zero, then 1, 10, 100. The nine intermediate ticks the
            # axis used to carry were reading aids for numbers that are now in the CSV.
            ticks_at = [t for t in PCSZ_CLASH_TICKS if t <= hi * 1.6]
            ax.set_yticks(tf(ticks_at))
            ax.set_yticklabels([str(t) for t in ticks_at])
            # The headroom the per-violin mean/median strip needed is gone with it.
            ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.06)
            # TURNED (2026-09-14): on the ECDF's narrower canvas eight names laid flat ran
            # into each other -- Pocket2Mol into DiffSBDD into DecompDiff. Anchored at the
            # right so each name ends under its own violin.
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(ticks, fontsize=9.5, rotation=30, ha="right")
            ax.set_ylabel("Steric clashes\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            # NO TITLE and NO per-violin mean/med strip (2026-09-14), as on the strain ECDF:
            # the bin is in the file name (`clash-violin-by-size-<bin>`) and every number the
            # strip carried is a column of the CSV written beside the figure.
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_violin_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


# ── fig-posecheck-clash-box-by-method ──────────────────────────────────────────────────
# The rotatable-bond grid's box format on ONE panel: method along x, clashes up y. There is
# nothing to facet here -- this is every pose the method made, at every ligand size -- so the
# 3x3 block collapses to a single axes and the bins live in the violin above.
# THE FILL IS THE METHOD, not its median (2026-09-14). The grid needed a colourmap because a
# method owns nine panels there and the fill was the only place its median could be read; on
# one axes the eight medians are side by side already, so the fill goes back to carrying
# identity -- the same soft() tint the violin and the ECDF give that method -- and the
# colourbar that encoded the median comes off.
# 0.8x the grid's 2.0 (2026-09-14): a box here is far wider than one in a nine-panel grid,
# and the same weight read heavier across it.
PCSZ_CBOX_MED_LW = 2.0 * 0.8
# Wider again (2026-09-14), on top of the width the colourbar gave back. TALL is its own
# factor because the x names went up to the y name's size: nine of them at 30 degrees eat
# height that the boxes used to have.
PCSZ_CBOX_WIDE, PCSZ_CBOX_TALL = 1.15, 1.2
# ROTATED (2026-09-14): the methods run DOWN the y axis and the clashes across x. Nine rows
# want height rather than width, and the names stop needing their 30-degree tilt -- upright
# against a y axis is the main thing rotating buys. Kept as its own pair rather than swapping
# the two above, so going back to the upright panel stays one edit.
# Width at 0.6x (2026-09-14), back up from 0.4. The method names sit OUTSIDE the axes but
# INSIDE the figure, and constrained layout fits everything within figsize -- so a cut here
# comes out of the drawing area alone, the names keeping their full width. At 0.4 that left
# the nine boxes about 285 px of a 492 px image to share, with the labels taking 42 % of it,
# and the median spread the panel exists to show (4 against 8) stopped being legible.
PCSZ_CBOX_WIDE_V, PCSZ_CBOX_TALL_V = 0.5, 1.55
# The method names' tilt. A rotated label's horizontal footprint is w*cos(t) + h*sin(t), so
# for names this long against this type size the saving is modest until the angle is steep:
# roughly 5 % at 30 degrees, 17 % at 45, 36 % at 60. Level is easier to read, so this is the
# smallest angle that buys anything rather than the one that buys most.
PCSZ_CBOX_NAME_ROT = 30
# Minor ticks, as a fraction of the frame's weight. They SUBDIVIDE a decade rather than bound
# the panel, and carrying the frame's full 2.0 made the eight of them between each pair of
# decades read as more axis lines instead of as gradations on one.
PCSZ_CBOX_MINOR_LW = 0.5
# WHERE THE VALUE AXIS STOPS (2026-09-14). Left to itself it runs to the largest flier any
# method produced -- DiffSBDD's 333 -- which spends the right third of the panel on a handful
# of points and squeezes all nine boxes into the left half. DiffSBDD is the ONLY method that
# reaches past this: the next largest maximum is TargetDiff's 167, and every other method
# tops out between 26 and 97, so the cap costs one method's extreme tail and nothing else.
PCSZ_CBOX_HI = 200.0
# HOW MUCH WHITE GOES INTO A FILL (2026-09-14). The fills already carry EXACTLY the soft()
# hexes the ECDF and the per-atom panel draw their lines in -- checked on the rendered
# pixels, #EE9190 / #F0A6C0 / #8DCB92 / #C6B46A / #8291E8 / #7FC4D1 match byte for byte
# across the two files. They still read more saturated, and that is an area effect, not a
# palette drift: a filled box is ~100x the ink of a 3 px line of the same colour. So the
# FILL, and only the fill, is mixed toward white. The line palette is untouched, which is
# what keeps a method the same colour everywhere. This is the step the rotatable-bond grid
# gets for free by running its colourmap through pale().
# NO WHITE AT ALL (2026-09-14): the fills are the soft() hexes exactly as the ECDF and the
# per-atom panel draw their lines, so a method is literally one colour across the three
# figures. 0.42 read washed out and 0.2 was still a second colour for the same method; the
# area effect that motivated the blend is worth living with. This stays a knob rather than
# being deleted -- one number brings the tint back if the solid blocks prove too heavy.
PCSZ_CBOX_TINT = 0.0
# The frame's weight. Raised from this family's 1.1 to the rotatable-bond grid's 2.0
# (2026-09-14) because the panel carries the grid's boxes and type at 14 pt, and 1.1 read
# thin under both. The rule between the crystal ligands and the methods takes the SAME
# weight, because it is frame and not data -- 79 poses against ~7,500 is a different
# population, not a ninth method.
#
# DOWN TO 1.0 (2026-09-15), in two steps: 2.0 read too heavy, then the target became the JSD
# summary's proportions.
#
# WEIGHT IS RELATIVE TO THE CANVAS, which is the thing that makes this number look wrong on
# its own. jsd.py already says so -- its 2.2 pt exists because that canvas is far wider and
# gets scaled down in the document. This panel is the opposite extreme: 2.77 in wide against
# the JSD summary's 13.32, so an identical pt reads 4.8x heavier here. At 2.0 this was
# 0.72 pt/in against the JSD summary's 0.165; 1.0 lands at 0.36, roughly halfway, which is
# the correction that was asked for rather than the full column-scaled one (that would have
# meant 0.46 pt, a hairline in the raster).
#
# IT NO LONGER MATCHES RB_GRID_AXIS_LW, still 2.0 -- the two figures were deliberately given
# one frame weight and now differ. If that pairing is wanted back, raise THIS, not that one.
#
# It carries two other things down with it, both intended: the minor ticks (x
# PCSZ_CBOX_MINOR_LW, so 0.50) and the reference/method rule.
PCSZ_CBOX_AXIS_LW = 1.0
# The method names, two points off the y name (2026-09-14), then 0.9x on request
# (2026-09-15). Kept as an offset from PCSZ_LABEL_FS with the scale applied after, so the
# relationship to the family size still reads at a glance instead of becoming a bare number.
PCSZ_CBOX_NAME_FS = (PCSZ_LABEL_FS - 2) * 0.9
# The axis title, a little under the family's PCSZ_LABEL_FS (2026-09-15, on request). Scoped
# to this figure rather than lowering PCSZ_LABEL_FS itself, which the strain ECDF, the
# per-atom panels and the rotatable-bond figures all read -- dropping it there would retype
# five figures nobody asked to touch. THIS PANEL HAS ONLY AN X TITLE: it is drawn sideways,
# so the methods are the y tick labels and there is no y name to match.
# The same 0.9x as the method names, so the two move together (2026-09-15). NOT EVERY NUMBER
# IN THIS PANEL SCALES: the decade tick labels take their size from the shared style -- this
# figure's tick_params sets only `width` -- so they are unaffected and would need a constant
# of their own to follow.
PCSZ_CBOX_TITLE_FS = (PCSZ_LABEL_FS - 1.5) * 0.9
# AIR ON EITHER SIDE OF THE RULE (2026-09-14). Boxes are 0.66 wide on a 1.0 pitch, so two
# neighbours are 0.34 apart; at the 2 pt frame weight this was tuned against, the rule ate
# most of that, leaving less air between the reference box and AR's than between the
# reference box and the left spine. The methods slide right by this much so the rule sits in
# a margin of its own.
#
# SIZED FOR A 2 PT RULE AND LEFT ALONE when the rule dropped to 1.0 (2026-09-15). A rule at
# half the weight it was tuned against needs much less margin, so this is now well more
# generous than it has to be -- not wrong, just loose. It is the knob to trim if the
# reference box reads too far from AR's.
PCSZ_CBOX_GAP = 0.5
# The x padding past the outermost box centre, at each end.
PCSZ_CBOX_PAD = 0.7
# ONE SCALE OVER BOTH OF THE ABOVE (2026-09-14), so "tighten the whitespace" stays a single
# number instead of two that drift apart. The box width and the 1.0 pitch are untouched: only
# air moves, so the boxes keep their size and the panel just stops carrying as much empty
# space.
#
# 0.88, UP 1.1x FROM 0.8 ON REQUEST (2026-09-15): the rule's air goes 0.40 -> 0.44 and the
# end padding 0.56 -> 0.62. It moves in the opposite direction to the 0.9x type above, which
# is the point -- smaller glyphs with more room around them.
PCSZ_CBOX_MARGIN = 0.88


def _pcsz_tint(hexv, f=PCSZ_CBOX_TINT):
    """`hexv` mixed f of the way to white, as an rgb triple."""
    r, g, b = (int(hexv[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return tuple(c + (1.0 - c) * f for c in (r, g, b))
# WHERE THE CRYSTAL LIGANDS' CLASHES COME FROM, and why not from REF_ROOT like their strain.
# REF_ROOT's run pose-scored against the 10 A crop, which deletes protein a pose could clash
# with and so under-counts; every method on this axis is scored against the whole receptor.
# This root pose-scored with scope="full", so its reference is on the same footing. Nothing
# was recomputed for it: three independent full-scope runs (this one, reproduction/
# res_test_100 and 260827 base) carry the reference clash for all 79 pockets and agree on
# every one of them, as they must -- counting clashes is deterministic, unlike strain.
PCSZ_REF_CLASH_ROOT = (f"{E}/260908_fusion_default_cv2_scratch_8gpu/samples/"
                       "samples_ep350_test79_n100")


@figure("fig-posecheck-clash-box-by-method", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_box_by_method(out):
    """Steric clashes per pose, every method, all ligand sizes pooled."""
    data_by_label, series, _ref, methods = _pcsz_data()
    labels = [label for label, *_ in series
              if len(data_by_label[label][PCSZ_POOLED]["clash"]) >= 20]
    vals = [np.asarray(data_by_label[l][PCSZ_POOLED]["clash"], dtype=float) for l in labels]
    # THE CRYSTAL LIGANDS LEAD (2026-09-14), from the whole-receptor run named above. They are
    # the benchmark every method is read against, so they are the leftmost box, and n=79 here
    # against ~7,500 a method: one crystal pose per pocket is all there is.
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    ref_full, _ = _pcsz_load_local(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    labels = [PCSZ_REF_NAME] + labels
    vals = [np.asarray(ref_full[PCSZ_POOLED]["clash"], dtype=float)] + vals
    meds = [float(np.median(v)) for v in vals]
    # PROPER log10, not log10(x+1) (2026-09-14). The +1 existed so a zero-clash pose had
    # somewhere to sit on the axis -- but the floor is 10^0 now and those poses fall below it
    # either way, so all the +1 still bought was a compressed bottom decade: it doubles 1
    # while leaving 100 untouched. Measured on the drawn panel, 10^0->10^1 spanned 130 px
    # against 169 px for 10^1->10^2, a ratio of 1.30 under tick labels that promise 1.00.
    # Zeros clamp to 0.5, below the floor, and are clipped exactly as before -- so nothing
    # visible changes except that the decades are finally even.
    # THE VIOLIN KEEPS ITS +1 and must: its axis carries a real 0 tick, so a zero-clash pose
    # has a place to land there and the transform is still doing work.
    tf = lambda v: np.log10(np.maximum(np.asarray(v, dtype=float), 0.5))
    with plt.rc_context(PCSZ_RC):
        fig, ax = plt.subplots(figsize=(PCSZ_ECDF_SIZE[0] * PCSZ_CBOX_WIDE_V,
                                        PCSZ_ECDF_SIZE[1] * PCSZ_CBOX_TALL_V),
                               layout="constrained")
        fig.patch.set_facecolor(PCSZ_BG)
        # The grid follows the VALUE axis, which rotating moved to x. A rule through every
        # category centre is a picket fence rather than a reading aid -- which is the whole
        # reason _pcsz_style takes the axis as an argument.
        _pcsz_style(ax, grid_axis="x")
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(PCSZ_CBOX_AXIS_LW)
        ax.tick_params(which="major", width=PCSZ_CBOX_AXIS_LW)
        # THE GRID'S WHISKERS (2026-09-14): 1.5 x IQR on the RAW counts, the way the
        # rotatable-bond grid gets them by handing raw values to a log axes. Handing this one
        # tf(v) instead put the fences 1.5 IQR out in LOG space, which ran AR's whisker to 33
        # where the raw fence is 14 -- the two "same format" figures were quoting different
        # statistics. Quartiles are order statistics and do not care which space they are
        # found in; the fences do. So the box is computed on the counts and only its
        # POSITIONS are transformed for drawing.
        stats = [{k: tf(s[k]) for k in ("med", "q1", "q3", "whislo", "whishi", "fliers")}
                 for s in (matplotlib.cbook.boxplot_stats(v, whis=1.5)[0] for v in vals)]
        # The reference holds position 0; the methods start one full pitch plus the gap out.
        gap = PCSZ_CBOX_GAP * PCSZ_CBOX_MARGIN
        pos = [0.0] + [i + gap for i in range(1, len(stats))]
        bp = ax.bxp(stats, positions=pos, widths=0.66, orientation="horizontal",
                    showfliers=True, patch_artist=True, manage_ticks=False, zorder=5,
                    # The house grey the whiskers already wear (INK, #514F52), not this
                    # family's pure black. Rotating the panel turned each method's outliers
                    # from a short spike above its box into a long band beside it, and at
                    # black they read as the loudest thing in the figure -- louder than the
                    # boxes whose comparison is the point. At 0.45 over white this lands
                    # near #B0B0B0 against the black version's #8C8C8C.
                    flierprops=dict(marker="d", markersize=2.4, markerfacecolor=INK,
                                    markeredgecolor="none", alpha=0.45))
        for box, label in zip(bp["boxes"], labels):
            # The method's own tint, the one it wears in the violin and the ECDF beside it.
            # soft() is keyed on the palette's "Reference ligand", not on the drawn name.
            box.set(facecolor=_pcsz_tint(REF_COLOR if label == PCSZ_REF_NAME
                                         else soft(label)),
                    edgecolor="none", linewidth=0)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                # THE GRID'S WHISKER INK (2026-09-14), not this family's. The rotatable-bond
                # grid draws whiskers and caps in the house INK -- #514F52, a warm grey --
                # while this family overrides its furniture to pure black; at the same 0.9
                # weight that black read as a harder, heavier line than the grid's. Only the
                # box's own furniture moves: the spines, ticks and names stay black.
                art.set(color=INK, linewidth=0.9)
        for m in bp["medians"]:
            m.set(color=RB_GRID_MED_COLOR, linewidth=PCSZ_CBOX_MED_LW, solid_capstyle="butt")
        # Tens of thousands of flier markers as vector paths make the PDF unplaceable.
        for fl in bp["fliers"]:
            fl.set_rasterized(True)
        # Capped: fliers past PCSZ_CBOX_HI fall OFF the panel rather than being clipped onto
        # its edge, because piling them on the edge would draw a spike at 200 that no method
        # actually has. The run log below still prints each method's true maximum.
        hi = min(max(float(v.max()) for v in vals), PCSZ_CBOX_HI)
        # THE VALUE AXIS IS X NOW (2026-09-14, rotated): decade names as powers of ten with
        # the log minor ticks between them, as the per-atom panel has them, so the two still
        # read as a pair of log axes -- evenly, now that tf is a true log10. The floor is
        # 10^0, so there is no 0 tick and the poses AT zero (1.6% of TargetDiff's, 10.9% of
        # AR's) sit below it, their whisker running off the LEFT edge. That those poses are
        # invisible is worth saying in the caption: a tenth of AR's poses clash with nothing
        # at all, which is the best thing about it and the panel cannot show it.
        decades = [t for t in PCSZ_CLASH_TICKS if t and t <= hi * 1.6]
        ax.set_xticks(tf(decades))
        ax.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in decades])
        ax.set_xticks(tf([k * t for t in decades for k in range(2, 10) if k * t <= hi]),
                      minor=True)
        ax.tick_params(axis="x", which="minor", direction="out", length=2.0,
                       width=PCSZ_CBOX_AXIS_LW * PCSZ_CBOX_MINOR_LW, colors=PCSZ_AXIS)
        ax.set_xlim(tf(1), tf(hi) + 0.06)
        # INVERTED, so position 0 -- the crystal ligands -- is the TOP row. A stack of rows is
        # read downward and matplotlib puts 0 at the bottom, which would have buried the
        # benchmark under the eight methods it exists to be compared against.
        ax.set_ylim(pos[-1] + PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN,
                    pos[0] - PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN)
        # The crystal ligands are the benchmark the methods are read against, so a rule
        # separates them rather than leaving them to look like a ninth method. It takes the
        # frame's colour AND weight exactly, with no alpha: at 0.45 the same black rendered as
        # a mid grey that read as a third kind of line -- neither the black spines it belongs
        # with nor the grey whiskers inside the panel.
        ax.axhline((pos[0] + pos[1]) / 2.0, color=PCSZ_AXIS, lw=PCSZ_CBOX_AXIS_LW, zorder=4)
        ax.set_yticks(pos)
        # TILTED, ANCHORED WHERE THE NAME ENDS (2026-09-14). rotation_mode="anchor" pins the
        # label's right end to its tick and rotates about that point, so the tick sits
        # exactly where the name stops and each name points at its own row.
        # Measured, this puts a label's bounding-box CENTRE a median 40 px below its tick
        # (54 px for the longest) -- but that is the geometry of an anchored rotation, not a
        # misalignment, because the eye follows a tilted name to its END. Centring the
        # rotated box on the tick instead scores 0 px by that measure and reads WORSE: no
        # part of the name then touches its tick and it floats between two rows.
        ax.set_yticklabels([l if l == PCSZ_REF_NAME else display(l) for l in labels],
                           fontsize=PCSZ_CBOX_NAME_FS, rotation=PCSZ_CBOX_NAME_ROT,
                           ha="right", va="center", rotation_mode="anchor")
        ax.set_xlabel("Steric clashes", fontsize=PCSZ_CBOX_TITLE_FS, labelpad=PCSZ_YPAD)
        save(fig, out, "clash_box_by_method")
    write_csv(out, "clash_box_by_method",
              ["method", "n_clash", "clash_mean", "clash_median", "q25", "q75", "zero_frac"],
              [[l, len(v), round(float(v.mean()), 2), round(float(np.median(v)), 1),
                round(float(np.percentile(v, 25)), 1), round(float(np.percentile(v, 75)), 1),
                round(float((v == 0).mean()), 4)]
               for l, v in zip(labels, vals)])
    for l, v, m in zip(labels, vals, meds):
        print(f"    {l:16s} n={len(v):5d}  median {m:4.1f}  mean {v.mean():5.2f}  "
              f"zero-clash {100 * (v == 0).mean():4.1f}%  max {v.max():.0f}")
    print(f"  wrote {out}")


# ── fig-posecheck-clash-per-atom-all-methods ───────────────────────────────────────────
# The strain per-atom all-methods panel's twin, for clashes (2026-09-14).
#
# WHY THE EXISTING clash_per_atom_* FIGURES COULD NOT SIMPLY GAIN FIVE MORE LINES. They draw
# out of the ARMS run trees, and those trees cannot answer this question for eight methods:
#   1. THE FIVE BASELINES HAVE NO PER-MOLECULE POSECHECK THERE AT ALL -- 0 of ~37,000 samples
#      under baselines_pose/* carry posecheck.clashes. Their per-molecule counts exist only in
#      the svr12 exports, which is where the violin, the ECDF and the box already read them.
#   2. THE RECEPTOR SCOPE DIFFERS. The ARMS roots pose-scored against the 10 A crop
#      (pose_receptor_scope "crop", or absent, which metrics.py reads as crop); every other
#      clash figure in this family scores against the whole receptor. On the SAME arm and the
#      same 79 pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole
#      receptor reads 6.0 / 7.48 / 97 -- the crop deletes protein the pose could clash with,
#      so it under-counts, and it truncates the tail hardest.
# So this figure takes its five baselines from the exports, its three local arms from
# PCSZ_LOCAL (scope "full") and its reference from PCSZ_REF_CLASH_ROOT, and never from ARMS.
# It keeps the strain twin's windowing (+-RB_WIN atoms, drawn where >= MIN_N molecules pool)
# so the two panels are read the same way, and its house is the ECDF's, like every panel here.
PCSZ_CATOM_STATS = {"median": lambda v: float(np.median(v)),
                    "mean": lambda v: float(np.mean(v))}
# Dash per method, keyed as PCSZ_LINE_LW is: both tables are (.., lw, ls) last-two.
PCSZ_LINE_LS = {**{ALIASES.get(lab, lab): ls for _, lab, _, ls in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): ls for lab, _, _, ls in PCSZ_LOCAL}}


def _pcsz_atom_rows(root, keep, reference=False):
    """[{n, c}] per molecule for one whole-receptor run tree, over the pockets in `keep`."""
    rows = []
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            c = (m.get("posecheck") or {}).get("clashes")
            if c is not None:
                rows.append({"n": m["n_atoms"], "c": float(c)})
    return rows


def _pcsz_atom_keep():
    """The pockets every arm on this axis has: the baselines' 79, narrowed by ours."""
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    return keep


@figure("fig-posecheck-clash-per-atom-all-methods", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_per_atom_all_methods(out):
    """Steric clashes against ligand size, all eight methods and the crystal ligands."""
    keep = _pcsz_atom_keep()
    series = {}
    for key, label, *_ in PCSZ_BASELINES:
        # The exports' rows are already {n, s, c}; this reads the same file the violin does.
        series[label], _ = _rb_atom_baseline_rows(key, keep)
    for label, root, *_ in PCSZ_LOCAL:
        series[label] = _pcsz_atom_rows(root, keep)
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)

    labels = _pcsz_ordered([label for _, label, *_ in PCSZ_BASELINES]
                           + [lab for lab, *_ in PCSZ_LOCAL])
    per = {lab: by_size(series[lab], "c") for lab in labels}
    ref_per = by_size(refrows, "c")
    xs = list(range(RB_X_LO, RB_X_HI + 1))

    for stat, f in PCSZ_CATOM_STATS.items():
        with plt.rc_context(PCSZ_RC):
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR,
                    lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=SOLID, zorder=4,
                    solid_capstyle="round")
            for lab in labels:
                key = ALIASES.get(lab, lab)
                ax.plot(xs, _rb_curve(per[lab], xs, f), color=soft(lab),
                        lw=PCSZ_LINE_LW[key] * PCSZ_LW_SCALE, ls=PCSZ_LINE_LS[key],
                        zorder=5, solid_capstyle="round")
            ax.set_ylim(bottom=0)
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
            ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
            ax.set_ylabel(f"Clashes {stat}\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_per_atom_all_methods_{stat}")

    rows = []
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        cur = {s: (reference_curve(p, xs, f) if lab == PCSZ_REF_NAME
                   else _rb_curve(p, xs, f)) for s, f in PCSZ_CATOM_STATS.items()}
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ()))]
                        + ["" if cur[s][i] is None else round(cur[s][i], 2)
                           for s in ("median", "mean")])
    write_csv(out, "clash_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "clash_median", "clash_mean"], rows)

    print(f"  {len(keep)} pockets · {len(labels)} methods + reference · whole-receptor "
          f"scope · x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, drawn "
          f"where it pools ≥{MIN_N} molecules")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>6d}" for a in heads) + f" {'drawn':>9s}   (clash median at n)")
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        med = (reference_curve(p, xs, PCSZ_CATOM_STATS["median"]) if lab == PCSZ_REF_NAME
               else _rb_curve(p, xs, PCSZ_CATOM_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:6.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>6s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        span = f"{drawn[0]}-{drawn[-1]}" if drawn else "—"
        n_mol = sum(len(v) for v in p.values())
        mean_at = np.mean([n for n, v in p.items() for _ in v]) if n_mol else float("nan")
        print(f"  {lab:16s} {n_mol:6d} {mean_at:6.1f} " + " ".join(cells) + f" {span:>9s}")
    print(f"  wrote {out}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-legend — the per-atom all-methods strain key, as its own image
# ════════════════════════════════════════════════════════════════════════════════
# strain_per_atom_all_methods_{mean,median} dropped their key on request (2026-09-13); this
# draws that key alone, to be placed beside or under the panels. The layout is the Vina
# per-atom v3 key's (dock-per-atom-v3-*-all): the reference on a centred line of its own
# over the eight methods in 2 rows x 4 columns reading across, one frame round both. No n:
# the reference is the section's own crystal ligands, and the panel itself says nothing
# about counts.
#
# THE SWATCHES ARE THE PANEL'S. The dash comes from the shared METHOD_DASH table through
# dash(), the WEIGHT from the ECDF family (PCSZ_LINE_LW x PCSZ_LW_SCALE) and the colour
# through soft(), in RB_ORDER. PCSZ_LINE_LW is read at call time: it lives in a part
# assembled after this one.
#
# THIS KEY USED TO LIE. It read the rotatable-bond family's own dash table, which had drifted
# a whole method out of step with the ECDF's -- so the legend told a reader that Pocket2Mol
# was the dotted line when the panel drew DiffSBDD that way. One table now feeds both.
PLEG_NCOL = 4
# SWATCH LENGTH, in font sizes. The shared legend() fixes 1.9, and at the panel's weights that
# is shorter than one period of DecompDiff's (9, 3) dash -- its swatch read as a solid line,
# i.e. as one of the local arms -- and the reference showed a dash and a half. 3.6 carries at
# least two periods of every pattern in RB_BASE_DASH.
PLEG_HANDLE_LEN = 3.6


def _pleg_method_handles():
    out = []
    for lab in RB_ORDER:
        out.append(Line2D([], [], color=soft(lab), ls=dash(lab),
                          lw=PCSZ_LINE_LW[ALIASES.get(lab, lab)] * PCSZ_LW_SCALE,
                          solid_capstyle="round", label=display(lab)))
    return out


def _pleg_legend(fig, handles, **kw):
    """legend()'s look -- white face, the axis-pen frame, INK text -- with a swatch long
    enough to show a dash. Its frame is removed again by _vpa_one_frame."""
    leg = fig.legend(handles=handles, frameon=True, fontsize=11.5, handlelength=PLEG_HANDLE_LEN,
                     handletextpad=0.6, labelspacing=0.3, borderpad=0.4, borderaxespad=0.39,
                     facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(AXIS_LW)
    # above the shared frame _vpa_one_frame draws at zorder 6, as legend() does -- without it
    # the white frame is painted over every entry and the image is an empty box
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


@figure("fig-posecheck-strain-legend", folder="fig-posecheck/strain-energy")
def draw_posecheck_strain_legend(out):
    """The key the per-atom all-methods strain panels dropped: reference on top, methods 2x4."""
    use_style()
    # THE REFERENCE TAKES ITS LINE FROM THE SAME TABLE (2026-09-14). It used to hardcode
    # DASH, which made this key show a dashed crystal-ligand swatch while both panels it
    # describes draw that series SOLID -- the reference went solid on 2026-09-14 and the key
    # was never told. It is built here rather than in _pleg_method_handles because it sits on
    # its own centred line, which is exactly how it escaped the earlier repointing.
    ref = [Line2D([], [], color=REF_COLOR, ls=dash(REF_LABEL),
                  lw=PCSZ_REF_LW * PCSZ_LW_SCALE,
                  dash_capstyle="round", label=REF_LABEL)]
    # PCSZ_RC for the panel's 170 dpi and its crop to the ink, so the key and the panels it
    # sits beside are rasterised the same way.
    with plt.rc_context(PCSZ_RC):
        fig = plt.figure(figsize=(10.0, 1.6))
        fig.patch.set_facecolor("white")
        # Two legends in one frame, exactly as the v3 key: a legend column is as wide as its
        # widest entry, so the reference cannot be a cell of the grid without unevening it.
        legs = [_pleg_legend(fig, _vpa_row_major(_pleg_method_handles(), PLEG_NCOL),
                             loc="lower center", ncol=PLEG_NCOL, bbox_to_anchor=(0.5, 0.05),
                             columnspacing=1.5)]
        fig.canvas.draw()
        h = legs[0].get_window_extent(fig.canvas.get_renderer()).height / \
            (fig.get_size_inches()[1] * fig.dpi)
        legs.append(_pleg_legend(fig, ref, loc="lower center", ncol=1,
                                 bbox_to_anchor=(0.5, 0.05 + h + VPA_KEY_ROW_GAP)))
        _vpa_one_frame(fig, legs)
        print(f"  key: {REF_LABEL} over {len(RB_ORDER)} methods in {PLEG_NCOL} columns · "
              f"dash from METHOD_DASH, weight from PCSZ_LINE_LW, colour soft()")
        save(fig, out, "strain_per_atom_all_methods_legend")


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
                      clip=None, stem, legend_loc="upper left", rows=None, ref=None):
    """One statistic, one panel, one file. Which statistic you are looking at is carried by
    the filename and by the y-axis name, exactly as the 3-line figures carry it.

    `rows`/`ref` override where the molecules come from. Strain passes neither and reads
    pose_data() as it always has; CLASHES pass the whole-receptor trees, because the
    pose_data() rows are crop-scored and a crop under-counts clashes."""
    _, p79_rows, refrows = pose_data()
    p79_rows = p79_rows if rows is None else rows
    refrows = refrows if ref is None else ref
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


@figure("fig-posecheck-clash-per-atom", folder="fig-posecheck/clash",
        needs=("frozenenc_probes/posecheck_full/",))
def draw_posecheck_clash_per_atom(out):
    """PoseCheck steric clashes against heavy-atom count — mean and median, core only.

    WHOLE-RECEPTOR SCOPE (2026-09-14). This used to read pose_data(), i.e. the ARMS run
    trees, and those pose-scored against the 10 A crop: on the same arm over the same 79
    pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole receptor
    reads 6.0 / 7.48 / 97, and VoxBind 4.0 / 5.23 / 42 against 5.0 / 6.23 / 78. A crop
    deletes protein the pose could clash with, so it under-counts and truncates the tail
    hardest. Every other clash figure in this family scores against the whole receptor, so
    this one reads the same posecheck_full trees the violin and the box do.

    No `all`: see core_only(). The five published baselines carry NO per-molecule PoseCheck
    in the ARMS trees at all -- 0 of ~37,000 samples -- so `all` here was only ever core plus
    TargetDiff, which is neither the two-arm comparison nor the whole field. The whole field
    (seven published baselines, ours, and the crystal ligands) is
    clash_per_atom_all_methods, which reads the svr12 exports for the five."""
    use_style()
    keep = _pcsz_atom_keep()
    # The run roots are keyed by the ARMS key itself: posecheck_full/<key>.
    rows = {os.path.basename(root): _pcsz_atom_rows(root, keep)
            for _, root, *_ in PCSZ_LOCAL}
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    arms = [a for a in ARMS if a[1] in CORE and a[1] in rows]
    per = {key: by_size(rows[key], "c") for _, key, _ in arms}
    # The x range is the CLASH range now, not the strain range _pc_ranges() derives. That
    # sharing existed because both panels read one molecule set; this figure no longer does,
    # so registering the two count for count would be a coincidence, not a property.
    xs = x_range(per, arms)
    for stat, f in PC_STATS:
        _pc_per_atom_stat(out, "c", xs, arms, "core", "Clashes", "", stat, f,
                          log=False, stem=f"clash_per_atom_{stat}",
                          legend_loc="upper left", rows=rows, ref=refrows)
    # Its own CSV: _pc_curve_csv writes strain AND clashes out of pose_data(), whose clash
    # columns are the crop-scored ones this figure just stopped drawing.
    ref_per = by_size(refrows, "c")
    csv_rows = []
    for lab, key, _ in arms:
        cur = {s: model_curve(per[key], xs, fn) for s, fn in PC_STATS}
        csv_rows += [[lab, a, len(per[key].get(a, ()))]
                     + ["" if cur[s][i] is None else round(cur[s][i], 3)
                        for s, _ in PC_STATS] for i, a in enumerate(xs)]
    cur = {s: reference_curve(ref_per, xs, fn) for s, fn in PC_STATS}
    csv_rows += [[REF_LABEL, a, ""] + ["" if cur[s][i] is None else round(cur[s][i], 3)
                                       for s, _ in PC_STATS] for i, a in enumerate(xs)]
    write_csv(out, "clash_per_atom",
              ["arm", "heavy_atoms", "n", "clash_mean", "clash_median"], csv_rows)
    print(f"  {len(keep)} pockets · whole-receptor scope · "
          f"{', '.join(l for l, *_ in arms)} + reference · x = {xs[0]}-{xs[-1]}")




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
# The per-method dashes that used to live here are now METHOD_DASH in 00_core.py, read
# through dash(). This table had drifted a whole method out of step with the ECDF family's,
# so the same baseline was drawn with two different patterns depending on the figure.
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
# What the crystal ligands are CALLED in the eight-method keys (2026-09-14), matching the
# strain ECDF. The shared REF_LABEL still names them in the data and in every other family.
RB_REF_NAME = "Reference"
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
    """Weight from provenance, DASH FROM THE SHARED TABLE.

    The weight still keys on RB_LOCAL_LABELS: the three arms we run locally are drawn heavier
    in this family. The dash no longer does. It used to return solid for every local arm,
    which handed TargetDiff -- a published baseline we happen to run ourselves -- the solid
    line that means OURS in every other figure. dash() knows the difference."""
    lw = MODEL_LW if label in RB_LOCAL_LABELS else RB_BASE_LW
    return lw, dash(label)

def _rb_panel_order(labels):
    """Same order as `_rb_order`, with the crystal ligands FIRST -- they are the table's
    first row and the thing every other panel is read against."""
    return [REF_LABEL] + _rb_order(labels)


def _rb_handles(labels, solid=False):
    """`solid` for the box figure: nothing in it is a dashed line, so a dashed swatch in the
    key advertises an encoding the panel does not use. The crystal ligands are SOLID in both
    (2026-09-14), named "Reference" and drawn through display(), so this family's keys read
    like the strain ECDF's and the per-atom panel's -- solid marks the series a reader returns
    to, every baseline carries a dash, and VoxBind shows its sigma as a subscript."""
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=SOLID, label=RB_REF_NAME)]
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        h.append(Line2D([], [], color=soft(lab), lw=MODEL_LW if solid else lw,
                        ls="-" if solid else ls, label=display(lab)))
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
    # Solid, like the key's swatch and like the ECDF's crystal ligands (2026-09-14). The
    # three-arm core figures above keep the dash -- there the reference is one of three lines,
    # not one of nine, and nothing else in those panels competes for solid.
    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=SOLID, zorder=4, solid_capstyle="round")
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
                ls=SOLID, zorder=6, solid_capstyle="round")
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
        # Solid here too, matching the ECDF it pairs with (2026-09-14); AR is already dashed
        # on this axis through RB_BASE_DASH.
        ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR,
                lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=SOLID, zorder=4, solid_capstyle="round")
        dropped = []
        for lab in labels:
            y = _rb_curve(per[lab], xs, f)
            if clip:
                over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
                if over:
                    dropped.append((lab, over))
            _, ls = _rb_style_of(lab)
            # The ECDF's weights, not this family's (2026-09-13): the two panels are read as a
            # pair, and our arms were 3.06 here against 3.9-4.4 there. KeyError rather than a
            # default -- a method missing from that table is a rename to fix, not a thin line.
            lw = PCSZ_LINE_LW[ALIASES.get(lab, lab)]
            ax.plot(xs, y, color=soft(lab), lw=lw * PCSZ_LW_SCALE, ls=ls, zorder=5,
                    solid_capstyle="round")
        ax.set_yscale("log")
        if clip:
            ax.set_ylim(top=clip)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
        ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
        # Two lines again (2026-09-14): it was pulled onto one on 2026-09-13, and the unit now
        # goes back under the name, matching the ECDF's y name beside it.
        ax.set_ylabel(f"Strain {stat}\n(kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS,
                      labelpad=PCSZ_YPAD)
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
# THE NORM IS A FIXED 0-1000 kcal/mol (was 0-800; see RB_GRID_NORM), ticked every 200. Over
# the eight methods' 64 boxes the median is 136 and the 95th percentile 694; only FuncBind at
# six bonds (5,638) lies above 1000, and it saturates onto the top colour with the colorbar's
# arrow saying so (the run log names every saturated box). The price of
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
# THE PAPER'S OWN SCALE (2026-09-14). VoxBind Fig. 12/13 colours each box by its median on
# matplotlib's coolwarm, and this used to be a house-built lookalike assembled out of the
# SHARED PALETTE: CoDE's soft blue at 0, through both pale tints, to VoxBind's sand at the
# top. That made the magnitude channel collide with the identity one -- a periwinkle box
# meant "low median" here and "CoDE" in every other figure, a sand one "high median" here
# and "VoxBind" there -- and a reader carrying the palette between figures reads it wrong.
# coolwarm keeps what the house version was reaching for, a light middle so the dark median
# bars and the fliers stay legible, while being far more saturated at both ends than any
# pastel in SOFT, so it cannot be mistaken for a method.
#
# IT IS DIVERGING AND THE QUANTITY IS NOT: median strain runs 0-800 with no meaningful
# midpoint at 400, so the white band is an artefact of the map rather than a feature of the
# data. That is the paper's choice and matching it is the point here, but a sequential map
# would be the more honest encoding if we ever stop matching.
# THE RAMP IS THE HOUSE ONE: HEAT_CMAP, single-hue light-to-dark, decided in 00_core
# because fig-posebusters-valid-heatmap keys its cells with the same object. It used to
# be built here and read across from there; it moved up on request (2026-09-15) so
# neither figure owns the other's colours.
# TOP AT 1000, NOT 800 (2026-09-15, on request, with the two-stop ramp #FAFDFB -> #3C52DD):
# "1000 and over is the end colour". FuncBind at seven bonds (982) now takes its own shade just
# under the tail instead of saturating; FuncBind at six (5,638) is the one box above, and the
# bar's arrow still says so. Ticks stay every 200, so the bar reads 0-1000 in six numbers.
RB_GRID_NORM, RB_GRID_CTICK = (0.0, 1000.0), 200.0
RB_GRID_TALL = 0.9975 * 1.1  # per row, as a share of PANEL_H
# 0.8x the width the three-arm box figures use (2026-09-13). The nine panels keep their type
# and line weights, so pulling the canvas in is what makes the block read fuller; the boxes
# narrow with the axes, the colourbar keeps its share of the figure.
RB_GRID_WIDE = 0.8 * 1.2 * 1.15
# The outer names are a step above the house 15.5 pt, and the y name sits further off its
# tick labels -- on a 3x3 block the house sizes read as small.
RB_GRID_LABEL_FS, RB_GRID_YPAD = 27, 18
# Everything else in the block moves with the names (2026-09-14), so it reads like the strain
# ECDF, which spends 16 pt on a canvas half this wide. Panel names, the tick numbers under and
# beside every panel, and the colourbar's own ticks -- left at the house 14/14/12 they turned
# into fine print as soon as the axis names went to 22.
RB_GRID_TITLE_FS, RB_GRID_TICK_FS = 22, 18
RB_GRID_CBAR_TICK_FS = RB_GRID_TICK_FS   # one size for every number in the figure
# SET AGAINST THE STRAIN ECDF (2026-09-14, user's call): that figure spends 16.1 pt of type
# and a 1.1 pt spine on a 6.95 in canvas; this one is 14.45 in, so matching it at equal placed
# width would mean 33 pt and 2.5, and matching it per PANEL -- a grid panel is 2.8 x 2.6 in
# against the ECDF axes' 6.0 x 2.8 -- would mean 16 pt and 1.1. These are the geometric mean of
# the two, for a block placed wider than the ECDF but not twice as wide.
# THE PANEL TICK NUMBERS ARE SMALLER THAN THE BAR'S (2026-09-14) even though both were set to
# 20: measured off the PNG the two render at the same 48 px, but the bar's are turned a quarter
# turn and a vertical string of digits reads smaller than a horizontal one. They came down to
# 16 to match what the bar LOOKS like, the bar's own numbers followed them down, and both then
# settled at 18 -- one number size for the whole figure, a step under the panel names.
RB_GRID_MED_COLOR = "#323232"   # the median bar, a touch lighter than INK
# The x name's gap above it, DOUBLED (2026-09-13): measured off the drawn PNG it was 27 px at
# 220 dpi, i.e. 8.8 pt from the tick labels; a labelpad of 15 measures back as 17.7 pt, which
# is that doubled (the pad is spent from the axes bbox, so it is ~3 pt more than the gap it
# buys). In points, like the y pad, which is why the name moved off fig.supxlabel and onto an
# axes -- see the call.
RB_GRID_XPAD = 15
# The colourbar's name is the block's THIRD outer name, so it carries the same size as the
# two axis names rather than a step below them (2026-09-13).
RB_GRID_CBAR_FS = RB_GRID_LABEL_FS
# Spines and MAJOR tick marks 1.2x the house weight: nine small panels read as washed out at
# 1.35, and 1.6x was too heavy. Minor ticks (the log decades' 2-9) keep their own weight.
RB_GRID_AXIS_LW = 2.0
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
    cmap = HEAT_CMAP
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(min(all_meds) < norm.vmin, max(all_meds) > norm.vmax)]

    nrow = -(-len(panels) // RB_GRID_COLS)
    fig, axes = plt.subplots(nrow, RB_GRID_COLS, sharex=True, sharey=True,
                             figsize=(FIG_W * RB_BOX_WIDE_1 * RB_GRID_WIDE,
                                      PANEL_H * RB_GRID_TALL * nrow),
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
                # No outline (2026-09-14): the fill IS the median through the colourmap, and a
                # dark edge on nine panels of eight boxes read as a grid of frames.
                box.set(facecolor=cmap(norm(float(np.median(v)))), edgecolor="none",
                        linewidth=0)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=INK, linewidth=0.9)
            for med in bp["medians"]:
                # Up from 1.3, settled at 2.0 (2026-09-14; 2.6 was heavy): with the box outline
                # gone this is the only dark mark on the fill, and it is the number the colour
                # encodes.
                med.set(color=RB_GRID_MED_COLOR, linewidth=2.0, solid_capstyle="butt")
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
        ax.tick_params(labelsize=RB_GRID_TICK_FS)   # after furniture(), which sets the house 14
        # EVERY decade keeps its label. At 16 pt the default log locator decided 10^0..10^5 no
        # longer fit and thinned them to every second one, dropping 10^5 -- the axis top -- with
        # them. numticks high enough that the thinning never triggers on this range.
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=99))
        # An unlabelled tick where the axes meet, matching the y axis's 10^0 in that corner:
        # the first bond count sits 0.6 in, so without it the x axis looks cut short.
        counts = list(range(0, RB_GRID_X_MAX + 1))
        ax.set_xticks([xlo] + counts, [""] + [str(x) for x in counts])
        # "Reference", not the shared REF_LABEL, and the gap to the panel doubled (2026-09-14).
        ax.set_title("Reference" if lab == REF_LABEL else display(lab),
                     fontsize=RB_GRID_TITLE_FS, color=INK, pad=10)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    # Both outer names now sit on an axes rather than on the figure: only an axes label takes a
    # labelpad, and fig.supxlabel has no way to spend points on its gap. The bottom-middle axes
    # is the block's centre with three full rows, so the name also stops being centred on the
    # figure -- which included the colourbar -- and centres on the PANELS instead.
    axes[nrow - 1, RB_GRID_COLS // 2].set_xlabel(RB_X_LABEL, fontsize=RB_GRID_LABEL_FS,
                                                 color=INK, labelpad=RB_GRID_XPAD)
    # The y name goes on the MIDDLE row's left axes rather than fig.supylabel, because only an
    # axes label takes a labelpad -- supylabel sits flush against the tick labels. With three
    # rows the middle axes' centre is the block's centre.
    # TWO LINES (2026-09-14), broken at the comma the sentence already has: the unit stays
    # with the name it belongs to and the stage qualifier gets its own line, which is how the
    # ECDF and the per-atom panel set their y names.
    # THE PAD IS UNCHANGED ON PURPOSE. labelpad is measured to the NEAREST edge of the label,
    # so the 63 px it buys between the tick numbers and the name survives the second line;
    # what grows is the label block itself (~81 px wide to ~170), and under layout=
    # "constrained" that width comes out of the panels rather than out of the canvas.
    axes[nrow // 2, 0].set_ylabel(f"UFF strain energy (kcal mol⁻¹),\n{qualifier}",
                                  fontsize=RB_GRID_LABEL_FS, color=INK,
                                  labelpad=RB_GRID_YPAD)
    # The shared bar (00_core): ramp, shape, pad and the quarter-turned numbers come from
    # there, the point sizes and the tick step are this figure's. `extend` is computed above
    # off the data, so the arrow appears exactly when a median saturates.
    heat_colorbar(fig, axes, norm, "Median strain energy (kcal mol⁻¹)",
                  label_fs=RB_GRID_CBAR_FS, tick_fs=RB_GRID_CBAR_TICK_FS, extend=extend,
                  ticks=np.arange(RB_GRID_NORM[0], RB_GRID_NORM[1] + RB_GRID_CTICK,
                                  RB_GRID_CTICK))
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
