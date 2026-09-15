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
    # CoE (LaTeX: \oursC) is our coords-only ablation of CoDE, so the blue ladder is where
    # it belongs by family -- and is exactly where it cannot go. The Vina per-atom family
    # draws CoDE through soft() as periwinkle #8291E8 and CoE is a FOCUS arm beside it
    # there, both solid at model weight: two steps of one blue would read as one curve.
    # Deep teal is the nearest family no method holds at full saturation -- AR's #17A2B8 is
    # lighter and only ever drawn as thin dashed context, so weight and lightness separate
    # them wherever both appear.
    "CoE":              "#0F766E",   # deep teal — ours, coords-only
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
    # The coords-only ablation, which every table spells differently: "Ours C" is the
    # similarity CSV's key (build_reference_similarity.py's PLAIN map), "Ours · coords" the
    # drug-design table's row name, "VoxBind+CoE" the results-bundle folder.
    "ours_c": "CoE", "Ours C": "CoE", "Ours · coords": "CoE", "Ours &middot; coords": "CoE",
    "VoxBind+CoE": "CoE", "VoxBind + CoE": "CoE", "\\oursC": "CoE", "COE": "CoE",
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
