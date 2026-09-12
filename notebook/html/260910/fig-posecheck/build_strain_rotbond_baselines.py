#!/usr/bin/env python3
"""build_strain_rotbond_baselines.py — strain against rotatable bonds, ALL methods.

    strain_rotbond_all_methods_{median,mean}.*  lines, every method, per exact bond count
    strain_box_per_rotbond_all_methods.*        strain, every method on one axis
    rotbond_distribution_all_methods.*          where each method puts its ligands, 3x3
    strain_rotbond_all_methods.{json,csv}       the numbers behind both

`build_strain_per_rotbond.py` next door draws the same metric for the three arms we run
locally. This adds the published baselines, whose per-molecule strain has been in
`posecheck_<Method>.json` all along but which could not be put on this axis before,
because those exports carry no SMILES and rotatable bonds cannot be counted without one.
The molecules themselves are now in the results bundle, so the count can be recovered.

WHERE THE ROTATABLE BONDS COME FROM, AND WHY THE JOIN IS SAFE. The baselines' molecules
live in `results/task2-drugdesign/<M>/samples/meta/` as the TargetDiff meta format: one
list per test pocket of {mol, smiles, ligand_filename, pred_pos}. `export_posecheck_json.py`
built `posecheck_<Method>.json` from THE SAME meta, walking each pocket's entries in order
and skipping the ones whose `mol` is None -- so the two are the same molecules in the same
order, and `load_meta` here reproduces its loader exactly (base + `_part2` concatenated per
pocket; `_gap.pt` is present in the bundle but that loader does not use it, so neither does
this one).

That is an argument, not evidence, so the join is CHECKED rather than trusted: for every
pocket the heavy-atom sequence recomputed from the meta must equal the `n` sequence in the
export, position by position, and a mismatch aborts. It matches 100/100 pockets for all
five methods, on exactly 7,655 / 7,772 / 7,720 / 6,427 / 7,895 molecules over the 79
pockets --
the totals `../fig-posecheck/README.md` already records from
`baselines/_eval/summary_density79.json`.

FUNCBIND IS IN, since 2026-09-10. Its meta reached the bundle that day and joins cleanly:
100/100 pockets, 9,992 molecules, the export's own total. Before that only its strain
export was here, and its shard SDFs under `funcbind/artifacts/reproduction/crossdocked/
paper_run` were tried as a substitute and FAILED this same check -- 1 pocket of 100 matched,
pocket 0 held 99 molecules against the export's 100, atom counts disagreeing from the first
record. Those shards are a different sampling run from the one PoseCheck scored, and the
pocket mapping is not the problem (`pocket_ligand_filename` agrees with the shard target
dirs); do not reach for them again if the meta ever goes missing.

MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN. The baselines were
scored against the whole `*_rec.pdb` receptor; the local arms here are read from the
pocket10 crop in their target `metrics.json`, the same source the sibling figure uses.
Strain is a property of the ligand's own conformer -- UFF relaxation under a position
constraint -- so receptor scope cannot enter it, and measuring the same molecules both ways
confirms it does not: median |relative difference| 0.6-1.0 %, which is the run-to-run noise
`num_confs=50` already carries, and pooled medians move under 1 % (60.1 vs 62.6, 80.5 vs
82.9, 346.2 vs 350.4). Do NOT extend this to clashes or interactions: those are
receptor-dependent and a crop cannot see an atom it does not contain.

THREE FIGURES, AND THE BOXES ARE THE ONE TO READ. All eight methods are dodged inside each
bond count on ONE axis: the comparison that matters is between methods AT a bond count, and
small multiples -- one panel per method, the paper's Fig. 12/13 form -- were built first and
dropped because a grid makes exactly that comparison hardest. The share of each method's
ligands at each count is now its OWN figure (3x3, log y) rather than a strip under the
boxes: the two answer different questions, are read at different times, and nine share
panels do not fit under a box panel at any useful size. The median/mean LINES keep the
summary view, where colour plus a dash pattern per baseline is the identity channel
`build_posecheck_all_by_atom_range.py` already uses for nine series.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posecheck/build_strain_rotbond_baselines.py
    ... --stat median        # only the median line figure
    ... --kind box           # only the binned boxes
"""
import argparse
import collections
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

from rdkit import Chem, RDLogger                                     # noqa: E402
from rdkit.Chem import rdMolDescriptors                              # noqa: E402

RDLogger.DisableLog("rdApp.*")

pc.REF_WIN, pc.MIN_REF = 1, 12          # as in the sibling: this axis is a third as long

# Walk up to the repository root by looking for the bundle itself rather than counting
# directory levels -- this file sits four deep and a miscount silently resolves to
# notebook/results, which exists.
BUNDLE = next((os.path.join(d, "results/task2-drugdesign")
               for d in (HERE, *(os.path.dirname(HERE[:i]) for i in
                                 range(len(HERE), 0, -1) if HERE[i - 1] == os.sep))
               if os.path.isdir(os.path.join(d, "results/task2-drugdesign"))
               and os.path.isdir(os.path.join(d, "voxbind"))), None)
if BUNDLE is None:
    raise SystemExit("results/task2-drugdesign not found above " + HERE)

# label, bundle folder, meta stem. DecompDiff's meta is the reference-prior run, which is
# the one export_posecheck_json.py scored.
BASELINES = [
    ("AR",         "AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
    # Arrived in the bundle on 2026-09-10 and joins cleanly: 100/100 pockets, 9,992
    # molecules. Until then only its strain export was here and the shard SDFs under
    # funcbind/artifacts were a different sampling run -- see the docstring.
    ("FuncBind",   "FuncBind",   "FuncBind"),
]
# Thin and dashed: with eight models on one axis hue alone is not enough, and the three
# local arms are the subject while these are context. Same channel split as
# build_posecheck_all_by_atom_range.py.
BASE_LW = 1.6
BASE_DASH = {"AR": (0, (5, 2)), "Pocket2Mol": (0, (1, 1.6)),
             "DiffSBDD": (0, (6, 2, 1, 2)), "DecompDiff": (0, (9, 3)),
             "FuncBind": (0, (3, 1.4, 1, 1.4))}

X_LABEL = "Number of rotatable bonds in ligand"
XTICK = 1
STRAIN_FLOOR, WHIS = 1e-2, (5, 95)
BOX_GAP = 0.30               # of a bond count's width, left clear between neighbouring groups
# The box panel is wider and taller than the house figure -- eight boxes have to fit inside
# each of thirteen counts -- and the share panels are a 3x3 block of small ones.
BOX_WIDE_1, BOX_TALL, DIST_COLS, DIST_WIDE, DIST_TALL = 1.72, 1.22, 3, 1.34, 0.66
BOX_YLIM = (1e0, 1e5)
# THE AXIS STOPS AT 12 ROTATABLE BONDS, because each INDIVIDUAL count past it is thin and
# stretching the axis to the last molecule anyone made (25, one VoxBind ligand) spent two
# thirds of the width on a handful of boxes.
#
# THE CUMULATIVE TAIL IS NOT NEGLIGIBLE, THOUGH, and the figure must not imply it is: 5.0 %
# of VoxBind's ligands, 3.1 % of DecompDiff's and 2.8 % of DiffSBDD's have more than 12
# rotatable bonds (Pocket2Mol is the outlier at 0.0 %). So every share panel PRINTS its own
# excluded percentage rather than letting the cap pass silently. Raise X_MAX to see them.
X_MAX = 12
# Above this many counts the axis is split over two rows so the boxes stay wide enough to
# read; at X_MAX = 12 it is one row. Raise X_MAX and the split comes back on its own.
SPLIT_ABOVE = 14
# Boxes are drawn far further into the tail than the sibling figures' MIN_N=25 allows,
# so every method covers its own full range instead of being clipped to the narrowest
# one. Ten molecules is the floor for a box to carry quartiles at all; the share figure
# is what tells the reader which end of the axis is thin. The crystal ligands are a LINE
# in the box figure, not a ninth box series, so this floor never applies to them -- theirs
# is pose_common's MIN_REF over the +-REF_WIN window, and it is why their line stops at 9.
GRID_MIN_N = 10
TAIL = 1e4
# Floor for the log-scaled share panels; below this a bond count is empty, not rare.
DIST_FLOOR = 0.05
STATS = {"median": lambda v: float(np.median(v)), "mean": lambda v: float(np.mean(v))}
KEY_H = 0.92                 # inches of key strip under the line panels; see lines()

# The boxes bin the axis; the lines keep every count. Half-open, so these are {0}, {1,2},
# {3,4}, {5,6}, {7,8,9}, {10+}.
BIN_EDGES = [0, 1, 3, 5, 7, 10, 10 ** 6]
BIN_LABELS = ["0", "1–2", "3–4", "5–6", "7–9", "10+"]

# The three arms this section runs locally, addressed by pose_common's stable KEYS --
# their LABELS moved ("VoxBind + Ours" -> "Ours") on 2026-09-10 and may move again.
LOCAL_KEYS = ("targetdiff", "vanilla", "ours_v1")
REF_KEY = "ours_v1"          # any arm carries the same crystal ligand per pocket
LOCAL_LABELS = []            # filled by load_all() from ARMS, in LOCAL_KEYS order

# OUR ARM IS LABELLED CoDE, whatever pose_common currently spells it. That file has called
# it "VoxBind + Ours" and then "Ours"; the method's name is CoDE and it is what
# ../../260827/table_drug_design.tex's last row carries. method_colors resolves every
# spelling to the same blue, so relabelling here cannot desynchronise the palette.
RELABEL = {"ours_v1": "CoDE"}

# THE ROW ORDER OF ../../260827/table_drug_design.tex, which is canonical for this section:
# Reference, AR, Pocket2Mol, DiffSBDD, TargetDiff, DecompDiff, VoxBind, FuncBind, then ours
# last. Legends, panels and exports all read from this, so a method cannot sit in one order
# in the legend and another in the share panels.
# The table's order, except that VoxBind is pulled down next to CoDE so the model our arm
# modifies sits immediately before it and the two read as a pair.
ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff", "FuncBind", "VoxBind",
         "CoDE"]

_RB = {}


def rot_bonds(smiles):
    if not smiles:
        return None
    if smiles not in _RB:
        m = Chem.MolFromSmiles(smiles)
        _RB[smiles] = None if m is None else int(rdMolDescriptors.CalcNumRotatableBonds(m))
    return _RB[smiles]


# ── the baselines ────────────────────────────────────────────────────────────────
def load_meta(folder, stem):
    """export_posecheck_json.py's loader, reproduced exactly: base, then `_part2`
    concatenated PER POCKET. `_gap.pt` ships in the bundle and that loader ignores it, so
    including it here would shift every molecule after the split."""
    d = os.path.join(BUNDLE, folder, "samples/meta")
    meta = torch.load(os.path.join(d, f"{stem}.pt"), weights_only=False)
    part2 = os.path.join(d, f"{stem}_part2.pt")
    if os.path.exists(part2):
        meta = [a + b for a, b in zip(meta, torch.load(part2, weights_only=False))]
    return meta


def baseline_rows(label, folder, stem, keep):
    """Rows shaped like pose_common's, for the p79 pockets, with the join checked.

    THE CHECK RUNS OVER EVERY POCKET THE EXPORT HOLDS -- all 100 -- while only the p79 ones
    become rows. Checking just the 79 that are drawn would leave the other 21 as evidence
    nobody looked at, and they cost nothing: the meta is already in memory and the export
    already carries them."""
    meta = load_meta(folder, stem)
    export = json.load(open(os.path.join(HERE, f"posecheck_{label}.json")))
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
            rb = rot_bonds(e.get("smiles"))
            unparsed += rb is None
            rows.append({"n": m["n"], "s": m["s"], "c": m["c"], "rb": rb})
    return rows, unparsed, len(export["molecules"]), checked


def load_all():
    """(label -> rows) for every drawn method, plus the crystal-ligand rows.

    THE THREE LOCAL ARMS ARE PICKED BY KEY, NOT BY WALKING pose_common.ARMS. That list grew
    from three arms to eight on 2026-09-10 -- the five published baselines were added to it,
    pointing at `exps/baselines_pose/<m>`, which another job is still filling. Those trees
    carry PoseBusters but no `posecheck` block yet, so walking ARMS here would load three
    methods' worth of rows with `s=None` and silently empty every range this builder
    computes. Selecting `targetdiff`/`vanilla`/`ours_v1` out of it keeps the roots in sync
    with that file while ignoring arms it cannot yet serve.

    WHEN `exps/baselines_pose/` IS COMPLETE, the bundle join below becomes unnecessary:
    those trees carry SMILES and will carry strain, so `build_strain_per_rotbond.py` will
    cover every method through pose_common alone and this builder can be retired. Until
    then the bundle is the only source with strain for the baselines."""
    by_key = {key: (lab, root) for lab, key, root in pc.ARMS}
    series, notes = {}, {}
    local = []
    for key in LOCAL_KEYS:
        if key not in by_key:
            raise SystemExit(f"pose_common.ARMS no longer defines {key!r}")
        lab, root = by_key[key]
        lab = RELABEL.get(key, lab)
        rows = [r for t in pc.P79 for r in pc.rows_of(os.path.join(root, t))]
        for r in rows:
            r["rb"] = rot_bonds(r.get("smi"))
        series[lab] = rows
        notes[lab] = {"source": f"{root} (pocket10 crop)", "n_rows": len(rows)}
        local.append(lab)
    LOCAL_LABELS[:] = local

    ref_root = by_key[REF_KEY][1]
    refrows = [r for t in pc.P79
               for r in pc.rows_of(os.path.join(ref_root, t), reference=True)]
    for r in refrows:
        r["rb"] = rot_bonds(r.get("smi"))

    keep = {int(t.split("_")[1]) for t in pc.P79}
    for lab, folder, stem in BASELINES:
        rows, bad, n_all, checked = baseline_rows(lab, folder, stem, keep)
        series[lab] = rows
        notes[lab] = {"source": f"results/task2-drugdesign/{folder} meta x posecheck_{lab}.json"
                                " (whole receptor)",
                      "n_rows": len(rows), "n_all_pockets": n_all, "smiles_unparsed": bad,
                      "pockets_join_checked": checked}
    return series, refrows, notes


def order(labels):
    """The drug-design table's row order, ours last. Anything the table does not name is a
    bug rather than something to append quietly, so it raises."""
    unknown = [l for l in labels if l not in ORDER]
    if unknown:
        raise SystemExit(f"not in table_drug_design.tex's order: {unknown}")
    return [l for l in ORDER if l in labels]


def style_of(label):
    return (pc.MODEL_LW, "-") if label in LOCAL_LABELS else (BASE_LW, BASE_DASH[label])


def panel_order(labels):
    """Same order as `order`, with the crystal ligands FIRST -- they are the table's first
    row and the thing every other panel is read against."""
    return [pc.REF_LABEL] + order(labels)


# ── figures ──────────────────────────────────────────────────────────────────────
def save(fig, stem):
    """pose_common.save appends a `_<variant>` suffix; this family has no core/all split --
    every figure it writes draws every method there is data for -- so it saves without one
    rather than leaving a dangling underscore in the filename."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), facecolor="white")
    plt.close(fig)


def handles(labels, solid=False):
    """`solid` for the box figure: nothing in it is a dashed line, so a dashed swatch in the
    key advertises an encoding the panel does not use. The crystal ligands keep their dash
    either way -- there they really are a dashed line."""
    h = [Line2D([], [], color=pc.REF_COLOR, lw=pc.REF_LW, ls=pc.DASH, label=pc.REF_LABEL)]
    for lab in labels:
        lw, ls = style_of(lab)
        h.append(Line2D([], [], color=color(lab), lw=pc.MODEL_LW if solid else lw,
                        ls="-" if solid else ls, label=lab))
    return h


def line_curve(per_lab, xs, f):
    """The statistic at each exact bond count, null where that count holds fewer than
    pose_common.MIN_N molecules. THE AXIS IS FIXED AT 0-X_MAX (see `main`) rather than cut
    back to the counts every method can answer: past nine bonds the methods thin out at
    very different rates -- Pocket2Mol has 40 ligands at nine and single figures at eleven,
    CoDE and VoxBind still have hundreds -- and the old rule let the emptiest method decide
    where everyone's line stopped. Now each line simply ends where its own method ran out,
    which is the more informative thing to show, and the crystal ligands' dashed line ends
    earlier still: 79 of them cannot fill a window at twelve bonds."""
    return [f(per_lab[x]) if len(per_lab.get(x, ())) >= pc.MIN_N else None for x in xs]


def lines(xs, per, ref_per, labels, stat):
    """One statistic, one panel, and the key in a strip of its own beneath it.

    NINE SERIES DO NOT LEAVE A CORNER FREE. Inside the axes this key covered FuncBind's
    spike at six bonds and everything above ~500 kcal/mol on the left half -- the part of
    the figure that carries the finding. Under the panel it covers nothing, and it is the
    same 3x3 block the box figure's key is."""
    f = STATS[stat]
    fig, (ax, key) = plt.subplots(2, 1, figsize=(pc.FIG_W, pc.PANEL_H + KEY_H), dpi=220,
                                  gridspec_kw={"height_ratios": [pc.PANEL_H, KEY_H]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key.axis("off")
    ax.plot(xs, pc.reference_curve(ref_per, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
            ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab in labels:
        lw, ls = style_of(lab)
        ax.plot(xs, line_curve(per[lab], xs, f), color=color(lab), lw=lw, ls=ls,
                zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    pc.furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=X_LABEL,
                 xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=XTICK)
    pc.legend(key, handles(labels), loc="center", ncol=3, fontsize=11)
    pc.fit(fig, pad=0.5)
    save(fig, f"strain_rotbond_all_methods_{stat}")


def _axis(labels, refrows, series):
    """The shared x range and per-method rows both figures are drawn over, so the boxes and
    the share panels cannot end up covering different molecules."""
    panels = panel_order(labels)
    rows_of = {lab: series[lab] for lab in labels}
    rows_of[pc.REF_LABEL] = refrows
    per = {lab: pc.by_size(rows_of[lab], "s", key="rb") for lab in panels}
    dist = {lab: collections.Counter(r["rb"] for r in rows_of[lab] if r["rb"] is not None)
            for lab in panels}
    # 0 to X_MAX, the same span `main` gives the line figures, so the three are read
    # against one axis BY CONSTRUCTION rather than by today's data happening to agree.
    # Boxes still appear only where a method has GRID_MIN_N molecules, so pinning the axis
    # adds no box; it only stops the axis shrinking when the tail thins.
    xs = list(range(0, X_MAX + 1))
    beyond = {lab: 100 * sum(n for x, n in dist[lab].items() if x > X_MAX)
              / max(sum(dist[lab].values()), 1) for lab in panels}
    return panels, per, dist, xs, beyond


def strain_boxes(series, refrows, labels):
    """Strain against rotatable bonds, every method on one axis, dodged inside each count.

    The paper's Fig. 12 shape: one panel of boxplots per number of rotatable bonds, all
    methods together. (The paper packs 151 boxes into one row and colours each box by its
    median strain, spending colour on the value; here colour stays the method key it is in
    every other 260910 figure, and the legend carries it.)

    The crystal ligands stay the dashed grey line rather than becoming a ninth box series:
    79 of them over thirteen bond counts cannot fill a box at every count, and as a line
    they are the same ruler here that they are in every sibling figure."""
    panels, per, dist, xs, beyond = _axis(labels, refrows, series)
    if len(xs) > SPLIT_ABOVE:
        split = len(xs) - len(xs) // 2                 # low row takes the extra count
        bands = [xs[:split], xs[split:]]
    else:
        bands = [xs]
    ref_line = {x: v for x, v in
                zip(xs, pc.reference_curve(per[pc.REF_LABEL], xs, STATS["median"]))}

    fig, axes = plt.subplots(len(bands), 1, figsize=(pc.FIG_W * BOX_WIDE_1,
                                                     pc.PANEL_H * BOX_TALL * len(bands)),
                             dpi=220, squeeze=False)
    fig.patch.set_facecolor("white")
    n = len(labels)
    slot = (1.0 - BOX_GAP) / n
    for band, ax in zip(bands, axes.ravel()):
        ax.set_facecolor("white")
        for i, lab in enumerate(labels):
            offs = (i - (n - 1) / 2) * slot
            drawn = [x for x in band if x in per[lab] and len(per[lab][x]) >= GRID_MIN_N]
            if not drawn:
                continue
            col = color(lab)
            bp = ax.boxplot([np.clip(per[lab][x], STRAIN_FLOOR, None) for x in drawn],
                            positions=[x + offs for x in drawn], widths=slot * 0.88,
                            whis=WHIS, showfliers=False, patch_artist=True, zorder=5,
                            manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=0.8)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=col, linewidth=0.8)
            for med in bp["medians"]:
                med.set(color=pc.INK, linewidth=1.1, solid_capstyle="butt")
        ax.plot(band, [ref_line.get(x) for x in band], color=pc.REF_COLOR, lw=pc.REF_LW,
                ls=pc.DASH, zorder=6, dash_capstyle="round")
        ax.set_yscale("log")
        # A FIXED FIVE DECADES, AND THE TAIL IS ALLOWED TO RUN OFF THE TOP. FuncBind's
        # 95th percentile at 6 rotatable bonds reaches ~1e11; autoscaling to it stretched
        # the panel over fourteen decades and pressed every box into the bottom fifth. The
        # window is pinned so the boxes -- which all sit between 1e0 and 1e5 -- stay legible
        # across every rebuild, and the whiskers that leave the top simply leave it. How
        # much tail each method carries is reported as `strain_gt_1e4`, not drawn.
        ax.set_ylim(*BOX_YLIM)
        pc.furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{WHIS[0]:g}–{WHIS[1]:g}th pct "
                                "whiskers",
                     xlabel=X_LABEL, xlim=(band[0] - 0.62, band[-1] + 0.62), xloc=1)
        ax.set_xticks(band)
    # Nine entries -- the crystal ligands plus eight methods -- in three columns, so the
    # key is a 3x3 block rather than one long strip across the top of the panel.
    # Lower right: the boxes climb left-to-right, so the empty corner is under the high
    # bond counts, where only the lower whiskers reach.
    pc.legend(axes.ravel()[0], handles(labels, solid=True), loc="lower right",
              fontsize=10, ncol=3)
    pc.fit(fig, pad=0.5)
    save(fig, "strain_box_per_rotbond_all_methods")


def rotbond_distribution(series, refrows, labels):
    """Where each method puts its ligands on the same axis the boxes use -- its own share
    at each rotatable-bond count, one panel per method in a 3x3 block.

    ITS OWN FIGURE, not a strip under the boxes. The two answer different questions and are
    read at different times: the boxes compare methods at a bond count, this compares the
    bond counts a method produces. Sharing a canvas forced one to be a third the height of
    the other, and a nine-panel block does not fit under a box panel at any useful size.

    LOG y. The share spans two decades inside 0-12 bonds -- Pocket2Mol puts 33 % at one bond
    and 0.1 % at twelve -- and on a linear axis everything under ~2 % is a flat line on the
    floor."""
    panels, per, dist, xs, beyond = _axis(labels, refrows, series)
    ncol = DIST_COLS
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(pc.FIG_W * DIST_WIDE,
                                                 pc.PANEL_H * DIST_TALL * nrow),
                             dpi=220, sharex=True, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for i, lab in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor("white")
        col = pc.REF_COLOR if lab == pc.REF_LABEL else color(lab)
        total = sum(dist[lab].values())
        pct = [100 * dist[lab].get(x, 0) / total for x in xs]
        ax.step(xs, pct, where="mid", color=col, lw=pc.DIST_LW, zorder=3)
        ax.fill_between(xs, pct, DIST_FLOOR, step="mid", color=col, alpha=pc.DIST_FILL,
                        lw=0, zorder=2)
        ax.set_yscale("log")
        # The x name goes under the BOTTOM ROW only; repeated under all nine it is wider
        # than a panel and the copies overprint each other.
        pc.furniture(ax, ylabel="% of ligands" if i % ncol == 0 else None,
                     xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=2)
        ax.set_ylim(bottom=DIST_FLOOR)
        ax.set_title(lab, fontsize=12, color=pc.INK, loc="left", pad=4)
        ax.text(0.97, 0.07, f">{X_MAX} bonds: {beyond[lab]:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=pc.AXIS, zorder=6)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    # ONE x name for the whole block, centred under the bottom row. sharex already leaves
    # the tick labels on that row alone; three copies of the name did not fit -- each is
    # wider than a panel -- and overprinted each other.
    fig.supxlabel(X_LABEL, fontsize=15.5, color=pc.INK)
    pc.fit(fig, pad=0.5, h_pad=0.9)
    save(fig, "rotbond_distribution_all_methods")


def bin_of(rb):
    return min(int(np.searchsorted(BIN_EDGES, rb, side="right")) - 1, len(BIN_LABELS) - 1)


# ── exports ──────────────────────────────────────────────────────────────────────
def tail_share(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > TAIL for x in v) / len(v), 2) if v else None


def exports(xs, series, per, ref_per, refrows, labels, notes):
    out = {
        "built": "2026-09-10",
        "metric": "PoseCheck strain against RDKit strict rotatable-bond count",
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79)},
        "join": ("baselines: results/task2-drugdesign/<M>/samples/meta (base + _part2, per "
                 "pocket) x posecheck_<M>.json, verified position-by-position on the "
                 "heavy-atom sequence"),
        "join_checked": ("every baseline's heavy-atom sequence was verified against its "
                         "export on ALL 100 pockets (see each arm's pockets_join_checked); "
                         "rows are kept for the 79 drawn"),
        "n_columns": ("`n` is the molecules at that exact count. `n_window` is the sample "
                      "the value on the row was computed from -- identical for the models, "
                      f"which use no window, and the +-{pc.REF_WIN}-count pool for the "
                      "crystal ligands, which do"),
        "scope_note": ("baselines scored whole-receptor, local arms on the pocket10 crop; "
                       "measured median |rel diff| 0.6-1.0 % on strain, which is "
                       "receptor-independent by construction"),
        "bins": {"labels": BIN_LABELS, "edges": BIN_EDGES[:-1] + ["inf"]},
        "rotatable_bonds": xs, "arms": {},
    }
    for lab in labels:
        rows = series[lab]
        out["arms"][lab] = {
            **notes[lab],
            "n_scored": sum(len(v) for v in per[lab].values()),
            "strain_gt_1e4": tail_share(rows),
            "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2),
            **{f"strain_{s}": line_curve(per[lab], xs, STATS[s]) for s in STATS},
            "bins": {b: (round(float(np.median(v)), 2) if v else None) for b, v in
                     ((BIN_LABELS[i], [r["s"] for r in rows if r["s"] is not None
                                       and r["rb"] is not None and bin_of(r["rb"]) == i])
                      for i in range(len(BIN_LABELS)))},
        }
    ref_window = [sum(len(v) for x, v in ref_per.items() if abs(x - a) <= pc.REF_WIN)
                  for a in xs]
    out["arms"][pc.REF_LABEL] = {
        "source": "crystal ligand of each of the 79 pockets",
        "window": f"+-{pc.REF_WIN} counts, drawn where it pools >= {pc.MIN_REF} ligands",
        "n": [len(ref_per.get(a, ())) for a in xs],
        "n_window": ref_window,
        "n_scored": sum(len(v) for v in ref_per.values()),
        "strain_gt_1e4": tail_share(refrows),
        "atoms_mean": round(float(np.mean([r["n"] for r in refrows])), 2),
        **{f"strain_{s}": pc.reference_curve(ref_per, xs, STATS[s]) for s in STATS},
    }
    json.dump(out, open(os.path.join(HERE, "strain_rotbond_all_methods.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "strain_rotbond_all_methods.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "rotatable_bonds", "n", "n_window", "strain_median",
                    "strain_mean"])
        for lab in labels:
            for i, x in enumerate(xs):
                d, n = out["arms"][lab], len(per[lab].get(x, ()))
                w.writerow([lab, x, n, n,          # the models are drawn per exact count
                            "" if d["strain_median"][i] is None else round(d["strain_median"][i], 3),
                            "" if d["strain_mean"][i] is None else round(d["strain_mean"][i], 3)])
        d = out["arms"][pc.REF_LABEL]
        for i, x in enumerate(xs):
            w.writerow([pc.REF_LABEL, x, len(ref_per.get(x, ())), ref_window[i],
                        "" if d["strain_median"][i] is None else round(d["strain_median"][i], 3),
                        "" if d["strain_mean"][i] is None else round(d["strain_mean"][i], 3)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stat", choices=("median", "mean", "both"), default="both")
    ap.add_argument("--kind", choices=("box", "line", "both"), default="both")
    args = ap.parse_args()
    stats = ("median", "mean") if args.stat == "both" else (args.stat,)
    kinds = ("box", "line") if args.kind == "both" else (args.kind,)

    pc.use_style()
    series, refrows, notes = load_all()
    labels = order(series)
    per = {lab: pc.by_size(series[lab], "s", key="rb") for lab in labels}
    ref_per = pc.by_size(refrows, "s", key="rb")

    # 0 to X_MAX, always -- the same span the box and share figures cover, so the three
    # are read against one axis. Where a method (or the reference) is too thin at a count,
    # its own line stops; the axis does not.
    xs = list(range(0, X_MAX + 1))
    if "line" in kinds:
        for s in stats:
            lines(xs, per, ref_per, labels, s)
    if "box" in kinds:
        strain_boxes(series, refrows, labels)
        rotbond_distribution(series, refrows, labels)
    out = exports(xs, series, per, ref_per, refrows, labels, notes)

    print(f"{len(pc.P79)} pockets · {len(labels)} methods + reference · "
          f"x = {xs[0]}-{xs[-1]} rotatable bonds "
          f"(each line drawn where its own method has ≥{pc.MIN_N} molecules)\n")
    for lab in labels:
        drawn = [x for x in xs if len(per[lab].get(x, ())) >= pc.MIN_N]
        # A break INSIDE a method's span is reported rather than collapsed into the
        # endpoints -- "0-12" over a line with a hole in it would be a false summary.
        gaps = [x for x in range(drawn[0], drawn[-1] + 1) if x not in drawn] if drawn else []
        print(f"  {lab:12s} line drawn {drawn[0]}-{drawn[-1]}"
              + (f", broken at {gaps}" if gaps else "") if drawn else
              f"  {lab:12s} nowhere thick enough to draw")
    ref_drawn = [x for x, v in zip(xs, pc.reference_curve(ref_per, xs, STATS["median"]))
                 if v is not None]
    print(f"  {pc.REF_LABEL:12s} line drawn {ref_drawn[0]}-{ref_drawn[-1]} "
          f"(±{pc.REF_WIN} window, ≥{pc.MIN_REF} ligands)\n")
    head = BIN_LABELS
    print(f"{'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{b:>8s}" for b in head) + f" {'>1e4':>7s}   (strain median by bin)")
    for lab in labels + [pc.REF_LABEL]:
        d = out["arms"][lab]
        if lab == pc.REF_LABEL:
            rows = refrows
            cells = [[r["s"] for r in rows if r["s"] is not None and r["rb"] is not None
                      and bin_of(r["rb"]) == i] for i in range(len(BIN_LABELS))]
            vals = [f"{np.median(v):8.1f}" if len(v) >= pc.MIN_REF else f"{'—':>8s}"
                    for v in cells]
        else:
            vals = [f"{d['bins'][b]:8.1f}" if d["bins"][b] is not None else f"{'—':>8s}"
                    for b in head]
        print(f"{lab:16s} {d['n_scored']:6d} {d['atoms_mean']:6.1f} " + " ".join(vals)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
