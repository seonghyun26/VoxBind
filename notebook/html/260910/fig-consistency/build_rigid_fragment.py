#!/usr/bin/env python3
"""build_rigid_fragment.py — rigid-fragment consistency against fragment size.

    rigid_violin_{core,all}.*                   one violin per exact fragment size to 9,
                                                then 10+: range, IQR, median and mean
    rigid_violin_paperbins_{core,all}.*         the same, over the paper's coarser bins
    rigid_per_size_{median,mean}_{core,all}.*   RMSD against rigid-fragment size,
                                                over each arm's fragment-size distribution
    rigid_ecdf_pair_{core,all}.*                the RMSD distribution, all fragments and
                                                the 7+ atom fragments, on one shared y
    rigid_fragment_per_size.{json,csv}          the numbers behind the curves
    rigid_fragment_summary.json                 pooled numbers (p79 and all_pockets)

THE METRIC. A rigid fragment has no internal degrees of freedom -- a benzene ring, an
amide, a fused bicycle each has exactly one correct shape, so a force field has no reason
to move it. Optimise the molecule with MMFF, cut every rotatable bond, and RMSD each
fragment against its own optimised self after Kabsch superposition. If it moved, the
generated geometry was wrong to begin with. Lower is better; the unit is Angstrom.

It is the complement of the two metrics next door. Strain (../fig-posecheck) relaxes under
a 0.1 A position constraint precisely so that local errors wash out and only GLOBAL
conformational strain is reported -- exactly what this measures instead. PoseBusters
(../fig-posebusters) asks the same local question but only through pass/fail thresholds;
this is continuous. A model can be good at one and bad at another, and here it is.

WHAT IS ON THE X AXIS IS THE FRAGMENT, NOT THE LIGAND. Every other size-resolved figure in
260910 plots against heavy atoms in the ligand. This one cannot: the metric has no
per-molecule value, only per-fragment ones, and fragment size is what drives it (a 2-atom
fragment is a bond length, a 14-atom fragment is a fused ring system). So the axis, the
distribution strip and MIN_N all count FRAGMENTS. Do not read a point here against a point
in fig-posecheck at the same x.

THE REFERENCE WINDOW IS NARROWER HERE, for the same reason. pose_common windows the crystal
ligands over +-4 atoms because ligand sizes run 5-45; fragment sizes run 2-25 and the
distribution is spiky (a third of all fragments have exactly 2 atoms, and 6 -- benzene --
is the next spike), so a +-4 window would pool a bond length with a fused bicycle. This
module narrows it to +-1 before drawing anything.

Everything else -- arm list, pocket set, colours, furniture -- comes from ../pose_common.py
and ../method_colors.py, so an arm reads the same here as in every other 260910 figure. The
evaluation itself is voxbind/exps/frozenenc_probes/eval_rigid_fragments.py; this only
draws what that wrote.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-consistency/build_rigid_fragment.py
"""
import collections
import csv
import json
import os
import statistics as st
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

# The reference SHARE in the distribution strip is still rolled, but over +-1 atom rather
# than pose_common's +-4: fragment sizes run 2-25, and a +-4 roll would smear a strip whose
# whole point is where the spikes are. The reference STATISTIC does not use this at all --
# see ref_curve. Set before anything is drawn.
pc.REF_WIN, pc.MIN_REF = 1, 12

# The crystal reference is drawn per exact fragment size, where at least this many crystal
# fragments have that size.
MIN_REF_EXACT = 8

X_LABEL = "Number of heavy atoms in rigid fragment"
XTICK = 2
RESULTS, REFERENCE = "rigid_fragment_results.json", "rigid_fragment_reference.json"
STATS = (("median", lambda v: float(np.median(v))),
         ("mean", lambda v: float(np.mean(v))))
# TWO BINNINGS, AND THE FINE ONE IS THE DEFAULT READ.
#
# `fine` is one bin per exact fragment size up to 9, then a 10+ tail. Every model arm holds
# 567-10,952 fragments at each of those sizes, so nothing needs pooling, and pooling costs
# something real: the `paper` binning puts benzene (6 atoms, where both VoxBind arms sit at
# crystal quality, 21% of all their fragments) in one bin with the thin and much worse
# 5-atom fragments, which hides the finding.
#
# `paper` is what eval_rigid_fragments.py writes and what ../../260827 reported -- roughly
# log-spaced, chosen there to match the form the VoxBind paper reports. It is kept so the
# numbers stay directly comparable to that write-up, under its own filename. It is NOT
# kept because the data asks for it: only the 79 crystal ligands are thin enough to need
# pooling (9-15 fragments at sizes 4, 5, 8 and 9), and see MIN_BODY for how that is
# handled instead.
BINNINGS = {
    "fine": ([(n, n + 1) for n in range(2, 10)] + [(10, 10 ** 6)],
             [str(n) for n in range(2, 10)] + ["10+"]),
    "paper": ([(2, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, 10 ** 6)],
              ["2", "3–4", "5–6", "7–9", "10–13", "14+"]),
}
# The stem each binning is saved under. The fine one keeps the plain name: two figures that
# differ in how they group the data must not be able to sit in a folder under one name.
BINNING_STEM = {"fine": "rigid_violin", "paper": "rigid_violin_paperbins"}
# The tables and the run summary stay on the paper bins, so they keep lining up with 260827.
PAPER_BINS, BIN_LABELS = BINNINGS["paper"]
# A KDE fitted to fewer than this many values is a shape invented from noise. Below it the
# violin is drawn as its glyphs alone -- range, IQR, median, mean -- which are honest at any
# n. In practice this only ever touches the crystal reference.
MIN_BODY = 20
# The right-hand ECDF panel: where both models leave the crystal ligands behind.
BIG_FRAG = 7


# Below this many labelled decades on the axis, the 2/3/5 minor ticks get labels too.
MIN_DECADE_LABELS = 3


def log_y_ticks(ax):
    """A log y axis holding one or two decade labels leaves the reader almost no scale to
    read the curve against -- the mean panel spans 0.02 to 0.7 and shows `10^-1`, alone.
    Where that happens, label the 2/3/5 minor ticks as plain decimals as well.

    The test counts the labels the axis will actually SHOW, not the span of the data: a log
    axis autoscales out to the enclosing decades, so a 0.025-0.72 curve reports a span of
    ~1.7 decades while still carrying two labels."""
    lo, hi = ax.get_ylim()
    if len([t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]) >= MIN_DECADE_LABELS:
        return
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5), numticks=20))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="minor", labelsize=12.5, colors=pc.AXIS,
                   length=2.5, width=pc.AXIS_LW)


def ref_curve(per, xs, f):
    """The crystal ligands at each EXACT fragment size, from xs[0] up to the first size
    with fewer than MIN_REF_EXACT of them.

    NOT pose_common.reference_curve, and the difference matters. That windows the crystal
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
        if len(v) < MIN_REF_EXACT:
            break
        out.append(f(v))
    return out + [None] * (len(xs) - len(out))


def contiguous(xs):
    """The leading run of consecutive sizes. pose_common.x_range keeps every count where
    all arms clear MIN_N, which for LIGAND sizes is a dense range; fragment sizes are not
    dense out in the tail -- 25, 26 and 27 fall under the floor while 28 (a common fused
    system) clears it -- and a line drawn from 24 to 28 would cross three counts that were
    dropped for being too thin. Truncating is the same rule the rest of the folder follows:
    no point is invented, and nothing is interpolated over a gap."""
    for i, (a, b) in enumerate(zip(xs, xs[1:])):
        if b != a + 1:
            return xs[:i + 1]
    return xs


# ── load ─────────────────────────────────────────────────────────────────────────
def load(path):
    """{target: {n_mols, n_mmff_failed, fragments:[(size, rmsd), ...]}} for every target
    the run scored. Filtering to the 79 happens in the caller, so the pooled all-pockets
    numbers stay available for the summary."""
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    return {t["target"]: t for t in j["per_target"]}


def pooled(per_target, targets=None):
    """(fragment rows, molecule count, MMFF failure count) over a set of targets."""
    names = sorted(per_target) if targets is None else [t for t in targets
                                                        if t in per_target]
    rows = [tuple(f) for t in names for f in per_target[t]["fragments"]]
    return (rows,
            sum(per_target[t]["n_mols"] for t in names),
            sum(per_target[t]["n_mmff_failed"] for t in names))


def by_frag_size(rows):
    out = collections.defaultdict(list)
    for n, v in rows:
        out[n].append(v)
    return out


ARM_DATA = {key: load(os.path.join(root, RESULTS)) for _, key, root in pc.ARMS}
REF_DATA = load(os.path.join(pc.REF_ROOT, REFERENCE))
P79_FRAGS = {key: pooled(ARM_DATA[key], pc.P79) for key in ARM_DATA}
REF_FRAGS = pooled(REF_DATA, pc.P79)
PER = {key: by_frag_size(P79_FRAGS[key][0]) for key in ARM_DATA}
REF_PER = by_frag_size(REF_FRAGS[0])


# ── figure 1: RMSD per fragment size, over the fragment-size distribution ────────
def fig_per_size(xs, arms, variant, stat, f):
    """One statistic, one panel, one file -- the rule the 3-line figures set. The median is
    the one to read: the RMSD distribution inside a size is heavy-tailed (a handful of
    fragments the force field rebuilds outright sit an order of magnitude above the bulk),
    so the mean tracks that tail rather than the typical fragment. Both are drawn, apart,
    and the filename says which."""
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(pc.FIG_W, pc.STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": pc.HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    top.plot(xs, ref_curve(REF_PER, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
             ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, pc.model_curve(PER[key], xs, f), color=color(lab), lw=pc.MODEL_LW,
                 zorder=5, solid_capstyle="round")
    top.set_yscale("log")
    # Two lines, as the strain figures do it: on one line the label is taller than this
    # panel and runs into the distribution strip's own label.
    pc.furniture(top, ylabel=f"Fragment RMSD {stat}\n(Å)",
                 xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=XTICK)
    log_y_ticks(top)
    # Lower right, not the house default upper left: every curve climbs to the right, so
    # upper left is where TargetDiff's rise is and the opaque legend box would cover it.
    pc.legend(top, pc.arm_handles(arms), loc="lower right", fontsize=11.5)

    pc.size_distribution(bot, xs, PER, REF_PER, arms)
    pc.furniture(bot, ylabel="% of fragments", xlabel=X_LABEL,
                 xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=XTICK)
    fig.align_ylabels((top, bot))
    pc.fit(fig, pad=0.5, h_pad=pc.H_PAD)
    pc.save(fig, HERE, f"rigid_per_size_{stat}", variant)


# ── figure 2: the paper's own form ──────────────────────────────────────────────
# The violin y axis is floored here. The 2-atom bin reaches 8.6e-08 -- a bond MMFF did not
# move at all -- which is seven decades below that bin's median, and an axis that reached it
# would squash every violin in the figure into a line. Values below the floor are drawn AT
# it and the share that hits it is reported in the run log and the README.
VIOLIN_FLOOR = 1e-4
VIOLIN_TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]


def binned(rows, f, binning="paper"):
    """[(label, value, n), ...] over a binning, skipping a bin nothing lands in."""
    bins, labels = BINNINGS[binning]
    out = []
    for (lo, hi), lab in zip(bins, labels):
        v = [r[1] for r in rows if lo <= r[0] < hi]
        if v:
            out.append((lab, f(v), len(v)))
    return out


def bin_values(rows, binning="paper"):
    """{bin label: np.array of RMSDs}."""
    bins, labels = BINNINGS[binning]
    return {lab: np.array([r[1] for r in rows if lo <= r[0] < hi])
            for (lo, hi), lab in zip(bins, labels)}


def fig_violin(arms, variant, binning):
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
    violins in ../fig-posecheck.
    """
    labels_all = BINNINGS[binning][1]
    series = [(pc.REF_LABEL, pc.REF_COLOR, REF_FRAGS[0])] + \
             [(lab, color(lab), P79_FRAGS[key][0]) for lab, key, _ in arms]
    per = [(lab, col, bin_values(rows, binning)) for lab, col, rows in series]
    labels = [l for l in labels_all if any(len(v[l]) for _, _, v in per)]
    idx = np.arange(len(labels))
    slot = 0.86 / len(series)
    t = lambda v: np.log10(np.clip(v, VIOLIN_FLOOR, None))

    fig, ax = plt.subplots(figsize=(pc.FIG_W * 1.52, pc.PANEL_H * 1.30), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    clipped, thin = {}, {}
    for i, (lab, col, vals) in enumerate(per):
        pos = idx - 0.43 + slot * (i + 0.5)
        keep = [(x, vals[l]) for x, l in zip(pos, labels) if len(vals[l])]
        clipped[lab] = sum(int((v < VIOLIN_FLOOR).sum()) for _, v in keep)
        body_at = [(x, v) for x, v in keep if len(v) >= MIN_BODY]
        thin[lab] = [l for l, (_, v) in zip(labels, keep) if len(v) < MIN_BODY]
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
            if lab == pc.REF_LABEL:     # the benchmark, as in every other figure here
                body.set_hatch("///")
        for x, v in keep:
            lo, q1, med, q3, hi = np.percentile(v, [0, 25, 50, 75, 100])
            ax.vlines(x, t(lo), t(hi), color=pc.INK, lw=0.9, zorder=3)
            ax.hlines([t(lo), t(hi)], x - slot * 0.16, x + slot * 0.16, color=pc.INK,
                      lw=0.9, zorder=3)
            ax.vlines(x, t(q1), t(q3), color=pc.INK, lw=4.0, zorder=4)
            ax.plot(x, t(med), "o", mfc="white", mec=pc.INK, mew=0.7, ms=4.4, zorder=6)
            ax.plot(x, t(v.mean()), "D", mfc=pc.INK, mec="white", mew=0.8, ms=3.8,
                    zorder=6)

    pc.furniture(ax, ylabel="Fragment RMSD (Å)",
                 xlabel="Rigid fragment size (heavy atoms)", xloc=None)
    ax.grid(False, axis="x")      # the groups are the categories; an x rule is only ink
    # The axis is linear in log10(RMSD) -- see the docstring -- so the ticks are placed by
    # hand and labelled as powers of ten, which is what the axis actually is and what the
    # other log-scaled figures in this folder show.
    ax.set_yticks([np.log10(v) for v in VIOLIN_TICKS])
    ax.set_yticklabels([f"$10^{{{int(round(np.log10(v)))}}}$" for v in VIOLIN_TICKS])
    ax.set_ylim(np.log10(VIOLIN_FLOOR) - 0.12,
                max(t(v).max() for _, _, vals in per for v in vals.values() if len(v)) + 0.30)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.58, len(labels) - 0.42)

    swatches = [plt.Rectangle((0, 0), 1, 1, facecolor=col, alpha=0.62, edgecolor=col,
                              hatch="///" if lab == pc.REF_LABEL else None, label=lab)
                for lab, col, _ in per]
    ax.add_artist(pc.legend(ax, swatches, loc="lower right", fontsize=11.5))
    glyphs = [
        Line2D([], [], color=pc.INK, lw=0.9, label="min–max"),
        Line2D([], [], color=pc.INK, lw=4.0, label="IQR"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec=pc.INK, mew=0.7, ms=4.4,
               label="median"),
        Line2D([], [], ls="none", marker="D", mfc=pc.INK, mec="white", mew=0.8, ms=3.8,
               label="mean"),
    ]
    pc.legend(ax, glyphs, loc="upper left", ncol=4, fontsize=10.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, BINNING_STEM[binning], variant)
    return clipped, thin


# ── figure 3: the distribution itself ───────────────────────────────────────────
def fig_ecdf_pair(arms, variant):
    """Two panels on one shared y, the pose ECDF-pair layout. Left, every fragment; right,
    only the 7+ atom fragments. The split is there because the left panel is dominated by
    the 2- and 3-atom fragments -- more than half of every arm's total -- whose RMSD is a
    bond length and where all four lines sit on top of each other. The right panel is the
    same curve over the fragments that actually carry ring and conjugation geometry."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(pc.FIG_W * 1.52, pc.PANEL_H),
                                      dpi=220, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")

    def ecdf(ax, keep):
        def draw(rows, col, lw, ls):
            v = np.sort(np.clip([r[1] for r in rows if keep(r[0])], 1e-4, None))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=lw, ls=ls,
                    zorder=4 if ls != "-" else 5)
        for lab, key, _ in arms:
            draw(P79_FRAGS[key][0], color(lab), pc.MODEL_LW, "-")
        draw(REF_FRAGS[0], pc.REF_COLOR, pc.REF_LW, pc.DASH)

    ecdf(left, lambda n: True)
    left.set_xscale("log")
    left.set_xlim(1e-3, 3e0)
    pc.furniture(left, ylabel="Cumulative share", xlabel="Fragment RMSD (Å)", xloc=None)
    left.set_ylim(0, 1.0)
    pc.legend(left, pc.arm_handles(arms), loc="upper left", fontsize=11.5)

    ecdf(right, lambda n: n >= BIG_FRAG)
    right.set_xscale("log")
    right.set_xlim(1e-3, 3e0)
    pc.furniture(right, ylabel=None,
                 xlabel=f"Fragment RMSD (Å), {BIG_FRAG}+ atom fragments", xloc=None)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "rigid_ecdf_pair", variant)


# ── data exports ────────────────────────────────────────────────────────────────
def block(rows, n_mols, n_failed):
    v = [r[1] for r in rows]
    binned = {}
    for lo, hi in PAPER_BINS:
        sel = [r[1] for r in rows if lo <= r[0] < hi]
        if sel:
            name = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
            binned[name] = {"n": len(sel), "median": round(st.median(sel), 4),
                            "mean": round(float(np.mean(sel)), 4)}
    return {
        "n_molecules": n_mols,
        "n_mmff_failed": n_failed,
        "mmff_failure_rate_pct": round(100 * n_failed / n_mols, 2) if n_mols else None,
        "n_fragments": len(rows),
        "frag_atoms_mean": round(float(np.mean([r[0] for r in rows])), 2) if rows else None,
        "rmsd_median": round(st.median(v), 4) if v else None,
        "rmsd_mean": round(float(np.mean(v)), 4) if v else None,
        "rmsd_q25": round(float(np.percentile(v, 25)), 4) if v else None,
        "rmsd_q75": round(float(np.percentile(v, 75)), 4) if v else None,
        "by_fragment_size": binned,
    }


def exports(xs):
    summary = {
        "built": "2026-09-09",
        "metric": "rigid-fragment consistency (VoxBind paper Fig. 11b), MMFF / RDKit",
        "definition": ("MMFF-optimise the molecule, cut every rotatable bond, Kabsch-"
                       "superimpose each fragment on its optimised self and take the "
                       "heavy-atom RMSD. Angstrom, lower is better."),
        "evaluation": "voxbind/exps/frozenenc_probes/eval_rigid_fragments.py",
        "caveat": ("molecules MMFF cannot parameterise are excluded from every number "
                   "here; the exclusion rate differs by arm and is reported per arm"),
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79),
                       "targets": pc.P79},
        "arms": {},
        "reference_ligand": block(*REF_FRAGS),
    }
    for lab, key, root in pc.ARMS:
        summary["arms"][lab] = {
            "root": root, "key": key, "pockets_all": len(ARM_DATA[key]),
            "p79": block(*P79_FRAGS[key]),
            "all_pockets": block(*pooled(ARM_DATA[key])),
        }
    with open(os.path.join(HERE, "rigid_fragment_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    per_size = {"fragment_atoms": xs, "n_pockets": len(pc.P79), "arms": {}}
    for lab, key, _ in pc.ARMS:
        per_size["arms"][lab] = {
            "n": [len(PER[key].get(a, ())) for a in xs],
            **{f"rmsd_{stat}": pc.model_curve(PER[key], xs, f) for stat, f in STATS},
        }
    per_size["arms"][pc.REF_LABEL] = {
        "note": (f"per exact fragment size, null past the first size holding fewer than "
                 f"{MIN_REF_EXACT} crystal fragments; nothing is windowed"),
        "n": [len(REF_PER.get(a, ())) for a in xs],
        **{f"rmsd_{stat}": ref_curve(REF_PER, xs, f) for stat, f in STATS},
    }
    with open(os.path.join(HERE, "rigid_fragment_per_size.json"), "w",
              encoding="utf-8") as fh:
        json.dump(per_size, fh, indent=1, ensure_ascii=False)

    fields = [f"rmsd_{stat}" for stat, _ in STATS]
    with open(os.path.join(HERE, "rigid_fragment_per_size.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "fragment_atoms", "n"] + fields)
        for arm, d in per_size["arms"].items():
            for i, a in enumerate(xs):
                w.writerow([arm, a, d["n"][i]]
                           + ["" if d[f][i] is None else round(d[f][i], 4) for f in fields])

    # the violin figure's numbers, flat -- every glyph it draws, per arm and bin
    with open(os.path.join(HERE, "rigid_fragment_by_bin.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["binning", "arm", "fragment_size_bin", "n", "min", "q25", "median",
                    "mean", "q75", "max"])
        for binning in BINNINGS:
            for lab in [l for l, _, _ in pc.ARMS] + [pc.REF_LABEL]:
                rows = REF_FRAGS[0] if lab == pc.REF_LABEL \
                    else P79_FRAGS[dict((l, k) for l, k, _ in pc.ARMS)[lab]][0]
                for name, v in bin_values(rows, binning).items():
                    if not len(v):
                        continue
                    lo, q1, med, q3, hi = np.percentile(v, [0, 25, 50, 75, 100])
                    w.writerow([binning, lab, name, len(v)] + [round(float(x), 4) for x in
                               (lo, q1, med, v.mean(), q3, hi)])
    return summary


def main():
    pc.use_style()
    ranges, clipped, thin = {}, {}, {}
    for variant, arms in pc.variants():
        xs = contiguous(pc.x_range(PER, arms))
        ranges[variant] = xs
        for stat, f in STATS:
            fig_per_size(xs, arms, variant, stat, f)
        for binning in BINNINGS:
            clipped[variant], thin[(variant, binning)] = fig_violin(arms, variant, binning)
        fig_ecdf_pair(arms, variant)
    summary = exports(ranges["all"])

    print(f"{len(pc.P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (fragment sizes where every drawn arm has >={pc.MIN_N} fragments,\n"
          + "  truncated at the first gap: "
          + ", ".join(f"{v} drops {sorted(set(pc.x_range(PER, a)) - set(ranges[v]))}"
                      for v, a in pc.variants()) + ")\n")
    head = f"{'arm':16s} {'frags':>7s} {'med':>7s} {'mean':>7s} " \
           f"{'mols':>6s} {'MMFF fail':>10s}"
    print(head)
    for lab in [l for l, _, _ in pc.ARMS] + [pc.REF_LABEL]:
        r = summary["arms"][lab]["p79"] if lab != pc.REF_LABEL \
            else summary["reference_ligand"]
        print(f"{lab:16s} {r['n_fragments']:7d} {r['rmsd_median']:7.4f} "
              f"{r['rmsd_mean']:7.4f} {r['n_molecules']:6d} "
              f"{r['mmff_failure_rate_pct']:9.1f}%")

    print(f"\n{'fragment size':>14s} " + "".join(f"{l:>18s}" for l in
          [l for l, _, _ in pc.ARMS] + [pc.REF_LABEL]))
    for name in summary["reference_ligand"]["by_fragment_size"]:
        cells = ""
        for lab in [l for l, _, _ in pc.ARMS] + [pc.REF_LABEL]:
            b = (summary["arms"][lab]["p79"] if lab != pc.REF_LABEL
                 else summary["reference_ligand"])["by_fragment_size"].get(name)
            cells += f"{b['median']:10.4f} ({b['n']:>5d})" if b else f"{'':>18s}"
        print(f"{name:>14s} {cells}")
    hit = clipped["all"]
    if any(hit.values()):
        tot = {lab: (summary["arms"][lab]["p79"] if lab != pc.REF_LABEL
                     else summary["reference_ligand"])["n_fragments"]
               for lab in hit}
        print(f"\nrigid_violin: fragments drawn AT the {VIOLIN_FLOOR:g} A axis floor "
              f"(all in the 2-atom bin): "
              + ", ".join(f"{lab} {n} of {tot[lab]} ({100 * n / tot[lab]:.2f}%)"
                          for lab, n in hit.items() if n))
    for binning in BINNINGS:
        lean = {lab: b for lab, b in thin[("all", binning)].items() if b}
        if lean:
            print(f"{BINNING_STEM[binning]}: fewer than {MIN_BODY} fragments, so drawn as "
                  f"glyphs with no KDE body: "
                  + "; ".join(f"{lab} at {', '.join(b)}" for lab, b in lean.items()))
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
