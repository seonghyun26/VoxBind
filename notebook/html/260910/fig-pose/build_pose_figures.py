#!/usr/bin/env python3
"""build_pose_figures.py — pose quality (PoseBusters + PoseCheck) in the 260903 3-line style.

    pb_valid_per_atom.{png,svg,pdf}     validity against ligand size, + the size mix
    pb_check_failures.{png,svg,pdf}     which checks fail, per method
    posecheck_per_atom.{png,svg,pdf}    strain and clashes against ligand size
    strain_clash_ecdf_pair.{png,svg,pdf} the two PoseCheck distributions, pooled
    pose_summary.json                   coverage + pooled numbers (p79 and all_pockets)
    pose_by_atom_range.{json,csv}       the per-bin numbers
    pose_check_failures.json            per-check failure counts and rates
    pose_per_molecule_<arm>.json        per-molecule export, for merging elsewhere

WHAT IS NEW. 260903 reported PoseCheck for three arms. PoseBusters had only ever run on
33/79 of Ours v1 and 53/100 of vanilla, and never on TargetDiff or Ours v2;
`voxbind/scripts/85_fill_pose_eval_4runs.sh` finished it on 2026-09-09, so this is the
first build where all four arms carry BOTH metrics on every pocket, plus the per-check
breakdown that says why a validity rate is what it is.

THE POCKET SET IS THE 79, NOT EACH ARM'S OWN COVERAGE. TargetDiff and vanilla hold 100
pockets and Ours v2 holds 92, but Ours v1 only sampled the 79 with usable deposited
electron density (`frozenenc_probes/p79_targets.json`), and all four cover those. Every
figure is therefore like-for-like; each arm's own full-coverage number is in
pose_summary.json under "all_pockets" and is NOT comparable across arms. target_71 is in:
it is unscoreable on the docking side only, both pose metrics work on it.

SIZE IS THE CONFOUND, SO SIZE IS THE X AXIS. Every pose check gets harder as the molecule
grows -- more rings to pucker, more angles to strain, more atoms to reach the protein --
and the arms draw different size mixes (24.9 heavy atoms for Ours v1 against TargetDiff's
22.2). A pooled rate is therefore partly a report of the size mix, the same trap the Vina
numbers have, which is why the headline figure is per-atom with the size distribution
drawn underneath it rather than a single pooled bar. pose_summary.json also carries
`pb_valid_rate_size_standardized`: each arm's validity within every 1-atom stratum,
re-weighted by one common size distribution.

STYLE. 260903/build_vina_3line.py and 260910/fig-vina-per-atom: no panel titles, warm
near-black furniture (#514F52), left+bottom spines only, dotted mid-grey rules, live text
in the SVG and TrueType in the PDF. Colour is the identity of the arm. The house pair --
VoxBind sand #F5B27E, VoxBind + Ours periwinkle #8291E8, reference grey #9aa0a6 -- is kept
exactly, so an arm reads the same here as it does in the Vina figures. The two arms that
house palette has no entry for take a hue each rather than a shade of an existing one:
TargetDiff a muted teal, because it is not a VoxBind model and should not read as a shade
of one, and Ours v2 a light periwinkle, because it IS the same family as Ours v1 and
should read as one.

    /opt/conda/envs/voxbind/bin/python notebook/html/260910/fig-pose/build_pose_figures.py
"""
import collections
import csv
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"

# label, key, run root, colour
ARMS = [
    ("TargetDiff",    "targetdiff", "/home1/irteam/base_drug/eval/targetdiff",                             "#5FA8A0"),
    ("VoxBind",       "vanilla",    f"{E}/_vanilla_ep923/samples/full_eval_ep923",                          "#F5B27E"),
    ("VoxBind + Ours", "ours_v1",   f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350", "#8291E8"),
    ("Ours v2",       "ours_v2",    f"{E}/samples_reference_receptor_ed_ep350",                             "#AEB7EE"),
]
REF_LABEL, REF_COLOR = "Reference ligand", "#9aa0a6"
REF_ROOT = ARMS[2][2]        # any arm carries the same crystal ligand per pocket

P79 = json.load(open(f"{E}/frozenenc_probes/p79_targets.json"))
EDGES = [0, 16, 21, 26, 31, 10 ** 6]
LABELS = ["≤15", "16–20", "21–25", "26–30", ">30"]

# A heavy-atom count is plotted only where EVERY arm has this many molecules, so all four
# curves start and stop together and none carries a tail the others cannot answer.
MIN_N = 25
# The reference window: +-REF_WIN atoms, kept where it pools at least MIN_REF ligands.
# There is one crystal ligand per pocket -- 79 over ~40 sizes -- so a per-count reference
# curve would be noise, and its share would be a picket fence of 5%-tall spikes.
REF_WIN, MIN_REF = 4, 12
# Smoothing for the VALIDITY curves only -- see model_curve. Strain and clash medians are
# drawn unrolled.
VALID_WIN = 2
# A PoseBusters check is drawn only where some method fails it at least this often.
MIN_FAIL_PCT = 0.5

INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
SOLID, DASH = (0, ()), (0, (4, 2.6))
MODEL_LW, REF_LW = 2.35, 1.5
AXIS_LW, GRID_LW = 1.35, 1.1
DIST_LW, DIST_FILL = 1.7, 0.16

# GEOMETRY, in inches, off the 3-line base figure of 7.25 x 2.77.
WIDE, TALL = 1.05, 1.155
FIG_W = 7.25 * WIDE
STACK_H = 2.77 * TALL + 1.62          # a full panel plus the distribution strip
HEIGHT_RATIOS = (2.62, 1.05)
# The gap between stacked panels goes through tight_layout's h_pad, NOT gridspec_kw's
# hspace: an explicit hspace makes tight_layout call the axes "not compatible", warn, and
# leave ~12% of the figure as dead white at the top.
H_PAD = 0.2
XTICK_STEP = 5

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
}


# ── load ─────────────────────────────────────────────────────────────────────────
def rows_of(target_dir, reference=False):
    """One record per molecule. `s=None` means the UFF relaxation did not converge --
    kept as None rather than 0, because dropping a value is not the same as scoring it."""
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
            "s": float(s) if isinstance(s, (int, float)) and np.isfinite(s) else None,
            "c": float(pc["clashes"]) if pc.get("clashes") is not None else None,
            "v": pb["valid"] if isinstance(pb.get("valid"), bool) else None,
            "f": sorted(k for k, ok in (pb.get("checks") or {}).items() if ok is False),
        })
    return out


def load(root):
    names = sorted(d for d in os.listdir(root) if d.startswith("target_")
                   and os.path.isdir(os.path.join(root, d)))
    return {t: rows_of(os.path.join(root, t)) for t in names}


DATA = {key: load(root) for _, key, root, _ in ARMS}
REF = {t: rows_of(os.path.join(REF_ROOT, t), reference=True) for t in P79}
REFROWS = [r for t in P79 for r in REF[t]]


def pool(key, targets=P79):
    return [r for t in targets for r in DATA[key].get(t, [])]


P79_ROWS = {key: pool(key) for _, key, _, _ in ARMS}


def by_size(rows, field):
    out = collections.defaultdict(list)
    for r in rows:
        if r[field] is not None:
            out[r["n"]].append(r[field])
    return out


def bin_of(n):
    return min(int(np.searchsorted(EDGES, n, side="right")) - 1, len(LABELS) - 1)


def rate(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


# Direct standardization: each arm's rate WITHIN every 1-heavy-atom stratum, re-weighted
# by one common size distribution (all four arms pooled), so what is left is pose quality
# at matched size. Strata where the arm holds <10 molecules are dropped, not extrapolated.
STD_W = collections.Counter(r["n"] for key in P79_ROWS for r in P79_ROWS[key]
                            if r["v"] is not None)


def std_rate(rows):
    per = by_size(rows, "v")
    num = den = 0.0
    for n, w in STD_W.items():
        if len(per.get(n, ())) >= 10:
            num += w * float(np.mean(per[n])); den += w
    return 100 * num / den if den else float("nan")


def model_curve(per, xs, f, win=0):
    """Value at each exact heavy-atom count, or over a centred +-win window.

    A median of a continuous quantity (strain, clashes) is stable at one count -- each arm
    has 100-300 molecules there -- so those curves are drawn unrolled, exactly as the Vina
    per-atom figure draws its medians. VALIDITY IS NOT: it is a proportion, and at n=150 a
    per-count rate carries about +-4 pp of sampling noise, enough that four arms separated
    by 2-10 pp cross each other repeatedly on nothing. Its curves use win=2, which pools
    ~750 molecules per point and takes that to about +-1.8 pp. The window is the only
    smoothing anywhere here; nothing is interpolated and no point is invented."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= win for v in vals] if win \
            else per.get(a, [])
        out.append(f(p) if p else None)
    return out


def reference_curve(per, xs, f):
    """A centred +-REF_WIN window; None where it is too thin to mean anything, which
    leaves a gap in the line rather than an invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(p) if len(p) >= MIN_REF else None)
    return out


def share(per, xs):
    total = sum(len(v) for v in per.values())
    return [100.0 * len(per.get(a, ())) / total for a in xs] if total else [], total


def rolled(values, xs):
    """The reference's shares under the same +-REF_WIN window -- averaged, not summed, so
    it stays on the same scale as the models' per-count shares and all five read against
    one axis."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= REF_WIN]) for a in xs]


# ── house furniture ──────────────────────────────────────────────────────────────
def furniture(ax, *, ylabel, xlabel=None, xlim=None, xloc=XTICK_STEP):
    ax.set_ylabel(ylabel, fontsize=15.5, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15.5, labelpad=9)
    if xlim:
        ax.set_xlim(*xlim)
    if xloc:
        ax.xaxis.set_major_locator(MultipleLocator(xloc))
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)


def legend(ax, handles, loc="upper right", ncol=1, fontsize=12.5):
    """Opaque white with a rule in the axis pen, so the dotted grid does not run through
    the labels but the box itself stays quiet."""
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=True, fontsize=fontsize,
                    handlelength=1.9, handletextpad=0.55, labelspacing=0.3,
                    borderpad=0.4, borderaxespad=0.39, facecolor="white",
                    edgecolor=LEGEND_EDGE, framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def arm_handles(include_ref=True):
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH, label=REF_LABEL)] \
        if include_ref else []
    return h + [Line2D([], [], color=c, lw=MODEL_LW, ls="-", label=lab)
                for lab, _, _, c in ARMS]


def fit(fig, **kw):
    """tight_layout, then give back whatever it under-reserved: at this type size it can
    leave a rotated y label overrunning the figure edge, and the label is then silently
    SLICED OFF in the raster. get_tightbbox reports the overrun after the fact."""
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


def save(fig, stem):
    """PNG to look at, SVG and PDF to place -- both vector, both with live text."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), facecolor="white")
    plt.close(fig)


def size_distribution(ax, xs, per_arm, ref_per):
    """The bottom strip: where each set puts its molecules, as a share of its own -- which
    is what makes the panel above it trustworthy or not, and is itself the finding, since
    the arms differ in the sizes they generate as much as in per-size pose quality."""
    for (lab, key, _, col) in ARMS:
        pct, _ = share(per_arm[key], xs)
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, step="mid", color=col, alpha=DIST_FILL, lw=0, zorder=2)
    raw, _ = share(ref_per, xs)
    ax.plot(xs, rolled(raw, xs), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    ax.set_ylim(bottom=0)


# ── figure 1: PoseBusters validity per heavy-atom count ──────────────────────────
def fig_pb_per_atom():
    per = {key: by_size(P79_ROWS[key], "v") for _, key, _, _ in ARMS}
    ref_per = by_size(REFROWS, "v")
    xs = [a for a in sorted(set.intersection(*(set(p) for p in per.values())))
          if all(len(per[k][a]) >= MIN_N for k in per)]
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    y = reference_curve(ref_per, xs, lambda v: 100 * float(np.mean(v)))
    top.plot(xs, y, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    for lab, key, _, col in ARMS:
        top.plot(xs, model_curve(per[key], xs, lambda v: 100 * float(np.mean(v)),
                                 win=VALID_WIN),
                 color=col, lw=MODEL_LW, zorder=5)
    furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    legend(top, arm_handles(), loc="lower left")

    size_distribution(bot, xs, per, ref_per)
    furniture(bot, ylabel="% of ligands", xlabel="Number of heavy atoms in ligand",
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, "pb_valid_per_atom")
    return xs


# ── figure 2: which checks fail ──────────────────────────────────────────────────
def fig_check_failures():
    fails = {}
    for lab, key, _, _ in ARMS:
        rows = [r for r in P79_ROWS[key] if r["v"] is not None]
        cnt = collections.Counter(c for r in rows for c in r["f"])
        fails[key] = {"n_mols": len(rows), "counts": dict(cnt),
                      "rates": {k: 100 * v / len(rows) for k, v in cnt.items()}}
    rrows = [r for r in REFROWS if r["v"] is not None]
    cnt = collections.Counter(c for r in rrows for c in r["f"])
    fails["reference"] = {"n_mols": len(rrows), "counts": dict(cnt),
                          "rates": {k: 100 * v / max(1, len(rrows)) for k, v in cnt.items()}}

    # A check no method fails above MIN_FAIL_PCT is a row of white space: it says only
    # that PoseBusters ran it. The full counts stay in pose_check_failures.json.
    names = sorted({k for g in fails.values() for k in g["counts"]
                    if max(h["rates"].get(k, 0) for h in fails.values()) >= MIN_FAIL_PCT},
                   key=lambda k: -max(g["rates"].get(k, 0) for g in fails.values()))
    fig, ax = plt.subplots(figsize=(FIG_W, 0.52 * len(names) + 1.5), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(len(names))[::-1]
    h = 0.19
    for i, (lab, key, _, col) in enumerate(ARMS):
        ax.barh(ys + (i - (len(ARMS) - 1) / 2) * h,
                [fails[key]["rates"].get(k, 0) for k in names],
                height=h, color=col, edgecolor=col, lw=0.8, zorder=3)
    ax.plot([fails["reference"]["rates"].get(k, 0) for k in names], ys, "|",
            color=REF_COLOR, ms=20, mew=REF_LW + 0.4, zorder=5)
    furniture(ax, ylabel=None, xlabel="Molecules failing the check (%)", xloc=5)
    ax.set_ylabel("")
    ax.grid(False, axis="y")
    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("_", " ") for k in names], fontsize=13)
    ax.set_ylim(-0.6, len(names) - 0.4)
    handles = [Line2D([], [], color=REF_COLOR, lw=0, marker="|", ms=13,
                      mew=REF_LW + 0.4, label=REF_LABEL)] + \
              [Patch(facecolor=c, edgecolor=c, label=lab) for lab, _, _, c in ARMS]
    legend(ax, handles, loc="lower right")
    fit(fig, pad=0.5)
    save(fig, "pb_check_failures")
    return fails


# ── figure 3: PoseCheck strain and clashes per heavy-atom count ──────────────────
def fig_posecheck_per_atom(xs):
    s_per = {key: by_size(P79_ROWS[key], "s") for _, key, _, _ in ARMS}
    c_per = {key: by_size(P79_ROWS[key], "c") for _, key, _, _ in ARMS}
    s_ref, c_ref = by_size(REFROWS, "s"), by_size(REFROWS, "c")
    med = lambda v: float(np.median(v))

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0)})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    # Strain is heavy-tailed enough that its mean reports the UFF failure rate rather than
    # the strain (vanilla: mean 2.26e8, median 62), so the median is the only honest
    # central value here. Log y because the four arms span 30 to 600 kcal/mol.
    top.plot(xs, reference_curve(s_ref, xs, med), color=REF_COLOR, lw=REF_LW, ls=DASH,
             zorder=4)
    for lab, key, _, col in ARMS:
        top.plot(xs, model_curve(s_per[key], xs, med), color=col, lw=MODEL_LW, zorder=5)
    top.set_yscale("log")
    furniture(top, ylabel="Strain median\n(kcal mol⁻¹)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    # The strain panel has no free corner: the curves climb left-to-right across it and
    # the two steep ones cross whatever the upper left leaves. The clash panel below has a
    # large empty upper left (no median is above 7 before x=20), so the figure's one key
    # goes there and serves both panels.

    bot.plot(xs, reference_curve(c_ref, xs, med), color=REF_COLOR, lw=REF_LW, ls=DASH,
             zorder=4)
    for lab, key, _, col in ARMS:
        bot.plot(xs, model_curve(c_per[key], xs, med), color=col, lw=MODEL_LW, zorder=5)
    furniture(bot, ylabel="Clash median", xlabel="Number of heavy atoms in ligand",
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    bot.set_ylim(bottom=0)
    legend(bot, arm_handles(), loc="upper left", fontsize=11.5)
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, "posecheck_per_atom")


# ── figure 4: the two PoseCheck distributions, pooled ────────────────────────────
def fig_ecdf_pair():
    """Two panels on one shared y, the 3-line pair layout. ECDFs rather than violins: the
    question is what share of a method's poses sit under a given strain or clash count,
    and a cumulative curve answers it directly at every threshold instead of at three
    quartiles."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(FIG_W * 1.52, 2.77 * TALL), dpi=220,
                                      sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")

    def ecdf(ax, field, clip):
        for lab, key, _, col in ARMS:
            v = np.array([r[field] for r in P79_ROWS[key] if r[field] is not None])
            x = np.sort(np.clip(v, clip, None))
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=MODEL_LW, zorder=5)
        v = np.array([r[field] for r in REFROWS if r[field] is not None])
        x = np.sort(np.clip(v, clip, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=REF_COLOR, lw=REF_LW, ls=DASH,
                zorder=4)

    ecdf(left, "s", 1e-2)
    left.set_xscale("log")
    left.set_xlim(1e-1, 3e3)
    furniture(left, ylabel="Cumulative share",
              xlabel="PoseCheck strain (kcal mol⁻¹)", xloc=None)
    left.set_ylim(0, 1.0)
    legend(left, arm_handles(), loc="upper left", fontsize=11.5)

    ecdf(right, "c", 0)
    furniture(right, ylabel=None, xlabel="PoseCheck steric clashes", xloc=5)
    right.set_ylabel("")
    right.set_xlim(0, 30)
    fit(fig, pad=0.5)
    save(fig, "strain_clash_ecdf_pair")


# ── data exports ─────────────────────────────────────────────────────────────────
def block(rows):
    s = [r["s"] for r in rows if r["s"] is not None]
    c = [r["c"] for r in rows if r["c"] is not None]
    v = [r["v"] for r in rows if r["v"] is not None]
    std = std_rate(rows)
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_strain": len(s),
        "strain_median": round(st.median(s), 1) if s else None,
        "strain_q25": round(float(np.percentile(s, 25)), 1) if s else None,
        "strain_q75": round(float(np.percentile(s, 75)), 1) if s else None,
        "clash_mean": round(float(np.mean(c)), 2) if c else None,
        "clash_median": round(float(np.median(c)), 1) if c else None,
        "n_posebusters": len(v),
        "pb_valid_rate": round(rate(v) / 100, 4) if v else None,
        "pb_valid_rate_size_standardized": round(std / 100, 4) if np.isfinite(std) else None,
    }


def exports(fails):
    summary = {
        "built": "2026-09-09",
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(P79), "targets": P79},
        "note": ("p79 is the like-for-like set every arm covers; all_pockets is each arm's "
                 "own coverage and is NOT comparable across arms. Strain is the posecheck "
                 "1.3.1 definition, not the VoxBind paper's."),
        "arms": {}, "reference_ligand": block(REFROWS),
    }
    for lab, key, root, _ in ARMS:
        summary["arms"][lab] = {
            "root": root, "key": key,
            "pockets_all": len(DATA[key]),
            "pockets_p79": len([t for t in P79 if t in DATA[key]]),
            "p79": block(P79_ROWS[key]),
            "all_pockets": block(pool(key, sorted(DATA[key]))),
        }
    json.dump(summary, open(os.path.join(HERE, "pose_summary.json"), "w"),
              indent=1, ensure_ascii=False)

    by_bin = {"bins": LABELS, "edges": EDGES[:-1] + ["inf"], "n_pockets": len(P79), "arms": {}}
    for lab, key, _, _ in ARMS:
        by_bin["arms"][lab] = [block([r for r in P79_ROWS[key] if bin_of(r["n"]) == b])
                               for b in range(len(LABELS))]
    by_bin["arms"][REF_LABEL] = [block([r for r in REFROWS if bin_of(r["n"]) == b])
                                 for b in range(len(LABELS))]
    json.dump(by_bin, open(os.path.join(HERE, "pose_by_atom_range.json"), "w"),
              indent=1, ensure_ascii=False)

    cols = ["n_molecules", "atoms_mean", "strain_median", "strain_q25", "strain_q75",
            "clash_mean", "clash_median", "n_posebusters", "pb_valid_rate"]
    with open(os.path.join(HERE, "pose_by_atom_range.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "bin"] + cols)
        for arm, rowset in by_bin["arms"].items():
            for lab, r in zip(LABELS, rowset):
                wr.writerow([arm, lab] + [r[c] for c in cols])

    json.dump({"n_pockets": len(P79),
               "arms": {lab: fails[key] for lab, key, _, _ in ARMS},
               "reference_ligand": fails["reference"]},
              open(os.path.join(HERE, "pose_check_failures.json"), "w"),
              indent=1, ensure_ascii=False)

    for lab, key, root, _ in ARMS:
        mols = [dict(r, t=t) for t in sorted(DATA[key]) for r in DATA[key][t]]
        json.dump({"arm": lab, "root": root, "n_pockets": len(DATA[key]),
                   "n_molecules": len(mols), "p79_targets": P79,
                   "fields": {"t": "target dir", "n": "heavy atoms",
                              "s": "PoseCheck strain (null = UFF relaxation did not converge)",
                              "c": "PoseCheck clashes", "v": "PoseBusters valid",
                              "f": "failed PoseBusters checks"},
                   "molecules": mols},
                  open(os.path.join(HERE, f"pose_per_molecule_{key}.json"), "w"),
                  ensure_ascii=False)
    return summary, by_bin


def main():
    plt.rcParams.update(RC)
    xs = fig_pb_per_atom()
    fails = fig_check_failures()
    fig_posecheck_per_atom(xs)
    fig_ecdf_pair()
    summary, by_bin = exports(fails)

    print(f"79-pocket set · {len(P79)} pockets · per-atom x range {xs[0]}-{xs[-1]} "
          f"(counts where every arm has >={MIN_N} molecules)\n")
    print(f"{'arm':16s} {'atoms':>6s} {'clash med':>10s} {'strain med':>11s} "
          f"{'PB-valid':>9s} {'size-std':>9s} {'mols':>7s}")
    for lab, key, _, _ in ARMS:
        r = summary["arms"][lab]["p79"]
        print(f"{lab:16s} {r['atoms_mean']:6.1f} {r['clash_median']:10.1f} "
              f"{r['strain_median']:11.1f} {100 * r['pb_valid_rate']:8.1f}% "
              f"{100 * r['pb_valid_rate_size_standardized']:8.1f}% {r['n_molecules']:7d}")
    r = summary["reference_ligand"]
    print(f"{REF_LABEL:16s} {r['atoms_mean']:6.1f} {r['clash_median']:10.1f} "
          f"{r['strain_median']:11.1f} {100 * r['pb_valid_rate']:8.1f}% "
          f"{'—':>9s} {r['n_molecules']:7d}")
    print(f"\nPB-valid by bin ({' · '.join(LABELS)}):")
    for lab, key, _, _ in ARMS:
        print(f"  {lab:16s} " + "  ".join(
            f"{100 * b['pb_valid_rate']:5.1f}%" if b["pb_valid_rate"] is not None else "    -"
            for b in by_bin["arms"][lab]))
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
