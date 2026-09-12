"""pose_common.py — the data and the house furniture shared by the pose-quality figures.

Two folders draw from the same per-target `metrics.json` trees and must agree exactly on
which pockets, which molecules and which arms they cover:

    fig-posebusters/   PoseBusters — dock-mode validity and the per-check breakdown
    fig-posecheck/     PoseCheck   — strain and steric clashes

Each folder carries the code that draws ITS OWN figures. What lives here is only what a
disagreement between the two would silently corrupt: the arm list, the pocket set, the
loader, the statistics, and the 3-line style furniture. Colour is next door in
method_colors.py for the same reason.

THE POCKET SET IS THE 79, NOT EACH ARM'S OWN COVERAGE. TargetDiff holds 100 pockets, but
Ours v1 only sampled the 79 with usable deposited electron density
(`frozenenc_probes/p79_targets.json`) and every arm covers those. So every figure is
like-for-like; each arm's own full-coverage number goes in the summary JSON under
"all_pockets" and is NOT comparable across arms. target_71 is in: it is unscoreable on the
docking side only, both pose metrics work on it.

TWO VARIANTS OF EVERY FIGURE. `core` is the comparison this section is making -- VoxBind
against VoxBind + Ours, with the crystal ligand as the benchmark, the same three series the
Vina 3-line figures carry. `all` adds every baseline the figure has data for. The split is
the one the Vina figures already make, not a subset picked after seeing the numbers, and
the variant is part of the FILENAME: two figures that differ in which arms they draw must
not be able to sit in a folder under one name.

STYLE. 260910/fig-vina-3line/build_vina_3line.py and 260910/fig-vina-per-atom: no panel titles, warm
near-black furniture (#514F52), left+bottom spines only, dotted mid-grey rules, live text
in the SVG and TrueType in the PDF.
"""
import collections
import json
import os
import statistics as st
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from method_colors import color, display                              # noqa: E402

E = "/home1/irteam/VoxBind/voxbind/exps"

# label, key, run root — drawn in this order, so our method lands on top.
#
# The five published baselines were sampled on another box and arrive as TargetDiff-format
# meta bundles; voxbind/scripts/tools/stage_baseline_samples.py writes them out as ordinary
# sample dirs under exps/baselines_pose/ and 86_pose_eval_baselines.sh scores them with the
# same metrics.py as everything else, against the same hard-linked pocket10 crops. They
# carry PoseBusters only -- their PoseCheck lives in fig-posecheck's whole-receptor
# figures, at a scope these crops are not comparable with -- so builders take the arms
# that actually hold the field they draw (see `arms_for`) rather than a hard-coded list.
ARMS = [
    ("AR",         "ar",         f"{E}/baselines_pose/ar"),
    ("Pocket2Mol", "pocket2mol", f"{E}/baselines_pose/pocket2mol"),
    ("DiffSBDD",   "diffsbdd",   f"{E}/baselines_pose/diffsbdd"),
    ("DecompDiff", "decompdiff", f"{E}/baselines_pose/decompdiff"),
    ("FuncBind",   "funcbind",   f"{E}/baselines_pose/funcbind"),
    ("TargetDiff", "targetdiff", "/home1/irteam/base_drug/eval/targetdiff"),
    ("VoxBind",    "vanilla",    f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    # These labels are DATA KEYS -- they head CSV columns and JSON objects -- so they stay
    # plain. What a legend shows comes from method_colors.display(), which is where vanilla
    # VoxBind picks up its sigma=0.9 subscript.
    ("CoDE",       "ours_v1",    f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]
# Ours v2 (exps/samples_reference_receptor_ed_ep350, 92 pockets) was dropped from this
# section on 2026-09-09. It is still evaluated -- both metrics are complete on it and
# 85_fill_pose_eval_4runs.sh still fills it -- it is just not one of the arms reported
# here. Re-adding it is one line.
# `core` is the comparison this section makes: our method against the model it modifies,
# with the crystal ligand as the benchmark. `all` is every arm the figure has data for.
CORE = ("vanilla", "ours_v1")
REF_LABEL = "Reference ligand"
REF_COLOR = color(REF_LABEL)
# The crystal ligand is the same molecule in every run that carries one, so any of OUR
# roots will do -- but it is resolved BY KEY, never by position. It used to be ARMS[2][2],
# which silently became DiffSBDD the moment the baselines were prepended to ARMS, and the
# staged baseline dirs have no reference block at all.
REF_ROOT = next(root for _, key, root in ARMS if key == "ours_v1")

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

INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
SOLID, DASH = (0, ()), (0, (4, 2.6))
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
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
}


def use_style():
    plt.rcParams.update(RC)


# ── load ─────────────────────────────────────────────────────────────────────────
def rows_of(target_dir, reference=False):
    """One record per molecule: heavy atoms, PoseCheck strain and clashes, its interaction
    fingerprint, PoseBusters validity and the checks it failed. `s=None` means the UFF
    relaxation did not converge -- kept as None rather than 0, because dropping a value is
    not the same as scoring it.

    `ifp` IS THE POSECHECK FINGERPRINT, NOT `m["interactions"]`. A molecule record carries
    two things under that name and they are unrelated: `m["interactions"]` is the docking
    side's contact/clash geometry, while `m["posecheck"]["interactions"]` is the ProLIF
    fingerprint -- {"Hydrophobic": 4, "VdWContact": 12, ...}. Only the second one is here.
    A TYPE THE MOLECULE DOES NOT MAKE IS ABSENT FROM THE DICT, NOT ZERO IN IT, so read it
    with .get(type, 0) and never with len() or a sum over its keys.
    """
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


def load_arms():
    """(per-arm {target: rows}, p79 rows per arm, the 79 crystal-ligand rows)."""
    data = {}
    for _, key, root in ARMS:
        names = sorted(d for d in os.listdir(root) if d.startswith("target_")
                       and os.path.isdir(os.path.join(root, d)))
        data[key] = {t: rows_of(os.path.join(root, t)) for t in names}
    p79_rows = {key: [r for t in P79 for r in data[key].get(t, [])] for _, key, _ in ARMS}
    refrows = [r for t in P79 for r in rows_of(os.path.join(REF_ROOT, t), reference=True)]
    return data, p79_rows, refrows


def arms_for(field, data, arms=None):
    """The arms that carry `field` across the WHOLE 79-pocket set, in ARMS order.

    An arm with no data for a metric must not become an empty curve and a legend row that
    says it was measured and lost -- it was never measured. The baselines hold PoseBusters
    and not PoseCheck, so the two folders end up with different arm lists from one ARMS.

    The test is per POCKET, not "has any value at all", and that is the point: a scoring
    run in progress leaves an arm with a handful of finished pockets, and pooling those
    would draw a complete-looking curve over 4% of the data. An arm is in only when every
    p79 pocket it holds has at least one scored molecule."""
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
    """(name, arms) for each figure variant, core first. Pass `field` and the loaded
    per-arm data to drop arms that do not carry that metric over the whole pocket set."""
    arms = ARMS if field is None else arms_for(field, data)
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


# ── statistics ───────────────────────────────────────────────────────────────────
def by_size(rows, field, key="n"):
    """{x: [values]}, where x is the row's `key` -- heavy atoms by default. `key` exists so
    a builder can resolve the same metric against another per-molecule integer (rotatable
    bonds, in build_strain_per_rotbond.py) through the same statistics and the same
    x_range/model_curve/reference_curve below, rather than forking them."""
    out = collections.defaultdict(list)
    for r in rows:
        if r[field] is not None and r.get(key) is not None:
            out[r[key]].append(r[field])
    return out


def bin_of(n):
    return min(int(np.searchsorted(EDGES, n, side="right")) - 1, len(BIN_LABELS) - 1)


def rate(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def model_curve(per, xs, f, win=0):
    """Value at each exact heavy-atom count, or over a centred +-win window.

    A median of a continuous quantity (strain, clashes) is stable at one count -- each arm
    has 100-300 molecules there -- so those curves are drawn unrolled, exactly as the Vina
    per-atom figure draws its medians. A PROPORTION IS NOT: at n=150 a per-count validity
    rate carries about +-4 pp of sampling noise, enough that arms separated by 2-10 pp
    cross each other repeatedly on nothing, so those curves pass win=2 (~750 molecules per
    point, about +-1.8 pp). The window is the only smoothing anywhere; nothing is
    interpolated and no point is invented."""
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
    it stays on the same scale as the models' per-count shares and all read against one
    axis."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= REF_WIN]) for a in xs]


# ── house furniture ──────────────────────────────────────────────────────────────
def furniture(ax, *, ylabel, xlabel=None, xlim=None, xloc=XTICK_STEP):
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
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)


def legend(target, handles, loc="upper right", ncol=1, fontsize=12.5, **kw):
    """Opaque white with a rule in the axis pen, so the dotted grid does not run through
    the labels but the box itself stays quiet.

    `target` is an Axes or a FIGURE. Both carry .legend() with this signature, and a key
    that has to sit outside the data area -- the check-failure bars fill every corner of
    their axes -- belongs to the figure, not to the axes it would otherwise cover."""
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
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH, label=REF_LABEL)] \
        if include_ref else []
    return h + [Line2D([], [], color=color(lab), lw=MODEL_LW, ls="-", label=display(lab))
                for lab, _, _ in arms]


def fit(fig, **kw):
    """tight_layout, then give back whatever it under-reserved: at this type size it can
    leave a rotated y label overrunning the figure edge even though it has just run, and
    the label is then silently SLICED OFF in the raster. get_tightbbox reports the overrun
    after the fact, so hand back exactly that much -- re-running tight_layout with a rect
    would re-apply `pad` on top of the correction and cost plot area."""
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


def save(fig, out_dir, stem, variant):
    """PNG to look at, SVG and PDF to place -- both vector, both with live text. The
    variant is part of the filename, never only of the content."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}_{variant}.{ext}"), facecolor="white")
    plt.close(fig)


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
