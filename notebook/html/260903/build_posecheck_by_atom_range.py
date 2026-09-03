"""build_posecheck_by_atom_range.py — Section 2 of 260903_meeting.html.

PoseCheck stratified by generated-molecule heavy-atom count, for the three methods the
meeting cares about: TargetDiff, vanilla VoxBind, Ours v1.

Why stratify at all: strain and clashes both rise with molecule size (Spearman +0.52 and
+0.28 on this data), and the three methods draw different size mixes -- Ours v1 puts 29%
of its molecules above 30 heavy atoms against vanilla's 24% and TargetDiff's 19%. A
pooled number therefore partly reports the size mix rather than pose quality. Inside a
bin the methods are drawing comparably-sized molecules, so the comparison is like for
like.

Emits, over the same 78 pockets used everywhere else in this project:
  * posecheck_by_atom_range.json   the numbers for the 2.1 table
  * strain_ecdf_<bin>.{png,svg}    cumulative probability of strain, one per bin
  * clash_violin_<bin>.{png,svg}   steric-clash distribution, one per bin

Statistic notes. Strain is drawn as an ECDF on a log axis because it is heavy-tailed:
4-7% of molecules per method fail UFF relaxation and land at 1e4-1e10, which makes the
arithmetic mean report the failure rate rather than the strain. Clashes are small
integers, so a violin plot (with the median and quartiles marked) shows the whole
distribution rather than hiding it behind a mean.

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_posecheck_by_atom_range.py
"""
import collections
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E = "/home1/irteam/VoxBind/voxbind/exps"
HERE = os.path.dirname(os.path.abspath(__file__))

# Palette matches results_drug_design.html: baselines cool, ours green.
METHODS = [
    ("TargetDiff",     "targetdiff", "/home1/irteam/base_drug/eval/targetdiff",           "#3a7bd5", 1.9),
    ("VoxBind σ=0.9",  "vanilla",    f"{E}/_vanilla_ep923/samples/full_eval_ep923",       "#e08a1e", 2.4),
    ("Ours · v1",      "ours_v1",    f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350", "#1e7a4d", 3.0),
]
REF_ROOT, REF_COLOR = METHODS[2][2], "#2b3440"

P78 = [int(t.split("_")[-1]) for t in
       json.load(open(f"{E}/frozenenc_probes/p78_targets.json"))]

# Edges from the meeting doc's own placeholder, kept because every bin holds >=900
# molecules per method, which is enough for a stable ECDF and violin.
EDGES = [0, 16, 21, 26, 31, 10 ** 6]
LABELS = ["≤15", "16–20", "21–25", "26–30", ">30"]
SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30"]
BG = "#fbfaf7"


def bin_of(n):
    return min(np.searchsorted(EDGES, n, side="right") - 1, len(LABELS) - 1)


def pull(root, reference=False):
    """(bin index) -> {'strain': [...], 'clash': [...]} over the 78 pockets."""
    out = collections.defaultdict(lambda: {"strain": [], "clash": []})
    for i in P78:
        path = f"{root}/target_{i:02d}/metrics.json"
        if not os.path.exists(path):
            continue
        j = json.load(open(path))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m:
                continue
            pc = m.get("posecheck") or {}
            n = m.get("n_atoms")
            if not n:
                continue
            b = bin_of(n)
            s = pc.get("strain")
            # A failed relaxation writes null, not 0 — dropping it is not the same as
            # scoring it, so the per-bin n is reported alongside every number.
            if s is not None and np.isfinite(s):
                out[b]["strain"].append(float(s))
            c = pc.get("clashes")
            if c is not None:
                out[b]["clash"].append(float(c))
    return out


DATA = {key: pull(root) for _, key, root, _, _ in METHODS}
REF = pull(REF_ROOT, reference=True)


def style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#aeb6c4")
    ax.grid(color="#e6e9ef", lw=0.85)
    ax.set_axisbelow(True)


XFLOOR, XTOP = 1e-2, 3e3

for b, (lab, slug) in enumerate(zip(LABELS, SLUGS)):
    # ── strain ECDF ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fig.patch.set_facecolor(BG)
    style(ax)
    for name, key, _, col, lw in METHODS:
        v = np.array(DATA[key][b]["strain"])
        if len(v) < 20:
            continue
        x = np.sort(np.clip(v, XFLOOR, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=lw,
                label=f"{name}  ·  median {st.median(v):.0f}", solid_capstyle="round")
    r = np.array(REF[b]["strain"])
    if len(r) >= 3:
        x = np.sort(np.clip(r, XFLOOR, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=REF_COLOR, lw=1.8, ls="--",
                label=f"Reference  ·  median {st.median(r):.0f}  (n={len(r)})")
    ax.set_xscale("log")
    ax.set_xlim(XFLOOR, XTOP)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("UFF strain energy (kcal mol⁻¹) · log scale", fontsize=11.5)
    ax.set_ylabel("cumulative probability", fontsize=11.5)
    ax.set_title(f"Strain energy — {lab} heavy atoms", fontsize=14, fontweight="620",
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    n_txt = " · ".join(f"{k}: n={len(DATA[k][b]['strain']):,}" for _, k, _, _, _ in METHODS)
    ax.text(0, -0.155, n_txt + f" · x clipped to {XTOP:.0e}", transform=ax.transAxes,
            fontsize=9, color="#7a8699")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{HERE}/strain_ecdf_{slug}.{ext}", dpi=170, facecolor=BG,
                    bbox_inches="tight")
    plt.close(fig)

    # ── clash violin ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.patch.set_facecolor(BG)
    style(ax)
    series, cols, ticks = [], [], []
    for name, key, _, col, _ in METHODS:
        v = DATA[key][b]["clash"]
        if len(v) < 20:
            continue
        series.append(v); cols.append(col); ticks.append(name.replace(" · ", "\n").replace(" σ", "\nσ"))
    rv = REF[b]["clash"]
    if len(rv) >= 3:
        series.append(rv); cols.append(REF_COLOR); ticks.append(f"Reference\n(n={len(rv)})")
    # Log-like y, done properly. TargetDiff's tail reaches 148 while vanilla and Ours sit
    # at 4-8, so a linear axis squashes the range that matters. A plain log axis is not an
    # option: 1.6-5.5% of poses have ZERO clashes and log(0) is undefined. Setting
    # yscale="symlog" would also mis-shape the violin, because matplotlib fits the KDE in
    # data space and only the display is warped. So the KDE is fitted on log10(x+1) and the
    # ticks are relabelled with the original counts -- an undistorted density on a log-like
    # axis, with zero kept as a real value at the bottom.
    tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
    parts = ax.violinplot([tf(v) for v in series], showextrema=False, widths=0.82)
    for body, col in zip(parts["bodies"], cols):
        body.set_facecolor(col); body.set_alpha(0.55); body.set_edgecolor(col); body.set_linewidth(1.2)
    for i, v in enumerate(series, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
        ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
    TICKS = [0, 1, 2, 5, 10, 20, 50, 100]
    hi = max(max(v) for v in series)
    TICKS = [t for t in TICKS if t <= hi * 1.6]
    ax.set_yticks(tf(TICKS)); ax.set_yticklabels([str(t) for t in TICKS])
    ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.16)
    for i, v in enumerate(series, start=1):
        ax.text(i, ax.get_ylim()[1], f"mean {np.mean(v):.2f}\nmed {np.median(v):.0f}",
                ha="center", va="top", fontsize=9.5, color="#3a4352", linespacing=1.35)
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels(ticks, fontsize=10.5)
    ax.set_ylabel("steric clashes per pose  ·  log-spaced", fontsize=11.5)
    ax.set_title(f"Steric clashes — {lab} heavy atoms", fontsize=14, fontweight="620",
                 loc="left", pad=10)
    ax.text(0, -0.16, "thick bar = IQR · white dot = median · axis is log10(clashes+1), "
                      "relabelled with real counts · zero is not the target: crystal poses clash too",
            transform=ax.transAxes, fontsize=9, color="#7a8699")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{HERE}/clash_violin_{slug}.{ext}", dpi=170, facecolor=BG,
                    bbox_inches="tight")
    plt.close(fig)

# ── numbers for the 2.1 table ─────────────────────────────────────────────────
summary = {"bins": LABELS, "edges": EDGES[:-1] + ["inf"], "n_pockets": len(P78), "methods": {}}
for name, key, _, _, _ in METHODS:
    rows = []
    for b in range(len(LABELS)):
        s_, c_ = DATA[key][b]["strain"], DATA[key][b]["clash"]
        rows.append({
            "n": len(s_),
            "strain_median": round(st.median(s_), 1) if s_ else None,
            "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
            "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
            "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
            "clash_median": round(float(np.median(c_)), 1) if c_ else None,
        })
    summary["methods"][name] = rows
ref_rows = []
for b in range(len(LABELS)):
    s_, c_ = REF[b]["strain"], REF[b]["clash"]
    ref_rows.append({"n": len(s_),
                     "strain_median": round(st.median(s_), 1) if s_ else None,
                     "clash_mean": round(float(np.mean(c_)), 2) if c_ else None})
summary["methods"]["Reference ligand"] = ref_rows
json.dump(summary, open(f"{HERE}/posecheck_by_atom_range.json", "w"), indent=1)

print(f"{'bin':7s} {'method':15s} {'n':>6s} {'strain med':>11s} {'IQR':>17s} {'clash mean':>11s} {'med':>5s}")
for b, lab in enumerate(LABELS):
    for name, key, _, _, _ in METHODS:
        r = summary["methods"][name][b]
        print(f"{lab:7s} {name:15s} {r['n']:6d} {r['strain_median']:11.1f} "
              f"{r['strain_q25']:7.1f}–{r['strain_q75']:<9.1f} {r['clash_mean']:11.2f} {r['clash_median']:5.1f}")
    rr = summary["methods"]["Reference ligand"][b]
    print(f"{'':7s} {'Reference':15s} {rr['n']:6d} "
          f"{rr['strain_median'] if rr['strain_median'] is not None else float('nan'):11.1f} "
          f"{'':17s} {rr['clash_mean'] if rr['clash_mean'] is not None else float('nan'):11.2f}")
    print()
print("wrote posecheck_by_atom_range.json + 10 figures")
