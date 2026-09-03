#!/usr/bin/env python3
"""build_strain_growth.py — Figure 8 of results_drug_design.html.

Figure 7 shows how strain sits against ligand size; this one isolates how fast it *grows*,
which is the claim Figure 7's caption makes ("one to two orders of magnitude, steeper for
the weaker models") and cannot show directly because the curves sit at different levels
and cover different size ranges.

    Top    median strain divided by that method's own fitted value at the anchor size, log y,
           with the fitted log-linear trend drawn through it. Removing the vertical offset is
           what makes the slopes comparable by eye.
    Bottom the fitted slope itself, in decades of strain per 10 heavy atoms.

Endpoint ratios are deliberately NOT the summary: v2 stops at 43 heavy atoms where vanilla
reaches 49, so "how many orders from first to last size" partly measures how far a model's
size distribution reaches. The fitted slope is range-independent.

Rendered as inline SVG with the same per-method checkbox toggles as Figures 2/3 and 5-7 —
toggling a method hides its curve, its trend line and its slope bar together.

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/build_strain_growth.py
"""
import collections
import glob
import json
import math
import os
import statistics as st
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_vina_by_pocket import to_inline_svg, block, splice   # noqa: E402

EXPS = "/home1/irteam/VoxBind/voxbind/exps"
DOC = os.path.join(os.path.dirname(HERE), "results_drug_design.html")
V1 = os.path.join(EXPS, "voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350")
VANILLA = os.path.join(EXPS, "_vanilla_ep923/samples/full_eval_ep923")

# key, label, run root, colour — same series identity as Figures 5-7's toggle rows.
SERIES = [
    ("reference",  "Reference ligand", VANILLA, "#8e9baa"),
    ("targetdiff", "TargetDiff",       "/home1/irteam/base_drug/eval/targetdiff", "#3498db"),
    ("vanilla",    "VoxBind σ=0.9",    VANILLA, "#1abc9c"),
    ("ours",       "Ours · v1",        V1, "#2ecc71"),
    ("ours_v2",    "Ours · v2",        os.path.join(EXPS, "samples_reference_receptor_ed_ep350"), "#7fcf9d"),
]
MIN_N, XMAX = 20, 50      # same cut-offs as Figures 4 and 7 so the three line up
MIN_N_REF = 3             # 78 crystal ligands total: the model threshold would empty the curve
ANCHOR = 6                # smallest heavy-atom count every series reaches


def pocket_set():
    return sorted({os.path.basename(d) for d in glob.glob(os.path.join(V1, "target_*"))} - {"target_71"})


def median_by_size(key, root, targets):
    """{heavy atoms: median strain} over the 78 pockets, at the same cut-offs as Figure 7."""
    buckets = collections.defaultdict(list)
    for t in targets:
        path = os.path.join(root, t, "metrics.json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path, encoding="utf-8"))
        rows = [doc.get("reference")] if key == "reference" else (doc.get("samples") or [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            pc, n = r.get("posecheck") or {}, r.get("n_atoms")
            if isinstance(pc, dict) and pc.get("strain") is not None and n:
                buckets[n].append(float(pc["strain"]))
    floor = MIN_N_REF if key == "reference" else MIN_N
    return {a: st.median(v) for a, v in buckets.items() if len(v) >= floor and a <= XMAX}


def fit(medians):
    """Least-squares log10(strain) ~ atoms. Returns (slope per atom, intercept)."""
    xs = sorted(medians)
    ys = [math.log10(medians[a]) for a in xs]
    mx, my = st.mean(xs), st.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def render(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": "#1f2937", "axes.labelcolor": "#1f2937",
        "xtick.color": "#8a94a6", "ytick.color": "#8a94a6",
    })
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(12.4, 7.4), dpi=100,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=.42))
    fig.patch.set_facecolor("white")

    fits = {}
    for key, label, _, colour in SERIES:
        medians = data[key]
        if not medians:
            continue
        slope, intercept = fit(medians)
        fits[key] = slope * 10                       # decades per 10 heavy atoms
        anchor_value = 10 ** (intercept + slope * ANCHOR)
        xs = sorted(medians)
        (ln,) = ax.plot(xs, [medians[a] / anchor_value for a in xs], color=colour, lw=1.15,
                        marker="o", markersize=2.8, markerfacecolor=colour,
                        markeredgecolor="none", alpha=.75, zorder=3)
        ln.set_gid(f"sr-{key}-points")
        (tr,) = ax.plot(xs, [10 ** (intercept + slope * a) / anchor_value for a in xs],
                        color=colour, lw=2.4, zorder=4)
        tr.set_gid(f"sr-{key}-trend")

    ax.set_yscale("log")
    ax.axhline(1.0, color="#c4cad4", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("Number of ligand heavy atoms", fontsize=12.3, labelpad=8)
    ax.set_ylabel(f"median strain, relative to\nits own fitted value at {ANCHOR} heavy atoms",
                  fontsize=11.6, labelpad=10)
    ax.grid(color="#e6e9ef", lw=.85)
    ax.set_axisbelow(True)

    order = sorted(fits, key=lambda k: fits[k])
    colours = {k: c for k, _, _, c in SERIES}
    for i, key in enumerate(order):
        bar = bx.barh(i, fits[key], height=.62, color=colours[key], linewidth=0)[0]
        bar.set_gid(f"sr-{key}-slope")
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([dict((k, lab) for k, lab, _, _ in SERIES)[k] for k in order], fontsize=10.5)
    bx.set_xlabel("fitted growth rate  (decades of strain per 10 heavy atoms)",
                  fontsize=12.3, labelpad=8)
    bx.set_xlim(0, max(fits.values()) * 1.12)
    bx.xaxis.grid(True, color="#e6e9ef", lw=.85)
    bx.set_axisbelow(True)

    for a in (ax, bx):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#c4cad4")
    fig.tight_layout()
    return fig, fits


def main():
    targets = pocket_set()
    data = {key: median_by_size(key, root, targets) for key, _, root, _ in SERIES}
    fig, fits = render(data)
    svg = to_inline_svg(fig, "fig8")
    plt.close(fig)

    items = [(k, lab, c) for k, lab, _, c in SERIES]
    html = open(DOC, encoding="utf-8").read()
    html = splice(html, "fig8", block("fig8", svg, items))
    open(DOC, "w", encoding="utf-8").write(html)

    print(f"fig8: svg {len(svg) // 1024} KB, spliced into {DOC}")
    for key, label, _, _ in SERIES:
        if key in fits:
            xs = sorted(data[key])
            span = data[key][xs[-1]] / data[key][xs[0]]
            print(f"    {label:17s} slope {fits[key]:+.2f} decades/10 atoms   "
                  f"sizes {xs[0]}-{xs[-1]}  endpoint span {span:6.1f}x "
                  f"({math.log10(span):.2f} orders)")


if __name__ == "__main__":
    main()
