#!/usr/bin/env python3
"""build_mcp_finetune_size.py — does more MCP fine-tuning buy binding, or just size?

    mcp_finetune_size.{png,svg}   median Vina Dock per heavy-atom bin (top) over the
                                  molecule count each arm puts in that bin (bottom)
    mcp_finetune_size.csv         the numbers behind both panels

THE ARGUMENT. Vina rewards size, so an arm that draws smaller peptides is penalised and
an arm that draws larger ones is flattered, regardless of whether it binds better. Binning
by heavy-atom count removes that: inside a bin every arm draws the same-sized peptides, so
a gap that survives the bin is a binding difference. This is §4 of the 260827 note applied
to the MCP family, and it is the reason the pooled numbers in the tables cannot be read on
their own.

Colours come from ../method_colors.py, where the four arms are registered as an ordinal
ramp -- they are one model at four amounts of fine-tuning, not four methods.
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method_colors import color

FB = "/home1/irteam/funcbind/artifacts/reproduction/mcpp"
# Four steps of one brown ramp separate cleanly in the bars, but the top panel's lines
# cross, and at a crossing two adjacent steps of a lightness ladder are genuinely hard to
# tell apart. So each arm carries a second identity channel -- line style -- the same way
# this note's published baselines do. The two solid lines are the lightest and the darkest
# step, which is the pair that never needed the help.
ARMS = [
    ("vanilla",        "FuncBind vanilla",  f"{FB}/cmp10/_eval/vanilla/eval_docking_results.json",      "-"),
    ("fine-tune 3.17M", "ft_3.17M",         f"{FB}/cmp10/_eval/finetuned/eval_docking_results.json",    (0, (1, 1.6))),
    ("fine-tune 8.21M", "ft_8.21M",         f"{FB}/cmp10_r10/_eval/finetuned/eval_docking_results.json", (0, (5, 2.2))),
    ("fine-tune 26.1M", "ft_26.1M",         f"{FB}/cmp10_r14/_eval/finetuned/eval_docking_results.json", "-"),
]
BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 89)]
LBL = ["20–29", "30–39", "40–49", "50–59", "60+"]
MIN_N = 5          # fewer molecules than this in a bin is a data point, not a trend

D = {lab: json.load(open(p))["per_target"] for lab, _, p, _ in ARMS}
# Only targets every arm produced molecules for: comparing an arm against a target where
# another arm returned nothing is not a comparison.
COMMON = sorted(set.intersection(*[
    {e["target"] for e in per if e.get("vina_dock") is not None} for per in D.values()]))


def mols(lab):
    return [(m["n_atoms"], m["vina_dock"])
            for e in D[lab] if e["target"] in COMMON
            for m in (e.get("per_mol") or [])
            if m.get("vina_dock") is not None and m.get("n_atoms")]


M = {lab: mols(lab) for lab, _, _, _ in ARMS}
binned = {lab: [[d for a, d in v if lo <= a <= hi] for lo, hi in BINS] for lab, v in M.items()}

fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 7.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.13))
x = np.arange(len(BINS))

for lab, key, _, style in ARMS:
    col = color(key)
    ys = [np.median(b) if len(b) >= MIN_N else np.nan for b in binned[lab]]
    ax.plot(x, ys, color=col, lw=2.0, ls=style, marker="o", ms=8.5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.6, label=lab)
    # An arm that runs out of molecules early gets its label above the point; to the
    # right, the arms that continue would run through it.
    last = max(i for i, y in enumerate(ys) if not np.isnan(y))
    kw = dict(xytext=(10, -2), ha="left") if last == len(BINS) - 1 else dict(xytext=(0, 23), ha="center")
    ax.annotate(lab, (last, ys[last]), textcoords="offset points",
                fontsize=10.2, color=col, fontweight="600", va="center", **kw)

ax.set_ylabel("Vina Dock, median\n(kcal/mol)", fontsize=11.3)
ax.grid(color="#e6e9ef", lw=.85); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10.0, ncols=4, loc="lower center", bbox_to_anchor=(.5, 1.01))
ax.set_xlim(-.45, len(BINS) - .55)
ax.margins(y=.20)

w = .21
for i, (lab, key, _, _s) in enumerate(ARMS):
    counts = [len(b) for b in binned[lab]]
    bx.bar(x + (i - 1.5) * w, counts, width=w * .86, color=color(key), linewidth=0)
    for xi, c in enumerate(counts):
        if c:
            bx.annotate(str(c), (xi + (i - 1.5) * w, c), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.8, color="#7a8699")

bx.set_ylabel("molecules", fontsize=11.3)
bx.set_xlabel("heavy atoms", fontsize=11.3)
bx.set_xticks(x); bx.set_xticklabels(LBL)
bx.grid(axis="y", color="#e6e9ef", lw=.85); bx.set_axisbelow(True)

for a in (ax, bx):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("#c4cad4")
    a.tick_params(labelsize=10.2, colors="#5b6678")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_finetune_size")
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")

rows = ["arm,bin,n,dock_median"]
for lab, _, _, _ in ARMS:
    for l, b in zip(LBL, binned[lab]):
        med = f"{np.median(b):.3f}" if len(b) >= MIN_N else ""
        rows.append(f"{lab},{l},{len(b)},{med}")
open(f"{OUT}.csv", "w").write("\n".join(rows) + "\n")

print("common targets:", len(COMMON), COMMON)
for lab, _, _, _ in ARMS:
    ha = [a for a, _ in M[lab]]
    print(f"{lab:17s} n={len(M[lab]):4d} median heavy={np.median(ha):5.1f} "
          f"bins={[len(b) for b in binned[lab]]} "
          f"medians={[round(float(np.median(b)),2) if len(b)>=MIN_N else None for b in binned[lab]]}")
print("wrote", OUT + ".{png,svg,csv}")
