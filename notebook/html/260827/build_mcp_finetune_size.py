"""build_mcp_finetune_size.py — Figure for §5 of 260827_meeting.html.

The MCP receptor-ED fine-tune's apparent gains are a size effect, and this is the
figure that shows it. Top: median Vina Dock inside each heavy-atom bin, where all
three arms draw the same-sized peptides, so a surviving gap is not size. Bottom:
how many molecules each arm puts in that bin -- which is what actually moved
between the 3.17M and 8.21M checkpoints.

Same construction as §4's build_vina_by_size_curve.py (bin, then compare inside
the bin); binned rather than per-atom-count because the fine-tune arms have ~100
molecules, not ~7,800. All caption text lives in the HTML, not in the image.
"""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FB = "/home1/irteam/funcbind/artifacts/reproduction/mcpp"
SRC = {
    "vanilla":        f"{FB}/cmp10/_eval/vanilla/eval_docking_results.json",
    "fine-tune 3.17M": f"{FB}/cmp10/_eval/finetuned/eval_docking_results.json",
    "fine-tune 8.21M": f"{FB}/cmp10_r10/_eval/finetuned/eval_docking_results.json",
}
# Categorical slots 1-3 of the validated palette: they clear the all-pairs CVD and
# normal-vision floors together. Aqua sits below 3:1 on white, so the arm it marks
# carries a direct label (the relief rule), and §5's tables repeat every number.
COL = {"vanilla": "#2a78d6", "fine-tune 3.17M": "#eb6834", "fine-tune 8.21M": "#1baf7a"}
MIN_N = 5          # a bin with fewer molecules than this is a data point, not a trend

D = {k: json.load(open(v))["per_target"] for k, v in SRC.items()}
# target_07 yielded molecules for the 8.21M arm only; comparing arms on it would be
# comparing one arm against nothing.
COMMON = sorted(set.intersection(*[
    {e["target"] for e in per if e.get("vina_dock") is not None} for per in D.values()]))

BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 89)]
LBL = ["20–29", "30–39", "40–49", "50–59", "60+"]


def mols(arm):
    out = []
    for e in D[arm]:
        if e["target"] not in COMMON:
            continue
        for m in e.get("per_mol") or []:
            if m.get("vina_dock") is not None and m.get("n_atoms"):
                out.append((m["n_atoms"], m["vina_dock"]))
    return out


M = {k: mols(k) for k in SRC}
binned = {k: [[d for a, d in v if lo <= a <= hi] for lo, hi in BINS] for k, v in M.items()}

fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 6.8), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.13))
x = np.arange(len(BINS))

for k in SRC:
    ys = [np.median(b) if len(b) >= MIN_N else np.nan for b in binned[k]]
    ax.plot(x, ys, color=COL[k], lw=2.0, marker="o", ms=8.5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.6, label=k)
    # No per-point n here: the bottom panel already labels every count, and three
    # series x five bins of "n=" collide with each other and with the markers.
    last = max(i for i, y in enumerate(ys) if not np.isnan(y))
    # A series that stops early gets its label above the point instead of to the
    # right, where the arms that continue would run through it.
    kw = dict(xytext=(10, -2), ha="left") if last == len(BINS) - 1 else dict(xytext=(0, 15), ha="center")
    ax.annotate(k, (last, ys[last]), textcoords="offset points",
                fontsize=10.4, color=COL[k], fontweight="600", va="center", **kw)

ax.set_ylabel("Vina Dock, median\n(kcal/mol)", fontsize=11.3)
ax.grid(color="#e6e9ef", lw=.85); ax.set_axisbelow(True)
# Above the panel, not inside it: at "lower left" the legend sat on top of the
# vanilla line, which is the series the figure is arguing against.
ax.legend(frameon=False, fontsize=10.2, ncols=3, loc="lower center",
          bbox_to_anchor=(.5, 1.01))
ax.set_xlim(-.45, len(BINS) - .55)
ax.margins(y=.20)

w = .27
for i, k in enumerate(SRC):
    counts = [len(b) for b in binned[k]]
    # 2px of surface between neighbouring bars: the gap is the spacer, not a border.
    bx.bar(x + (i - 1) * w, counts, width=w * .88, color=COL[k], alpha=.92, linewidth=0)
    for xi, c in enumerate(counts):
        if c:
            bx.annotate(str(c), (xi + (i - 1) * w, c), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.4, color="#7a8699")

bx.set_ylabel("molecules", fontsize=11.3)
bx.set_xlabel("heavy atoms", fontsize=11.3)
bx.set_xticks(x); bx.set_xticklabels(LBL)
bx.grid(axis="y", color="#e6e9ef", lw=.85); bx.set_axisbelow(True)
# Median sizes are stated in the caption and the §5 table; drawn as vertical rules
# here they crossed the bars and their count labels, and two of the three landed a
# single atom apart.

for a in (ax, bx):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("#c4cad4")
    a.tick_params(labelsize=10.2, colors="#5b6678")

OUT = "/home1/irteam/VoxBind/notebook/html/260827/mcp_finetune_size"
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")

print("common targets:", len(COMMON))
for k in SRC:
    print(f"{k:18s} n={len(M[k]):4d} median heavy={np.median([a for a,_ in M[k]]):.0f} "
          f"bins={[len(b) for b in binned[k]]} "
          f"medians={[round(float(np.median(b)),2) if len(b)>=MIN_N else None for b in binned[k]]}")
print("wrote", OUT + ".{png,svg}")
