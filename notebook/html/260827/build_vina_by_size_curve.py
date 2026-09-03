"""build_vina_by_size_curve.py — Figure 4 of results_drug_design.html.

Vina Dock against ligand size at every atom count, plus the size distribution.

Top: mean Dock per exact heavy-atom count with a bootstrap band. Bottom: how many
molecules each model puts at that size -- which is what makes the top panel's tails
trustworthy or not. All caption text lives in the HTML, not in the image.
"""
import json, collections, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E = "/home1/irteam/VoxBind/voxbind/exps"
SRC = {"vanilla": f"{E}/_vanilla_ep923/samples/full_eval_ep923",
       "ours_v1": f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
       "v2":      f"{E}/samples_reference_receptor_ed_ep350",
       "v3":      f"{E}/samples_frozen_v3_mask090_ep561"}
D = {k: {int(e["target"].split("_")[-1]): e
         for e in json.load(open(f"{v}/eval_docking_results.json"))["per_target"]}
     for k, v in SRC.items()}
P78 = sorted((set(D["vanilla"]) & set(D["ours_v1"]) & set(D["v2"]) & set(D["v3"])) - {71})

def by_size(k):
    d = collections.defaultdict(list)
    for i in P78:
        for pm in (D[k][i].get("per_mol") or []):
            if pm.get("vina_dock") is not None and pm.get("n_atoms"):
                d[pm["n_atoms"]].append(pm["vina_dock"])
    return d

REF = collections.defaultdict(list)
for i in P78:
    r = (json.load(open(f"{SRC['vanilla']}/target_{i:02d}/metrics.json")).get("reference") or {})
    dd = D["vanilla"][i].get("ref_vina_dock")
    if r.get("n_atoms") and dd is not None:
        REF[r["n_atoms"]].append(dd)

VAN, OUR = "#6b7688", "#2f6f4f"
MODELS = [("VoxBind σ=0.9 (vanilla)", "vanilla", VAN),
          ("Ours · v1  (density-conditioned)", "ours_v1", OUR)]
S = {k: by_size(k) for _, k, _ in MODELS}
rng = np.random.default_rng(260827)
MIN_N, XMAX = 20, 50            # 50 keeps 98.4% of molecules; past it the counts are single digits

fig, (ax, bx) = plt.subplots(
    2, 1, figsize=(12.4, 6.9), sharex=True,
    gridspec_kw=dict(height_ratios=[2.8, 1.15], hspace=.07))

for name, key, col in MODELS:
    xs = sorted(a for a, v in S[key].items() if len(v) >= MIN_N and a <= XMAX)
    mu = [st.mean(S[key][a]) for a in xs]
    band = np.array([np.percentile(
        rng.choice(np.asarray(S[key][a]), (1500, len(S[key][a])), replace=True).mean(1),
        [2.5, 97.5]) for a in xs])
    ax.fill_between(xs, band[:, 0], band[:, 1], color=col, alpha=.17, linewidth=0, zorder=2)
    ax.plot(xs, mu, color=col, lw=2.6, zorder=4, label=name,
            solid_capstyle="round")

rx = [a for a in sorted(REF) if a <= XMAX]
ax.scatter(rx, [st.mean(REF[a]) for a in rx], marker="*", s=125, color="#c1442e",
           zorder=5, edgecolor="white", linewidth=.5, label="Reference ligand")

# Vanilla's best size, and where it starts losing ground -- the whole point of the plot.
vx = sorted(a for a, v in S["vanilla"].items() if len(v) >= MIN_N and a <= XMAX)
turn = min(vx, key=lambda a: st.mean(S["vanilla"][a]))
ax.axvspan(turn, XMAX, color="#b07a17", alpha=.055, zorder=0)
ax.axvline(turn, color="#b07a17", lw=1.1, ls=(0, (5, 3)), zorder=1)
bx.axvline(turn, color="#b07a17", lw=1.1, ls=(0, (5, 3)), zorder=1)

allmu = [st.mean(v) for k in S for a, v in S[k].items() if len(v) >= MIN_N and a <= XMAX]
ylo, yhi = min(allmu) - 1.5, max(allmu) + 0.7
ax.set_ylim(ylo, yhi)
ax.annotate(f"vanilla bottoms out at {turn} atoms\nand degrades beyond it",
            xy=(turn, ylo + .45), xytext=(turn + 1.4, ylo + .45),
            color="#8a6413", fontsize=10.8, va="bottom", linespacing=1.45, zorder=6)

ax.set_ylabel("Vina Dock   (kcal/mol, lower is better)", fontsize=12.3, labelpad=10)
ax.set_title("Vina Dock across the full ligand-size range — 78 CrossDocked pockets",
             fontsize=13.6, fontweight="620", pad=13)
ax.grid(color="#e6e9ef", lw=.85); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11.5, loc="lower left", handlelength=1.9,
          borderaxespad=.9, labelspacing=.55)

for j, (_, key, col) in enumerate(MODELS):
    xs = [a for a in sorted(S[key]) if a <= XMAX]
    bx.bar([a + (j - .5) * .42 for a in xs], [len(S[key][a]) for a in xs],
           width=.42, color=col, alpha=.9, linewidth=0)
bx.set_xlabel("Number of generated ligand heavy atoms", fontsize=12.3, labelpad=9)
bx.set_ylabel("generated\nmolecules", fontsize=11.3, labelpad=10)
bx.grid(axis="y", color="#e6e9ef", lw=.85); bx.set_axisbelow(True)
bx.set_xlim(1.5, XMAX + .5); bx.set_xticks(range(5, XMAX + 1, 5))

# Reference sizes on their own axis: there are 78 crystal ligands against ~15k generated
# molecules, so they cannot share the bars' scale. Raw counts peak at 6 and are far too
# spiky to read, so the line is a Gaussian-smoothed histogram (sigma = 1.6 atoms) and
# the right axis is labelled as smoothed to keep that explicit.
from scipy.ndimage import gaussian_filter1d
grid = np.arange(0, XMAX + 1)
rawref = np.array([len(REF.get(a, [])) for a in grid], dtype=float)
smooth = gaussian_filter1d(rawref, sigma=1.6, mode="constant")
rx2 = bx.twinx()
rx2.fill_between(grid, smooth, color="#c1442e", alpha=.10, linewidth=0, zorder=1)
rx2.plot(grid, smooth, color="#c1442e", lw=2.0, zorder=3,
         label=f"Reference ligand sizes ({int(rawref.sum())} of {sum(len(v) for v in REF.values())}, smoothed)")
# Fixed head-room on the reference axis: the smoothed peak is ~1.8, so a hard cap of 4
# keeps the red curve low in the panel and out of the bars, and keeps the axis stable
# across rebuilds instead of floating with the peak.
rx2.set_ylim(0, 4)
rx2.set_ylabel("reference\nligands", fontsize=11.3, color="#c1442e", labelpad=12)
rx2.tick_params(axis="y", colors="#c1442e", labelsize=9.5)
rx2.spines["right"].set_color("#e3b6ae")
for sp in ("top", "left", "bottom"): rx2.spines[sp].set_visible(False)
rx2.legend(frameon=False, fontsize=10.2, loc="upper right", labelcolor="#c1442e",
           handlelength=1.6, borderaxespad=.2)

for a in (ax, bx):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color("#c4cad4")

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"/home1/irteam/VoxBind/notebook/html/260827/vina_by_size_curve.{ext}",
                dpi=170, bbox_inches="tight", facecolor="white")
print(f"ref smoothed peak={max(smooth):.2f} (right axis capped at 4)\n"
      f"turn={turn}  ylim=({ylo:.2f},{yhi:.2f})  molecules<=({XMAX}) = "
      f"{sum(len(v) for k in S for a,v in S[k].items() if a<=XMAX)}")
