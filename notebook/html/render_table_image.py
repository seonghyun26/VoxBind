"""render_table_image.py — render the results Table 1 (Spearman ρ) + leakage dumbbell to a PNG,
so the grouped table can be viewed as an image (no HTML renderer available)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "260715"))
import build_results as br
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLS = [("lp_edrscc_v2", "v2"), ("lp_edrscc_v2_cl1", "+CL1"), ("lp_edrscc_v2_cl12", "+CL1+2"),
        ("lp_edrscc_v2_cl123", "+CL1+2+3"), ("CASF-leaky", "CASF"), ("CASF-nt", "CASF−train")]

def rho(m, key):
    if key == "CASF-leaky":
        d = br.load_casf(m, "leaky")
    elif key == "CASF-nt":
        d = br.load_casf(m, "nontrain")
    else:
        d = br.load(m, key)
    return d["rho"][0] if d and d.get("rho") and d["rho"][0] is not None else np.nan

methods = sorted(br.ORDER, key=lambda m: -np.nanmean([rho(m, c) for c, _ in COLS[:4]]))
M = np.array([[rho(m, c) for c, _ in COLS] for m in methods])
# "best" per column excludes leaked methods (Nesso-1, IPNet-frozen) — same as the HTML table
_leak = np.array([m in br.LEAKED for m in methods])
best = {}
for j in range(M.shape[1]):
    col = M[:, j].copy(); col[_leak] = np.nan
    if not np.all(np.isnan(col)):
        best[j] = methods[int(np.nanargmax(col))]

fig = plt.figure(figsize=(11.6, 11.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.32, 1.0], hspace=0.20)

# ── top: Spearman ρ heatmap-table ────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
im = ax.imshow(M, cmap="RdYlGn", vmin=0.43, vmax=0.72, aspect="auto")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([l for _, l in COLS], fontsize=10.5)
# y-labels: annotate each method's frozen pretrained backbone + a leaked marker
def _ylab(m):
    s = m + (f"  [{br.BACKBONE[m][0]}]" if m in br.BACKBONE else "")
    return s + ("  ⚠leaked" if m in br.LEAKED else "")
ax.set_yticks(range(len(methods))); ax.set_yticklabels([_ylab(m) for m in methods], fontsize=10.5)
for tick, m in zip(ax.get_yticklabels(), methods):
    if m in br.LEAKED:
        tick.set_color("#b03030"); tick.set_fontweight("bold")   # leaked (PDBbind-trained) — red
        continue
    bb = br.BACKBONE.get(m)
    if bb and bb[1] == "esm":
        tick.set_color("#b25a10"); tick.set_fontweight("bold")   # ESM users pop
    elif bb:
        tick.set_color("#5b6678")                                 # other pretrained backbone
ax.set_title("Binding-affinity Spearman ρ  —  LP-PDBBind no-leak tiers  |  CASF-2016 (trained on v2)",
             fontsize=12.5, fontweight="bold", pad=12)
for i in range(len(methods)):
    for j in range(len(COLS)):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#888"); continue
        isbest = best.get(j) == methods[i]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold" if isbest else "normal", color="#0a0a0a")
        if isbest:
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="#111", lw=2.2))
ax.axvline(3.5, color="#1c2433", lw=3)  # divider before CASF block
ax.text(4.5, -0.9, "trained on v2 train, tested on CASF", ha="center", fontsize=8.5, color="#555", style="italic")
ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.4); ax.tick_params(which="minor", length=0)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Spearman ρ")

# ── bottom: CASF leakage dumbbell ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
dd = []
for m in br.ORDER:
    lk, nt = br.load_casf(m, "leaky"), br.load_casf(m, "nontrain")
    if lk and nt and lk["rho"][0] is not None and nt["rho"][0] is not None:
        dd.append((m, nt["rho"][0], lk["rho"][0]))
dd.sort(key=lambda t: t[2]-t[1])   # smallest gap at bottom, biggest at top after invert
ys = range(len(dd))
for y, (m, nt, lk) in zip(ys, dd):
    gap = lk-nt
    col = "#d64545" if gap >= 0.10 else ("#9aa3b2" if gap < 0.03 else "#3f7fc4")
    ax2.plot([nt, lk], [y, y], color=col, lw=2.6, alpha=.85, zorder=1)
    ax2.scatter([nt], [y], s=55, color=col, zorder=2)
    ax2.scatter([lk], [y], s=55, facecolor="white", edgecolor=col, lw=2, zorder=2)
    ax2.text(max(lk, nt)+.006, y, f"{gap:+.3f}", va="center", fontsize=8.8, color=col, fontweight="bold")
ax2.set_yticks(list(ys)); ax2.set_yticklabels([m for m, _, _ in dd], fontsize=10)
ax2.set_xlabel("Spearman ρ", fontsize=10)
ax2.set_title("CASF-2016 leakage gap  —  non-train (honest, ●)  →  leaky (train-overlap, ○)",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xlim(0.42, 0.90); ax2.grid(axis="x", color="#eceff3", lw=1); ax2.set_axisbelow(True)
for s in ("top", "right", "left"): ax2.spines[s].set_visible(False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_table.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print("wrote", out)
