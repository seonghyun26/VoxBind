"""Bar chart of Pearson r per experiment for Table 2 (H-bond-count regression).

Values are transcribed verbatim from Table 2 of notebook/html/260611/260611_meeting.html
(frozen-encoder MLP probe, mean ± std over 3 seeds; the baseline is a deterministic
1-feature linear fit, no seed variance). Plots Validation vs Test Pearson r side by
side so the val→test overfit gap is visible. Output mirrors the other figure scripts:
PNG + base64 .b64.txt in notebook/html/260611/ for self-contained HTML embedding.
"""
import base64
import io
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "notebook/html/260611/table2_pearson.png"
OUT_B64 = ROOT / "notebook/html/260611/table2_pearson.b64.txt"

# label, val_r, val_sd, test_r, test_sd, is_baseline   (from Table 2)
ROWS = [
    ("Density+gradmag\nonly (ViT)",      0.708, 0.008, 0.657, 0.004, False),
    ("Coordinates\nonly (ViT)",          0.640, 0.010, 0.685, 0.004, False),
    ("Coords+dens+grad\n(ViT)",          0.707, 0.001, 0.672, 0.001, False),
    ("Coords+dens+grad\n(RoPE-3D)",      0.692, 0.004, 0.673, 0.004, False),
    ("Ligand N/O/S\ncount (baseline)",   0.529, 0.000, 0.416, 0.000, True),
]

labels   = [r[0] for r in ROWS]
val_r    = np.array([r[1] for r in ROWS])
val_sd   = np.array([r[2] for r in ROWS])
test_r   = np.array([r[3] for r in ROWS])
test_sd  = np.array([r[4] for r in ROWS])
is_base  = np.array([r[5] for r in ROWS])

# report palette
INK      = "#1c2433"
INK_SOFT = "#5b6678"
VAL_C    = "#9bb0c3"   # soft slate-blue — validation
TEST_C   = "#2e7d5b"   # green accent    — test (the headline metric)

x = np.arange(len(ROWS))
w = 0.38

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

def draw(group_x, heights, sds, color, label):
    enc = ~is_base
    bars = ax.bar(group_x, heights, w, color=color, edgecolor="white",
                  linewidth=0.6, label=label, zorder=3)
    # baseline bars: marked as non-encoder (hatched, dimmer)
    for b, base in zip(bars, is_base):
        if base:
            b.set_alpha(0.45)
            b.set_hatch("////")
    # error bars only where there is seed variance (encoders)
    ax.errorbar(group_x[enc], heights[enc], yerr=sds[enc], fmt="none",
                ecolor=INK_SOFT, elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    # value labels
    for xi, h, sd, base in zip(group_x, heights, sds, is_base):
        ax.text(xi, h + (sd if not base else 0) + 0.006, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8.2, color=INK)

draw(x - w / 2, val_r,  val_sd,  VAL_C,  "Validation Pearson $r$")
draw(x + w / 2, test_r, test_sd, TEST_C, "Test Pearson $r$")

# baseline test-r reference line (the floor every encoder clears)
ax.axhline(0.416, ls="--", lw=1.0, color=INK_SOFT, alpha=0.7, zorder=1)
ax.text(-0.45, 0.416 + 0.005, "baseline test $r$", ha="left", va="bottom",
        fontsize=7.8, color=INK_SOFT, style="italic")

ax.set_ylim(0.35, 0.78)
ax.set_ylabel("Pearson $r$", fontsize=11, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.0, color=INK)
ax.tick_params(axis="y", labelsize=9, colors=INK_SOFT)
ax.tick_params(axis="x", length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9d2dc")
ax.yaxis.grid(True, color="#e7ecf1", lw=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_title("Table 2  ·  Pearson $r$ per experiment — H-bond-count regression",
             fontsize=12.5, color=INK, pad=26, loc="left")
fig.text(0.125, 0.905,
         "Frozen-encoder MLP probe, mean ± std over 3 seeds (LP-PDBBind split). "
         "Baseline is a deterministic 1-feature fit (no variance). "
         "Wide val→test gaps flag overfitting.",
         ha="left", va="center", fontsize=8.6, color=INK_SOFT)

ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2,
          bbox_to_anchor=(1.0, 1.02))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
print("saved", OUT_PNG)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=170, facecolor="white", bbox_inches="tight",
            pad_inches=0.12)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
OUT_B64.write_text(b64)
print("b64 bytes:", len(b64))
