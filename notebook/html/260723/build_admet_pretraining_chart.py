"""Recreate the ADMET comparison chart from slide 19."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("admet_scratch_vs_pretraining.png")

INK = "#1B2333"
MUTED = "#5B6472"
TEAL = "#0E9488"
GRAY = "#94A3B8"
GRID = "#E5E9F0"

tasks = [
    "half_life\n(Spearman ρ)",
    "clearance_hep\n(Spearman ρ)",
    "cyp2c9_sub\n(AUPRC)",
    "cyp2d6_sub\n(AUPRC)",
]
scratch = np.array([0.194, 0.319, 0.338, 0.705])
pretrained = np.array([0.419, 0.400, 0.516, 0.740])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    }
)

fig, ax = plt.subplots(figsize=(12.8, 6.2), facecolor="white")
x = np.arange(len(tasks))
width = 0.34

bars_scratch = ax.bar(
    x - width / 2,
    scratch,
    width,
    label="Scratch",
    color=GRAY,
    edgecolor="none",
)
bars_pretrained = ax.bar(
    x + width / 2,
    pretrained,
    width,
    label="With pretraining",
    color=TEAL,
    edgecolor="none",
)

for bars in (bars_scratch, bars_pretrained):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.014,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=15, color=INK, linespacing=1.25)
ax.set_ylim(0, 0.86)
ax.set_yticks(np.arange(0, 0.81, 0.2))
ax.tick_params(axis="y", colors=MUTED, labelsize=13, length=0)
ax.tick_params(axis="x", length=0, pad=12)
ax.yaxis.grid(True, color=GRID, linewidth=1.2)
ax.set_axisbelow(True)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.spines["bottom"].set_linewidth(1.2)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, 1.10),
    ncol=2,
    frameon=False,
    fontsize=15,
    handlelength=1.15,
    handletextpad=0.55,
    columnspacing=1.8,
)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.86)
fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
plt.close(fig)

print(OUT)
