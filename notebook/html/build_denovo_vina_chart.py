#!/usr/bin/env python3
"""Render the Table-4 Vina averages/medians with paper and reproduction groups."""
import base64
import html
import io
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = {
    "Reference": "#8e9baa",
    "AR": "#9b59b6",
    "Pocket2Mol": "#e67e22",
    "DiffSBDD": "#e74c3c",
    "TargetDiff": "#3498db",
    "DecompDiff": "#34495e",
    "DecompDiff\n(ref-informed)": "#71869b",
    "DecompDiff\n(ref-free)": "#a6b3c0",
    "VoxBind σ0.9\n(paper)": "#1abc9c",
    "VoxBind σ1.0\n(paper)": "#e83e8c",
    "VoxBind σ0.9\n(reproduced)": "#a8dbc0",
    "VoxBind σ1.0\n(reproduced)": "#52bb7e",
    "Ours": "#2ecc71",
}

DISPLAY = {
    "VoxBind σ0.9\n(paper)": "VoxBind\nσ0.9",
    "VoxBind σ1.0\n(paper)": "VoxBind\nσ1.0",
    "DecompDiff\n(ref-informed)": "DecompDiff\nref-informed",
    "DecompDiff\n(ref-free)": "DecompDiff\nref-free",
    "VoxBind σ0.9\n(reproduced)": "VoxBind\nσ0.9",
    "VoxBind σ1.0\n(reproduced)": "VoxBind\nσ1.0",
}


def _text(fragment):
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("\xa0", " ").strip()


def _number(fragment):
    text = _text(fragment).replace("−", "-").replace("—", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _label(method_html):
    text = " ".join(_text(method_html).split())
    reproduced = 'class="tag repro"' in method_html
    if text.startswith("Reference"):
        return "Reference"
    if text.startswith("AR "):
        return "AR"
    if text.startswith("Pocket2Mol"):
        return "Pocket2Mol"
    if text.startswith("DiffSBDD"):
        return "DiffSBDD"
    if text.startswith("TargetDiff"):
        return "TargetDiff"
    if text.startswith("DecompDiff"):
        if reproduced and "ref-informed" in text:
            return "DecompDiff\n(ref-informed)"
        if reproduced and "ref-free" in text:
            return "DecompDiff\n(ref-free)"
        return "DecompDiff"
    if text.startswith("VoxBind"):
        sigma = "σ0.9" if "0.9" in text else "σ1.0"
        source = "reproduced" if reproduced else "paper"
        return f"VoxBind {sigma}\n({source})"
    if text.startswith("Ours"):
        return "Ours"
    raise ValueError(f"Unknown Table-4 method: {text}")


def parse_rows(table):
    """Read method names and the first six Vina cells directly from the rendered Table 4."""
    rows = []
    for row_html in re.findall(r"<tr\b.*?</tr>", table, flags=re.S):
        cells = re.findall(r"<td\b.*?</td>", row_html, flags=re.S)
        if not cells or "col-method" not in cells[0]:
            continue
        values = [_number(cell) for cell in cells[1:7]]
        if len(values) != 6 or any(value is None for value in values):
            continue
        rows.append({"label": _label(cells[0]), "values": values})
    return rows


def render(table):
    """Return an embedded PNG <img>, preserving the original Matplotlib chart convention."""
    rows = parse_rows(table)
    labels = [row["label"] for row in rows]
    colors = [COLORS[label] for label in labels]
    metrics = (("Vina Score", 0, 1), ("Vina Min", 2, 3), ("Vina Dock", 4, 5))
    paper_n = labels.index("DecompDiff\n(ref-informed)")
    x = list(range(len(rows)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#1f2937",
        "text.color": "#1f2937",
        "axes.labelcolor": "#1f2937",
        "xtick.color": "#8a94a6",
        "ytick.color": "#8a94a6",
    })
    fig, axes = plt.subplots(1, 3, figsize=(19.6, 6.5), dpi=100)
    fig.patch.set_facecolor("white")

    for ax, (title, avg_i, med_i) in zip(axes, metrics):
        averages = [row["values"][avg_i] for row in rows]
        medians = [row["values"][med_i] for row in rows]
        bars = ax.bar(x, averages, width=0.72, color=colors, edgecolor="none", zorder=2)
        bars[-1].set_edgecolor("#111827")
        bars[-1].set_linewidth(1.8)
        for i, (avg, med) in enumerate(zip(averages, medians)):
            ax.hlines(med, i - 0.34, i + 0.34, color="#566176", linewidth=3, zorder=4)
            ax.text(i, avg - 0.15, f"{avg:.2f}", ha="center", va="top", fontsize=8.5,
                    color="#165c35" if i == len(rows) - 1 else "#566176",
                    fontweight="bold" if i == len(rows) - 1 else "normal")
        ax.axvline(paper_n - 0.5, color="#aeb9c8", linewidth=1.8, linestyle=(0, (2, 2)), zorder=1)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x, [DISPLAY.get(label, label) for label in labels],
                      rotation=40, ha="right", fontsize=8.5)
        ax.yaxis.grid(True, color="#e1e6ee", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        vals = averages + medians
        ax.set_ylim(min(vals) - 0.55, max(vals) + 0.75)

    axes[0].set_ylabel("kcal/mol  (lower = stronger)", fontsize=12)
    fig.suptitle(
        "Vina Score / Min / Dock     ·     paper: 100 pockets     ·     "
        "reproduction: DecompDiff 98/100; VoxBind/Ours 79 density pockets",
        fontsize=14, y=0.985,
    )
    fig.legend(
        handles=[Patch(facecolor="#9aa5b4", label="Average (bar)"),
                 Line2D([0], [0], color="#566176", lw=3, label="Median (tick)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2,
        frameon=True, facecolor="white", edgecolor="#dfe4eb",
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.79, bottom=0.30, wspace=0.12)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<img src="data:image/png;base64,{data}" '
            'style="width:100%;max-width:1960px;border-radius:6px" '
            'alt="Vina Score, Min, and Dock bars; reproduced DecompDiff is grouped with reproduced VoxBind and Ours">')
