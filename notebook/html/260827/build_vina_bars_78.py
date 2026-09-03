#!/usr/bin/env python3
"""build_vina_bars_78.py — Figure-1 (Vina Score / Min / Dock) for results_drug_design.html.

Renders the 78-pocket Vina bars in the exact convention used by Figure 4 of
notebook/html/results.html (build_denovo_vina_chart.py): one panel per Vina flavour,
vertical bar = pocket average, thick horizontal tick = pooled-molecule median, a dashed
divider between the baseline group and the Ours group, and a black outline plus a bold
dark-green label on the best bar in each panel. A fourth panel carries High affinity in
the same style — the table has no median for it, so that panel is bars only and reads
upward from zero.

Values are parsed straight out of Table 1 of results_drug_design.html, so the figure can
never drift from the table. Rows still sampling (TBA) keep their slot as a hatched
placeholder; a metric that does not apply to a row (the reference ligand's own high
affinity) is left blank with an em dash, matching the table.

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/build_vina_bars_78.py
"""
import html
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "results_drug_design.html")
OUT = os.path.join(HERE, "denovo_vina_78")

# Shared with the method swatches in results_drug_design.html Table 1 — keep in sync.
COLORS = {
    "Reference":   "#8e9baa",
    "AR":          "#9b59b6",
    "Pocket2Mol":  "#e67e22",
    "DecompDiff":  "#34495e",
    "FuncBind":    "#c0699a",
    "TargetDiff":  "#3498db",
    "VoxBind\nσ0.9": "#1abc9c",
    "Ours · v1":   "#2ecc71",
    "Ours · v2":   "#7fcf9d",
    "Ours · v3":   "#52bb7e",
}
# First row of the Ours group — the dashed divider is drawn just before it.
OURS_HEAD = "Ours · v1"


def _text(fragment):
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("\xa0", " ").strip()


def _number(fragment):
    text = _text(fragment).replace("−", "-").replace("—", "")
    if "TBA" in text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _label(method_html):
    text = " ".join(_text(method_html).split())
    for key in ("Reference", "AR", "Pocket2Mol", "DecompDiff", "FuncBind", "TargetDiff"):
        if text.startswith(key):
            return key
    if text.startswith("VoxBind"):
        return "VoxBind\nσ0.9"
    match = re.match(r"Ours\s*·?\s*(v\d)", text)
    if match:
        return f"Ours · {match.group(1)}"
    raise ValueError(f"Unknown Table-1 method: {text}")


def parse_rows(table):
    """Method label plus the seven leading Table-1 cells: Score/Min/Dock × avg/med, then High aff."""
    rows = []
    for row_html in re.findall(r"<tr\b.*?</tr>", table, flags=re.S):
        cells = re.findall(r"<td\b.*?</td>", row_html, flags=re.S)
        if not cells or "col-method" not in cells[0]:
            continue
        values = [_number(cell) for cell in cells[1:8]]
        if len(values) != 7:
            continue
        rows.append({"label": _label(cells[0]), "values": values})
    return rows


def read_table1():
    with open(DOC, encoding="utf-8") as handle:
        doc = handle.read()
    start = doc.index("Table 1 ")
    return doc[start:doc.index("</table>", start)]


def render(rows):
    labels = [row["label"] for row in rows]
    colors = [COLORS[label] for label in labels]
    # (title, average index, median index or None, higher-is-better)
    metrics = (("Vina Score", 0, 1, False), ("Vina Min", 2, 3, False),
               ("Vina Dock", 4, 5, False), ("High affinity (%)", 6, None, True))
    divider = labels.index(OURS_HEAD)
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
    fig, axes = plt.subplots(1, 4, figsize=(19.6, 6.5), dpi=100)
    fig.patch.set_facecolor("white")

    # rows that are still sampling — hatched in every panel, so read them once
    pending = {i for i, row in enumerate(rows) if all(v is None for v in row["values"])}

    for ax, (title, avg_i, med_i, higher) in zip(axes, metrics):
        averages = [row["values"][avg_i] for row in rows]
        medians = [row["values"][med_i] for row in rows] if med_i is not None else [None] * len(rows)
        filled = [i for i, avg in enumerate(averages) if avg is not None]
        best_i = max(filled, key=lambda i: averages[i]) if higher else min(filled, key=lambda i: averages[i])
        vals = [v for v in averages + medians if v is not None]
        # Vina panels hang downward from the top; a percentage panel reads upward from zero.
        lo, hi = (0.0, max(vals) * 1.18) if higher else (min(vals) - 0.55, max(vals) + 0.75)
        pad = (hi - lo) * 0.022

        for i in x:
            avg, med = averages[i], medians[i]
            if i in pending:                      # still sampling — keep the slot, mark it TBA
                ax.bar(i, hi - lo, bottom=lo, width=0.72, color="#f2f4f7",
                       edgecolor="#d7dce4", hatch="//", linewidth=0.8, zorder=1)
                ax.text(i, (lo + hi) / 2, "TBA", ha="center", va="center",
                        fontsize=9, color="#9aa5b4")
                continue
            if avg is None:                       # metric does not apply to this row
                ax.text(i, lo + (hi - lo) * 0.035, "n/a", ha="center", va="bottom",
                        fontsize=9, color="#9aa5b4", style="italic")
                continue
            bar = ax.bar(i, avg, width=0.72, color=colors[i], edgecolor="none", zorder=2)[0]
            if i == best_i:
                bar.set_edgecolor("#111827")
                bar.set_linewidth(1.8)
            if med is not None:
                ax.hlines(med, i - 0.34, i + 0.34, color="#566176", linewidth=3, zorder=4)
            label = f"{avg:.1f}" if higher else f"{avg:.2f}"
            ax.text(i, avg + pad if higher else avg - pad, label, ha="center",
                    va="bottom" if higher else "top", fontsize=8.5,
                    color="#165c35" if i == best_i else "#566176",
                    fontweight="bold" if i == best_i else "normal")

        ax.axvline(divider - 0.5, color="#aeb9c8", linewidth=1.8, linestyle=(0, (2, 2)), zorder=3)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8.5)
        ax.yaxis.grid(True, color="#e1e6ee", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(lo, hi)

    axes[0].set_ylabel("kcal/mol  (lower = stronger)", fontsize=12)
    axes[3].set_ylabel("% out-docking the reference  (higher = better)", fontsize=11, labelpad=6)
    fig.suptitle(
        "Vina Score / Min / Dock and high affinity     ·     78 CrossDocked pockets     ·     "
        "~100 molecules per pocket, exhaustiveness 16, pocket10 crop",
        fontsize=14, y=0.985,
    )
    fig.legend(
        handles=[Patch(facecolor="#9aa5b4", label="Average (bar)"),
                 Line2D([0], [0], color="#566176", lw=3, label="Median (tick)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2,
        frameon=True, facecolor="white", edgecolor="#dfe4eb",
    )
    fig.subplots_adjust(left=0.042, right=0.988, top=0.79, bottom=0.30, wspace=0.16)
    return fig


def main():
    rows = parse_rows(read_table1())
    fig = render(rows)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}.{ext}", format=ext, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}.png / .svg — {len(rows)} rows: "
          + ", ".join(r["label"].replace(chr(10), " ") for r in rows))


if __name__ == "__main__":
    main()
