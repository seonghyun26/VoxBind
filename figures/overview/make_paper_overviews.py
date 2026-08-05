#!/usr/bin/env python3
"""Clean, paper-oriented overviews of the current two-tower + VoxBind adapter path."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "regular",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

INK = "#17212B"
MUTED = "#63717E"
LINE = "#C7D0D9"
BG = "#F7F9FB"
POCKET = "#3E73A8"
POCKET_BG = "#EAF2F9"
LIGAND = "#D65F7C"
LIGAND_BG = "#FBEAF0"
DENS = "#289C8E"
DENS_BG = "#E4F5F2"
MASK = "#8D65A7"
MASK_BG = "#F1EAF5"
MODEL = "#486A94"
MODEL_BG = "#E8EEF6"
TRAIN = "#E58A24"
TRAIN_BG = "#FFF1DF"
OUTCOL = "#CB554A"
OUT_BG = "#FBEAE8"


def canvas(w=1800, h=950):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=18, color=INK, weight="regular", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=10, **kw)


def box(ax, x, y, w, h, fc="white", ec=LINE, lw=1.5, radius=18, ls="-", z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.015,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.1, rad=0, ls="-", z=6):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                        color=color, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=z)
    ax.add_patch(p)
    return p


def section(ax, x, y, letter, title, width):
    text(ax, x, y, f"({letter})", 20, INK, "bold", ha="left")
    text(ax, x + 58, y, title, 20, INK, "bold", ha="left")
    ax.plot([x, x + width], [y + 28, y + 28], color=LINE, lw=1.2, zorder=1)


def badge(ax, x, y, label, fc, ec, tc, w=86):
    box(ax, x, y, w, 26, fc, ec, 1.0, 13, z=7)
    text(ax, x + w / 2, y + 13, label, 11, tc, "bold")


def module(ax, x, y, w, h, title, subtitle, fc, ec, status=None, title_size=17):
    box(ax, x, y, w, h, fc, ec, 1.7, 17)
    if status == "frozen":
        badge(ax, x + 10, y + 9, "FROZEN", "#F3F6F8", "#CAD3DC", MUTED, 78)
        ty = y + 50
    elif status == "trainable":
        badge(ax, x + 10, y + 9, "TRAINABLE", TRAIN_BG, TRAIN, TRAIN, 98)
        ty = y + 53
    else:
        ty = y + 35
    text(ax, x + w / 2, ty, title, title_size, ec, "bold")
    text(ax, x + w / 2, y + h - 28, subtitle, 14, MUTED)


def stack(ax, x, y, w, h, groups, title, font=13):
    text(ax, x + w / 2, y - 18, title, 14, INK, "bold")
    box(ax, x, y, w, h, "white", LINE, 1.4, 13)
    total = sum(g[2] for g in groups)
    yy = y + 6
    usable = h - 12
    for label, color, frac in groups:
        hh = usable * frac / total
        ax.add_patch(Rectangle((x + 6, yy), w - 12, hh - 2,
                               facecolor=color, edgecolor="none", zorder=3))
        text(ax, x + w / 2, yy + (hh - 2) / 2, label, font, INK, "bold")
        yy += hh


def grid_icon(ax, x, y, w=102, h=82, color=POCKET):
    for off, alpha in ((12, 0.14), (6, 0.24), (0, 0.38)):
        box(ax, x + off, y - off, w, h, color, color, 1.0, 8, z=3)
        for i in range(1, 5):
            ax.plot([x + off + 7, x + off + w - 7], [y - off + i * h / 5] * 2,
                    color="white", alpha=0.75, lw=0.7, zorder=4)
        for j in range(1, 6):
            xx = x + off + j * w / 6
            ax.plot([xx, xx], [y - off + 7, y - off + h - 7],
                    color="white", alpha=0.75, lw=0.7, zorder=4)


def mask_icon(ax, x, y, size=96):
    box(ax, x, y, size, size, "white", LINE, 1.3, 12)
    n, gap, pad = 5, 3, 8
    cell = (size - 2 * pad - 4 * gap) / n
    visible = {(0, 0), (0, 4), (2, 2), (3, 1), (4, 4), (1, 3)}
    for i in range(n):
        for j in range(n):
            on = (i, j) in visible
            ax.add_patch(Rectangle((x + pad + j * (cell + gap), y + pad + i * (cell + gap)),
                                   cell, cell, facecolor=(DENS_BG if on else "#D7DEE5"),
                                   edgecolor=(DENS if on else "none"), lw=0.7, zorder=4))


def complex_icon(ax, x, y, w=220, h=165):
    ax.add_patch(Rectangle((x, y + 12), w - 18, h - 18, facecolor="#F8FAFC",
                           edgecolor=LINE, lw=1.3, zorder=2))
    ax.add_patch(Polygon([(x, y + 12), (x + 18, y), (x + w, y), (x + w - 18, y + 12)],
                         closed=True, facecolor="#EDF2F6", edgecolor=LINE, lw=1.1, zorder=2))
    ax.add_patch(Polygon([(x + w - 18, y + 12), (x + w, y), (x + w, y + h - 18),
                          (x + w - 18, y + h)], closed=True, facecolor="#E8EEF3",
                         edgecolor=LINE, lw=1.1, zorder=2))
    for k in range(1, 4):
        xx = x + (w - 18) * k / 4
        yy = y + 12 + (h - 18) * k / 4
        ax.plot([xx, xx], [y + 12, y + h - 6], color="#D8E0E7", lw=0.7, zorder=2)
        ax.plot([x, x + w - 18], [yy, yy], color="#D8E0E7", lw=0.7, zorder=2)
    for s, a in ((0.86, .10), (.62, .15), (.40, .22)):
        ax.add_patch(Ellipse((x + .5 * w, y + .55 * h), (w - 28) * s, (h - 28) * s,
                             fc=DENS, ec=DENS, alpha=a, zorder=3))
    pts = [(0.22, .68, POCKET), (.36, .47, POCKET), (.50, .61, LIGAND),
           (.63, .42, LIGAND), (.76, .61, POCKET)]
    for a, b in zip(pts[:-1], pts[1:]):
        ax.plot([x + a[0] * (w - 18), x + b[0] * (w - 18)],
                [y + a[1] * h, y + b[1] * h], color="#788590", lw=2, zorder=5)
    for px, py, c in pts:
        ax.add_patch(Circle((x + px * (w - 18), y + py * h), 9, fc=c, ec="white", lw=1.2, zorder=6))


def cdg_data_scene(ax, x, y, w=340, h=245):
    """Illustrate pocket, ligand, and electron density as distinct aligned data."""
    ax.add_patch(Rectangle((x, y + 14), w - 22, h - 22, facecolor="#F8FAFC",
                           edgecolor=LINE, lw=1.3, zorder=2))
    ax.add_patch(Polygon([(x, y + 14), (x + 22, y), (x + w, y), (x + w - 22, y + 14)],
                         closed=True, facecolor="#EEF3F7", edgecolor=LINE, lw=1.1, zorder=2))
    ax.add_patch(Polygon([(x + w - 22, y + 14), (x + w, y), (x + w, y + h - 22),
                          (x + w - 22, y + h)], closed=True, facecolor="#E8EEF3",
                         edgecolor=LINE, lw=1.1, zorder=2))
    for k in range(1, 5):
        xx = x + (w - 22) * k / 5
        yy = y + 14 + (h - 22) * k / 5
        ax.plot([xx, xx], [y + 14, y + h - 8], color="#DCE3E9", lw=.7, zorder=2)
        ax.plot([x, x + w - 22], [yy, yy], color="#DCE3E9", lw=.7, zorder=2)

    # Experimental ED envelope around the binding site.
    for scale, alpha in ((1.0, .08), (.76, .12), (.53, .17), (.31, .24)):
        ax.add_patch(Ellipse((x + .52 * w, y + .55 * h), (w - 52) * scale,
                             (h - 54) * scale, fc=DENS, ec=DENS, alpha=alpha, zorder=3))

    # Pocket atoms form a blue shell around the central ligand.
    pocket_pts = [(0.16, .54), (.24, .32), (.38, .23), (.63, .23), (.79, .36),
                  (.84, .61), (.72, .78), (.47, .83), (.26, .73)]
    for a, b in zip(pocket_pts, pocket_pts[1:] + pocket_pts[:1]):
        ax.plot([x + a[0] * (w - 22), x + b[0] * (w - 22)],
                [y + a[1] * h, y + b[1] * h], color=POCKET, lw=1.4, alpha=.55, zorder=4)
    for px, py in pocket_pts:
        ax.add_patch(Circle((x + px * (w - 22), y + py * h), 9, fc=POCKET,
                            ec="white", lw=1.2, zorder=6))

    # Ligand is a compact pink molecule at the center.
    lig_pts = [(0.42, .53), (.51, .43), (.61, .52), (.55, .65), (.43, .66)]
    for a, b in zip(lig_pts[:-1], lig_pts[1:]):
        ax.plot([x + a[0] * (w - 22), x + b[0] * (w - 22)],
                [y + a[1] * h, y + b[1] * h], color="#8C4A5A", lw=2.4, zorder=7)
    for px, py in lig_pts:
        ax.add_patch(Circle((x + px * (w - 22), y + py * h), 10, fc=LIGAND,
                            ec="white", lw=1.3, zorder=8))


def modality_card(ax, x, y, w, title, subtitle, fc, ec, kind):
    box(ax, x, y, w, 132, fc, ec, 1.5, 16)
    if kind == "ligand":
        pts = [(x + 30, y + 60), (x + 54, y + 42), (x + 78, y + 61), (x + 58, y + 80)]
        for a, b in zip(pts[:-1], pts[1:]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#8C4A5A", lw=2.0, zorder=5)
        for px, py in pts:
            ax.add_patch(Circle((px, py), 6.5, fc=LIGAND, ec="white", lw=1, zorder=6))
    elif kind == "pocket":
        for px, py in ((28, 44), (52, 62), (78, 42), (88, 76), (42, 83)):
            ax.add_patch(Circle((x + px, y + py), 7, fc=POCKET, ec="white", lw=1, zorder=6))
    else:
        for s, a in ((76, .12), (54, .18), (30, .28)):
            ax.add_patch(Ellipse((x + w / 2, y + 61), s, s * .64, fc=DENS, ec=DENS,
                                 alpha=a, zorder=4))
    text(ax, x + w / 2, y + 104, title, 13, ec, "bold")
    text(ax, x + w / 2, y + 122, subtitle, 11, MUTED)


def save(fig, stem):
    for ext in ("svg", "pdf", "png"):
        kw = {"facecolor": "white", "bbox_inches": None, "pad_inches": 0}
        if ext == "png":
            kw["dpi"] = 180
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)


def pretraining():
    fig, ax = canvas(1800, 950)
    text(ax, 900, 42, "Spatial two-tower pre-training", 31, INK, "bold")
    text(ax, 900, 78, "Electron-density partitioning, independent masked autoencoding, and a frozen-token affinity check",
         16, MUTED)

    section(ax, 55, 132, "a", "Density partitioning", 345)
    section(ax, 460, 132, "b", "Independent MAE towers", 820)
    section(ax, 1340, 132, "c", "Affinity & transfer", 405)
    ax.plot([430, 430], [120, 875], color=LINE, lw=1.0)
    ax.plot([1310, 1310], [120, 875], color=LINE, lw=1.0)

    # Input construction.
    complex_icon(ax, 106, 192, 220, 165)
    text(ax, 216, 382, "Aligned complex + 2Fo−Fc ED", 16, INK, "bold")
    box(ax, 72, 424, 288, 148, MASK_BG, MASK, 1.5, 17)
    text(ax, 216, 452, r"$M_P=\mathbb{1}[\sum_c V_{P,c}>\tau]$", 18)
    ax.plot([96, 336], [476, 476], color="#CDBADD", lw=1.0)
    text(ax, 216, 508, r"$\rho_P=M_P\odot\rho_{exp}$", 18, POCKET, "bold")
    text(ax, 216, 544, r"$\rho_L=(1-M_P)\odot\rho_{exp}$", 18, LIGAND, "bold")
    arrow(ax, 216, 574, 140, 632, POCKET, rad=.05)
    arrow(ax, 216, 574, 300, 632, LIGAND, rad=-.05)
    stack(ax, 58, 648, 155, 170,
          [("Pocket atoms ×4", POCKET_BG, 4), ("ρP + |∇ρP|", DENS_BG, 2), ("MP", MASK_BG, 1)],
          "Pocket · 7 ch", 11)
    stack(ax, 232, 648, 160, 170,
          [("Ligand atoms ×7", LIGAND_BG, 7), ("ρL + |∇ρL|", DENS_BG, 2), ("1−MP", MASK_BG, 1)],
          "Ligand · 10 ch", 11)
    text(ax, 216, 856, "Shared center, orientation, crop, and resolution", 13, MUTED)

    # Pocket MAE lane.
    box(ax, 470, 184, 810, 286, POCKET_BG, "#B8CCE0", 1.4, 22)
    badge(ax, 492, 202, "POCKET", POCKET, POCKET, "white", 92)
    stack(ax, 500, 282, 108, 130,
          [("VP ×4", POCKET_BG, 4), ("ρP, GP", DENS_BG, 2), ("MP", MASK_BG, 1)], "7-ch input", 11)
    arrow(ax, 614, 347, 655, 347, POCKET)
    mask_icon(ax, 661, 299, 96)
    text(ax, 709, 422, "75% block mask", 13, MUTED)
    arrow(ax, 763, 347, 802, 347, POCKET)
    module(ax, 808, 272, 250, 150, "ChannelViT encoder", "12 blocks · D=512 · [4,2,1]",
           MODEL_BG, MODEL, None, 15)
    arrow(ax, 1064, 347, 1105, 347, POCKET)
    grid_icon(ax, 1112, 316, 98, 76, POCKET)
    text(ax, 1167, 417, r"$T_P$: 8³ × 512", 15, POCKET, "bold")
    arrow(ax, 1158, 428, 1000, 447, MUTED, 1.7, .04)
    text(ax, 875, 447, r"reconstruct hidden voxels · $\mathcal{L}_{MAE}$", 14, MUTED)

    # Ligand MAE lane.
    box(ax, 470, 505, 810, 286, LIGAND_BG, "#E5B9C5", 1.4, 22)
    badge(ax, 492, 523, "LIGAND", LIGAND, LIGAND, "white", 92)
    stack(ax, 500, 603, 108, 130,
          [("VL ×7", LIGAND_BG, 7), ("ρL, GL", DENS_BG, 2), ("1−MP", MASK_BG, 1)], "10-ch input", 11)
    arrow(ax, 614, 668, 655, 668, LIGAND)
    mask_icon(ax, 661, 620, 96)
    text(ax, 709, 743, "75% block mask", 13, MUTED)
    arrow(ax, 763, 668, 802, 668, LIGAND)
    module(ax, 808, 593, 250, 150, "ChannelViT encoder", "12 blocks · D=512 · [7,2,1]",
           MODEL_BG, MODEL, None, 15)
    arrow(ax, 1064, 668, 1105, 668, LIGAND)
    grid_icon(ax, 1112, 637, 98, 76, LIGAND)
    text(ax, 1167, 738, r"$T_L$: 8³ × 512", 15, LIGAND, "bold")
    arrow(ax, 1158, 749, 1000, 768, MUTED, 1.7, .04)
    text(ax, 875, 768, r"reconstruct hidden voxels · $\mathcal{L}_{MAE}$", 14, MUTED)
    text(ax, 875, 838, "Two separate 50-epoch runs · 112k PLINDER-v2 complexes", 14, MUTED)

    # Frozen-token probe and transfer.
    box(ax, 1352, 188, 380, 398, BG, LINE, 1.4, 21, "--")
    badge(ax, 1372, 205, "EVALUATION", "#F3F6F8", LINE, MUTED, 96)
    box(ax, 1380, 260, 132, 70, POCKET_BG, POCKET, 1.3, 14)
    text(ax, 1446, 295, r"frozen $T_P$", 15, POCKET, "bold")
    box(ax, 1572, 260, 132, 70, LIGAND_BG, LIGAND, 1.3, 14)
    text(ax, 1638, 295, r"frozen $T_L$", 15, LIGAND, "bold")
    arrow(ax, 1446, 334, 1503, 382, POCKET)
    arrow(ax, 1638, 334, 1581, 382, LIGAND)
    module(ax, 1420, 386, 244, 120, "Cross-attention ×2", "bidirectional · 3D RoPE",
           TRAIN_BG, TRAIN, None, 17)
    arrow(ax, 1542, 510, 1542, 538, OUTCOL)
    text(ax, 1542, 558, r"mean pool → MLP → $\hat{pK}$", 16, OUTCOL, "bold")
    text(ax, 1542, 610, "The towers remain unchanged", 13, MUTED)

    box(ax, 1370, 680, 344, 146, POCKET_BG, POCKET, 1.7, 19)
    badge(ax, 1388, 697, "TRANSFER", POCKET, POCKET, "white", 88)
    text(ax, 1542, 748, "Pocket encoder checkpoint", 18, POCKET, "bold")
    arrow(ax, 1542, 775, 1542, 799, POCKET)
    text(ax, 1542, 814, "VoxBind spatial adapter", 16, INK, "bold")
    text(ax, 1542, 864, "Only the pre-fusion pocket path is reused", 13, MUTED)

    # Modality legend.
    for x, c, lab in ((500, POCKET, "pocket"), (650, LIGAND, "ligand"),
                      (795, DENS, "electron density"), (1025, MASK, "region mask"),
                      (1220, TRAIN, "trainable head"), (1460, MODEL_BG, "frozen module")):
        ax.add_patch(Rectangle((x, 902), 22, 18, fc=c, ec=(LINE if c == MODEL_BG else c), lw=1.0))
        text(ax, x + 31, 911, lab, 13, MUTED, ha="left")

    save(fig, "pretraining_overview")


def voxbind_adapter():
    fig, ax = canvas(1800, 930)
    text(ax, 900, 42, "Transfer of the pretrained pocket encoder into VoxBind", 30, INK, "bold")
    text(ax, 900, 77, "A zero-initialized spatial adapter adds experimental-ED context while the pocket tower and VoxBind remain frozen",
         16, MUTED)

    section(ax, 55, 132, "a", "Conditions", 340)
    section(ax, 435, 132, "b", "Frozen backbone + trainable residual", 835)
    section(ax, 1310, 132, "c", "Denoising and sampling", 435)
    ax.plot([410, 410], [120, 855], color=LINE, lw=1.0)
    ax.plot([1290, 1290], [120, 855], color=LINE, lw=1.0)

    # Inputs.
    box(ax, 70, 190, 145, 86, LIGAND_BG, LIGAND, 1.5, 16)
    text(ax, 142, 220, r"Noisy ligand $y$", 16, LIGAND, "bold")
    text(ax, 142, 250, "7 × 64³ · σ=0.9", 13, MUTED)
    box(ax, 235, 190, 145, 86, POCKET_BG, POCKET, 1.5, 16)
    text(ax, 307, 220, r"Pocket $V_P$", 16, POCKET, "bold")
    text(ax, 307, 250, "4 × 64³", 13, MUTED)
    box(ax, 70, 320, 145, 86, DENS_BG, DENS, 1.5, 16)
    text(ax, 142, 350, r"Experimental $\rho$", 16, DENS, "bold")
    text(ax, 142, 380, "1 × 64³", 13, MUTED)
    box(ax, 235, 310, 145, 106, MASK_BG, MASK, 1.5, 16)
    text(ax, 307, 338, "Protein vdW mask", 14, MASK, "bold")
    text(ax, 307, 370, r"$M_P=\mathbb{1}[\sum V_P>\tau]$", 15)
    text(ax, 307, 397, r"$\rho_P=M_P\odot\rho$", 15)

    arrow(ax, 307, 420, 307, 472, MASK)
    arrow(ax, 142, 410, 210, 482, DENS, rad=-.08)
    stack(ax, 124, 500, 210, 190,
          [("Pocket atoms ×4", POCKET_BG, 4), ("ρP", DENS_BG, 1),
           ("|∇ρP|", "#EAF4E6", 1), ("MP", MASK_BG, 1)],
          "Pocket-tower input · 7 ch", 13)
    text(ax, 229, 732, "No ligand coordinates or ligand ED", 14, MUTED, "bold")
    text(ax, 229, 758, "enter the conditioning path", 14, MUTED)

    # Frozen VoxBind coordinate backbone.
    text(ax, 460, 182, "Original coordinate path", 18, INK, "bold", ha="left")
    module(ax, 480, 220, 210, 112, "Ligand encoder", "ResidualBlock → C/2",
           MODEL_BG, MODEL, "frozen", 17)
    module(ax, 480, 358, 210, 112, "Pocket encoder", "ResidualBlock → C/2",
           MODEL_BG, MODEL, "frozen", 17)
    # Route the ligand arrow under the neighboring pocket card to keep inputs distinct.
    ax.plot([142, 142, 448], [276, 296, 296], color=LIGAND, lw=2.1, zorder=5)
    arrow(ax, 448, 296, 474, 276, LIGAND)
    arrow(ax, 380, 233, 474, 414, POCKET, rad=-.10)
    arrow(ax, 696, 276, 748, 337, MODEL, rad=.05)
    arrow(ax, 696, 414, 748, 347, MODEL, rad=-.05)
    ax.add_patch(Circle((770, 342), 23, fc="white", ec=MODEL, lw=1.7, zorder=8))
    text(ax, 770, 342, "+", 25, MODEL, "bold")
    arrow(ax, 795, 342, 838, 342, MODEL)
    box(ax, 844, 300, 150, 84, "#F3F6F8", MODEL, 1.4, 16)
    text(ax, 919, 327, r"$x_{base}$", 18, MODEL, "bold")
    text(ax, 919, 358, "C/2 × 64³", 13, MUTED)

    # Transfer branch.
    text(ax, 460, 530, "Transferred experimental-ED path", 18, INK, "bold", ha="left")
    arrow(ax, 334, 595, 466, 595, POCKET)
    module(ax, 472, 548, 264, 132, "Pocket ChannelViT", "12 blocks · D=512 · [4,2,1]",
           POCKET_BG, POCKET, "frozen", 17)
    arrow(ax, 742, 614, 780, 614, POCKET)
    grid_icon(ax, 787, 582, 98, 76, POCKET)
    text(ax, 842, 684, "group-pooled 8³ tokens", 14, POCKET, "bold")
    arrow(ax, 896, 614, 934, 614, TRAIN)
    module(ax, 940, 548, 238, 132, "PocketAdapter", "1×1 proj · ↑8 · 3×3",
           TRAIN_BG, TRAIN, "trainable", 18)
    text(ax, 1059, 708, "3×3 output is zero-initialized", 13, TRAIN, "bold")

    # Residual fusion.
    arrow(ax, 998, 342, 1125, 342, MODEL)
    arrow(ax, 1178, 614, 1148, 370, TRAIN, 2.5, -.13)
    ax.add_patch(Circle((1150, 342), 24, fc="white", ec=INK, lw=1.8, zorder=8))
    text(ax, 1150, 342, "+", 26, INK, "bold")
    text(ax, 1130, 424, r"$\Delta x$", 16, TRAIN, "bold")
    arrow(ax, 1176, 342, 1210, 342, INK)
    box(ax, 1216, 300, 62, 84, "#F3F6F8", INK, 1.4, 15)
    text(ax, 1247, 327, "fused", 12, MUTED, "bold")
    text(ax, 1247, 358, r"$x$", 19, INK, "bold")

    box(ax, 474, 770, 782, 82, "#FFF9F1", "#EDC386", 1.4, 17)
    badge(ax, 492, 788, "UPDATED", TRAIN_BG, TRAIN, TRAIN, 86)
    text(ax, 597, 801, "PocketAdapter only", 16, TRAIN, "bold", ha="left")
    text(ax, 492, 832, "Frozen: both coordinate encoders · pocket ChannelViT · 3D U-Net · output head",
         14, MUTED, ha="left")

    # Denoising and outputs.
    arrow(ax, 1282, 342, 1334, 342, INK)
    module(ax, 1340, 234, 392, 178, "VoxBind 3D U-Net", "UNet3D → activation → ligand head",
           MODEL_BG, MODEL, "frozen", 22)
    arrow(ax, 1536, 418, 1536, 456, OUTCOL)
    box(ax, 1420, 462, 232, 86, OUT_BG, OUTCOL, 1.7, 17)
    text(ax, 1536, 490, r"Denoised ligand $\hat{x}$", 18, OUTCOL, "bold")
    text(ax, 1536, 522, "7 × 64³", 14, MUTED)
    arrow(ax, 1536, 552, 1435, 620, OUTCOL, rad=.08)
    arrow(ax, 1536, 552, 1636, 620, OUTCOL, rad=-.08)

    box(ax, 1332, 624, 210, 122, "white", LINE, 1.4, 16)
    text(ax, 1437, 651, "Adapter training", 16, INK, "bold")
    text(ax, 1437, 688, r"$\|\hat{x}-x_{clean}\|^2$", 18)
    text(ax, 1437, 720, "gradients → adapter only", 13, MUTED)
    box(ax, 1558, 624, 192, 122, "white", LINE, 1.4, 16)
    text(ax, 1654, 651, "Generation", 16, INK, "bold")
    text(ax, 1654, 688, r"$s_\theta=(\hat{x}-y)/\sigma^2$", 16)
    text(ax, 1654, 720, "conditional score", 13, MUTED)
    arrow(ax, 1654, 750, 1654, 786, OUTCOL)
    box(ax, 1420, 790, 330, 64, OUT_BG, OUTCOL, 1.7, 17)
    text(ax, 1585, 812, "Walk–jump sampling", 17, OUTCOL, "bold")
    text(ax, 1585, 837, "Langevin walk → denoising jump", 13, MUTED)

    text(ax, 900, 892, "At initialization, Δx=0: the integrated model exactly reproduces the frozen VoxBind baseline.",
         14, MUTED, "bold")

    save(fig, "voxbind_adapter_overview")


def single_cdg_encoder():
    """Overview of the earlier single-encoder C+D+G pipeline."""
    fig, ax = canvas(1800, 930)
    text(ax, 900, 42, "Single-encoder CDG pipeline", 31, INK, "bold")
    text(ax, 900, 78,
         "Pocket coordinates, ligand coordinates, and experimental electron density are learned jointly in one spatial encoder",
         16, MUTED)

    section(ax, 55, 132, "a", "Aligned input data", 390)
    section(ax, 500, 132, "b", "Joint CDG pre-training", 760)
    section(ax, 1310, 132, "c", "Downstream reuse", 435)
    ax.plot([475, 475], [120, 872], color=LINE, lw=1.0)
    ax.plot([1285, 1285], [120, 872], color=LINE, lw=1.0)

    # (a) Make all three modalities explicit in the raw aligned data.
    cdg_data_scene(ax, 90, 202, 340, 245)
    badge(ax, 74, 470, "POCKET", POCKET_BG, POCKET, POCKET, 92)
    badge(ax, 188, 470, "LIGAND", LIGAND_BG, LIGAND, LIGAND, 92)
    badge(ax, 302, 470, "2Fo−Fc ED", DENS_BG, DENS, DENS, 106)
    text(ax, 250, 520, "One pocket-centered 64³ frame", 15, MUTED, "bold")

    modality_card(ax, 58, 570, 118, r"$V_L$", "ligand · 7 ch",
                  LIGAND_BG, LIGAND, "ligand")
    modality_card(ax, 190, 570, 118, r"$V_P$", "pocket · 4 ch",
                  POCKET_BG, POCKET, "pocket")
    modality_card(ax, 322, 570, 118, r"$\rho,\;|\nabla\rho|$", "ED · 2 ch",
                  DENS_BG, DENS, "density")
    arrow(ax, 117, 708, 224, 762, LIGAND, rad=-.05)
    arrow(ax, 249, 708, 249, 756, POCKET)
    arrow(ax, 381, 708, 276, 762, DENS, rad=.05)
    box(ax, 112, 766, 274, 72, "#F4F7FA", INK, 1.5, 17)
    text(ax, 249, 789, "Aligned CDG tensor", 17, INK, "bold")
    text(ax, 249, 817, "7 + 4 + 2 = 13 channels", 14, MUTED)
    text(ax, 249, 868, "C: coordinates · D: density · G: density gradient", 13, MUTED)

    # (b) One encoder jointly processes all modalities.
    ax.plot([386, 455, 455, 515], [802, 802, 439, 439], color=INK, lw=2.1, zorder=5)
    arrow(ax, 515, 439, 525, 439, INK)
    stack(ax, 530, 310, 154, 258,
          [("Ligand coords\nVL ×7", LIGAND_BG, 7),
           ("Pocket coords\nVP ×4", POCKET_BG, 4),
           ("ρ", DENS_BG, 1),
           ("|∇ρ|", "#EAF4E6", 1)],
          "Joint CDG input · 13 ch", 12)
    arrow(ax, 690, 439, 736, 439, INK)
    mask_icon(ax, 742, 391, 96)
    text(ax, 790, 511, "75% spatial mask", 14, MUTED)
    arrow(ax, 844, 439, 888, 439, INK)

    box(ax, 894, 272, 274, 334, MODEL_BG, MODEL, 1.8, 21)
    badge(ax, 916, 292, "ONE ENCODER", MODEL, MODEL, "white", 112)
    text(ax, 1031, 354, "CDG ChannelViT", 22, MODEL, "bold")
    text(ax, 1031, 393, "channel-grouped patch tokens", 15, MUTED)
    # Visible modality groups entering one shared representation.
    badge(ax, 922, 425, "LIGAND C", LIGAND_BG, LIGAND, LIGAND, 96)
    badge(ax, 1030, 425, "POCKET C", POCKET_BG, POCKET, POCKET, 96)
    badge(ax, 1138, 425, "D + G", DENS_BG, DENS, DENS, 78)
    arrow(ax, 970, 460, 1018, 505, LIGAND, 1.6)
    arrow(ax, 1078, 460, 1036, 505, POCKET, 1.6)
    arrow(ax, 1177, 460, 1054, 505, DENS, 1.6)
    box(ax, 952, 510, 158, 58, "white", MODEL, 1.3, 14)
    text(ax, 1031, 539, "shared 3D context", 15, MODEL, "bold")
    text(ax, 1031, 585, "single spatial representation", 14, MUTED)

    arrow(ax, 1174, 439, 1207, 439, MODEL)
    grid_icon(ax, 1210, 402, 64, 64, MODEL)
    text(ax, 1242, 492, "joint tokens", 14, MODEL, "bold")

    # Reconstruction loop communicates the MAE objective without layer detail.
    arrow(ax, 1240, 512, 1110, 660, MUTED, 1.8, .16)
    box(ax, 762, 650, 378, 96, "#F8FAFC", LINE, 1.4, 17)
    text(ax, 951, 680, "Reconstruct all masked modalities", 17, INK, "bold")
    text(ax, 951, 716, r"pocket + ligand + ED · $\mathcal{L}_{MAE}$ on hidden voxels", 14, MUTED)
    arrow(ax, 756, 698, 610, 576, MUTED, 1.8, -.08)
    box(ax, 634, 790, 514, 70, "#F4F7FA", LINE, 1.2, 17)
    text(ax, 891, 812, "Joint representation of pocket–ligand geometry + ED", 15, INK, "bold")
    text(ax, 891, 840, "No early separation into pocket and ligand towers", 13, MUTED)

    # (c) Two simple downstream routes.
    box(ax, 1326, 208, 406, 248, BG, LINE, 1.4, 20, "--")
    badge(ax, 1346, 226, "AFFINITY", "#F3F6F8", LINE, MUTED, 86)
    box(ax, 1350, 282, 124, 66, MODEL_BG, MODEL, 1.3, 14)
    text(ax, 1412, 315, "joint tokens", 12, MODEL, "bold")
    arrow(ax, 1480, 315, 1508, 315, MODEL)
    box(ax, 1514, 282, 86, 66, "white", LINE, 1.3, 14)
    text(ax, 1557, 305, "global", 13, MUTED)
    text(ax, 1557, 327, "pool", 13, INK, "bold")
    arrow(ax, 1606, 315, 1640, 315, TRAIN)
    box(ax, 1646, 282, 62, 66, TRAIN_BG, TRAIN, 1.3, 14)
    text(ax, 1677, 315, "MLP", 14, TRAIN, "bold")
    arrow(ax, 1677, 354, 1677, 383, OUTCOL)
    text(ax, 1677, 407, r"Affinity $\hat{pK}$", 17, OUTCOL, "bold")
    text(ax, 1529, 435, "Frozen encoder · lightweight supervised head", 13, MUTED)

    box(ax, 1326, 492, 406, 354, POCKET_BG, POCKET, 1.5, 20)
    badge(ax, 1346, 510, "VOXBIND", POCKET, POCKET, "white", 86)
    text(ax, 1529, 565, "Generation-time CDG input", 17, INK, "bold")
    box(ax, 1360, 592, 338, 68, "white", LINE, 1.3, 14)
    text(ax, 1529, 615, r"$[\;V_L=0,\;V_P,\;\rho,\;|\nabla\rho|\;]$", 18)
    text(ax, 1529, 644, "ligand masked · pocket + ED retained", 11, MUTED)
    arrow(ax, 1529, 666, 1529, 696, POCKET)
    box(ax, 1362, 700, 164, 62, MODEL_BG, MODEL, 1.3, 14)
    badge(ax, 1372, 708, "FROZEN", "#F3F6F8", LINE, MUTED, 66)
    text(ax, 1444, 742, "CDG encoder", 14, MODEL, "bold")
    arrow(ax, 1532, 731, 1561, 731, TRAIN)
    box(ax, 1567, 700, 136, 62, TRAIN_BG, TRAIN, 1.3, 14)
    text(ax, 1635, 721, "spatial fusion", 12, TRAIN, "bold")
    text(ax, 1635, 744, "into VoxBind", 12, MUTED)
    arrow(ax, 1635, 768, 1635, 798, OUTCOL)
    text(ax, 1635, 820, "denoise → walk–jump", 15, OUTCOL, "bold")
    text(ax, 1529, 872, "Same frozen encoder · task-specific heads", 13, MUTED)

    save(fig, "single_cdg_encoder_overview")


if __name__ == "__main__":
    pretraining()
    voxbind_adapter()
    single_cdg_encoder()
    print(f"Wrote paper overview figures to {OUT}")
