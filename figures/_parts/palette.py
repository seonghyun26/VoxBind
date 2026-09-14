# ══════════════════════════════════════════════════════════════════════════════════════
# fig-palette-methods — the shared method palette, drawn and exported
#
# THE PALETTE IS DEFINED ONCE, IN 00_core.py: COLORS (the saturated hue every method owns),
# SOFT (the lighter tint the eight-method figures draw through soft()) and PALE (the fill
# tints). This figure does not define anything -- it READS those three tables, so the sheet
# and the CSV cannot drift from what the figures actually draw. Regenerate it after any
# palette change and the CSV is the file to hand to a co-author, a slide deck or a notebook
# that has to match these figures without importing draw.py.
#
# WHY THREE TABLES AND NOT ONE. A method is one hue everywhere, but the same hue cannot do
# every job: nine series on one axis need the light end (soft), a box fill under a dark
# median needs lighter still (pale), and a three-line panel wants the saturated original.
# Reading the sheet left to right is reading those three jobs.
# ══════════════════════════════════════════════════════════════════════════════════════

# The nine series of the eight-method PoseCheck figures, in the drawn order, then everything
# else COLORS registers -- our v2 arm and the MCP fine-tune ramp, which are the same FuncBind
# brown darkening with training rather than four categorical hues.
PAL_ORDER = ["Reference ligand", "AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind",
             "TargetDiff", "VoxBind", "CoDE"]
PAL_EXTRA = ["Ours v2", "FuncBind vanilla", "FuncBind ft 3.17M", "FuncBind ft 8.21M",
             "FuncBind ft 26.1M"]
PAL_COLS = ["COLORS · color()", "SOFT · soft()", "PALE · pale()"]
PAL_SW_W, PAL_SW_H = 0.88, 0.66      # swatch, in the one-unit-per-column grid
PAL_ROW_H = 0.42                     # inches per row


def _pal_ink(hexv):
    """Black or white for text ON a swatch, by that swatch's luminance."""
    r, g, b = (int(hexv[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def _pal_rows():
    """(canonical key, palette, soft-or-None, pale-or-None, drawn label) per method.

    SOFT and PALE are read with .get: most methods have no entry, and that is information --
    soft() falls back to the palette colour for them, which is what the sheet shows."""
    rows = []
    for lab in PAL_ORDER + PAL_EXTRA:
        key = ALIASES.get(lab, lab)
        rows.append((key, COLORS[key], SOFT.get(key), PALE.get(key), DISPLAY.get(key, key)))
    return rows


@figure("fig-palette-methods", folder="palette")
def draw_palette_methods(out):
    """Every method's registered tints, as a swatch sheet and a CSV."""
    rows = _pal_rows()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8.4, PAL_ROW_H * len(rows) + 1.25), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_xlim(-2.35, len(PAL_COLS) + 0.05)
        # The bottom edge clears the last row AND the footnote under it.
        ax.set_ylim(-len(rows) - 0.3, 1.45)
        ax.axis("off")
        for j, name in enumerate(PAL_COLS):
            ax.text(j + PAL_SW_W / 2, 0.72, name, ha="center", va="center", fontsize=11.5,
                    color=INK)
        for i, (key, c, s, p, disp) in enumerate(rows):
            y = -i
            ax.text(-0.16, y, disp, ha="right", va="center", fontsize=12, color=INK)
            for j, hexv in enumerate((c, s, p)):
                if not hexv:
                    # No entry: soft()/pale() do not invent one, and neither does this sheet.
                    ax.text(j + PAL_SW_W / 2, y, "—", ha="center", va="center", fontsize=11,
                            color=AXIS)
                    continue
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j, y - PAL_SW_H / 2), PAL_SW_W, PAL_SW_H, facecolor=hexv,
                    edgecolor=LEGEND_EDGE, linewidth=0.6))
                ax.text(j + PAL_SW_W / 2, y, hexv.upper(), ha="center", va="center",
                        fontsize=9, color=_pal_ink(hexv))
        ax.text(-2.3, -len(rows) + 0.1,
                "soft() falls back to the palette colour where SOFT has no entry; "
                "pale() raises instead.",
                ha="left", va="center", fontsize=9.5, color=AXIS)
        fit(fig, pad=0.4)
        save(fig, out, "palette_methods")
    write_csv(out, "palette_methods",
              ["method", "palette_hex", "soft_hex", "pale_hex", "legend_label"],
              [[k, c, s or "", p or "", d] for k, c, s, p, d in rows])
    print(f"  {len(rows)} methods · {sum(1 for r in rows if r[2])} with a soft tint · "
          f"{sum(1 for r in rows if r[3])} with a pale tint")
    print(f"  wrote {out}")
