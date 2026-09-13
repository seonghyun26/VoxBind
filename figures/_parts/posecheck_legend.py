

# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-legend — the per-atom all-methods strain key, as its own image
# ════════════════════════════════════════════════════════════════════════════════
# strain_per_atom_all_methods_{mean,median} dropped their key on request (2026-09-13); this
# draws that key alone, to be placed beside or under the panels. The layout is the Vina
# per-atom v3 key's (dock-per-atom-v3-*-all): the reference on a centred line of its own
# over the eight methods in 2 rows x 4 columns reading across, one frame round both. No n:
# the reference is the section's own crystal ligands, and the panel itself says nothing
# about counts.
#
# THE SWATCHES ARE THE PANEL'S, not the rotatable-bond figures' or the ECDF's -- the three
# strain families do not draw a method identically. The per-atom panel takes its dash from
# the rotatable-bond family (RB_BASE_DASH, solid for the local arms), its WEIGHT from the
# ECDF family (PCSZ_LINE_LW x PCSZ_LW_SCALE) and its colour through soft(), in RB_ORDER.
# Every one of those is read at call time: they live in parts assembled after this one.
PLEG_NCOL = 4
# SWATCH LENGTH, in font sizes. The shared legend() fixes 1.9, and at the panel's weights that
# is shorter than one period of DecompDiff's (9, 3) dash -- its swatch read as a solid line,
# i.e. as one of the local arms -- and the reference showed a dash and a half. 3.6 carries at
# least two periods of every pattern in RB_BASE_DASH.
PLEG_HANDLE_LEN = 3.6


def _pleg_method_handles():
    out = []
    for lab in RB_ORDER:
        out.append(Line2D([], [], color=soft(lab), ls=RB_BASE_DASH.get(lab, "-"),
                          lw=PCSZ_LINE_LW[ALIASES.get(lab, lab)] * PCSZ_LW_SCALE,
                          solid_capstyle="round", label=display(lab)))
    return out


def _pleg_legend(fig, handles, **kw):
    """legend()'s look -- white face, the axis-pen frame, INK text -- with a swatch long
    enough to show a dash. Its frame is removed again by _vpa_one_frame."""
    leg = fig.legend(handles=handles, frameon=True, fontsize=11.5, handlelength=PLEG_HANDLE_LEN,
                     handletextpad=0.6, labelspacing=0.3, borderpad=0.4, borderaxespad=0.39,
                     facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(AXIS_LW)
    # above the shared frame _vpa_one_frame draws at zorder 6, as legend() does -- without it
    # the white frame is painted over every entry and the image is an empty box
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


@figure("fig-posecheck-strain-legend", folder="fig-posecheck/strain-energy")
def draw_posecheck_strain_legend(out):
    """The key the per-atom all-methods strain panels dropped: reference on top, methods 2x4."""
    use_style()
    ref = [Line2D([], [], color=REF_COLOR, ls=DASH, lw=PCSZ_REF_LW * PCSZ_LW_SCALE,
                  dash_capstyle="round", label=REF_LABEL)]
    # PCSZ_RC for the panel's 170 dpi and its crop to the ink, so the key and the panels it
    # sits beside are rasterised the same way.
    with plt.rc_context(PCSZ_RC):
        fig = plt.figure(figsize=(10.0, 1.6))
        fig.patch.set_facecolor("white")
        # Two legends in one frame, exactly as the v3 key: a legend column is as wide as its
        # widest entry, so the reference cannot be a cell of the grid without unevening it.
        legs = [_pleg_legend(fig, _vpa_row_major(_pleg_method_handles(), PLEG_NCOL),
                             loc="lower center", ncol=PLEG_NCOL, bbox_to_anchor=(0.5, 0.05),
                             columnspacing=1.5)]
        fig.canvas.draw()
        h = legs[0].get_window_extent(fig.canvas.get_renderer()).height / \
            (fig.get_size_inches()[1] * fig.dpi)
        legs.append(_pleg_legend(fig, ref, loc="lower center", ncol=1,
                                 bbox_to_anchor=(0.5, 0.05 + h + VPA_KEY_ROW_GAP)))
        _vpa_one_frame(fig, legs)
        print(f"  key: {REF_LABEL} over {len(RB_ORDER)} methods in {PLEG_NCOL} columns · "
              f"dash from RB_BASE_DASH, weight from PCSZ_LINE_LW, colour soft()")
        save(fig, out, "strain_per_atom_all_methods_legend")
