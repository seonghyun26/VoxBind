

# ════════════════════════════════════════════════════════════════════════════════
# fig-posebusters-{valid-per-atom,valid-heatmap,check-failures,sucos-ecdf,sucos-per-atom}
# ════════════════════════════════════════════════════════════════════════════════
# SIZE IS THE CONFOUND, SO SIZE IS THE X AXIS. Every PoseBusters check gets harder as the
# molecule grows -- more rings to pucker, more angles to strain, more atoms to reach the
# protein -- and the arms draw different size mixes (24.9 heavy atoms for CoDE against
# TargetDiff's 22.2). A pooled validity rate is therefore partly a report of the size mix,
# the same trap the Vina numbers have, which is why the headline figure is per-atom with
# the size distribution drawn underneath it rather than a single pooled bar.
#
# SuCOS is the one check PoseBusters' `gen` config adds over `dock`: shape overlap x
# pharmacophore-feature overlap against the pocket's crystal ligand. It is NOT folded into
# `valid` -- every other column asks "is this pose physically possible", SuCOS asks "does it
# sit where the crystal ligand sits", and a de novo model is not trying to reproduce the
# crystal ligand. Its per-molecule values are recomputed by
# notebook/html/260910/fig-posebusters/build_sucos.py, which needs RDKit and the posebusters
# package (the `moleval` env); this file only draws what that script exported.
#
# THE FOUR CHECKED-IN sucos_*.png WERE RASTERIZED IN THAT `moleval` ENV, and it carries
# FreeType 2.14.3 against `voxbind`'s 2.6.1. Same matplotlib, same data, same code -- but a
# different glyph rasterizer, and tight_layout measures the axes off the text it lays out,
# so a redraw here is sub-pixel-shifted against them over the whole figure. Drawing them
# from `voxbind` is correct and identical to what build_sucos.py itself produces there; it
# just cannot be byte-compared against PNGs made by the other env's FreeType.

# Validity is a proportion, so its per-count curve is smoothed over +-PB_VALID_WIN atoms --
# see model_curve for why the strain and clash curves are not.
PB_VALID_WIN = 2
# A check no method fails above this often is a row of white space in the breakdown: it
# says only that PoseBusters ran it. The full counts stay in the JSON the recompute script
# writes. The filter stays a RATE even though the bars are counts -- it asks "is this row
# informative", and the arms hold different numbers of molecules.
PB_MIN_FAIL_PCT = 0.5
# The check-failure breakdown, split into stacked panels by how often a check fails, each on
# a linear axis scaled to itself (2026-09-14, on request; it was one symlog axis). The median
# rate over the eight arms puts them in three decades:
#   ~10%   bond angles 9.7, min distance 7.0, steric clash 5.7, ring non-flatness 5.4,
#          bond lengths 3.2
#   ~1%    internal energy 1.6, volume overlap 0.36
#   <=0.1% double bond flatness 0.26, aromatic ring flatness 0.02 -- zero for most sets
# Fixed rather than recomputed, so `core` and `all` split the same way; inside a panel the
# checks keep the figure's order (worst first). A check shown but in no panel is an error.
PB_FAIL_PANELS = (
    ("bond_lengths", "bond_angles", "non-aromatic_ring_non-flatness",
     "minimum_distance_to_protein", "internal_steric_clash"),
    ("volume_overlap_with_protein", "internal_energy"),
    ("aromatic_ring_flatness", "double_bond_flatness"),
)
# THE HEAT MAP'S GROUPS. The 20 scored dock-mode checks partitioned into what each one is
# actually asking about, so `valid` can be read as "passes all five" instead of as one
# number. EVERY CHECK IS IN EXACTLY ONE GROUP and the partition is asserted against the
# data: a PoseBusters release that adds a check breaks the build rather than quietly
# dropping it out of the picture, and no check can be double-counted into two groups.
#
# Six of the nine in `Protein clash & overlap` are VACUOUS HERE and pass by construction:
# the receptor every arm is scored against is the pocket10 crop, which carries no HETATM
# and no HOH, so the cofactor and water checks have nothing to measure. They stay in the
# group because the group is a partition of `valid` and dropping them would make the
# columns stop multiplying out to it -- not because they carry information.
#
# `Ring pucker & internal strain` is not one of the four groups this was asked for, and it
# exists because the partition has to be complete: `non-aromatic_ring_non-flatness` is the
# single worst check for the voxel arms (21-23%), and a breakdown that left it out would
# show CoDE passing every column it draws while failing 33% of molecules.
PB_GROUPS = (
    ("Bond geometry", ("bond_lengths", "bond_angles")),
    ("Aromatic ring flatness", ("aromatic_ring_flatness",)),
    ("Valence & connectivity", ("sanitization", "all_atoms_connected", "inchi_convertible",
                                "no_radicals")),
    ("Protein clash & overlap", ("minimum_distance_to_protein", "volume_overlap_with_protein",
                                 "protein-ligand_maximum_distance",
                                 "minimum_distance_to_organic_cofactors",
                                 "minimum_distance_to_inorganic_cofactors",
                                 "minimum_distance_to_waters",
                                 "volume_overlap_with_organic_cofactors",
                                 "volume_overlap_with_inorganic_cofactors",
                                 "volume_overlap_with_waters")),
    ("Ring pucker & internal strain", ("non-aromatic_ring_non-flatness",
                                       "internal_steric_clash", "internal_energy",
                                       "double_bond_flatness")),
)
PB_HEAT_COL0 = "PoseBusters valid"
# THE RAMP IS THE DATA'S OWN RANGE: it ends at 100, a rate's ceiling, and starts at the
# WORST CELL DRAWN (2026-09-14, on request) -- FuncBind's 50.1% valid. A 0-100 ramp would
# spend half its range on ground no method stands on and paint the whole map one shade,
# which is the failure mode of a heat map whose numbers all sit near the top.
#
# Both variants take the range of the ALL-ARMS map, so a cell is the same colour in `core`
# as in `all`. Scaling `core` to its own worst cell (CoDE's 67.5) would make our two arms
# look as far apart as the eight are.
PB_HEAT_VMAX = 100.0
# THE COLOURBAR IS THE PER-ROTBOND GRID'S (2026-09-14, on request): vertical on the right,
# shrink 0.92, aspect 38, RB_GRID_CBAR_PAD off the panel, numbers turned a quarter turn so
# they read along the bar, no outline. Those constants are READ FROM THE ROTBOND FAMILY at
# call time rather than copied here -- a part is pasted into draw.py above posecheck_rotbond,
# so the names do not exist while this module executes, but they do by the time a figure is
# drawn, and reading them is what keeps the two bars from drifting apart.
# The colourmap is that grid's coolwarm REVERSED. In the strain grid warm means more strain,
# i.e. worse; this cell is a PASS rate, so without the reversal the same red would mean good
# here and bad there in two heat maps of the same eight methods.
PB_HEAT_CMAP = "coolwarm_r"
# Point sizes are NOT copied from the rotbond grid: that figure is three rows of nine panels
# and carries 18 pt numbers, which on this 8.4 x 6.2 in map would tower over the 12.5 pt in
# the cells. Same bar, this figure's scale.
PB_HEAT_CBAR_FS, PB_HEAT_CBAR_TICK_FS = 13, 11.5
PB_HEAT_ROW_H = 0.44             # inches per method row
PB_HEAT_WRAP = 12                # characters per line of a column heading
PB_SUCOS_THRESHOLD = 0.4         # gen.yml's own value
PB_BIN_COLS = ["n_molecules", "atoms_mean", "n_posebusters", "pb_valid_rate",
               "pb_valid_rate_size_standardized"]


def _posebusters_rate(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def _posebusters_std_weights(data, p79_rows):
    """The common size distribution for direct standardization -- every arm pooled.

    The weights come from the arms that are actually IN the comparison, not from every key
    the loader returned: a scoring run in progress leaves partial rows in p79_rows, and
    folding those into the standard population moves every arm's standardized rate."""
    return collections.Counter(
        r["n"] for _, key, _ in arms_for("v", data) for r in p79_rows[key]
        if r["v"] is not None)


def _posebusters_std_rate(rows, weights):
    """Each arm's rate WITHIN every 1-heavy-atom stratum, re-weighted by one common size
    distribution, so what is left is validity at matched size. Strata where the arm holds
    <10 molecules are dropped, not extrapolated. This is the number to compare arms with;
    the crude rate is what the arm actually produced, and the two answer different
    questions."""
    per = by_size(rows, "v")
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * float(np.mean(per[n])); den += w
    return 100 * num / den if den else float("nan")


def _posebusters_block(rows, weights):
    v = [r["v"] for r in rows if r["v"] is not None]
    std = _posebusters_std_rate(rows, weights)
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_posebusters": len(v),
        "pb_valid_rate": round(_posebusters_rate(v) / 100, 4) if v else None,
        "pb_valid_rate_size_standardized": round(std / 100, 4) if np.isfinite(std) else None,
    }


# ── figure 1: validity per heavy-atom count, over each arm's size distribution ───
def _posebusters_valid_panel(out, arms, variant, p79_rows, refrows):
    per = {key: by_size(p79_rows[key], "v") for _, key, _ in arms}
    ref_per = by_size(refrows, "v")
    xs = x_range(per, arms)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    rate100 = lambda v: 100 * float(np.mean(v))
    top.plot(xs, reference_curve(ref_per, xs, rate100), color=REF_COLOR,
             lw=REF_LW, ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, model_curve(per[key], xs, rate100, win=PB_VALID_WIN),
                 color=soft(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    legend(top, arm_handles(arms, paint=soft), loc="lower left", fontsize=11.5)

    size_distribution(bot, xs, per, ref_per, arms, paint=soft)
    furniture(bot, ylabel="% of ligands", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    fig.align_ylabels((top, bot))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, out, f"pb_valid_per_atom_{variant}")
    return xs


# ── figure 2: which checks fail ─────────────────────────────────────────────────
def _posebusters_check_failures(data, p79_rows, refrows):
    """{key: {n_mols, counts, rates}} over every arm, plus the crystal ligands."""
    out = {}
    for lab, key, _ in arms_for("v", data):
        rows = [r for r in p79_rows[key] if r["v"] is not None]
        cnt = collections.Counter(c for r in rows for c in r["f"])
        out[key] = {"n_mols": len(rows), "counts": dict(cnt),
                    "rates": {k: 100 * v / len(rows) for k, v in cnt.items()}}
    rrows = [r for r in refrows if r["v"] is not None]
    cnt = collections.Counter(c for r in rrows for c in r["f"])
    out["reference"] = {"n_mols": len(rrows), "counts": dict(cnt),
                        "rates": {k: 100 * v / max(1, len(rrows)) for k, v in cnt.items()}}
    return out


def _posebusters_wrap_check(name):
    """The PoseBusters check name, wrapped instead of abbreviated.

    `non-aromatic_ring_non-flatness` on one line is 30 characters and was eating 2.9 in of
    a 7.6 in figure -- nearly half the width -- as a tick label. Abbreviating it is the
    wrong fix: this check passes when a non-aromatic ring is sufficiently NON-flat (it is
    `check_nonflat: True` in dock.yml, threshold 0.1 A), so failing it means a saturated
    ring came out planar, and every shortening of that name I tried either flipped its
    sense or read as the aromatic check next to it. Wrapping is free and exact.

    textwrap is not one of draw.py's shared imports and this is the only figure that needs
    it, so it is fetched here rather than at the top of the file. Its exact wrapping --
    including breaking on the hyphens -- is what the tick labels are, so this must stay
    textwrap and not a hand-rolled splitter."""
    return "\n".join(__import__("textwrap").wrap(name.replace("_", " "), 18))


def _posebusters_failures_panel(out, arms, variant, fails):
    """Rows are checks, bars are the SHARE of each set's molecules that fail them.

    Rates, not counts, and that is what lets the crystal ligands be an ordinary bar here:
    on a count axis 79 of them against ~7,900 generated molecules put their worst row at 2
    molecules, invisible beside a bar of 1,822, and they had to be drawn as a rate-matched
    marker instead. The price is that a rate hides its denominator -- the reference's 2.5%
    IS those 2 molecules, and it carries about +-1.8 points of binomial noise against the
    arms' +-0.2 -- so its bar alone keeps its n in the key. Counts for every arm and every
    check stay in posebusters_check_failures.json.
    """
    # The reference is one more series, first in every group and first in the key.
    series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    shown = [fails[key] for _, key in series]
    names = sorted({k for g in shown for k in g["counts"]
                    if max(h["rates"].get(k, 0) for h in shown) >= PB_MIN_FAIL_PCT},
                   key=lambda k: -max(g["rates"].get(k, 0) for g in shown))
    # One row of the y axis is 1.0 apart, so the bars of a group must fit inside that:
    # a fixed height works for three series and silently overlaps the neighbouring groups
    # at nine (9 x 0.19 = 1.71), which reads as bars detached from their labels. Derive it.
    h = 0.86 / len(series)
    # THE KEY GOES OUTSIDE THE AXES, centred along the bottom, 3 x 3. There is no empty
    # corner inside:
    # the long bars fill the top and the right, and the in-axes box this used to carry
    # reached far enough left to bury the bottom rows' bars -- `double bond flatness`
    # looked empty while AR, DecompDiff and Pocket2Mol were failing it 3.1, 1.8 and 1.4% of
    # the time. Outside it covers nothing. Three columns only fit because the bars are
    # rates now: an arm's own n no longer has to be in the key for its bar to be readable.
    ncol = 3
    leg_rows = -(-len(series) // ncol)
    # Row pitch: enough that nine thin bars stay readable, without turning a 7.6 in wide
    # figure into a 10 in tall one. Plus the strip the key needs at the bottom.
    # STACKED PANELS, LINEAR X (2026-09-14, on request; it was one symlog axis). The rates run
    # 0.01% to 23%, and on one linear axis everything under ~2% was a stub against FuncBind's
    # 23%; symlog fixed that but made every bar's length a log reading. Three panels -- see
    # PB_FAIL_PANELS -- each give a decade of checks its own linear scale, so a length is a
    # rate again, and a zero is still a bar of nothing. Panel height = its check count, so
    # the bars are the same thickness in every panel; the x scales differ, so each keeps
    # its own tick labels.
    panels = [[k for k in names if k in p] for p in PB_FAIL_PANELS]
    leftover = [k for k in names if not any(k in p for p in PB_FAIL_PANELS)]
    if leftover:
        raise ValueError(f"PoseBusters checks in no PB_FAIL_PANELS panel: {leftover}")
    panels = [p for p in panels if p]
    fig = plt.figure(figsize=(FIG_W, (0.40 if len(arms) <= 3 else 0.62) * len(names)
                              + 1.6 + 0.45 * (len(panels) - 1) + 0.30 * leg_rows + 0.2),
                     dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[len(p) for p in panels])
    for j, panel in enumerate(panels):
        ax = fig.add_subplot(gs[j, 0])
        ax.set_facecolor("white")
        ys = np.arange(len(panel))[::-1]
        for i, (lab, key) in enumerate(series):
            # top of the group downwards, so the key's order IS the order of the bars
            ax.barh(ys + ((len(series) - 1) / 2 - i) * h,
                    [fails[key]["rates"].get(k, 0.0) for k in panel],
                    height=h, color=soft(lab), edgecolor=soft(lab), lw=0.8, zorder=3)
        top = max(g["rates"].get(k, 0.0) for g in shown for k in panel)
        furniture(ax, ylabel=None, xlim=(0, top * 1.08), xloc=None,
                  xlabel="Molecules failing the check (%)" if j == len(panels) - 1 else None)
        ax.grid(False, axis="y")
        ax.set_yticks(ys)
        ax.set_yticklabels([_posebusters_wrap_check(k) for k in panel], fontsize=12)
        ax.set_ylim(-0.6, len(panel) - 0.4)
    # soft() tints, the eight-method figures' colours (2026-09-14, on request)
    handles = [Patch(facecolor=soft(lab), edgecolor=soft(lab),
                     label=display(lab) + ("  (n=79)" if key == "reference" else ""))
               for lab, key in series]
    fit(fig, pad=0.5)
    # tight_layout does not see a FIGURE legend, so it has just laid the axes out over the
    # strip the key occupies. Place the key, measure what it actually took, and RAISE the
    # axes by that much -- reserving the measured height keeps the tick labels and the x
    # label, which sit BELOW the axes box and would otherwise end up behind an opaque
    # legend; reserving up to its top edge instead would bury them. Columns are dropped
    # if the key overruns the figure, which is cheaper than trusting a width estimate.
    fs = 12.5 if len(arms) <= 3 else 11
    for c in range(ncol, 0, -1):
        leg = legend(fig, handles, loc="lower center", ncol=c, fontsize=fs,
                     bbox_to_anchor=(0.5, 0.008))
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        if (bb.x0 >= 0.003 and bb.x1 <= 0.997) or c == 1:
            break
        leg.remove()
    fig.subplots_adjust(bottom=min(0.6, fig.subplotpars.bottom
                                  + (bb.y1 - bb.y0) + 0.016))
    save(fig, out, f"pb_check_failures_{variant}")


@figure("fig-posebusters-valid-per-atom", needs=("metrics.json (posebusters block)",))
def draw_posebusters_valid_per_atom(out):
    """PoseBusters dock-mode validity against ligand size, over each arm's size mix."""
    data, p79_rows, refrows = pose_data()
    use_style()
    weights = _posebusters_std_weights(data, p79_rows)
    ranges = {}
    for variant, arms in variants("v", data):
        ranges[variant] = _posebusters_valid_panel(out, arms, variant, p79_rows, refrows)

    scored = arms_for("v", data)
    pooled = {lab: _posebusters_block(p79_rows[key], weights) for lab, key, _ in scored}
    ref_block = _posebusters_block(refrows, weights)
    by_bin = {lab: [_posebusters_block([r for r in p79_rows[key] if bin_of(r["n"]) == b],
                                       weights)
                    for b in range(len(BIN_LABELS))] for lab, key, _ in scored}
    by_bin[REF_LABEL] = [_posebusters_block([r for r in refrows if bin_of(r["n"]) == b],
                                            weights)
                         for b in range(len(BIN_LABELS))]

    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    print(f"  {'arm':16s} {'atoms':>6s} {'PB-valid':>9s} {'size-std':>9s} {'mols':>7s}")
    for lab, key, _ in scored:
        r = pooled[lab]
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {100 * r['pb_valid_rate']:8.1f}% "
              f"{100 * r['pb_valid_rate_size_standardized']:8.1f}% {r['n_molecules']:7d}")
    print(f"  {REF_LABEL:16s} {ref_block['atoms_mean']:6.1f} "
          f"{100 * ref_block['pb_valid_rate']:8.1f}% {'—':>9s} "
          f"{ref_block['n_molecules']:7d}")
    print(f"\n  PB-valid by bin ({' · '.join(BIN_LABELS)}):")
    for lab, key, _ in scored:
        print(f"    {lab:16s} " + "  ".join(
            f"{100 * b['pb_valid_rate']:5.1f}%" if b["pb_valid_rate"] is not None else "    -"
            for b in by_bin[lab]))

    write_csv(out, "posebusters_by_atom_range", ["arm", "bin"] + PB_BIN_COLS,
              [[arm, lab] + [r[c] for c in PB_BIN_COLS]
               for arm, rowset in by_bin.items()
               for lab, r in zip(BIN_LABELS, rowset)])


@figure("fig-posebusters-check-failures", needs=("metrics.json (posebusters block)",))
def draw_posebusters_check_failures(out):
    """Which PoseBusters checks fail, per method, as a share of its own molecules."""
    data, p79_rows, refrows = pose_data()
    use_style()
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    for variant, arms in variants("v", data):
        _posebusters_failures_panel(out, arms, variant, fails)


# ── validity heat map ───────────────────────────────────────────────────────────
def _posebusters_group_rates(rows, checked):
    """(% passing each group, % valid, n scored) for one set of molecules.

    A GROUP IS PASSED WHEN THE MOLECULE FAILS NOTHING IN IT, which is per-molecule and not
    recoverable from the per-check rates: two checks each failing 5% of molecules are one
    column at 90% if they fail different molecules and at 95% if they fail the same ones.
    That is also why the group columns do not multiply out to the `valid` column."""
    scored = [r for r in rows if r["v"] is not None]
    n = len(scored)
    if not n:
        return {}, float("nan"), 0
    seen = {c for r in scored for c in r["f"]}
    if not seen <= checked:
        raise KeyError(f"PoseBusters check outside PB_GROUPS: {sorted(seen - checked)} -- "
                       "add it to a group, or the heat map stops being a partition of `valid`")
    out = {}
    for name, checks in PB_GROUPS:
        want = set(checks)
        out[name] = 100 * sum(not (want & set(r["f"])) for r in scored) / n
    return out, 100 * sum(bool(r["v"]) for r in scored) / n, n


def _posebusters_heatmap_grid(arms, p79_rows, refrows):
    """(column names, row labels, the rates, n per row) -- the map, before it is drawn."""
    series = [(REF_LABEL, refrows)] + [(lab, p79_rows[key]) for lab, key, _ in arms]
    checked = {c for _, checks in PB_GROUPS for c in checks}
    cols = [PB_HEAT_COL0] + [name for name, _ in PB_GROUPS]
    grid, ns = [], []
    for lab, rows in series:
        groups, valid, n = _posebusters_group_rates(rows, checked)
        grid.append([valid] + [groups[name] for name, _ in PB_GROUPS])
        ns.append(n)
    return cols, [lab for lab, _ in series], np.array(grid, float), ns


def _posebusters_heatmap_panel(out, variant, built, norm):
    cols, labels, grid, ns = built
    series = list(zip(labels, ns))

    # The constant was 2.25 while the bar lay along the bottom; with it now on the right it
    # only has to buy the three-line column headings. The floor is for `core`: three rows
    # leave a vertical bar at aspect 38 too short for its own name and tick numbers.
    # CONSTRAINED LAYOUT, as the rotbond grid uses -- tight_layout does not see a colourbar's
    # label, and fit()'s overrun correction moves the main axes, not the bar's, so the name
    # came out sliced off the right edge of the core map.
    fig, ax = plt.subplots(figsize=(FIG_W * 1.16,
                                    max(PB_HEAT_ROW_H * len(series) + 1.3, 3.3)),
                           dpi=220, layout="constrained")
    fig.patch.set_facecolor("white")
    cmap = matplotlib.colormaps[PB_HEAT_CMAP]
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(series)):
        for j in range(len(cols)):
            v = grid[i, j]
            # White on a dark cell, ink on a light one, off the fill's own luminance: a
            # diverging map is dark at BOTH ends, so a single threshold on the value would
            # put ink on the dark red of a 50% cell and white on the pale middle.
            r, g, b, _ = cmap(norm(v))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=12.5,
                    color="white" if dark else INK, zorder=3)
    # The first column is the aggregate, not a sixth category, and the groups do not multiply
    # out to it (see _posebusters_group_rates) -- a rule wide enough to read as a break says so.
    ax.axvline(0.5, color="white", lw=5, zorder=4)
    ax.set_xticks(np.arange(len(cols)))
    # Wrapped at PB_HEAT_WRAP, not at the 18 the check-failure tick labels use: a column here
    # is ~1.2 in wide, and at 18 the group names ran into each other across the header.
    ax.set_xticklabels(["\n".join(__import__("textwrap").wrap(c, PB_HEAT_WRAP))
                        for c in cols], fontsize=12.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(series)))
    ax.set_yticklabels([f"{display(lab)}  ({n:,})" for lab, n in series], fontsize=13)
    ax.tick_params(length=0, colors=INK, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # White rules on the cell boundaries rather than a frame: the cells are the figure.
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(series), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      shrink=0.92, aspect=38, pad=RB_GRID_CBAR_PAD)
    cb.set_label("% of molecules passing", fontsize=PB_HEAT_CBAR_FS, color=INK,
                 labelpad=RB_GRID_CBAR_LABELPAD)
    cb.ax.tick_params(labelsize=PB_HEAT_CBAR_TICK_FS, colors=AXIS, width=AXIS_LW)
    plt.setp(cb.ax.get_yticklabels(), rotation=90, va="center", ha="left")
    cb.outline.set_visible(False)
    save(fig, out, f"pb_valid_heatmap_{variant}")


@figure("fig-posebusters-valid-heatmap", needs=("metrics.json (posebusters block)",))
def draw_posebusters_valid_heatmap(out):
    """PoseBusters validity as a method x check-group heat map, with the per-check table."""
    data, p79_rows, refrows = pose_data()
    use_style()
    drawn = {variant: _posebusters_heatmap_grid(arms, p79_rows, refrows)
             for variant, arms in variants("v", data)}
    # One ramp for both maps, floored at the worst cell of the widest one.
    norm = matplotlib.colors.Normalize(float(drawn["all"][2].min()), PB_HEAT_VMAX)
    for variant, built in drawn.items():
        _posebusters_heatmap_panel(out, variant, built, norm)

    cols, labels, grid, ns = drawn["all"]
    worst = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"  colour ramp {norm.vmin:.1f}-{norm.vmax:.0f}%, floored at the worst cell: "
          f"{labels[worst[0]]} / {cols[worst[1]]}")
    print(f"  {'method':16s} {'n':>7s} " + " ".join(f"{c.split(' ')[0][:9]:>9s}" for c in cols))
    for lab, row, n in zip(labels, grid, ns):
        print(f"  {lab:16s} {n:7,d} " + " ".join(f"{v:8.1f}%" for v in row))
    write_csv(out, "posebusters_valid_heatmap",
              ["method", "n_molecules"] + [f"pct_pass_{c}" for c in cols],
              [[lab, n] + [round(float(v), 2) for v in row]
               for lab, row, n in zip(labels, grid, ns)])

    # The per-check table the groups are built from -- a group at 99.9% can be one check at
    # 99.9% or four at 100 and one at 99.9, and only this says which.
    checks = [c for _, cs in PB_GROUPS for c in cs]
    group_of = {c: name for name, cs in PB_GROUPS for c in cs}
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    rows = []
    for lab, key, _ in [(REF_LABEL, "reference", None)] + list(arms_for("v", data)):
        g = fails[key]
        rows += [[lab, g["n_mols"], group_of[c], c, g["counts"].get(c, 0),
                  round(100 - g["rates"].get(c, 0.0), 3)] for c in checks]
    write_csv(out, "posebusters_check_pass_rates",
              ["method", "n_molecules", "group", "check", "n_failed", "pct_pass"], rows)


# ── SuCOS ───────────────────────────────────────────────────────────────────────
_PB_SUCOS_CACHE = {}


def _posebusters_sucos_rows():
    """{key: [{t, n, sucos}]}, from the per-molecule export.

    build_sucos.py computes these with `check_sucos(..., sucos_threshold=0.4)` -- exactly
    what gen.yml configures -- and caches them beside itself; it needs RDKit, posebusters
    and the `moleval` env, none of which drawing does. Reading its export is the same code
    path that script itself takes on a warm cache."""
    if not _PB_SUCOS_CACHE:
        for _, key, _ in ARMS:
            path = legacy("fig-posebusters", f"sucos_per_molecule_{key}.json")
            _PB_SUCOS_CACHE[key] = json.load(open(path))["molecules"]
    return _PB_SUCOS_CACHE


def _posebusters_sucos_p79():
    """The same, restricted to the 79-pocket like-for-like set."""
    p79 = set(P79)
    return {key: [r for r in rows if r["t"] in p79]
            for key, rows in _posebusters_sucos_rows().items()}


def _posebusters_sucos_standardized(rows, weights, f):
    """The arm's SuCOS WITHIN each 1-heavy-atom stratum, re-weighted by `weights`. SuCOS
    turns out to be nearly flat in size, so this barely moves -- which is itself worth
    recording, because every other metric in this section is size-confounded."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["n"]].append(r["sucos"])
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * f(per[n]); den += w
    return num / den if den else float("nan")


def _posebusters_sucos_paired(a_rows, b_rows):
    """Per-pocket mean SuCOS, a - b, over the pockets both cover. Paired because the
    pockets differ from each other far more than the arms do: an unpaired comparison of
    two arms is mostly a comparison of which pockets each happened to cover."""
    def by_pocket(rows):
        d = collections.defaultdict(list)
        for r in rows:
            d[r["t"]].append(r["sucos"])
        return {t: float(np.mean(v)) for t, v in d.items()}
    a, b = by_pocket(a_rows), by_pocket(b_rows)
    ts = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in ts]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    half = 1.96 * sd / max(1, len(diff)) ** 0.5
    return {"n_pockets": len(ts), "mean_diff": round(float(np.mean(diff)), 4),
            "median_diff": round(st.median(diff), 4),
            "ci95": [round(float(np.mean(diff)) - half, 4),
                     round(float(np.mean(diff)) + half, 4)],
            "a_higher_in": sum(d > 0 for d in diff)}


@figure("fig-posebusters-sucos-ecdf", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_ecdf(out):
    """SuCOS against the pocket's crystal ligand, as a distribution, with gen's 0.4 mark."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    for variant, arms in variants():
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for lab, key, _ in arms:
            v = np.sort([r["sucos"] for r in p79_rows[key]])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=soft(lab),
                    lw=MODEL_LW, zorder=5, solid_capstyle="round")
        # gen.yml would call everything left of this line invalid.
        ax.axvline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
        # Horizontal and high: rotated against the line it ran straight through the curves.
        ax.text(PB_SUCOS_THRESHOLD + 0.015, 0.97, f"gen threshold {PB_SUCOS_THRESHOLD}",
                ha="left", va="top", fontsize=11, color=INK)
        furniture(ax, ylabel="Cumulative share",
                  xlabel="SuCOS vs the pocket's crystal ligand", xloc=None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        legend(ax, arm_handles(arms, include_ref=False, paint=soft), loc="upper left",
               fontsize=11.5)
        fit(fig, pad=0.5)
        save(fig, out, f"sucos_ecdf_{variant}")

    # One common size distribution -- every arm pooled -- for direct standardization.
    weights = collections.Counter(r["n"] for rows in p79_rows.values() for r in rows)
    print(f"\n  {'arm':16s} {'mean':>7s} {'median':>7s} {'q25':>7s} {'q75':>7s} "
          f"{'std mean':>9s} {'>=0.4':>8s} {'mols':>7s}")
    rates = {}
    for lab, key, _ in ARMS:
        v = [r["sucos"] for r in p79_rows[key]]
        over = 100 * sum(x >= PB_SUCOS_THRESHOLD for x in v) / len(v)
        rates[lab] = round(over / 100, 4)
        std = round(_posebusters_sucos_standardized(p79_rows[key], weights, np.mean), 4)
        print(f"  {lab:16s} {np.mean(v):7.3f} {st.median(v):7.3f} "
              f"{np.percentile(v, 25):7.3f} {np.percentile(v, 75):7.3f} "
              f"{std:9.3f} {over:7.1f}% {len(v):7d}")

    # The claim this figure exists to support, stated as a paired test rather than a
    # difference of two pooled means.
    print()
    for name, (a, b) in (("CoDE - VoxBind", ("ours_v1", "vanilla")),
                         ("CoDE - TargetDiff", ("ours_v1", "targetdiff")),
                         ("CoDE - DecompDiff", ("ours_v1", "decompdiff"))):
        d = _posebusters_sucos_paired(p79_rows[a], p79_rows[b])
        print(f"    {name:30s} {d['mean_diff']:+.4f} "
              f"(95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}), "
              f"higher in {d['a_higher_in']}/{d['n_pockets']} pockets")

    # What running config="gen" instead of "dock" would have cost, stated rather than
    # implied: `valid` there is all-must-pass INCLUDING sucos_within_threshold.
    try:
        pb = json.load(open(legacy("fig-posebusters", "posebusters_summary.json")))
        print(f"\n  {'arm':16s} {'dock valid':>11s} {'gen valid (upper bound)':>24s}")
        for lab, key, _ in ARMS:
            # An arm whose PoseBusters run has not finished is simply absent from that
            # summary; skip its row rather than dropping the whole table.
            if lab not in pb.get("arms", {}):
                continue
            dock = pb["arms"][lab]["p79"]["pb_valid_rate"]
            print(f"  {lab:16s} {100 * dock:10.1f}% {100 * dock * rates[lab]:23.1f}%")
        print("    (upper bound: the product assumes the two are independent; the true gen\n"
              "     rate is the share passing BOTH, which cannot exceed either factor)")
    except (OSError, KeyError):
        pass


@figure("fig-posebusters-sucos-per-atom", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_per_atom(out):
    """SuCOS against ligand size — mean and median, core and all."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    ranges = {}
    for variant, arms in variants():
        per = {key: by_size(p79_rows[key], "sucos") for _, key, _ in arms}
        xs = x_range(per, arms)
        for stat, f in (("mean", lambda v: float(np.mean(v))),
                        ("median", lambda v: float(np.median(v)))):
            fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            for lab, key, _ in arms:
                ax.plot(xs, model_curve(per[key], xs, f), color=soft(lab),
                        lw=MODEL_LW, zorder=5, solid_capstyle="round")
            ax.axhline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
            furniture(ax, ylabel=f"SuCOS {stat}", xlabel=X_LABEL,
                      xlim=(xs[0] - 0.6, xs[-1] + 0.6))
            ax.set_ylim(0, 1)
            legend(ax, arm_handles(arms, include_ref=False, paint=soft), loc="upper left",
                   fontsize=11.5)
            fit(fig, pad=0.5)
            save(fig, out, f"sucos_per_atom_{stat}_{variant}")
        ranges[variant] = xs
    print("  per-atom x range: "
          + " · ".join(f"{v} {xs[0]}-{xs[-1]}" for v, xs in ranges.items()))
