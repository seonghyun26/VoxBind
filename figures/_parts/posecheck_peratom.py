

# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-atom, fig-posecheck-clash-per-atom, fig-posecheck-ecdf-pair
# ════════════════════════════════════════════════════════════════════════════════
# The size-resolved view of PoseCheck strain and steric clashes: what each arm's poses cost
# at a given ligand size, against the crystal ligand at that size.
#
# MEAN AND MEDIAN ARE SEPARATE FIGURES, which is what the 3-line figures do
# (vina_dock_3line_mean / _median) and here it is not optional. Strain's mean is not a
# location statistic: 6.6% of molecules fail UFF relaxation and land between 1e4 and 1e13,
# and one of those at a thin heavy-atom count moves that count's mean by four decades. The
# two sit three to four decades apart, so overlaid, the mean's spikes cross the whole panel
# and bury the medians -- which are tight, ordered, and the thing worth reading. Drawn
# apart, the contrast is itself the argument for reporting the median: in the mean figure
# the arms are tangled with no order at all, in the median figure they separate cleanly.
#
# The mean figure is still clipped to the bulk -- the 3-line figure's own answer (see its
# `_v3_limits`) -- and the excluded points are NAMED in the run log rather than squashing
# everything else into two decades. Clashes have no such problem and are drawn the same way
# for symmetry.
PC_STATS = (("mean", lambda v: float(np.mean(v))),
            ("median", lambda v: float(np.median(v))))
PC_STRAIN_CLIP = 1e6
PC_STRAIN_UNIT = "\n(kcal mol⁻¹)"
# The ECDF's x floors: strain is drawn on a log axis, so a value at or below zero has no
# place on it and is clipped onto the first decade rather than dropped; clashes are counts
# and floor at 0.
PC_ECDF_FLOOR = {"s": 1e-2, "c": 0}
PC_ECDF_W = 1.52                       # two panels side by side, off the one-panel width


def _pc_ranges():
    """(variant, arms, xs) per variant, plus the line saying what was drawn.

    THE X RANGE IS THE STRAIN RANGE, for the clash figure too. The builder computed it once
    from field "s" and handed the same xs to both, so the two panels register against each
    other count for count. Resolving it against "c" instead would silently shift the clash
    figure: a UFF relaxation that did not converge drops a molecule from the strain counts
    and not from the clash ones, so the two fields do not reach MIN_N at the same sizes."""
    data, p79_rows, _ = pose_data()
    out = []
    for variant, arms in variants("s", data):
        per = {key: by_size(p79_rows[key], "s") for _, key, _ in arms}
        out.append((variant, arms, x_range(per, arms)))
    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, _, xs in out)
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    return out


def _pc_per_atom_stat(out, field, xs, arms, variant, name, unit, stat, f, *, log,
                      clip=None, stem, legend_loc="upper left", rows=None, ref=None):
    """One statistic, one panel, one file. Which statistic you are looking at is carried by
    the filename and by the y-axis name, exactly as the 3-line figures carry it.

    `rows`/`ref` override where the molecules come from. Strain passes neither and reads
    pose_data() as it always has; CLASHES pass the whole-receptor trees, because the
    pose_data() rows are crop-scored and a crop under-counts clashes."""
    _, p79_rows, refrows = pose_data()
    p79_rows = p79_rows if rows is None else rows
    refrows = refrows if ref is None else ref
    per = {key: by_size(p79_rows[key], field) for _, key, _ in arms}
    ref_per = by_size(refrows, field)
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dropped = []
    ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per[key], xs, f)
        if clip:
            over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
            if over:
                dropped.append((lab, over))
        ax.plot(xs, y, color=color(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if clip:
        ax.set_ylim(top=clip)
    furniture(ax, ylabel=f"{name} {stat}{unit}", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    legend(ax, arm_handles(arms), loc=legend_loc, fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"{stem}_{variant}")
    return dropped


def _pc_block(rows):
    """The pooled numbers the run log reports for one set of molecules. These used to fill
    posecheck_summary.json too; that export belongs to the recompute script, so only what
    the table prints survives here."""
    s = [r["s"] for r in rows if r["s"] is not None]
    c = [r["c"] for r in rows if r["c"] is not None]
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "strain_mean": round(float(np.mean(s)), 1) if s else None,
        "strain_median": round(st.median(s), 1) if s else None,
        "clash_median": round(float(np.median(c)), 1) if c else None,
    }


def _pc_table():
    """Pooled strain and clashes per arm over the 79-pocket set, and the crystal ligand."""
    data, p79_rows, refrows = pose_data()
    print(f"  {'arm':16s} {'atoms':>6s} {'strain med':>11s} {'strain mean':>12s} "
          f"{'clash med':>10s} {'mols':>7s}")
    for lab, key, _ in arms_for("s", data):
        r = _pc_block(p79_rows[key])
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
              f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")
    r = _pc_block(refrows)
    print(f"  {REF_LABEL:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
          f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")


def _pc_curve_csv(out, xs):
    """The curves themselves, beside the figure, so a reader can check one without
    re-running it. Strain and clashes go in ONE table at ONE xs -- four curves over the
    same molecules at the same heavy-atom counts -- so the strain folder and the clash
    folder each get the whole thing rather than half of it. xs is the `all` variant's
    range, the wider of the two, and `core` is a subset of these rows."""
    data, p79_rows, refrows = pose_data()
    fields = ["strain_mean", "strain_median", "clash_mean", "clash_median"]
    cells = lambda curves, i: [("" if curves[f][i] is None else round(curves[f][i], 3))
                               for f in fields]
    rows = []
    for lab, key, _ in arms_for("s", data):
        s_per, c_per = by_size(p79_rows[key], "s"), by_size(p79_rows[key], "c")
        curves = {"strain_mean": model_curve(s_per, xs, PC_STATS[0][1]),
                  "strain_median": model_curve(s_per, xs, PC_STATS[1][1]),
                  "clash_mean": model_curve(c_per, xs, PC_STATS[0][1]),
                  "clash_median": model_curve(c_per, xs, PC_STATS[1][1])}
        rows += [[lab, a, len(s_per.get(a, ()))] + cells(curves, i)
                 for i, a in enumerate(xs)]
    # The reference's own curves come from the centred +-REF_WIN window, not from a count,
    # so it has no per-count n to report.
    s_ref, c_ref = by_size(refrows, "s"), by_size(refrows, "c")
    curves = {"strain_mean": reference_curve(s_ref, xs, PC_STATS[0][1]),
              "strain_median": reference_curve(s_ref, xs, PC_STATS[1][1]),
              "clash_mean": reference_curve(c_ref, xs, PC_STATS[0][1]),
              "clash_median": reference_curve(c_ref, xs, PC_STATS[1][1])}
    rows += [[REF_LABEL, a, ""] + cells(curves, i) for i, a in enumerate(xs)]
    write_csv(out, "posecheck_per_atom", ["arm", "heavy_atoms", "n"] + fields, rows)


@figure("fig-posecheck-strain-per-atom", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain)",))
def draw_posecheck_strain_per_atom(out):
    """PoseCheck strain against heavy-atom count — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_per_atom_all_methods."""
    use_style()
    drops, all_xs = {}, None
    for variant, arms, xs in core_only(_pc_ranges()):
        drops[variant] = _pc_per_atom_stat(
            out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT, *PC_STATS[0],
            log=True, clip=PC_STRAIN_CLIP, stem="strain_per_atom_mean",
            legend_loc="lower right")
        _pc_per_atom_stat(out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT,
                          *PC_STATS[1], log=True, stem="strain_per_atom_median",
                          legend_loc="upper left")
        all_xs = xs
    _pc_table()
    # Named, not hidden: the mean figure clips to the bulk, so say which points that leaves
    # off the panel and how far above they went.
    if drops["core"]:
        print(f"\n  strain_per_atom_mean_core: above the {PC_STRAIN_CLIP:.0e} clip, "
              f"off-panel (a failed UFF relaxation at a thin count moves that count's "
              f"mean):")
        for lab, over in drops["core"]:
            pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
            print(f"    {lab:16s} {len(over):2d} of {len(all_xs):2d}: {pts}")
    _pc_curve_csv(out, all_xs)


@figure("fig-posecheck-clash-per-atom", folder="fig-posecheck/clash",
        needs=("frozenenc_probes/posecheck_full/",))
def draw_posecheck_clash_per_atom(out):
    """PoseCheck steric clashes against heavy-atom count — mean and median, core only.

    WHOLE-RECEPTOR SCOPE (2026-09-14). This used to read pose_data(), i.e. the ARMS run
    trees, and those pose-scored against the 10 A crop: on the same arm over the same 79
    pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole receptor
    reads 6.0 / 7.48 / 97, and VoxBind 4.0 / 5.23 / 42 against 5.0 / 6.23 / 78. A crop
    deletes protein the pose could clash with, so it under-counts and truncates the tail
    hardest. Every other clash figure in this family scores against the whole receptor, so
    this one reads the same posecheck_full trees the violin and the box do.

    No `all`: see core_only(). The five published baselines carry NO per-molecule PoseCheck
    in the ARMS trees at all -- 0 of ~37,000 samples -- so `all` here was only ever core plus
    TargetDiff, which is neither the two-arm comparison nor the whole field. The whole field
    (seven published baselines, ours, and the crystal ligands) is
    clash_per_atom_all_methods, which reads the svr12 exports for the five."""
    use_style()
    keep = _pcsz_atom_keep()
    # The run roots are keyed by the ARMS key itself: posecheck_full/<key>.
    rows = {os.path.basename(root): _pcsz_atom_rows(root, keep)
            for _, root, *_ in PCSZ_LOCAL}
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    arms = [a for a in ARMS if a[1] in CORE and a[1] in rows]
    per = {key: by_size(rows[key], "c") for _, key, _ in arms}
    # The x range is the CLASH range now, not the strain range _pc_ranges() derives. That
    # sharing existed because both panels read one molecule set; this figure no longer does,
    # so registering the two count for count would be a coincidence, not a property.
    xs = x_range(per, arms)
    for stat, f in PC_STATS:
        _pc_per_atom_stat(out, "c", xs, arms, "core", "Clashes", "", stat, f,
                          log=False, stem=f"clash_per_atom_{stat}",
                          legend_loc="upper left", rows=rows, ref=refrows)
    # Its own CSV: _pc_curve_csv writes strain AND clashes out of pose_data(), whose clash
    # columns are the crop-scored ones this figure just stopped drawing.
    ref_per = by_size(refrows, "c")
    csv_rows = []
    for lab, key, _ in arms:
        cur = {s: model_curve(per[key], xs, fn) for s, fn in PC_STATS}
        csv_rows += [[lab, a, len(per[key].get(a, ()))]
                     + ["" if cur[s][i] is None else round(cur[s][i], 3)
                        for s, _ in PC_STATS] for i, a in enumerate(xs)]
    cur = {s: reference_curve(ref_per, xs, fn) for s, fn in PC_STATS}
    csv_rows += [[REF_LABEL, a, ""] + ["" if cur[s][i] is None else round(cur[s][i], 3)
                                       for s, _ in PC_STATS] for i, a in enumerate(xs)]
    write_csv(out, "clash_per_atom",
              ["arm", "heavy_atoms", "n", "clash_mean", "clash_median"], csv_rows)
    print(f"  {len(keep)} pockets · whole-receptor scope · "
          f"{', '.join(l for l, *_ in arms)} + reference · x = {xs[0]}-{xs[-1]}")


