

# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-ecdf-by-size / fig-posecheck-clash-violin-by-size
# ════════════════════════════════════════════════════════════════════════════════
# Every method on one axis -- the five published baselines PLUS the three we ran here
# (TargetDiff, vanilla VoxBind, Ours v1) -- stratified by generated-molecule heavy-atom
# count. Six panels each: the five size bins plus `all sizes`.
#
# WHY THIS FAMILY DOES NOT USE pose_data(). It reads a different scoring of the same
# molecules, and a different pocket set, for two reasons:
#
#  1. prj-denovo/baselines is not on this box, so the five baselines are read from the
#     per-molecule exports the legacy folder ships (`fig-posecheck/posecheck_<Method>.json`).
#     The loader is checked on every run: the per-bin numbers it reproduces from the
#     exports must match the stored `posecheck_baselines_by_atom_range.json` on all five
#     methods x five bins x three statistics, or the build aborts instead of drawing.
#
#  2. THE RUNS' MAIN SAMPLES DIRECTORIES ARE THE WRONG PROTOCOL. There PoseCheck was
#     scored against the *pocket10 crop*; the five baselines were scored against the whole
#     `*_rec.pdb`. Clashes are receptor-dependent and the crop hides some of them --
#     pooled clash mean 5.25 crop vs 6.27 full for vanilla, 6.49 vs 7.55 for Ours v1,
#     10.39 vs 11.11 for TargetDiff (see posecheck_crop_vs_full.json). Mixing the two
#     would have handed our three a ~1-clash head start. Our three are therefore read from
#     `frozenenc_probes/posecheck_full/`, the whole-receptor re-scoring, whose pooled
#     numbers reproduce `ours_posecheck_full.json` exactly.
#
# POCKET SUBSET. The baselines cover all 79 electron-density pockets; the whole-receptor
# re-scoring of our three now covers all 79 as well: target_71 was missing only because
# scripts 70/71/72 default to `p78_targets.json`, a list that exists for a *docking*
# failure (pdb2pqr30 rejects that pocket's 10 A crop) which PoseCheck never had. Every
# series is still restricted to the pockets both sides share, computed at run time, so no
# method is scored on a pocket another one never saw; the console prints the shared count
# and anything dropped.
#
# THE REFERENCE LIGAND is drawn on the strain figure only. PoseCheck's strain
# (`calculate_strain_energy(mol, num_confs=50)`) never touches the receptor, so the
# crystal ligand's strain is protocol-independent and can be read from our local run. Its
# *clashes* are not: the only local reference scoring is against the crop, so it would sit
# ~0.3-1 clash low next to eight whole-receptor series, and it is left off the violin
# rather than quietly compared. (The whole-receptor reference clash means from the
# baselines' own run live on in the legacy posecheck_all_by_atom_range.json.)
#
# STRAIN IS NOISY BY CONSTRUCTION. `num_confs=50` random conformers means re-scoring the
# same pose gives a slightly different answer: over 7,349 molecules scored twice here the
# median relative difference is 1.0%, p90 8.9%. Fine for an ECDF, not for ranking two
# methods a few percent apart.
#
# COLOURS come from the shared palette, so the key carries across figures; only line
# weight and dash are decided here. Our two are drawn thick and solid; the five published
# baselines are context, so they are thin and each carries its own dash pattern -- with
# nine series on one axis hue alone cannot be the identity channel.
#
# STYLE. This family predates the house style at the top of this file and keeps its own:
# 12 pt type, BLACK spines and tick labels. The violin keeps its panel title -- the size bin
# has nowhere else to live inside the frame -- but the strain ECDF dropped its title on
# request (2026-09-13), so there the bin is named by the file only.

# key -> (label, linewidth, linestyle). Dash is a second identity channel for the five
# published baselines: they are drawn thin, five at a time, and colour alone is not enough
# at that weight.
PCSZ_BASELINES = [
    ("AR",         "AR",         1.6, "-"),
    ("Pocket2Mol", "Pocket2Mol", 1.6, (0, (5, 2))),
    ("DiffSBDD",   "DiffSBDD",   1.6, (0, (1, 1.6))),
    ("DecompDiff", "DecompDiff", 1.6, (0, (6, 2, 1, 2))),
    ("FuncBind",   "FuncBind",   1.6, (0, (9, 3))),
]
# TargetDiff is violet and still dashed: it used to be orange, which put two BASELINES in
# the same hue family as each other, and the dash costs nothing now that the collision is
# gone. These labels are the drawn labels, not the ARMS keys -- they reach color() through
# the alias table.
PCSZ_LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", 2.4, (0, (6, 2))),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    3.0, "-"),
    ("CoDE",          f"{E}/frozenenc_probes/posecheck_full/ours_v1",    3.4, "-"),
]
# Weight per METHOD, keyed by the canonical name so the other family can read it: the two
# figures are a pair and a series that is thicker in one of them reads as a different series
# (2026-09-13). NOT one flat weight -- our two arms are the subject and the five published
# baselines are context, which is the same reason they are dashed and these are solid.
PCSZ_REF_LW = 2.0
PCSZ_LINE_LW = {**{ALIASES.get(lab, lab): lw for _, lab, lw, _ in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): lw for lab, _, lw, _ in PCSZ_LOCAL}}

# The five shared size bins plus the pooled one every molecule ALSO lands in.
PCSZ_LABELS = BIN_LABELS + ["all sizes"]
PCSZ_SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30", "all"]
PCSZ_POOLED = len(PCSZ_LABELS) - 1

# This family's own furniture: a white ground and BLACK spines/ticks (the house INK/AXIS
# are the warm near-black of the newer figures), with the shared GRID and LEGEND_EDGE.
PCSZ_BG, PCSZ_INK, PCSZ_AXIS = "#ffffff", "#000000", "#000000"
# The axis starts at 10 kcal/mol. Below that a pose is effectively unstrained, and spending
# decades of width on it pushed the region where the methods separate into the right half.
# Values under the floor are clipped onto it, so the curve enters the axis at its true share.
PCSZ_XFLOOR, PCSZ_XTOP = 1e1, 3e3
# 80 % of the original 8.6 x 5.6 in canvas both ways, then 1.2x wider again so the lower-right
# key leaves room for the curves' long right tails, then 0.9x both ways (2026-09-13). The
# per-atom all-methods figure reads its aspect from this, so keep scaling both sides together
# unless that one should change too.
# PCSZ_SHRINK pulls the canvas in once more (2026-09-13) WITHOUT touching the type sizes or
# the dpi, so the same ink fills a smaller frame -- that, not a font change, is what makes the
# panel read fuller. PCSZ_LW_SCALE thickens the curves by the same argument: at this canvas
# the nine series were drawn for a frame a fifth wider. The trailing 1.1x is WIDTH ONLY
# (2026-09-13): the lower-right key and the curves' right tails were tight against each other
# once the canvas came in.
PCSZ_SHRINK, PCSZ_LW_SCALE = 0.85, 1.3
PCSZ_ECDF_SIZE = (8.6 * 0.8 * 1.2 * 0.9 * PCSZ_SHRINK * 1.1, 5.6 * 0.8 * 0.9 * PCSZ_SHRINK)
# Axis-name size for the ECDF, and for the per-atom all-methods figure drawn in its style.
# 1.4x the family's original 11.5 pt (2026-09-13).
PCSZ_LABEL_FS = 11.5 * 1.4
PCSZ_RC = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": PCSZ_INK, "axes.labelcolor": PCSZ_INK,
    "xtick.color": PCSZ_AXIS, "ytick.color": PCSZ_AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
    # save() takes no dpi/bbox arguments, and this family is drawn at 170 dpi and cropped
    # to its own ink rather than sized to the figure. Both ride in on the savefig.* rc
    # keys, which Figure.savefig and print_figure fall back to when the kwargs are absent.
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
}


def _pcsz_empty():
    return {b: {"strain": [], "clash": []} for b in range(len(PCSZ_LABELS))}


def _pcsz_add(out, n, s, c):
    for b in (bin_of(n), PCSZ_POOLED):
        if s is not None and np.isfinite(s):
            out[b]["strain"].append(float(s))
        if c is not None and np.isfinite(c):
            out[b]["clash"].append(float(c))


def _pcsz_load_baseline(method, keep):
    """One of the five, from its per-molecule export. `keep` is a set of pocket indices."""
    d = json.load(open(legacy("fig-posecheck", f"posecheck_{method}.json"),
                       encoding="utf-8"))
    out = _pcsz_empty()
    for m in d["molecules"]:
        if m["p"] in keep:
            _pcsz_add(out, m["n"], m["s"], m["c"])
    return out, set(d["density79_pockets"])


def _pcsz_metrics(root):
    """Every per-target metrics.json under one run root, in path order."""
    return sorted(str(p) for p in Path(root).glob("target_*/metrics.json"))


def _pcsz_load_local(root, keep, reference=False):
    """One of ours, from per-target metrics.json."""
    out = _pcsz_empty()
    seen = set()
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        seen.add(idx)
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            pc = m.get("posecheck") or {}
            _pcsz_add(out, m["n_atoms"], pc.get("strain"), pc.get("clashes"))
    return out, seen


def _pcsz_pockets_of(root):
    return {int(os.path.basename(os.path.dirname(p)).split("_")[1])
            for p in _pcsz_metrics(root)}


def _pcsz_stats(cell):
    s_, c_ = cell["strain"], cell["clash"]
    return {
        "n_strain": len(s_), "n_clash": len(c_),
        "strain_median": round(st.median(s_), 1) if s_ else None,
        "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
        "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
        "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
        "clash_median": round(float(np.median(c_)), 1) if c_ else None,
    }


_PCSZ_CACHE = {}


def _pcsz_data():
    """(per-method {bin: {strain, clash}}, the drawn series, the reference, the summary).

    Cached for the life of the process: the two figures below are the same nine series
    read once and drawn twice, and the verification pass alone re-bins every baseline
    molecule five times."""
    if "d" in _PCSZ_CACHE:
        return _PCSZ_CACHE["d"]

    # ── the pocket subsets ────────────────────────────────────────────────────
    probe = json.load(open(legacy("fig-posecheck", "posecheck_AR.json"), encoding="utf-8"))
    density79 = set(probe["density79_pockets"])
    shared = set(density79)
    for _, root, *_ in PCSZ_LOCAL:
        shared &= _pcsz_pockets_of(root)
    dropped = sorted(density79 - shared)
    print(f"  pockets: baselines {len(density79)} · shared with our three {len(shared)}"
          f" · dropped {['target_%02d' % i for i in dropped]}")

    # ── verification: the exports must reproduce the stored per-bin numbers ───
    stored = json.load(open(legacy("fig-posecheck",
                                   "posecheck_baselines_by_atom_range.json"),
                            encoding="utf-8"))
    mismatch = 0
    for key, label, *_ in PCSZ_BASELINES:
        data, _ = _pcsz_load_baseline(key, density79)
        for b in range(len(EDGES) - 1):
            want, got = stored["methods"][label][b], _pcsz_stats(data[b])
            for a, c in (("n", "n_strain"), ("strain_median", "strain_median"),
                         ("clash_mean", "clash_mean")):
                if want.get(a) != got.get(c):
                    mismatch += 1
                    print(f"    MISMATCH {label} bin {PCSZ_LABELS[b]} {a}: "
                          f"stored {want.get(a)} != rebuilt {got.get(c)}")
    if mismatch:
        raise SystemExit(f"loader disagrees with the stored baseline JSON on {mismatch} "
                         f"values -- refusing to plot")
    print("  verification: exports reproduce posecheck_baselines_by_atom_range.json "
          "on 5 methods x 5 bins x 3 statistics, 0 mismatches")

    # ── the data actually plotted, all on the shared pockets ──────────────────
    data_by_label, series = {}, []
    for key, label, lw, ls in PCSZ_BASELINES:
        d79, _ = _pcsz_load_baseline(key, density79)
        dsh, _ = _pcsz_load_baseline(key, shared)
        data_by_label[label] = dsh
        series.append((label, color(label), lw, ls))
        a, b = _pcsz_stats(d79[PCSZ_POOLED]), _pcsz_stats(dsh[PCSZ_POOLED])
        print(f"    {label:12s} 79 pockets n={a['n_clash']:5d} clash {a['clash_mean']:5.2f} "
              f"strain {a['strain_median']:7.1f}   ->  {len(shared)} pockets "
              f"n={b['n_clash']:5d} clash {b['clash_mean']:5.2f} strain {b['strain_median']:7.1f}")
    for label, root, lw, ls in PCSZ_LOCAL:
        data_by_label[label], _ = _pcsz_load_local(root, shared)
        series.append((label, color(label), lw, ls))
    ref, _ = _pcsz_load_local(REF_ROOT, shared, reference=True)

    # ── the numbers behind both figures ──────────────────────────────────────
    methods = {label: [_pcsz_stats(data_by_label[label][b])
                       for b in range(len(PCSZ_LABELS))] for label, *_ in series}
    methods[REF_LABEL] = [
        {**_pcsz_stats(ref[b]), "clash_mean": None, "clash_median": None,
         "note": "strain only"} for b in range(len(PCSZ_LABELS))]

    print(f"\n  {'bin':10s} {'method':14s} {'n':>6s} {'strain med':>11s} {'IQR':>19s} "
          f"{'clash mean':>11s} {'med':>5s}")
    for b, lab in enumerate(PCSZ_LABELS):
        for label in methods:
            r = methods[label][b]
            if not r["n_strain"]:
                continue
            iqr = (f"{r['strain_q25']:7.1f}–{r['strain_q75']:<10.1f}"
                   if r["strain_q25"] is not None else " " * 18)
            cm = f"{r['clash_mean']:11.2f}" if r["clash_mean"] is not None else f"{'—':>11s}"
            cd = f"{r['clash_median']:5.1f}" if r["clash_median"] is not None else f"{'—':>5s}"
            print(f"  {lab:10s} {label:14s} {r['n_strain']:6d} {r['strain_median']:11.1f} "
                  f"{iqr} {cm} {cd}")
        print()

    _PCSZ_CACHE["d"] = (data_by_label, series, ref, methods)
    return _PCSZ_CACHE["d"]


def _pcsz_style(ax, *, grid_axis="both"):
    """The vina figures' axis furniture: black spines and outward ticks, dotted grid.

    `grid_axis` is "y" for the violin -- its x is categorical, so a vertical rule through
    every category centre is a picket fence, not a reading aid."""
    ax.set_facecolor(PCSZ_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(labelsize=11, direction="out", length=3.5, width=1.1, pad=4,
                   colors=PCSZ_AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.9, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)


PCSZ_KEY_NCOL = 2
PCSZ_KEY_KW = dict(fontsize=9.5, handlelength=1.6, handletextpad=0.5, labelspacing=0.32,
                   borderpad=0.4, columnspacing=1.2)
# The frame's face, translucent so a curve running under the key is still followable, still
# opaque enough to keep the dotted grid out of the text (2026-09-13). Set on the FACE, not as
# the artist's alpha, which would fade the grey rule with it.
PCSZ_KEY_FACE = (1.0, 1.0, 1.0, 0.82)
# The reference line is anchored on the grid's top edge, so the gap between them is the two
# legends' borderpad back to back. Pull it down by this much of the axes to close half of it.
PCSZ_KEY_GAP = 0.022


def _pcsz_key(fig, ax):
    """The ECDF key: the crystal ligands on a centred line of their own, over the eight methods
    in 4 rows x 2 columns reading across, in one frame in the lower right.

    Built like the Vina per-atom key (_vpa_legend): TWO legends inside ONE rectangle. A legend
    column is as wide as its widest entry and matplotlib cannot span a cell, so the reference
    cannot sit centred over the grid as a cell of it. The grid is placed first, the reference
    legend is centred on it and stacked on top, both frames come off, and a single rectangle is
    drawn round their union. Call AFTER the layout is final: positions are axes fractions of
    the axes as laid out when this runs."""
    pairs = list(zip(*ax.get_legend_handles_labels()))
    ref = [p for p in pairs if p[1] == REF_LABEL]
    methods = [p for p in pairs if p[1] != REF_LABEL]
    rows = -(-len(methods) // PCSZ_KEY_NCOL)
    # matplotlib fills a legend column by column; interleave so the grid reads across.
    methods = [methods[r * PCSZ_KEY_NCOL + c] for c in range(PCSZ_KEY_NCOL)
               for r in range(rows) if r * PCSZ_KEY_NCOL + c < len(methods)]
    grid = ax.legend(*zip(*methods), loc="lower right", ncol=PCSZ_KEY_NCOL, borderaxespad=0.5,
                     frameon=False, **PCSZ_KEY_KW)
    legs = [grid]
    render = fig.canvas.get_renderer
    if ref:
        ax.add_artist(grid)          # the next ax.legend() would otherwise replace it
        fig.canvas.draw()
        bb = grid.get_window_extent(render()).transformed(ax.transAxes.inverted())
        legs.append(ax.legend(*zip(*ref), loc="lower center", borderaxespad=0, frameon=False,
                              bbox_to_anchor=((bb.x0 + bb.x1) / 2, bb.y1 - PCSZ_KEY_GAP),
                              **PCSZ_KEY_KW))
    fig.canvas.draw()
    boxes = [l.get_window_extent(render()).transformed(ax.transAxes.inverted()) for l in legs]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    # An AXES artist, not a figure one: a figure-level patch is drawn after the whole axes and
    # would cover the legend text. Translucent white under the text, over the curves.
    ax.add_artist(matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes, facecolor=PCSZ_KEY_FACE,
        edgecolor=LEGEND_EDGE, linewidth=0.7, zorder=5, clip_on=False))
    for l in legs:
        l.set_zorder(6)
        for text in l.get_texts():   # identity rides the swatch, not the ink
            text.set_color(PCSZ_INK)


def _pcsz_csv(out, methods):
    write_csv(out, "posecheck_all_by_atom_range",
              ["bin", "method", "n_strain", "strain_median", "strain_q25", "strain_q75",
               "n_clash", "clash_mean", "clash_median"],
              [[lab, label, r["n_strain"], r["strain_median"], r["strain_q25"],
                r["strain_q75"], r["n_clash"], r["clash_mean"], r["clash_median"]]
               for b, lab in enumerate(PCSZ_LABELS)
               for label, r in ((k, v[b]) for k, v in methods.items())])


@figure("fig-posecheck-strain-ecdf-by-size", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_strain_ecdf_by_size(out):
    """UFF strain ECDF, nine methods on one log axis, per heavy-atom bin."""
    data_by_label, series, ref, methods = _pcsz_data()
    with plt.rc_context(PCSZ_RC):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            for label, colour, lw, ls in series:
                v = np.asarray(data_by_label[label][b]["strain"], dtype=float)
                if v.size < 20:
                    continue
                x = np.sort(np.clip(v, PCSZ_XFLOOR, None))
                # soft(): CoDE in its lighter tint, as in the eight-method strain line figures
                # (2026-09-13). The clash violin below still draws the palette colour.
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=soft(label),
                        lw=lw * PCSZ_LW_SCALE, ls=ls,
                        label=label, solid_capstyle="round")
            r = np.asarray(ref[b]["strain"], dtype=float)
            if r.size >= 3:
                x = np.sort(np.clip(r, PCSZ_XFLOOR, None))
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR,
                        lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=(0, (3, 2)),
                        label=REF_LABEL)
            ax.set_xscale("log")
            ax.set_xlim(PCSZ_XFLOOR, PCSZ_XTOP)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("UFF strain energy (kcal mol⁻¹) · log scale", fontsize=PCSZ_LABEL_FS)
            ax.set_ylabel("Cumulative probability", fontsize=PCSZ_LABEL_FS)
            # No title (removed on request 2026-09-13): the size bin is carried by the file
            # name (`ecdf-by-size-<bin>`) and by whatever caption places the figure.
            # Nine series; the legend names them only -- the medians and n are in the CSV and
            # the run log, not the key. The crystal ligands head it on a centred line of their
            # own, over the eight methods in 4 x 2 (see _pcsz_key); lower right, opaque white
            # with a grey rule so the dotted grid does not run through the text. The key is
            # placed after tight_layout, because its frame is measured in axes fractions.
            fig.tight_layout(pad=0.5)
            _pcsz_key(fig, ax)
            save(fig, out, f"strain_ecdf_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


@figure("fig-posecheck-clash-violin-by-size", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_violin_by_size(out):
    """Steric clashes per pose, eight methods as violins, per heavy-atom bin."""
    data_by_label, series, _ref, methods = _pcsz_data()
    with plt.rc_context(PCSZ_RC):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            fig, ax = plt.subplots(figsize=(10.6, 5.6))
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax, grid_axis="y")
            vals, cols, ticks = [], [], []
            for label, colour, lw, ls in series:
                v = data_by_label[label][b]["clash"]
                if len(v) < 20:
                    continue
                vals.append(v)
                cols.append(colour)
                ticks.append(label.replace(" · ", "\n").replace(" σ", "\nσ"))
            # KDE fitted on log10(x+1) and the ticks relabelled with real counts: a plain
            # log axis is impossible because a few percent of poses have zero clashes, and
            # symlog would warp only the display while the KDE stayed in data space.
            tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
            parts = ax.violinplot([tf(v) for v in vals], showextrema=False, widths=0.82)
            for body, colour in zip(parts["bodies"], cols):
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)
                body.set_linewidth(1.2)
            for i, v in enumerate(vals, start=1):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
                ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
            hi = max(max(v) for v in vals)
            ticks_at = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= hi * 1.6]
            ax.set_yticks(tf(ticks_at))
            ax.set_yticklabels([str(t) for t in ticks_at])
            ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.20)
            for i, v in enumerate(vals, start=1):
                ax.text(i, ax.get_ylim()[1],
                        f"mean {np.mean(v):.2f}\nmed {np.median(v):.0f}",
                        ha="center", va="top", fontsize=8.5, color="#3a4352",
                        linespacing=1.35)
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(ticks, fontsize=9)
            ax.set_ylabel("steric clashes per pose  ·  log-spaced", fontsize=11.5)
            ax.set_title(f"Steric clashes — {lab} heavy atoms", fontsize=14,
                         fontweight="620", loc="left", pad=10)
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_violin_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")
