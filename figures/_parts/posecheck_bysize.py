

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
# The dashes come from METHOD_DASH in 00_core.py now (2026-09-14), so this family and the
# rotatable-bond one cannot drift apart again. AR and Pocket2Mol both carried (5, 2) here,
# which left them separable by hue alone; dash() gives Pocket2Mol its own pattern.
PCSZ_BASELINES = [
    ("AR",         "AR",         1.6, dash("AR")),
    ("Pocket2Mol", "Pocket2Mol", 1.6, dash("Pocket2Mol")),
    ("DiffSBDD",   "DiffSBDD",   1.6, dash("DiffSBDD")),
    ("DecompDiff", "DecompDiff", 1.6, dash("DecompDiff")),
    ("FuncBind",   "FuncBind",   1.6, dash("FuncBind")),
]
# TargetDiff is violet and still dashed: it used to be orange, which put two BASELINES in
# the same hue family as each other, and the dash costs nothing now that the collision is
# gone. These labels are the drawn labels, not the ARMS keys -- they reach color() through
# the alias table.
PCSZ_LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", 1.6,
     dash("TargetDiff")),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    3.4,
     dash("VoxBind σ=0.9")),
    ("CoDE",          f"{E}/frozenenc_probes/posecheck_full/ours_v1",    3.8, dash("CoDE")),
]
# Weight per METHOD, keyed by the canonical name so the other family can read it: the two
# figures are a pair and a series that is thicker in one of them reads as a different series
# (2026-09-13). NOT one flat weight -- our two arms are the subject and the five published
# baselines are context, which is the same reason they are dashed and these are solid.
# ONE WEIGHT FOR EVERY SERIES THAT IS NOT OURS (2026-09-14): the five published baselines,
# TargetDiff and the crystal ligands all draw at 1.6, and dash alone tells them apart. Only
# VoxBind (3.4) and CoDE (3.8) are heavier, which is the whole point of the weight channel.
PCSZ_REF_LW = 1.6
# What the crystal ligands are CALLED in this family's key (2026-09-14). The shared REF_LABEL
# ("Reference ligand") still names them everywhere else; only the drawn key is shortened, and
# _pcsz_key splits the key on this name.
PCSZ_REF_NAME = "Reference"
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
# The clash axis is log10(x+1), so it can hold the zero-clash poses; these are the labels put
# back on it in real counts (2026-09-14).
PCSZ_CLASH_TICKS = [0, 1, 10, 100]
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
PCSZ_ECDF_SIZE = (8.6 * 0.8 * 1.2 * 0.9 * PCSZ_SHRINK * 1.1,
                  5.6 * 0.8 * 0.9 * PCSZ_SHRINK * 0.9)   # trailing 0.9 is HEIGHT only
# Axis-name size for the ECDF, and for the per-atom all-methods figure drawn in its style.
# 1.4x the family's original 11.5 pt (2026-09-13).
PCSZ_LABEL_FS = 11.5 * 1.4
# The y name's gap off its tick labels, DOUBLED (2026-09-14). Measured off the drawn PNGs the
# matplotlib default (labelpad 4.0) buys 5.5 pt here and 6.8 pt on the per-atom panel; a pad of
# 10 is twice that on both. Shared, so the pair keeps one left margin.
PCSZ_YPAD = 10
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
# handlelength 1.6 showed barely one repeat of a dash pattern, so the five baselines' second
# identity channel was unreadable in the key; 2.8 fits two to three repeats of every pattern.
PCSZ_KEY_KW = dict(fontsize=9.5, handlelength=2.8, handletextpad=0.5, labelspacing=0.32,
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
    ref = [p for p in pairs if p[1] == PCSZ_REF_NAME]
    methods = [p for p in pairs if p[1] != PCSZ_REF_NAME]
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
                # display(): the key carries VoxBind's sigma as a subscript (2026-09-14). It is
                # the ONLY name DISPLAY overrides, so every other entry is unchanged, and the
                # series keeps its plain label everywhere else -- CSV, violin ticks, data keys.
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=soft(label),
                        lw=lw * PCSZ_LW_SCALE, ls=ls,
                        label=display(label), solid_capstyle="round")
            r = np.asarray(ref[b]["strain"], dtype=float)
            if r.size >= 3:
                x = np.sort(np.clip(r, PCSZ_XFLOOR, None))
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR,
                        # SOLID, and AR took its dash (2026-09-14): solid was the one style no
                        # baseline had, so it now marks the crystal ligands and our two arms --
                        # the series a reader returns to -- and every baseline carries a dash.
                        lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls="-",
                        label=PCSZ_REF_NAME, solid_capstyle="round")
            ax.set_xscale("log")
            ax.set_xlim(PCSZ_XFLOOR, PCSZ_XTOP)
            ax.set_ylim(0, 1.0)
            # "· log scale" dropped (2026-09-14); the decade ticks say it.
            ax.set_xlabel("UFF strain energy (kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS)
            # Two lines (2026-09-14), as on the per-atom panel it pairs with.
            ax.set_ylabel("Cumulative\nprobability", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
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
            # The strain ECDF's canvas (2026-09-14), so the two PoseCheck measurements are the
            # same shape on a page; its rc is already in force for the whole family.
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax, grid_axis="y")
            vals, cols, ticks = [], [], []
            for label, colour, lw, ls in series:
                v = data_by_label[label][b]["clash"]
                if len(v) < 20:
                    continue
                vals.append(v)
                # soft(), as every other eight-method figure draws (2026-09-14). `colour` off
                # the series table is the saturated palette entry, which now only the
                # three-arm and docking figures use.
                cols.append(soft(label))
                ticks.append(display(label))
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
            # DECADES ONLY (2026-09-14): zero, then 1, 10, 100. The nine intermediate ticks the
            # axis used to carry were reading aids for numbers that are now in the CSV.
            ticks_at = [t for t in PCSZ_CLASH_TICKS if t <= hi * 1.6]
            ax.set_yticks(tf(ticks_at))
            ax.set_yticklabels([str(t) for t in ticks_at])
            # The headroom the per-violin mean/median strip needed is gone with it.
            ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.06)
            # TURNED (2026-09-14): on the ECDF's narrower canvas eight names laid flat ran
            # into each other -- Pocket2Mol into DiffSBDD into DecompDiff. Anchored at the
            # right so each name ends under its own violin.
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(ticks, fontsize=9.5, rotation=30, ha="right")
            ax.set_ylabel("Steric clashes\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            # NO TITLE and NO per-violin mean/med strip (2026-09-14), as on the strain ECDF:
            # the bin is in the file name (`clash-violin-by-size-<bin>`) and every number the
            # strip carried is a column of the CSV written beside the figure.
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_violin_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


# ── fig-posecheck-clash-box-by-method ──────────────────────────────────────────────────
# The rotatable-bond grid's box format on ONE panel: method along x, clashes up y. There is
# nothing to facet here -- this is every pose the method made, at every ligand size -- so the
# 3x3 block collapses to a single axes and the bins live in the violin above.
# THE FILL IS THE METHOD, not its median (2026-09-14). The grid needed a colourmap because a
# method owns nine panels there and the fill was the only place its median could be read; on
# one axes the eight medians are side by side already, so the fill goes back to carrying
# identity -- the same soft() tint the violin and the ECDF give that method -- and the
# colourbar that encoded the median comes off.
# 0.8x the grid's 2.0 (2026-09-14): a box here is far wider than one in a nine-panel grid,
# and the same weight read heavier across it.
PCSZ_CBOX_MED_LW = 2.0 * 0.8
# Wider again (2026-09-14), on top of the width the colourbar gave back. TALL is its own
# factor because the x names went up to the y name's size: nine of them at 30 degrees eat
# height that the boxes used to have.
PCSZ_CBOX_WIDE, PCSZ_CBOX_TALL = 1.15, 1.2
# ROTATED (2026-09-14): the methods run DOWN the y axis and the clashes across x. Nine rows
# want height rather than width, and the names stop needing their 30-degree tilt -- upright
# against a y axis is the main thing rotating buys. Kept as its own pair rather than swapping
# the two above, so going back to the upright panel stays one edit.
# Width at 0.6x (2026-09-14), back up from 0.4. The method names sit OUTSIDE the axes but
# INSIDE the figure, and constrained layout fits everything within figsize -- so a cut here
# comes out of the drawing area alone, the names keeping their full width. At 0.4 that left
# the nine boxes about 285 px of a 492 px image to share, with the labels taking 42 % of it,
# and the median spread the panel exists to show (4 against 8) stopped being legible.
PCSZ_CBOX_WIDE_V, PCSZ_CBOX_TALL_V = 0.5, 1.55
# The method names' tilt. A rotated label's horizontal footprint is w*cos(t) + h*sin(t), so
# for names this long against this type size the saving is modest until the angle is steep:
# roughly 5 % at 30 degrees, 17 % at 45, 36 % at 60. Level is easier to read, so this is the
# smallest angle that buys anything rather than the one that buys most.
PCSZ_CBOX_NAME_ROT = 30
# Minor ticks, as a fraction of the frame's weight. They SUBDIVIDE a decade rather than bound
# the panel, and carrying the frame's full 2.0 made the eight of them between each pair of
# decades read as more axis lines instead of as gradations on one.
PCSZ_CBOX_MINOR_LW = 0.5
# WHERE THE VALUE AXIS STOPS (2026-09-14). Left to itself it runs to the largest flier any
# method produced -- DiffSBDD's 333 -- which spends the right third of the panel on a handful
# of points and squeezes all nine boxes into the left half. DiffSBDD is the ONLY method that
# reaches past this: the next largest maximum is TargetDiff's 167, and every other method
# tops out between 26 and 97, so the cap costs one method's extreme tail and nothing else.
PCSZ_CBOX_HI = 200.0
# HOW MUCH WHITE GOES INTO A FILL (2026-09-14). The fills already carry EXACTLY the soft()
# hexes the ECDF and the per-atom panel draw their lines in -- checked on the rendered
# pixels, #EE9190 / #F0A6C0 / #8DCB92 / #C6B46A / #8291E8 / #7FC4D1 match byte for byte
# across the two files. They still read more saturated, and that is an area effect, not a
# palette drift: a filled box is ~100x the ink of a 3 px line of the same colour. So the
# FILL, and only the fill, is mixed toward white. The line palette is untouched, which is
# what keeps a method the same colour everywhere. This is the step the rotatable-bond grid
# gets for free by running its colourmap through pale().
# NO WHITE AT ALL (2026-09-14): the fills are the soft() hexes exactly as the ECDF and the
# per-atom panel draw their lines, so a method is literally one colour across the three
# figures. 0.42 read washed out and 0.2 was still a second colour for the same method; the
# area effect that motivated the blend is worth living with. This stays a knob rather than
# being deleted -- one number brings the tint back if the solid blocks prove too heavy.
PCSZ_CBOX_TINT = 0.0
# The frame's weight, up from this family's 1.1 to the rotatable-bond grid's (2026-09-14):
# the panel carries the grid's boxes and now type at 14 pt, and 1.1 read thin under both.
# The rule between the crystal ligands and the methods takes the SAME weight, because it is
# frame and not data -- 79 poses against ~7,500 is a different population, not a ninth method.
PCSZ_CBOX_AXIS_LW = 2.0
# The method names, two points off the y name (2026-09-14).
PCSZ_CBOX_NAME_FS = PCSZ_LABEL_FS - 2
# AIR ON EITHER SIDE OF THE RULE (2026-09-14). Boxes are 0.66 wide on a 1.0 pitch, so two
# neighbours are 0.34 apart; at a 2 pt frame weight the rule ate most of that, leaving less
# air between the reference box and AR's than between the reference box and the left spine.
# The methods slide right by this much so the rule sits in a margin of its own.
PCSZ_CBOX_GAP = 0.5
# The x padding past the outermost box centre, at each end.
PCSZ_CBOX_PAD = 0.7
# ONE SCALE OVER BOTH OF THE ABOVE (2026-09-14), so "tighten the whitespace" stays a single
# number instead of two that drift apart. 0.8 puts the rule's air at 0.40 and the end padding
# at 0.56. The box width and the 1.0 pitch are untouched: only air moves, so the boxes keep
# their size and the panel just stops carrying as much empty space.
PCSZ_CBOX_MARGIN = 0.8


def _pcsz_tint(hexv, f=PCSZ_CBOX_TINT):
    """`hexv` mixed f of the way to white, as an rgb triple."""
    r, g, b = (int(hexv[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return tuple(c + (1.0 - c) * f for c in (r, g, b))
# WHERE THE CRYSTAL LIGANDS' CLASHES COME FROM, and why not from REF_ROOT like their strain.
# REF_ROOT's run pose-scored against the 10 A crop, which deletes protein a pose could clash
# with and so under-counts; every method on this axis is scored against the whole receptor.
# This root pose-scored with scope="full", so its reference is on the same footing. Nothing
# was recomputed for it: three independent full-scope runs (this one, reproduction/
# res_test_100 and 260827 base) carry the reference clash for all 79 pockets and agree on
# every one of them, as they must -- counting clashes is deterministic, unlike strain.
PCSZ_REF_CLASH_ROOT = (f"{E}/260908_fusion_default_cv2_scratch_8gpu/samples/"
                       "samples_ep350_test79_n100")


@figure("fig-posecheck-clash-box-by-method", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_box_by_method(out):
    """Steric clashes per pose, every method, all ligand sizes pooled."""
    data_by_label, series, _ref, methods = _pcsz_data()
    labels = [label for label, *_ in series
              if len(data_by_label[label][PCSZ_POOLED]["clash"]) >= 20]
    vals = [np.asarray(data_by_label[l][PCSZ_POOLED]["clash"], dtype=float) for l in labels]
    # THE CRYSTAL LIGANDS LEAD (2026-09-14), from the whole-receptor run named above. They are
    # the benchmark every method is read against, so they are the leftmost box, and n=79 here
    # against ~7,500 a method: one crystal pose per pocket is all there is.
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    ref_full, _ = _pcsz_load_local(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    labels = [PCSZ_REF_NAME] + labels
    vals = [np.asarray(ref_full[PCSZ_POOLED]["clash"], dtype=float)] + vals
    meds = [float(np.median(v)) for v in vals]
    # PROPER log10, not log10(x+1) (2026-09-14). The +1 existed so a zero-clash pose had
    # somewhere to sit on the axis -- but the floor is 10^0 now and those poses fall below it
    # either way, so all the +1 still bought was a compressed bottom decade: it doubles 1
    # while leaving 100 untouched. Measured on the drawn panel, 10^0->10^1 spanned 130 px
    # against 169 px for 10^1->10^2, a ratio of 1.30 under tick labels that promise 1.00.
    # Zeros clamp to 0.5, below the floor, and are clipped exactly as before -- so nothing
    # visible changes except that the decades are finally even.
    # THE VIOLIN KEEPS ITS +1 and must: its axis carries a real 0 tick, so a zero-clash pose
    # has a place to land there and the transform is still doing work.
    tf = lambda v: np.log10(np.maximum(np.asarray(v, dtype=float), 0.5))
    with plt.rc_context(PCSZ_RC):
        fig, ax = plt.subplots(figsize=(PCSZ_ECDF_SIZE[0] * PCSZ_CBOX_WIDE_V,
                                        PCSZ_ECDF_SIZE[1] * PCSZ_CBOX_TALL_V),
                               layout="constrained")
        fig.patch.set_facecolor(PCSZ_BG)
        # The grid follows the VALUE axis, which rotating moved to x. A rule through every
        # category centre is a picket fence rather than a reading aid -- which is the whole
        # reason _pcsz_style takes the axis as an argument.
        _pcsz_style(ax, grid_axis="x")
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(PCSZ_CBOX_AXIS_LW)
        ax.tick_params(which="major", width=PCSZ_CBOX_AXIS_LW)
        # THE GRID'S WHISKERS (2026-09-14): 1.5 x IQR on the RAW counts, the way the
        # rotatable-bond grid gets them by handing raw values to a log axes. Handing this one
        # tf(v) instead put the fences 1.5 IQR out in LOG space, which ran AR's whisker to 33
        # where the raw fence is 14 -- the two "same format" figures were quoting different
        # statistics. Quartiles are order statistics and do not care which space they are
        # found in; the fences do. So the box is computed on the counts and only its
        # POSITIONS are transformed for drawing.
        stats = [{k: tf(s[k]) for k in ("med", "q1", "q3", "whislo", "whishi", "fliers")}
                 for s in (matplotlib.cbook.boxplot_stats(v, whis=1.5)[0] for v in vals)]
        # The reference holds position 0; the methods start one full pitch plus the gap out.
        gap = PCSZ_CBOX_GAP * PCSZ_CBOX_MARGIN
        pos = [0.0] + [i + gap for i in range(1, len(stats))]
        bp = ax.bxp(stats, positions=pos, widths=0.66, orientation="horizontal",
                    showfliers=True, patch_artist=True, manage_ticks=False, zorder=5,
                    # The house grey the whiskers already wear (INK, #514F52), not this
                    # family's pure black. Rotating the panel turned each method's outliers
                    # from a short spike above its box into a long band beside it, and at
                    # black they read as the loudest thing in the figure -- louder than the
                    # boxes whose comparison is the point. At 0.45 over white this lands
                    # near #B0B0B0 against the black version's #8C8C8C.
                    flierprops=dict(marker="d", markersize=2.4, markerfacecolor=INK,
                                    markeredgecolor="none", alpha=0.45))
        for box, label in zip(bp["boxes"], labels):
            # The method's own tint, the one it wears in the violin and the ECDF beside it.
            # soft() is keyed on the palette's "Reference ligand", not on the drawn name.
            box.set(facecolor=_pcsz_tint(REF_COLOR if label == PCSZ_REF_NAME
                                         else soft(label)),
                    edgecolor="none", linewidth=0)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                # THE GRID'S WHISKER INK (2026-09-14), not this family's. The rotatable-bond
                # grid draws whiskers and caps in the house INK -- #514F52, a warm grey --
                # while this family overrides its furniture to pure black; at the same 0.9
                # weight that black read as a harder, heavier line than the grid's. Only the
                # box's own furniture moves: the spines, ticks and names stay black.
                art.set(color=INK, linewidth=0.9)
        for m in bp["medians"]:
            m.set(color=RB_GRID_MED_COLOR, linewidth=PCSZ_CBOX_MED_LW, solid_capstyle="butt")
        # Tens of thousands of flier markers as vector paths make the PDF unplaceable.
        for fl in bp["fliers"]:
            fl.set_rasterized(True)
        # Capped: fliers past PCSZ_CBOX_HI fall OFF the panel rather than being clipped onto
        # its edge, because piling them on the edge would draw a spike at 200 that no method
        # actually has. The run log below still prints each method's true maximum.
        hi = min(max(float(v.max()) for v in vals), PCSZ_CBOX_HI)
        # THE VALUE AXIS IS X NOW (2026-09-14, rotated): decade names as powers of ten with
        # the log minor ticks between them, as the per-atom panel has them, so the two still
        # read as a pair of log axes -- evenly, now that tf is a true log10. The floor is
        # 10^0, so there is no 0 tick and the poses AT zero (1.6% of TargetDiff's, 10.9% of
        # AR's) sit below it, their whisker running off the LEFT edge. That those poses are
        # invisible is worth saying in the caption: a tenth of AR's poses clash with nothing
        # at all, which is the best thing about it and the panel cannot show it.
        decades = [t for t in PCSZ_CLASH_TICKS if t and t <= hi * 1.6]
        ax.set_xticks(tf(decades))
        ax.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in decades])
        ax.set_xticks(tf([k * t for t in decades for k in range(2, 10) if k * t <= hi]),
                      minor=True)
        ax.tick_params(axis="x", which="minor", direction="out", length=2.0,
                       width=PCSZ_CBOX_AXIS_LW * PCSZ_CBOX_MINOR_LW, colors=PCSZ_AXIS)
        ax.set_xlim(tf(1), tf(hi) + 0.06)
        # INVERTED, so position 0 -- the crystal ligands -- is the TOP row. A stack of rows is
        # read downward and matplotlib puts 0 at the bottom, which would have buried the
        # benchmark under the eight methods it exists to be compared against.
        ax.set_ylim(pos[-1] + PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN,
                    pos[0] - PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN)
        # The crystal ligands are the benchmark the methods are read against, so a rule
        # separates them rather than leaving them to look like a ninth method. It takes the
        # frame's colour AND weight exactly, with no alpha: at 0.45 the same black rendered as
        # a mid grey that read as a third kind of line -- neither the black spines it belongs
        # with nor the grey whiskers inside the panel.
        ax.axhline((pos[0] + pos[1]) / 2.0, color=PCSZ_AXIS, lw=PCSZ_CBOX_AXIS_LW, zorder=4)
        ax.set_yticks(pos)
        # TILTED, ANCHORED WHERE THE NAME ENDS (2026-09-14). rotation_mode="anchor" pins the
        # label's right end to its tick and rotates about that point, so the tick sits
        # exactly where the name stops and each name points at its own row.
        # Measured, this puts a label's bounding-box CENTRE a median 40 px below its tick
        # (54 px for the longest) -- but that is the geometry of an anchored rotation, not a
        # misalignment, because the eye follows a tilted name to its END. Centring the
        # rotated box on the tick instead scores 0 px by that measure and reads WORSE: no
        # part of the name then touches its tick and it floats between two rows.
        ax.set_yticklabels([l if l == PCSZ_REF_NAME else display(l) for l in labels],
                           fontsize=PCSZ_CBOX_NAME_FS, rotation=PCSZ_CBOX_NAME_ROT,
                           ha="right", va="center", rotation_mode="anchor")
        ax.set_xlabel("Steric clashes", fontsize=PCSZ_LABEL_FS, labelpad=PCSZ_YPAD)
        save(fig, out, "clash_box_by_method")
    write_csv(out, "clash_box_by_method",
              ["method", "n_clash", "clash_mean", "clash_median", "q25", "q75", "zero_frac"],
              [[l, len(v), round(float(v.mean()), 2), round(float(np.median(v)), 1),
                round(float(np.percentile(v, 25)), 1), round(float(np.percentile(v, 75)), 1),
                round(float((v == 0).mean()), 4)]
               for l, v in zip(labels, vals)])
    for l, v, m in zip(labels, vals, meds):
        print(f"    {l:16s} n={len(v):5d}  median {m:4.1f}  mean {v.mean():5.2f}  "
              f"zero-clash {100 * (v == 0).mean():4.1f}%  max {v.max():.0f}")
    print(f"  wrote {out}")


# ── fig-posecheck-clash-per-atom-all-methods ───────────────────────────────────────────
# The strain per-atom all-methods panel's twin, for clashes (2026-09-14).
#
# WHY THE EXISTING clash_per_atom_* FIGURES COULD NOT SIMPLY GAIN FIVE MORE LINES. They draw
# out of the ARMS run trees, and those trees cannot answer this question for eight methods:
#   1. THE FIVE BASELINES HAVE NO PER-MOLECULE POSECHECK THERE AT ALL -- 0 of ~37,000 samples
#      under baselines_pose/* carry posecheck.clashes. Their per-molecule counts exist only in
#      the svr12 exports, which is where the violin, the ECDF and the box already read them.
#   2. THE RECEPTOR SCOPE DIFFERS. The ARMS roots pose-scored against the 10 A crop
#      (pose_receptor_scope "crop", or absent, which metrics.py reads as crop); every other
#      clash figure in this family scores against the whole receptor. On the SAME arm and the
#      same 79 pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole
#      receptor reads 6.0 / 7.48 / 97 -- the crop deletes protein the pose could clash with,
#      so it under-counts, and it truncates the tail hardest.
# So this figure takes its five baselines from the exports, its three local arms from
# PCSZ_LOCAL (scope "full") and its reference from PCSZ_REF_CLASH_ROOT, and never from ARMS.
# It keeps the strain twin's windowing (+-RB_WIN atoms, drawn where >= MIN_N molecules pool)
# so the two panels are read the same way, and its house is the ECDF's, like every panel here.
PCSZ_CATOM_STATS = {"median": lambda v: float(np.median(v)),
                    "mean": lambda v: float(np.mean(v))}
# Dash per method, keyed as PCSZ_LINE_LW is: both tables are (.., lw, ls) last-two.
PCSZ_LINE_LS = {**{ALIASES.get(lab, lab): ls for _, lab, _, ls in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): ls for lab, _, _, ls in PCSZ_LOCAL}}


def _pcsz_atom_rows(root, keep, reference=False):
    """[{n, c}] per molecule for one whole-receptor run tree, over the pockets in `keep`."""
    rows = []
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            c = (m.get("posecheck") or {}).get("clashes")
            if c is not None:
                rows.append({"n": m["n_atoms"], "c": float(c)})
    return rows


def _pcsz_atom_keep():
    """The pockets every arm on this axis has: the baselines' 79, narrowed by ours."""
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    return keep


@figure("fig-posecheck-clash-per-atom-all-methods", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_per_atom_all_methods(out):
    """Steric clashes against ligand size, all eight methods and the crystal ligands."""
    keep = _pcsz_atom_keep()
    series = {}
    for key, label, *_ in PCSZ_BASELINES:
        # The exports' rows are already {n, s, c}; this reads the same file the violin does.
        series[label], _ = _rb_atom_baseline_rows(key, keep)
    for label, root, *_ in PCSZ_LOCAL:
        series[label] = _pcsz_atom_rows(root, keep)
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)

    labels = [label for _, label, *_ in PCSZ_BASELINES] + [lab for lab, *_ in PCSZ_LOCAL]
    per = {lab: by_size(series[lab], "c") for lab in labels}
    ref_per = by_size(refrows, "c")
    xs = list(range(RB_X_LO, RB_X_HI + 1))

    for stat, f in PCSZ_CATOM_STATS.items():
        with plt.rc_context(PCSZ_RC):
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR,
                    lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=SOLID, zorder=4,
                    solid_capstyle="round")
            for lab in labels:
                key = ALIASES.get(lab, lab)
                ax.plot(xs, _rb_curve(per[lab], xs, f), color=soft(lab),
                        lw=PCSZ_LINE_LW[key] * PCSZ_LW_SCALE, ls=PCSZ_LINE_LS[key],
                        zorder=5, solid_capstyle="round")
            ax.set_ylim(bottom=0)
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
            ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
            ax.set_ylabel(f"Clashes {stat}\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_per_atom_all_methods_{stat}")

    rows = []
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        cur = {s: (reference_curve(p, xs, f) if lab == PCSZ_REF_NAME
                   else _rb_curve(p, xs, f)) for s, f in PCSZ_CATOM_STATS.items()}
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ()))]
                        + ["" if cur[s][i] is None else round(cur[s][i], 2)
                           for s in ("median", "mean")])
    write_csv(out, "clash_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "clash_median", "clash_mean"], rows)

    print(f"  {len(keep)} pockets · {len(labels)} methods + reference · whole-receptor "
          f"scope · x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, drawn "
          f"where it pools ≥{MIN_N} molecules")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>6d}" for a in heads) + f" {'drawn':>9s}   (clash median at n)")
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        med = (reference_curve(p, xs, PCSZ_CATOM_STATS["median"]) if lab == PCSZ_REF_NAME
               else _rb_curve(p, xs, PCSZ_CATOM_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:6.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>6s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        span = f"{drawn[0]}-{drawn[-1]}" if drawn else "—"
        n_mol = sum(len(v) for v in p.values())
        mean_at = np.mean([n for n, v in p.items() for _ in v]) if n_mol else float("nan")
        print(f"  {lab:16s} {n_mol:6d} {mean_at:6.1f} " + " ".join(cells) + f" {span:>9s}")
    print(f"  wrote {out}")
