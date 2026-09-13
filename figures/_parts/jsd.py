

# ════════════════════════════════════════════════════════════════════════════════
# fig-jsd-{bond,bond-length,pair,atom-type,summary,ring-size,n-rings,aromatic,rings}
# ════════════════════════════════════════════════════════════════════════════════
# The distribution metrics of TargetDiff and VoxBind: how closely each arm's bond lengths,
# atom-pair distances, element mix and rings follow CrossDocked ligands.
#
#     bond         bond-distance JSD per bond type          VoxBind Table 2
#     bond-length  the length histograms those JSDs compare
#     pair         all-atom (<12 Å) and C-C (<2 Å) pair distances   TargetDiff Fig. 2
#     atom-type    heavy-atom element shares
#     summary      the four headline JSDs side by side
#     ring-size    share of rings by size, 3-9               TargetDiff Table 2
#     n-rings      rings per molecule                        VoxBind Fig. 10, middle
#     aromatic     aromatic share of atoms and of rings      VoxBind Fig. 10, bottom
#     rings        all three, one column per method          VoxBind Fig. 10
#
# THESE ONLY DRAW. voxbind/scripts/89_eval_crossdocked_jsd.sh computes everything into one
# JSON; its tool's docstring carries the protocol and the self-check that pins it to the
# published TargetDiff row. What a reader of the figures needs from it:
#
# THE REFERENCE DRAWN IS EVERY CROSSDOCKED LIGAND of the train+test split, each distinct
# molecule weighing one (100,100 files are 8,829 molecules; one is cross-docked into 869
# pockets). The published papers score against the 100 test ligands instead, which are
# too few: a model sampling the training distribution exactly would score C=N JSD 0.52
# against them (16 bonds). Those numbers are still computed -- `jsd_test` in the JSON, the
# CSVs here and the appendix tables of jsd-tables.ipynb -- and so is a pose-weighted
# reference, as the check that the de-duplication does not move anything.
#
# THE ARMS stand on the 79 electron-density pockets (TargetDiff and VoxBind hold 100). JSD
# is scipy's JS DISTANCE, as both papers print it.
JSD_JSON = BUNDLE / "_shared" / "260913_crossdocked_jsd" / "crossdocked_jsd.json"
JSD_NEEDS = ("crossdocked_jsd.json (voxbind/scripts/89_eval_crossdocked_jsd.sh)",)
# Bond lengths are SCORED in 0.005 Å bins; they are DRAWN four bins at a time, so a
# histogram reads as a shape rather than a comb. The JSD columns still come from the
# scoring bins.
JSD_LEN_MERGE = 4
JSD_CC_XLIM = (1.0, 2.0)        # C-C pairs under 2 Å are bonds; nothing sits below 1.0
JSD_NRINGS_MAX = 8              # rings per molecule: the last bar pools 8 and more
JSD_AROM_MERGE = 2              # 20 scored aromatic-fraction bins drawn as 10
JSD_SUMMARY = (
    ("Mean bond-distance JSD ↓", lambda j: j["bond_mean"]),
    ("All-atom pair JSD ↓", lambda j: j["pair"]["All_12A"]),
    ("C–C pair JSD ↓", lambda j: j["pair"]["CC_2A"]),
    ("Atom-type JSD ↓", lambda j: j["atom_type"]),
)
# (CSV source label, arm key) for every reference the eval scores against; the first is
# the one the figures draw.
JSD_REFERENCES = (("vs CrossDocked train+test (distinct molecules)", "jsd"),
                  ("vs CrossDocked train+test (pose-weighted)", "jsd_pose_weighted"),
                  ("vs CrossDocked test (100)", "jsd_test"))
_JSD_CACHE = {}


def _jsd_data():
    """(the eval JSON, the ARMS it covers in ARMS order). An arm the JSON lacks is named
    and skipped rather than drawn empty."""
    if "d" not in _JSD_CACHE:
        if not JSD_JSON.exists():
            raise FileNotFoundError(f"missing input {JSD_JSON}\n"
                                    f"  rebuild it with bash voxbind/scripts/89_eval_crossdocked_jsd.sh")
        d = json.load(open(JSD_JSON))
        if "reference_test" not in d:
            raise ValueError(f"{JSD_JSON} predates the train+test reference; "
                             f"rerun bash voxbind/scripts/89_eval_crossdocked_jsd.sh")
        arms = [a for a in ARMS if a[0] in d["arms"]]
        if not arms:
            raise ValueError(f"{JSD_JSON} holds none of the ARMS labels")
        missing = [a[0] for a in ARMS if a[0] not in d["arms"]]
        pockets = {d["arms"][lab]["n_pockets"] for lab, _, _ in arms}
        ref = d["reference"]
        print(f"  [{os.path.relpath(JSD_JSON, REPO)}]")
        print(f"  reference: CrossDocked train+test, {ref['n_files']:,} ligand files = "
              f"{ref['n_unique_ligands']:,} distinct molecules, each weighing 1 · "
              f"arms over {'/'.join(map(str, sorted(pockets)))} pockets")
        sc = d.get("selfcheck")
        if sc:
            print(f"  selfcheck vs published TargetDiff (test reference): max |Δ| bond JSD "
                  f"{sc['max_abs_diff_bond_jsd']:.4f}, ring % {sc['max_abs_diff_ring_pct']:.2f}"
                  f" -> {'PASS' if sc['pass'] else 'FAIL'}")
        if missing:
            print("  not in the JSON, not drawn: " + ", ".join(missing))
        _JSD_CACHE["d"] = (d, arms)
    return _JSD_CACHE["d"]


def _jsd_ref_key():
    """The reference's legend entry, with its size -- distinct molecules, not files."""
    ref = _jsd_data()[0]["reference"]
    return f"{ref['label']}, {ref['n_unique_ligands']:,} molecules"


def _jsd_ref_sets(d):
    """(CSV label, summary) of every reference whose histograms a CSV should carry."""
    return [(f"{REF_LABEL} (train+test)", d["reference"]),
            (f"{REF_LABEL} (train+test, pose-weighted)", d["reference_pose_weighted"]),
            (f"{REF_LABEL} (test)", d["reference_test"])]


def _jsd_variants(arms):
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


def _jsd_new(w, h):
    fig = plt.figure(figsize=(w, h), dpi=220)
    fig.patch.set_facecolor("white")
    return fig


def _jsd_key_above(fig, handles, ncol, fontsize=11.5):
    """A figure-level key in a strip above every panel. Laid out AFTER fit(), because
    tight_layout does not know about figure legends: the panels are fitted first, then
    pushed down by exactly the overlap the key makes with the tallest one."""
    fit(fig, pad=0.5)
    leg = legend(fig, handles, loc="upper center", ncol=ncol, fontsize=fontsize,
                 bbox_to_anchor=(0.5, 0.995))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = leg.get_window_extent(rend).transformed(inv).y0
    y1 = max(ax.get_tightbbox(rend).transformed(inv).y1 for ax in fig.axes)
    if y1 > y0 - 0.012:
        fig.subplots_adjust(top=fig.subplotpars.top - (y1 - y0 + 0.012))


def _jsd_handles(arms, *, ref=True, patch=False):
    def one(label, col, ls="-", lw=MODEL_LW):
        if patch:
            return Patch(facecolor=col, edgecolor=col, label=label)
        return Line2D([], [], color=col, lw=lw, ls=ls, label=label)
    hs = [one(_jsd_ref_key(), REF_COLOR, DASH, REF_LW)] if ref else []
    return hs + [one(display(lab), color(lab)) for lab, _, _ in arms]


def _jsd_key_cols(n, wide):
    return n if n <= 3 else (5 if wide else 3)


def _jsd_grouped(ax, groups, series):
    """Grouped bars: one group per x category, one bar per (label, values, colour) in
    `series` order, so the key's order is the bars' order. Bar width is derived from the
    series count -- a fixed width overlaps neighbouring groups at nine series."""
    x = np.arange(len(groups))
    w = 0.84 / len(series)
    for i, (_, vals, col) in enumerate(series):
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, width=w, color=col,
               edgecolor="white", lw=0.4 if len(series) > 4 else 0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, len(groups) - 0.5)


def _jsd_round(v, nd=4):
    """A JSON number for a CSV cell. None is a JSD with nothing to compare (an arm with no
    bond of that type) and stays an empty cell rather than becoming 0."""
    return "" if v is None else round(v, nd)


def _jsd_pct(counts):
    counts = np.asarray(counts, float)
    return 100 * counts / counts.sum() if counts.sum() else counts


# ── bond distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-bond", needs=JSD_NEEDS)
def draw_jsd_bond(out):
    """Bond-distance JSD per bond type against the CrossDocked ligands (VoxBind Table 2)."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * 1.45, PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(lab, [d["arms"][lab]["jsd"]["bond"][t] for t in types]
                   + [d["arms"][lab]["jsd"]["bond_mean"]], color(lab)) for lab, _, _ in arms]
        _jsd_grouped(ax, types + ["mean"], series)
        top = max(v for _, vals, _ in series for v in vals if v is not None)
        furniture(ax, ylabel="JSD to CrossDocked\nligands ↓", xloc=None)
        # AFTER furniture(), whose tick_params would reset the size
        ax.set_xticklabels(types + ["mean"], fontsize=13)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.axvline(len(types) - 0.5, color=AXIS, lw=GRID_LW, zorder=1)
        ax.set_ylim(0, top * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, ref=False, patch=True),
                       _jsd_key_cols(len(arms), wide))
        save(fig, out, f"bond_{variant}")

    rows = []
    for source, key in JSD_REFERENCES:
        rows += [[lab, source, d["arms"][lab]["n_mols"]]
                 + [_jsd_round(d["arms"][lab][key]["bond"][t]) for t in types]
                 + [_jsd_round(d["arms"][lab][key]["bond_mean"])] for lab, _, _ in all_arms]
    for lab, s in _jsd_ref_sets(d):
        rows.append([f"{lab}: bonds", "reference bond count (weighted where de-duplicated)",
                     s["n_mols"]] + [s["bond_n"][t] for t in types] + [""])
    for lab, vals in d["published"]["bond_jsd"].items():
        rows.append([lab, "published, vs CrossDocked test (100 pockets)", ""] + vals
                    + [round(float(np.mean(vals)), 4)])
    write_csv(out, "bond_jsd", ["set", "source", "n_mols"] + types + ["mean"], rows)


def _jsd_method_grid(out, d, arms, kinds, *, xlabel, stem):
    """Rows are methods, columns are what is measured. Every panel is ONE method's
    distribution over the reference's dashed outline, with that panel's JSD in its corner.

    The overlay puts nine curves in one axes, where the arm you are looking for is the one
    you cannot see; here each is read against the reference alone. y is shared down a
    column, so a method's peak height still compares with the other methods'.

    `kinds` is [(column title, set -> (pct, edges), arm -> JSD, xlim, x tick step)]."""
    n, m = len(arms), len(kinds)
    fig = _jsd_new(1.95 * m + 1.7, 1.12 * n + 1.5)
    axes = fig.subplots(n, m, sharex="col", sharey="col", squeeze=False)
    for j, (title, dist, jsd_of, xlim, xstep) in enumerate(kinds):
        ref_pct, _ = dist(d["reference"])
        top = float(ref_pct.max())
        for i, (lab, _, _) in enumerate(arms):
            ax = axes[i][j]
            ax.set_facecolor("white")
            pct, edges = dist(d["arms"][lab])
            top = max(top, float(pct.max()))
            ax.stairs(pct, edges, color=color(lab), fill=True, alpha=DIST_FILL + 0.14, lw=0, zorder=2)
            ax.stairs(pct, edges, color=color(lab), lw=DIST_LW - 0.4, zorder=3)
            ax.stairs(ref_pct, edges, color=REF_COLOR, lw=REF_LW - 0.2, ls=DASH, zorder=4)
            furniture(ax, xloc=xstep, xlim=xlim, xlabel=xlabel if i == n - 1 else None)
            ax.tick_params(labelsize=9.5, length=2.5)
            ax.xaxis.label.set_size(11.5)
            v = jsd_of(d["arms"][lab])
            ax.text(0.97, 0.93, "JSD —" if v is None else f"JSD {v:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
            if j == 0:
                ax.set_ylabel(display(lab), fontsize=12, labelpad=8)
            if i == 0:
                ax.set_title(title, fontsize=11, color=INK, pad=6)
        axes[0][j].set_ylim(0, top * 1.12)
    # The key names the reference only: the y unit is in each column head, and a longer key
    # outran the two-column pair grid and was cut off at both edges.
    _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                label=_jsd_ref_key())], 1, fontsize=11)
    save(fig, out, stem)


@figure("fig-jsd-bond-length", needs=JSD_NEEDS)
def draw_jsd_bond_length(out):
    """The bond-length histograms behind fig-jsd-bond, one panel per bond type."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    bins = np.asarray(d["protocol"]["bond_bins"])
    step = float(bins[1] - bins[0])

    def drawn(counts):
        """Share of ALL of a set's bonds per merged bin. Index 0 and the last index are the
        under/overflow bins (searchsorted over the edges): they count in the denominator,
        so a set with many out-of-range bonds shows less mass, and are not drawn."""
        counts = np.asarray(counts, float)
        inner = counts[1:-1]
        inner = np.concatenate([inner, np.zeros((-len(inner)) % JSD_LEN_MERGE)])
        merged = inner.reshape(-1, JSD_LEN_MERGE).sum(1)
        edges = bins[0] + step * JSD_LEN_MERGE * np.arange(len(merged) + 1)
        # 118 inner bins pad to 120, but the last scoring edge is 1.69 Å: longer bonds are
        # overflow, so the last bar must not claim [1.69, 1.70) it holds nothing from
        edges[-1] = bins[-1]
        return (100 * merged / counts.sum() if counts.sum() else merged), edges

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x bond-type grid (see _jsd_method_grid); core keeps the
            # overlay, where two curves over the reference still read directly
            _jsd_method_grid(out, d, arms, [
                (f"{t}\n% per {step * JSD_LEN_MERGE:.2f} Å",
                 (lambda s, t=t: drawn(s["bond_counts"][t])),
                 (lambda a, t=t: a["jsd"]["bond"][t]), (bins[0], bins[-1]), 0.2) for t in types],
                xlabel="Bond length (Å)", stem=f"bond_length_{variant}")
            continue
        fig = _jsd_new(FIG_W * 1.55, PANEL_H * 1.62)
        axes = fig.subplots(2, 4)
        for i, t in enumerate(types):
            ax = axes.flat[i]
            ax.set_facecolor("white")
            for lab, _, _ in arms:
                pct, edges = drawn(d["arms"][lab]["bond_counts"][t])
                ax.stairs(pct, edges, color=color(lab), lw=DIST_LW, zorder=3)
                ax.stairs(pct, edges, color=color(lab), fill=True, alpha=DIST_FILL, lw=0, zorder=2)
            pct, edges = drawn(d["reference"]["bond_counts"][t])
            ax.stairs(pct, edges, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xloc=0.2, xlim=(bins[0], bins[0] + step * 120),
                      ylabel=f"% of bonds\nper {step * JSD_LEN_MERGE:.2f} Å" if i % 4 == 0 else None,
                      xlabel="Bond length (Å)" if i >= 4 else None)
            ax.tick_params(labelsize=11.5)
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.set_ylim(bottom=0)
            # a panel LABEL, not a title
            ax.set_title(t, loc="left", fontsize=12.5, color=INK, pad=5)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"bond_length_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for t in types:
            c = s["bond_counts"][t]
            total = sum(c)
            lo = [float("-inf")] + bins.tolist()
            hi = bins.tolist() + [float("inf")]
            rows += [[lab, t, round(a, 3), round(b, 3), n, round(100 * n / total, 4) if total else 0]
                     for a, b, n in zip(lo, hi, c)]
    write_csv(out, "bond_length_hist", ["set", "bond_type", "len_lo", "len_hi", "n", "pct"], rows)


# ── pair distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-pair", needs=JSD_NEEDS)
def draw_jsd_pair(out):
    """All-atom (<12 Å) and C–C (<2 Å) pair-distance distributions (TargetDiff Fig. 2)."""
    d, all_arms = _jsd_data()
    use_style()
    panels = (("All_12A", "All heavy-atom pairs · distance (Å)", None),
              ("CC_2A", "C–C pairs under 2 Å · distance (Å)", JSD_CC_XLIM))

    def dist(key):
        edges = np.asarray(d["protocol"]["pair_bins"][key])
        return lambda s: (_jsd_pct(s["pair_counts"][key])[1:-1], edges)

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x pair-kind grid (see _jsd_method_grid); core keeps the overlay
            step = {k: float(np.diff(d["protocol"]["pair_bins"][k][:2])[0]) for k, _, _ in panels}
            _jsd_method_grid(out, d, arms, [
                (f"All heavy-atom pairs < 12 Å\n% per {step['All_12A']:.2f} Å", dist("All_12A"),
                 lambda a: a["jsd"]["pair"]["All_12A"], (0, 12), 4),
                (f"C–C pairs < 2 Å\n% per {step['CC_2A']:.2f} Å", dist("CC_2A"),
                 lambda a: a["jsd"]["pair"]["CC_2A"], JSD_CC_XLIM, 0.5)],
                xlabel="Distance (Å)", stem=f"pair_{variant}")
            continue
        fig = _jsd_new(FIG_W * 1.45, PANEL_H * 1.05)
        axes = fig.subplots(1, 2)
        for ax, (key, xlabel, xlim) in zip(axes, panels):
            ax.set_facecolor("white")
            edges = np.asarray(d["protocol"]["pair_bins"][key])
            for lab, _, _ in arms:
                ax.stairs(_jsd_pct(d["arms"][lab]["pair_counts"][key])[1:-1], edges,
                          color=color(lab), lw=DIST_LW, zorder=3)
            ax.stairs(_jsd_pct(d["reference"]["pair_counts"][key])[1:-1], edges,
                      color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xlabel=xlabel, xloc=None, xlim=xlim or (edges[0], edges[-1]),
                      ylabel=f"% of pairs per {edges[1] - edges[0]:.2f} Å")
            ax.tick_params(labelsize=12.5)
            ax.xaxis.label.set_size(13.5)
            ax.yaxis.label.set_size(13.5)
            ax.set_ylim(bottom=0)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"pair_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for key, _, _ in panels:
            edges = d["protocol"]["pair_bins"][key]
            pct = _jsd_pct(s["pair_counts"][key])
            rows += [[lab, key, round(a, 4), round(b, 4), n, round(float(q), 4)]
                     for a, b, n, q in zip(edges[:-1], edges[1:], s["pair_counts"][key][1:-1], pct[1:-1])]
    write_csv(out, "pair_hist", ["set", "pairs", "dist_lo", "dist_hi", "n", "pct"], rows)


# ── atom types and the headline numbers ─────────────────────────────────────────
@figure("fig-jsd-atom-type", needs=JSD_NEEDS)
def draw_jsd_atom_type(out):
    """Heavy-atom element shares against the CrossDocked ligands, log scale."""
    d, all_arms = _jsd_data()
    use_style()
    elems = d["protocol"]["atom_types"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        # Log, because C sits at ~70% and Cl at ~0.5%: on a linear axis everything past O
        # is a stub. An exact zero cannot sit on a log axis and is left as no bar.
        ax.set_yscale("log")
        series = [(REF_LABEL, [100 * d["reference"]["atom_frac"][e] or None for e in elems], REF_COLOR)]
        series += [(lab, [100 * d["arms"][lab]["atom_frac"][e] or None for e in elems], color(lab))
                   for lab, _, _ in arms]
        _jsd_grouped(ax, elems, series)
        furniture(ax, ylabel="% of heavy atoms\n(log scale)", xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.05, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"atom_type_{variant}")

    rows = [[lab, s["n_mols"]] + [round(100 * s["atom_frac"][e], 3) for e in elems]
            + [round(100 * s["atom_other_frac"], 3), "", "", ""] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, d["arms"][lab]["n_mols"]] + [round(100 * d["arms"][lab]["atom_frac"][e], 3) for e in elems]
             + [round(100 * d["arms"][lab]["atom_other_frac"], 3)]
             + [_jsd_round(d["arms"][lab][key]["atom_type"]) for _, key in JSD_REFERENCES]
             for lab, _, _ in all_arms]
    write_csv(out, "atom_type", ["set", "n_mols"] + [f"pct_{e}" for e in elems]
              + ["pct_other_not_scored"] + [f"atom_type_{key}" for _, key in JSD_REFERENCES], rows)


@figure("fig-jsd-summary", needs=JSD_NEEDS)
def draw_jsd_summary(out):
    """Mean bond, all-atom pair, C–C pair and atom-type JSD per arm, side by side."""
    d, all_arms = _jsd_data()
    use_style()
    for variant, arms in _jsd_variants(all_arms):
        fig = _jsd_new(FIG_W * 1.75, 0.5 * len(arms) + 1.55)
        axes = fig.subplots(1, len(JSD_SUMMARY), sharey=True)
        ys = np.arange(len(arms))[::-1]
        for ax, (name, get) in zip(axes, JSD_SUMMARY):
            ax.set_facecolor("white")
            vals = [get(d["arms"][lab]["jsd"]) for lab, _, _ in arms]
            ax.barh(ys, [np.nan if v is None else v for v in vals], height=0.66,
                    color=[color(lab) for lab, _, _ in arms], zorder=3)
            top = max([v for v in vals if v is not None], default=1.0)
            # every value is labelled: this figure IS the table, drawn. A None is a JSD with
            # nothing to compare and says so, rather than drawing as a zero-length "best"
            for y, v in zip(ys, vals):
                ax.text((v or 0) + top * 0.03, y, "—" if v is None else f"{v:.3f}",
                        va="center", ha="left", fontsize=11.5, color=INK, zorder=4)
            furniture(ax, xlabel=name, xloc=None, xlim=(0, top * 1.38))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.label.set_size(13)
            ax.tick_params(labelsize=11.5)
            ax.grid(False, axis="y")
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels([display(lab) for lab, _, _ in arms], fontsize=13)
        axes[0].set_ylim(-0.6, len(arms) - 0.4)
        fit(fig, pad=0.5)
        save(fig, out, f"summary_{variant}")
    head = ["set", "n_mols", "n_disconnected_dropped"]
    cols = ("bond_mean", "pair_all_12A", "pair_cc_2A", "atom_type")
    for _, key in JSD_REFERENCES:
        head += [f"{c}_{key}" for c in cols]
    write_csv(out, "summary", head,
              [[lab, d["arms"][lab]["n_mols"], d["arms"][lab]["n_disconnected"]]
               + [_jsd_round(get(d["arms"][lab][key])) for _, key in JSD_REFERENCES for _, get in JSD_SUMMARY]
               for lab, _, _ in all_arms])


# ── rings ───────────────────────────────────────────────────────────────────────
@figure("fig-jsd-ring-size", needs=JSD_NEEDS)
def draw_jsd_ring_size(out):
    """Share of rings by size 3–9 (TargetDiff Table 2), with the table itself as CSV."""
    d, all_arms = _jsd_data()
    use_style()
    sizes = d["published"]["ring_size_pct_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(REF_LABEL, [d["reference"]["ring_size_pct"][s] for s in sizes], REF_COLOR)]
        series += [(lab, [d["arms"][lab]["ring_size_pct"][s] for s in sizes], color(lab))
                   for lab, _, _ in arms]
        # Log, because the shares span three decades -- 6-rings at 50-83%, 7/8/9-rings and the
        # reference's 3-rings at 0.1-2% -- and on a linear axis every size but 5 and 6 was a
        # stub. A share of exactly zero has no place on a log axis and draws no bar.
        ax.set_yscale("log")
        _jsd_grouped(ax, [f"{s}-ring" for s in sizes],
                     [(lab, [v or None for v in vals], col) for lab, vals, col in series])
        furniture(ax, ylabel="% of rings, sizes 3–9\n(log scale)", xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        # floor at 0.01%: a 0.05% floor swallowed the 0.05-0.06% eight-rings of CoDE and FuncBind
        ax.set_ylim(0.01, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"ring_size_{variant}")

    head = ["set", "source"] + sizes + ["10+ of all rings"]
    rows = [[lab, "reference"] + [round(s["ring_size_pct"][z], 1) for z in sizes]
            + [round(s["ring_size_pct"]["10+ of all"], 1)] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, f"this eval ({d['arms'][lab]['n_pockets']} pockets)"]
             + [round(d["arms"][lab]["ring_size_pct"][s], 1) for s in sizes]
             + [round(d["arms"][lab]["ring_size_pct"]["10+ of all"], 1)] for lab, _, _ in all_arms]
    rows += [[lab, "published TargetDiff Table 2 (reference = test)"] + vals + [""]
             for lab, vals in d["published"]["ring_size_pct"].items()]
    write_csv(out, "ring_size_pct", head, rows)


@figure("fig-jsd-n-rings", needs=JSD_NEEDS)
def draw_jsd_n_rings(out):
    """Share of molecules by number of rings (VoxBind Fig. 10, middle row), every method in one axes."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(REF_LABEL, _jsd_ring_rows(d["reference"])[1], REF_COLOR)]
        series += [(lab, _jsd_ring_rows(d["arms"][lab])[1], color(lab)) for lab, _, _ in arms]
        _jsd_grouped(ax, labels, series)
        furniture(ax, ylabel="% of molecules", xlabel="Rings per molecule", xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"n_rings_{variant}")

    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    write_csv(out, "n_rings", ["set", "mean_rings_per_molecule"] + [f"pct_{b}" for b in labels],
              [[lab, round(s["mean_n_rings"], 3)] + [round(float(q), 3) for q in _jsd_ring_rows(s)[1]]
               for lab, s in sets])


@figure("fig-jsd-aromatic", needs=JSD_NEEDS)
def draw_jsd_aromatic(out):
    """Aromatic share of each molecule's heavy atoms (VoxBind Fig. 10, bottom row) and of its rings."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [f"{k / 10:.1f}–{(k + 1) / 10:.1f}" for k in range(10)]
    # Two definitions, one panel each. VoxBind Fig. 10 plots the aromatic share of a
    # molecule's heavy ATOMS; the share of its RINGS that are aromatic (every ring bond
    # aromatic) says the same thing without the ring-free atoms diluting it, and is only
    # defined for a molecule that has a ring -- so its panel counts those molecules alone.
    panels = (
        ("Aromatic share of a molecule's heavy atoms",
         lambda s: _jsd_ring_rows(s)[2], "mean_arom_atom_frac", "% of molecules"),
        ("Aromatic share of a molecule's rings (molecules with ≥1 ring)",
         lambda s: _jsd_pct(s["arom_ring_frac_counts"]), "mean_arom_ring_frac", "% of molecules\nwith a ring"),
    )
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H * 1.9)
        axes = fig.subplots(2, 1)
        for ax, (xlabel, get, _, ylabel) in zip(axes, panels):
            ax.set_facecolor("white")
            series = [(REF_LABEL, get(d["reference"]), REF_COLOR)]
            series += [(lab, get(d["arms"][lab]), color(lab)) for lab, _, _ in arms]
            _jsd_grouped(ax, labels, series)
            furniture(ax, ylabel=ylabel, xlabel=xlabel, xloc=None)
            ax.tick_params(axis="x", length=0, labelsize=11 if wide else 10)
            ax.grid(False, axis="x")
            ax.xaxis.label.set_size(13.5)
            ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"aromatic_{variant}")

    rows = []
    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    for name, (_, get, mean_key, _) in zip(("atoms", "rings"), panels):
        rows += [[lab, name, _jsd_round(s[mean_key])] + [round(float(q), 3) for q in get(s)]
                 for lab, s in sets]
    write_csv(out, "aromatic", ["set", "aromatic_share_of", "mean"] + [f"pct_{b}" for b in labels], rows)


JSD_RING_ROWS = (
    ([str(k) for k in range(3, 10)] + ["≥10"], "Ring size\n% of rings", (0, 2, 4, 6, 7)),
    ([str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"],
     "Rings per molecule\n% of molecules", (0, 2, 4, 6, 8)),
    (None, "Aromatic atom fraction\n% of molecules", None),
)


def _jsd_ring_rows(s):
    """The three Fig. 10 histograms of one set, in percent: ring sizes over ALL rings
    (10+ pooled), rings per molecule (8+ pooled), aromatic share of heavy atoms."""
    rs = {int(k): v for k, v in s["ring_size_counts"].items()}
    sizes = [rs.get(k, 0) for k in range(3, 10)] + [sum(v for k, v in rs.items() if k >= 10)]
    nr = {int(k): v for k, v in s["n_rings_counts"].items()}
    counts = [nr.get(k, 0) for k in range(JSD_NRINGS_MAX)] \
        + [sum(v for k, v in nr.items() if k >= JSD_NRINGS_MAX)]
    arom = np.asarray(s["arom_frac_counts"], float).reshape(-1, JSD_AROM_MERGE).sum(1)
    return [_jsd_pct(sizes), _jsd_pct(counts), _jsd_pct(arom)]


@figure("fig-jsd-rings", needs=JSD_NEEDS)
def draw_jsd_rings(out):
    """VoxBind Fig. 10: ring sizes, rings per molecule and aromatic fraction, one column per arm."""
    d, all_arms = _jsd_data()
    use_style()
    ref_rows = _jsd_ring_rows(d["reference"])
    for variant, arms in _jsd_variants(all_arms):
        n = len(arms)
        tick_fs = 10 if n > 2 else 11.5
        fig = _jsd_new(max(FIG_W, 2.05 * n + 1.5), 7.2)
        # The paper's layout: a column per method, the reference as the same dashed outline
        # in every panel, and y shared along a row so columns compare by eye. Column heads
        # name the method -- the one place this family puts text above an axes.
        axes = fig.subplots(3, n, sharey="row", squeeze=False)
        per = {lab: _jsd_ring_rows(d["arms"][lab]) for lab, _, _ in arms}
        for c, (lab, _, _) in enumerate(arms):
            rows = per[lab]
            for r, (labels, ylabel, ticks) in enumerate(JSD_RING_ROWS):
                ax = axes[r][c]
                ax.set_facecolor("white")
                if labels is None:
                    edges = np.linspace(0, 1, len(rows[r]) + 1)
                    x = (edges[:-1] + edges[1:]) / 2
                    width = (edges[1] - edges[0]) * 0.86
                else:
                    x = np.arange(len(rows[r]))
                    edges = np.arange(len(rows[r]) + 1) - 0.5
                    width = 0.78
                ax.bar(x, rows[r], width=width, color=color(lab), lw=0, zorder=3)
                ax.stairs(ref_rows[r], edges, color=REF_COLOR, lw=REF_LW, ls=DASH,
                          baseline=None, zorder=4)
                furniture(ax, xloc=None, ylabel=ylabel if c == 0 else None)
                ax.grid(False, axis="x")
                ax.tick_params(labelsize=tick_fs)
                ax.yaxis.label.set_size(12)
                if labels is None:
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(["0", "0.5", "1"])
                else:
                    ax.set_xlim(edges[0], edges[-1])
                    ax.set_xticks(list(ticks))
                    ax.set_xticklabels([labels[i] for i in ticks])
            axes[0][c].set_title(display(lab), fontsize=13, color=INK, pad=7)
        # The row's y range is set ONCE, from every column and the reference. A per-panel
        # set_ylim(bottom=0) freezes the shared top at whatever the columns drawn so far
        # reached, and clipped FuncBind's 83% six-ring bar at AR's 70%.
        for r in range(len(JSD_RING_ROWS)):
            top = max(max(float(per[lab][r].max()) for lab in per), float(ref_rows[r].max()))
            axes[r][0].set_ylim(0, top * 1.08)
        _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                    label=_jsd_ref_key())], 1)
        save(fig, out, f"rings_{variant}")

    rows = []
    names = ("ring_size", "rings_per_molecule", "aromatic_fraction")
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for name, (labels, _, _), pct in zip(names, JSD_RING_ROWS, _jsd_ring_rows(s)):
            if labels is None:
                labels = [f"{i / len(pct):.1f}-{(i + 1) / len(pct):.1f}" for i in range(len(pct))]
            rows += [[lab, name, b, round(float(q), 3)] for b, q in zip(labels, pct)]
    write_csv(out, "rings_hist", ["set", "histogram", "bin", "pct"], rows)
