

# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-rotbond, fig-posecheck-strain-box-per-rotbond,
# fig-posecheck-strain-rotbond-all-methods, fig-posecheck-strain-per-atom-all-methods
# ════════════════════════════════════════════════════════════════════════════════
# STRAIN AGAINST THE NUMBER OF ROTATABLE BONDS, and the eight-method version of the same
# metric on both axes. The VoxBind paper's Fig. 12/13 form.
#
# WHY THE AXIS IS WORTH SWAPPING. Strain is conformational: it is the energy a pose carries
# because its torsions are not where the force field would put them, so the number of
# torsions a molecule HAS is the more direct explanatory variable, and heavy-atom count is
# its proxy. The two are not interchangeable -- a fused polycyclic and a long-chain ligand
# of equal size have very different torsional freedom -- and the ranking of the arms is
# allowed to differ between the two views.
#
# ROTATABLE BONDS ARE COUNTED FROM THE SMILES that `metrics.json` already records, with
# RDKit's default (strict) `CalcNumRotatableBonds`, which excludes amides, terminal bonds
# and ring bonds. It is a topological descriptor, so reading it off the recorded SMILES
# rather than the pose gives the same integer with no risk of a sample-to-SDF index slip.
# Molecules whose SMILES will not parse are dropped and counted in the run log.
#
# THE CRYSTAL REFERENCE WINDOW IS +-1 HERE, NOT the header's +-4. There is one crystal
# ligand per pocket, so a per-count reference curve over 79 ligands would be noise -- that
# is why the window exists at all. But rotatable-bond counts run 0-14 where ligand sizes run
# 5-45, and a +-4 window there spans two thirds of the axis and would flatten the reference
# into a near-constant line. +-1 pools 12-29 ligands per point and the line stops at 9,
# where the window falls under the minimum, rather than being extended into an invented
# value. The originals got this by MUTATING pose_common.REF_WIN at import; in one process
# that draws every figure a module-level override would silently re-window the per-atom
# figures next door, so the narrow window is a local constant and a local curve function.
#
# THE BOXES ARE THE PRIMARY FIGURE AND THE LINES ARE KEPT BESIDE THEM. A median line says
# where an arm sits; it cannot say whether two arms a factor of 1.5 apart are actually
# separated, and on a metric whose distribution spans four decades inside a single bond
# count that is the question. The boxes answer it -- and they show something the lines
# cannot: the arms' inter-quartile ranges overlap almost completely at every count, so the
# median ordering is a shift of a wide distribution, not a separation of two narrow ones.
#
# WHISKERS ARE THE 5TH AND 95TH PERCENTILES AND FLIERS ARE NOT DRAWN. Not a cosmetic
# choice: 4-7% of molecules relax to 1e4-1e13, so Tukey whiskers with fliers would put
# single points nine decades above the boxes and squash every box in the figure into a
# line. The percentile whisker is stated on the y axis, and the tail it leaves out is
# reported as a number rather than drawn. NO MEAN MARKER ON THE BOXES either: within one arm
# and one bond count the mean sits at 1e4-1e11 while the box sits near 1e2, so every marker
# would land far above its own box and drag the axis with it. The mean has its own line
# figure, where being unreadable at least reads as the finding it is.
#
# MEAN AND MEDIAN ARE SEPARATE LINE FIGURES, as everywhere in 260910. THE MEDIAN IS THE
# FIGURE TO READ; the mean is here so that claim can be checked, not as an alternative. AND
# THE PER-ROTBOND MEAN PANEL IS NOT CLIPPED, where the per-atom one is: 15 rotatable-bond
# counts pool 500-1,100 molecules each, so nearly every point catches one of the conformers
# that relax to 1e8-1e13, and a 1e6 clip left the curve as disconnected fragments with most
# of it above the panel. Drawn whole it spans nine decades, sits 6-9 decades above the
# crystal ligands and has no ordering at all -- which is the honest picture of what a mean
# does to this metric, and it is the argument for the median rather than something to hide
# behind an axis limit.
RB_REF_WIN, RB_MIN_REF = 1, 12
RB_X_LABEL = "Number of rotatable bonds in ligand"
RB_XTICK = 1
# Above this a UFF relaxation has effectively failed rather than found a lower conformer
# (the converged bulk sits under ~1e3). Nothing is filtered on it; it only names the tail
# that makes the mean unreadable, so the mean figure can be read for what it is.
RB_TAIL = 1e4
RB_WHIS = (5, 95)
# Strain is floored before boxing, at the value strain_clash_ecdf_pair already floors it to.
# A rigid ligand can relax to ~0, and on a log axis a single 1e-11 at 0 rotatable bonds
# pulled the panel down through fifteen decades and flattened every box in it. Values are
# CLIPPED, not dropped, so the whisker rests on the floor and the count is unchanged.
RB_STRAIN_FLOOR = 1e-2
RB_STATS = {"median": lambda v: float(np.median(v)),
            "mean": lambda v: float(np.mean(v))}
RB_Y_LABEL = "Strain {stat}\n(kcal mol⁻¹)"
# The mean curve sits high and rises; the median curve sits low and rises. Each legend goes
# in the corner its own curves leave empty.
RB_LEGEND_LOC = {"median": "upper left", "mean": "lower right"}
# The three-arm boxes: the ECDF pair's canvas width, and the share of each count's slot left
# as white space between groups.
RB_BOX_WIDE, RB_BOX_GAP = 1.52, 0.28

# ── the eight-method versions ────────────────────────────────────────────────────
# `posecheck_<Method>.json` holds the published baselines' per-molecule strain, and has all
# along -- but those exports carry no SMILES, so rotatable bonds could not be counted from
# them. The molecules themselves are in the results bundle, so the count is recovered by
# joining the two.
#
# WHERE THE ROTATABLE BONDS COME FROM, AND WHY THE JOIN IS SAFE. The baselines' molecules
# live in `results/task2-drugdesign/<M>/samples/meta/` as the TargetDiff meta format: one
# list per test pocket of {mol, smiles, ligand_filename, pred_pos}. `export_posecheck_json.py`
# built `posecheck_<Method>.json` from THE SAME meta, walking each pocket's entries in order
# and skipping the ones whose `mol` is None -- so the two are the same molecules in the same
# order, and `_rb_load_meta` reproduces its loader exactly (base + `_part2` concatenated per
# pocket; `_gap.pt` is present in the bundle but that loader does not use it, so neither
# does this one).
#
# That is an argument, not evidence, so the join is CHECKED rather than trusted: for every
# pocket the heavy-atom sequence recomputed from the meta must equal the `n` sequence in the
# export, position by position, and a mismatch aborts. It matches 100/100 pockets for all
# five methods.
#
# MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN. The baselines were
# scored against the whole `*_rec.pdb` receptor; the local arms are read from the pocket10
# crop in their target `metrics.json`. Strain is a property of the ligand's own conformer --
# UFF relaxation under a position constraint -- so receptor scope cannot enter it, and
# measuring the same molecules both ways confirms it does not: median |relative difference|
# 0.6-1.0 %, which is the run-to-run noise `num_confs=50` already carries. Do NOT extend
# this to clashes or interactions: those are receptor-dependent and a crop cannot see an
# atom it does not contain.
#
# label, bundle folder, meta stem. DecompDiff's meta is the reference-prior run, which is
# the one export_posecheck_json.py scored. FuncBind's meta reached the bundle on 2026-09-10
# and joins cleanly (100/100 pockets, 9,992 molecules); its shard SDFs under
# `funcbind/artifacts/reproduction/crossdocked/paper_run` were tried as a substitute and
# FAILED this same check -- a different sampling run from the one PoseCheck scored -- so do
# not reach for them again if the meta ever goes missing.
RB_BASELINES = [
    ("AR",         "AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
    ("FuncBind",   "FuncBind",   "FuncBind"),
]
# Thin and dashed: with eight models on one axis hue alone is not enough, and the three
# local arms are the subject while these are context. Same channel split the nine-series
# by-atom-range figures already use.
RB_BASE_LW = 1.6
RB_BASE_DASH = {"AR": (0, (5, 2)), "Pocket2Mol": (0, (1, 1.6)),
                "DiffSBDD": (0, (6, 2, 1, 2)), "DecompDiff": (0, (9, 3)),
                "FuncBind": (0, (3, 1.4, 1, 1.4))}
# The eight-method figures draw every method through soft(): CoDE's lighter tint, the one the
# Vina per-atom family (dock-per-atom-v*) carries, and the palette colour for everyone else.
# The shared #4363D8 was the only saturated hue among nine series (changed 2026-09-13). The
# three-arm figures above keep the full colour: their key comes from arm_handles().
RB_ALL_BOX_GAP = 0.30        # of a bond count's width, left clear between neighbouring groups
# The eight-method box panel is wider and taller than the house figure -- eight boxes have
# to fit inside each of thirteen counts -- and the share panels are a 3x3 block of small ones.
RB_BOX_WIDE_1, RB_BOX_TALL = 1.72, 1.22
RB_DIST_COLS, RB_DIST_WIDE, RB_DIST_TALL = 3, 1.34, 0.66
RB_BOX_YLIM = (1e0, 1e5)
# THE AXIS STOPS AT 12 ROTATABLE BONDS, because each INDIVIDUAL count past it is thin and
# stretching the axis to the last molecule anyone made (25, one VoxBind ligand) spent two
# thirds of the width on a handful of boxes.
#
# THE CUMULATIVE TAIL IS NOT NEGLIGIBLE, THOUGH, and the figure must not imply it is: 5.0 %
# of VoxBind's ligands, 3.1 % of DecompDiff's and 2.8 % of DiffSBDD's have more than 12
# rotatable bonds (Pocket2Mol is the outlier at 0.0 %). So every share panel PRINTS its own
# excluded percentage rather than letting the cap pass silently. Raise RB_X_MAX to see them.
RB_X_MAX = 12
# Above this many counts the axis is split over two rows so the boxes stay wide enough to
# read; at RB_X_MAX = 12 it is one row. Raise RB_X_MAX and the split comes back on its own.
RB_SPLIT_ABOVE = 14
# Boxes are drawn further into the tail than the sibling figures' MIN_N=25 allows, so every
# method covers its own full range instead of being clipped to the narrowest one. Ten
# molecules is the floor for a box to carry quartiles at all; the share figure is what tells
# the reader which end of the axis is thin. The crystal ligands are a LINE in the box
# figure, not a ninth box series, so this floor never applies to them -- theirs is
# RB_MIN_REF over the +-RB_REF_WIN window, and it is why their line stops at 9.
RB_GRID_MIN_N = 10
# Floor for the log-scaled share panels; below this a bond count is empty, not rare.
RB_DIST_FLOOR = 0.05
RB_KEY_H = 0.92              # inches of key strip under the line panels; see _rb_lines()
# The summary table bins the axis; the figures keep every count. Half-open, so these are
# {0}, {1,2}, {3,4}, {5,6}, {7,8,9}, {10+}.
RB_BIN_EDGES = [0, 1, 3, 5, 7, 10, 10 ** 6]
RB_BIN_LABELS = ["0", "1–2", "3–4", "5–6", "7–9", "10+"]

# The three arms this section runs locally, addressed by their stable KEYS -- their LABELS
# moved ("VoxBind + Ours" -> "Ours" -> "CoDE") and may move again. They are picked BY KEY
# rather than by walking ARMS: that list grew from three arms to eight on 2026-09-10 when
# the five published baselines were added to it, pointing at `exps/baselines_pose/<m>`,
# which another job is still filling. Those trees carry PoseBusters but no `posecheck` block
# yet, so walking ARMS here would load three methods' worth of rows with `s=None` and
# silently empty every range these builders compute.
#
# WHEN `exps/baselines_pose/` IS COMPLETE the bundle join becomes unnecessary: those trees
# carry SMILES and will carry strain, so the three-arm figures above will cover every method
# through the shared loader alone and these two can be retired.
RB_LOCAL_KEYS = ("targetdiff", "vanilla", "ours_v1")
RB_LOCAL_LABELS = []         # filled by the loaders from ARMS, in RB_LOCAL_KEYS order
RB_RELABEL = {"ours_v1": "CoDE"}
# THE ROW ORDER OF 260827/table_drug_design.tex, which is canonical for this section, except
# that VoxBind is pulled down next to CoDE so the model our arm modifies sits immediately
# before it and the two read as a pair. Legends, panels and exports all read from this, so a
# method cannot sit in one order in the legend and another in the share panels.
RB_ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff", "FuncBind",
            "VoxBind", "CoDE"]

# The per-ATOM eight-method figure. No join is needed there at all: unlike the rotatable-bond
# axis, which had to go back to the results bundle for SMILES, heavy-atom count is already
# in `posecheck_<Method>.json`. Only the p79 pockets are kept, by the export's own index.
RB_ATOM_BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind"]
RB_X_LO, RB_X_HI = 5, 44
# DecompDiff PUTS EXACTLY ONE SIZE IN EACH POCKET, AND IT IS THE CRYSTAL LIGAND'S -- checked
# on every run, not assumed. Its reference prior fixes the atom budget per pocket, so on
# that axis DecompDiff is not a distribution but a comb over the 79 reference sizes, with
# exact zeros at 7, 24, 30, 34, 36, 39-41, 43-44 atoms. So EVERY model curve pools a +-1
# atom window: drawn per exact count DecompDiff's line would break ten times inside the axis
# on molecules that were never going to exist. The window is one atom wide, the same
# treatment for all eight methods, and it moves a median of a continuous quantity almost
# nowhere at these counts. The one hole it does not fill -- 40 atoms, where no reference
# sits within +-1 -- is left as a break rather than widened away.
RB_WIN = 1
RB_MEAN_CLIP = 1e6           # the per-atom mean panel's top, as in the three-arm figure

_RB_CHEM = None
_RB_CACHE = {}


def _rb_rdkit():
    """RDKit, imported on first use rather than at the top of draw.py. The chemistry toolkit
    costs seconds to import and is the only thing in this file the other figure families do
    not need, so a run that draws none of these four never pays for it."""
    global _RB_CHEM
    if _RB_CHEM is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
        _RB_CHEM = (Chem, rdMolDescriptors)
    return _RB_CHEM


def _rb_rot_bonds(smiles):
    """RDKit's strict rotatable-bond count, or None if the SMILES will not parse. Cached:
    the arms carry ~60k molecules and many repeat."""
    if not smiles:
        return None
    if smiles not in _RB_CACHE:
        chem, desc = _rb_rdkit()
        mol = chem.MolFromSmiles(smiles)
        _RB_CACHE[smiles] = None if mol is None else int(desc.CalcNumRotatableBonds(mol))
    return _RB_CACHE[smiles]


def _rb_attach(rows):
    """Add `rb` to each row in place, and report how many rows could not get one."""
    bad = 0
    for r in rows:
        r["rb"] = _rb_rot_bonds(r.get("smi"))
        bad += r["rb"] is None
    return bad


def _rb_reference_curve(per, xs, f):
    """reference_curve over the NARROW +-RB_REF_WIN window -- see the banner. None where the
    window is too thin to mean anything, which leaves a gap in the line rather than an
    invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= RB_REF_WIN for v in vals]
        out.append(f(p) if len(p) >= RB_MIN_REF else None)
    return out


def _rb_ref_window_n(per, xs):
    """The reference's n is the WINDOWED pool, not the exact-count one: the plotted value
    comes from +-RB_REF_WIN, so an n beside it that counted only the exact bond count would
    read as 4 ligands supporting a point that 16 produced."""
    return [sum(len(v) for x2, v in per.items() if abs(x2 - x) <= RB_REF_WIN) for x in xs]


def _rb_cell(v):
    return "" if v is None else round(v, 3)


def _rb_tail_share(per):
    """% of an arm's molecules whose relaxation ran past RB_TAIL -- the tail that decides its
    mean curve, and the reason the mean curve is not a location statistic."""
    vals = [v for vs in per.values() for v in vs]
    return round(100 * sum(v > RB_TAIL for v in vals) / len(vals), 2) if vals else None


def _rb_tail_share_rows(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > RB_TAIL for x in v) / len(v), 2) if v else None


# ── the three local arms, per rotatable bond ─────────────────────────────────────
def _rb_local():
    """(per-arm {rb: [strain]}, the crystal ligands' own, the (name, arms) variants, dropped).

    THE ARMS ARE THE ONES THAT CARRY STRAIN, not all of ARMS. That list grew to eight on
    2026-09-10 and the five staged baselines hold PoseBusters only, so `variants()` with no
    field would put five all-None curves in `all` and leave x_range intersecting an empty
    set. `variants("s", data)` is the test the per-atom sibling already applies."""
    data, p79_rows, refrows = pose_data()
    bad = sum(_rb_attach(rows) for rows in p79_rows.values()) + _rb_attach(refrows)
    per_arm = {key: by_size(p79_rows[key], "s", key="rb") for _, key, _ in ARMS}
    ref_per = by_size(refrows, "s", key="rb")
    return per_arm, ref_per, variants("s", data), bad


def _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per):
    """One statistic, one panel, one file — the per-atom builder's layout with this axis."""
    f = RB_STATS[stat]
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per_arm[key], xs, f)
        ax.plot(xs, y, color=color(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")

    ax.set_yscale("log")
    furniture(ax, ylabel=RB_Y_LABEL.format(stat=stat), xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.35, xs[-1] + 0.35), xloc=RB_XTICK)
    legend(ax, arm_handles(arms), loc=RB_LEGEND_LOC[stat], fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_per_rotbond_{stat}_{variant}")


def _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per):
    """The distribution at each rotatable-bond count, one box per arm, dodged within the
    count. Wider canvas than the line panels (the ECDF pair's width, already in the house)
    because this draws 15 counts x 2-3 arms of boxes on one axis."""
    fig, ax = plt.subplots(figsize=(FIG_W * RB_BOX_WIDE, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Dodge: n boxes centred on the count, leaving RB_BOX_GAP of the slot as white space so
    # neighbouring counts stay visually separate groups.
    n = len(arms)
    slot = (1.0 - RB_BOX_GAP) / n
    for i, (lab, key, _) in enumerate(arms):
        offs = (i - (n - 1) / 2) * slot
        vals = [np.clip(per_arm[key].get(x, []), RB_STRAIN_FLOOR, None) for x in xs]
        col = color(lab)
        bp = ax.boxplot(vals, positions=[x + offs for x in xs], widths=slot * 0.86,
                        whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                        manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=1.05)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=col, linewidth=1.05)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=1.5, solid_capstyle="butt")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]), color=REF_COLOR,
            lw=REF_LW, ls=DASH, zorder=6, dash_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(bottom=RB_STRAIN_FLOOR)
    furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers",
              xlabel=RB_X_LABEL, xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    legend(ax, arm_handles(arms), loc="upper left", fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_box_per_rotbond_{variant}")


def _rb_local_csv(out, xs, arms, per_arm, ref_per):
    """The curves themselves plus the per-count molecule counts, so a reader can check a
    figure without re-running it — and so the rotatable-bond DISTRIBUTION is on the record.
    It is not drawn as a strip (the per-atom figure this mirrors has no strip either), but
    the arms differ in it, and any claim about a curve has to be read against `n`."""
    rows, summary = [], {}
    for lab, key, _ in arms:
        med = model_curve(per_arm[key], xs, RB_STATS["median"])
        mean = model_curve(per_arm[key], xs, RB_STATS["mean"])
        summary[lab] = {"n_scored": sum(len(v) for v in per_arm[key].values()),
                        "strain_gt_1e4": _rb_tail_share(per_arm[key]), "median": med}
        for i, x in enumerate(xs):
            rows.append([lab, x, len(per_arm[key].get(x, ())),
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    summary[REF_LABEL] = {"n_scored": sum(len(v) for v in ref_per.values()),
                          "strain_gt_1e4": _rb_tail_share(ref_per), "median": med}
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, _rb_ref_window_n(ref_per, xs)[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_rotbond",
              ["arm", "rotatable_bonds", "n", "strain_median", "strain_mean"], rows)
    return summary


def _rb_local_log(xs, summary, ranges, bad):
    print(f"  {len(P79)}-pocket set · "
          + " · ".join(f"{v} x {r[0]}-{r[-1]} rotatable bonds" for v, r in ranges.items())
          + f" (counts where every drawn arm has ≥{MIN_N} molecules)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")
    header = [0, 2, 4, 6, 8, 10, 12]
    print("  strain MEDIAN at a given rotatable-bond count, and the tail behind the mean")
    print(f"  {'arm':16s} {'mols':>7s} " + " ".join(f"{'rb=' + str(h):>7s}" for h in header)
          + f" {'>1e4':>7s}")
    for lab, d in summary.items():
        cells = []
        for h in header:
            v = d["median"][xs.index(h)] if h in xs else None
            cells.append(f"{v:7.1f}" if v is not None else f"{'—':>7s}")
        print(f"  {lab:16s} {d['n_scored']:7d} " + " ".join(cells)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print("  The mean figure is drawn unclipped and spans ~9 decades: the tail above is "
          "what puts it there. Read the median.")


@figure("fig-posecheck-strain-per-rotbond", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_per_rotbond(out):
    """Strain per rotatable bond — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    ranges, all_arms = {}, []
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        ranges[variant] = xs
        all_arms = arms
        for stat in ("median", "mean"):
            _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per)
    summary = _rb_local_csv(out, ranges["core"], all_arms, per_arm, ref_per)
    _rb_local_log(ranges["core"], summary, ranges, bad)


@figure("fig-posecheck-strain-box-per-rotbond", folder="fig-posecheck/strain-energy",
        needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_box_per_rotbond(out):
    """The strain DISTRIBUTION at each rotatable-bond count, as boxes — core only.

    No `all`: see core_only(). The eight-method view is strain_box_per_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per)
        print(f"  {variant:4s} {len(arms)} arms x {xs[0]}-{xs[-1]} rotatable bonds "
              f"({RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers, no fliers)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")


# ── all eight methods ────────────────────────────────────────────────────────────
def _rb_bundle():
    """The published baselines' sample bundle, which is where their molecules are."""
    p = REPO / "results" / "task2-drugdesign"
    if not p.is_dir():
        raise FileNotFoundError(f"missing the baseline sample bundle {p}")
    return p


def _rb_load_meta(folder, stem):
    """export_posecheck_json.py's loader, reproduced exactly: base, then `_part2`
    concatenated PER POCKET. `_gap.pt` ships in the bundle and that loader ignores it, so
    including it here would shift every molecule after the split."""
    import torch                    # local, like RDKit above: nothing else here needs it
    d = _rb_bundle() / folder / "samples" / "meta"
    meta = torch.load(d / f"{stem}.pt", weights_only=False)
    part2 = d / f"{stem}_part2.pt"
    if part2.exists():
        meta = [a + b for a, b in zip(meta, torch.load(part2, weights_only=False))]
    return meta


def _rb_baseline_rows(label, folder, stem, keep):
    """Rows shaped like the shared loader's, for the p79 pockets, with the join checked.

    THE CHECK RUNS OVER EVERY POCKET THE EXPORT HOLDS -- all 100 -- while only the p79 ones
    become rows. Checking just the 79 that are drawn would leave the other 21 as evidence
    nobody looked at, and they cost nothing: the meta is already in memory and the export
    already carries them."""
    meta = _rb_load_meta(folder, stem)
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    by_pocket = collections.defaultdict(list)
    for m in export["molecules"]:
        by_pocket[m["p"]].append(m)

    rows, unparsed, checked = [], 0, 0
    for p in sorted(by_pocket):
        entries = [e for e in meta[p] if e.get("mol") is not None]
        scored = by_pocket[p]
        mine = [int(e["mol"].GetNumAtoms()) for e in entries[:len(scored)]]
        if mine != [m["n"] for m in scored]:
            raise SystemExit(f"{label}: meta/export heavy-atom sequence differs at pocket "
                             f"{p} -- the join is not valid, refusing to guess")
        checked += 1
        if p not in keep:
            continue
        for e, m in zip(entries, scored):
            rb = _rb_rot_bonds(e.get("smiles"))
            unparsed += rb is None
            rows.append({"n": m["n"], "s": m["s"], "c": m["c"], "rb": rb})
    return rows, unparsed, checked


def _rb_locals_from_arms():
    """(label -> rows) for the three arms we run locally, picked BY KEY out of ARMS, plus
    the crystal-ligand rows. See the banner for why it is by key and not a walk."""
    _, p79_rows, refrows = pose_data()
    by_key = {key: (lab, root) for lab, key, root in ARMS}
    series, local = {}, []
    for key in RB_LOCAL_KEYS:
        if key not in by_key:
            raise SystemExit(f"ARMS no longer defines {key!r}")
        lab = RB_RELABEL.get(key, by_key[key][0])
        series[lab] = p79_rows[key]
        local.append(lab)
    RB_LOCAL_LABELS[:] = local
    return series, refrows


def _rb_load_all():
    """(label -> rows) for every drawn method on the rotatable-bond axis, plus the crystal
    ligands and the join-check counts."""
    series, refrows = _rb_locals_from_arms()
    for rows in series.values():
        _rb_attach(rows)
    _rb_attach(refrows)

    keep = {int(t.split("_")[1]) for t in P79}
    checks = []
    for lab, folder, stem in RB_BASELINES:
        rows, bad, checked = _rb_baseline_rows(lab, folder, stem, keep)
        series[lab] = rows
        checks.append((lab, len(rows), bad, checked))
    return series, refrows, checks


def _rb_order(labels):
    """The drug-design table's row order, ours last. Anything the table does not name is a
    bug rather than something to append quietly, so it raises."""
    unknown = [l for l in labels if l not in RB_ORDER]
    if unknown:
        raise SystemExit(f"not in table_drug_design.tex's order: {unknown}")
    return [l for l in RB_ORDER if l in labels]


def _rb_style_of(label):
    return (MODEL_LW, "-") if label in RB_LOCAL_LABELS else (RB_BASE_LW, RB_BASE_DASH[label])

def _rb_panel_order(labels):
    """Same order as `_rb_order`, with the crystal ligands FIRST -- they are the table's
    first row and the thing every other panel is read against."""
    return [REF_LABEL] + _rb_order(labels)


def _rb_handles(labels, solid=False):
    """`solid` for the box figure: nothing in it is a dashed line, so a dashed swatch in the
    key advertises an encoding the panel does not use. The crystal ligands keep their dash
    either way -- there they really are a dashed line."""
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH, label=REF_LABEL)]
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        h.append(Line2D([], [], color=soft(lab), lw=MODEL_LW if solid else lw,
                        ls="-" if solid else ls, label=lab))
    return h


def _rb_line_curve(per_lab, xs, f):
    """The statistic at each exact bond count, null where that count holds fewer than MIN_N
    molecules. THE AXIS IS FIXED AT 0-RB_X_MAX rather than cut back to the counts every
    method can answer: past nine bonds the methods thin out at very different rates --
    Pocket2Mol has 40 ligands at nine and single figures at eleven, CoDE and VoxBind still
    have hundreds -- and the old rule let the emptiest method decide where everyone's line
    stopped. Now each line simply ends where its own method ran out, which is the more
    informative thing to show, and the crystal ligands' dashed line ends earlier still: 79
    of them cannot fill a window at twelve bonds."""
    return [f(per_lab[x]) if len(per_lab.get(x, ())) >= MIN_N else None for x in xs]


def _rb_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, and the key in a strip of its own beneath it.

    NINE SERIES DO NOT LEAVE A CORNER FREE. Inside the axes this key covered FuncBind's
    spike at six bonds and everything above ~500 kcal/mol on the left half -- the part of
    the figure that carries the finding. Under the panel it covers nothing, and it is the
    same 3x3 block the box figure's key is."""
    f = RB_STATS[stat]
    fig, (ax, key) = plt.subplots(2, 1, figsize=(FIG_W, PANEL_H + RB_KEY_H), dpi=220,
                                  gridspec_kw={"height_ratios": [PANEL_H, RB_KEY_H]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key.axis("off")
    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        ax.plot(xs, _rb_line_curve(per[lab], xs, f), color=soft(lab), lw=lw, ls=ls,
                zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    legend(key, _rb_handles(labels), loc="center", ncol=3, fontsize=11)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_rotbond_all_methods_{stat}")


def _rb_axis(labels, refrows, series):
    """The shared x range and per-method rows both figures are drawn over, so the boxes and
    the share panels cannot end up covering different molecules."""
    panels = _rb_panel_order(labels)
    rows_by_lab = {lab: series[lab] for lab in labels}
    rows_by_lab[REF_LABEL] = refrows
    per = {lab: by_size(rows_by_lab[lab], "s", key="rb") for lab in panels}
    dist = {lab: collections.Counter(r["rb"] for r in rows_by_lab[lab]
                                     if r["rb"] is not None) for lab in panels}
    # 0 to RB_X_MAX, the same span the line figures get, so the three are read against one
    # axis BY CONSTRUCTION rather than by today's data happening to agree. Boxes still
    # appear only where a method has RB_GRID_MIN_N molecules, so pinning the axis adds no
    # box; it only stops the axis shrinking when the tail thins.
    xs = list(range(0, RB_X_MAX + 1))
    beyond = {lab: 100 * sum(n for x, n in dist[lab].items() if x > RB_X_MAX)
              / max(sum(dist[lab].values()), 1) for lab in panels}
    return panels, per, dist, xs, beyond


def _rb_all_boxes(out, series, refrows, labels):
    """Strain against rotatable bonds, every method on one axis, dodged inside each count.

    The paper's Fig. 12 shape: one panel of boxplots per number of rotatable bonds, all
    methods together. (The paper packs 151 boxes into one row and colours each box by its
    median strain, spending colour on the value; here colour stays the method key it is in
    every other 260910 figure, and the legend carries it.)

    The crystal ligands stay the dashed grey line rather than becoming a ninth box series:
    79 of them over thirteen bond counts cannot fill a box at every count, and as a line
    they are the same ruler here that they are in every sibling figure."""
    panels, per, dist, xs, beyond = _rb_axis(labels, refrows, series)
    if len(xs) > RB_SPLIT_ABOVE:
        split = len(xs) - len(xs) // 2                 # low row takes the extra count
        bands = [xs[:split], xs[split:]]
    else:
        bands = [xs]
    ref_line = {x: v for x, v in
                zip(xs, _rb_reference_curve(per[REF_LABEL], xs, RB_STATS["median"]))}

    fig, axes = plt.subplots(len(bands), 1,
                             figsize=(FIG_W * RB_BOX_WIDE_1,
                                      PANEL_H * RB_BOX_TALL * len(bands)),
                             dpi=220, squeeze=False)
    fig.patch.set_facecolor("white")
    n = len(labels)
    slot = (1.0 - RB_ALL_BOX_GAP) / n
    for band, ax in zip(bands, axes.ravel()):
        ax.set_facecolor("white")
        for i, lab in enumerate(labels):
            offs = (i - (n - 1) / 2) * slot
            drawn = [x for x in band if x in per[lab] and len(per[lab][x]) >= RB_GRID_MIN_N]
            if not drawn:
                continue
            col = soft(lab)
            bp = ax.boxplot([np.clip(per[lab][x], RB_STRAIN_FLOOR, None) for x in drawn],
                            positions=[x + offs for x in drawn], widths=slot * 0.88,
                            whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                            manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=0.8)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=col, linewidth=0.8)
            for med in bp["medians"]:
                med.set(color=INK, linewidth=1.1, solid_capstyle="butt")
        ax.plot(band, [ref_line.get(x) for x in band], color=REF_COLOR, lw=REF_LW,
                ls=DASH, zorder=6, dash_capstyle="round")
        ax.set_yscale("log")
        # A FIXED FIVE DECADES, AND THE TAIL IS ALLOWED TO RUN OFF THE TOP. FuncBind's
        # 95th percentile at 6 rotatable bonds reaches ~1e11; autoscaling to it stretched
        # the panel over fourteen decades and pressed every box into the bottom fifth. The
        # window is pinned so the boxes -- which all sit between 1e0 and 1e5 -- stay legible
        # across every rebuild, and the whiskers that leave the top simply leave it. How
        # much tail each method carries is reported as `strain_gt_1e4`, not drawn.
        ax.set_ylim(*RB_BOX_YLIM)
        furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct "
                             "whiskers",
                  xlabel=RB_X_LABEL, xlim=(band[0] - 0.62, band[-1] + 0.62), xloc=1)
        ax.set_xticks(band)
    # Nine entries -- the crystal ligands plus eight methods -- in three columns, so the key
    # is a 3x3 block rather than one long strip across the top of the panel. Lower right:
    # the boxes climb left-to-right, so the empty corner is under the high bond counts,
    # where only the lower whiskers reach.
    legend(axes.ravel()[0], _rb_handles(labels, solid=True), loc="lower right",
           fontsize=10, ncol=3)
    fit(fig, pad=0.5)
    save(fig, out, "strain_box_per_rotbond_all_methods")


def _rb_distribution(out, series, refrows, labels):
    """Where each method puts its ligands on the same axis the boxes use -- its own share
    at each rotatable-bond count, one panel per method in a 3x3 block.

    ITS OWN FIGURE, not a strip under the boxes. The two answer different questions and are
    read at different times: the boxes compare methods at a bond count, this compares the
    bond counts a method produces. Sharing a canvas forced one to be a third the height of
    the other, and a nine-panel block does not fit under a box panel at any useful size.

    LOG y. The share spans two decades inside 0-12 bonds -- Pocket2Mol puts 33 % at one bond
    and 0.1 % at twelve -- and on a linear axis everything under ~2 % is a flat line on the
    floor."""
    panels, per, dist, xs, beyond = _rb_axis(labels, refrows, series)
    ncol = RB_DIST_COLS
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(FIG_W * RB_DIST_WIDE, PANEL_H * RB_DIST_TALL * nrow),
                             dpi=220, sharex=True, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for i, lab in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor("white")
        col = REF_COLOR if lab == REF_LABEL else soft(lab)
        total = sum(dist[lab].values())
        pct = [100 * dist[lab].get(x, 0) / total for x in xs]
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, RB_DIST_FLOOR, step="mid", color=col, alpha=DIST_FILL,
                        lw=0, zorder=2)
        ax.set_yscale("log")
        # The x name goes under the BOTTOM ROW only; repeated under all nine it is wider
        # than a panel and the copies overprint each other.
        furniture(ax, ylabel="% of ligands" if i % ncol == 0 else None,
                  xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=2)
        ax.set_ylim(bottom=RB_DIST_FLOOR)
        ax.set_title(lab, fontsize=12, color=INK, loc="left", pad=4)
        ax.text(0.97, 0.07, f">{RB_X_MAX} bonds: {beyond[lab]:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=AXIS, zorder=6)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    # ONE x name for the whole block, centred under the bottom row. sharex already leaves
    # the tick labels on that row alone; three copies of the name did not fit -- each is
    # wider than a panel -- and overprinted each other.
    fig.supxlabel(RB_X_LABEL, fontsize=15.5, color=INK)
    fit(fig, pad=0.5, h_pad=0.9)
    save(fig, out, "rotbond_distribution_all_methods")


def _rb_bin_of(rb):
    return min(int(np.searchsorted(RB_BIN_EDGES, rb, side="right")) - 1,
               len(RB_BIN_LABELS) - 1)


def _rb_all_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels:
        med = _rb_line_curve(per[lab], xs, RB_STATS["median"])
        mean = _rb_line_curve(per[lab], xs, RB_STATS["mean"])
        for i, x in enumerate(xs):
            n = len(per[lab].get(x, ()))
            # the models are drawn per exact count, so their window IS that count
            rows.append([lab, x, n, n, _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    win = _rb_ref_window_n(ref_per, xs)
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, len(ref_per.get(x, ())), win[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_rotbond_all_methods",
              ["arm", "rotatable_bonds", "n", "n_window", "strain_median", "strain_mean"],
              rows)


@figure("fig-posecheck-strain-rotbond-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_all_methods(out):
    """Strain against rotatable bonds, all eight methods: lines, boxes and the share block."""
    use_style()
    series, refrows, checks = _rb_load_all()
    labels = _rb_order(series)
    per = {lab: by_size(series[lab], "s", key="rb") for lab in labels}
    ref_per = by_size(refrows, "s", key="rb")

    # 0 to RB_X_MAX, always -- the same span the box and share figures cover, so the three
    # are read against one axis. Where a method (or the reference) is too thin at a count,
    # its own line stops; the axis does not.
    xs = list(range(0, RB_X_MAX + 1))
    for s in ("median", "mean"):
        _rb_lines(out, xs, per, ref_per, labels, s)
    _rb_all_boxes(out, series, refrows, labels)
    _rb_distribution(out, series, refrows, labels)
    _rb_all_csv(out, xs, per, ref_per, labels)

    for lab, n_rows, bad, checked in checks:
        print(f"  {lab:12s} join checked on {checked} pockets · {n_rows:,} p79 molecules"
              + (f" · {bad} SMILES would not parse" if bad else ""))
    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {xs[0]}-{xs[-1]} rotatable bonds "
          f"(each line drawn where its own method has ≥{MIN_N} molecules)")
    for lab in labels:
        drawn = [x for x in xs if len(per[lab].get(x, ())) >= MIN_N]
        # A break INSIDE a method's span is reported rather than collapsed into the
        # endpoints -- "0-12" over a line with a hole in it would be a false summary.
        gaps = [x for x in range(drawn[0], drawn[-1] + 1) if x not in drawn] if drawn else []
        print(f"    {lab:12s} line drawn {drawn[0]}-{drawn[-1]}"
              + (f", broken at {gaps}" if gaps else "") if drawn else
              f"    {lab:12s} nowhere thick enough to draw")
    ref_drawn = [x for x, v in zip(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]))
                 if v is not None]
    print(f"    {REF_LABEL:12s} line drawn {ref_drawn[0]}-{ref_drawn[-1]} "
          f"(±{RB_REF_WIN} window, ≥{RB_MIN_REF} ligands)")
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{b:>8s}" for b in RB_BIN_LABELS)
          + f" {'>1e4':>7s}   (strain median by bin)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        cells = [[r["s"] for r in rows if r["s"] is not None and r["rb"] is not None
                  and _rb_bin_of(r["rb"]) == i] for i in range(len(RB_BIN_LABELS))]
        floor = RB_MIN_REF if lab == REF_LABEL else 1
        vals = [f"{np.median(v):8.1f}" if len(v) >= floor else f"{'—':>8s}" for v in cells]
        n_scored = sum(len(v) for v in (ref_per if lab == REF_LABEL else per[lab]).values())
        print(f"  {lab:16s} {n_scored:6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(vals) + f" {_rb_tail_share_rows(rows):6.2f}%")


# ── all eight methods, against ligand SIZE ───────────────────────────────────────
def _rb_atom_baseline_rows(label, keep):
    """The export's p79 molecules, as shared-loader-shaped rows."""
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    rows = [{"n": m["n"], "s": m["s"], "c": m["c"]}
            for m in export["molecules"] if m["p"] in keep]
    return rows, len(export["molecules"])


def _rb_atom_load_all():
    series, refrows = _rb_locals_from_arms()
    keep = {int(t.split("_")[1]) for t in P79}
    for lab in RB_ATOM_BASELINES:
        series[lab], _ = _rb_atom_baseline_rows(lab, keep)
    return series, refrows


def _rb_check_decompdiff():
    """The banner's claim, re-checked on every run: one size per pocket, and it is the
    crystal ligand's. If DecompDiff's export is ever replaced by the non-reference-prior
    run this stops holding, and the +-1 window stops being something this figure needs."""
    export = json.load(open(legacy("fig-posecheck", "posecheck_DecompDiff.json")))
    ref = {}
    for t in P79:
        r = rows_of(os.path.join(REF_ROOT, t), reference=True)
        if r:
            ref[int(t.split("_")[1])] = r[0]["n"]
    sizes = collections.defaultdict(set)
    for m in export["molecules"]:
        if m["p"] in ref:
            sizes[m["p"]].add(m["n"])
    one = sum(len(v) == 1 for v in sizes.values())
    same = sum(len(v) == 1 and next(iter(v)) == ref[p] for p, v in sizes.items())
    return one, same, len(sizes)


def _rb_pool(per_lab, a):
    """The molecules behind the point at `a`: its +-RB_WIN neighbours, CLIPPED TO THE AXIS.

    The clip is not cosmetic. DecompDiff has no molecule at 43 or 44 heavy atoms and 125 at
    45+, so an unclipped window put a point at 44 computed entirely from molecules the same
    file reports as beyond the axis -- a plotted value representing nothing near where it
    was plotted. Inside the axis the window still interpolates across its comb, which is
    what it is for; at the edges it no longer extrapolates from outside."""
    return [v for n, vals in per_lab.items()
            if abs(n - a) <= RB_WIN and RB_X_LO <= n <= RB_X_HI for v in vals]


def _rb_curve(per_lab, xs, f):
    """The statistic over that window, None where it is too thin to mean anything -- which
    leaves a gap rather than an invented value, the same rule reference_curve applies to the
    crystal ligands."""
    return [f(p) if len(p) >= MIN_N else None
            for p in (_rb_pool(per_lab, a) for a in xs)]


def _rb_atom_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, NO KEY.

    The key was dropped on request (2026-09-13). It used to be a 3x3 block in a strip of its
    own beneath the panel -- nine series leave no corner of the axes free -- and the strip
    went with it, so the panel is the house single-panel size. The colours and dashes are the
    ones every sibling figure keys (strain_rotbond_all_methods_*), which is where to read them.

    THE MEAN PANEL IS CLIPPED TO THE BULK and the points that leaves off are NAMED in the
    run log: 4-13 % of molecules fail UFF relaxation and land between 1e4 and 1e13, so one
    of them at a thin count carries that count's mean four decades up and an unclipped axis
    spends fourteen decades on it.

    THE REFERENCE KEEPS THE HEADER'S +-4 WINDOW HERE, not the narrow one the rotatable-bond
    figures use: this axis runs 5-44 atoms, which is what that window was set for."""
    f = RB_STATS[stat]
    clip = RB_MEAN_CLIP if stat == "mean" else None
    # DRAWN IN THE STRAIN ECDF'S HOUSE (2026-09-13), so the two strain figures sit side by side
    # as a pair: its canvas (PCSZ_ECDF_SIZE), its rc (12 pt type, black ink, 170 dpi cropped to
    # the ink), its axis furniture and its label size. Read at call time -- the ECDF family is
    # another part of this file. The y name is one line, as the ECDF's are.
    with plt.rc_context(PCSZ_RC):
        fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
        fig.patch.set_facecolor(PCSZ_BG)
        _pcsz_style(ax)
        ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
                ls=DASH, zorder=4, dash_capstyle="round")
        dropped = []
        for lab in labels:
            y = _rb_curve(per[lab], xs, f)
            if clip:
                over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
                if over:
                    dropped.append((lab, over))
            lw, ls = _rb_style_of(lab)
            ax.plot(xs, y, color=soft(lab), lw=lw, ls=ls, zorder=5, solid_capstyle="round")
        ax.set_yscale("log")
        if clip:
            ax.set_ylim(top=clip)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
        ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
        ax.set_ylabel(f"Strain {stat} (kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS)
        fig.tight_layout(pad=0.5)
        save(fig, out, f"strain_per_atom_all_methods_{stat}")
    return dropped


def _rb_atom_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels + [REF_LABEL]:
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        mean = (reference_curve(p, xs, RB_STATS["mean"]) if lab == REF_LABEL
                else _rb_curve(p, xs, RB_STATS["mean"]))
        win = ([sum(len(v) for x, v in p.items() if abs(x - a) <= REF_WIN) for a in xs]
               if lab == REF_LABEL else [len(_rb_pool(p, a)) for a in xs])
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ())), win[i],
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "n_window", "strain_median", "strain_mean"], rows)
    return rows


@figure("fig-posecheck-strain-per-atom-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "metrics.json (posecheck.strain)"))
def draw_posecheck_strain_per_atom_all_methods(out):
    """Strain against ligand size, all eight methods — mean and median."""
    use_style()
    series, refrows = _rb_atom_load_all()
    labels = _rb_order(series)
    dd = _rb_check_decompdiff()
    per = {lab: by_size(series[lab], "s") for lab in labels}
    ref_per = by_size(refrows, "s")
    xs = list(range(RB_X_LO, RB_X_HI + 1))
    drops = {s: _rb_atom_lines(out, xs, per, ref_per, labels, s) for s in ("median", "mean")}
    _rb_atom_csv(out, xs, per, ref_per, labels)

    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, "
          f"drawn where it pools ≥{MIN_N} molecules")
    print(f"  DecompDiff: {dd[0]}/{dd[2]} pockets hold a single heavy-atom count and "
          f"{dd[1]}/{dd[2]} of them equal the crystal ligand's — it is size-matched to the "
          f"reference by construction")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>7d}" for a in heads)
          + f" {'drawn':>9s} {'>44 at':>7s} {'>1e4':>7s}   (strain median at n atoms)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:7.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>7s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        gaps = [a for a in range(drawn[0], drawn[-1] + 1) if a not in drawn] if drawn else []
        span = (f"{drawn[0]}-{drawn[-1]}" + (f"*{len(gaps)}" if gaps else "")) if drawn \
            else "—"
        beyond = round(100 * sum(r["n"] > RB_X_HI for r in rows) / max(len(rows), 1), 2)
        tail = _rb_tail_share_rows(rows)
        print(f"  {lab:16s} {sum(len(v) for v in p.values()):6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(cells) + f" {span:>9s} {beyond:6.1f}% "
              + (f"{tail:6.2f}%" if tail is not None else f"{'—':>7s}"))
        if gaps:
            print(f"    * {lab}'s line breaks inside its span at {gaps} heavy atoms — too "
                  f"few molecules in the window there, not a drawing error")
    for lab, over in drops.get("mean", []):
        pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
        print(f"    strain_per_atom_all_methods_mean · off-panel above {RB_MEAN_CLIP:.0e}: "
              f"{lab:12s} {len(over):2d} of {len(xs)}: {pts}")


# ── all eight methods, the paper's small-multiples form ──────────────────────────
# VoxBind Fig. 12 (arXiv 2405.03961, appendix): ONE PANEL PER METHOD, rotatable bonds on x,
# Tukey boxes WITH fliers, and each box FILLED BY ITS OWN MEDIAN on a diverging colormap. The
# dodged figure above answers "which method is lower at this count"; this one answers "how
# does each method's strain climb with torsional freedom", and the colour lets the reader
# compare levels across panels without lining up y values.
#
# WHAT IS KEPT FROM THE PAPER, WHAT IS NOT. Kept: the grid, x = 0-7, Tukey whiskers (1.5 IQR)
# with fliers, log y, a LINEAR colour norm = median. Changed: nine panels (the crystal ligands
# plus eight methods) in 3x3 rather than 2x4, the house fonts and furniture, and the colormap.
#
# THE COLORMAP IS BUILT FROM THE HOUSE PALETTE, not coolwarm: a light version of CoDE's blue at
# the low end running straight to a light version of VoxBind's sand at the high end, so the
# figure reads in the same two hues as every other 260910 figure.
#
# THE NORM IS A FIXED 0-800 kcal/mol, ticked every 200. Over the eight methods' 64 boxes the
# median is 136 and the 95th percentile 694; only FuncBind at seven and six bonds (982, 5,638)
# lie above 800, and they saturate onto the top colour with the colorbar's arrow saying so
# (the run log names every saturated box). The price of
# a linear norm is that medians under ~100 -- every crystal-ligand box, and VoxBind/CoDE at low
# counts -- share the darkest blues; their differences are read off the y axis, as in the paper.
#
# THE AXIS STOPS AT 7 AS THE PAPER'S DOES, which leaves a real share of every method off it
# (more than the dodged figure's cut at 12). That share is printed per method and written to
# the CSV rather than implied away; raise RB_GRID_X_MAX to see it.
#
# THE Y WINDOW IS THE DODGED FIGURE'S FIXED FIVE DECADES. Fliers that relaxed past 1e5 leave
# the top, as they do in the paper; how many per box is in the CSV (`n_above_ylim`).
#
# STAGES. The paper draws this twice, on the generated pose (Fig. 12) and after a local
# force-field minimisation (Fig. 13). Only `generated` exists in the data: no per-molecule
# minimised strain has been computed for any method yet. A stage is a field name on the rows,
# so adding the minimised one is a new entry here once the rows carry it.
RB_GRID_X_MAX = 7
RB_GRID_COLS = 3
# The crystal ligands are 79 molecules over eight counts; at the methods' floor of
# RB_GRID_MIN_N most of their boxes would vanish. Three is the least that gives a box a median
# and quartiles that are not the same point, and the CSV carries n for every box.
RB_GRID_REF_MIN_N = 3
# (position, colour) stops, all from the shared tables (2026-09-13): CoDE's SOFT blue at 0,
# through its PALE tint and VoxBind's PALE tint, to VoxBind's palette sand at the top -- the
# ends carry the colour, the middle stays light enough for the dark median bars and fliers.
RB_GRID_CMAP_STOPS = [(0.0, soft("CoDE")), (1 / 3, pale("CoDE")), (2 / 3, pale("VoxBind")),
                      (1.0, color("VoxBind"))]
RB_GRID_NORM, RB_GRID_CTICK = (0.0, 800.0), 200.0
RB_GRID_TALL = 0.9975        # per row, as a share of PANEL_H
# The outer names are a step above the house 15.5 pt, and the y name sits further off its
# tick labels -- on a 3x3 block the house sizes read as small.
RB_GRID_LABEL_FS, RB_GRID_CBAR_FS, RB_GRID_YPAD = 17, 15, 16
# Spines and MAJOR tick marks 1.2x the house weight: nine small panels read as washed out at
# 1.35, and 1.6x was too heavy. Minor ticks (the log decades' 2-9) keep their own weight.
RB_GRID_AXIS_LW = AXIS_LW * 1.2
# stage -> (row field, y-axis qualifier)
RB_GRID_STAGES = {"generated": ("s", "generated pose")}


def _rb_grid_boxes(rows, field, floor):
    """[(bond count, clipped values)] for every count on the axis that holds `floor` rows."""
    per = by_size(rows, field, key="rb")
    return [(x, np.clip(per[x], RB_STRAIN_FLOOR, None))
            for x in range(0, RB_GRID_X_MAX + 1) if len(per.get(x, ())) >= floor]


def _rb_grid(out, series, refrows, labels, stage):
    field, qualifier = RB_GRID_STAGES[stage]
    panels = _rb_panel_order(labels)
    rows_by_lab = {**{lab: series[lab] for lab in labels}, REF_LABEL: refrows}
    boxes = {lab: _rb_grid_boxes(rows_by_lab[lab], field,
                                 RB_GRID_REF_MIN_N if lab == REF_LABEL else RB_GRID_MIN_N)
             for lab in panels}

    all_meds = [float(np.median(v)) for lab in panels for _, v in boxes[lab]]
    norm = matplotlib.colors.Normalize(*RB_GRID_NORM)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rb_grid", RB_GRID_CMAP_STOPS)
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(min(all_meds) < norm.vmin, max(all_meds) > norm.vmax)]

    nrow = -(-len(panels) // RB_GRID_COLS)
    fig, axes = plt.subplots(nrow, RB_GRID_COLS, sharex=True, sharey=True,
                             figsize=(FIG_W * RB_BOX_WIDE_1, PANEL_H * RB_GRID_TALL * nrow),
                             dpi=220, squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for ax, lab in zip(flat, panels):
        ax.set_facecolor("white")
        drawn = boxes[lab]
        if drawn:
            bp = ax.boxplot([v for _, v in drawn], positions=[x for x, _ in drawn],
                            widths=0.72, whis=1.5, showfliers=True, patch_artist=True,
                            manage_ticks=False, zorder=5,
                            flierprops=dict(marker="d", markersize=2.6, markerfacecolor=INK,
                                            markeredgecolor="none", alpha=0.5))
            for box, (_, v) in zip(bp["boxes"], drawn):
                box.set(facecolor=cmap(norm(float(np.median(v)))), edgecolor=INK,
                        linewidth=0.9)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=INK, linewidth=0.9)
            for med in bp["medians"]:
                med.set(color=INK, linewidth=1.3, solid_capstyle="butt")
            # Tens of thousands of flier markers as vector paths make the PDF/SVG unplaceable;
            # the points are rasterised inside an otherwise vector figure.
            for fl in bp["fliers"]:
                fl.set_rasterized(True)
        ax.set_yscale("log")
        ax.set_ylim(*RB_BOX_YLIM)
        xlo = -0.6
        furniture(ax, xlim=(xlo, RB_GRID_X_MAX + 0.6), xloc=1)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(RB_GRID_AXIS_LW)
        ax.tick_params(which="major", width=RB_GRID_AXIS_LW)
        # An unlabelled tick where the axes meet, matching the y axis's 10^0 in that corner:
        # the first bond count sits 0.6 in, so without it the x axis looks cut short.
        counts = list(range(0, RB_GRID_X_MAX + 1))
        ax.set_xticks([xlo] + counts, [""] + [str(x) for x in counts])
        ax.set_title(display(lab), fontsize=14, color=INK, pad=5)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    fig.supxlabel(RB_X_LABEL, fontsize=RB_GRID_LABEL_FS, color=INK)
    # The y name goes on the MIDDLE row's left axes rather than fig.supylabel, because only an
    # axes label takes a labelpad -- supylabel sits flush against the tick labels. With three
    # rows the middle axes' centre is the block's centre.
    axes[nrow // 2, 0].set_ylabel(f"UFF strain energy (kcal mol⁻¹), {qualifier}",
                                  fontsize=RB_GRID_LABEL_FS, color=INK,
                                  labelpad=RB_GRID_YPAD)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, extend=extend,
                      shrink=0.92, aspect=38, pad=0.012,
                      ticks=np.arange(RB_GRID_NORM[0], RB_GRID_NORM[1] + RB_GRID_CTICK,
                                      RB_GRID_CTICK))
    cb.set_label("Median strain energy (kcal mol⁻¹)", fontsize=RB_GRID_CBAR_FS, color=INK)
    cb.ax.tick_params(labelsize=12, colors=AXIS, width=AXIS_LW)
    cb.outline.set_edgecolor(AXIS)
    cb.outline.set_linewidth(AXIS_LW)
    save(fig, out, f"strain_box_per_rotbond_grid_{stage}")

    csv_rows = []
    for lab in panels:
        for x, v in boxes[lab]:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            csv_rows.append([lab, x, len(v), round(float(med), 3), round(float(q1), 3),
                             round(float(q3), 3), int(np.sum(v > RB_BOX_YLIM[1]))])
    write_csv(out, f"strain_box_per_rotbond_grid_{stage}",
              ["method", "rotatable_bonds", "n", "strain_median", "strain_q25", "strain_q75",
               "n_above_ylim"], csv_rows)

    sat = [(lab, x, round(float(np.median(v)))) for lab in panels for x, v in boxes[lab]
           if not norm.vmin <= float(np.median(v)) <= norm.vmax]
    print(f"  {stage}: {len(panels)} panels · x = 0-{RB_GRID_X_MAX} rotatable bonds · "
          f"colour linear {norm.vmin:g}-{norm.vmax:g} kcal/mol (extend={extend}) · "
          f"saturated boxes {sat}")
    for lab in panels:
        rows = [r for r in rows_by_lab[lab] if r[field] is not None and r["rb"] is not None]
        beyond = 100 * sum(r["rb"] > RB_GRID_X_MAX for r in rows) / max(len(rows), 1)
        top = sum(int(np.sum(v > RB_BOX_YLIM[1])) for _, v in boxes[lab])
        print(f"    {lab:16s} {len(boxes[lab])} boxes · {len(rows):6d} scored · "
              f">{RB_GRID_X_MAX} bonds {beyond:5.1f}% (not drawn) · {top} fliers above "
              f"{RB_BOX_YLIM[1]:.0e}")


@figure("fig-posecheck-strain-rotbond-grid", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_grid(out):
    """Strain per rotatable bond, one panel per method, boxes coloured by median (VoxBind
    Fig. 12 form)."""
    use_style()
    series, refrows, _ = _rb_load_all()
    labels = _rb_order(series)
    for stage in RB_GRID_STAGES:
        _rb_grid(out, series, refrows, labels, stage)
