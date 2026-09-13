

# ════════════════════════════════════════════════════════════════════════════════
# fig-vina-per-atom — Vina Dock against ligand size, in the fig-vina-3line style
# ════════════════════════════════════════════════════════════════════════════════
# TWELVE FILES: three reference variants x two statistics x two arm sets, plus one CSV
# behind all of them. Which statistic a figure shows is carried by its filename and by its
# y-axis name ("Vina Dock / mean" / "Vina Dock / median", set over two lines so the pair of
# panels keeps one narrow left margin), exactly as in the 3-line figures -- there are no
# panel titles, so the y name is the only thing telling two otherwise identical figures
# apart. One CSV serves all twelve: the per-arm columns do not depend on which arms are
# drawn, and `plotted_in`, `reference_source` and `reference_firm` are the columns that
# carry what does.
#
# ── WHICH LIGANDS THE GREY CURVE IS ──────────────────────────────────────────────
# There are two possible reference populations. Which one is drawn is stated on EVERY run,
# in the progress line and in the CSV's `reference_source` and `n_reference_total` columns.
# The KEY names it only when it is the CrossDocked set -- "the 79" quietly passed off as
# "CrossDocked" is the mistake this guards against, and that mistake only runs one way.
#
# WHY THE KEY IS ASYMMETRIC, since a matching "(79 pockets)" looks more careful and is not.
# Every other figure in this section draws "Reference ligand" and means these same 79
# crystal ligands. Annotating it here alone would put one key at odds with its neighbours
# while describing an identical population, and a reader moving between them would
# reasonably infer that THIS reference is some other set. It is not. The ambiguity an
# annotation prevents does not exist until a second population does, and at that point the
# CrossDocked label carries the distinction exactly where it is needed. The bare label also
# keeps v1 and v2 byte-identical to the figures this port was verified against, which is
# the check the whole family rests on: between an annotation that is redundant in every
# figure already meaning the 79, and losing that check, the annotation goes. Provenance is
# not lost either way -- the log and the CSV carry it unconditionally, which is where a
# reader who needs it will look.
#
#   p79          The 79 crystal ligands of the test set, one per pocket: the docked value
#                out of each run's eval_docking_results_full79.json, the ligand's own
#                heavy-atom count out of that pocket's metrics.json `reference` block.
#                Drawn as plain "Reference ligand", the same key every other figure in the
#                section uses for these same molecules. This is the fallback.
#   crossdocked  The whole CrossDocked set, docked per ligand on another box and dropped
#                into the Dropbox-backed bundle at
#                results/task2-drugdesign/Reference/crossdocked_reference_dock.json,
#                which arrives with `bash results/dropbox_pull.sh Reference`. Drawn as
#                "Reference ligand (CrossDocked, n=...)". Schema, so the upload matches:
#
#                    {"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1},
#                                                    ...]}
#
#                `ligands` must be a non-empty list and every entry must carry both
#                `n_atoms` and `vina_dock`; anything else in the file is ignored, and
#                `protocol` is echoed into the run log so the figure states what it was
#                docked under. A file that does not parse fails LOUDLY with its path and
#                this schema rather than falling back to the 79 behind the reader's back.
#
# The file decides: crossdocked when it is there, p79 when it is not.
# VOXBIND_VINA_REFERENCE=p79|crossdocked forces one, and errors if the forced one is not
# available. The choice is printed either way.
#
# THIS IS WHY THE THINNESS LOGIC STAYS even though CrossDocked will mostly retire it. With
# tens of thousands of ligands behind it the rolling window is never thin and the fade
# never fires, which is the whole point of the upload; with the 79 it is thin at both ends
# and the fade is the honesty cue. The code does not need to know which -- it fades what is
# thin.
#
# ── TWO ARM SETS ─────────────────────────────────────────────────────────────────
# The same split every other figure in this section makes:
#
#   core  VoxBind and CoDE -- the two arms the figure exists to compare, and the only two
#         with an area under their distribution.
#   all   every arm that ACTUALLY HAS per-molecule Vina on this box, which today is those
#         two plus TargetDiff.
#
# WHAT AN ARM NEEDS TO BE IN `all`: `<its root>/eval_docking_results_full79.json`, carrying
# `per_target[].per_mol[]` entries with both `n_atoms` and `vina_dock`. An arm whose file is
# absent is SKIPPED WITH A PRINTED NOTE naming the path, not a crash and not a silent drop,
# so the day a file lands `all` simply grows -- adding a method is data, not surgery.
#
# The five published baselines are listed in VPA_ARMS and are skipped today. AR,
# Pocket2Mol, DiffSBDD, DecompDiff and FuncBind are staged here as sample directories
# (exps/baselines_pose/<key>/target_XX/) and scored for PoseBusters and PoseCheck, but
# their metrics.json carries `docking: "none"`: the Vina run behind
# results/task2-drugdesign/_shared/baselines_eval/summary_density79.json happened on svr12
# and only its AGGREGATES came back, so there is no (heavy atoms, vina_dock) pair per
# molecule to bin. Docking them here under this protocol is ~89 CPU-hours per method
# (79 pockets x ~100 molecules, exhaustiveness 32, whole receptor). So `all` DOES NOT MEAN
# "all eight published methods" today, and the run log says which arms it did mean.
#
# The two sets do not share an x span, and that is the point of drawing both: x runs only
# where EVERY drawn arm clears MIN_N, so dropping TargetDiff buys a count back at the right
# end (core reaches 45, all stops at 44) and removes the two interior counts where
# TargetDiff alone is thin.
#
# ── THREE VARIANTS ───────────────────────────────────────────────────────────────
# ONE FUNCTION. They differ only in what the crystal reference does, over the three knobs
# in VPA_VARIANTS below: whether it is on the Vina panel, whether it is on the distribution
# strip, and how thin its rolling window may get before it stops being drawn at full
# strength. They are kept together on purpose: split apart they would drift, and then the
# "same figure, one choice different" claim would quietly stop being true.
#
#   v1  The reference's Vina curve is GATED: drawn only where its rolling window holds at
#       least VPA_REF_FIRM ligands, so on the 79 the grey stops at 36 atoms. Its
#       distribution below is smoothed and filled like the models'.
#   v2  The reference's Vina curve runs the full x range, FADED where the window thins
#       (below VPA_REF_FIRM, from 37 atoms up on the 79). Its distribution is smoothed but
#       UNFILLED, so the two model areas underneath stay readable.
#
#       v2's cost is real and visible: on the 79 the reference median at 42-44 is two
#       crystal ligands, TNKS1 (target_72, -15.99) and AKT1 (target_80, -14.66), and
#       drawing them stretches y to -15.5, which compresses the -3..-11 band where the arms
#       are actually being compared. v1 spends nothing on y but ends its grey eight atoms
#       short of where the panel below it ends, which reads as missing data rather than as
#       thin data. Neither is free; that is why both are built.
#   v3  v2's Vina panel with the gate opened all the way, and NO reference on the
#       distribution strip. The grey curve is drawn at every count whose +-REF_WIN window
#       holds so much as one ligand -- nothing is dropped for being thin, only faded below
#       VPA_REF_FIRM, which is the honesty cue doing that job -- and the strip below carries
#       the model arms alone. The reference keeps its key entry, because it is still drawn.
#
#       WHAT V3 GIVES UP. It used to drop the reference from BOTH panels, and its selling
#       point was the y axis: nothing on the panel reached past the models, so the -3..-11
#       band they are compared in filled the height. Drawing the grey again hands that
#       back -- on the 79, TNKS1 and AKT1 sit under 42-44 atoms and stretch y to -15.5, the
#       same squeeze v2 pays. What it keeps is the benchmark on the panel where the
#       comparison is actually made, and what it buys is the strip: with no grey step or
#       fill there, the model areas are read without a fourth series over them.
#
#       ON THE 79 THE OPEN GATE CHANGES NOTHING. Over the plotted span that window never
#       falls below three ligands -- its minimum, at 43-45 atoms -- which v2's floor already
#       admits, so v3's grey curve and v2's are the same line and the two variants differ
#       only in the strip below. The open gate is what the figure does when x reaches
#       further right: the 79 crystal ligands span 6 to 57 heavy atoms but only five sit
#       past 40, and from 46 up the window is down to one or two. Against the CrossDocked
#       reference it is moot in the other direction -- nothing is thin.
#
# ── WHAT THEY SHOW ───────────────────────────────────────────────────────────────
# Two stacked panels over one x axis, the ligand's heavy-atom count.
#
#   top     Vina Dock mean or median at each exact heavy-atom count, for every arm in the
#           set, and the reference ligands.
#   bottom  How many molecules each set puts at each size, as a share of its own molecules
#           -- a share and not a count, because the reference set and ~7,900 generated
#           molecules do not share a count axis. This is what makes the top panel's tails
#           trustworthy or not, and is itself the finding the figure exists to guard
#           against: the arms differ in the size distribution they generate at least as
#           much as in per-atom binding quality (see the size-confound note; raw pooled
#           Vina Dock is ~80% a size statistic). In v1 and v2 the reference is on this
#           panel too, so "does the model generate ligands the size of the real one" is
#           readable without a second figure; v3 is the variant that gives that up.
#
# EVERY ARM IS GATED THE SAME WAY, per arm rather than per figure. A heavy-atom count
# enters x only if EVERY drawn arm has MIN_N molecules there, which sets the two ends;
# inside that span an arm that dips below MIN_N (TargetDiff does, at 38 and 41 atoms, which
# is why only `all` is broken there) has its curve BROKEN rather than interpolated, because
# a bridge over a count where one method generated 15 molecules is a drawn claim about data
# that is not there. The distribution panel underneath keeps drawing at those counts --
# being thin is exactly what it is for.
#
# ONE STATISTIC PER FIGURE. Both were drawn in one panel at first, dash against solid, and
# it said nothing extra for the ink: at every size the two run within ~0.2 kcal/mol of each
# other, so six curves were three curves drawn twice and the pair merely thickened and
# blurred each series. They are separate figures instead. That near-agreement IS a result
# -- the Ours/VoxBind gap is the whole distribution shifting, not a tail dragging the mean
# -- and it is what makes the two figures nearly interchangeable; the mean is the slightly
# kinder one to CoDE (below VoxBind at 35/40 sizes against the median's 31/40, mean
# per-size gap -0.441 against -0.388 on the `all` span), because CoDE's size distribution
# reaches further right.
#
# INK IS THE SECOND CHANNEL, and it says what a series is FOR: the two arms being compared
# are the thickest, solid, and the only ones with an area under their distribution;
# everything else is thinner and DASHED, because it is what the pair is read against rather
# than a further competitor, and four filled areas on the lower panel would be a stack no
# one can read through. That is also the rule any newly docked baseline arrives under: it
# joins as context ink, not as a third solid curve. CoDE is periwinkle #8291E8 here, the
# lighter tint of our blue this Vina family carries, as the palette records.
#
# PROTOCOL. The published-baseline protocol, exactly as in the 3-line figures: whole
# `*_rec.pdb` receptor, exhaustiveness 32, all 79 pockets, from
# `eval_docking_results_full79.json`. The same runs, so this figure and the 3-line figures
# sit on one axis -- TargetDiff included, which 74_dock_targetdiff_full79.sh re-docked here
# for exactly that reason. The CrossDocked reference is docked elsewhere, so its own
# `protocol` block is echoed into the run log rather than assumed to match.
#
# WHY NO CONFIDENCE BANDS. The thing a band would guard against here is already drawn: x is
# clipped to the counts where EVERY arm has at least MIN_N molecules, so no curve has a
# tail the others cannot answer, and the bottom panel shows how thin each end actually is.
# The per-count sample sizes are in the CSV.
#
# THE REFERENCE IS ROLLED. On the 79 there is exactly ONE crystal ligand per pocket, 1-6 at
# any exact heavy-atom count, so anything per-count off it is noise: raw, its distribution
# is a picket fence of 1.3%-tall steps reaching 7.6% where six pockets happen to share a
# size, and that spike, not the models, would set the lower panel's y scale. Both its
# curves are therefore a centred rolling window of +-REF_WIN atoms, and the window stays
# for the CrossDocked set so the two sources are read the same way. The models are NOT
# rolled, in either panel: they have hundreds of molecules per count.
VPA_BASENAME = "eval_docking_results_full79.json"
VPA_PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# The CrossDocked reference drop, REPO-relative because this tree gets copied between
# boxes. The folder is the Dropbox-backed bundle, so the upload lands with
# `bash results/dropbox_pull.sh Reference` and needs no sync path of its own.
VPA_REF_DOCK = ("results", "task2-drugdesign", "Reference",
                "crossdocked_reference_dock.json")
VPA_REF_SOURCES = ("p79", "crossdocked")
VPA_REF_ENV = "VOXBIND_VINA_REFERENCE"
VPA_REF_SCHEMA = ('{"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1}, '
                  '...]}  -- extra keys ignored')

# The whole difference between the three figures, and all of it about the reference.
# `ref_vina` puts it on the top panel, `ref_dist` on the strip below; it earns its key
# entry by being on either. `ref_gate` names the floor its rolling window has to clear
# before a point is drawn at all -- VPA_REF_GATE maps the three settings to that number --
# and everything except "hard" fades the stretch below VPA_REF_FIRM rather than cutting it.
# `ref_dist_smooth` rolls the reference's distribution; `ref_dist_fill` gives it an area
# under it like the two compared arms have.
VPA_VARIANTS = {
    "v1": dict(ref_vina=True, ref_dist=True, ref_gate="hard",
               ref_dist_smooth=True, ref_dist_fill=True),
    "v2": dict(ref_vina=True, ref_dist=True, ref_gate="fade",
               ref_dist_smooth=True, ref_dist_fill=False),
    "v3": dict(ref_vina=True, ref_dist=False, ref_gate="full",
               ref_dist_smooth=True, ref_dist_fill=False),
}

VPA_FIELD, VPA_REF_FIELD = "vina_dock", "ref_vina_dock"
VPA_STATS = ("mean", "median")       # one figure each, per variant and per arm set

# The y names are set over TWO LINES. One line each ("Vina Dock median", "% of ligands")
# put the widest label's length into the left margin of a 7.6-inch figure; broken after
# the metric they are the same words in half the width, and align_ylabels below keeps the
# two panels' labels on one left edge. The statistic still lives in the y name and
# nowhere else, exactly as in the 3-line figures.
VPA_Y_LABEL = "Vina Dock\n{stat}"
VPA_D_LABEL = "Generated\nfraction"

# These are DATA KEYS as well as labels (they head this folder's CSV columns), so they
# stay plain; display() dresses them for a legend -- VoxBind picks up its sigma=0.9
# subscript there. CoDE is \textsc{CoDE} in the .tex files; matplotlib has no small caps
# without a TeX backend, so the figures carry the plain string.
VPA_VOX_LABEL, VPA_OUR_LABEL = "VoxBind", "CoDE"
VPA_OUR_COLOR = soft(VPA_OUR_LABEL)

# Line weights in points. VPA_MODEL_BOOST widens ONLY the two arms the figure exists to
# compare -- both their Vina curves and their distribution steps -- and is applied as a
# factor on the shared base weights rather than baked into them, so the gap it opens
# against the context series' thinner ink stays visible as a decision.
VPA_MODEL_BOOST = 1.2
VPA_MODEL_LW, VPA_BASE_LW = MODEL_LW * VPA_MODEL_BOOST, 1.9
VPA_DIST_LW = DIST_LW * VPA_MODEL_BOOST

# The reference dash is the shared DASH: legible at hairline weights, and it does not
# shimmer where it runs close to a model curve. VPA_BASE_DASH is the context arms', longer
# so the two kinds of dashed series are told apart by rhythm as well as by colour.
VPA_BASE_DASH = (0, (6, 2))

# label, run root, line width, dash -- DRAWN IN THIS ORDER, so ours lands on top and the
# context series sit under it. Roots are REPO-relative (E) or resolved off the sibling
# checkout (BASEDRUG); nothing here is an absolute path. The five baselines are listed and
# will be SKIPPED until their eval_docking_results_full79.json exists, which is how a newly
# docked method joins `all` without a code change.
VPA_ARMS = [
    ("AR",          f"{E}/baselines_pose/ar",         VPA_BASE_LW, VPA_BASE_DASH),
    ("Pocket2Mol",  f"{E}/baselines_pose/pocket2mol", VPA_BASE_LW, VPA_BASE_DASH),
    ("DiffSBDD",    f"{E}/baselines_pose/diffsbdd",   VPA_BASE_LW, VPA_BASE_DASH),
    ("DecompDiff",  f"{E}/baselines_pose/decompdiff", VPA_BASE_LW, VPA_BASE_DASH),
    ("FuncBind",    f"{E}/baselines_pose/funcbind",   VPA_BASE_LW, VPA_BASE_DASH),
    ("TargetDiff",  str(BASEDRUG / "eval" / "targetdiff"), VPA_BASE_LW, VPA_BASE_DASH),
    (VPA_VOX_LABEL, f"{E}/_vanilla_ep923/samples/full_eval_ep923", VPA_MODEL_LW, "-"),
    (VPA_OUR_LABEL, f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
     VPA_MODEL_LW, "-"),
]
# The two arms this figure exists to compare: the only ones drawn solid, and the only ones
# given an area under their distribution. Everything else on the panel is context -- and
# `core` is exactly this pair with the context removed, which is why one tuple names both.
VPA_FOCUS = (VPA_VOX_LABEL, VPA_OUR_LABEL)
# Every arm takes its colour from the shared palette, so a method is the same colour here
# as in the PoseBusters and PoseCheck figures. CoDE is the one override: this Vina family
# carries the lighter tint of our blue.
VPA_COLORS = {lab: (VPA_OUR_COLOR if lab == VPA_OUR_LABEL else color(lab))
              for lab, *_ in VPA_ARMS}
VPA_COLORS[REF_LABEL] = REF_COLOR
# The p79 crystal ligand is the same molecule in every run that carries one, but it is
# resolved BY LABEL, never by position in VPA_ARMS -- the staged baselines have no
# reference block at all, and a positional pick would silently become one of them.
VPA_REF_ROOT = next(root for lab, root, _, _ in VPA_ARMS if lab == VPA_VOX_LABEL)

# The reference window is +-REF_WIN atoms, and these are the three floors it can be held
# to. VPA_REF_FIRM is where it holds enough ligands to be read as a curve (v1 gates there,
# v2 and v3 fade below it); VPA_REF_FLOOR is v2's floor for drawing a point at all;
# VPA_REF_ANY is v3's, which refuses only a window holding nothing. The curve is computed
# once at VPA_REF_ANY and masked down, so one array serves all three gates and the CSV
# reports every value any of them can draw.
VPA_REF_FIRM, VPA_REF_FLOOR, VPA_REF_ANY = 12, 3, 1
VPA_REF_GATE = {"hard": VPA_REF_FIRM, "fade": VPA_REF_FLOOR, "full": VPA_REF_ANY}
VPA_REF_THIN_ALPHA = 0.42


def _vpa_slug(label):
    """The CSV's column prefix for an arm: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


# Bundle folder for an arm that has no run tree here. The five published baselines were
# docked on svr12 and only their aggregates came back until 2026-09-13, when the bundle
# gained <Method>/eval/vina_docking/per_molecule.csv -- target, n_atoms and the three Vina
# columns per molecule, which is exactly what this figure bins. They are read from THERE.
# Our own arms keep their run tree: the bundle's `VoxBind-vanilla` is res_test_100, a
# DIFFERENT run from the _vanilla_ep923 this figure has always drawn, and swapping the
# source underneath an arm would silently redraw it.
VPA_BUNDLE = {"AR": "AR", "Pocket2Mol": "Pocket2Mol", "DiffSBDD": "DiffSBDD",
              "DecompDiff": "DecompDiff", "FuncBind": "FuncBind"}


def _vpa_load_bundle(label):
    """{heavy atoms: [vina dock, ...]} over the 79 density pockets, from the bundle CSV."""
    folder = VPA_BUNDLE.get(label)
    rows = vina_rows(folder, density79=True) if folder else None
    if not rows:
        return None, 0
    sizes = collections.defaultdict(list)
    for r in rows:
        if r.get(VPA_FIELD) is not None and r.get("n"):
            sizes[r["n"]].append(r[VPA_FIELD])
    return (sizes, len({r["target"] for r in rows})) if sizes else (None, 0)


def _vpa_load(root, label=None):
    """{heavy atoms: [vina dock, ...]} pooled over the 79 pockets of one run, or None if
    this arm has not been docked here -- an absent file is a method waiting for its Vina
    run, not an error, and the caller says so in the log instead of drawing an empty
    curve. An arm with no run tree falls to its bundle CSV (see VPA_BUNDLE)."""
    path = os.path.join(root, VPA_BASENAME)
    if not os.path.exists(path):
        return _vpa_load_bundle(label) if label else (None, 0)
    with open(path, encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes = collections.defaultdict(list)
    for t in per_target:
        for m in t.get("per_mol") or []:
            if isinstance(m, dict) and m.get(VPA_FIELD) is not None and m.get("n_atoms"):
                sizes[m["n_atoms"]].append(m[VPA_FIELD])
    return sizes, len(per_target)


def _vpa_load_reference(root):
    """The p79 reference: {heavy atoms: [crystal-ligand vina dock, ...]}, one entry per
    pocket.

    The docked value is in the results file, but the ligand's own atom count is not -- it
    lives in that pocket's metrics.json, under `reference`. Pockets missing either are
    dropped and named by the caller."""
    with open(os.path.join(root, VPA_BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes, dropped = collections.defaultdict(list), []
    for t in per_target:
        path = os.path.join(root, t["target"], "metrics.json")
        n_atoms = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                n_atoms = (json.load(handle).get("reference") or {}).get("n_atoms")
        value = t.get(VPA_REF_FIELD)
        if n_atoms and value is not None:
            sizes[n_atoms].append(value)
        else:
            dropped.append(t["target"])
    return sizes, dropped


def _vpa_load_crossdocked(path):
    """The CrossDocked reference drop: {heavy atoms: [vina dock, ...]} over every ligand in
    it, plus whatever `protocol` block it carries.

    Every failure raises with the path AND the schema. This file is written on another box
    and pulled in, so the realistic failure is a shape mismatch, and the one thing that
    must never happen is falling back to the 79 while the key still says CrossDocked."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    ligands = payload.get("ligands") if isinstance(payload, dict) else None
    if not isinstance(ligands, list) or not ligands:
        raise ValueError(f"{path}: needs a non-empty 'ligands' list\n"
                         f"  schema: {VPA_REF_SCHEMA}")
    sizes = collections.defaultdict(list)
    for i, m in enumerate(ligands):
        if not isinstance(m, dict) or m.get("n_atoms") is None \
                or m.get(VPA_FIELD) is None:
            raise ValueError(
                f"{path}: ligand {i} has no 'n_atoms' and/or '{VPA_FIELD}'\n"
                f"  schema: {VPA_REF_SCHEMA}")
        sizes[int(m["n_atoms"])].append(float(m[VPA_FIELD]))
    return sizes, payload.get("protocol")


def _vpa_reference_set():
    """(source, {heavy atoms: [vina dock]}, the key's label, the log line, ligand count).

    The file decides -- CrossDocked when the drop is there, the 79 when it is not -- and
    VPA_REF_ENV forces one either way, erroring rather than silently substituting the
    other. Whichever wins, its name goes in the key, in the log and in the CSV."""
    path = REPO.joinpath(*VPA_REF_DOCK)
    forced = os.environ.get(VPA_REF_ENV) or None
    if forced is not None and forced not in VPA_REF_SOURCES:
        raise ValueError(f"{VPA_REF_ENV}={forced!r}: expected one of "
                         f"{' or '.join(VPA_REF_SOURCES)}")
    if forced == "crossdocked" and not path.exists():
        raise FileNotFoundError(
            f"{VPA_REF_ENV}=crossdocked, but {path} is not there\n"
            f"  pull it with: bash results/dropbox_pull.sh Reference\n"
            f"  schema: {VPA_REF_SCHEMA}")
    if forced == "p79" or (forced is None and not path.exists()):
        sizes, dropped = _vpa_load_reference(VPA_REF_ROOT)
        n = sum(len(v) for v in sizes.values())
        note = (f"p79 · {n} crystal ligands, one per pocket, from {VPA_BASENAME}"
                + (f" (dropped {', '.join(dropped)})" if dropped else ""))
        return "p79", sizes, REF_LABEL, note, n
    sizes, protocol = _vpa_load_crossdocked(path)
    n = sum(len(v) for v in sizes.values())
    if isinstance(protocol, dict):
        protocol = ", ".join(f"{k}={v}" for k, v in protocol.items())
    note = (f"crossdocked · {n:,} ligands from {path.relative_to(REPO)}"
            + (f" · docked under {protocol}" if protocol else ""))
    # One line. It is roughly three times the width of a method label, which is exactly
    # why it cannot sit INSIDE the method grid: a legend column is as wide as its widest
    # entry, so dropping this into one would stretch that column and break the 2 x 4. It
    # gets its own centred key above the grid instead.
    #
    # The count alone carries the distinction the annotation exists for: the p79 set is 79
    # and this one is six figures, so "(n=100,090)" cannot be mistaken for the test
    # pockets' ligands, and the bare "Reference ligand" keeps meaning the 79 exactly as it
    # does in every other figure of the section. Which population it is, and under what
    # protocol, is printed in full on every run and carried in the CSV.
    return "crossdocked", sizes, f"{REF_LABEL} (n={n:,})", note, n


def _vpa_arm_sets(arms):
    """(name, arms) for the two sets, core first. `arms` is what actually has data."""
    return (("core", [a for a in arms if a[0] in VPA_FOCUS]), ("all", list(arms)))


def _vpa_model_curve(sizes, xs, stat):
    """mean/median at each exact heavy-atom count, None below MIN_N.

    None and not the value: a count an arm put fewer than MIN_N molecules at is a number
    the other arms cannot be asked to answer, and plotting it would also let matplotlib
    draw a straight segment over it as though the arm had been measured there."""
    f = st.mean if stat == "mean" else st.median
    return [f(sizes[a]) if len(sizes.get(a, ())) >= MIN_N else None for a in xs]


def _vpa_window_counts(sizes, xs):
    """How many reference ligands each plotted point's rolling window pools."""
    return [sum(len(v) for n, v in sizes.items() if abs(n - a) <= REF_WIN) for a in xs]


def _vpa_reference_curve(sizes, xs, stat, floor):
    """The reference over a centred +-REF_WIN window; None where the window holds fewer
    than `floor` ligands, which leaves a gap in the line rather than an invented value."""
    f = st.mean if stat == "mean" else st.median
    out = []
    for a in xs:
        pool = [v for n, vals in sizes.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(pool) if len(pool) >= floor else None)
    return out


def _vpa_gated(values, counts, gate):
    """The reference curve held to one variant's floor: the value where its window clears
    VPA_REF_GATE[gate], None below. The curve itself is computed once at VPA_REF_ANY, so
    the three gates are three masks over one array rather than three recomputations."""
    floor = VPA_REF_GATE[gate]
    return [v if c >= floor else None for v, c in zip(values, counts)]


def _vpa_split_firm(values, counts):
    """The reference curve cut into the part its window can carry and the part it cannot.

    Returns (firm, thin): both full-length, each holding None wherever the other holds the
    value. A point on the boundary belongs to BOTH, so the solid and the faded stretch meet
    rather than leaving a gap -- a gap would say "no data here", which is the opposite of
    what the fade is for."""
    firm_at = [v is not None and c >= VPA_REF_FIRM for v, c in zip(values, counts)]
    n = len(firm_at)
    edge = [firm_at[i] and any(not firm_at[j] for j in (i - 1, i + 1) if 0 <= j < n)
            for i in range(n)]
    firm = [v if f else None for v, f in zip(values, firm_at)]
    thin = [v if (not f) or e else None for v, f, e in zip(values, firm_at, edge)]
    return firm, thin


def _vpa_furniture(ax, *, ylabel, xlabel, xs):
    """The 3-line figure's axes, over the plotted heavy-atom span. The y names are two
    lines each, so they are centred on one another rather than left-ragged."""
    furniture(ax, ylabel=ylabel, xlabel=xlabel, xlim=(xs[0] - 0.6, xs[-1] + 0.6),
              xloc=XTICK_STEP)   # ticks and vertical rules both, unlike the 3-line figure
                                 # where the rules fall between the labelled ranks
    ax.yaxis.label.set_multialignment("center")


# Above this many entries the key stops fitting in the panel's empty corner and starts
# covering the curves it explains. `core` has three and keeps the corner; `all` has nine
# and takes a strip under the figure instead.
VPA_KEY_INSIDE_MAX = 4
# Inside that strip the reference is NOT one of the methods -- it is the benchmark they are
# read against -- so it takes its own centred line above a 2 rows x 4 columns grid of the
# eight methods, which is the shape that fits this figure's width without reaching the axes.
VPA_KEY_NCOL = 4
# The two lines are laid out as two legends and then wrapped in one frame; this is the gap
# left between them, in figure fractions, INSIDE that shared frame.
VPA_KEY_ROW_GAP = 0.002


def _vpa_row_major(handles, ncol):
    """Reorder for matplotlib's column-major legend fill, so the key reads ACROSS.

    A legend with ncol=4 lays its entries down column 1, then column 2 -- so the drawing
    order the eye expects along a row is not the order it gets. Interleaving here puts the
    entries back in reading order."""
    rows = -(-len(handles) // ncol)
    out = []
    for c in range(ncol):
        for r in range(rows):
            i = r * ncol + c
            if i < len(handles):
                out.append(handles[i])
    return out


def _vpa_legend(ax, series, ref_label, *, fig=None):
    """The key, drawn in plotting order, its handles carrying each series' dash so an
    identity is readable from the key alone.

    WHERE IT GOES DEPENDS ON HOW MANY THERE ARE. Three series fit in the empty upper right
    and belong there -- next to the data, costing no height. Nine do not: at this panel
    size a nine-row box reaches more than halfway down and across, and in the `all` variant
    it sat squarely over the x axis. Past VPA_KEY_INSIDE_MAX the key moves beneath the
    figure as a reference row over a 2x4 grid of methods.

    `ref_label` is bare "Reference ligand" for the 79 -- the section's own name for them --
    and names the size only for the CrossDocked set, which is the one that would otherwise
    be read as the 79 (see the banner)."""
    handle = lambda lab, colour, lw, style: Line2D(
        [], [], color=colour, lw=lw, ls=style,
        label=ref_label if lab == REF_LABEL else display(lab))
    handles = [handle(*a[:4]) for a in series]
    if len(handles) <= VPA_KEY_INSIDE_MAX or fig is None:
        legend(ax, handles, loc="upper right")
        return []

    ref = [h for a, h in zip(series, handles) if a[0] == REF_LABEL]
    methods = [h for a, h in zip(series, handles) if a[0] != REF_LABEL]
    ncol = VPA_KEY_NCOL

    # ONE box, built as TWO legends. A legend column is as wide as its widest entry, so
    # the reference -- one line, and about three times the width of a method name -- cannot
    # be a cell of the method grid without stretching whichever column it landed in, and
    # the 2 x 4 would stop being a grid. matplotlib cannot span a cell either. So the two
    # lines get their own layouts, stacked and each centred, and then the frames come off
    # and a single rectangle is drawn round the pair: the reader sees one key, while
    # matplotlib still lays out the grid and the reference independently, which is what
    # keeps the grid even and the reference genuinely centred over it.
    legs = [legend(fig, _vpa_row_major(methods, ncol), loc="lower center", ncol=ncol,
                   fontsize=11.5, bbox_to_anchor=(0.5, 0.008), columnspacing=1.5)]
    if not ref:
        return legs
    fig.canvas.draw()
    h = legs[0].get_window_extent(fig.canvas.get_renderer()).height / \
        fig.get_size_inches()[1] / fig.dpi
    legs.append(legend(fig, ref, loc="lower center", ncol=1, fontsize=11.5,
                       bbox_to_anchor=(0.5, 0.008 + h + VPA_KEY_ROW_GAP)))
    return legs + [_vpa_one_frame(fig, legs)]


def _vpa_one_frame(fig, legs):
    """Take the frames off stacked keys and draw ONE around their union, so they read as a
    single box. Returns the patch, which is then the outermost thing the axes must clear.

    A legend's window extent already includes its own borderpad, so the union sits exactly
    where the individual frames did -- no padding is added here, or the pair would gain a
    ring the single-legend figures do not have."""
    fig.canvas.draw()
    boxes = [l.get_window_extent(fig.canvas.get_renderer()) for l in legs]
    for l in legs:
        l.set_frame_on(False)
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(min(b.x0 for b in boxes), min(b.y0 for b in boxes)),
         (max(b.x1 for b in boxes), max(b.y1 for b in boxes))])
    patch = matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, figure=fig,
        facecolor="white", edgecolor=LEGEND_EDGE, linewidth=AXIS_LW, zorder=6)
    fig.add_artist(patch)
    return patch


def _vpa_clear_keys(fig, legs, axes):
    """Move the axes up until the LOWEST THING THEY DRAW clears the figure-level keys.

    tight_layout lays the axes out over the whole figure and cannot see a legend that
    belongs to the FIGURE, so it puts the x label underneath one. Raising `bottom` is not
    enough on its own either: the x label hangs BELOW the axes box and travels up with it,
    so the thing that has to clear the key is the axes' tight bounding box, not its frame.
    Measure both and close the gap -- the same after-the-fact correction fit() makes for an
    overrunning y label, and it keeps holding when the key gains a row or the type moves."""
    if not legs:
        return
    for _ in range(3):                      # a shift changes the layout; re-measure
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        h = fig.get_size_inches()[1] * fig.dpi
        key_top = max(l.get_window_extent(r).y1 for l in legs) / h
        drawn_bottom = min(a.get_tightbbox(r).y0 for a in axes) / h
        gap = key_top + 0.02 - drawn_bottom
        if gap <= 0.001:
            break
        sp = fig.subplotpars
        fig.subplots_adjust(bottom=min(sp.bottom + gap, sp.top - 0.1))


def _vpa_resolve(arms, data, ref):
    """Everything that depends on WHICH arms are drawn, for one arm set.

    x is the only thing the two sets really argue about, and it cascades: the per-count
    curves, the window counts and the shares are all evaluated over it, and the reference's
    rolled share is truncated by its two ends. Per-arm values themselves do not depend on
    the set -- an arm's mean at 30 atoms is its mean at 30 atoms -- which is what lets one
    CSV stand behind both."""
    firm = [a for a in sorted(set.intersection(*(set(data[lab]) for lab, *_ in arms)))
            if all(len(data[lab].get(a, ())) >= MIN_N for lab, *_ in arms)]
    xs = list(range(firm[0], firm[-1] + 1))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, xs, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], xs, stat)
    exact, totals = {}, {}
    for label, sizes in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]:
        exact[label], totals[label] = share(sizes, xs)
    return {"arms": arms, "xs": xs, "curves": curves, "exact": exact, "totals": totals,
            "counts": _vpa_window_counts(ref, xs),
            "thin": {lab: [a for a in xs if len(data[lab].get(a, ())) < MIN_N]
                     for lab, *_ in arms}}


def _vpa_build(out, name, cfg, stat, aset, S, ref_label):
    """One variant at one statistic on one arm set: the figure, and the line about it for
    the run log."""
    stem = f"vina_dock_per_atom_{name}_{stat}_{aset}"
    xs, curves, counts = S["xs"], S["curves"], S["counts"]

    # Reference first and ours last, so the series being read sits on top -- the same
    # ordering the 3-line figures use. The key is the union of the two panels: a series
    # drawn on either one belongs in it, which is how v3 keeps its reference entry while
    # taking the grey off the strip below.
    series = [(lab, VPA_COLORS[lab], lw, style, z)
              for z, (lab, _, lw, style) in enumerate(S["arms"], start=3)]
    if cfg["ref_vina"] or cfg["ref_dist"]:
        series.insert(0, (REF_LABEL, REF_COLOR, REF_LW, DASH, 2))
    dist_series = [s for s in series if s[0] != REF_LABEL or cfg["ref_dist"]]

    shares = dict(S["exact"])
    if cfg["ref_dist_smooth"]:
        shares[REF_LABEL] = rolled(S["exact"][REF_LABEL], xs)
    ref_values = _vpa_gated(curves[(REF_LABEL, stat)], counts, cfg["ref_gate"])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw=dict(height_ratios=list(HEIGHT_RATIOS)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bx.set_facecolor("white")

    for label, colour, lw, style, z in series:
        if label != REF_LABEL:
            ax.plot(xs, curves[(label, stat)], color=colour, lw=lw, ls=style, zorder=z,
                    solid_capstyle="round", dash_capstyle="round")
        elif not cfg["ref_vina"]:
            continue
        elif cfg["ref_gate"] == "hard":
            ax.plot(xs, ref_values, color=colour, lw=lw, ls=style, zorder=z,
                    dash_capstyle="round")
        else:
            firm, thin = _vpa_split_firm(ref_values, counts)
            for ys, alpha in ((thin, VPA_REF_THIN_ALPHA), (firm, 1.0)):
                ax.plot(xs, ys, color=colour, lw=lw, ls=style, zorder=z, alpha=alpha,
                        dash_capstyle="round")

    # The same identities below: colour for the series, dash for the ones it is read
    # against. The arms are per-count steps; the reference is a curve when it is smoothed,
    # because drawing a rolling average stepped would claim a precision it lost. Only the
    # two VPA_FOCUS arms get an area -- four filled bands would be a stack, not a
    # comparison -- and in `core` that is both of them.
    for label, colour, lw, style, z in dist_series:
        stepped = label != REF_LABEL or not cfg["ref_dist_smooth"]
        filled = label in VPA_FOCUS or (label == REF_LABEL and cfg["ref_dist_fill"])
        if filled:
            bx.fill_between(xs, shares[label], step="mid" if stepped else None,
                            color=colour, alpha=DIST_FILL, linewidth=0, zorder=z)
        draw = bx.step if stepped else bx.plot
        draw(xs, shares[label], color=colour, ls=style, zorder=z + 5,
             lw=VPA_DIST_LW if label in VPA_FOCUS else lw,
             solid_capstyle="round", dash_capstyle="round",
             **(dict(where="mid") if stepped else {}))

    _vpa_furniture(ax, ylabel=VPA_Y_LABEL.format(stat=stat), xlabel=None, xs=xs)
    _vpa_furniture(bx, ylabel=VPA_D_LABEL, xlabel=X_LABEL, xs=xs)
    ax.margins(y=0.075)              # the default 5% puts ours on the bottom spine
    # Every 2 kcal/mol. The locator, left to itself over a ~9 kcal/mol span, picks 3 and
    # labels -3/-6/-9, which reads as a coarser axis than the 3-line figures' step of 2.
    ax.yaxis.set_major_locator(MultipleLocator(2))
    # Headroom over what is ON THIS PANEL, which is why it reads dist_series: v3's grey is
    # on the panel above only, and letting its rolled share set this scale would leave the
    # strip short by the height of a series that is not in it.
    bx.set_ylim(0, max(v for lab, *_ in dist_series for v in shares[lab]) * 1.13)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if name == "v3":
        # v3's strip is ticked 0/4/8 % on request (2026-09-13) rather than the locator's
        # 0/3/6; the other variants keep the locator.
        bx.yaxis.set_major_locator(MultipleLocator(4))
    # "Generated fraction" is the name; the per-cent sign on the ticks is where the unit
    # is stated, so the label does not have to carry a parenthesis to stay honest.
    bx.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    # Each y label is otherwise placed against ITS OWN tick labels, so "Vina Dock median"
    # (widest tick "-10") and "Generated fraction" (widest tick "4%") sit at two different
    # x and the pair reads as a ragged left edge. align_ylabels puts both at the outer of
    # the two, which is the only way they line up without hard-coding a coordinate.
    fig.align_ylabels((ax, bx))
    legs = _vpa_legend(ax, series, ref_label, fig=fig)
    fit(fig, pad=0.5, h_pad=H_PAD)
    _vpa_clear_keys(fig, legs, (ax, bx))
    lo, hi = ax.get_ylim()
    save(fig, out, stem)

    drawn = [a for a, v in zip(xs, ref_values) if v is not None]
    where = (f"reference {drawn[0]}..{drawn[-1]} atoms" if cfg["ref_vina"]
             else "no reference")
    if not cfg["ref_dist"]:
        where += ", off the strip"
    return f"    {stem:40s} x {xs[0]}..{xs[-1]}   {where}   y {lo:.1f} .. {hi:.1f}"


def _vpa_write_csv(out, sets, arms, data, ref, source, n_ref):
    """One table behind all twelve figures.

    It is one table because almost nothing in it is per figure. An arm's count, mean,
    median and share at a heavy-atom count are properties of that arm, not of the picture
    it is drawn in, so the arm columns are evaluated once over the UNION of the two x spans
    -- which means a row may carry a value at a count only `core` plots, and `plotted_in`
    is the column that says so.

    Four columns carry what does differ. `plotted_in` names the arm sets whose x span
    contains the row. `reference_source` and `n_reference_total` name the POPULATION behind
    every reference column -- p79 or crossdocked, and how many ligands -- which is constant
    down the table and repeated anyway, because a flat CSV read apart from its figure has
    nowhere else to put it and a reference column whose population is unstated is the one
    thing this figure will not ship. `reference_firm` is yes wherever the rolling window
    holds at least VPA_REF_FIRM ligands, which is the whole of the variant difference on
    the Vina panel: v1 draws the reference exactly there, v2 and v3 draw beyond it and
    fade. And the reference's ROLLED share is the one number that genuinely depends on the
    span, because a moving average is truncated at its ends -- hence one column per arm
    set, empty where that set does not reach."""
    union = sorted(set().union(*(set(S["xs"]) for S in sets.values())))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, union, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], union, stat)
    counts = _vpa_window_counts(ref, union)
    exact = {label: share(sizes, union)[0] for label, sizes
             in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]}
    rolled_ref = {name: dict(zip(S["xs"], rolled(S["exact"][REF_LABEL], S["xs"])))
                  for name, S in sets.items()}
    spans = {name: set(S["xs"]) for name, S in sets.items()}

    header = ["heavy_atoms", "plotted_in"]
    for label, *_ in arms:
        k = _vpa_slug(label)
        header += [f"n_{k}", f"{k}_mean", f"{k}_median", f"{k}_pct"]
    header += ["reference_source", "n_reference_total", "n_reference_exact",
               "n_reference_window", "reference_firm", "reference_mean",
               "reference_median", "reference_pct_exact"]
    header += [f"reference_pct_rolled_{name}" for name in sets]
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    rows = []
    for i, a in enumerate(union):
        row = [a, "+".join(n for n, span in spans.items() if a in span)]
        for label, *_ in arms:
            row += [len(data[label].get(a, ())),
                    fmt(curves[(label, "mean")][i]),
                    fmt(curves[(label, "median")][i]),
                    f"{exact[label][i]:.3f}"]
        row += [source, n_ref, len(ref.get(a, ())), counts[i],
                "yes" if counts[i] >= VPA_REF_FIRM else "no",
                fmt(curves[(REF_LABEL, "mean")][i]),
                fmt(curves[(REF_LABEL, "median")][i]),
                f"{exact[REF_LABEL][i]:.3f}"]
        row += [fmt(rolled_ref[name].get(a)) for name in sets]
        rows.append(row)
    return write_csv(out, "vina_dock_per_atom", header, rows)


@figure("fig-vina-per-atom", needs=(VPA_BASENAME, "crossdocked_reference_dock.json"))
def draw_vina_per_atom(out):
    """Vina Dock against heavy-atom count, over a size-distribution strip — three
    reference variants (gated / faded / strip-free) x mean and median x core and all."""
    data, pockets, arms, missing = {}, {}, [], []
    for arm in VPA_ARMS:
        label, root = arm[0], arm[1]
        sizes, n_pockets = _vpa_load(root, label)
        if sizes is None:
            missing.append((label, os.path.join(root, VPA_BASENAME)))
            continue
        data[label], pockets[label] = sizes, n_pockets
        arms.append(arm)
    absent = [lab for lab in VPA_FOCUS if lab not in data]
    if absent:
        raise FileNotFoundError(
            f"the compared arms are the figure: {', '.join(absent)} has no {VPA_BASENAME}")
    source, ref, ref_label, ref_note, n_ref = _vpa_reference_set()

    use_style()
    print(f"  {VPA_PROTOCOL}")
    print("  arms: " + " · ".join(f"{lab} {pockets[lab]} pockets" for lab, *_ in arms))
    for label, path in missing:
        print(f"    {label} skipped: not docked here, no "
              f"{os.path.relpath(path, REPO)}")
    print(f"  reference: {ref_note}")
    print(f"    drawn as {ref_label!r}")
    for label, by in [(REF_LABEL, ref)] + [(lab, data[lab]) for lab, *_ in arms]:
        vals = [n for n, v in by.items() for _ in v]
        print(f"    size of {label:16s} mean {st.mean(vals):5.2f}  "
              f"median {st.median(vals):5.1f} heavy atoms")

    # The two ends of x are where EVERY arm IN THE SET clears MIN_N, so the sets do not
    # share a span. Inside them x stays contiguous and an arm that dips below is broken by
    # _vpa_model_curve() instead, so one thin patch in one arm no longer truncates the
    # figure for all of them.
    sets = {}
    for aset, subset in _vpa_arm_sets(arms):
        S = sets[aset] = _vpa_resolve(subset, data, ref)
        xs, curves = S["xs"], S["curves"]
        print(f"  {aset}: " + " + ".join(lab for lab, *_ in subset)
              + f" · x = {xs[0]}..{xs[-1]} heavy atoms, all drawn arms >= {MIN_N} "
              f"molecules at both ends; broken inside where an arm is not: "
              + (", ".join(f"{lab} at {v}" for lab, v in S["thin"].items() if v)
                 or "nowhere"))
        for label in [REF_LABEL] + [lab for lab, *_ in subset]:
            print(f"    {label:16s} n={S['totals'][label]:6d}  "
                  f"{sum(S['exact'][label]):.1f}% of its molecules inside the plotted "
                  f"x range")
        for stat in VPA_STATS:
            both = [i for i in range(len(xs))
                    if curves[(VPA_VOX_LABEL, stat)][i] is not None
                    and curves[(VPA_OUR_LABEL, stat)][i] is not None]
            wins = [i for i in both
                    if curves[(VPA_OUR_LABEL, stat)][i] <= curves[(VPA_VOX_LABEL, stat)][i]]
            gap = st.mean([curves[(VPA_OUR_LABEL, stat)][i]
                           - curves[(VPA_VOX_LABEL, stat)][i] for i in both])
            print(f"    {stat:6s}: CoDE lower than VoxBind at {len(wins)}/{len(both)} "
                  f"sizes, mean per-size gap {gap:+.3f} kcal/mol")

    for name, cfg in VPA_VARIANTS.items():
        panels = ("Vina panel and strip" if cfg["ref_dist"] else "Vina panel only")
        tail = ("gated off below it" if cfg["ref_gate"] == "hard" else
                f"faded to alpha {VPA_REF_THIN_ALPHA} below it, down to a window of "
                f"{VPA_REF_GATE[cfg['ref_gate']]}")
        print(f"  {name}: reference on the {panels}, firm at >= {VPA_REF_FIRM} ligands in "
              f"its ±{REF_WIN}-atom window and {tail}; distribution "
              + ("rolled" if cfg["ref_dist_smooth"] else "per-count")
              + (", filled" if cfg["ref_dist_fill"] else ", unfilled"))
        for stat in VPA_STATS:
            for aset in sets:
                print(_vpa_build(out, name, cfg, stat, aset, sets[aset], ref_label))
    _vpa_write_csv(out, sets, arms, data, ref, source, n_ref)
    print("  vina_dock_per_atom.csv -- one table behind all twelve")
