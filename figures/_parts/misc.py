

# ════════════════════════════════════════════════════════════════════════════════
# fig-molweight-{distribution,ecdf} — the size prior every other figure is read against
# ════════════════════════════════════════════════════════════════════════════════
# NOT a quality metric -- nothing here says an arm is better -- it is the size prior every
# quality metric in 260910 is read against. Vina score grows with molecule size, strain
# grows with size, rigid-fragment RMSD grows with fragment size. So "which arm wins" is
# only meaningful once you know whether the arms are drawing from the same weight
# distribution, and they are not.
#
# MOLECULAR WEIGHT IS READ STRAIGHT FROM `target_*/samples.sdf`, through RDKit's
# `Descriptors.MolWt` (average mass, implicit hydrogens included).
#
# It used to come from the SMILES in each target's `metrics.json`, and that had to change
# to cover the five published baselines: their sample dirs are staged and complete, but
# their `metrics.json` files are written by the PoseBusters scoring run and appear as it
# finishes each pocket. Keying a size prior to the progress of a scoring run is the wrong
# dependency -- the molecules exist either way -- so this reads the molecules. The two
# agree: over the three arms that have both, SDF and SMILES give the same weight for every
# molecule (the load prints the check), because the recorded SMILES was derived from the
# same SDF entry.
#
# THE ARMS AND THE POCKET SET ARE THE SHARED ONES -- the 79 electron-density pockets. Arms
# hold different numbers of molecules (7,287 for TargetDiff against 7,888 for VoxBind)
# because that is what each sampled; nothing is filtered here. An arm is drawn only if it
# has a samples.sdf for all 79 pockets, which is the same rule arms_for() applies to
# metrics -- an arm mid-stage must not become a curve over part of the data.
MW_X_LABEL = "Molecular weight (Da)"
MW_BIN = 20                  # Da; ~52 bins over the occupied range
MW_XMAX = 800                # the plotted range; the share above it is named in the log
MW_XTICK = 100
# The crystal reference is 79 molecules over ~35 occupied bins -- about two per bin, so its
# raw histogram is a picket fence of 2.5%-tall spikes that says nothing. It is rolled over
# +-MW_REF_ROLL bins and AVERAGED, which keeps it on the same per-bin scale as the models'
# shares, exactly as size_distribution() rolls the reference over ligand size.
MW_REF_ROLL = 2
MW_EDGES = np.arange(0, MW_XMAX + MW_BIN, MW_BIN)
MW_CENTRES = MW_EDGES[:-1] + MW_BIN / 2


def _mw_sdf_weights(paths):
    """(molecular weights, unsanitisable entries, disconnected entries dropped).

    DISCONNECTED MOLECULES ARE DROPPED, because notebook/webapp/metrics.py drops them
    before scoring anything else -- so keeping them would put this figure on a different
    molecule set than the strain, clash and PoseBusters figures for the same run. It
    matters for exactly one arm: TargetDiff emits 511 of 7,798 (6.6%) over the 79 pockets
    and every other arm emits none. Their weight is also not a ligand's weight -- it is
    the sum of two or more pieces that never bonded.

    RDKit is imported HERE, not at the top of draw.py: it is the only figure-specific
    dependency in the file, and a top-level import would make every other figure -- and
    --list -- refuse to run on a box that has no rdkit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    out, bad, disc = [], 0, 0
    for path in paths:
        for m in Chem.SDMolSupplier(path):
            if m is None:
                bad += 1
                continue
            if len(Chem.GetMolFrags(m)) > 1:
                disc += 1
                continue
            out.append(Descriptors.MolWt(m))
    return np.asarray(out), bad, disc


def _mw_samples(root, targets=None):
    """The samples.sdf under `root`, over `targets` or over every target it holds."""
    names = sorted(d for d in os.listdir(root) if d.startswith("target_")) \
        if targets is None else targets
    paths = [os.path.join(root, t, "samples.sdf") for t in names]
    return [p for p in paths if os.path.exists(p)]


def _mw_reference_paths():
    """The deposited ligand in each of the 79 pockets: the one SDF in the target dir that
    is not samples.sdf. Every arm that has one carries the same molecule, so this reads
    ours -- the staged baseline dirs hold no reference at all."""
    out = []
    for t in P79:
        d = os.path.join(REF_ROOT, t)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".sdf") and f != "samples.sdf"]
    return out


def _mw_share(v):
    """Percent of a set's molecules in each bin. Molecules beyond MW_XMAX are NOT folded
    into the last bin -- that would put a spike where the data has a tail -- so the shares
    sum to slightly under 100 and the remainder is reported."""
    counts, _ = np.histogram(v, bins=MW_EDGES)
    return 100 * counts / len(v), counts


def _mw_rolled(pct):
    """The reference's shares under a centred +-MW_REF_ROLL-bin window, averaged."""
    k = 2 * MW_REF_ROLL + 1
    pad = np.pad(pct, MW_REF_ROLL, mode="constant")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _mw_block(v, bad=0, disc=0):
    """n, centre, spread and tails for one set -- what the load summary prints."""
    return {
        "n_molecules": int(len(v)), "n_sdf_unsanitisable": int(bad),
        "n_disconnected_dropped": int(disc),
        "mw_mean": round(float(v.mean()), 1),
        "mw_median": round(float(np.median(v)), 1),
        "mw_q25": round(float(np.percentile(v, 25)), 1),
        "mw_q75": round(float(np.percentile(v, 75)), 1),
        "mw_p1": round(float(np.percentile(v, 1)), 1),
        "mw_p99": round(float(np.percentile(v, 99)), 1),
        "mw_min": round(float(v.min()), 1), "mw_max": round(float(v.max()), 1),
        "pct_over_500": round(100 * float((v > 500).mean()), 2),
        f"pct_over_{MW_XMAX}": round(100 * float((v > MW_XMAX).mean()), 2),
    }


def _mw_smiles_crosscheck(arms, mw):
    """Molecules whose recorded SMILES gives a different weight than their SDF entry, over
    the arms that have metrics.json. Reported, never silently trusted: this builder changed
    source and the check is what says the change was free."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _, p79_rows, _ = pose_data()
    out = {}
    for lab, key, root in arms:
        smi = [r["smi"] for r in p79_rows[key] if r.get("smi")]
        if not smi:
            continue
        sdf = mw[key]
        if len(smi) != len(sdf):
            out[lab] = (f"{len(smi)} SMILES vs {len(sdf)} SDF — not comparable; "
                        f"metrics.json is still being written for this arm")
            continue
        mism = sum(1 for a, b in zip(smi, sdf)
                   if abs(Descriptors.MolWt(Chem.MolFromSmiles(a)) - b) > 0.05)
        out[lab] = f"{mism} of {len(sdf)} differ"
    return out


_MW_CACHE = {}


def _mw_data():
    """(arms that have samples over all 79 pockets, {key: weights}, reference weights).

    Cached for the life of the process: the distribution and the ECDF are the same ~60k
    molecules read out of ~630 SDF files, and drawing both used to mean reading them twice.
    The per-arm summary and the SMILES cross-check print from HERE rather than from a draw
    function, so they appear once per run however many of the two figures are asked for.

    The original also built a weights-over-EVERY-pocket set; it fed only mw_summary.json,
    which the recompute script still owns, so it is not built here."""
    if "d" not in _MW_CACHE:
        t0 = time.time()
        # An arm is in only if it holds a samples.sdf for every one of the 79 pockets.
        arms = [a for a in ARMS if len(_mw_samples(a[2], P79)) == len(P79)]
        missing = [a[0] for a in ARMS if a not in arms]
        mw, bad, disc = {}, {}, {}
        for _, key, root in arms:
            mw[key], bad[key], disc[key] = _mw_sdf_weights(_mw_samples(root, P79))
        mw_ref, bad_ref, _ = _mw_sdf_weights(_mw_reference_paths())
        _MW_CACHE["d"] = (arms, mw, mw_ref)

        n = sum(len(v) for v in mw.values())
        print(f"  [read {n:,} molecules from {len(arms)} arms' samples.sdf "
              f"in {time.time()-t0:.1f}s]")
        print(f"  {len(P79)} pockets · molecular weight from target_*/samples.sdf, "
              f"single-component molecules only · {MW_BIN} Da bins to {MW_XMAX} Da\n")
        print(f"  {'arm':16s} {'mols':>6s} {'median':>8s} {'mean':>7s} {'IQR':>15s} "
              f"{'>500 Da':>8s} {f'>{MW_XMAX} Da':>9s} {'disconn.':>9s}")
        blocks = {lab: _mw_block(mw[key], bad[key], disc[key]) for lab, key, _ in arms}
        blocks[REF_LABEL] = _mw_block(mw_ref, bad_ref)
        for lab in [l for l, _, _ in arms] + [REF_LABEL]:
            b = blocks[lab]
            print(f"  {lab:16s} {b['n_molecules']:6d} {b['mw_median']:8.1f} "
                  f"{b['mw_mean']:7.1f} {b['mw_q25']:7.1f}–{b['mw_q75']:<7.1f} "
                  f"{b['pct_over_500']:7.2f}% {b[f'pct_over_{MW_XMAX}']:8.2f}% "
                  f"{b['n_disconnected_dropped']:9d}")
        if missing:
            print(f"\n  not drawn — no samples.sdf over all {len(P79)} pockets: "
                  + ", ".join(missing))
        print("\n  SDF vs recorded SMILES, molecules whose weight differs by >0.05 Da:")
        for lab, msg in _mw_smiles_crosscheck(arms, mw).items():
            print(f"    {lab:16s} {msg}")
    return _MW_CACHE["d"]


def _mw_variants(arms):
    """(name, arms) for each figure variant, core first -- variants() over the arms that
    have samples, which is a different list from the arms that have a given metric."""
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


@figure("fig-molweight-distribution", needs=("target_*/samples.sdf",))
def draw_molweight_distribution(out):
    """Share of an arm's molecules per 20 Da bin, core and all arms."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Filled steps read well for the two or three arms of `core`; at nine they stack
        # into one opaque mass and the arm you are looking for is the one you cannot see.
        # So the fill is dropped once the figure carries more than the core arms, and the
        # step lines alone do the work -- the same reason the posecheck family draws its
        # eight-method figures as unfilled curves.
        fill = len(arms) <= len(CORE)
        for lab, key, _ in arms:
            pct, _ = _mw_share(mw[key])
            col = color(lab)
            ax.step(MW_CENTRES, pct, where="mid", color=col, lw=DIST_LW + 0.35, zorder=3)
            if fill:
                ax.fill_between(MW_CENTRES, pct, step="mid", color=col, alpha=DIST_FILL,
                                lw=0, zorder=2)
        pct, _ = _mw_share(mw_ref)
        ax.plot(MW_CENTRES, _mw_rolled(pct), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4,
                dash_capstyle="round")

        # "% of molecules" is per ARM -- each curve is normalised by its own total, which
        # is the only way a 79-molecule reference and a 7,888-molecule arm share an axis
        furniture(ax, ylabel=f"% of molecules\nper {MW_BIN} Da", xlabel=MW_X_LABEL,
                  xlim=(0, MW_XMAX), xloc=MW_XTICK)
        ax.set_ylim(bottom=0)
        legend(ax, arm_handles(arms), loc="upper right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_distribution_{variant}")

    rows = []
    for lab, key, _ in all_arms:
        pct, counts = _mw_share(mw[key])
        rows += [[lab, lo, lo + MW_BIN, int(n), round(float(q), 3)]
                 for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    pct, counts = _mw_share(mw_ref)
    rows += [[REF_LABEL, lo, lo + MW_BIN, int(n), round(float(q), 3)]
             for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    write_csv(out, "mw_histogram", ["arm", "mw_bin_lo", "mw_bin_hi", "n", "pct"], rows)


@figure("fig-molweight-ecdf", needs=("target_*/samples.sdf",))
def draw_molweight_ecdf(out):
    """The distribution figure answers "where does this arm put its molecules"; this one
    answers "how much of an arm sits below any given weight", which is the form to read a
    shift between two arms off. No binning and no rolling -- the reference's 79 molecules
    are 79 honest steps."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        def ecdf(v, col, lw, ls):
            x = np.sort(v)
            ax.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=col, lw=lw, ls=ls,
                    zorder=4 if ls != "-" else 5, solid_capstyle="round")

        for lab, key, _ in arms:
            ecdf(mw[key], color(lab), MODEL_LW, "-")
        ecdf(mw_ref, REF_COLOR, REF_LW, DASH)

        furniture(ax, ylabel="Cumulative share (%)", xlabel=MW_X_LABEL, xlim=(0, MW_XMAX),
                  xloc=MW_XTICK)
        ax.set_ylim(0, 100)
        legend(ax, arm_handles(arms), loc="lower right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_ecdf_{variant}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size and fig-ensemble-* — drawn under matplotlib's own defaults
# ════════════════════════════════════════════════════════════════════════════════
# These three figures predate 260910's house style and never applied it: each ran as its own
# process, under stock rcParams. draw.py is ONE process, so a figure drawn after a styled
# one would silently inherit 15 pt DejaVu in warm near-black, and one drawn BEFORE a styled
# one would leave the style stripped for it. MISC_PLAIN_RC is rcParams as they stood when
# draw.py was imported -- before any use_style() can have run -- which is exactly the
# fresh-process state the originals drew in, and it is applied through rc_context so the
# reset lasts only as long as the figure. `backend` is left out: restoring it through
# rcParams is not how a backend is set, and rc_context drops it for the same reason.
MISC_PLAIN_RC = {k: v for k, v in plt.rcParams.copy().items() if k != "backend"}


def _misc_savefig(fig, out, stems, **kw):
    """save() writes png+svg+pdf at the figure's own dpi and no bbox; these families set
    their own dpi and a tight bbox and ship a different pair of formats, and changing that
    changes the image. So they keep their original savefig call."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for path in stems:
        fig.savefig(out / path, **kw)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size — does more MCP fine-tuning buy binding, or just size?
# ════════════════════════════════════════════════════════════════════════════════
# THE ARGUMENT. Vina rewards size, so an arm that draws smaller peptides is penalised and
# an arm that draws larger ones is flattered, regardless of whether it binds better. Binning
# by heavy-atom count removes that: inside a bin every arm draws the same-sized peptides, so
# a gap that survives the bin is a binding difference. This is §4 of the 260827 note applied
# to the MCP family, and it is the reason the pooled numbers in the tables cannot be read on
# their own.
#
# Colours come from the shared palette, where the four arms are registered as an ordinal
# ramp -- they are one model at four amounts of fine-tuning, not four methods.
MCP_FB = str(FUNCBIND / "artifacts" / "reproduction" / "mcpp")
# Four steps of one brown ramp separate cleanly in the bars, but the top panel's lines
# cross, and at a crossing two adjacent steps of a lightness ladder are genuinely hard to
# tell apart. So each arm carries a second identity channel -- line style -- the same way
# this note's published baselines do. The two solid lines are the lightest and the darkest
# step, which is the pair that never needed the help.
MCP_ARMS = [
    ("vanilla",        "FuncBind vanilla",  f"{MCP_FB}/cmp10/_eval/vanilla/eval_docking_results.json",      "-"),
    ("fine-tune 3.17M", "ft_3.17M",         f"{MCP_FB}/cmp10/_eval/finetuned/eval_docking_results.json",    (0, (1, 1.6))),
    ("fine-tune 8.21M", "ft_8.21M",         f"{MCP_FB}/cmp10_r10/_eval/finetuned/eval_docking_results.json", (0, (5, 2.2))),
    ("fine-tune 26.1M", "ft_26.1M",         f"{MCP_FB}/cmp10_r14/_eval/finetuned/eval_docking_results.json", "-"),
]
MCP_BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 89)]
MCP_LBL = ["20–29", "30–39", "40–49", "50–59", "60+"]
MCP_MIN_N = 5      # fewer molecules than this in a bin is a data point, not a trend


@figure("fig-mcp-finetune-size", needs=("mcpp/*/_eval/*/eval_docking_results.json",))
def draw_mcp_finetune_size(out):
    """Median Vina Dock per heavy-atom bin (top) over the molecule count each arm puts in
    that bin (bottom)."""
    d = {lab: json.load(open(p))["per_target"] for lab, _, p, _ in MCP_ARMS}
    # Only targets every arm produced molecules for: comparing an arm against a target where
    # another arm returned nothing is not a comparison.
    common = sorted(set.intersection(*[
        {e["target"] for e in per if e.get("vina_dock") is not None} for per in d.values()]))

    def mols(lab):
        return [(m["n_atoms"], m["vina_dock"])
                for e in d[lab] if e["target"] in common
                for m in (e.get("per_mol") or [])
                if m.get("vina_dock") is not None and m.get("n_atoms")]

    m = {lab: mols(lab) for lab, _, _, _ in MCP_ARMS}
    binned = {lab: [[v for a, v in vals if lo <= a <= hi] for lo, hi in MCP_BINS]
              for lab, vals in m.items()}

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 7.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.13))
        x = np.arange(len(MCP_BINS))

        for lab, key, _, style in MCP_ARMS:
            col = color(key)
            ys = [np.median(b) if len(b) >= MCP_MIN_N else np.nan for b in binned[lab]]
            ax.plot(x, ys, color=col, lw=2.0, ls=style, marker="o", ms=8.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.6, label=lab)
            # An arm that runs out of molecules early gets its label above the point; to the
            # right, the arms that continue would run through it.
            last = max(i for i, y in enumerate(ys) if not np.isnan(y))
            kw = dict(xytext=(10, -2), ha="left") if last == len(MCP_BINS) - 1 \
                else dict(xytext=(0, 23), ha="center")
            ax.annotate(lab, (last, ys[last]), textcoords="offset points",
                        fontsize=10.2, color=col, fontweight="600", va="center", **kw)

        ax.set_ylabel("Vina Dock, median\n(kcal/mol)", fontsize=11.3)
        ax.grid(color="#e6e9ef", lw=.85)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10.0, ncols=4, loc="lower center",
                  bbox_to_anchor=(.5, 1.01))
        ax.set_xlim(-.45, len(MCP_BINS) - .55)
        ax.margins(y=.20)

        w = .21
        for i, (lab, key, _, _s) in enumerate(MCP_ARMS):
            counts = [len(b) for b in binned[lab]]
            bx.bar(x + (i - 1.5) * w, counts, width=w * .86, color=color(key), linewidth=0)
            for xi, c in enumerate(counts):
                if c:
                    bx.annotate(str(c), (xi + (i - 1.5) * w, c), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=7.8, color="#7a8699")

        bx.set_ylabel("molecules", fontsize=11.3)
        bx.set_xlabel("heavy atoms", fontsize=11.3)
        bx.set_xticks(x)
        bx.set_xticklabels(MCP_LBL)
        bx.grid(axis="y", color="#e6e9ef", lw=.85)
        bx.set_axisbelow(True)

        for a in (ax, bx):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color("#c4cad4")
            a.tick_params(labelsize=10.2, colors="#5b6678")

        _misc_savefig(fig, out, [f"mcp_finetune_size.{ext}" for ext in ("png", "svg")],
                      dpi=170, bbox_inches="tight", facecolor="white")

    write_csv(out, "mcp_finetune_size", ["arm", "bin", "n", "dock_median"],
              [[lab, l, len(b), f"{np.median(b):.3f}" if len(b) >= MCP_MIN_N else ""]
               for lab, _, _, _ in MCP_ARMS for l, b in zip(MCP_LBL, binned[lab])])

    print("  common targets:", len(common), common)
    for lab, _, _, _ in MCP_ARMS:
        ha = [a for a, _ in m[lab]]
        print(f"  {lab:17s} n={len(m[lab]):4d} median heavy={np.median(ha):5.1f} "
              f"bins={[len(b) for b in binned[lab]]} "
              f"medians={[round(float(np.median(b)), 2) if len(b) >= MCP_MIN_N else None for b in binned[lab]]}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-ensemble-{6set-3metric,vs-alone} — probe-side feature ensembles
# ════════════════════════════════════════════════════════════════════════════════
# Does bolting a frozen baseline encoder onto the champion's features buy anything the
# champion does not already have? `fig1` is six cohorts x three metrics for the champion
# and for champion+partner; `fig2` asks the sharper question -- is the ensemble better than
# the PARTNER alone, or is the partner simply carrying it?
#
# ensemble_results.json is produced by notebook/html/260910/fig-ensemble/gen_ensemble_data.py,
# which reads feature .pt files that live on the box the probe ran on; it is a recompute
# script, not a drawing script, and it stays there.
#
# These are exploratory panels: bold suptitles, per-panel titles, matplotlib's own tab
# colours. They are deliberately NOT restyled into the house style -- they are not §1/§2
# figures and were never drawn as such.
ENS_COH = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
ENS_COHL = ["FULL", "CL3", "CL3\nID60", "CL3\nID30", "CASF\nnontrain", "CASF\nclean"]
ENS_PARTNERS = ["GET", "EGNN", "CheapNet", "ProFSA"]
ENS_COLORS = {"GET": "#1f77b4", "EGNN": "#2ca02c", "CheapNet": "#ff7f0e", "ProFSA": "#d62728"}
ENS_METRICS = [("r", "Pearson r", False), ("rho", "Spearman ρ", False), ("rmse", "RMSE", True)]


def _ens_results():
    return json.load(open(legacy("fig-ensemble", "ensemble_results.json")))


def _ens_series(d, metric):
    return [d.get(c, {}).get(metric, np.nan) for c in ENS_COH]


@figure("fig-ensemble-6set-3metric", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_6set_3metric(out):
    """Six cohorts x three metrics: champion baseline + champion⊕{GET,EGNN,CheapNet,ProFSA}."""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, (mk, ml, lower_better) in zip(axes, ENS_METRICS):
            ax.plot(x, _ens_series(r["champion"], mk), "-o", color="black", lw=2.6, ms=7,
                    label="champion (C+D+G)", zorder=5)
            for p in ENS_PARTNERS:
                ax.plot(x, _ens_series(r["ensembles"][p], mk), "--o", color=ENS_COLORS[p],
                        lw=1.8, ms=5, label=f"⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=9)
            ax.set_title(ml + ("  (↓ better)" if lower_better else "  (↑ better)"),
                         fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.axvspan(3.5, 5.5, color="grey", alpha=0.06)      # shade CASF holdout region
            if mk == "r":
                ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
        fig.suptitle("Probe-side feature ensemble: champion ⊕ baseline encoder (PCA-64), "
                     f"5-seed, n_shared={r['n_shared']}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _misc_savefig(fig, out,
                      [f"fig1_ensemble_6set_3metric.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig1")

    print("\n  Δρ (ensemble − champion) per cohort:")
    print(f"  {'cohort':<15}" + "".join(f"{p:>10}" for p in ENS_PARTNERS))
    for c in ENS_COH:
        row = f"  {c:<15}"
        for p in ENS_PARTNERS:
            d = r["ensembles"][p].get(c, {}).get("rho", np.nan) \
                - r["champion"].get(c, {}).get("rho", np.nan)
            row += f"{d:>+10.3f}"
        print(row)


@figure("fig-ensemble-vs-alone", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_vs_alone(out):
    """Spearman rho, champion vs partner-ALONE vs ensemble per cohort — does the ensemble
    beat the partner alone?"""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
        for ax, p in zip(axes2, ENS_PARTNERS):
            w = 0.27
            ax.bar(x - w, _ens_series(r["champion"], "rho"), w, color="black", label="champion")
            ax.bar(x, _ens_series(r["alone"][p], "rho"), w, color=ENS_COLORS[p], alpha=0.45,
                   label=f"{p} alone")
            ax.bar(x + w, _ens_series(r["ensembles"][p], "rho"), w, color=ENS_COLORS[p],
                   label=f"champion ⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=8)
            ax.set_title(p, fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(0.40, 0.75)
            ax.legend(fontsize=8, loc="lower left")
        axes2[0].set_ylabel("Spearman ρ", fontsize=11)
        fig2.suptitle("Is the ensemble better than the partner ALONE?  (Spearman ρ per cohort)",
                      fontsize=13, fontweight="bold")
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        _misc_savefig(fig2, out, [f"fig2_ensemble_vs_alone.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig2")
