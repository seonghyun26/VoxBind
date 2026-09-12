#!/usr/bin/env python3
"""build_posebusters_figures.py — PoseBusters dock-mode validity, in the 3-line style.

    pb_valid_per_atom_{core,all}.*     validity against ligand size, + the size mix
    pb_check_failures_{core,all}.*     which checks fail, per method
    posebusters_summary.json           coverage + pooled rates (p79 and all_pockets)
    posebusters_by_atom_range.{json,csv}   the per-bin numbers
    posebusters_check_failures.json    per-check failure counts and rates
    posebusters_per_molecule_<arm>.json    per-molecule export, for merging elsewhere

PoseCheck (strain, clashes) is the sibling folder, ../fig-posecheck. The two split by
metric and share ../pose_common.py, which owns the arm list, the pocket set, the loader
and the house furniture -- everything a disagreement between them would silently corrupt.
Each folder carries the code that draws its own figures.

WHAT IS NEW. PoseBusters had only ever run on 33/79 of Ours v1 and 53/100 of vanilla, and
never on TargetDiff; `voxbind/scripts/85_fill_pose_eval_4runs.sh` finished it on
2026-09-09, so this is the first build where every arm carries it on every pocket -- and
the first with the per-check breakdown that says WHY a validity rate is what it is.

SIZE IS THE CONFOUND, SO SIZE IS THE X AXIS. Every check gets harder as the molecule grows
-- more rings to pucker, more angles to strain, more atoms to reach the protein -- and the
arms draw different size mixes (24.9 heavy atoms for Ours v1 against TargetDiff's 22.2). A
pooled rate is therefore partly a report of the size mix, the same trap the Vina numbers
have, which is why the headline figure is per-atom with the size distribution drawn
underneath it rather than a single pooled bar. posebusters_summary.json also carries
`pb_valid_rate_size_standardized`: each arm's validity within every 1-atom stratum,
re-weighted by one common size distribution.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posebusters/build_posebusters_figures.py
"""
import collections
import csv
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color, display                             # noqa: E402

# Validity is a proportion, so its per-count curve is smoothed over +-VALID_WIN atoms --
# see pose_common.model_curve for why the strain and clash curves are not.
VALID_WIN = 2
# A check no method fails above this often is a row of white space in the breakdown: it
# says only that PoseBusters ran it. The full counts stay in the JSON. The filter stays a
# RATE even though the bars are counts -- it asks "is this row informative", and the arms
# hold different numbers of molecules.
MIN_FAIL_PCT = 0.5

DATA, P79_ROWS, REFROWS = pc.load_arms()


# Direct standardization: each arm's rate WITHIN every 1-heavy-atom stratum, re-weighted by
# one common size distribution (every arm pooled), so what is left is validity at matched
# size. Strata where the arm holds <10 molecules are dropped, not extrapolated. This is the
# number to compare arms with; the crude rate is what the arm actually produced, and the
# two answer different questions.
# The weights come from the arms that are actually IN the comparison, not from every key
# the loader returned: a scoring run in progress leaves partial rows in P79_ROWS, and
# folding those into the standard population moves every arm's standardized rate.
STD_W = collections.Counter(
    r["n"] for _, key, _ in pc.arms_for("v", DATA) for r in P79_ROWS[key]
    if r["v"] is not None)


def std_rate(rows):
    per = pc.by_size(rows, "v")
    num = den = 0.0
    for n, w in STD_W.items():
        if len(per.get(n, ())) >= 10:
            num += w * float(np.mean(per[n])); den += w
    return 100 * num / den if den else float("nan")


# ── figure 1: validity per heavy-atom count, over each arm's size distribution ───
def fig_valid_per_atom(arms, variant):
    per = {key: pc.by_size(P79_ROWS[key], "v") for _, key, _ in arms}
    ref_per = pc.by_size(REFROWS, "v")
    xs = pc.x_range(per, arms)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(pc.FIG_W, pc.STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": pc.HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    rate100 = lambda v: 100 * float(np.mean(v))
    top.plot(xs, pc.reference_curve(ref_per, xs, rate100), color=pc.REF_COLOR,
             lw=pc.REF_LW, ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, pc.model_curve(per[key], xs, rate100, win=VALID_WIN),
                 color=color(lab), lw=pc.MODEL_LW, zorder=5, solid_capstyle="round")
    pc.furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    pc.legend(top, pc.arm_handles(arms), loc="lower left", fontsize=11.5)

    pc.size_distribution(bot, xs, per, ref_per, arms)
    pc.furniture(bot, ylabel="% of ligands", xlabel=pc.X_LABEL,
                 xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    fig.align_ylabels((top, bot))
    pc.fit(fig, pad=0.5, h_pad=pc.H_PAD)
    pc.save(fig, HERE, "pb_valid_per_atom", variant)
    return xs


# ── figure 2: which checks fail ─────────────────────────────────────────────────
def check_failures():
    """{key: {n_mols, counts, rates}} over every arm, plus the crystal ligands."""
    out = {}
    for lab, key, _ in pc.arms_for("v", DATA):
        rows = [r for r in P79_ROWS[key] if r["v"] is not None]
        cnt = collections.Counter(c for r in rows for c in r["f"])
        out[key] = {"n_mols": len(rows), "counts": dict(cnt),
                    "rates": {k: 100 * v / len(rows) for k, v in cnt.items()}}
    rrows = [r for r in REFROWS if r["v"] is not None]
    cnt = collections.Counter(c for r in rrows for c in r["f"])
    out["reference"] = {"n_mols": len(rrows), "counts": dict(cnt),
                        "rates": {k: 100 * v / max(1, len(rrows)) for k, v in cnt.items()}}
    return out


def wrap_check(name):
    """The PoseBusters check name, wrapped instead of abbreviated.

    `non-aromatic_ring_non-flatness` on one line is 30 characters and was eating 2.9 in of
    a 7.6 in figure -- nearly half the width -- as a tick label. Abbreviating it is the
    wrong fix: this check passes when a non-aromatic ring is sufficiently NON-flat (it is
    `check_nonflat: True` in dock.yml, threshold 0.1 A), so failing it means a saturated
    ring came out planar, and every shortening of that name I tried either flipped its
    sense or read as the aromatic check next to it. Wrapping is free and exact."""
    return "\n".join(textwrap.wrap(name.replace("_", " "), 18))


def fig_check_failures(arms, variant, fails):
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
    series = [(pc.REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    shown = [fails[key] for _, key in series]
    names = sorted({k for g in shown for k in g["counts"]
                    if max(h["rates"].get(k, 0) for h in shown) >= MIN_FAIL_PCT},
                   key=lambda k: -max(g["rates"].get(k, 0) for g in shown))
    # One row of the y axis is 1.0 apart, so the bars of a group must fit inside that:
    # a fixed height works for three series and silently overlaps the neighbouring groups
    # at nine (9 x 0.19 = 1.71), which reads as bars detached from their labels. Derive it.
    h = 0.86 / len(series)
    # THE KEY GOES OUTSIDE THE AXES, bottom left, 3 x 3. There is no empty corner inside:
    # the long bars fill the top and the right, and the in-axes box this used to carry
    # reached far enough left to bury the bottom rows' bars -- `double bond flatness`
    # looked empty while AR, DecompDiff and Pocket2Mol were failing it 3.1, 1.8 and 1.4% of
    # the time. Outside it covers nothing. Three columns only fit because the bars are
    # rates now: an arm's own n no longer has to be in the key for its bar to be readable.
    ncol = 3
    leg_rows = -(-len(series) // ncol)
    # Row pitch: enough that nine thin bars stay readable, without turning a 7.6 in wide
    # figure into a 10 in tall one. Plus the strip the key needs at the bottom.
    fig, ax = plt.subplots(
        figsize=(pc.FIG_W, (0.40 if len(arms) <= 3 else 0.62) * len(names)
                 + 1.6 + 0.30 * leg_rows + 0.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(len(names))[::-1]
    for i, (lab, key) in enumerate(series):
        # top of the group downwards, so the key's order IS the order of the bars
        ax.barh(ys + ((len(series) - 1) / 2 - i) * h,
                [fails[key]["rates"].get(k, 0.0) for k in names],
                height=h, color=color(lab), edgecolor=color(lab), lw=0.8, zorder=3)

    top = max(g["rates"].get(k, 0.0) for g in shown for k in names)
    pc.furniture(ax, ylabel=None, xlabel="Molecules failing the check (%)",
                 xlim=(0, top * 1.03), xloc=None)
    ax.grid(False, axis="y")
    ax.set_yticks(ys)
    ax.set_yticklabels([wrap_check(k) for k in names], fontsize=12)
    ax.set_ylim(-0.6, len(names) - 0.4)
    handles = [Patch(facecolor=color(lab), edgecolor=color(lab),
                     label=display(lab) + ("  (n=79)" if key == "reference" else ""))
               for lab, key in series]
    pc.fit(fig, pad=0.5)
    # tight_layout does not see a FIGURE legend, so it has just laid the axes out over the
    # strip the key occupies. Place the key, measure what it actually took, and RAISE the
    # axes by that much -- reserving the measured height keeps the tick labels and the x
    # label, which sit BELOW the axes box and would otherwise end up behind an opaque
    # legend; reserving up to its top edge instead would bury them. Columns are dropped
    # if the key overruns the figure, which is cheaper than trusting a width estimate.
    fs = 12.5 if len(arms) <= 3 else 11
    for c in range(ncol, 0, -1):
        leg = pc.legend(fig, handles, loc="lower left", ncol=c, fontsize=fs,
                        bbox_to_anchor=(0.008, 0.008))
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        if bb.x1 <= 0.997 or c == 1:
            break
        leg.remove()
    fig.subplots_adjust(bottom=min(0.6, fig.subplotpars.bottom
                                   + (bb.y1 - bb.y0) + 0.016))
    pc.save(fig, HERE, "pb_check_failures", variant)


# ── data exports ────────────────────────────────────────────────────────────────
def block(rows):
    v = [r["v"] for r in rows if r["v"] is not None]
    std = std_rate(rows)
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_posebusters": len(v),
        "pb_valid_rate": round(pc.rate(v) / 100, 4) if v else None,
        "pb_valid_rate_size_standardized": round(std / 100, 4) if np.isfinite(std) else None,
    }


def exports(fails):
    summary = {
        "built": "2026-09-09", "metric": "PoseBusters dock-mode validity",
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79),
                       "targets": pc.P79},
        "note": ("p79 is the like-for-like set every arm covers; all_pockets is each arm's "
                 "own coverage and is NOT comparable across arms."),
        "arms": {}, "reference_ligand": block(REFROWS),
    }
    for lab, key, root in pc.arms_for("v", DATA):
        pockets = sorted(DATA[key])
        summary["arms"][lab] = {
            "root": root, "key": key, "pockets_all": len(pockets),
            "pockets_p79": len([t for t in pc.P79 if t in DATA[key]]),
            "p79": block(P79_ROWS[key]),
            "all_pockets": block([r for t in pockets for r in DATA[key][t]]),
        }
    json.dump(summary, open(os.path.join(HERE, "posebusters_summary.json"), "w"),
              indent=1, ensure_ascii=False)

    by_bin = {"bins": pc.BIN_LABELS, "edges": pc.EDGES[:-1] + ["inf"],
              "n_pockets": len(pc.P79), "arms": {}}
    for lab, key, _ in pc.arms_for("v", DATA):
        by_bin["arms"][lab] = [block([r for r in P79_ROWS[key] if pc.bin_of(r["n"]) == b])
                               for b in range(len(pc.BIN_LABELS))]
    by_bin["arms"][pc.REF_LABEL] = [
        block([r for r in REFROWS if pc.bin_of(r["n"]) == b])
        for b in range(len(pc.BIN_LABELS))]
    json.dump(by_bin, open(os.path.join(HERE, "posebusters_by_atom_range.json"), "w"),
              indent=1, ensure_ascii=False)

    cols = ["n_molecules", "atoms_mean", "n_posebusters", "pb_valid_rate",
            "pb_valid_rate_size_standardized"]
    with open(os.path.join(HERE, "posebusters_by_atom_range.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "bin"] + cols)
        for arm, rowset in by_bin["arms"].items():
            for lab, r in zip(pc.BIN_LABELS, rowset):
                w.writerow([arm, lab] + [r[c] for c in cols])

    json.dump({"n_pockets": len(pc.P79),
               "arms": {lab: fails[key] for lab, key, _ in pc.arms_for("v", DATA)},
               "reference_ligand": fails["reference"]},
              open(os.path.join(HERE, "posebusters_check_failures.json"), "w"),
              indent=1, ensure_ascii=False)

    for lab, key, root in pc.arms_for("v", DATA):
        mols = [{"t": t, "n": r["n"], "v": r["v"], "f": r["f"]}
                for t in sorted(DATA[key]) for r in DATA[key][t]]
        json.dump({"arm": lab, "root": root, "n_pockets": len(DATA[key]),
                   "n_molecules": len(mols), "p79_targets": pc.P79,
                   "fields": {"t": "target dir", "n": "heavy atoms",
                              "v": "PoseBusters valid", "f": "failed checks"},
                   "molecules": mols},
                  open(os.path.join(HERE, f"posebusters_per_molecule_{key}.json"), "w"),
                  ensure_ascii=False)
    return summary, by_bin


def main():
    pc.use_style()
    fails = check_failures()
    ranges = {}
    for variant, arms in pc.variants("v", DATA):
        ranges[variant] = fig_valid_per_atom(arms, variant)
        fig_check_failures(arms, variant, fails)
    summary, by_bin = exports(fails)

    print(f"79-pocket set · {len(pc.P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (counts where every drawn arm has >={pc.MIN_N} molecules)\n")
    print(f"{'arm':16s} {'atoms':>6s} {'PB-valid':>9s} {'size-std':>9s} {'mols':>7s}")
    for lab, key, _ in pc.arms_for("v", DATA):
        r = summary["arms"][lab]["p79"]
        print(f"{lab:16s} {r['atoms_mean']:6.1f} {100 * r['pb_valid_rate']:8.1f}% "
              f"{100 * r['pb_valid_rate_size_standardized']:8.1f}% {r['n_molecules']:7d}")
    r = summary["reference_ligand"]
    print(f"{pc.REF_LABEL:16s} {r['atoms_mean']:6.1f} {100 * r['pb_valid_rate']:8.1f}% "
          f"{'—':>9s} {r['n_molecules']:7d}")
    print(f"\nPB-valid by bin ({' · '.join(pc.BIN_LABELS)}):")
    for lab, key, _ in pc.arms_for("v", DATA):
        print(f"  {lab:16s} " + "  ".join(
            f"{100 * b['pb_valid_rate']:5.1f}%" if b["pb_valid_rate"] is not None else "    -"
            for b in by_bin["arms"][lab]))
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
