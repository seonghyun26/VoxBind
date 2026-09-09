#!/usr/bin/env python3
"""build_molecular_weight.py — what molecular weights each arm actually produces.

    mw_distribution_{core,all}.*   share of an arm's molecules per 20 Da bin
    mw_ecdf_{core,all}.*           the same distributions, cumulative
    mw_histogram.csv               the plotted bins, counts and shares
    mw_summary.json                n, mean, median, quartiles and tails per arm

WHY THIS IS ITS OWN FOLDER. It is not a quality metric -- nothing here says an arm is
better -- it is the size prior every quality metric in 260910 is read against. Vina score
grows with molecule size, strain grows with size, rigid-fragment RMSD grows with fragment
size. So "which arm wins" is only meaningful once you know whether the arms are drawing
from the same weight distribution, and they are not: see the README.

MOLECULAR WEIGHT COMES FROM THE SMILES `metrics.json` ALREADY RECORDS, through RDKit's
`Descriptors.MolWt` (average mass, implicit hydrogens included). Reading it off the
recorded SMILES rather than re-parsing the SDF gives the same number with no risk of a
sample-to-SDF index slip, and it is the same string every other 260910 figure identifies a
molecule by. Every SMILES in all three arms parses; the run log says so each time.

THE ARMS AND THE POCKET SET ARE ../pose_common.py's -- the 79 electron-density pockets,
the same molecules the pose figures score. TargetDiff carries fewer molecules than the
other two (7,287 against 7,888 and 7,873) because that is what it sampled, not because
anything is filtered here.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-molecular-weight/build_molecular_weight.py
"""
import csv
import json
import os
import statistics as st
import sys

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

RDLogger.DisableLog("rdApp.*")

X_LABEL = "Molecular weight (Da)"
BIN = 20                     # Da; ~52 bins over the occupied range
XMAX = 800                   # the plotted range; the share above it is named in the log
XTICK = 100
# The crystal reference is 79 molecules over ~35 occupied bins -- about two per bin, so its
# raw histogram is a picket fence of 2.5%-tall spikes that says nothing. It is rolled over
# +-REF_ROLL bins and AVERAGED, which keeps it on the same per-bin scale as the models'
# shares, exactly as pose_common.size_distribution rolls it over ligand size.
REF_ROLL = 2

DATA, P79_ROWS, REFROWS = pc.load_arms()


def weights(rows):
    """(molecular weights, count of SMILES that would not parse)."""
    out, bad = [], 0
    for r in rows:
        m = Chem.MolFromSmiles(r["smi"]) if r.get("smi") else None
        if m is None:
            bad += 1
            continue
        out.append(Descriptors.MolWt(m))
    return np.asarray(out), bad


MW = {key: weights(P79_ROWS[key])[0] for _, key, _ in pc.ARMS}
MW_ALL = {key: weights([r for t in sorted(DATA[key]) for r in DATA[key][t]])[0]
          for _, key, _ in pc.ARMS}
MW_REF, BAD_REF = weights(REFROWS)
BAD = {key: weights(P79_ROWS[key])[1] for _, key, _ in pc.ARMS}

EDGES = np.arange(0, XMAX + BIN, BIN)
CENTRES = EDGES[:-1] + BIN / 2


def share(v):
    """Percent of a set's molecules in each bin. Molecules beyond XMAX are NOT folded into
    the last bin -- that would put a spike where the data has a tail -- so the shares sum
    to slightly under 100 and the remainder is reported."""
    counts, _ = np.histogram(v, bins=EDGES)
    return 100 * counts / len(v), counts


def rolled(pct):
    """The reference's shares under a centred +-REF_ROLL-bin window, averaged."""
    k = 2 * REF_ROLL + 1
    pad = np.pad(pct, REF_ROLL, mode="constant")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


# ── figure 1: the distribution ──────────────────────────────────────────────────
def fig_distribution(arms, variant):
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for lab, key, _ in arms:
        pct, _ = share(MW[key])
        col = color(lab)
        ax.step(CENTRES, pct, where="mid", color=col, lw=pc.DIST_LW + 0.35, zorder=3)
        ax.fill_between(CENTRES, pct, step="mid", color=col, alpha=pc.DIST_FILL, lw=0,
                        zorder=2)
    pct, _ = share(MW_REF)
    ax.plot(CENTRES, rolled(pct), color=pc.REF_COLOR, lw=pc.REF_LW, ls=pc.DASH, zorder=4,
            dash_capstyle="round")

    # "% of molecules" is per ARM -- each curve is normalised by its own total, which is
    # the only way a 79-molecule reference and a 7,888-molecule arm share an axis
    pc.furniture(ax, ylabel=f"% of molecules\nper {BIN} Da", xlabel=X_LABEL,
                 xlim=(0, XMAX), xloc=XTICK)
    ax.set_ylim(bottom=0)
    pc.legend(ax, pc.arm_handles(arms), loc="upper right", fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "mw_distribution", variant)


# ── figure 2: the same, cumulative ──────────────────────────────────────────────
def fig_ecdf(arms, variant):
    """The distribution figure answers "where does this arm put its molecules"; this one
    answers "how much of an arm sits below any given weight", which is the form to read a
    shift between two arms off. No binning and no rolling -- the reference's 79 molecules
    are 79 honest steps."""
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def ecdf(v, col, lw, ls):
        x = np.sort(v)
        ax.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=col, lw=lw, ls=ls,
                zorder=4 if ls != "-" else 5, solid_capstyle="round")

    for lab, key, _ in arms:
        ecdf(MW[key], color(lab), pc.MODEL_LW, "-")
    ecdf(MW_REF, pc.REF_COLOR, pc.REF_LW, pc.DASH)

    pc.furniture(ax, ylabel="Cumulative share (%)", xlabel=X_LABEL, xlim=(0, XMAX),
                 xloc=XTICK)
    ax.set_ylim(0, 100)
    pc.legend(ax, pc.arm_handles(arms), loc="lower right", fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "mw_ecdf", variant)


# ── data exports ────────────────────────────────────────────────────────────────
def block(v, bad=0):
    return {
        "n_molecules": int(len(v)), "n_smiles_unparsed": int(bad),
        "mw_mean": round(float(v.mean()), 1),
        "mw_median": round(float(np.median(v)), 1),
        "mw_q25": round(float(np.percentile(v, 25)), 1),
        "mw_q75": round(float(np.percentile(v, 75)), 1),
        "mw_p1": round(float(np.percentile(v, 1)), 1),
        "mw_p99": round(float(np.percentile(v, 99)), 1),
        "mw_min": round(float(v.min()), 1), "mw_max": round(float(v.max()), 1),
        "pct_over_500": round(100 * float((v > 500).mean()), 2),
        f"pct_over_{XMAX}": round(100 * float((v > XMAX).mean()), 2),
    }


def exports():
    summary = {
        "built": "2026-09-09",
        "metric": "molecular weight (RDKit Descriptors.MolWt) of every generated molecule",
        "source": ("the SMILES each target's metrics.json records -- the same string the "
                   "other 260910 figures identify a molecule by"),
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79),
                       "targets": pc.P79},
        "histogram": {"bin_da": BIN, "x_max": XMAX,
                      "reference_roll_bins": REF_ROLL},
        "arms": {}, "reference_ligand": block(MW_REF, BAD_REF),
    }
    for lab, key, root in pc.ARMS:
        summary["arms"][lab] = {
            "root": root, "key": key, "pockets_all": len(DATA[key]),
            "p79": block(MW[key], BAD[key]),
            "all_pockets": block(MW_ALL[key]),
        }
    with open(os.path.join(HERE, "mw_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "mw_histogram.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mw_bin_lo", "mw_bin_hi", "n", "pct"])
        for lab, key, _ in pc.ARMS:
            pct, counts = share(MW[key])
            for lo, n, q in zip(EDGES[:-1], counts, pct):
                w.writerow([lab, lo, lo + BIN, int(n), round(float(q), 3)])
        pct, counts = share(MW_REF)
        for lo, n, q in zip(EDGES[:-1], counts, pct):
            w.writerow([pc.REF_LABEL, lo, lo + BIN, int(n), round(float(q), 3)])
    return summary


def main():
    pc.use_style()
    for variant, arms in pc.variants():
        fig_distribution(arms, variant)
        fig_ecdf(arms, variant)
    summary = exports()

    print(f"{len(pc.P79)} pockets · molecular weight from the recorded SMILES · "
          f"{BIN} Da bins to {XMAX} Da\n")
    print(f"{'arm':16s} {'mols':>6s} {'median':>8s} {'mean':>7s} {'IQR':>15s} "
          f"{'>500 Da':>8s} {f'>{XMAX} Da':>9s} {'unparsed':>9s}")
    for lab in [l for l, _, _ in pc.ARMS] + [pc.REF_LABEL]:
        b = (summary["arms"][lab]["p79"] if lab != pc.REF_LABEL
             else summary["reference_ligand"])
        print(f"{lab:16s} {b['n_molecules']:6d} {b['mw_median']:8.1f} {b['mw_mean']:7.1f} "
              f"{b['mw_q25']:7.1f}–{b['mw_q75']:<7.1f} {b['pct_over_500']:7.2f}% "
              f"{b[f'pct_over_{XMAX}']:8.2f}% {b['n_smiles_unparsed']:9d}")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
