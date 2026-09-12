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

MOLECULAR WEIGHT IS READ STRAIGHT FROM `target_*/samples.sdf`, through RDKit's
`Descriptors.MolWt` (average mass, implicit hydrogens included).

It used to come from the SMILES in each target's `metrics.json`, and that had to change to
cover the five published baselines: their sample dirs are staged and complete, but their
`metrics.json` files are written by the PoseBusters scoring run and appear as it finishes
each pocket. Keying a size prior to the progress of a scoring run is the wrong dependency
-- the molecules exist either way -- so this reads the molecules. The two agree: over the
three arms that have both, SDF and SMILES give the same weight for every molecule (the run
log prints the check), because the recorded SMILES was derived from the same SDF entry.

THE ARMS AND THE POCKET SET ARE ../pose_common.py's -- the 79 electron-density pockets.
Arms hold different numbers of molecules (7,287 for TargetDiff against 7,888 for VoxBind)
because that is what each sampled; nothing is filtered here. An arm is drawn only if it
has a samples.sdf for all 79 pockets, which is the same rule pose_common.arms_for applies
to metrics -- an arm mid-stage must not become a curve over part of the data.

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

def variants():
    """(name, arms), core first -- pose_common.variants over the arms that have samples."""
    return (("core", [a for a in ARMS if a[1] in pc.CORE]), ("all", ARMS))


def sdf_weights(paths):
    """(molecular weights, unsanitisable entries, disconnected entries dropped).

    DISCONNECTED MOLECULES ARE DROPPED, because notebook/webapp/metrics.py drops them
    before scoring anything else -- so keeping them would put this figure on a different
    molecule set than the strain, clash and PoseBusters figures for the same run. It
    matters for exactly one arm: TargetDiff emits 511 of 7,798 (6.6%) over the 79 pockets
    and every other arm emits none. Their weight is also not a ligand's weight -- it is
    the sum of two or more pieces that never bonded."""
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


def samples(root, targets=None):
    """The samples.sdf under `root`, over `targets` or over every target it holds."""
    names = sorted(d for d in os.listdir(root) if d.startswith("target_")) \
        if targets is None else targets
    paths = [os.path.join(root, t, "samples.sdf") for t in names]
    return [p for p in paths if os.path.exists(p)]


def reference_paths():
    """The deposited ligand in each of the 79 pockets: the one SDF in the target dir that
    is not samples.sdf. Every arm that has one carries the same molecule, so this reads
    ours -- the staged baseline dirs hold no reference at all."""
    out = []
    for t in pc.P79:
        d = os.path.join(pc.REF_ROOT, t)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".sdf") and f != "samples.sdf"]
    return out


# An arm is in only if it holds a samples.sdf for every one of the 79 pockets.
ARMS = [a for a in pc.ARMS if len(samples(a[2], pc.P79)) == len(pc.P79)]
MISSING = [a[0] for a in pc.ARMS if a not in ARMS]

MW, BAD, DISC, MW_ALL = {}, {}, {}, {}
for _, key, root in ARMS:
    MW[key], BAD[key], DISC[key] = sdf_weights(samples(root, pc.P79))
    MW_ALL[key], _, _ = sdf_weights(samples(root))
MW_REF, BAD_REF, DISC_REF = sdf_weights(reference_paths())


def smiles_crosscheck():
    """Molecules whose recorded SMILES gives a different weight than their SDF entry, over
    the arms that have metrics.json. Reported, never silently trusted: this builder changed
    source and the check is what says the change was free."""
    data, p79_rows, _ = pc.load_arms()
    out = {}
    for lab, key, root in ARMS:
        smi = [r["smi"] for r in p79_rows[key] if r.get("smi")]
        if not smi:
            continue
        sdf = MW[key]
        if len(smi) != len(sdf):
            out[lab] = (f"{len(smi)} SMILES vs {len(sdf)} SDF — not comparable; "
                        f"metrics.json is still being written for this arm")
            continue
        mism = sum(1 for a, b in zip(smi, sdf)
                   if abs(Descriptors.MolWt(Chem.MolFromSmiles(a)) - b) > 0.05)
        out[lab] = f"{mism} of {len(sdf)} differ"
    return out

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

    # Filled steps read well for the two or three arms of `core`; at nine they stack into
    # one opaque mass and the arm you are looking for is the one you cannot see. So the
    # fill is dropped once the figure carries more than the core arms, and the step lines
    # alone do the work -- the same reason ../fig-posecheck draws its eight-method figures
    # as unfilled curves.
    fill = len(arms) <= len(pc.CORE)
    for lab, key, _ in arms:
        pct, _ = share(MW[key])
        col = color(lab)
        ax.step(CENTRES, pct, where="mid", color=col, lw=pc.DIST_LW + 0.35, zorder=3)
        if fill:
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
    pc.legend(ax, pc.arm_handles(arms), loc="upper right",
              ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
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
    pc.legend(ax, pc.arm_handles(arms), loc="lower right",
              ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "mw_ecdf", variant)


# ── data exports ────────────────────────────────────────────────────────────────
def block(v, bad=0, disc=0):
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
        f"pct_over_{XMAX}": round(100 * float((v > XMAX).mean()), 2),
    }


def exports():
    summary = {
        "built": "2026-09-09",
        "metric": "molecular weight (RDKit Descriptors.MolWt) of every generated molecule",
        "source": "target_*/samples.sdf, single-component molecules only",
        "dropped": ("molecules that are not one connected component, as "
                    "notebook/webapp/metrics.py drops them; only TargetDiff emits any"),
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79),
                       "targets": pc.P79},
        "histogram": {"bin_da": BIN, "x_max": XMAX,
                      "reference_roll_bins": REF_ROLL},
        "arms": {}, "reference_ligand": block(MW_REF, BAD_REF),
    }
    for lab, key, root in ARMS:
        summary["arms"][lab] = {
            "root": root, "key": key, "pockets_all": len(samples(root)),
            "p79": block(MW[key], BAD[key], DISC[key]),
            "all_pockets": block(MW_ALL[key]),
        }
    with open(os.path.join(HERE, "mw_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "mw_histogram.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mw_bin_lo", "mw_bin_hi", "n", "pct"])
        for lab, key, _ in ARMS:
            pct, counts = share(MW[key])
            for lo, n, q in zip(EDGES[:-1], counts, pct):
                w.writerow([lab, lo, lo + BIN, int(n), round(float(q), 3)])
        pct, counts = share(MW_REF)
        for lo, n, q in zip(EDGES[:-1], counts, pct):
            w.writerow([pc.REF_LABEL, lo, lo + BIN, int(n), round(float(q), 3)])
    return summary


def main():
    pc.use_style()
    for variant, arms in variants():
        fig_distribution(arms, variant)
        fig_ecdf(arms, variant)
    summary = exports()

    print(f"{len(pc.P79)} pockets · molecular weight from target_*/samples.sdf, "
          f"single-component molecules only · {BIN} Da bins to {XMAX} Da\n")
    print(f"{'arm':16s} {'mols':>6s} {'median':>8s} {'mean':>7s} {'IQR':>15s} "
          f"{'>500 Da':>8s} {f'>{XMAX} Da':>9s} {'disconn.':>9s}")
    for lab in [l for l, _, _ in ARMS] + [pc.REF_LABEL]:
        b = (summary["arms"][lab]["p79"] if lab != pc.REF_LABEL
             else summary["reference_ligand"])
        print(f"{lab:16s} {b['n_molecules']:6d} {b['mw_median']:8.1f} {b['mw_mean']:7.1f} "
              f"{b['mw_q25']:7.1f}–{b['mw_q75']:<7.1f} {b['pct_over_500']:7.2f}% "
              f"{b[f'pct_over_{XMAX}']:8.2f}% {b['n_disconnected_dropped']:9d}")
    if MISSING:
        print(f"\nnot drawn — no samples.sdf over all {len(pc.P79)} pockets: "
              + ", ".join(MISSING))
    print("\nSDF vs recorded SMILES, molecules whose weight differs by >0.05 Da:")
    for lab, msg in smiles_crosscheck().items():
        print(f"  {lab:16s} {msg}")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
