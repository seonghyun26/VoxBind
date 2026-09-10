#!/usr/bin/env python3
"""build_posecheck_per_atom.py — PoseCheck strain and clashes against ligand size.

    strain_per_atom_{mean,median}_{core,all}.*   strain against heavy-atom count
    clash_per_atom_{mean,median}_{core,all}.*    clashes against heavy-atom count
    strain_clash_ecdf_pair_{core,all}.*          both distributions, pooled, one shared y
    posecheck_per_atom.{json,csv}                the numbers behind the curves
    posecheck_summary.json                       pooled numbers (p79 and all_pockets)
    posecheck_per_molecule_<arm>.json            per-molecule export

This is the size-resolved view of the three arms we run locally. The sibling builders in
this folder cover the other cuts of the same metric:

  build_posecheck_all_by_atom_range.py       all eight methods, binned, ECDF + violin
  build_posecheck_baselines_by_atom_range.py the five published baselines alone
  export_posecheck_json.py                   their per-molecule exports

PoseBusters (dock-mode validity) is the sibling folder, ../fig-posebusters. The two split
by metric and share ../pose_common.py, which owns the arm list, the pocket set, the loader
and the house furniture -- everything a disagreement between them would silently corrupt.

MEAN AND MEDIAN ARE SEPARATE FIGURES, which is what the 3-line figures do
(vina_dock_3line_mean / _median) and here it is not optional. Strain's mean is not a
location statistic: 6.6% of molecules fail UFF relaxation and land between 1e4 and 1e13,
and one of those at a thin heavy-atom count moves that count's mean by four decades. The
two sit three to four decades apart, so overlaid, the mean's spikes cross the whole panel
and bury the medians -- which are tight, ordered, and the thing worth reading. Drawn apart,
the contrast is itself the argument for reporting the median: in the mean figure the arms
are tangled with no order at all, in the median figure they separate cleanly.

The mean figure is still clipped to the bulk -- the 3-line figure's own answer (see its
`limits`) -- and the excluded points are NAMED in the run log rather than squashing
everything else into two decades. Clashes have no such problem and are drawn the same way
for symmetry.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posecheck/build_posecheck_per_atom.py
"""
import csv
import json
import os
import statistics as st
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

STATS = (("mean", lambda v: float(np.mean(v))),
         ("median", lambda v: float(np.median(v))))
STRAIN_CLIP = 1e6

DATA, P79_ROWS, REFROWS = pc.load_arms()


def per_atom_stat(field, xs, arms, variant, name, unit, stat, f, *, log, clip=None,
                  stem, legend_loc="upper left"):
    """One statistic, one panel, one file. Which statistic you are looking at is carried by
    the filename and by the y-axis name, exactly as the 3-line figures carry it."""
    per = {key: pc.by_size(P79_ROWS[key], field) for _, key, _ in arms}
    ref_per = pc.by_size(REFROWS, field)
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dropped = []
    ax.plot(xs, pc.reference_curve(ref_per, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
            ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = pc.model_curve(per[key], xs, f)
        if clip:
            over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
            if over:
                dropped.append((lab, over))
        ax.plot(xs, y, color=color(lab), lw=pc.MODEL_LW, zorder=5, solid_capstyle="round")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if clip:
        ax.set_ylim(top=clip)
    pc.furniture(ax, ylabel=f"{name} {stat}{unit}", xlabel=pc.X_LABEL,
                 xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    pc.legend(ax, pc.arm_handles(arms), loc=legend_loc, fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, stem, variant)
    return dropped


def fig_strain(xs, arms, variant):
    dropped = per_atom_stat("s", xs, arms, variant, "Strain", "\n(kcal mol⁻¹)", *STATS[0],
                            log=True, clip=STRAIN_CLIP, stem="strain_per_atom_mean",
                            legend_loc="lower right")
    per_atom_stat("s", xs, arms, variant, "Strain", "\n(kcal mol⁻¹)", *STATS[1],
                  log=True, stem="strain_per_atom_median", legend_loc="upper left")
    return dropped


def fig_clash(xs, arms, variant):
    for stat, f in STATS:
        per_atom_stat("c", xs, arms, variant, "Clashes", "", stat, f, log=False,
                      stem=f"clash_per_atom_{stat}", legend_loc="upper left")


def fig_ecdf_pair(arms, variant):
    """Two panels on one shared y, the 3-line pair layout. ECDFs rather than violins: the
    question is what share of a method's poses sit under a given strain or clash count, and
    a cumulative curve answers it at every threshold instead of at three quartiles."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(pc.FIG_W * 1.52, pc.PANEL_H),
                                      dpi=220, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")

    def ecdf(ax, field, clip):
        for lab, key, _ in arms:
            v = np.array([r[field] for r in P79_ROWS[key] if r[field] is not None])
            x = np.sort(np.clip(v, clip, None))
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color(lab),
                    lw=pc.MODEL_LW, zorder=5)
        v = np.array([r[field] for r in REFROWS if r[field] is not None])
        x = np.sort(np.clip(v, clip, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=pc.REF_COLOR, lw=pc.REF_LW,
                ls=pc.DASH, zorder=4)

    ecdf(left, "s", 1e-2)
    left.set_xscale("log")
    left.set_xlim(1e-1, 3e3)
    pc.furniture(left, ylabel="Cumulative share",
                 xlabel="PoseCheck strain (kcal mol⁻¹)", xloc=None)
    left.set_ylim(0, 1.0)
    pc.legend(left, pc.arm_handles(arms), loc="upper left", fontsize=11.5)

    ecdf(right, "c", 0)
    pc.furniture(right, ylabel=None, xlabel="PoseCheck steric clashes", xloc=5)
    right.set_xlim(0, 30)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "strain_clash_ecdf_pair", variant)


# ── data exports ────────────────────────────────────────────────────────────────
def block(rows):
    s = [r["s"] for r in rows if r["s"] is not None]
    c = [r["c"] for r in rows if r["c"] is not None]
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_strain": len(s),
        "strain_mean": round(float(np.mean(s)), 1) if s else None,
        "strain_median": round(st.median(s), 1) if s else None,
        "strain_q25": round(float(np.percentile(s, 25)), 1) if s else None,
        "strain_q75": round(float(np.percentile(s, 75)), 1) if s else None,
        "clash_mean": round(float(np.mean(c)), 2) if c else None,
        "clash_median": round(float(np.median(c)), 1) if c else None,
    }


def exports(xs):
    summary = {
        "built": "2026-09-09", "metric": "PoseCheck strain and steric clashes",
        "definition": ("strain is the posecheck 1.3.1 definition, NOT the VoxBind paper's "
                       "-- ~10x smaller, never place the two side by side"),
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79),
                       "targets": pc.P79},
        "arms": {}, "reference_ligand": block(REFROWS),
    }
    for lab, key, root in pc.arms_for("s", DATA):
        pockets = sorted(DATA[key])
        summary["arms"][lab] = {
            "root": root, "key": key, "pockets_all": len(pockets),
            "p79": block(P79_ROWS[key]),
            "all_pockets": block([r for t in pockets for r in DATA[key][t]]),
        }
    json.dump(summary, open(os.path.join(HERE, "posecheck_summary.json"), "w"),
              indent=1, ensure_ascii=False)

    # The curves themselves, so a reader can check a figure without re-running it.
    per_atom = {"heavy_atoms": xs, "n_pockets": len(pc.P79), "arms": {}}
    for lab, key, _ in pc.arms_for("s", DATA):
        s_per, c_per = pc.by_size(P79_ROWS[key], "s"), pc.by_size(P79_ROWS[key], "c")
        per_atom["arms"][lab] = {
            "n": [len(s_per.get(a, ())) for a in xs],
            "strain_mean": pc.model_curve(s_per, xs, STATS[0][1]),
            "strain_median": pc.model_curve(s_per, xs, STATS[1][1]),
            "clash_mean": pc.model_curve(c_per, xs, STATS[0][1]),
            "clash_median": pc.model_curve(c_per, xs, STATS[1][1]),
        }
    s_ref, c_ref = pc.by_size(REFROWS, "s"), pc.by_size(REFROWS, "c")
    per_atom["arms"][pc.REF_LABEL] = {
        "note": f"centred +-{pc.REF_WIN}-atom window, >= {pc.MIN_REF} ligands",
        "strain_mean": pc.reference_curve(s_ref, xs, STATS[0][1]),
        "strain_median": pc.reference_curve(s_ref, xs, STATS[1][1]),
        "clash_mean": pc.reference_curve(c_ref, xs, STATS[0][1]),
        "clash_median": pc.reference_curve(c_ref, xs, STATS[1][1]),
    }
    json.dump(per_atom, open(os.path.join(HERE, "posecheck_per_atom.json"), "w"),
              indent=1, ensure_ascii=False)

    fields = ["strain_mean", "strain_median", "clash_mean", "clash_median"]
    with open(os.path.join(HERE, "posecheck_per_atom.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "heavy_atoms", "n"] + fields)
        for arm, d in per_atom["arms"].items():
            for i, a in enumerate(xs):
                n = d.get("n", [""] * len(xs))[i] if "n" in d else ""
                w.writerow([arm, a, n]
                           + [("" if d[f][i] is None else round(d[f][i], 3)) for f in fields])

    for lab, key, root in pc.arms_for("s", DATA):
        mols = [{"t": t, "n": r["n"], "s": r["s"], "c": r["c"]}
                for t in sorted(DATA[key]) for r in DATA[key][t]]
        json.dump({"arm": lab, "root": root, "n_pockets": len(DATA[key]),
                   "n_molecules": len(mols), "p79_targets": pc.P79,
                   "fields": {"t": "target dir", "n": "heavy atoms",
                              "s": "strain (null = UFF relaxation did not converge)",
                              "c": "steric clashes"},
                   "molecules": mols},
                  open(os.path.join(HERE, f"posecheck_per_molecule_{key}.json"), "w"),
                  ensure_ascii=False)
    return summary


def main():
    pc.use_style()
    ranges, drops = {}, {}
    for variant, arms in pc.variants("s", DATA):
        per = {key: pc.by_size(P79_ROWS[key], "s") for _, key, _ in arms}
        xs = pc.x_range(per, arms)
        ranges[variant] = xs
        drops[variant] = fig_strain(xs, arms, variant)
        fig_clash(xs, arms, variant)
        fig_ecdf_pair(arms, variant)
    summary = exports(ranges["all"])

    print(f"79-pocket set · {len(pc.P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (counts where every drawn arm has >={pc.MIN_N} molecules)\n")
    print(f"{'arm':16s} {'atoms':>6s} {'strain med':>11s} {'strain mean':>12s} "
          f"{'clash med':>10s} {'mols':>7s}")
    for lab, key, _ in pc.arms_for("s", DATA):
        r = summary["arms"][lab]["p79"]
        print(f"{lab:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
              f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")
    r = summary["reference_ligand"]
    print(f"{pc.REF_LABEL:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
          f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")

    # Named, not hidden: the mean figure clips to the bulk, so say which points that leaves
    # off the panel and how far above they went.
    if drops["all"]:
        print(f"\nstrain_per_atom_mean_all: above the {STRAIN_CLIP:.0e} clip, off-panel "
              f"(a failed UFF relaxation at a thin count moves that count's mean):")
        for lab, over in drops["all"]:
            pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
            print(f"  {lab:16s} {len(over):2d} of {len(ranges['all']):2d}: {pts}")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
