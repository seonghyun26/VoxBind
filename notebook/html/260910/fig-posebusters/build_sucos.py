#!/usr/bin/env python3
"""build_sucos.py — SuCOS: the one check PoseBusters' `gen` config adds over `dock`.

    sucos_ecdf_{core,all}.*                distribution, with the 0.4 threshold marked
    sucos_per_atom_{mean,median}_{core,all}.*   SuCOS against ligand size
    sucos_summary.json                     per-arm mean/median and threshold rate
    sucos_per_molecule_<arm>.json          per-molecule values

WHAT SuCOS IS. Shape overlap x pharmacophore-feature overlap between the generated pose and
a reference ligand -- here each pocket's crystal ligand. 1.0 is a perfect superposition,
0 is no overlap at all. It needs `mol_true`, which is why `dock` (our default) does not
have it and `gen` does.

IT IS NOT A VALIDITY CHECK, AND IT IS NOT FOLDED INTO `valid`. Every other PoseBusters
column asks "is this pose physically possible"; SuCOS asks "does it sit where the crystal
ligand sits". A de novo model is not trying to reproduce the crystal ligand -- that is the
job redocking benchmarks measure -- so a low SuCOS is a statement about novelty and pocket
occupancy, not about a broken molecule. Running `PoseBusters(config="gen")` would silently
merge the two: gen's chosen binary output is `sucos_within_threshold` at 0.4, so it enters
the all-must-pass `valid` and collapses it. This script therefore computes the SAME
quantity with the SAME parameter -- `check_sucos(..., sucos_threshold=0.4)`, exactly what
gen.yml configures -- and keeps it as its own number. main() prints what gen-mode `valid`
would have been, so the cost of that conflation is on the record rather than implied.

A second reason not to run the full gen config: it would re-run all 20 dock checks,
including the 50-conformer energy ensemble, to obtain one extra column. SuCOS needs only
the two molecules and no protein, and this way costs ~1.4 s per pocket.

MOLECULE SET. samples.sdf read with the same filter metrics.py uses -- RDKit-sanitizable
and single-fragment -- so the molecules here are the ones every other figure in this folder
and ../fig-posecheck draws. Reference ligand via notebook/webapp/data.load_gt_mols, the
same lookup the metrics pipeline uses.

Runs in the `moleval` env, which is where posebusters lives:

    /opt/conda/envs/moleval/bin/python \\
        notebook/html/260910/fig-posebusters/build_sucos.py
    CACHE=0 ... build_sucos.py     # ignore sucos_per_molecule_*.json and recompute
"""
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/home1/irteam/VoxBind/notebook/webapp")
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

SUCOS_THRESHOLD = 0.4            # gen.yml's own value
CACHE = os.environ.get("CACHE", "1") != "0"


def compute(key, root):
    """[{t, n, sucos}] over every pocket the run holds."""
    from rdkit import Chem, RDLogger
    from posebusters.modules.sucos import check_sucos
    from data import load_gt_mols
    RDLogger.DisableLog("rdApp.*")           # a few samples fail sanitization; that is data

    out, skipped = [], 0
    for t in sorted(d for d in os.listdir(root) if d.startswith("target_")
                    and os.path.isdir(os.path.join(root, d))):
        d = os.path.join(root, t)
        sdf = os.path.join(d, "samples.sdf")
        gt = load_gt_mols(Path(d))   # data.py takes a Path
        if not os.path.exists(sdf) or not gt:
            skipped += 1
            continue
        ref = gt[0]                          # TargetDiff convention: one per pocket
        for m in Chem.SDMolSupplier(sdf, sanitize=True):
            if m is None:
                continue
            try:
                if "." in Chem.MolToSmiles(m):    # same filter metrics.py applies
                    continue
                s = check_sucos(m, ref, sucos_threshold=SUCOS_THRESHOLD)["results"]["sucos"]
            except Exception:                # noqa: BLE001 — one bad pose is not a run
                continue
            if s is None or not np.isfinite(s):
                continue
            out.append({"t": t, "n": m.GetNumHeavyAtoms(), "sucos": float(s)})
    return out, skipped


def load(key, root, label):
    path = os.path.join(HERE, f"sucos_per_molecule_{key}.json")
    if CACHE and os.path.exists(path):
        return json.load(open(path))["molecules"]
    rows, skipped = compute(key, root)
    json.dump({"arm": label, "root": root, "threshold": SUCOS_THRESHOLD,
               "n_molecules": len(rows), "pockets_skipped": skipped,
               "p79_targets": pc.P79,
               "fields": {"t": "target dir", "n": "heavy atoms",
                          "sucos": "SuCOS vs the pocket's crystal ligand"},
               "molecules": rows}, open(path, "w"))
    print(f"  computed {label}: {len(rows)} molecules"
          + (f", {skipped} pockets skipped" if skipped else ""))
    return rows


def fig_ecdf(rows_by_arm, arms, variant):
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for lab, key, _ in arms:
        v = np.sort([r["sucos"] for r in rows_by_arm[key]])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color(lab),
                lw=pc.MODEL_LW, zorder=5, solid_capstyle="round")
    # gen.yml would call everything left of this line invalid.
    ax.axvline(SUCOS_THRESHOLD, color=pc.INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
    # Horizontal and high: rotated against the line it ran straight through the curves.
    ax.text(SUCOS_THRESHOLD + 0.015, 0.97, f"gen threshold {SUCOS_THRESHOLD}",
            ha="left", va="top", fontsize=11, color=pc.INK)
    pc.furniture(ax, ylabel="Cumulative share",
                 xlabel="SuCOS vs the pocket's crystal ligand", xloc=None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pc.legend(ax, pc.arm_handles(arms, include_ref=False), loc="upper left", fontsize=11.5)
    pc.fit(fig, pad=0.5)
    pc.save(fig, HERE, "sucos_ecdf", variant)


def fig_per_atom(rows_by_arm, arms, variant):
    per = {key: pc.by_size(rows_by_arm[key], "sucos") for _, key, _ in arms}
    xs = pc.x_range(per, arms)
    for stat, f in (("mean", lambda v: float(np.mean(v))),
                    ("median", lambda v: float(np.median(v)))):
        fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for lab, key, _ in arms:
            ax.plot(xs, pc.model_curve(per[key], xs, f), color=color(lab),
                    lw=pc.MODEL_LW, zorder=5, solid_capstyle="round")
        ax.axhline(SUCOS_THRESHOLD, color=pc.INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
        pc.furniture(ax, ylabel=f"SuCOS {stat}", xlabel=pc.X_LABEL,
                     xlim=(xs[0] - 0.6, xs[-1] + 0.6))
        ax.set_ylim(0, 1)
        pc.legend(ax, pc.arm_handles(arms, include_ref=False), loc="upper left",
                  fontsize=11.5)
        pc.fit(fig, pad=0.5)
        pc.save(fig, HERE, f"sucos_per_atom_{stat}", variant)
    return xs


def std_weights(p79_rows):
    """One common size distribution -- every arm pooled -- for direct standardization."""
    return collections.Counter(r["n"] for rows in p79_rows.values() for r in rows)


def standardized(rows, weights, f):
    """The arm's SuCOS WITHIN each 1-heavy-atom stratum, re-weighted by `weights`. SuCOS
    turns out to be nearly flat in size, so this barely moves -- which is itself worth
    recording, because every other metric in this section is size-confounded."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["n"]].append(r["sucos"])
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * f(per[n]); den += w
    return num / den if den else float("nan")


def paired(a_rows, b_rows):
    """Per-pocket mean SuCOS, a - b, over the pockets both cover. Paired because the
    pockets differ from each other far more than the arms do: an unpaired comparison of
    two arms is mostly a comparison of which pockets each happened to cover."""
    def by_pocket(rows):
        d = collections.defaultdict(list)
        for r in rows:
            d[r["t"]].append(r["sucos"])
        return {t: float(np.mean(v)) for t, v in d.items()}
    a, b = by_pocket(a_rows), by_pocket(b_rows)
    ts = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in ts]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    half = 1.96 * sd / max(1, len(diff)) ** 0.5
    return {"n_pockets": len(ts), "mean_diff": round(float(np.mean(diff)), 4),
            "median_diff": round(st.median(diff), 4),
            "ci95": [round(float(np.mean(diff)) - half, 4),
                     round(float(np.mean(diff)) + half, 4)],
            "a_higher_in": sum(d > 0 for d in diff)}


def main():
    pc.use_style()
    rows_by_arm = {}
    for lab, key, root in pc.ARMS:
        rows_by_arm[key] = load(key, root, lab)
    p79 = set(pc.P79)
    p79_rows = {key: [r for r in rows if r["t"] in p79] for key, rows in rows_by_arm.items()}

    ranges = {}
    for variant, arms in pc.variants():
        fig_ecdf(p79_rows, arms, variant)
        ranges[variant] = fig_per_atom(p79_rows, arms, variant)

    summary = {"built": "2026-09-09", "metric": "SuCOS vs the pocket's crystal ligand",
               "threshold": SUCOS_THRESHOLD, "source": "posebusters gen.yml's sucos module",
               "note": ("NOT folded into PoseBusters `valid`: it measures overlap with the "
                        "crystal ligand, not whether the pose is physically possible."),
               "pocket_set": {"name": "p79", "n": len(pc.P79)}, "arms": {}}
    print(f"\n{'arm':16s} {'mean':>7s} {'median':>7s} {'q25':>7s} {'q75':>7s} "
          f"{'std mean':>9s} {'>=0.4':>8s} {'mols':>7s}")
    weights = std_weights(p79_rows)
    for lab, key, _ in pc.ARMS:
        v = [r["sucos"] for r in p79_rows[key]]
        over = 100 * sum(x >= SUCOS_THRESHOLD for x in v) / len(v)
        summary["arms"][lab] = {
            "n_molecules": len(v), "sucos_mean": round(float(np.mean(v)), 4),
            "sucos_median": round(float(np.median(v)), 4),
            "sucos_q25": round(float(np.percentile(v, 25)), 4),
            "sucos_q75": round(float(np.percentile(v, 75)), 4),
            "sucos_mean_size_standardized":
                round(standardized(p79_rows[key], weights, np.mean), 4),
            "sucos_median_size_standardized":
                round(standardized(p79_rows[key], weights, np.median), 4),
            "within_threshold_rate": round(over / 100, 4),
        }
        print(f"{lab:16s} {np.mean(v):7.3f} {st.median(v):7.3f} "
              f"{np.percentile(v, 25):7.3f} {np.percentile(v, 75):7.3f} "
              f"{summary['arms'][lab]['sucos_mean_size_standardized']:9.3f} "
              f"{over:7.1f}% {len(v):7d}")

    # The claim this figure exists to support, stated as a paired test rather than a
    # difference of two pooled means.
    summary["paired"] = {
        "VoxBind + Ours - VoxBind": paired(p79_rows["ours_v1"], p79_rows["vanilla"]),
        "VoxBind + Ours - TargetDiff": paired(p79_rows["ours_v1"], p79_rows["targetdiff"]),
    }
    print()
    for name, d in summary["paired"].items():
        print(f"  {name:30s} {d['mean_diff']:+.4f} "
              f"(95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}), "
              f"higher in {d['a_higher_in']}/{d['n_pockets']} pockets")
    json.dump(summary, open(os.path.join(HERE, "sucos_summary.json"), "w"),
              indent=1, ensure_ascii=False)

    # What running config="gen" instead of "dock" would have cost, stated rather than
    # implied: `valid` there is all-must-pass INCLUDING sucos_within_threshold.
    try:
        pb = json.load(open(os.path.join(HERE, "posebusters_summary.json")))
        print(f"\n{'arm':16s} {'dock valid':>11s} {'gen valid (upper bound)':>24s}")
        for lab, key, _ in pc.ARMS:
            dock = pb["arms"][lab]["p79"]["pb_valid_rate"]
            gen = dock * summary["arms"][lab]["within_threshold_rate"]
            print(f"{lab:16s} {100 * dock:10.1f}% {100 * gen:23.1f}%")
        print("  (upper bound: the product assumes the two are independent; the true gen\n"
              "   rate is the share passing BOTH, which cannot exceed either factor)")
    except (OSError, KeyError):
        pass
    print(f"\nper-atom x range: "
          + " · ".join(f"{v} {xs[0]}-{xs[-1]}" for v, xs in ranges.items()))
    print(f"wrote {HERE}")


if __name__ == "__main__":
    main()
