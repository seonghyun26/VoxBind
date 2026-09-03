"""build_strain_by_size_curve_mean.py — Figure 9 of results_drug_design.html.

PoseCheck strain energy against ligand size at every heavy-atom count.

Companion to Figure 4, which asks the same size-control question of Vina Dock: is a
model's strain a property of the poses it makes, or just of the molecules it happens to
draw?  Same 78 pockets, same panel construction, same MIN_N / XMAX cut-offs, so the two
figures can be read against each other.

The statistic is the GEOMETRIC MEAN -- the companion to Figure 7's median, drawn because
"does the mean tell a different story?" is a fair question of a skewed distribution.

The ARITHMETIC mean cannot be used: 4-7% of molecules per model fail UFF relaxation and
land at 1e4-1e10 kcal/mol, which drags the pooled arithmetic mean to 2.3e8 (vanilla) and
4.3e9 (Ours v1) -- numbers that describe the failure rate, not the strain. The geometric
mean is the arithmetic mean of log-strain, which is the natural centre for a positive
heavy-tailed quantity already being drawn on a log axis, and unlike the median it still
moves when the whole upper half of the distribution shifts.

What it shows: pooled, Ours v1 remains worse than vanilla on EVERY statistic (median 63
vs 82, geometric mean 88 vs 130, 5%-trimmed mean 2.5k vs 5.5k). The mean does not rescue
the pooled comparison. Where it does change the picture is the large end -- at 36+ heavy
atoms the ordering flips on all three, and by MORE under the mean than the median
(median 373 -> 222, a 1.7x gap; 5%-trimmed mean 4481 -> 1024, a 4.4x gap).

Everything comes from target_*/metrics.json, which carries n_atoms and the PoseCheck
block on the same sample record -- no join against the docking JSON is needed.

Top: median strain per exact heavy-atom count with a bootstrap band.  Bottom: how many
molecules each model puts at that size, which is what makes the top panel's tails
trustworthy or not.  All caption text lives in the HTML, not in the image.

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/build_strain_by_size_curve.py
"""
import collections
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E = "/home1/irteam/VoxBind/voxbind/exps"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "strain_by_size_curve_mean")

# Colours are Figure 5/6's, so a curve keeps its identity across section 2.
# (label, key, root, colour, linewidth). The key becomes the SVG gid prefix, which is
# what lets results_drug_design.html toggle a method without re-rendering the figure.
MODELS = [
    ("TargetDiff", "targetdiff", "/home1/irteam/base_drug/eval/targetdiff", "#3a7bd5", 1.9),
    ("VoxBind σ=0.9", "vanilla", f"{E}/_vanilla_ep923/samples/full_eval_ep923", "#e08a1e", 2.4),
    ("Ours", "ours", f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350", "#1e7a4d", 3.0),
    ("Ours · v2", "ours_v2", f"{E}/samples_reference_receptor_ed_ep350", "#7fc7a2", 1.9),
]
REF_COLOR = "#2b3440"

# Same pocket set as Figures 5 and 6: every pocket Ours v1 covers, minus target_71.
V1 = MODELS[2][2]
P78 = sorted(
    {d for d in os.listdir(V1) if d.startswith("target_")} - {"target_71"}
)

MIN_N, XMAX = 20, 50            # matches Figure 4 so the two panels line up
BOOT, SEED = 1500, 260827
# One crystal reference ligand relaxes to 0.001 kcal/mol. Left unclamped it stretches a
# log axis over four empty decades and squeezes every curve into the top third, so it is
# drawn at the floor and called out -- the same convention Figure 5 uses on its x axis.
YFLOOR, YTOP = 1.5, 5.0e3


def geomean(values):
    """Geometric mean, i.e. exp(mean(log x)).

    Strain can be <= 0 when the 50-conformer global relaxation fails to beat the
    locally-relaxed pose; those cannot be logged, so they are floored at 1e-9. That
    affects a handful of values and pulls the result down slightly, which is the
    conservative direction for the model being defended here.
    """
    a = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    return float(np.exp(np.log(a).mean()))


def by_size(root):
    """heavy-atom count -> list of per-molecule strain values, over the 78 pockets."""
    out = collections.defaultdict(list)
    for t in P78:
        path = os.path.join(root, t, "metrics.json")
        if not os.path.exists(path):
            continue
        for s in (json.load(open(path)).get("samples") or []):
            pc = s.get("posecheck") or {}
            strain, n = pc.get("strain"), s.get("n_atoms")
            # A failed relaxation writes null, not 0 -- dropping it is not the same as
            # scoring it, and the bottom panel shows how many survived at each size.
            if strain is not None and n and np.isfinite(strain):
                out[n].append(float(strain))
    return out


def ref_by_size():
    out = collections.defaultdict(list)
    for t in P78:
        path = os.path.join(V1, t, "metrics.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path)).get("reference") or {}
        strain = (r.get("posecheck") or {}).get("strain")
        n = r.get("n_atoms")
        if strain is not None and n and np.isfinite(strain):
            out[n].append(float(strain))
    return out


S = {name: by_size(root) for name, _, root, _, _ in MODELS}
REF = ref_by_size()
rng = np.random.default_rng(SEED)

fig, (ax, bx) = plt.subplots(
    2, 1, figsize=(12.4, 6.9), sharex=True,
    gridspec_kw=dict(height_ratios=[2.8, 1.15], hspace=0.07))
fig.patch.set_facecolor("#fbfaf7")
for a in (ax, bx):
    a.set_facecolor("#fbfaf7")
    for side in ("top", "right"):
        a.spines[side].set_visible(False)

for name, key, _, col, lw in MODELS:
    d = S[name]
    xs = sorted(a for a, v in d.items() if len(v) >= MIN_N and a <= XMAX)
    if not xs:
        continue
    med = [geomean(d[a]) for a in xs]
    band = np.array([
        np.percentile(
            np.exp(np.log(np.clip(
                rng.choice(np.asarray(d[a]), (BOOT, len(d[a])), replace=True),
                1e-9, None)).mean(axis=1)),
            [2.5, 97.5])
        for a in xs])
    ax.fill_between(xs, band[:, 0], band[:, 1], color=col, alpha=0.16,
                    linewidth=0, zorder=2).set_gid(f"sr-{key}-band")
    (line,) = ax.plot(xs, med, color=col, lw=lw, zorder=4, label=name, solid_capstyle="round")
    line.set_gid(f"sr-{key}-line")
    (count,) = bx.plot(xs, [len(d[a]) for a in xs], color=col, lw=1.5, zorder=3)
    count.set_gid(f"sr-{key}-count")

rx = [a for a in sorted(REF) if a <= XMAX]
ry = [geomean(REF[a]) for a in rx]
n_clamped = sum(v < YFLOOR for v in ry)
ax.scatter(rx, [max(v, YFLOOR) for v in ry], marker="*", s=115, color=REF_COLOR,
           zorder=5, edgecolor="white", linewidth=0.5,
           label="Reference ligand (crystal)").set_gid("sr-reference-pts")

ax.set_yscale("log")
ax.set_ylim(YFLOOR, YTOP)
ax.set_ylabel("PoseCheck strain energy (kcal mol⁻¹)\ngeometric mean per atom count · log scale", fontsize=11)
legend = ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=2)
for key, text, handle in zip([m[1] for m in MODELS] + ["reference"],
                             legend.get_texts(), legend.legend_handles):
    text.set_gid(f"sr-{key}-legtext")
    handle.set_gid(f"sr-{key}-leghandle")
ax.grid(True, which="major", axis="y", color="#dfe3e8", lw=0.7, zorder=0)
ax.set_title("PoseCheck strain energy at every ligand size — geometric mean", fontsize=17, loc="left", pad=30)
ax.text(0, 1.012, "78 CrossDocked pockets · geometric mean per exact heavy-atom count · lower is better",
        transform=ax.transAxes, fontsize=11.5, color="#566072")

bx.set_yscale("log")
bx.set_ylabel("molecules\nat that size", fontsize=10)
bx.set_xlabel("Generated-ligand heavy-atom count", fontsize=12)
bx.grid(True, which="major", axis="y", color="#dfe3e8", lw=0.7, zorder=0)
bx.axhline(MIN_N, color="#b07a17", lw=1.0, ls=(0, (5, 3)), zorder=2)
bx.text(XMAX, MIN_N * 1.15, f"n = {MIN_N} cut-off", ha="right", va="bottom",
        fontsize=9.5, color="#b07a17")

# Two lines: one long line overruns the 12.4in canvas and gets clipped at both ends.
fig.text(0.5, 0.030,
         f"Drawn only where at least {MIN_N} molecules share an atom count · "
         f"bands are 95% bootstrap CIs of the geometric mean ({BOOT:,} resamples, seed {SEED})",
         ha="center", fontsize=9.5, color="#566072")
fig.text(0.5, 0.006,
         f"x ≤ {XMAX} heavy atoms"
         + (f" · {n_clamped} reference ligand below {YFLOOR} kcal/mol drawn at the floor"
            if n_clamped else ""),
         ha="center", fontsize=9.5, color="#566072")
fig.tight_layout(rect=(0, 0.048, 1, 1))
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=170, facecolor=fig.get_facecolor())
print("wrote", OUT + ".png / .svg")

summary = {}
for name, _, _, _, _ in MODELS:
    d = S[name]
    xs = sorted(a for a, v in d.items() if len(v) >= MIN_N and a <= XMAX)
    summary[name] = {
        "sizes": xs,
        "geomean_strain": [round(geomean(d[a]), 3) for a in xs],
        "n_molecules": [len(d[a]) for a in xs],
    }
summary["Reference ligand"] = {
    "sizes": rx,
    "geomean_strain": [round(geomean(REF[a]), 3) for a in rx],
    "n_molecules": [len(REF[a]) for a in rx],
}
summary["_meta"] = {"n_pockets": len(P78), "min_n": MIN_N, "xmax": XMAX,
                    "bootstrap": BOOT, "seed": SEED}
json.dump(summary, open(f"{OUT}_summary.json", "w"), indent=1)
print("wrote", OUT + "_summary.json")
