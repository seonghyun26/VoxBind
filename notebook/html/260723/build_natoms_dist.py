#!/usr/bin/env python
"""Heavy-atom-count distribution for the three §2 de-novo generation runs
(vanilla VoxBind sigma 0.9, sigma 1.0, and Ours = frozen C+D+G encoder).

Per-molecule heavy-atom counts come from each run's full 79-density-pocket
docking eval JSON.  Writes fig_natoms_dist.svg and prints a summary table.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

RUNS = [
    ("VoxBind σ 0.9", "#38559b",
     "voxbind/exps/exp_sig0.9_v2/samples/full_eval_ep350/eval_docking_results.json"),
    ("VoxBind σ 1.0", "#b07a17",
     "voxbind/exps/exp_sig1.0_350ep/samples/full_eval_ep349/eval_docking_results.json"),
    ("Ours — frozen C+D+G", "#2f6f4f",
     "voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results.json"),
]
# CrossDocked dataset baseline = the reference (crystal) ligands of the same 79 pockets
REF_JSON = "voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results.json"
CD_ROOT = "voxbind/dataset/data/crossdocked_pocket10"


def natoms(path):
    d = json.load(open(os.path.join(REPO, path)))
    out = []
    for t in d.get("per_target", []):
        for m in t.get("per_mol", []):
            if m.get("n_atoms") is not None:
                out.append(int(m["n_atoms"]))
    return np.array(out, int)


def dataset_natoms():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    d = json.load(open(os.path.join(REPO, REF_JSON)))
    out = []
    for t in d.get("per_target", []):
        rl = t.get("ref_ligand")
        if not rl:
            continue
        sub, fn = rl.split("__", 1)
        p = os.path.join(REPO, CD_ROOT, sub, fn)
        m = Chem.MolFromMolFile(p, sanitize=False) if os.path.exists(p) else None
        if m is not None:
            out.append(sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1))
    return np.array(out, int)


data = [(lab, col, natoms(p)) for lab, col, p in RUNS]
ref = dataset_natoms()
print("CrossDocked dataset ref (%d ligands): mean=%.1f median=%.0f" % (len(ref), ref.mean(), np.median(ref)))

print("run                     N     mean  median  std   min  max   IQR(25-75)")
for lab, col, a in data + [("CrossDocked dataset", "#1c2433", ref)]:
    q25, q75 = np.percentile(a, [25, 75])
    print("%-22s %5d  %5.1f  %5.0f  %4.1f  %3d  %3d   %d-%d"
          % (lab, len(a), a.mean(), np.median(a), a.std(), a.min(), a.max(), q25, q75))

lo = min([a.min() for _, _, a in data] + [ref.min()])
hi = max([a.max() for _, _, a in data] + [ref.max()])
bins = np.arange(lo - 0.5, hi + 1.5, 1)

fig, ax = plt.subplots(figsize=(6.4, 3.7), dpi=110)
for lab, col, a in data:
    ax.hist(a, bins=bins, density=True, histtype="stepfilled",
            color=col, alpha=.16, zorder=2)
    ax.hist(a, bins=bins, density=True, histtype="step",
            color=col, lw=1.9, zorder=3, label="%s  (med %d)" % (lab, int(np.median(a))))
    ax.axvline(np.median(a), color=col, lw=1.1, ls="--", alpha=.7, zorder=4)
# CrossDocked dataset baseline (79 reference crystal ligands) — smooth KDE, not a spiky
# 79-point histogram; black dashed so it reads as the reference distribution.
from scipy.stats import gaussian_kde
xs = np.linspace(lo - 1, hi + 1, 400)
ax.plot(xs, gaussian_kde(ref)(xs), color="#1c2433", lw=1.8, ls=(0, (5, 2)),
        zorder=5, label="CrossDocked dataset  (med %d)" % int(np.median(ref)))
ax.axvline(np.median(ref), color="#1c2433", lw=1.1, ls="--", alpha=.8, zorder=6)
ax.set_xlabel("heavy-atom count per generated molecule", fontsize=10.5)
ax.set_ylabel("density (log scale)", fontsize=10.5)
ax.set_xlim(lo - 1, hi + 1)
# log y so the long right tail (large, rare molecules up to ~70 atoms) is visible
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.08)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", which="both", color="#e3e7ee", lw=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_natoms_dist.svg"))
fig.savefig(os.path.join(HERE, "fig_natoms_dist.png"), dpi=140)
plt.close(fig)
print("wrote fig_natoms_dist.svg + .png")
