#!/usr/bin/env python3
"""Build all raster figures used by 260723_meeting.html.

The macrocycle comparison uses the matched 79-density-pocket evaluations:
generated groups contain only molecules whose largest RDKit SSSR ring has at
least 12 atoms; the reference group contains one redocked crystal ligand for
each unique pocket where either model generated at least one macrocycle.

All outputs are high-resolution PNGs so the meeting page does not depend on
browser SVG support.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from rdkit import Chem, RDLogger
from scipy.stats import gaussian_kde

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

OURS_JSON = os.path.join(
    REPO,
    "voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/"
    "samples/full_eval_ep350/eval_docking_results.json",
)
SIG09_JSON = os.path.join(
    REPO,
    "voxbind/exps/exp_sig0.9_v2/samples/full_eval_ep350/"
    "eval_docking_results.json",
)
SIG10_JSON = os.path.join(
    REPO,
    "voxbind/exps/exp_sig1.0_350ep/samples/full_eval_ep349/"
    "eval_docking_results.json",
)
CD_ROOT = os.path.join(REPO, "voxbind/dataset/data/crossdocked_pocket10")

MACRO = 12
INK = "#1c2433"
SOFT = "#5b6678"
LINE = "#e3e7ee"
ACC = "#2f6f4f"
BLUE = "#38559b"
GOLD = "#b07a17"


plt.rcParams.update({
    "font.size": 11.5,
    "axes.edgecolor": "#aeb6c4",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SOFT,
    "ytick.color": SOFT,
    "font.family": "sans-serif",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def largest_ring_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    return max(sizes) if sizes else 0


def macro_records(data):
    """Return macrocycle observations and receptors with >=1 macrocycle."""
    rows = []
    receptors = set()
    for target in data.get("per_target", []):
        for mol in target.get("per_mol", []):
            dock = mol.get("vina_dock")
            if dock is None:
                continue
            ring = largest_ring_from_smiles(mol.get("smiles"))
            if ring is not None and ring >= MACRO:
                rows.append({"dock": float(dock), "ring": int(ring),
                             "receptor": target["receptor"]})
                receptors.add(target["receptor"])
    return rows, receptors


def reference_molecule(target, run_root):
    ref_ligand = target.get("ref_ligand")
    candidates = []
    if ref_ligand and "__" in ref_ligand:
        subdir, filename = ref_ligand.split("__", 1)
        candidates.append(os.path.join(CD_ROOT, subdir, filename))
    target_dir = os.path.join(run_root, target["target"])
    if os.path.isdir(target_dir):
        candidates.extend(
            p for p in sorted(
                os.path.join(target_dir, fn) for fn in os.listdir(target_dir)
                if fn.endswith(".sdf") and fn != "samples.sdf"
            )
        )
    for path in candidates:
        if not os.path.exists(path):
            continue
        mol = Chem.MolFromMolFile(path, sanitize=True, removeHs=False)
        if mol is not None:
            return mol
    return None


def largest_ring_from_mol(mol):
    if mol is None:
        return None
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    return max(sizes) if sizes else 0


def finish_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=LINE, lw=.8)
    ax.set_axisbelow(True)


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


def plot_macrocycle_prevalence():
    labels = ["VoxBind\ngenerated", "CrossDocked\n(unique lig.)", "PLINDER v2\n(unique lig.)"]
    n = np.array([618, 11685, 22050])
    n12 = np.array([10, 190, 360])
    n8 = np.array([17, 240, 463])
    pct12, pct8 = 100 * n12 / n, 100 * n8 / n

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(labels))
    h = .36
    ax.barh(y + h / 2, pct8, height=h, color=GOLD, alpha=.55,
            label="ring >= 8 atoms")
    ax.barh(y - h / 2, pct12, height=h, color=ACC,
            label="macrocycle (ring >= 12)")
    for yi, value in zip(y - h / 2, pct12):
        ax.text(value + .035, yi, f"{value:.2f}%", va="center", fontsize=10,
                color=ACC, fontweight="bold")
    for yi, value in zip(y + h / 2, pct8):
        ax.text(value + .035, yi, f"{value:.2f}%", va="center", fontsize=9.5,
                color=GOLD)
    ax.set_yticks(y, [f"{label}\n(n={count:,})" for label, count in zip(labels, n)])
    ax.invert_yaxis()
    ax.set_xlabel("frequency (% of ligands)")
    ax.set_xlim(0, max(pct8) * 1.22)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower right",
              bbox_to_anchor=(1, 1.01))
    finish_axis(ax, "x")
    save(fig, "fig_macrocycle_fraction.png")


def plot_macrocycle_size_distribution(ours_rows, sigma_rows, ref_rings, train_rings, plinder_rings):
    groups = [
        ("Ours — frozen C+D+G", ACC, Counter(r["ring"] for r in ours_rows)),
        ("VoxBind sigma 0.9", BLUE, Counter(r["ring"] for r in sigma_rows)),
        ("Reference ligands", INK, Counter(ref_rings)),
    ]
    sizes = np.arange(MACRO, max(max(c) for _, _, c in groups) + 1)
    x = np.arange(len(sizes))
    width = .27
    TRAIN = "#7a5195"       # CrossDocked = the generator's training set
    PRETRAIN = "#dd7e33"    # PLINDER v2 = the frozen density encoder's pretraining corpus
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for offset, (label, color, counts) in zip((-.27, 0, .27), groups):
        values = [counts.get(int(size), 0) for size in sizes]
        bars = ax.bar(x + offset, values, width=width, color=color, alpha=.9,
                      label=f"{label} (n={sum(values)})")
        if label == "Reference ligands":
            for bar, value in zip(bars, values):
                if value:
                    ax.text(bar.get_x() + bar.get_width() / 2, value + .55, str(value),
                            ha="center", va="bottom", fontsize=8, color=INK)
    # dataset macrocycle-size shapes on a shared secondary axis (compare shape, not the
    # generated raw counts): CrossDocked = generator training set, PLINDER v2 = the frozen
    # density encoder's pretraining corpus (instance-level, what the encoder actually sees)
    ax2 = ax.twinx()
    train_values = [train_rings.get(int(size), 0) for size in sizes]
    plinder_values = [plinder_rings.get(int(size), 0) for size in sizes]
    ax2.plot(x, train_values, color=TRAIN, lw=2, marker="o", ms=4, zorder=5,
             label=f"CrossDocked train (n={sum(train_rings.values())})")
    ax2.plot(x, plinder_values, color=PRETRAIN, lw=2, marker="s", ms=4, zorder=6,
             label=f"PLINDER v2 pretrain (n={sum(plinder_rings.values())})")
    ax2.set_ylabel("dataset macrocycle count", color=SOFT)
    ax2.tick_params(axis="y", colors=SOFT)
    ax2.set_ylim(0, max(max(train_values), max(plinder_values)) * 1.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#aeb6c4")
    ax.set_xticks(x, sizes, fontsize=8.5)
    ax.set_xlabel("macrocycle size (atoms in largest SSSR ring)")
    ax.set_ylabel("generated / reference count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8.2,
              ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", color=LINE, lw=.8)
    ax.set_axisbelow(True)
    save(fig, "fig_macrocycle_size_distribution.png")


def summary(values):
    values = np.asarray(values, float)
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "n": len(values), "mean": values.mean(), "median": np.median(values),
        "std": values.std(ddof=1), "q25": q25, "q75": q75,
        "min": values.min(), "max": values.max(),
    }


def plot_macrocycle_dock(ours_rows, sigma_rows, ref_dock):
    groups = [
        ("Ours\nmacrocycles", np.array([r["dock"] for r in ours_rows]), ACC),
        ("VoxBind sigma 0.9\nmacrocycles", np.array([r["dock"] for r in sigma_rows]), BLUE),
        ("Reference\nmatched pockets", np.array(ref_dock), INK),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    rng = np.random.default_rng(7)
    top = max(float(v.max()) for _, v, _ in groups)
    for i, (label, values, color) in enumerate(groups):
        jitter = rng.uniform(-.16, .16, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=17, color=color,
                   alpha=.34, edgecolors="none", zorder=2)
        box = ax.boxplot(values, positions=[i], widths=.46, showfliers=False,
                         patch_artist=True, zorder=3)
        box["boxes"][0].set(facecolor="white", edgecolor=color, lw=1.8, alpha=.94)
        for part in box["whiskers"] + box["caps"]:
            part.set(color=color, lw=1.5)
        box["medians"][0].set(color=color, lw=2.4)
        ax.text(i, top + .45, f"n={len(values)}", ha="center", va="bottom",
                color=color, fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10)
    ax.set_ylabel("Vina Dock score (kcal/mol) · lower = stronger")
    ax.set_ylim(min(float(v.min()) for _, v, _ in groups) - .7, top + 1.25)
    finish_axis(ax, "y")
    save(fig, "fig_dock_macro.png")


def group_substr(svg, group_id):
    start = svg.index(f'<g id="{group_id}">')
    depth = 0
    for match in re.finditer(r"<g\b[^>]*>|</g>", svg[start:]):
        if match.group(0).startswith("</g"):
            depth -= 1
            if depth == 0:
                return svg[start:start + match.end()]
        else:
            depth += 1
    raise RuntimeError(f"unbalanced SVG group: {group_id}")


def uses_xy(group):
    points = []
    for use in re.findall(r"<use\b[^>]*?/>", group):
        mx = re.search(r'\bx="([-\d.]+)"', use)
        my = re.search(r'\by="([-\d.]+)"', use)
        if mx and my:
            points.append((float(mx.group(1)), float(my.group(1))))
    return np.asarray(points, float)


def linmap(pixel_a, value_a, pixel_b, value_b):
    slope = (value_a - value_b) / (pixel_a - pixel_b)
    return slope, value_a - slope * pixel_a


def plot_dock_vs_atoms_from_svg():
    """Rasterize the preserved 618-sample diagnostic from its recoverable points."""
    with open(os.path.join(HERE, "fig_dock_natoms.svg")) as fh:
        svg = fh.read()
    non = uses_xy(group_substr(svg, "PathCollection_1"))
    macro = uses_xy(group_substr(svg, "PathCollection_2"))
    kx, cx = linmap(macro[:, 0].mean(), 36.5, non[:, 0].mean(), 22.3)
    ky, cy = linmap(macro[:, 1].mean(), -9.98, non[:, 1].mean(), -7.32)
    n_non = np.rint(kx * non[:, 0] + cx).astype(int)
    n_macro = np.rint(kx * macro[:, 0] + cx).astype(int)
    d_non = ky * non[:, 1] + cy
    d_macro = ky * macro[:, 1] + cy
    n_all = np.concatenate([n_non, n_macro]).astype(float)
    d_all = np.concatenate([d_non, d_macro])

    fig, ax = plt.subplots(figsize=(7.0, 4.35))
    ax.scatter(n_non, d_non, s=17, color=SOFT, alpha=.28, edgecolors="none",
               label=f"non-macrocycle (n={len(n_non)})", zorder=2)
    ax.scatter(n_macro, d_macro, s=48, color=ACC, alpha=.96, edgecolors="white",
               linewidths=.6, label=f"macrocycle (n={len(n_macro)})", zorder=4)
    coef = np.polyfit(n_all, d_all, 1)
    xr = np.array([n_all.min(), n_all.max()])
    ax.plot(xr, coef[0] * xr + coef[1], color=GOLD, lw=1.8, ls="--",
            label="size trend", zorder=3)
    ax.set_xlabel("heavy-atom count")
    ax.set_ylabel("Vina Dock score (kcal/mol) · lower = stronger")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    finish_axis(ax, "both")
    save(fig, "fig_dock_natoms.png")


def generated_atom_counts(data):
    return np.asarray([
        int(mol["n_atoms"])
        for target in data.get("per_target", [])
        for mol in target.get("per_mol", [])
        if mol.get("n_atoms") is not None
    ])


def reference_atom_counts(data, run_root):
    values = []
    for target in data.get("per_target", []):
        mol = reference_molecule(target, run_root)
        if mol is not None:
            values.append(sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()))
    return np.asarray(values, int)


def training_atom_counts():
    """Heavy-atom counts of the CrossDocked training ligands (data_train.pt).

    Each element is (pocket_dict, ligand_dict); the ligand is the small one and
    carries only heavy-atom channels, so its atom count == coords length.
    Cached to train_natoms.npy (loading the 685 MB tensor takes a while).
    """
    cache = os.path.join(HERE, "train_natoms.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    data = torch.load(os.path.join(REPO, "voxbind/dataset/data/data_train.pt"),
                      map_location="cpu", weights_only=False)
    counts = np.asarray([complex_[1]["coords"].shape[0] for complex_ in data], int)
    np.save(cache, counts)
    return counts


def training_macro_rings():
    """Largest-ring size per CrossDocked training ligand -> Counter of macrocycles.

    Reads ~100k training SDFs on first run (slow, a few minutes); the per-ligand
    largest-ring array is cached to train_largest_ring.npy.
    """
    cache = os.path.join(HERE, "train_largest_ring.npy")
    if os.path.exists(cache):
        rings = np.load(cache)
    else:
        import torch
        data = torch.load(os.path.join(REPO, "voxbind/dataset/data/data_train.pt"),
                          map_location="cpu", weights_only=False)
        rings = np.full(len(data), -1, dtype=np.int16)
        for i, ex in enumerate(data):
            path = os.path.join(CD_ROOT, ex[0]["id"].replace("_pocket10.pdb", ".sdf"))
            mol = Chem.MolFromMolFile(path, sanitize=False)
            if mol is None:
                continue
            try:
                Chem.FastFindRings(mol)
            except Exception:
                continue
            atom_rings = mol.GetRingInfo().AtomRings()
            rings[i] = max((len(r) for r in atom_rings), default=0)
        np.save(cache, rings)
    return Counter(int(v) for v in rings[rings >= MACRO])


def plot_atom_count_distribution(ours, sigma, sigma10, ref, train):
    groups = [
        ("Vanilla σ=0.9", BLUE, generated_atom_counts(sigma)),
        ("Vanilla σ=1.0", GOLD, generated_atom_counts(sigma10)),
        ("Frozen C+D+G", ACC, generated_atom_counts(ours)),
    ]
    lo = min([a.min() for _, _, a in groups] + [ref.min(), train.min()])
    hi = max([a.max() for _, _, a in groups] + [ref.max(), train.max()])
    bins = np.arange(lo - .5, hi + 1.5, 1)
    # focus the x-axis on where the mass is; a handful of >72-atom training
    # ligands would otherwise leave the right half of the plot empty
    hi_plot = max([a.max() for _, _, a in groups] + [ref.max()]) + 2

    fig, ax = plt.subplots(figsize=(7.6, 4.75))
    for label, color, values in groups:
        ax.hist(values, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=.13, zorder=2)
        ax.hist(values, bins=bins, density=True, histtype="step",
                color=color, lw=2, zorder=3,
                label=f"{label} (med {int(np.median(values))})")
    # dataset baselines as smooth KDE lines (neutral, so they read as references,
    # not competing with the coloured generator histograms)
    xs = np.linspace(lo - 1, hi + 1, 400)
    ax.plot(xs, gaussian_kde(train)(xs), color=SOFT, lw=2.2, zorder=4,
            label=f"CrossDocked train (med {int(np.median(train))})")
    ax.plot(xs, gaussian_kde(ref)(xs), color=INK, lw=1.9, ls=(0, (5, 2)),
            zorder=5, label=f"Test references (med {int(np.median(ref))})")
    ax.set_xlabel("heavy-atom count per molecule")
    ax.set_ylabel("density (log scale)")
    ax.set_xlim(lo - 1, hi_plot)
    # log y so the long right tail (large, rare molecules up to ~70 atoms) is visible
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.09)
    ax.legend(frameon=False, fontsize=8.4, loc="lower center",
              bbox_to_anchor=(.5, 1.015), ncol=3, columnspacing=1.25,
              handlelength=2.5, handletextpad=.55)
    finish_axis(ax, "y")
    ax.grid(axis="y", which="both", color=LINE, lw=.8)
    save(fig, "fig_natoms_dist.png")


def print_summary(label, values):
    stats = summary(values)
    print(
        f"{label}: n={stats['n']} mean={stats['mean']:.3f} "
        f"median={stats['median']:.3f} sd={stats['std']:.3f} "
        f"IQR={stats['q25']:.3f}..{stats['q75']:.3f} "
        f"range={stats['min']:.3f}..{stats['max']:.3f}"
    )


def main():
    ours = load_json(OURS_JSON)
    sigma = load_json(SIG09_JSON)
    sigma10 = load_json(SIG10_JSON)
    ours_rows, ours_pockets = macro_records(ours)
    sigma_rows, sigma_pockets = macro_records(sigma)
    macro_pockets = ours_pockets | sigma_pockets

    ours_by_receptor = {target["receptor"]: target for target in ours["per_target"]}
    ref_dock = [
        float(ours_by_receptor[receptor]["ref_vina_dock"])
        for receptor in sorted(macro_pockets)
        if ours_by_receptor.get(receptor, {}).get("ref_vina_dock") is not None
    ]
    ours_root = os.path.dirname(OURS_JSON)
    ref_rings = []
    for target in ours["per_target"]:
        ring = largest_ring_from_mol(reference_molecule(target, ours_root))
        if ring is not None and ring >= MACRO:
            ref_rings.append(int(ring))

    train_rings = training_macro_rings()
    with open(os.path.join(HERE, "macrocycle_data.json")) as fh:
        plinder_rings = Counter(
            int(v) for v in json.load(fh)["plinder_instances"]["macro_ring_sizes"])
    plot_macrocycle_prevalence()
    plot_macrocycle_size_distribution(ours_rows, sigma_rows, ref_rings, train_rings, plinder_rings)
    plot_macrocycle_dock(ours_rows, sigma_rows, ref_dock)
    plot_dock_vs_atoms_from_svg()
    ref_atoms = reference_atom_counts(ours, ours_root)
    train_atoms = training_atom_counts()
    plot_atom_count_distribution(ours, sigma, sigma10, ref_atoms, train_atoms)

    print("macrocycle pockets: ours=%d sigma0.9=%d union=%d intersection=%d" % (
        len(ours_pockets), len(sigma_pockets), len(macro_pockets),
        len(ours_pockets & sigma_pockets),
    ))
    print("ring-size counts ours:", dict(sorted(Counter(r["ring"] for r in ours_rows).items())))
    print("ring-size counts sigma0.9:", dict(sorted(Counter(r["ring"] for r in sigma_rows).items())))
    print("ring-size counts reference:", dict(sorted(Counter(ref_rings).items())))
    print_summary("ours macrocycles", [r["dock"] for r in ours_rows])
    print_summary("sigma0.9 macrocycles", [r["dock"] for r in sigma_rows])
    print_summary("matched reference", ref_dock)
    print("reference heavy atoms: n=%d mean=%.1f median=%.0f" %
          (len(ref_atoms), ref_atoms.mean(), np.median(ref_atoms)))
    print("training heavy atoms: n=%d mean=%.1f median=%.0f" %
          (len(train_atoms), train_atoms.mean(), np.median(train_atoms)))


if __name__ == "__main__":
    main()
