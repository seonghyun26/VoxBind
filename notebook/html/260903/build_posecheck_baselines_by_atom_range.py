"""build_posecheck_baselines_by_atom_range.py — the five published baselines, stratified
by generated-molecule heavy-atom count, in the same form as build_posecheck_by_atom_range.py.

Same reason for stratifying: strain and clashes both rise with molecule size, and the
methods draw different size mixes, so a pooled number partly reports the size mix rather
than pose quality. Inside a bin the methods are drawing comparably-sized molecules.

Data comes from prj-denovo/baselines, where the five baselines were sampled and scored
(100 pockets x 100 ligands, PoseCheck on the pose as generated), restricted here to the
79 test pockets whose receptor has usable deposited electron density -- the same subset
tables 1b/2b of baseline.html use. Reference ligands are pulled from a run that indexes
its targets as target_<data_id>; that mapping was checked against the split by canonical
SMILES, 100/100. PoseCheck was written
out per (pocket, chunk-of-20) and does not store the atom count, so the counts are read
back from the meta the chunks were built from -- an alignment the script re-checks on
every chunk (matching length AND ligand_filename) before using it, and refuses to plot
if any chunk disagrees.

The three methods in the sibling script (TargetDiff, vanilla VoxBind, Ours v1) live under
/home1/irteam and are not on this box. Their roots are still declared below: where they
exist they are merged into the same axes, so running this on that machine yields one
figure with all eight. Here it draws the five plus the crystal reference.

Writes into 260903/posecheck eval/:
  * posecheck_baselines_by_atom_range.json
  * strain_ecdf_<bin>.{png,svg}
  * clash_violin_<bin>.{png,svg}

    python notebook/html/260903/build_posecheck_baselines_by_atom_range.py
"""
import collections
import glob
import json
import os
import statistics as st
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "posecheck eval")
BASE = "/home/shpark/prj-denovo/baselines"
CHUNK = 20

# the 79 electron-density pockets, by ligand file (baselines) and by index (metrics.json)
_SEL = json.load(open(f"{BASE}/_shared_data/density_pockets_selected.json"))
KEEP_LIGANDS = set(_SEL["ligand_filenames"])
KEEP_IDX = set(_SEL["indices"])
N_POCKETS = len(KEEP_IDX)  # run_posecheck.py's chunk size; the alignment check below enforces it

# Colours follow build_denovo_vina_chart.py so a reader moving between the two charts
# keeps the same mental key. FuncBind is not in that palette, so it takes a brown that
# none of the others use.
BASELINES = [
    ("AR",         "AR",                   "#9b59b6"),
    ("Pocket2Mol", "Pocket2Mol",           "#e67e22"),
    ("DiffSBDD",   "DiffSBDD",             "#e74c3c"),
    ("DecompDiff", "DecompDiff_ref_prior", "#34495e"),
    ("FuncBind",   "FuncBind",             "#795548"),
]

# Merged in only where they exist -- see the module docstring.
E = "/home1/irteam/VoxBind/voxbind/exps"
REMOTE = [
    ("TargetDiff",    "/home1/irteam/base_drug/eval/targetdiff",                             "#3498db"),
    ("VoxBind σ=0.9", f"{E}/_vanilla_ep923/samples/full_eval_ep923",                          "#e08a1e"),
    ("Ours · v1",     f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350", "#1e7a4d"),
]
# The crystal ligand of each test pocket, from whichever local run carries a reference
# block; it is the same CrossDocked test set either way.
REF_ROOTS = [
    "/home/shpark/prj-denovo/Voxbind/voxbind/exps/260827_voxbind_base_8gpu/samples/samples_ep350_test",
    f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
]
REF_COLOR = "#2b3440"

EDGES = [0, 16, 21, 26, 31, 10 ** 6]
LABELS = ["≤15", "16–20", "21–25", "26–30", ">30"]
SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30"]
BG = "#fbfaf7"


def bin_of(n):
    return min(np.searchsorted(EDGES, n, side="right") - 1, len(LABELS) - 1)


def atom_counts(baseline):
    """Heavy-atom count of every molecule PoseCheck saw, per pocket, in chunk order."""
    meta = torch.load(f"{BASE}/_meta/{baseline}.pt", weights_only=False)
    part2 = f"{BASE}/_meta/{baseline}_part2.pt"
    if os.path.exists(part2):
        other = torch.load(part2, weights_only=False)
        meta = [a + b for a, b in zip(meta, other)]
    return meta


def pull_baseline(baseline):
    """(bin) -> {'strain': [...], 'clash': [...]} for one of the five."""
    meta = atom_counts(baseline)
    out = collections.defaultdict(lambda: {"strain": [], "clash": []})
    for path in sorted(glob.glob(f"{BASE}/_posecheck/{baseline}/pc_[0-9][0-9][0-9]_[0-9][0-9].pt")):
        stem = os.path.basename(path)[3:-3]
        idx, cid = (int(x) for x in stem.split("_"))
        d = torch.load(path, weights_only=False)
        if d["ligand_filename"] not in KEEP_LIGANDS:
            continue
        entries = meta[idx][cid * CHUNK:(cid + 1) * CHUNK]
        # refuse to guess: the chunk must be exactly the slice it was built from
        if len(entries) != d["n"] or entries[0]["ligand_filename"] != d["ligand_filename"]:
            raise SystemExit(f"chunk/meta mismatch at {baseline} pocket {idx} chunk {cid} "
                             f"-- rebuild _meta or re-run PoseCheck before plotting")
        clashes = np.asarray(d["clashes"], dtype=float)
        strain = np.asarray(d["strain"], dtype=float)
        for k, e in enumerate(entries):
            mol = e.get("mol")
            if mol is None:
                continue
            b = bin_of(mol.GetNumAtoms())
            if k < strain.size and np.isfinite(strain[k]):
                out[b]["strain"].append(float(strain[k]))
            if k < clashes.size and np.isfinite(clashes[k]):
                out[b]["clash"].append(float(clashes[k]))
    return out


def pull_metrics_json(root, reference=False):
    """Same shape, for a run that stores per-target metrics.json."""
    out = collections.defaultdict(lambda: {"strain": [], "clash": []})
    for path in sorted(glob.glob(f"{root}/target_*/metrics.json")):
        if int(os.path.basename(os.path.dirname(path)).split("_")[1]) not in KEEP_IDX:
            continue
        j = json.load(open(path))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m:
                continue
            pc = m.get("posecheck") or {}
            n = m.get("n_atoms")
            if not n:
                continue
            b = bin_of(n)
            s, c = pc.get("strain"), pc.get("clashes")
            if s is not None and np.isfinite(s):
                out[b]["strain"].append(float(s))
            if c is not None:
                out[b]["clash"].append(float(c))
    return out


METHODS, DATA = [], {}
for label, key, colour in BASELINES:
    DATA[label] = pull_baseline(key)
    METHODS.append((label, colour))
for label, root, colour in REMOTE:
    if os.path.isdir(root):
        DATA[label] = pull_metrics_json(root)
        METHODS.append((label, colour))

REF = None
for root in REF_ROOTS:
    if os.path.isdir(root):
        REF = pull_metrics_json(root, reference=True)
        break


def style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#aeb6c4")
    ax.grid(color="#e6e9ef", lw=0.85)
    ax.set_axisbelow(True)


os.makedirs(OUT, exist_ok=True)
XFLOOR, XTOP = 1e-2, 3e3

for b, (lab, slug) in enumerate(zip(LABELS, SLUGS)):
    # ── strain ECDF ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.patch.set_facecolor(BG)
    style(ax)
    for name, col in METHODS:
        v = np.array(DATA[name][b]["strain"])
        if len(v) < 20:
            continue
        x = np.sort(np.clip(v, XFLOOR, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=2.2,
                label=f"{name}  ·  median {st.median(v):.0f}", solid_capstyle="round")
    if REF is not None and len(REF[b]["strain"]) >= 3:
        r = np.array(REF[b]["strain"])
        x = np.sort(np.clip(r, XFLOOR, None))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=REF_COLOR, lw=1.8, ls="--",
                label=f"Reference  ·  median {st.median(r):.0f}  (n={len(r)})")
    ax.set_xscale("log")
    ax.set_xlim(XFLOOR, XTOP)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("UFF strain energy (kcal mol⁻¹) · log scale", fontsize=11.5)
    ax.set_ylabel("cumulative probability", fontsize=11.5)
    ax.set_title(f"Strain energy — {lab} heavy atoms", fontsize=14, fontweight="620",
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    n_txt = " · ".join(f"{n}: n={len(DATA[n][b]['strain']):,}" for n, _ in METHODS)
    ax.text(0, -0.155, f"{N_POCKETS} electron-density pockets · " + n_txt
            + f" · x clipped to {XTOP:.0e}", transform=ax.transAxes,
            fontsize=8.5, color="#7a8699")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/strain_ecdf_{slug}.{ext}", dpi=170, facecolor=BG,
                    bbox_inches="tight")
    plt.close(fig)

    # ── clash violin ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.patch.set_facecolor(BG)
    style(ax)
    series, cols, ticks = [], [], []
    for name, col in METHODS:
        v = DATA[name][b]["clash"]
        if len(v) < 20:
            continue
        series.append(v)
        cols.append(col)
        ticks.append(name.replace(" · ", "\n").replace(" σ", "\nσ"))
    if REF is not None and len(REF[b]["clash"]) >= 3:
        series.append(REF[b]["clash"])
        cols.append(REF_COLOR)
        ticks.append(f"Reference\n(n={len(REF[b]['clash'])})")
    # KDE fitted on log10(x+1) and the ticks relabelled with real counts: a plain log
    # axis is impossible because a few percent of poses have zero clashes, and symlog
    # would warp only the display while the KDE stayed in data space.
    tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
    parts = ax.violinplot([tf(v) for v in series], showextrema=False, widths=0.82)
    for body, col in zip(parts["bodies"], cols):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor(col)
        body.set_linewidth(1.2)
    for i, v in enumerate(series, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
        ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
    hi = max(max(v) for v in series)
    TICKS = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= hi * 1.6]
    ax.set_yticks(tf(TICKS))
    ax.set_yticklabels([str(t) for t in TICKS])
    ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.18)
    for i, v in enumerate(series, start=1):
        ax.text(i, ax.get_ylim()[1], f"mean {np.mean(v):.2f}\nmed {np.median(v):.0f}",
                ha="center", va="top", fontsize=9, color="#3a4352", linespacing=1.35)
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels(ticks, fontsize=9.5)
    ax.set_ylabel("steric clashes per pose  ·  log-spaced", fontsize=11.5)
    ax.set_title(f"Steric clashes — {lab} heavy atoms", fontsize=14, fontweight="620",
                 loc="left", pad=10)
    ax.text(0, -0.17, f"{N_POCKETS} electron-density pockets · thick bar = IQR · white dot = median"
                      " · axis is log10(clashes+1), relabelled with real counts"
                      " · zero is not the target: crystal poses clash too",
            transform=ax.transAxes, fontsize=8.5, color="#7a8699")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/clash_violin_{slug}.{ext}", dpi=170, facecolor=BG,
                    bbox_inches="tight")
    plt.close(fig)

# ── numbers ──────────────────────────────────────────────────────────────────
summary = {"bins": LABELS, "edges": EDGES[:-1] + ["inf"], "n_pockets": N_POCKETS,
           "source": f"prj-denovo/baselines, {N_POCKETS} electron-density pockets x 100 ligands",
           "methods": {}}
for name, _ in METHODS:
    rows = []
    for b in range(len(LABELS)):
        s_, c_ = DATA[name][b]["strain"], DATA[name][b]["clash"]
        rows.append({
            "n": len(s_),
            "strain_median": round(st.median(s_), 1) if s_ else None,
            "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
            "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
            "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
            "clash_median": round(float(np.median(c_)), 1) if c_ else None,
        })
    summary["methods"][name] = rows
if REF is not None:
    summary["methods"]["Reference ligand"] = [
        {"n": len(REF[b]["strain"]),
         "strain_median": round(st.median(REF[b]["strain"]), 1) if REF[b]["strain"] else None,
         "clash_mean": round(float(np.mean(REF[b]["clash"])), 2) if REF[b]["clash"] else None}
        for b in range(len(LABELS))]
json.dump(summary, open(f"{OUT}/posecheck_baselines_by_atom_range.json", "w"), indent=1)

print(f"{'bin':7s} {'method':14s} {'n':>6s} {'strain med':>11s} {'IQR':>18s} {'clash mean':>11s} {'med':>5s}")
for b, lab in enumerate(LABELS):
    for name, _ in METHODS:
        r = summary["methods"][name][b]
        if r["n"] == 0:
            continue
        print(f"{lab:7s} {name:14s} {r['n']:6d} {r['strain_median']:11.1f} "
              f"{r['strain_q25']:7.1f}–{r['strain_q75']:<10.1f} {r['clash_mean']:11.2f} {r['clash_median']:5.1f}")
    if REF is not None:
        rr = summary["methods"]["Reference ligand"][b]
        if rr["n"]:
            print(f"{'':7s} {'Reference':14s} {rr['n']:6d} {rr['strain_median']:11.1f} "
                  f"{'':18s} {rr['clash_mean']:11.2f}")
    print()
print(f"wrote {OUT}")
