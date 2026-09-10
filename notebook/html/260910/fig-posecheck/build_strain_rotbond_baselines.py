#!/usr/bin/env python3
"""build_strain_rotbond_baselines.py — strain against rotatable bonds, ALL methods.

    strain_rotbond_all_methods_{median,mean}.*  lines, every method, per exact bond count
    strain_box_rotbond_grid_all_methods.*       ONE PANEL PER METHOD, boxes across the axis
    strain_rotbond_all_methods.{json,csv}       the numbers behind both

`build_strain_per_rotbond.py` next door draws the same metric for the three arms we run
locally. This adds the published baselines, whose per-molecule strain has been in
`posecheck_<Method>.json` all along but which could not be put on this axis before,
because those exports carry no SMILES and rotatable bonds cannot be counted without one.
The molecules themselves are now in the results bundle, so the count can be recovered.

WHERE THE ROTATABLE BONDS COME FROM, AND WHY THE JOIN IS SAFE. The baselines' molecules
live in `results/task2-drugdesign/<M>/samples/meta/` as the TargetDiff meta format: one
list per test pocket of {mol, smiles, ligand_filename, pred_pos}. `export_posecheck_json.py`
built `posecheck_<Method>.json` from THE SAME meta, walking each pocket's entries in order
and skipping the ones whose `mol` is None -- so the two are the same molecules in the same
order, and `load_meta` here reproduces its loader exactly (base + `_part2` concatenated per
pocket; `_gap.pt` is present in the bundle but that loader does not use it, so neither does
this one).

That is an argument, not evidence, so the join is CHECKED rather than trusted: for every
pocket the heavy-atom sequence recomputed from the meta must equal the `n` sequence in the
export, position by position, and a mismatch aborts. It matches 100/100 pockets for all
four methods, on exactly 7,655 / 7,772 / 7,720 / 6,427 molecules over the 79 pockets --
the totals `../fig-posecheck/README.md` already records from
`baselines/_eval/summary_density79.json`.

FUNCBIND IS NOT HERE, and that is a finding rather than an omission. Its strain export
exists (`posecheck_FuncBind.json`, 9,992 molecules) but no meta for it reached this box.
Its shard SDFs under `funcbind/artifacts/reproduction/crossdocked/paper_run` were tried as
a substitute and FAIL the same check -- 1 pocket of 100 matches, pocket 0 holds 99
molecules against the export's 100, and the atom counts disagree from the first record.
They are a different sampling run, not the one PoseCheck scored. Joining them would have
attached the wrong molecule's torsion count to every strain value, silently.

MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN. The baselines were
scored against the whole `*_rec.pdb` receptor; the local arms here are read from the
pocket10 crop in their target `metrics.json`, the same source the sibling figure uses.
Strain is a property of the ligand's own conformer -- UFF relaxation under a position
constraint -- so receptor scope cannot enter it, and measuring the same molecules both ways
confirms it does not: median |relative difference| 0.6-1.0 %, which is the run-to-run noise
`num_confs=50` already carries, and pooled medians move under 1 % (60.1 vs 62.6, 80.5 vs
82.9, 346.2 vs 350.4). Do NOT extend this to clashes or interactions: those are
receptor-dependent and a crop cannot see an atom it does not contain.

TWO FORMS, BECAUSE EIGHT SERIES DO NOT FIT ONE AXIS. The sibling figure dodges its boxes
inside each bond count, which works at two or three arms and fails at eight: 120 boxes on
one axis, 0.17 in of width each, and the reader asked to compare eight thin slivers across
a group boundary. So the boxes here become SMALL MULTIPLES -- one panel per method, the
paper's Fig. 12/13 form -- sharing one y range, one x range, one whisker rule and the same
grey crystal-ligand line in every panel, so only the boxes differ between them. The lines
keep the all-on-one-axis view, where colour plus a dash pattern per baseline is the
identity channel `build_posecheck_all_by_atom_range.py` already uses for nine series.

    /opt/conda/envs/voxbind/bin/python \\
        notebook/html/260910/fig-posecheck/build_strain_rotbond_baselines.py
    ... --stat median        # only the median line figure
    ... --kind box           # only the binned boxes
"""
import argparse
import collections
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402
from method_colors import color                                      # noqa: E402

from rdkit import Chem, RDLogger                                     # noqa: E402
from rdkit.Chem import rdMolDescriptors                              # noqa: E402

RDLogger.DisableLog("rdApp.*")

pc.REF_WIN, pc.MIN_REF = 1, 12          # as in the sibling: this axis is a third as long

# Walk up to the repository root by looking for the bundle itself rather than counting
# directory levels -- this file sits four deep and a miscount silently resolves to
# notebook/results, which exists.
BUNDLE = next((os.path.join(d, "results/task2-drugdesign")
               for d in (HERE, *(os.path.dirname(HERE[:i]) for i in
                                 range(len(HERE), 0, -1) if HERE[i - 1] == os.sep))
               if os.path.isdir(os.path.join(d, "results/task2-drugdesign"))
               and os.path.isdir(os.path.join(d, "voxbind"))), None)
if BUNDLE is None:
    raise SystemExit("results/task2-drugdesign not found above " + HERE)

# label, bundle folder, meta stem. DecompDiff's meta is the reference-prior run, which is
# the one export_posecheck_json.py scored.
BASELINES = [
    ("AR",         "AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
]
# Thin and dashed: with seven models on one axis hue alone is not enough, and the three
# local arms are the subject while these are context. Same channel split as
# build_posecheck_all_by_atom_range.py.
BASE_LW = 1.6
BASE_DASH = {"AR": (0, (5, 2)), "Pocket2Mol": (0, (1, 1.6)),
             "DiffSBDD": (0, (6, 2, 1, 2)), "DecompDiff": (0, (9, 3))}

X_LABEL = "Number of rotatable bonds in ligand"
XTICK = 1
STRAIN_FLOOR, WHIS = 1e-2, (5, 95)
BOX_WIDE, BOX_GAP = 1.52, 0.30
# The small-multiple grid: two columns of wide panels rather than one row of narrow ones,
# because the x carries fifteen bond counts and each needs room for a box.
GRID_COLS, GRID_WIDE, GRID_TALL, GRID_XTICK, BOX_W = 2, 1.34, 1.06, 3, 0.62
# Boxes are drawn far further into the tail than the sibling figures' MIN_N=25 allows,
# so every method covers its own full range instead of being clipped to the narrowest
# one. Ten molecules is the floor for a box to carry quartiles at all; the share strip
# under each panel is what tells the reader which end of the axis is thin.
# The crystal ligands get a panel of their own, and 79 of them over ~12 bond counts
# cannot meet MIN_REF at every one. Their panel uses a lower floor than the dashed
# reference LINE does -- a box on 6 ligands is thin but it is honest and visibly thin,
# whereas the line is a summary and would read as solid at the same n.
GRID_MIN_N, GRID_MIN_REF = 10, 5
TAIL = 1e4
STATS = {"median": lambda v: float(np.median(v)), "mean": lambda v: float(np.mean(v))}
LEGEND_LOC = {"median": "upper left", "mean": "lower right"}

# The boxes bin the axis; the lines keep every count. Half-open, so these are {0}, {1,2},
# {3,4}, {5,6}, {7,8,9}, {10+}.
BIN_EDGES = [0, 1, 3, 5, 7, 10, 10 ** 6]
BIN_LABELS = ["0", "1–2", "3–4", "5–6", "7–9", "10+"]

_RB = {}


def rot_bonds(smiles):
    if not smiles:
        return None
    if smiles not in _RB:
        m = Chem.MolFromSmiles(smiles)
        _RB[smiles] = None if m is None else int(rdMolDescriptors.CalcNumRotatableBonds(m))
    return _RB[smiles]


# ── the baselines ────────────────────────────────────────────────────────────────
def load_meta(folder, stem):
    """export_posecheck_json.py's loader, reproduced exactly: base, then `_part2`
    concatenated PER POCKET. `_gap.pt` ships in the bundle and that loader ignores it, so
    including it here would shift every molecule after the split."""
    d = os.path.join(BUNDLE, folder, "samples/meta")
    meta = torch.load(os.path.join(d, f"{stem}.pt"), weights_only=False)
    part2 = os.path.join(d, f"{stem}_part2.pt")
    if os.path.exists(part2):
        meta = [a + b for a, b in zip(meta, torch.load(part2, weights_only=False))]
    return meta


def baseline_rows(label, folder, stem, keep):
    """Rows shaped like pose_common's, for the p79 pockets, with the join checked."""
    meta = load_meta(folder, stem)
    export = json.load(open(os.path.join(HERE, f"posecheck_{label}.json")))
    by_pocket = collections.defaultdict(list)
    for m in export["molecules"]:
        by_pocket[m["p"]].append(m)

    rows, unparsed = [], 0
    for p in sorted(keep):
        entries = [e for e in meta[p] if e.get("mol") is not None]
        scored = by_pocket[p]
        mine = [int(e["mol"].GetNumAtoms()) for e in entries[:len(scored)]]
        if mine != [m["n"] for m in scored]:
            raise SystemExit(f"{label}: meta/export heavy-atom sequence differs at pocket "
                             f"{p} -- the join is not valid, refusing to guess")
        for e, m in zip(entries, scored):
            rb = rot_bonds(e.get("smiles"))
            unparsed += rb is None
            rows.append({"n": m["n"], "s": m["s"], "c": m["c"], "rb": rb})
    return rows, unparsed, len(export["molecules"])


def load_all():
    """(label -> rows) for every drawn method, plus the crystal-ligand rows. The three
    local arms come through pose_common so their curves are identical, molecule for
    molecule, to the ones the sibling figure draws."""
    _, p79_rows, refrows = pc.load_arms()
    series, notes = {}, {}
    for lab, key, _ in pc.ARMS:
        rows = p79_rows[key]
        for r in rows:
            r["rb"] = rot_bonds(r.get("smi"))
        series[lab] = rows
        notes[lab] = {"source": "target metrics.json (pocket10 crop)", "n_rows": len(rows)}
    for r in refrows:
        r["rb"] = rot_bonds(r.get("smi"))

    keep = {int(t.split("_")[1]) for t in pc.P79}
    for lab, folder, stem in BASELINES:
        rows, bad, n_all = baseline_rows(lab, folder, stem, keep)
        series[lab] = rows
        notes[lab] = {"source": f"results/task2-drugdesign/{folder} meta x posecheck_{lab}.json"
                                " (whole receptor)",
                      "n_rows": len(rows), "n_all_pockets": n_all, "smiles_unparsed": bad}
    return series, refrows, notes


def order(labels):
    """Local arms first, then the baselines, so the legend reads subject-then-context."""
    local = [l for l, _, _ in pc.ARMS]
    return [l for l in local if l in labels] + [l for l, _, _ in BASELINES if l in labels]


def style_of(label):
    return ((pc.MODEL_LW, "-") if label in [l for l, _, _ in pc.ARMS]
            else (BASE_LW, BASE_DASH[label]))


# ── figures ──────────────────────────────────────────────────────────────────────
def save(fig, stem):
    """pose_common.save appends a `_<variant>` suffix; this family has no core/all split --
    every figure it writes draws every method there is data for -- so it saves without one
    rather than leaving a dangling underscore in the filename."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), facecolor="white")
    plt.close(fig)


def handles(labels):
    h = [Line2D([], [], color=pc.REF_COLOR, lw=pc.REF_LW, ls=pc.DASH, label=pc.REF_LABEL)]
    for lab in labels:
        lw, ls = style_of(lab)
        h.append(Line2D([], [], color=color(lab), lw=lw, ls=ls, label=lab))
    return h


def lines(xs, per, ref_per, labels, stat):
    f = STATS[stat]
    fig, ax = plt.subplots(figsize=(pc.FIG_W, pc.PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(xs, pc.reference_curve(ref_per, xs, f), color=pc.REF_COLOR, lw=pc.REF_LW,
            ls=pc.DASH, zorder=4, dash_capstyle="round")
    for lab in labels:
        lw, ls = style_of(lab)
        ax.plot(xs, pc.model_curve(per[lab], xs, f), color=color(lab), lw=lw, ls=ls,
                zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    pc.furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=X_LABEL,
                 xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=XTICK)
    pc.legend(ax, handles(labels), loc=LEGEND_LOC[stat], fontsize=10, ncol=2)
    pc.fit(fig, pad=0.5)
    save(fig, f"strain_rotbond_all_methods_{stat}")


def boxes_grid(series, refrows, labels):
    """ONE CELL PER METHOD -- its strain distribution across the axis, over its own
    rotatable-bond distribution. The paper's Fig. 12/13 form.

    With eight series this is the only box form that works: dodging them inside each bond
    count gave 0.17 in of width per box and asked the reader to compare eight thin slivers
    across a group boundary. Small multiples ask the easier question -- one method's shape
    at a glance -- and everything that has to be shared for the panels to be comparable IS
    shared: one y range, one x range, one whisker rule, and the same grey crystal-ligand
    line repeated in every panel as the common ruler.

    EVERY METHOD IS DRAWN OVER ITS OWN FULL RANGE, not clipped to where the others or the
    crystal ligands reach. The axis spans the union, so a method that stops at 11 rotatable
    bonds visibly stops while VoxBind runs past 20 -- that difference is a property of the
    models and the figure should show it. The grey reference line simply ends where the 79
    crystal ligands run out; being cut off is the honest thing for it to do.

    THE STRIP UNDER EACH PANEL IS WHAT MAKES THE THIN END READABLE. Boxes are drawn wherever
    a method has GRID_MIN_N molecules, which is far below the sibling figures' MIN_N -- so
    the axis reaches the tail, and the share strip immediately below says how little of the
    method's output lives there. Without it a box over 12 molecules and a box over 1,100
    look identical."""
    panels = labels + [pc.REF_LABEL]
    rows_of = {lab: series[lab] for lab in labels}
    rows_of[pc.REF_LABEL] = refrows

    per = {lab: pc.by_size(rows_of[lab], "s", key="rb") for lab in panels}
    # The strip counts EVERY molecule with a bond count, including the few whose UFF
    # relaxation failed and so contribute no box -- it is the output distribution, not the
    # scored-subset distribution.
    dist = {lab: collections.Counter(r["rb"] for r in rows_of[lab] if r["rb"] is not None)
            for lab in panels}
    floor = {lab: (GRID_MIN_REF if lab == pc.REF_LABEL else GRID_MIN_N) for lab in panels}
    valid = {lab: [x for x in sorted(per[lab]) if len(per[lab][x]) >= floor[lab]]
             for lab in panels}
    xs = list(range(min(v[0] for v in valid.values() if v),
                    max(v[-1] for v in valid.values() if v) + 1))
    ref_line = pc.reference_curve(per[pc.REF_LABEL], xs, STATS["median"])

    ncol = GRID_COLS
    nrow = -(-len(panels) // ncol)
    # Constrained layout, not pose_common's tight_layout pass: each cell is a main panel
    # plus a strip that must sit tight against it, while consecutive CELLS need a real gap.
    # tight_layout applies one spacing to every row and cannot make that distinction.
    fig = plt.figure(figsize=(pc.FIG_W * GRID_WIDE, pc.PANEL_H * GRID_TALL * nrow),
                     dpi=220, layout="constrained")
    fig.patch.set_facecolor("white")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.04, hspace=0.045, wspace=0.03)
    cells = fig.subfigures(nrow, ncol).ravel()

    top_ax = None
    for i, lab in enumerate(panels):
        sub = cells[i]
        sub.patch.set_facecolor("white")
        ax, strip = sub.subplots(2, 1, sharex=True,
                                 gridspec_kw={"height_ratios": pc.HEIGHT_RATIOS,
                                              "hspace": 0.06})
        for a in (ax, strip):
            a.set_facecolor("white")
        col = pc.REF_COLOR if lab == pc.REF_LABEL else color(lab)

        drawn = valid[lab]
        if drawn:
            bp = ax.boxplot([np.clip(per[lab][x], STRAIN_FLOOR, None) for x in drawn],
                            positions=drawn, widths=BOX_W, whis=WHIS, showfliers=False,
                            patch_artist=True, zorder=5, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.55, edgecolor=col, linewidth=1.0)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=col, linewidth=1.0)
            for med in bp["medians"]:
                med.set(color=pc.INK, linewidth=1.4, solid_capstyle="butt")
        # The crystal ligands as the same ruler in every panel, including their own.
        ax.plot(xs, ref_line, color=pc.REF_COLOR, lw=pc.REF_LW, ls=pc.DASH, zorder=6,
                dash_capstyle="round")
        ax.set_yscale("log")
        if top_ax is None:
            top_ax = ax
        else:
            ax.sharey(top_ax)
        pc.furniture(ax, ylabel="Strain (kcal mol⁻¹)" if i % ncol == 0 else None,
                     xlim=(xs[0] - 0.8, xs[-1] + 0.8), xloc=GRID_XTICK)
        ax.set_title(lab, fontsize=13, color=pc.INK, loc="left", pad=5)
        ax.tick_params(labelbottom=False)

        total = sum(dist[lab].values())
        pct = [100 * dist[lab].get(x, 0) / total for x in xs]
        strip.step(xs, pct, where="mid", color=col, lw=pc.DIST_LW, zorder=3)
        strip.fill_between(xs, pct, step="mid", color=col, alpha=pc.DIST_FILL, lw=0,
                           zorder=2)
        # The x name goes under the BOTTOM ROW only. Repeated under all eight cells it was
        # wider than a cell, overran into the neighbouring column and pushed the right-hand
        # panels off the figure; the whisker rule it used to carry lives in the README and
        # the docstring instead of costing a line of width in every cell.
        pc.furniture(strip, ylabel="% of ligands" if i % ncol == 0 else None,
                     xlabel=X_LABEL if i >= len(panels) - ncol else None,
                     xlim=(xs[0] - 0.8, xs[-1] + 0.8), xloc=GRID_XTICK)
        strip.set_ylim(bottom=0)

    for sub in cells[len(panels):]:
        sub.set_visible(False)
    top_ax.set_ylim(bottom=STRAIN_FLOOR)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"strain_box_rotbond_grid_all_methods.{ext}"),
                    facecolor="white")
    plt.close(fig)


def bin_of(rb):
    return min(int(np.searchsorted(BIN_EDGES, rb, side="right")) - 1, len(BIN_LABELS) - 1)


# ── exports ──────────────────────────────────────────────────────────────────────
def tail_share(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > TAIL for x in v) / len(v), 2) if v else None


def exports(xs, series, per, ref_per, refrows, labels, notes):
    out = {
        "built": "2026-09-09",
        "metric": "PoseCheck strain against RDKit strict rotatable-bond count",
        "pocket_set": {"name": "p79 electron-density pockets", "n": len(pc.P79)},
        "join": ("baselines: results/task2-drugdesign/<M>/samples/meta (base + _part2, per "
                 "pocket) x posecheck_<M>.json, verified position-by-position on the "
                 "heavy-atom sequence for all 100 pockets"),
        "excluded": {"FuncBind": ("strain export exists but no meta on this box; its shard "
                                  "SDFs are a different sampling run and fail the "
                                  "heavy-atom check (1/100 pockets)")},
        "scope_note": ("baselines scored whole-receptor, local arms on the pocket10 crop; "
                       "measured median |rel diff| 0.6-1.0 % on strain, which is "
                       "receptor-independent by construction"),
        "bins": {"labels": BIN_LABELS, "edges": BIN_EDGES[:-1] + ["inf"]},
        "rotatable_bonds": xs, "arms": {},
    }
    for lab in labels:
        rows = series[lab]
        out["arms"][lab] = {
            **notes[lab],
            "n_scored": sum(len(v) for v in per[lab].values()),
            "strain_gt_1e4": tail_share(rows),
            "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2),
            **{f"strain_{s}": pc.model_curve(per[lab], xs, STATS[s]) for s in STATS},
            "bins": {b: (round(float(np.median(v)), 2) if v else None) for b, v in
                     ((BIN_LABELS[i], [r["s"] for r in rows if r["s"] is not None
                                       and r["rb"] is not None and bin_of(r["rb"]) == i])
                      for i in range(len(BIN_LABELS)))},
        }
    out["arms"][pc.REF_LABEL] = {
        "source": "crystal ligand of each of the 79 pockets",
        "n_scored": sum(len(v) for v in ref_per.values()),
        "strain_gt_1e4": tail_share(refrows),
        "atoms_mean": round(float(np.mean([r["n"] for r in refrows])), 2),
        **{f"strain_{s}": pc.reference_curve(ref_per, xs, STATS[s]) for s in STATS},
    }
    json.dump(out, open(os.path.join(HERE, "strain_rotbond_all_methods.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(HERE, "strain_rotbond_all_methods.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "rotatable_bonds", "n", "strain_median", "strain_mean"])
        for lab in labels:
            for i, x in enumerate(xs):
                d = out["arms"][lab]
                w.writerow([lab, x, len(per[lab].get(x, ())),
                            "" if d["strain_median"][i] is None else round(d["strain_median"][i], 3),
                            "" if d["strain_mean"][i] is None else round(d["strain_mean"][i], 3)])
        d = out["arms"][pc.REF_LABEL]
        for i, x in enumerate(xs):
            w.writerow([pc.REF_LABEL, x, len(ref_per.get(x, ())),
                        "" if d["strain_median"][i] is None else round(d["strain_median"][i], 3),
                        "" if d["strain_mean"][i] is None else round(d["strain_mean"][i], 3)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stat", choices=("median", "mean", "both"), default="both")
    ap.add_argument("--kind", choices=("box", "line", "both"), default="both")
    args = ap.parse_args()
    stats = ("median", "mean") if args.stat == "both" else (args.stat,)
    kinds = ("box", "line") if args.kind == "both" else (args.kind,)

    pc.use_style()
    series, refrows, notes = load_all()
    labels = order(series)
    per = {lab: pc.by_size(series[lab], "s", key="rb") for lab in labels}
    ref_per = pc.by_size(refrows, "s", key="rb")

    common = set.intersection(*(set(per[l]) for l in labels))
    xs = [x for x in sorted(common)
          if all(len(per[l][x]) >= pc.MIN_N for l in labels)]
    if "line" in kinds:
        for s in stats:
            lines(xs, per, ref_per, labels, s)
    if "box" in kinds:
        boxes_grid(series, refrows, labels)
    out = exports(xs, series, per, ref_per, refrows, labels, notes)

    print(f"{len(pc.P79)} pockets · {len(labels)} methods + reference · "
          f"x = {xs[0]}-{xs[-1]} rotatable bonds "
          f"(counts where every drawn method has ≥{pc.MIN_N} molecules)\n")
    head = BIN_LABELS
    print(f"{'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{b:>8s}" for b in head) + f" {'>1e4':>7s}   (strain median by bin)")
    for lab in labels + [pc.REF_LABEL]:
        d = out["arms"][lab]
        if lab == pc.REF_LABEL:
            rows = refrows
            cells = [[r["s"] for r in rows if r["s"] is not None and r["rb"] is not None
                      and bin_of(r["rb"]) == i] for i in range(len(BIN_LABELS))]
            vals = [f"{np.median(v):8.1f}" if len(v) >= pc.MIN_REF else f"{'—':>8s}"
                    for v in cells]
        else:
            vals = [f"{d['bins'][b]:8.1f}" if d["bins"][b] is not None else f"{'—':>8s}"
                    for b in head]
        print(f"{lab:16s} {d['n_scored']:6d} {d['atoms_mean']:6.1f} " + " ".join(vals)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print(f"\nFuncBind excluded: {out['excluded']['FuncBind']}")
    print(f"\nwrote {HERE}")


if __name__ == "__main__":
    main()
