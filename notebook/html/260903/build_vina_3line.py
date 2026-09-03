#!/usr/bin/env python3
"""build_vina_3line.py — the clean three-line per-pocket Vina Dock figure.

Three figures, all over the same 79 pockets:

    vina_dock_3line_mean.{png,svg}     y = per-pocket MEAN   Vina Dock
    vina_dock_3line_median.{png,svg}   y = per-pocket MEDIAN Vina Dock
    vina_dock_3line_pair.{png,svg}     both of the above, side by side, shared y

Three series only — Reference ligand, VoxBind, VoxBind + Ours — no bands, no
per-pocket tick labels, no in-image title. x is the pockets sorted by
"VoxBind + Ours"'s own value for that statistic, so our curve is monotone and the
other two are read against it. In the paired figure each panel keeps its OWN
ordering (each is sorted by its own statistic), which is why both panels carry
the x label; only the y scale is shared, so the two panels are directly
comparable vertically.

PROTOCOL. The published-baseline protocol only — whole `*_rec.pdb` receptor,
exhaustiveness 32, all 79 pockets — the one 260903/baseline.html and
build_results_baseline_protocol.py use, so these curves sit on the same axis as
the AR/Pocket2Mol/DiffSBDD/DecompDiff/FuncBind table. Our older crop protocol
(`*_pocket10.pdb`, exhaustiveness 16, 78 pockets — target_71 is unscoreable on
the crop) is deliberately NOT plotted: the same crystal reference ligand docks to
-7.31 one way and -7.18 the other, so the two cannot share an axis. Note that
`eval_docking_results_full.json` in the same directory is a third thing again
(full receptor but exhaustiveness 16, 78 pockets); only the `_full79` file is the
baseline protocol.

REFERENCE LINE. `ref_vina_dock` is the crystal ligand's own dock score, one value
per pocket, and it is re-measured inside every run. Vina's dock mode is unseeded,
so the two runs disagree by up to ~0.7 kcal/mol on a pocket; the line plotted is
the mean of the runs that report it, and the worst per-pocket disagreement is
printed so the noise floor stays visible.

COLOUR. Slots 1 and 2 of the validated categorical theme (blue #2a78d6, orange
#eb6834) for the two models; the reference is a neutral grey dashed rule, which is
deliberately below the chroma floor — it is a baseline, not a third category, and
carries a second (dash) channel so identity is never colour-alone.

    /opt/conda/envs/voxbind/bin/python notebook/html/260903/build_vina_3line.py
"""
import csv
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"

VANILLA = f"{E}/_vanilla_ep923/samples/full_eval_ep923"
OURS = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
BASENAME = "eval_docking_results_full79.json"
PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

REF_LABEL, REF_COLOR = "Reference ligand", "#9aa0a6"
VOX_LABEL, VOX_COLOR = "VoxBind", "#2a78d6"
OUR_LABEL, OUR_COLOR = "VoxBind + Ours", "#eb6834"

INK, INK_SOFT, GRID, AXIS = "#1f2430", "#5b6270", "#e9eaec", "#c8ccd4"

RC = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
}


def load(root):
    """{target: {"mean", "median", "ref", "n"}} from one run's docking results."""
    with open(os.path.join(root, BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    out = {}
    for t in per_target:
        vals = [m["vina_dock"] for m in (t.get("per_mol") or [])
                if isinstance(m, dict) and m.get("vina_dock") is not None]
        if not vals:
            continue
        out[t["target"]] = {"mean": st.mean(vals), "median": st.median(vals),
                            "ref": t.get("ref_vina_dock"), "n": len(vals)}
    return out


def reference(vox, our, targets):
    """Per-pocket crystal-ligand dock, averaged over the runs that report it."""
    ref, worst = {}, (0.0, None)
    for t in targets:
        vals = [d[t]["ref"] for d in (vox, our) if d[t].get("ref") is not None]
        ref[t] = st.mean(vals) if vals else None
        if len(vals) == 2 and abs(vals[0] - vals[1]) > worst[0]:
            worst = (abs(vals[0] - vals[1]), t)
    return ref, worst


def series(vox, our, ref, targets, stat):
    """The three curves for one statistic, and the pocket order they are drawn in."""
    order = sorted(targets, key=lambda t: our[t][stat])
    return (order,
            [our[t][stat] for t in order],
            [vox[t][stat] for t in order],
            [ref[t] for t in order])


def limits(y_our, y_vox, y_ref):
    """Clip to the bulk instead of squashing 79 pockets for two runaway reference
    values; the caller names the excluded points, the image does not."""
    flat = sorted(v for v in y_our + y_vox + y_ref if v is not None)
    pct = lambda q: flat[int(round(q * (len(flat) - 1)))]
    return pct(0.01) - 0.45, pct(0.99) + 0.55


def draw(ax, order, y_our, y_vox, y_ref, ylim, *, legend, ylabel):
    """One panel: three lines, a hairline y grid, and nothing else."""
    x = list(range(1, len(order) + 1))
    lo, hi = ylim

    # The reference is one crystal-ligand value per pocket, so under an ordering set
    # by another series it is inherently jagged; it is drawn thin and recessive so it
    # reads as the benchmark it is and does not fight the two model curves.
    ax.plot(x, y_ref, color=REF_COLOR, lw=1.5, ls=(0, (4, 2.6)), alpha=0.62,
            zorder=2, label=REF_LABEL, solid_capstyle="round")
    ax.plot(x, y_vox, color=VOX_COLOR, lw=2.6, zorder=3,
            label=VOX_LABEL, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(x, y_our, color=OUR_COLOR, lw=3.4, zorder=4,
            label=OUR_LABEL, solid_capstyle="round", solid_joinstyle="round")

    ax.set_xlabel(f"pocket, ordered by {OUR_LABEL}  (strongest → weakest)",
                  fontsize=12.0, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.5, labelpad=10)
    ax.set_xlim(0.2, len(order) + 0.8)
    ax.set_ylim(lo, hi)
    ax.set_xticks([1] + list(range(10, len(order) + 1, 10)) + [len(order)])
    ax.tick_params(labelsize=11, length=0, pad=6)
    ax.yaxis.grid(True, color=GRID, lw=1.0)      # solid hairline, one shade off white
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(1.0)

    if legend:
        # Bottom-right is the one empty quadrant (every curve rises to the right), so
        # the legend sits there instead of over the reference line's spikes at the left.
        leg = ax.legend(loc="lower right", frameon=False, fontsize=12.5,
                        handlelength=2.6, handletextpad=0.9, labelspacing=0.7,
                        borderaxespad=0.9)
        for text in leg.get_texts():      # identity rides the swatch, not the ink
            text.set_color(INK)


def outside(order, curves, ylim):
    """Points the y-clip leaves off the panel, as (label, target, value)."""
    lo, hi = ylim
    return [(label, order[i], v) for label, ys in curves
            for i, v in enumerate(ys) if v is not None and not lo <= v <= hi]


def save(fig, stem):
    fig.savefig(f"{stem}.png", facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    plt.close(fig)


def main():
    vox, our = load(VANILLA), load(OURS)
    targets = sorted(set(vox) & set(our))
    ref, worst = reference(vox, our, targets)
    print(f"=== {PROTOCOL} ===")
    print(f"  pockets {len(targets)}   worst per-pocket reference disagreement "
          f"between runs: {worst[0]:.2f} kcal/mol @ {worst[1]}")

    curves = {stat: series(vox, our, ref, targets, stat) for stat in ("mean", "median")}
    spans = {stat: limits(*c[1:]) for stat, c in curves.items()}
    # The paired figure shares one y scale, so the panels are comparable vertically.
    shared = (min(s[0] for s in spans.values()), max(s[1] for s in spans.values()))

    plt.rcParams.update(RC)

    for stat, (order, y_our, y_vox, y_ref) in curves.items():
        fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        draw(ax, order, y_our, y_vox, y_ref, spans[stat], legend=True,
             ylabel=f"Vina Dock, per-pocket {stat}\n(kcal/mol, lower is better)")
        fig.subplots_adjust(left=0.098, right=0.975, top=0.965, bottom=0.145)
        stem = os.path.join(HERE, f"vina_dock_3line_{stat}")
        save(fig, stem)

        wins = sum(1 for t in order if our[t][stat] <= vox[t][stat])
        pooled = {REF_LABEL: st.mean([ref[t] for t in order]),
                  VOX_LABEL: st.mean([vox[t][stat] for t in order]),
                  OUR_LABEL: st.mean([our[t][stat] for t in order])}
        print(f"  {stat:6s} -> {os.path.basename(stem)}.png/.svg   "
              f"Ours stronger on {wins}/{len(order)} pockets "
              f"({100 * wins / len(order):.1f}%)   "
              + "  ".join(f"{k} {v:+.2f}" for k, v in pooled.items()))
        for label, target, value in outside(
                order, ((OUR_LABEL, y_our), (VOX_LABEL, y_vox), (REF_LABEL, y_ref)),
                spans[stat]):
            print(f"      outside the plotted y-range: {label} @ {target} = {value:+.2f}")

        with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["rank", "target", "n_mols_ours", "n_mols_voxbind",
                        "reference", f"voxbind_{stat}", f"ours_{stat}"])
            for i, t in enumerate(order, 1):
                w.writerow([i, t, our[t]["n"], vox[t]["n"], f"{ref[t]:.3f}",
                            f"{vox[t][stat]:.3f}", f"{our[t][stat]:.3f}"])

    # side by side, one shared y scale, one legend
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 5.5), dpi=220, sharey=True)
    fig.patch.set_facecolor("white")
    for i, (stat, (order, y_our, y_vox, y_ref)) in enumerate(curves.items()):
        ax = axes[i]
        ax.set_facecolor("white")
        draw(ax, order, y_our, y_vox, y_ref, shared, legend=(i == 0),
             ylabel="Vina Dock  (kcal/mol, lower is better)" if i == 0 else None)
        ax.set_title(f"per-pocket {stat}", loc="left", fontsize=13,
                     color=INK, pad=11)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.915, bottom=0.142, wspace=0.055)
    stem = os.path.join(HERE, "vina_dock_3line_pair")
    save(fig, stem)
    print(f"  pair   -> {os.path.basename(stem)}.png/.svg   "
          f"shared y {shared[0]:.2f} .. {shared[1]:.2f}")
    for stat, (order, y_our, y_vox, y_ref) in curves.items():
        for label, target, value in outside(
                order, ((OUR_LABEL, y_our), (VOX_LABEL, y_vox), (REF_LABEL, y_ref)),
                shared):
            print(f"      outside the plotted y-range ({stat}): "
                  f"{label} @ {target} = {value:+.2f}")


if __name__ == "__main__":
    main()
