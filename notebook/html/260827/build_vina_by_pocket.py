#!/usr/bin/env python3
"""build_vina_by_pocket.py — per-pocket Vina Dock, VoxBind-Figure-5 style.

Two wide plots (roughly 1:2, height:width), one per statistic:

    Figure 2   y = per-pocket MEAN   Vina Dock
    Figure 3   y = per-pocket MEDIAN Vina Dock

Both are spliced into results_drug_design.html as INLINE SVG with per-method checkbox
toggles, the same mechanism Figures 5-7 use: every artist of a series carries a
``sr-<key>-<part>`` gid, the whole SVG's ids are namespaced with the figure key, and the
shared ``fig-toggle-script`` shows/hides ``[id^="fig2-sr-ours-"]`` and friends. Nothing is
drawn over the curves and there is no in-image title or legend -- the toggle row above the
figure is the legend, and all caption text lives in the HTML.

x is the 78 shared pockets, ordered by Ours v1's own value for that statistic, so
v1 traces a monotone curve and every other model is read against it. The band
between v1 and vanilla is shaded — green where v1 docks stronger, red where it
does not — which is the per-pocket version of the paired test in Table A1.

Values come from each run's ``eval_docking_results.json`` (``per_target[].per_mol[].vina_dock``),
the same source the Table-1 averages are aggregated from. The crystal reference
ligand's own Vina Dock (``ref_vina_dock``, one value per pocket) is drawn as a grey line.

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/build_vina_by_pocket.py
"""
import io
import json
import os
import re
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = "/home1/irteam/VoxBind/voxbind/exps"
OUT = os.path.join(HERE, "denovo_vina_by_pocket")

# key -> (label, dir, colour, linewidth, zorder, linestyle)
# Series keys and colours match Figures 5-7's toggle rows so the whole page shares one legend.
RUNS = [
    ("targetdiff", "TargetDiff",    os.path.join("/home1/irteam/base_drug/eval", "targetdiff"), "#3498db", 1.0, 2, "-"),
    ("vanilla",    "VoxBind σ=0.9", os.path.join(EXPS, "_vanilla_ep923/samples/full_eval_ep923"), "#1abc9c", 1.9, 4, "-"),
    ("ours_v2",    "Ours · v2",     os.path.join(EXPS, "samples_reference_receptor_ed_ep350"), "#7fcf9d", 1.2, 2, (0, (5, 2))),
    ("ours_v3",    "Ours · v3",     os.path.join(EXPS, "samples_frozen_v3_mask090_ep561"), "#52bb7e", 1.2, 2, (0, (1, 1.7))),
    ("ours",       "Ours · v1",     os.path.join(EXPS, "voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
     "#2ecc71", 2.7, 6, "-"),
]
ANCHOR = "ours"               # the sort key, and the reference curve for the shaded band
BASELINE = "vanilla"          # what the band is measured against
REF_KEY, REF_LABEL, REF_COLOR = "reference", "Reference ligand", "#8e9baa"
DOC = os.path.join(os.path.dirname(HERE), "results_drug_design.html")


def load(root):
    """{target: {"mean": …, "median": …, "ref": …}} from one run's docking results."""
    path = os.path.join(root, "eval_docking_results.json")
    with open(path, encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    out = {}
    for t in per_target:
        vals = [m["vina_dock"] for m in t.get("per_mol") or []
                if isinstance(m, dict) and m.get("vina_dock") is not None]
        if not vals:
            continue
        out[t["target"]] = {"mean": st.mean(vals), "median": st.median(vals),
                            "ref": t.get("ref_vina_dock")}
    return out


def pocket_set(data):
    """The 78 shared pockets: every pocket Ours v1 covers, minus target_71 (no Vina number)."""
    return sorted(set(data[ANCHOR]) - {"target_71"})


def render(data, targets, stat, fig_key):
    """Matplotlib figure with every series gid-tagged ``sr-<key>-<part>``; no title, no legend."""
    order = sorted(targets, key=lambda t: data[ANCHOR][t][stat])
    x = list(range(len(order)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": "#1f2937", "axes.labelcolor": "#1f2937",
        "xtick.color": "#8a94a6", "ytick.color": "#8a94a6",
    })
    fig, ax = plt.subplots(figsize=(19, 9.5), dpi=100)
    fig.patch.set_facecolor("white")

    # crystal reference ligand, one value per pocket
    ref = [data[BASELINE][t]["ref"] for t in order]
    (ln,) = ax.plot(x, ref, color=REF_COLOR, lw=1.3, ls=(0, (3, 2)), zorder=3, alpha=0.7)
    ln.set_gid(f"sr-{REF_KEY}-line")

    # where v1 docks stronger than vanilla, and where it does not
    anchor = [data[ANCHOR][t][stat] for t in order]
    base = [data[BASELINE][t].get(stat) for t in order]
    both = [i for i in x if base[i] is not None]
    for where, colour, part in (([anchor[i] <= base[i] for i in both], "#2ecc71", "band-win"),
                                ([anchor[i] > base[i] for i in both], "#d64545", "band-loss")):
        poly = ax.fill_between(both, [anchor[i] for i in both], [base[i] for i in both],
                               where=where, interpolate=True, color=colour, alpha=0.15, zorder=1)
        poly.set_gid(f"sr-{ANCHOR}-{part}")     # the band belongs to v1: it vanishes with it

    series = {}
    for key, _, _, colour, lw, z, ls in RUNS:
        vals = [data[key].get(t, {}).get(stat) for t in order]
        series[key] = vals
        (ln,) = ax.plot(x, vals, color=colour, lw=lw, ls=ls, zorder=z,
                        marker="o" if key == ANCHOR else None, markersize=3.2,
                        markerfacecolor=colour, markeredgecolor="none")
        ln.set_gid(f"sr-{key}-line")

    # Clip to the bulk rather than squashing everything for one failed pocket; the clipped
    # points are returned so the HTML caption can name them (nothing is written on the curves).
    flat = sorted(v for vals in list(series.values()) + [ref] for v in vals if v is not None)
    lo, hi = flat[0] - 0.4, flat[int(0.995 * (len(flat) - 1))] + 0.5
    clipped = [(label, order[i], v) for key, label, *_ in RUNS
               for i, v in enumerate(series[key]) if v is not None and v > hi]

    ax.set_xlabel("pocket, ordered by Ours · v1 (strongest left)", fontsize=12.3, labelpad=9)
    ax.set_ylabel(f"Vina Dock {stat}   (kcal/mol, lower is better)", fontsize=12.3, labelpad=10)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("target_", "") for t in order], fontsize=7.5, rotation=90)
    ax.yaxis.grid(True, color="#e6e9ef", lw=.85)   # 78 x-ticks: a vertical grid would be a picket fence
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c4cad4")
    fig.tight_layout()

    wins = sum(1 for i in both if anchor[i] <= base[i])
    return fig, wins, clipped


def to_inline_svg(fig, fig_key):
    """Matplotlib SVG with every id namespaced by fig_key, so several figures can share a page."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor="white", bbox_inches="tight")
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg([^>]*?)\s+width="[^"]*"\s+height="[^"]*"', r"<svg\1", svg, count=1)
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{fig_key}-{m.group(1)}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f"url(#{fig_key}-{m.group(1)})", svg)
    svg = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{fig_key}-{m.group(1)}"', svg)
    return svg.strip()


def block(fig_key, svg, items=None):
    """The fig-block wrapper Figures 5-7 use: a toggle row acting as legend, then the SVG.

    ``items`` is [(series key, label, swatch colour), ...]; defaults to this figure's series."""
    if items is None:
        items = [(REF_KEY, REF_LABEL, REF_COLOR)] + [(k, lab, c) for k, lab, _, c, *_ in RUNS]
    toggles = "".join(
        f'<label class="fig-toggle-item"><input type="checkbox" checked data-series="{k}">'
        f'<span class="fig-swatch" style="background:{c}"></span>{lab}</label>'
        for k, lab, c in items)
    return (f'<div class="fig-block" data-fig="{fig_key}">'
            f'<div class="fig-toggle" role="group" aria-label="show or hide a method">{toggles}</div>'
            f'<div class="table-wrap fig-inline-svg" style="padding:16px">{svg}</div></div>')


def splice(html, fig_key, markup):
    """Idempotent replace between markers; first run swaps out the old <img> block."""
    a, b = f"<!-- {fig_key.upper()}:START -->", f"<!-- {fig_key.upper()}:END -->"
    payload = f"  {a}\n  {markup}\n  {b}"
    if a in html:
        return html[:html.index(a)] + payload.lstrip() + html[html.index(b) + len(b):]
    stat = "mean" if fig_key == "fig2" else "median"
    old = re.search(r'  <div class="table-wrap" style="padding:16px"><img '
                    r'src="260827/denovo_vina_by_pocket_%s\.png".*?</div>' % stat, html, re.S)
    if not old:
        raise SystemExit(f"{fig_key}: neither markers nor the original <img> block found")
    return html[:old.start()] + payload + html[old.end():]


def main():
    data = {key: load(root) for key, _, root, *_ in RUNS}
    targets = pocket_set(data)
    html = open(DOC, encoding="utf-8").read()
    for fig_key, stat in (("fig2", "mean"), ("fig3", "median")):
        fig, wins, clipped = render(data, targets, stat, fig_key)
        svg = to_inline_svg(fig, fig_key)
        plt.close(fig)
        html = splice(html, fig_key, block(fig_key, svg))
        print(f"{fig_key} ({stat}): {len(targets)} pockets · Ours v1 stronger on {wins} · "
              f"svg {len(svg) // 1024} KB")
        for label, target, v in clipped:
            print(f"    clipped off the top: {label} @ {target} = {v:+.2f}")
    open(DOC, "w", encoding="utf-8").write(html)
    print(f"spliced into {DOC}")


if __name__ == "__main__":
    main()
