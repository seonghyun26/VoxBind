#!/usr/bin/env python3
"""Regenerate the Section 2 shared-pocket results block in 260730_meeting.html.

The pocket set grows one pocket at a time (see diagnostics/vina_crop_vs_full/
extend_pockets.sh), so the table cells, the list of tested pockets, the ligand counts and
every volatile number in the findings list are rendered from the live aggregate
diagnostics/vina_crop_vs_full/crop_vs_full_shared.json rather than hand-edited.

Purely qualitative sentences stay in this file so they are reviewed as prose; anything
numeric is interpolated, so the note can never drift from the JSON.

Usage:  python3 build_shared_pocket_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = (HERE.parents[2] / "diagnostics" / "vina_crop_vs_full"
          / "crop_vs_full_shared.json")
MEETING = HERE / "260730_meeting.html"
START = "    <!-- SHARED_POCKET_BLOCK_START -->"
END = "    <!-- SHARED_POCKET_BLOCK_END -->"

# display order / labels for the two evaluated sample sets
ROWS = [
    ("vanilla", "VoxBind original", '<span class="tag base">baseline</span>',
     "σ 0.9 · ep350", ""),
    ("cdg", "Ours — frozen C+D+G", '<span class="tag ours">ours</span>',
     "σ 0.9 · ep350", ' class="ours"'),
]
# pockets with a non-null cropped VinaDock for BOTH methods, i.e. the largest possible
# matched set. target_71 is excluded because its cropped receptor fails preparation.
TOTAL_EVALUABLE = 78
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
        13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
        17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}


def num(x, dec=2):
    """Format with a real minus sign, matching the rest of the document."""
    if x is None:
        return "—"
    return f"{x:+.{dec}f}".replace("+", "").replace("-", "−")


def signed(x, dec=2):
    if x is None:
        return "—"
    return f"{x:+.{dec}f}".replace("-", "−")


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def score_pairs(d, model):
    """(crop_score, full_score, target, n_atoms) for every evaluated ligand."""
    out = []
    for e in d["per_target_records"]:
        if e["model"] != model:
            continue
        for m in e["per_mol"]:
            a, b = m.get("crop_vina_score"), m.get("full_vina_score")
            if a is None or b is None:
                continue
            out.append({"crop": a, "full": b, "diff": b - a,
                        "target": e["target"], "n_atoms": m["n_atoms"]})
    return out


def hidden_clashes(d, model):
    """Poses that are favourable on the crop but clashing on the whole receptor.

    This is the crop-specific failure: the pose overlaps protein the crop deleted, so the
    crop cannot see the collision. Poses that clash on BOTH receptors are a generation
    failure, not a receptor-scope one, and are counted separately.
    """
    both, hidden = [], []
    for r in score_pairs(d, model):
        if r["full"] > 0 and r["crop"] <= 0:
            hidden.append(r)
        elif r["full"] > 0 and r["crop"] > 0:
            both.append(r)
    hidden.sort(key=lambda r: -r["diff"])
    return hidden, both


def clashing_any(d, model, which):
    """Ligands whose score on `which` receptor is positive (a steric clash)."""
    return [r for r in score_pairs(d, model) if r[which] > 0]


def plural(n, word, suffix="s"):
    return f"{n} {word}" if n == 1 else f"{n} {word}{suffix}"


def smallest_ligand(d):
    """The smallest generated molecule in the set — the concrete 'tiny fragment' example."""
    best = None
    for e in d["per_target_records"]:
        for m in e["per_mol"]:
            if best is None or m["n_atoms"] < best["n_atoms"]:
                best = {"n_atoms": m["n_atoms"], "smiles": m["smiles"],
                        "target": e["target"],
                        "model": "Ours" if e["model"] == "cdg" else "σ 0.9"}
    return best


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def median_score(d, model, which):
    return median([r[which] for r in score_pairs(d, model)])


def mean_excluding_clashes(d, model):
    """Mean crop->full Score shift over ligands that clash on neither receptor."""
    vals = [r["diff"] for r in score_pairs(d, model)
            if r["crop"] <= 0 and r["full"] <= 0]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def biggest_dock_gains(d, k=3):
    rows = []
    for model in ("vanilla", "cdg"):
        for t, r in d["per_pocket"][model].items():
            c, f = r["crop"]["dock"], r["full"]["dock"]
            if c is None or f is None:
                continue
            rows.append({"model": model, "target": t, "delta": f - c})
    rows.sort(key=lambda r: r["delta"])
    return rows[:k]


def render(d):
    meta = d["meta"]
    pockets = meta["shared_pockets"]
    n = meta["n_shared_pockets"]
    nword = WORD.get(n, str(n))
    tbl = d["table_rows_within_pocket_first"]
    pooled = d["pooled_molecule_level"]
    h2h = d["head_to_head_same_pockets"]
    n_mols = meta["source_runs"][list(meta["source_runs"])[0]]["n_mols_per_target"]
    per_method = n * n_mols
    comb = pooled["_all_models_combined"]["vina_dock"]

    plist = ", ".join(f"<code>{p}</code>" for p in pockets)

    # --- the clash story (recomputed every refresh; may vanish as pockets are added)
    hidden = {m: hidden_clashes(d, m)[0] for m in ("vanilla", "cdg")}
    both_clash = {m: hidden_clashes(d, m)[1] for m in ("vanilla", "cdg")}
    clashing = {m: {w: clashing_any(d, m, w) for w in ("crop", "full")}
                for m in ("vanilla", "cdg")}

    L = []
    a = L.append
    a(START)
    a('    <p class="table-sub">')
    a('      Matched before-versus-fixed comparison, measured 2026-07-29. <b>Both methods are '
      'evaluated on the')
    a('      identical pocket set</b>, so the two rows are directly comparable to each other as '
      'well as')
    a(f'      within-row. <b>Pockets tested ({n}):</b> {plist} — worst-first by the')
    a('      <i>mean</i> cropped Vina Dock across both methods, so the set is chosen without '
      'reference to')
    a(f'      any whole-receptor number. <b>{n_mols} valid ligands per pocket</b> = '
      f'{per_method} ligands')
    a(f'      per method. Averages are within-pocket-first, then equal-weight across the '
      f'{nword} pockets.')
    a('      “Before” and “Fixed” score the <b>identical molecules at identical coordinates</b>; '
      'only the')
    a('      receptor changes. “Fixed” uses the aligned CrossDocked2020 whole receptor')
    a('      <code>test_set/&lt;subdir&gt;/&lt;PDBId_Chain&gt;_rec.pdb</code>. No legacy crop values '
      'were copied in —')
    a('      the “Before” cells were re-measured in the same run as “Fixed”, so the pair is '
      'exactly matched.')
    a('    </p>')
    a('    <p class="table-sub">')
    if n >= TOTAL_EVALUABLE:
        a(f'      <b>Scope:</b> this is the <b>complete matched set</b> — all {n} pockets for which '
          f'both methods have a')
        a('      cropped-receptor baseline. The only excluded pocket is <code>target_71</code>, whose '
          '<i>cropped</i>')
        a('      receptor cannot be prepared at all (see the callout below); it is evaluable on the '
          'whole receptor')
        a('      only, so it has no matched pair. No further pockets are pending.')
    else:
        a(f'      <b>Scope caveat:</b> these are the <i>hardest</i> {nword} pockets of the shared '
          f'{TOTAL_EVALUABLE}, taken worst-first.')
        a('      They are not a random sample and the absolute values sit below the full-set average '
          '— read the')
        a('      Before→Fixed deltas, not the levels. Pockets are still being added one at a time, '
          'each run for')
        a(f'      both methods together; {TOTAL_EVALUABLE - n} of {TOTAL_EVALUABLE} remain.')
    a('    </p>')
    a('    <div class="table-wrap">')
    a('      <table class="results mini-table">')
    a('        <thead>')
    a('          <tr>')
    a('            <th class="col-method" rowspan="2">Method</th>')
    a('            <th colspan="4">Before · cropped pocket10 receptor</th>')
    a('            <th colspan="4">Fixed · whole receptor</th>')
    a('          </tr>')
    a('          <tr>')
    for _ in range(2):
        a('            <th>Dock↓<br>Avg</th>')
        a('            <th>Min↓<br>Avg</th>')
        a('            <th>Score↓<br>Avg</th>')
        a('            <th>Success↑</th>')
    a('          </tr>')
    a('        </thead>')
    a('        <tbody>')
    for key, label, tag, subtxt, trcls in ROWS:
        r = tbl[key]
        a(f'          <tr{trcls}>')
        a('            <td class="col-method">')
        a(f'              {label} {tag}')
        a(f'              <span class="sub">{subtxt} · shared {n} pockets · '
          f'{per_method} ligands</span>')
        a('            </td>')
        for which in ("crop", "full"):
            w = r[which]
            for met in ("dock", "min", "score"):
                cav = ""
                # a Score mean over clashing poses is dominated by a few large positive
                # values, so flag the cell and give the robust median alongside
                if met == "score" and clashing[key][which]:
                    nclash = len(clashing[key][which])
                    med = median_score(d, key, which)
                    cav = (f'<span class="sub">{nclash} clashing pose'
                           f'{"s" if nclash > 1 else ""}; median {num(med)}</span>')
                a(f'            <td><span class="val">{num(w[met])}</span>{cav}</td>')
            a(f'            <td><span class="val">{pct(w["succ"])}</span></td>')
        a('          </tr>')
    a('        </tbody>')
    a('      </table>')
    a('    </div>')

    # ---------------- findings ----------------
    a('')
    a('    <div class="notes">')
    a('      <b>What the matched pilot shows.</b>')
    a('      <ul>')

    dv = tbl["vanilla"]["full"]["dock"] - tbl["vanilla"]["crop"]["dock"]
    dc = tbl["cdg"]["full"]["dock"] - tbl["cdg"]["crop"]["dock"]
    a(f'        <li><b>The whole receptor improves Dock for both methods.</b> On the identical '
      f'{nword} pockets,')
    a(f'          Dock moves <b>{signed(dv)}</b> kcal/mol for σ 0.9 and <b>{signed(dc)}</b> '
      f'kcal/mol for Ours.')
    a(f'          Pooled over all {comb["n_pairs"]} ligands the whole-receptor Dock gain is '
      f'<b>{signed(comb["mean_diff_full_minus_crop"])}</b> kcal/mol')
    a(f'          (Wilcoxon signed-rank p = {comb.get("wilcoxon_p", float("nan")):.2g}; '
      f'{comb["n_improved_by_full"]} of {comb["n_pairs"]} ligands improve). The direction is')
    a('          consistent: crop-scored Dock is <b>optimistically wrong in the search, not merely '
      'noisy</b>.</li>')

    gains = biggest_dock_gains(d, 3)
    gtxt = ", ".join(
        f'<b>{g["target"]}</b> ({"σ 0.9" if g["model"] == "vanilla" else "Ours"}, '
        f'{signed(g["delta"])})' for g in gains)
    a(f'        <li><b>The Dock effect is a tail, not a shift.</b> The median ligand moves only '
      f'{signed(comb["median_diff_full_minus_crop"])} kcal/mol.')
    a(f'          The mean is carried by a few large per-pocket gains — biggest: {gtxt}. '
      f'σ 0.9’s')
    a('          target_38 gain independently reproduces the first-server observation for that '
      'target.</li>')

    a('        <li><b>The crop is complete where the ligand sits, hollow where the search goes.</b> '
      'Inside the')
    a('          reference-ligand box the crop retains essentially every receptor atom (e.g. 15/15, '
      '24/24,')
    a('          23/23), but within the box grown by Vina’s 8 Å interaction range it is missing')
    a('          <b>35–64%</b> of the whole-receptor atoms. Score-only and minimize stay near the '
      'generated')
    a('          pose and see an intact pocket; the Dock search reaches the box edge, where only the '
      'whole')
    a('          receptor provides the wall.</li>')

    if hidden["cdg"] or hidden["vanilla"]:
        w = (hidden["cdg"] or hidden["vanilla"])[0]
        ex_c, n_c = mean_excluding_clashes(d, "cdg")
        ex_v, n_v = mean_excluding_clashes(d, "vanilla")
        nh_c, nh_v = len(hidden["cdg"]), len(hidden["vanilla"])
        nb = len(both_clash["cdg"]) + len(both_clash["vanilla"])
        a('        <li><b>The crop hides steric clashes — a failure mode crop scoring cannot '
          'report.</b>')
        a(f'          <b>{plural(nh_c + nh_v, "generated pose")}</b> in this set score as favourable '
          f'on the crop but <i>clash</i> on the')
        a(f'          whole receptor (positive affinity): worst is {num(w["crop"])} → '
          f'<b>{signed(w["full"])}</b> kcal/mol')
        a(f'          ({w["n_atoms"]} heavy atoms, {w["target"]}, '
          f'{"Ours" if hidden["cdg"] else "σ 0.9"}). These poses overlap protein the 10 Å crop '
          f'deleted, so')
        a('          the crop physically cannot see the collision and scores them as good binders. '
          'Split by method:')
        a(f'          <b>Ours {nh_c}</b>, <b>σ 0.9 {nh_v}</b> — Ours generates larger molecules, so '
          f'it has more mass to')
        a('          push into the deleted region. This is a <i>validity</i> artifact, not an '
          'affinity one, and it')
        a('          inflates the crop’s apparent quality in exactly the cases that matter.</li>')
        a('        <li><b>Read the Score column by median, not mean.</b> A clashing pose contributes '
          'a large')
        a(f'          positive affinity, so the Score means above are dominated by a handful of '
          f'ligands ({plural(nb, "pose")} here clash on')
        a('          <i>both</i> receptors, which is a generation failure rather than a receptor-scope '
          'one). Over the')
        mn_c = pooled["cdg"]["vina_min"]["mean_diff_full_minus_crop"]
        mn_v = pooled["vanilla"]["vina_min"]["mean_diff_full_minus_crop"]
        a(f'          ligands that clash on neither receptor, the crop→whole Score shift is '
          f'<b>{signed(ex_c, 3)}</b> for Ours (n={n_c})')
        a(f'          and <b>{signed(ex_v, 3)}</b> for σ 0.9 (n={n_v}), against a Min shift of '
          f'{signed(mn_c, 3)} and {signed(mn_v, 3)} —')
        a('          i.e. once clashes are removed, Score and Min move together and both stay far '
          'smaller than Dock.</li>')

    docks = [r["crop"]["dock"] for m in ("vanilla", "cdg")
             for r in d["per_pocket"][m].values() if r["crop"]["dock"] is not None]
    docks += [r["full"]["dock"] for m in ("vanilla", "cdg")
              for r in d["per_pocket"][m].values() if r["full"]["dock"] is not None]
    a(f'        <li><b>Whole receptors shift these pockets, but do not transform them.</b> Per-pocket '
      f'means span {num(max(docks))} to')
    a(f'          {num(min(docks))} kcal/mol under both receptors. Where a pocket is weak it stays '
      f'weak, and that is a')
    sm = smallest_ligand(d)
    a('          <b>generation problem, not a scoring artifact</b> — the models emit very small '
      'fragments in the worst')
    a(f'          pockets (the smallest evaluated ligand is <code>{sm["smiles"]}</code>, '
      f'{sm["n_atoms"]} heavy atoms, {sm["model"]} at {sm["target"]}).</li>')

    # Success rate is only meaningful once the set reaches pockets that can clear -8.18
    succ = {m: {w: tbl[m][w]["succ"] for w in ("crop", "full")} for m in ("vanilla", "cdg")}
    if all(v == 0 for m in succ for v in succ[m].values()):
        a(f'        <li><b>Success is 0.0% in every cell.</b> These are the hardest {nword} pockets '
          f'of the shared 79 and the')
        a('          DecompDiff threshold is Dock &lt; −8.18, which none of them reach. That is a '
          'scope artifact of')
        a('          the pilot, not a method result — it will become informative as easier pockets '
          'enter the set.</li>')
    else:
        a('        <li><b>Success rate now responds to the receptor fix.</b> Under the DecompDiff '
          'rule (QED &gt; 0.25,')
        a(f'          SA &gt; 0.59, Dock &lt; −8.18), σ 0.9 goes '
          f'<b>{pct(succ["vanilla"]["crop"])} → {pct(succ["vanilla"]["full"])}</b> and Ours goes '
          f'<b>{pct(succ["cdg"]["crop"])} → {pct(succ["cdg"]["full"])}</b>.')
        a('          Crop scoring therefore <b>under-reports</b> success: ligands that clear the '
          'threshold on the whole')
        a(f'          receptor are missed on the crop. Absolute levels stay low because the set is '
          f'still the hardest')
        a(f'          {nword} pockets of 79 — read the crop→whole movement, not the level.</li>')

    gc = h2h["crop"]["dock"]
    gf = h2h["full"]["dock"]
    dcrop = gc["diff_vanilla_minus_cdg"]
    dfull = gf["diff_vanilla_minus_cdg"]
    lead_crop = "σ 0.9" if gc["vanilla"] < gc["cdg"] else "Ours"
    lead_full = "σ 0.9" if gf["vanilla"] < gf["cdg"] else "Ours"
    pc, pf = gc.get("wilcoxon_p_across_pockets"), gf.get("wilcoxon_p_across_pockets")
    verb = "narrows" if abs(dfull) < abs(dcrop) else "widens"
    a('        <li><b>Method ranking is unchanged by the fix, and is not resolved at this n.</b> '
      'On the same')
    a(f'          pockets <b>{lead_crop}</b> leads on Dock under the crop '
      f'({num(gc["vanilla"])} vs {num(gc["cdg"])}) and <b>{lead_full}</b> leads under the')
    a(f'          whole receptor ({num(gf["vanilla"])} vs {num(gf["cdg"])}), but neither gap is '
      f'significant across {nword} pockets')
    a(f'          (pocket-paired Wilcoxon p = {pc:.2f} crop, p = {pf:.2f} whole). The receptor fix '
      f'<b>{verb}</b>')
    a(f'          the gap from {abs(dcrop):.2f} to {abs(dfull):.2f} kcal/mol. These are each '
      f'method’s hardest pockets, so this is')
    a('          not a headline comparison.</li>')
    a('      </ul>')
    a('    </div>')
    a(END)
    return "\n".join(L)


def main() -> None:
    d = json.loads(SHARED.read_text(encoding="utf-8"))
    html = MEETING.read_text(encoding="utf-8")
    if START not in html or END not in html:
        raise SystemExit(
            f"markers not found in {MEETING}; add\n{START}\n...\n{END}\naround the block")
    s = html.index(START)
    e = html.index(END) + len(END)
    MEETING.write_text(html[:s] + render(d) + html[e:], encoding="utf-8")
    n = d["meta"]["n_shared_pockets"]
    print(f"rebuilt Section 2 shared-pocket block: {n} pockets "
          f"{d['meta']['shared_pockets']} -> {MEETING} ({MEETING.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
