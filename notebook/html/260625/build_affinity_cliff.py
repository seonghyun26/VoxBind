#!/usr/bin/env python
"""Build results_affinity_cliff.html — activity-cliff analysis for the two canonical
Table-1 encoders, C (ViT) vs C+D+G (ChannelViT [7,4,2]), on lp_edrscc_v2.

PRIMARY grouping = target-level (SIFTS UniProt) — the expert MoleculeACE convention;
exact-sequence grouping is kept as a robustness check (§2). Static, no-JS, house style.

Reproduce:
    python voxbind/test/cliff_eval_canonical.py   # sanity + cached preds
    python voxbind/test/cliff_eval_target.py       # both groupings + primary detail
    python notebook/html/260625/build_affinity_cliff.py
"""
import json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/shpark/prj-denovo/VoxBind"
CAN  = json.load(open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"))   # sanity
T    = json.load(open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_target.json"))       # groupings + detail
OUT  = os.path.join(HERE, "results_affinity_cliff.html")

C, D = T["models"]
COL = {C: "#3498db", D: "#2ecc71"}
SAN = CAN["sanity"]
PRI, SEQ = T["groupings"]["uni"], T["groupings"]["seq"]      # primary / comparison
preds, pK = T["preds"], T["pK"]
DET = T["primary_detail"]
NT = T["n_test"]
avg = lambda m, p: float(np.mean(preds[m][p]))
def fmt(t, nd=3): return f"{t[0]:.{nd}f}<span class='sd'>±{t[1]:.{nd}f}</span>"
M = lambda G, m, k: G["metrics"][m][k]                       # metric accessor


# ── grouped bar SVG: 3 metric panels ─────────────────────────────────────────
def bars(panel_specs, w=1000, h=300):
    pad_l, pad_r, pad_t, pad_b = 12, 12, 30, 46
    pw = (w - pad_l - pad_r) / len(panel_specs)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;display:block;'
           f'margin:0 auto" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    for pi, (title, lo, hi, lower_better, vals) in enumerate(panel_specs):
        x0 = pad_l + pi * pw
        plot_l, plot_r = x0 + 40, x0 + pw - 16
        plot_t, plot_b = pad_t, h - pad_b
        for g in range(5):
            v = lo + (hi - lo) * g / 4
            yy = plot_b - (plot_b - plot_t) * (v - lo) / (hi - lo)
            out.append(f'<line x1="{plot_l:.1f}" y1="{yy:.1f}" x2="{plot_r:.1f}" y2="{yy:.1f}" stroke="#e6e9ef" stroke-width="1"/>')
            out.append(f'<text x="{plot_l-4:.1f}" y="{yy+3:.1f}" font-size="9.5" fill="#9aa3b2" text-anchor="end">{v:.2f}</text>')
        nb = len(vals); bw = min(46, (plot_r - plot_l) / (nb + 0.6))
        gap = (plot_r - plot_l - nb * bw) / (nb + 1)
        for bi, (lab, mean, sd, col, best) in enumerate(vals):
            bx = plot_l + gap + bi * (bw + gap)
            bh = (plot_b - plot_t) * (mean - lo) / (hi - lo); by = plot_b - bh
            stroke = ' stroke="#000" stroke-width="2.5"' if best else ''
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{col}"{stroke}/>')
            out.append(f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" font-size="9" fill="#5b6678" text-anchor="middle">{mean:.3f}</text>')
            if sd:
                ey0 = plot_b - (plot_b - plot_t) * (mean - sd - lo) / (hi - lo)
                ey1 = plot_b - (plot_b - plot_t) * (mean + sd - lo) / (hi - lo)
                cx = bx + bw / 2
                out.append(f'<line x1="{cx:.1f}" y1="{ey0:.1f}" x2="{cx:.1f}" y2="{ey1:.1f}" stroke="#566072" stroke-width="1"/>')
        out.append(f'<line x1="{plot_l:.1f}" y1="{plot_b:.1f}" x2="{plot_r:.1f}" y2="{plot_b:.1f}" stroke="#aeb6c4" stroke-width="1.5"/>')
        out.append(f'<text x="{(plot_l+plot_r)/2:.1f}" y="{h-26:.1f}" font-size="12" font-weight="600" fill="#1c2433" text-anchor="middle">{title}</text>')
        tip = ('&#8595; lower is better', '#b04a2f') if lower_better else ('&#8593; higher is better', '#2f6f4f')
        out.append(f'<text x="{(plot_l+plot_r)/2:.1f}" y="{h-12:.1f}" font-size="10" fill="{tip[1]}" text-anchor="middle">{tip[0]}</text>')
    out.append('</svg>')
    return "\n".join(out)

def vlist(G, metric):
    a, b = M(G, C, metric), M(G, D, metric)
    return [(C, a[0], a[1], COL[C], False), (D, b[0], b[1], COL[D], False)]

svg = bars([
    (f"RMSE on cliff mols (n={PRI['n_test_cliff']})", 1.30, 1.75, True,  vlist(PRI, "rmse_cliff")),
    ("RMSE on non-cliff mols",                        1.30, 1.75, True,  vlist(PRI, "rmse_noncliff")),
    (f"Sign-accuracy · test-test pairs (n={PRI['n_tt']})", 0.50, 0.80, False, vlist(PRI, "sign_acc")),
])

# ── §2 robustness comparison rows (exact-seq vs UniProt) ─────────────────────
def grp_rows():
    def row(G, primary):
        sa_c, sa_d = M(G, C, "sign_acc")[0], M(G, D, "sign_acc")[0]
        tag = ' <span class="tag">primary</span>' if primary else ''
        return f"""<tr>
        <td class="col-method">{G['name']}{tag}</td>
        <td><b>{G['n_pairs']}</b></td><td>{G['n_test_cliff']}</td><td>{G['n_tt']}</td>
        <td>{fmt(M(G,C,'rmse_cliff'))}</td><td>{fmt(M(G,D,'rmse_cliff'))}</td>
        <td{' class="best"' if sa_c>=sa_d else ''}>{fmt(M(G,C,'sign_acc'),2)}</td>
        <td{' class="best"' if sa_d>sa_c else ''}>{fmt(M(G,D,'sign_acc'),2)}</td>
        <td>{G['wilcoxon_p']:.2f}</td></tr>"""
    return row(SEQ, False) + row(PRI, True)

mult_p = PRI["n_pairs"] / SEQ["n_pairs"]
mult_tt = PRI["n_tt"] / SEQ["n_tt"]
sa_p_c, sa_p_d = M(PRI, C, "sign_acc")[0], M(PRI, D, "sign_acc")[0]
gap_C = M(PRI, C, "rmse_cliff")[0] - M(PRI, C, "rmse_noncliff")[0]
gap_D = M(PRI, D, "rmse_cliff")[0] - M(PRI, D, "rmse_noncliff")[0]

# ── §3 example cliffs (primary grouping; flag cross-construct) ────────────────
ex = DET["examples"]
n_xc_tt = sum(1 for e in ex if e["hi_split"] == "test" and e["lo_split"] == "test" and e["cross_construct"])
ex_rows = []
for e in ex[:11]:
    hi, lo = e["hi_pid"], e["lo_pid"]
    xc = '<span class="xc" title="same UniProt, different construct — missed by exact-sequence grouping">cross&#8209;construct</span>' if e["cross_construct"] else ""
    ex_rows.append(f"""<tr>
      <td class="mono">{hi} <span class="spl">{e['hi_split']}</span></td>
      <td class="mono">{lo} <span class="spl">{e['lo_split']}</span> {xc}</td>
      <td><b>{e['dpK']:.2f}</b></td>
      <td>{e['hi_pK']:.2f} / {e['lo_pK']:.2f}</td>
      <td class="mono sm">{e['hi_smiles']}<br><span style="color:#9aa3b2">{e['lo_smiles']}</span></td>
    </tr>""")

# test-test pairs ranking (primary grouping, top 14 by dpK)
tt = sorted([(a, b, pK[a] - pK[b] if pK[a] >= pK[b] else pK[b] - pK[a]) for a, b in
             [(x, y) if pK[x] >= pK[y] else (y, x) for x, y in PRI["tt"]]], key=lambda r: -r[2])
sa_rows = []
for hi, lo, dp in tt[:14]:
    cOK, dOK = avg(C,hi) > avg(C,lo), avg(D,hi) > avg(D,lo)
    sa_rows.append(f"""<tr>
      <td class="mono">{hi} &rsaquo; {lo}</td><td>{dp:.2f}</td>
      <td>{avg(C,hi):.2f} / {avg(C,lo):.2f} <span class="rk">{'&#10003;' if cOK else '&#10007;'}</span></td>
      <td>{avg(D,hi):.2f} / {avg(D,lo):.2f} <span class="rk">{'&#10003;' if dOK else '&#10007;'}</span></td>
    </tr>""")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activity-cliff analysis — C·ViT vs C+D+G·ChannelViT</title>
<style>
  :root {{ --ink:#1c2433; --ink-soft:#5b6678; --line:#e3e7ee; --line-strong:#aeb6c4;
    --bg:#f5f6f8; --card:#ffffff; --accent:#2f6f4f; --best-bg:#eaf5ee; --tbd:#b07a17; --tbd-bg:#fcf3e0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
  .page {{ max-width:1180px; margin:0 auto; padding:48px 24px 96px; }}
  header.doc {{ border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:36px; }}
  header.doc .date {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-soft); margin-bottom:6px; }}
  header.doc h1 {{ font-size:27px; font-weight:650; margin:0; letter-spacing:-0.01em; }}
  header.doc .sub {{ font-size:14px; color:var(--ink-soft); margin-top:8px; max-width:940px; }}
  section.block {{ margin-bottom:48px; }}
  h2.section-head {{ font-size:20px; font-weight:640; margin:0 0 14px; display:flex; align-items:center; gap:10px; }}
  h2.section-head .sec-num {{ display:inline-flex; align-items:center; justify-content:center;
    width:26px; height:26px; border-radius:7px; background:var(--ink); color:#fff; font-size:14px; font-weight:600; }}
  .table-title {{ font-size:18px; font-weight:620; margin:0 0 4px; }}
  .table-sub {{ font-size:13.5px; color:var(--ink-soft); margin:0 0 18px; max-width:1010px; }}
  .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:var(--card);
    box-shadow:0 1px 2px rgba(20,30,50,.04), 0 8px 24px rgba(20,30,50,.05); margin-bottom:14px; }}
  table.results {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
  table.results th, table.results td {{ padding:11px 12px; text-align:center; white-space:nowrap; }}
  table.results thead th {{ font-weight:600; color:var(--ink); background:#fafbfc; border-bottom:1px solid var(--line); font-size:12.5px; }}
  table.results tbody tr + tr td {{ border-top:1px solid var(--line); }}
  td.col-method, th.col-method {{ text-align:left !important; min-width:175px; }}
  td.col-method {{ font-weight:560; }}
  .val {{ font-weight:640; }}
  .sd {{ font-size:11px; color:var(--ink-soft); font-weight:400; margin-left:2px; }}
  .pe {{ display:inline-block; padding:1px 7px; border-radius:5px; font-size:11.5px; font-weight:600; }}
  .pe.c {{ background:#eaf1fb; color:#1d4ed8; }} .pe.d {{ background:#e7f6ec; color:#1d7a44; }}
  .tag {{ display:inline-block; padding:0 6px; border-radius:4px; font-size:10px; font-weight:700; background:#eef2fb; color:#1d4ed8; vertical-align:middle; letter-spacing:.03em; text-transform:uppercase; }}
  .xc {{ display:inline-block; padding:0 5px; border-radius:4px; font-size:9.5px; font-weight:700; background:#fdeee0; color:#b4671c; letter-spacing:.02em; }}
  .best {{ background:var(--best-bg); }}
  .legend {{ display:flex; justify-content:center; gap:18px; margin-top:10px; font-size:12.5px; color:var(--ink-soft); flex-wrap:wrap; }}
  .legend span.k {{ display:inline-flex; align-items:center; gap:6px; }}
  .legend i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  .callout {{ border-left:4px solid var(--accent); background:#f3f8f5; border-radius:0 10px 10px 0; padding:14px 18px; font-size:14px; margin:0 0 18px; }}
  .callout.warn {{ border-left-color:var(--tbd); background:var(--tbd-bg); }}
  .callout b {{ color:var(--ink); }}
  ul.notes {{ font-size:14px; color:var(--ink); padding-left:20px; margin:6px 0 0; }}
  ul.notes li {{ margin-bottom:9px; }}
  .mono {{ font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; }}
  .mono.sm {{ font-size:11px; white-space:normal; max-width:340px; word-break:break-all; }}
  .spl {{ font-size:10px; color:#fff; background:#aeb6c4; border-radius:4px; padding:0 5px; margin-left:3px; text-transform:uppercase; letter-spacing:.04em; }}
  .rk {{ font-weight:700; }} td .rk {{ margin-left:4px; }}
  .figcap {{ text-align:center; font-size:12px; color:#5b6678; margin-top:8px; }}
  code {{ background:#eef1f6; padding:1px 6px; border-radius:5px; font-size:12.5px; }}
  a {{ color:#1d4ed8; }}
</style>
</head>
<body>
<div class="page">
  <header class="doc">
    <div class="date">2026-06-25 &middot; VoxBind &middot; supplement to Table&nbsp;1</div>
    <h1>Activity-cliff analysis &mdash; C&nbsp;(ViT) vs C+D+G&nbsp;(ChannelViT)</h1>
    <div class="sub">
      Does the +0.036&nbsp;&rho; binding-affinity advantage of the density+gradmag <b>ChannelViT</b> encoder come from
      <b>activity cliffs</b> &mdash; near-identical ligands with a large potency gap, the regime a coords-only encoder
      should be structurally forced to mis-rank? We run the MoleculeACE consensus-cliff protocol on the two headline
      Table-1 encoders (frozen, <code>lp_edrscc_v2</code> Kd/Ki split), grouping analogs by <b>target (UniProt)</b> as
      MoleculeACE does. <b>Bottom line: no &mdash; the density gain is a bulk effect, not a cliff effect.</b>
    </div>
  </header>

  <section class="block">
    <h2 class="section-head"><span class="sec-num">0</span> Setup &amp; sanity</h2>
    <p class="table-sub">
      Frozen PLINDER-pretrained encoders, cached features, a fresh 3-seed MLP probe head (early-stopped on validation
      Spearman) re-trained to recover per-complex test predictions. The two encoders are <b>exactly</b> Table&nbsp;1's
      headline rows: <b>C&nbsp;(ViT)</b> = <span class="mono sm">atomblob_e99_v5_260622_plinder_otf_coords_mask050</span>,
      <b>C+D+G&nbsp;(ChannelViT&nbsp;[7,4,2])</b> = <span class="mono sm">atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742</span>.
      Cliffs are model-independent &mdash; defined purely by ligand similarity and &Delta;pK on the same test set.
    </p>
    <div class="table-wrap">
      <table class="results">
        <thead><tr>
          <th class="col-method">Encoder</th><th>Input</th><th>Architecture</th>
          <th>Test Pearson&nbsp;<i>r</i></th><th>Test Spearman&nbsp;&rho;</th><th>Table&nbsp;1&nbsp;&rho;</th><th><i>n</i>&nbsp;test</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class="col-method"><span class="pe c">C</span> &nbsp;coords-only</td>
            <td>pocket + ligand</td><td>ViT</td>
            <td>{SAN[C]['r'][0]:.3f}</td><td><span class="val">{SAN[C]['rho'][0]:.3f}</span> <span class="sd">±{SAN[C]['rho'][1]:.3f}</span></td>
            <td>0.605</td><td>{NT}</td>
          </tr>
          <tr>
            <td class="col-method"><span class="pe d">C+D+G</span> &nbsp;+ density + gradmag</td>
            <td>pocket + ligand + density + gradmag</td><td>ChannelViT&nbsp;[7,4,2]</td>
            <td>{SAN[D]['r'][0]:.3f}</td><td><span class="val">{SAN[D]['rho'][0]:.3f}</span> <span class="sd">±{SAN[D]['rho'][1]:.3f}</span></td>
            <td>0.637</td><td>{NT}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="figcap">Reproduces Table&nbsp;1's ranking (&rho;&nbsp;{SAN[C]['rho'][0]:.3f} / {SAN[D]['rho'][0]:.3f} &asymp; 0.605 / 0.637) &rarr; the right two encoders.
      The head is rank-optimized, so absolute RMSE is a head-protocol readout, not Table&nbsp;1's calibrated RMSE &mdash;
      lead with the scale-free sign-accuracy below.</p>
  </section>

  <section class="block">
    <h2 class="section-head"><span class="sec-num">1</span> Cliff result</h2>
    <p class="table-sub">
      <b>Consensus cliffs</b> (van&nbsp;Tilborg&nbsp;2022): a pair of ligands binding the <i>same target</i> with
      consensus structural similarity&nbsp;&ge;&nbsp;0.9 (Tanimoto&nbsp;/&nbsp;Murcko-scaffold&nbsp;/&nbsp;SMILES-Levenshtein)
      <b>and</b> |&Delta;pK|&nbsp;&gt;&nbsp;1 (a&nbsp;&gt;10&times; potency swing). Target = <b>UniProt accession</b>
      (SIFTS), the expert MoleculeACE convention &mdash; so analogs co-crystallised under different protein constructs
      still pair (see &sect;2). Pool = the {NT}-complex test set + train/val analogs of the same targets &rarr;
      <b>{PRI['n_pairs']} cliff pairs</b>, <b>{PRI['n_test_cliff']} test cliff molecules</b>,
      <b>{PRI['n_tt']} test&ndash;test pairs</b> (both members in test &rarr; usable for ranking).
    </p>
    {svg}
    <div class="legend">
      <span class="k"><i style="background:{COL[C]}"></i>C &middot; ViT</span>
      <span class="k"><i style="background:{COL[D]}"></i>C + D + G &middot; ChannelViT</span>
      <span class="k">error bars = &plusmn;1&nbsp;std (3 seeds)</span>
    </div>
    <p class="figcap" style="margin-top:14px">
      <b>Sign-accuracy</b> = fraction of the {PRI['n_tt']} test&ndash;test cliff pairs whose predicted potency order
      matches the truth (scale-free, pure ranking &mdash; the trustworthy cliff metric).
    </p>

    <div class="table-wrap" style="margin-top:16px">
      <table class="results">
        <thead><tr>
          <th class="col-method">Encoder</th>
          <th>RMSE&nbsp;cliff (n={PRI['n_test_cliff']})</th><th>RMSE&nbsp;non-cliff</th><th>cliff &minus; non-cliff</th>
          <th>Sign-acc (n={PRI['n_tt']})</th><th>|err|&nbsp;on cliffs</th><th>roughness&nbsp;&rho;</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class="col-method"><span class="pe c">C</span> ViT</td>
            <td>{fmt(M(PRI,C,'rmse_cliff'))}</td><td>{fmt(M(PRI,C,'rmse_noncliff'))}</td><td>+{gap_C:.3f}</td>
            <td>{fmt(M(PRI,C,'sign_acc'),2)}</td><td>{PRI['aerr'][C]:.3f}</td><td>{DET['roughslope'][C]:+.3f}</td>
          </tr>
          <tr>
            <td class="col-method"><span class="pe d">C+D+G</span> ChannelViT</td>
            <td>{fmt(M(PRI,D,'rmse_cliff'))}</td><td>{fmt(M(PRI,D,'rmse_noncliff'))}</td><td>+{gap_D:.3f}</td>
            <td class="best">{fmt(M(PRI,D,'sign_acc'),2)}</td><td>{PRI['aerr'][D]:.3f}</td><td>{DET['roughslope'][D]:+.3f}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="figcap">
      Paired Wilcoxon on per-cliff |error| (seed-averaged): C&nbsp;{PRI['aerr'][C]:.3f} vs C+D+G&nbsp;{PRI['aerr'][D]:.3f},
      <b>p&nbsp;=&nbsp;{PRI['wilcoxon_p']:.3f}</b> (not significant). &ldquo;roughness&nbsp;&rho;&rdquo; =
      Spearman(|error|, local SAR roughness) over the {DET['n_has_nb']} test mols with a same-target analog (Design&nbsp;B).
    </p>

    <div class="callout warn">
      <b>Density does not help on activity cliffs.</b> C+D+G&nbsp;ChannelViT is <b>+0.036&nbsp;&rho;</b> better overall,
      yet on the {PRI['n_tt']} test&ndash;test cliff pairs it is <b>no better</b>: sign-accuracy ties
      (C+D+G&nbsp;{sa_p_d:.2f}&nbsp;&asymp;&nbsp;C&nbsp;{sa_p_c:.2f}) and per-cliff error is statistically
      indistinguishable (Wilcoxon p&nbsp;=&nbsp;{PRI['wilcoxon_p']:.2f}). The overall affinity advantage lives in the
      <b>{PRI['n_noncliff']} non-cliff bulk</b> complexes, not in resolving near-identical analog pairs.
    </div>
  </section>

  <section class="block">
    <h2 class="section-head"><span class="sec-num">2</span> Robustness &mdash; grouping by sequence vs target</h2>
    <p class="table-sub">
      How a cliff&rsquo;s &ldquo;same target&rdquo; is defined matters. Grouping by <b>exact protein sequence</b> splits
      construct variants apart (truncations, point mutants, species) &mdash; e.g. our nine <b>Factor X</b> analogs span
      <b>five</b> exact-sequence groups, so a real cliff like <span class="mono">2j95</span> (pK&nbsp;8.4) vs
      <span class="mono">2j94</span> (pK&nbsp;6.3) is never paired. Grouping by <b>UniProt target</b> (the primary
      analysis, &sect;1) recovers them. The conclusion is the same under both &mdash; only the statistical power differs.
    </p>
    <div class="table-wrap">
      <table class="results">
        <thead>
          <tr><th class="col-method" rowspan="2">Grouping</th><th colspan="3">Cliff set</th>
              <th colspan="2">RMSE&nbsp;cliff</th><th colspan="2">Sign-accuracy</th><th rowspan="2">Wilcoxon&nbsp;p</th></tr>
          <tr><th>pairs</th><th>test mols</th><th>test&ndash;test</th><th>C</th><th>C+D+G</th><th>C</th><th>C+D+G</th></tr>
        </thead>
        <tbody>
          {grp_rows()}
        </tbody>
      </table>
    </div>
    <div class="callout">
      <b>Target grouping surfaces {mult_p:.1f}&times; more cliffs &mdash; and the verdict is unchanged.</b>
      Moving from exact-sequence to UniProt grows the set {SEQ['n_pairs']}&nbsp;&rarr;&nbsp;{PRI['n_pairs']} pairs
      (test&ndash;test {SEQ['n_tt']}&nbsp;&rarr;&nbsp;{PRI['n_tt']}; {n_xc_tt} of the {PRI['n_tt']} are
      <span class="xc">cross&#8209;construct</span>, invisible to sequence grouping). On the larger, more robust set the
      sign-accuracy is a clean <b>tie</b> (C+D+G&nbsp;{sa_p_d:.2f}&nbsp;&asymp;&nbsp;C&nbsp;{sa_p_c:.2f}); the apparent
      &ldquo;density ranks worse&rdquo; under exact-sequence (0.63&nbsp;vs&nbsp;0.71) was a {SEQ['n_tt']}-pair artifact.
      Density neither helps nor harms cliff ranking under either definition.
    </div>
    <p class="figcap">Identical encoders, split, seeds, and predictions throughout &mdash; only the cliff <i>definition</i>
      changes. The target grouping was motivated by the <a href="#mace">MoleculeACE feasibility check</a> (&sect;4).</p>
  </section>

  <section class="block">
    <h2 class="section-head"><span class="sec-num">3</span> Where the cliffs are</h2>
    <p class="table-sub">Largest-&Delta;pK consensus cliffs (UniProt grouping) &mdash; the cliff chemistry.
      <span class="xc">cross&#8209;construct</span> marks pairs the old exact-sequence grouping would have missed;
      per-pair ranking is in the next table.</p>
    <div class="table-wrap">
      <table class="results">
        <thead><tr>
          <th class="col-method">Potent (hi)</th><th class="col-method">Weak (lo)</th><th>&Delta;pK</th>
          <th>pK hi / lo</th><th class="col-method">SMILES (hi / lo)</th>
        </tr></thead>
        <tbody>
          {''.join(ex_rows)}
        </tbody>
      </table>
    </div>
    <p class="figcap">The biggest cliffs are functional-group / connectivity swaps &mdash; differences the coords +
      atom-type channel already resolves; e.g. DHFR <b>1c7f/1aku</b> and thrombin <b>3dux/3qto</b> are cross-construct
      pairs only the target grouping pairs.</p>

    <p class="table-title" style="margin-top:26px">Test&ndash;test cliff pairs &mdash; per-pair ranking (top 14 by &Delta;pK)</p>
    <div class="table-wrap">
      <table class="results">
        <thead><tr>
          <th class="col-method">pair (hi &rsaquo; lo)</th><th>&Delta;pK</th>
          <th>C&nbsp;(ViT) hi/lo</th><th>C+D+G&nbsp;(ChannelViT) hi/lo</th>
        </tr></thead>
        <tbody>
          {''.join(sa_rows)}
        </tbody>
      </table>
    </div>
  </section>

  <section class="block" id="mace">
    <h2 class="section-head"><span class="sec-num">4</span> Aside &mdash; can we get density for the MoleculeACE cliffs?</h2>
    <p class="table-sub">
      The target grouping was prompted by a feasibility check: could we run this test on the <b>expert-curated</b>
      MoleculeACE cliffs (30 ChEMBL targets, 48,714 molecules) with experimental density, rather than self-extracted
      PDBbind cliffs? We mapped every benchmark ligand <span class="mono sm">SMILES &rarr; InChIKey &rarr; wwPDB CCD code
      (50,507 ligands) &rarr; co-crystal with the right target &rarr; structure factors</span>.
    </p>
    <ul class="notes">
      <li><b>A wholesale density-MoleculeACE is impossible.</b> Only <b>2.7%</b> of benchmark molecules exist as a PDB
        ligand at all (1.4% of cliff mols) &mdash; most ChEMBL assay compounds were never crystallised. Across the 12
        best-crystallised targets, expert-cliffs&nbsp;&cap;&nbsp;density yields just <b>17 cliff pairs / 25 molecules</b>,
        ~94% from <b>Factor X + thrombin</b> (PPARs/kinases give many density-backed mols but no near-identical SAR pairs).</li>
      <li><b>And it is mostly already ours.</b> <b>21/25</b> entries are already in PDBbind; <b>19/25</b> already have
        ChannelViT features &mdash; a standalone MoleculeACE-density benchmark would add little.</li>
      <li><b>But it exposed the grouping choice.</b> None of the 17 MoleculeACE pairs coincided with our 42 exact-sequence
        pairs, yet 15/25 molecules were already in our test set &mdash; because MoleculeACE groups by <i>target</i> and we
        had grouped by <i>exact sequence</i>. That is what motivated adopting UniProt grouping as the primary analysis
        (&sect;1&ndash;2): {mult_p:.1f}&times; more cliffs from data we already had. Pipeline + manifest:
        <span class="mono sm">voxbind/dataset/moleculeace_pdb_map.py</span>,
        <span class="mono sm">dataset/data/moleculeace_density/</span>.</li>
    </ul>
  </section>

  <section class="block">
    <h2 class="section-head"><span class="sec-num">5</span> Interpretation</h2>
    <ul class="notes">
      <li><b>The +0.036&nbsp;&rho; density gain is a bulk effect, not a cliff effect.</b> C+D+G&nbsp;ChannelViT beats
        coords overall but is <i>not better</i> on cliffs &mdash; RMSE&nbsp;cliff n.s. (Wilcoxon p&nbsp;=&nbsp;{PRI['wilcoxon_p']:.2f}
        on the {PRI['n_tt']}-pair set) and sign-accuracy ties (C+D+G&nbsp;{sa_p_d:.2f}&nbsp;&asymp;&nbsp;C&nbsp;{sa_p_c:.2f}).
        Robust to the grouping choice (&sect;2).</li>
      <li><b>This reverses the 260618 PLINDER cliff finding.</b> The earlier C/C+D+G fused-ViT analysis on the
        IC50-inclusive split showed density <i>helping</i> cliff magnitude (RMSE&nbsp;cliff 1.78&nbsp;&rarr;&nbsp;1.55,
        p&nbsp;=&nbsp;0.016). On the canonical Kd/Ki conditions that gain is gone &mdash; consistent with the 260623
        result that most of the v1 density advantage was density absorbing <b>IC50 label-noise</b>, not binding signal.</li>
      <li><b>These cliffs are atom-swap / connectivity changes coords already encode.</b> The biggest are
        functional-group swaps (sulfonate&nbsp;&harr;&nbsp;phosphonate) or ring-rearrangement isomers (DHFR 1c7f/1aku)
        &mdash; visible to the coords + atom-type channel. Experimental density adds <b>calibration on hard-to-fit bulk
        complexes</b>, not discrimination between near-identical analogs. (Echoes the B-factor / H-bond probes: those,
        too, were largely coords-recoverable.)</li>
      <li><b>Caveat &mdash; RMSE is head-dependent.</b> The rank-optimized probe head reproduces Table-1&nbsp;&rho; but not
        its calibrated RMSE, so we lead with the scale-free sign-accuracy. Grouping sensitivity is settled in &sect;2:
        target-level grouping gives {mult_p:.1f}&times; the pairs and the conclusion is unchanged.</li>
    </ul>
  </section>

  <footer style="border-top:1px solid var(--line); padding-top:16px; font-size:12px; color:var(--ink-soft);">
    Reproduce: <code>python voxbind/test/cliff_eval_canonical.py</code> &amp; <code>cliff_eval_target.py</code> &rarr;
    JSON artifacts, then <code>python notebook/html/260625/build_affinity_cliff.py</code>. Cliff protocol vendored from
    MoleculeACE (van&nbsp;Tilborg&nbsp;et&nbsp;al., JCIM&nbsp;2022); UniProt grouping via SIFTS; split <code>{T['scheme']}</code>.
  </footer>
</div>
</body>
</html>
"""
open(OUT, "w").write(HTML)
print("wrote", OUT, f"({len(HTML)} bytes)")
