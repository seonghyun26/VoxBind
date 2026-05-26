"""Build the HTML report at notebook/html/260526_pretraining_meeting_summary.html."""
import json, os, glob

EXPS = "/home/shpark/prj-denovo/VoxBind/voxbind/exps"
OUT  = "/home/shpark/prj-denovo/VoxBind/notebook/html/260526_pretraining_meeting_summary.html"

with open("/tmp/curves.json") as f:
    curves = json.load(f)

def downsample(rows, stage, key="loss", every=5):
    pts = [(r["epoch"], r[key]) for r in rows
           if r.get("stage")==stage and r.get("epoch") is not None
           and key in r and r[key] is not None]
    if not pts: return []
    last_ep = pts[-1][0]
    keep = [p for p in pts if p[0] % every == 0 or p[0] == last_ep]
    seen, out = set(), []
    for ep, v in keep:
        if ep not in seen:
            out.append((ep, v)); seen.add(ep)
    return out

def sparkline(points, w=260, h=52, color="#2b6cb0"):
    if not points: return ""
    xs, ys = zip(*points)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin: ymax = ymin + 1e-9
    pad_x, pad_y = 8, 8
    def xpx(x): return pad_x + (x - xs[0]) / max(1, (xs[-1]-xs[0])) * (w - 2*pad_x)
    def ypx(y): return h - pad_y - (y - ymin) / (ymax - ymin) * (h - 2*pad_y)
    poly = " ".join(f"{xpx(x):.1f},{ypx(y):.1f}" for x, y in points)
    last_x, last_y = xpx(xs[-1]), ypx(ys[-1])
    first_x, first_y = xpx(xs[0]), ypx(ys[0])
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="background:#fafafa;border:1px solid #e2e8f0;border-radius:4px;vertical-align:middle">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{poly}"/>'
        f'<circle cx="{first_x}" cy="{first_y}" r="2.4" fill="{color}" opacity="0.55"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="3" fill="{color}"/>'
        f'<text x="{w-6}" y="{h-7}" text-anchor="end" font-size="10" fill="#4a5568">'
        f'ep{int(xs[-1])} · {ys[-1]:.3f}</text>'
        f'<text x="{pad_x}" y="{h-7}" text-anchor="start" font-size="10" fill="#a0aec0">'
        f'{ys[0]:.2f}→</text>'
        f'</svg>'
    )

sp = {
    "cnn_mae_val_loss":     sparkline(downsample(curves["cnn_mae"],    "val", "loss"),         color="#dd6b20"),
    "vit_mae_val_loss":     sparkline(downsample(curves["vit_mae"],    "val", "loss"),         color="#3182ce"),
    "vit_electra_val_rtd":  sparkline(downsample(curves["vit_electra"],"val", "pretext_loss"), color="#805ad5"),
    "vit_electra_val_str":  sparkline(downsample(curves["vit_electra"],"val", "L_str"),        color="#805ad5"),
    "down_cnn_miou":        sparkline(downsample(curves["down_cnn"],   "val", "miou"),         color="#dd6b20"),
    "down_vit_miou":        sparkline(downsample(curves["down_vit"],   "val", "miou"),         color="#3182ce"),
    "down_electra_miou":    sparkline(downsample(curves["down_electra"],"val","miou"),         color="#805ad5"),
}

def load_arm(root):
    out = {}
    for d in glob.glob(f"{root}/target_*"):
        name = os.path.basename(d)
        mpath = f"{d}/metrics.json"
        if not os.path.exists(mpath): continue
        try:
            with open(mpath) as f: m = json.load(f)
        except Exception: continue
        out[name] = m.get("aggregates", {})
    return out

A = {
    "baseline": load_arm(f"{EXPS}/260518_voxbind_10k_baseline/samples/res_ep99_test"),
    "aligned":  load_arm(f"{EXPS}/260519_voxbind_10k_density_aligned/samples/res_test_n10_ep138"),
    "vit_mae":  load_arm(f"{EXPS}/260523_voxbind_10k_density_vit_mae_frozen_e0379_snap/samples/res_ep379_fulltest"),
    "repro":    load_arm(f"{EXPS}/reproduction/samples/res_repro_fulltest"),
}

def overlap_means(arm_a, arm_b, field):
    keys = sorted(set(arm_a.keys()) & set(arm_b.keys()))
    xa = [arm_a[k].get(field) for k in keys if arm_a[k].get(field) is not None]
    xb = [arm_b[k].get(field) for k in keys if arm_b[k].get(field) is not None]
    if not xa or not xb: return None
    return len(keys), sum(xa)/len(xa), sum(xb)/len(xb)

def all_mean(arm, field):
    xs = [v.get(field) for v in arm.values() if v.get(field) is not None]
    return (sum(xs)/len(xs)) if xs else None

vm_vs_rep   = {f: overlap_means(A["vit_mae"], A["repro"],   f) for f in ["vina_score_mean","vina_dock_mean","high_affinity"]}
vm_vs_align = {f: overlap_means(A["vit_mae"], A["aligned"], f) for f in ["vina_score_mean","vina_dock_mean","high_affinity"]}
vm_vs_base  = {f: overlap_means(A["vit_mae"], A["baseline"],f) for f in ["vina_score_mean","vina_dock_mean","high_affinity"]}
align_vs_base = {f: overlap_means(A["aligned"], A["baseline"], f) for f in ["vina_score_mean","vina_dock_mean","high_affinity"]}

def fmt(x, p=2): return "--" if x is None else f"{x:.{p}f}"

html_parts = []
html_parts.append('<!doctype html><html lang="en"><head>')
html_parts.append('<meta charset="utf-8"/>')
html_parts.append('<title>VoxBind pre-training - research meeting summary (2026-05-26)</title>')
html_parts.append('''<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 980px; margin: 2.2em auto; padding: 0 1.5em; color: #1a202c; line-height: 1.45; }
  h1 { font-size: 1.5em; margin-bottom: 0.2em; border-bottom: 2px solid #2d3748; padding-bottom: 0.3em; }
  h2 { font-size: 1.15em; margin-top: 1.8em; border-left: 4px solid #2b6cb0; padding-left: 0.6em; color: #2d3748; }
  h3 { font-size: 1.0em; margin-top: 1.4em; color: #2d3748; }
  h4 { font-size: 0.92em; margin: 0.7em 0 0.3em; color: #4a5568; }
  .meta { color: #718096; font-size: 0.85em; margin-bottom: 1.5em; }
  .tldr { background: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 0.8em 1em;
          border-radius: 4px; margin: 1em 0; font-size: 0.96em; }
  .key  { background: #f0fff4; border-left: 4px solid #38a169; padding: 0.6em 0.9em;
          border-radius: 4px; margin: 0.8em 0; font-size: 0.9em; }
  .warn { background: #fffaf0; border-left: 4px solid #dd6b20; padding: 0.6em 0.9em;
          border-radius: 4px; margin: 0.8em 0; font-size: 0.9em; }
  table { border-collapse: collapse; width: 100%; margin: 0.7em 0; font-size: 0.86em; }
  th, td { padding: 5px 8px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }
  th { background: #f7fafc; font-weight: 600; color: #2d3748; }
  td.num { font-family: "SF Mono", Menlo, monospace; text-align: right; }
  th.num { text-align: right; }
  .good { color: #38a169; font-weight: 600; }
  .bad  { color: #e53e3e; font-weight: 600; }
  .neut { color: #4a5568; }
  code { background: #edf2f7; padding: 0.08em 0.35em; border-radius: 3px; font-size: 0.88em; }
  pre  { background: #f7fafc; padding: 0.7em 1em; border-radius: 4px; overflow-x: auto;
         font-size: 0.82em; border: 1px solid #e2e8f0; }
  .appendix { margin-top: 3em; padding-top: 1em; border-top: 2px dashed #cbd5e0;
              color: #4a5568; font-size: 0.92em; }
  .appendix h2 { border-left-color: #a0aec0; }
  .footnote { font-size: 0.8em; color: #718096; margin-top: 0.4em; }
  ol li, ul li { margin: 0.25em 0; }
</style></head><body>''')

html_parts.append('<h1>VoxBind density-encoder pre-training - meeting summary</h1>')
html_parts.append('<div class="meta">Hyun &middot; 2026-05-26 &middot; branch <code>impl/electra</code></div>')

# TL;DR
ha_vm  = vm_vs_rep['high_affinity'][1]*100
ha_rep = vm_vs_rep['high_affinity'][2]*100
ddock_align = vm_vs_align['vina_dock_mean'][1] - vm_vs_align['vina_dock_mean'][2]
ha_align_diff = (vm_vs_align['high_affinity'][1] - vm_vs_align['high_affinity'][2])*100
html_parts.append(f'''<div class="tldr">
<b>Headline.</b> A 4.5M-param 3-D <b>ViT-MAE</b> pre-trained on synthetic electron density, then dropped <i>frozen</i> into VoxBind's density branch and trained on a <b>10&times; smaller</b> data subset (10k vs 100k pocket crops), matches the full-data reproduction on Vina-dock within &plusmn;0.14 kcal/mol and effectively ties on <i>high-affinity %</i> ({fmt(ha_vm,1)} vs {fmt(ha_rep,1)}).
Versus a non-pre-trained aligned-density baseline at the same 10k data scale, ViT-MAE buys <b>{fmt(ddock_align,2)} kcal/mol Vina-dock</b> and <b>+{fmt(ha_align_diff,1)} pts high-affinity</b> on the 50-target overlap - bigger gain than adding the density branch itself.
</div>''')

# Section 1
html_parts.append('<h2>1. What we did &amp; why</h2>')
html_parts.append('''<p>VoxBind conditions a 3-D voxel diffusion model on a pocket density branch. Before this branch, density features were trained from scratch on a small downstream budget (10k pocket crops) and likely under-trained. We tested whether <b>self-supervised pre-training on synthetic density</b> can produce a strong encoder that downstream training only needs to fuse, not learn from scratch.</p>
<p>Three pre-training arms share the same synthetic-density pretext (sum-of-atom voxels &rarr; random Gaussian blur &sigma;&isin;[0.30, 0.80]&Aring; &rarr; per-sample z-score &rarr; additive noise). They differ only in <i>encoder</i> and <i>objective</i>:</p>
<table>
<thead><tr><th>Arm</th><th>Encoder</th><th>Objective</th><th>Why we ran it</th></tr></thead>
<tbody>
<tr><td><b>CNN-MAE</b></td>
    <td>4 &times; ResidualBlock (CNN)</td>
    <td>Block-mask + masked-voxel MSE + 11-ch atom-structure aux</td>
    <td>Shape-compatible drop-in for the existing CNN density encoder; cheapest baseline.</td></tr>
<tr><td><b>ViT-MAE 4.5M</b></td>
    <td>3-D ViT (patch 8&sup3;, D=192, L=6, H=6)</td>
    <td>Same MAE pretext as CNN</td>
    <td>Apples-to-apples test of whether attention helps over CNN at fixed compute &amp; pretext.</td></tr>
<tr><td><b>ViT-ELECTRA 4.5M</b></td>
    <td>3-D ViT, identical config</td>
    <td>Block-corrupt (swap/reblur/noise) + per-patch RTD BCE + same aux</td>
    <td>Tests whether replaced-token-detection (more sample-efficient in NLP) beats MAE on dense density volumes.</td></tr>
</tbody></table>
<p>All three encoders are <b>shape-compatible drop-ins</b> for <code>VoxBind.density_encoder</code>: same output shape, same fusion path, same downstream config. Only <code>density_encoder_type</code> and <code>density_pretrained_path</code> change.</p>''')

# Section 2.1 Pretraining
html_parts.append('<h2>2. Results</h2>')
html_parts.append('<h3>2.1 Pre-training convergence</h3>')
html_parts.append(f'''<table>
<thead><tr><th>Arm</th><th>Epochs done</th><th class="num">Val pretext</th><th class="num">Val L_str</th><th class="num">Val total</th><th>Val total (curve)</th></tr></thead>
<tbody>
<tr><td>CNN-MAE</td><td>55 / 100 <span class="footnote">killed early to free GPUs</span></td>
    <td class="num">0.0153</td><td class="num">0.1876</td><td class="num">0.2029</td><td>{sp["cnn_mae_val_loss"]}</td></tr>
<tr><td><b>ViT-MAE 4.5M</b></td><td>100 / 100</td>
    <td class="num"><b>0.0066</b></td><td class="num"><b>0.0556</b></td><td class="num"><b>0.0622</b></td><td>{sp["vit_mae_val_loss"]}</td></tr>
<tr><td>ViT-ELECTRA 4.5M</td><td>100 / 100</td>
    <td class="num">L_rtd 0.0025</td>
    <td class="num">0.2758</td><td class="num">0.2783</td><td>{sp["vit_electra_val_rtd"]}</td></tr>
</tbody></table>
<div class="key">
<b>Pre-training takeaways.</b>
<ul>
<li><b>ViT-MAE reconstructs ~3.3&times; tighter than CNN-MAE</b> at the same effective batch (val 0.062 vs 0.203). At matching ep55, ViT-MAE was already at 0.117 - so the gap is not just from CNN-MAE being killed.</li>
<li><b>ELECTRA solves its pretext too easily</b>: val L_rtd collapses to 0.0025, meaning the discriminator detects swap/reblur/noise corruptions almost perfectly. The encoder gets little representation pressure, so the auxiliary structure head ends up doing the work but tops out worse than under MAE (L_str 0.276 vs 0.056).</li>
<li><b>Diagnosis:</b> rule-based corruptions sit too far from the data manifold for a strong discriminator. The learned-generator path is implemented but not yet wired in.</li>
</ul>
</div>''')

# Section 2.2 Downstream
html_parts.append('<h3>2.2 Downstream diffusion training (frozen encoder, 10k subset)</h3>')
html_parts.append(f'''<table>
<thead><tr><th>Frozen encoder</th><th>Epochs</th><th class="num">Val loss</th><th class="num">Val mIoU</th><th>Val mIoU (curve)</th></tr></thead>
<tbody>
<tr><td>CNN-MAE (ep55)</td><td>200</td><td class="num">157.11</td><td class="num">0.578</td><td>{sp["down_cnn_miou"]}</td></tr>
<tr><td><b>ViT-MAE (ep99)</b></td><td>380</td><td class="num"><b>116.22</b></td><td class="num"><b>0.623</b></td><td>{sp["down_vit_miou"]}</td></tr>
<tr><td>ViT-ELECTRA (ep99)</td><td>157 / 200 <span class="footnote">still training</span></td>
    <td class="num">149.86</td><td class="num">0.589</td><td>{sp["down_electra_miou"]}</td></tr>
</tbody></table>
<div class="key"><b>Downstream takeaways.</b> ViT-MAE wins on both diffusion loss (<b>26% lower</b> than CNN-MAE) and mIoU (<b>+4.5 pts</b>). ViT-ELECTRA tracks CNN-MAE more than ViT-MAE so far - same encoder, but ELECTRA's weaker pretext signal carries through to downstream. (ELECTRA has ~43 epochs left.)</div>''')

# Section 2.3 Vina
def diff_cell(d, good_if_neg=True):
    cls = "good" if (good_if_neg and d < 0) or (not good_if_neg and d > 0) else "bad"
    sign = "+" if d > 0 else ""
    return f'<td class="num {cls}">{sign}{fmt(d)}</td>'

html_parts.append('<h3>2.3 Vina docking (overlap-only target means; kcal/mol, lower = better)</h3>')
html_parts.append('<p>ViT-MAE is the only pre-training arm with a fulltest sampling+docking run (79 targets). Each row compares ViT-MAE to a reference arm on the same target subset.</p>')

def row_pairwise(label, d, self_first=True):
    n = d['vina_score_mean'][0]
    vs_self = d['vina_score_mean'][1]; vs_ref = d['vina_score_mean'][2]
    vd_self = d['vina_dock_mean'][1];  vd_ref = d['vina_dock_mean'][2]
    ha_self = d['high_affinity'][1]*100; ha_ref = d['high_affinity'][2]*100
    # Vina deltas (self - ref): negative is better (good_if_neg=True default).
    # HA% delta (self - ref): positive is better (good_if_neg=False).
    return (
        f'<tr><td>{label}</td><td class="num">{n}</td>'
        f'<td class="num">{fmt(vs_self)}</td><td class="num">{fmt(vs_ref)}</td>'
        + diff_cell(vs_self - vs_ref) +
        f'<td class="num">{fmt(vd_self)}</td><td class="num">{fmt(vd_ref)}</td>'
        + diff_cell(vd_self - vd_ref) +
        f'<td class="num">{fmt(ha_self,1)} / {fmt(ha_ref,1)}</td>'
        + diff_cell(ha_self - ha_ref, good_if_neg=False) +
        '</tr>'
    )

html_parts.append('''<table>
<thead><tr>
  <th>Comparison (self vs ref)</th><th>n_overlap</th>
  <th class="num">v_score self</th><th class="num">v_score ref</th><th class="num">&Delta; score</th>
  <th class="num">v_dock self</th><th class="num">v_dock ref</th><th class="num">&Delta; dock</th>
  <th class="num">HA% self / ref</th><th class="num">&Delta; HA%</th>
</tr></thead><tbody>''')
html_parts.append(row_pairwise("<b>ViT-MAE</b> vs reproduction (full data)",         vm_vs_rep))
html_parts.append(row_pairwise("<b>ViT-MAE</b> vs aligned density (no SSL)",         vm_vs_align))
html_parts.append(row_pairwise("<b>ViT-MAE</b> vs 10k baseline (no density branch)", vm_vs_base))
html_parts.append(row_pairwise("aligned density vs 10k baseline (no density)",       align_vs_base))
html_parts.append('</tbody></table>')

html_parts.append(f'''<div class="key">
<b>Docking takeaways (effectiveness of what we did).</b>
<ol>
<li><b>The density branch alone</b> (no SSL, aligned crops) buys <b>{fmt(align_vs_base['vina_dock_mean'][1]-align_vs_base['vina_dock_mean'][2],2)} kcal/mol Vina-dock</b> and <b>+{fmt((align_vs_base['high_affinity'][1]-align_vs_base['high_affinity'][2])*100,1)} pts HA%</b> over no-density baseline.</li>
<li><b>Adding ViT-MAE pre-training on top</b> of the density branch buys an <b>additional {fmt(vm_vs_align['vina_dock_mean'][1]-vm_vs_align['vina_dock_mean'][2],2)} kcal/mol</b> and <b>+{fmt((vm_vs_align['high_affinity'][1]-vm_vs_align['high_affinity'][2])*100,1)} pts HA%</b> - <i>larger than the density branch's own contribution</i>.</li>
<li><b>Data efficiency:</b> ViT-MAE on a <b>10&times; smaller</b> training set lands within 0.14 kcal/mol Vina-dock of the full-data reproduction. Downstream looks feature-bound, not data-bound, at this scale.</li>
</ol>
</div>''')

# Section 3 conclusions
html_parts.append('<h2>3. Conclusions</h2>')
html_parts.append('''<ol>
<li><b>SSL pre-training on density works</b> - and contributes more than just having the density branch. Headline positive result.</li>
<li><b>ViT &gt; CNN as the density encoder</b> at this scale (4.5M params). ViT-MAE dominates CNN-MAE on pretext loss and downstream metrics.</li>
<li><b>MAE &gt; rule-based ELECTRA</b> on this pretext. Swap/reblur/noise corruptions are too distinguishable - the discriminator solves the RTD task trivially and provides no representation pressure. Final ELECTRA downstream still pending but trajectory is clear.</li>
<li><b>Downstream data efficiency:</b> a frozen pre-trained ViT-MAE encoder + 10k subset reaches full-data reproduction quality. Cheaper to pre-train once than to scale downstream data.</li>
</ol>''')

# Section 4 next experiments
html_parts.append('<h2>4. Suggested next experiments (to grow the contribution)</h2>')
html_parts.append('''<ol>
<li><b>Scale-up: ViT-MAE 40M.</b> The 42.6M config (D=512, L=12, H=8) is already written but not run. Tests whether the encoder is capacity-limited - current 4.5M is ViT-Tiny shaped. If MAE loss keeps dropping and downstream improves, we claim a scaling law for 3-D voxel SSL.</li>
<li><b>Unfreeze + fine-tune the encoder.</b> All current downstream runs use <code>density_freeze=true</code>. A short discriminative-LR fine-tune (encoder 1e-5, rest 1e-4, last 50 ep) usually buys another step. If it does <i>not</i> help here, that itself is a publishable finding.</li>
<li><b>Real-density pretext.</b> Current pretext is Gaussian-blurred atom voxels - a sanity-check distribution. Replace with real 2Fo-Fc maps from PDB or cryo-EM densities so the encoder learns features robust to true noise, resolution, and missing-density. This could be the headline contribution: "SSL transfers from real density &rarr; de novo ligand generation."</li>
<li><b>Learned-generator ELECTRA.</b> The <code>cfg.electra.generator</code> path is wired in but not enabled. A small CNN generator trained jointly produces distribution-matched corruptions and fixes the trivial-detection failure. Worth one run before declaring ELECTRA dead.</li>
<li><b>Bigger pretext corpus.</b> Pretext is synthetic from CrossDocked atoms only. Adding PDB-mined pockets (no x-ray needed) easily 10&times;'s the data - often the unlock for ViT scaling.</li>
<li><b>Cross-modal contrastive pretext.</b> Replace the structure-MSE aux head with an InfoNCE objective between density-encoder output and a separate ligand-atom encoder. Directly targets the cross-modal alignment we ultimately need for diffusion conditioning.</li>
<li><b>Linear-probing study.</b> Probe what the pre-trained encoder has actually learned: (a) atom-type classification, (b) secondary-structure recovery, (c) ligand-pose RMSD prediction. Gives interpretable mechanisms for the downstream Vina deltas - turns "it works" into "we explain why."</li>
<li><b>Fulltest sampling+docking for CNN-MAE and ViT-ELECTRA.</b> Immediate next step before any paper writeup - currently only ViT-MAE has 79-target Vina numbers.</li>
</ol>''')

# Appendix
html_parts.append('<div class="appendix">')

html_parts.append('<h2>Appendix A &middot; Architecture details</h2>')
html_parts.append('''<table>
<thead><tr><th>Component</th><th>CNN-MAE</th><th>ViT-MAE 4.5M</th><th>ViT-MAE 40M (planned)</th></tr></thead>
<tbody>
<tr><td>Patchifier / pre-conv</td><td>Conv3d(1&rarr;16, k=3)</td><td>Conv3d(1&rarr;192, k=8, s=8)</td><td>Conv3d(1&rarr;512, k=8, s=8)</td></tr>
<tr><td>Token / spatial grid</td><td>n/a (full 64&sup3;)</td><td>8&sup3;=512 tokens</td><td>8&sup3;=512 tokens</td></tr>
<tr><td>Encoder body</td><td>4 &times; ResidualBlock(16&rarr;16, gn=16, p=0.1)</td><td>6 &times; pre-LN ViT block, D=192, H=6, MLP=4&times;</td><td>12 &times; pre-LN ViT block, D=512, H=8, MLP=4&times;</td></tr>
<tr><td>Output feature shape</td><td>(B,16,64,64,64)</td><td>(B,16,64,64,64) via token&rarr;voxel reshape</td><td>(B,16,64,64,64) idem</td></tr>
<tr><td>Density head</td><td colspan="3">Conv3d(16&rarr;16) &rarr; SiLU &rarr; Conv3d(16&rarr;1)</td></tr>
<tr><td>RTD head (ELECTRA)</td><td colspan="3">Conv3d(16&rarr;16) &rarr; SiLU &rarr; Conv3d(16&rarr;1, k=8, s=8) &rarr; (B,1,8,8,8)</td></tr>
<tr><td>Structure aux head</td><td colspan="3">Conv3d(16&rarr;16) &rarr; SiLU &rarr; Conv3d(16&rarr;11)  // 7 ligand + 4 pocket</td></tr>
<tr><td><b>Total pretext-model params</b></td><td><b>0.08 M</b></td><td><b>4.47 M</b></td><td><b>42.6 M</b></td></tr>
<tr><td>Frozen encoder params loaded into VoxBind</td><td>0.056 M</td><td>4.45 M</td><td>~42.5 M</td></tr>
</tbody></table>
<p class="footnote">All three encoders produce (B, n_channels/2, G, G, G) and load drop-in into <code>VoxBind.density_encoder</code> via <code>encoder_state_dict_ema</code>. Downstream <code>density_proj</code> fusion path is identical.</p>''')

html_parts.append('<h2>Appendix B &middot; Pretext objective details</h2>')
html_parts.append('''<h4>Shared input synthesis</h4>
<pre>density = sum_over_channels(voxels_lig) + sum_over_channels(voxels_poc)
        &rarr; GaussianBlur3D(sigma ~ U[0.30, 0.80] Aa)
        &rarr; per-sample z-score
        &rarr; + N(0, 0.05^2)         # additive Gaussian noise</pre>

<h4>MAE pretext (CNN-MAE, ViT-MAE)</h4>
<ul>
<li>3-D block mask at 8&sup3; voxel blocks, mask_ratio = 0.60. Masked voxels are <b>zeroed</b> in the encoder input.</li>
<li>L = L_dens + L_str
  <ul>
    <li>L_dens = MSE(d_hat, d_clean) restricted to masked voxels.</li>
    <li>L_str  = pos-weighted MSE on 11-ch atom voxels. Voxels with target &gt; 0.05 get 100&times; weight to counter the ~99.8%-sparse target (without this, the network learns the constant-zero shortcut).</li>
  </ul>
</li>
</ul>

<h4>ELECTRA pretext (ViT-ELECTRA)</h4>
<ul>
<li>Same block mask, but masked blocks are <b>replaced</b> per batch by one of:
  <ul>
    <li><code>swap</code> (60%) - copy same region from a neighbour-shifted sample</li>
    <li><code>reblur</code> (30%) - re-blur the same atoms with a different sigma</li>
    <li><code>noise</code> (10%) - N(0,1) at the z-scored scale</li>
  </ul></li>
<li>Per-patch BCE-with-logits on RTD head over the 8&sup3;=512-patch grid. Target = 1 if any voxel in the patch was corrupted.</li>
<li>Structure aux head retained, identical loss as MAE.</li>
</ul>

<h4>End-of-pre-training loss decomposition (val, EMA)</h4>
<table>
<thead><tr><th>Arm</th><th class="num">Pretext (dens / RTD)</th><th class="num">L_str</th><th class="num">Total</th></tr></thead>
<tbody>
<tr><td>CNN-MAE (ep55)</td><td class="num">0.0153</td><td class="num">0.1876</td><td class="num">0.2029</td></tr>
<tr><td>ViT-MAE (ep99)</td><td class="num">0.0066</td><td class="num">0.0556</td><td class="num">0.0622</td></tr>
<tr><td>ViT-ELECTRA (ep99)</td><td class="num">0.0025 (RTD)</td><td class="num">0.2758</td><td class="num">0.2783</td></tr>
</tbody></table>''')

html_parts.append('<h2>Appendix C &middot; Launch / hyperparameters</h2>')
html_parts.append('''<table>
<thead><tr><th>Setting</th><th>CNN-MAE</th><th>ViT-MAE</th><th>ViT-ELECTRA</th></tr></thead>
<tbody>
<tr><td>Pretext data</td><td colspan="3">CrossDocked2020, full set (~100k pockets) - atom voxels only</td></tr>
<tr><td>GPUs &middot; bsz/rank &middot; eff batch</td>
    <td>6 &middot; 16 &middot; 96</td><td>6 &middot; 16 &middot; 96</td><td>5 &middot; 16 &middot; 80</td></tr>
<tr><td>Epochs done</td><td>55 / 100 (killed)</td><td>100</td><td>100</td></tr>
<tr><td>Optimizer</td><td colspan="3">AdamW, lr=1e-4, beta=(0.9,0.999); CNN wd=1e-2, ViT wd=5e-2; grad-clip 1.0</td></tr>
<tr><td>EMA decay</td><td colspan="3">0.999</td></tr>
<tr><td>Pretext run dir</td>
    <td><code>exps/260521_density_mae_posweight</code></td>
    <td><code>exps/260522_density_vit_mae_pretrain</code></td>
    <td><code>exps/260524_density_vit_electra_pretrain</code></td></tr>
<tr><td>Downstream run dir</td>
    <td><code>260521_voxbind_10k_density_mae_frozen</code></td>
    <td><code>260523_voxbind_10k_density_vit_mae_frozen</code></td>
    <td><code>260525_voxbind_10k_density_vit_electra_frozen</code></td></tr>
<tr><td>Downstream data</td>
    <td colspan="3">CrossDocked 10k subset, aligned x-ray crops, bsz=4/rank, 200 ep (ViT 380 ep w/ resume)</td></tr>
<tr><td>Downstream encoder mode</td><td colspan="3"><code>density_freeze=true</code> (requires_grad=False, dropout off)</td></tr>
</tbody></table>''')

html_parts.append('<h2>Appendix D &middot; Full-arm Vina aggregates (across all available targets)</h2>')

def arm_row(label, k):
    a = A[k]
    return (
        f'<tr><td>{label}</td><td class="num">{len(a)}</td>'
        f'<td class="num">{fmt(all_mean(a,"vina_score_mean"))}</td>'
        f'<td class="num">{fmt(all_mean(a,"vina_min_mean"))}</td>'
        f'<td class="num">{fmt(all_mean(a,"vina_dock_mean"))}</td>'
        f'<td class="num">{fmt(all_mean(a,"high_affinity")*100,1)}</td>'
        f'<td class="num">{fmt(all_mean(a,"qed_mean"))}</td>'
        f'<td class="num">{fmt(all_mean(a,"sa_mean"))}</td>'
        f'<td class="num">{fmt(all_mean(a,"diversity"))}</td></tr>'
    )

html_parts.append('''<table>
<thead><tr><th>Arm</th><th>n targets</th><th class="num">v_score</th><th class="num">v_min</th><th class="num">v_dock</th><th class="num">HA%</th><th class="num">QED</th><th class="num">SA</th><th class="num">Diversity</th></tr></thead>
<tbody>''')
html_parts.append(arm_row("10k baseline (no density)", "baseline"))
html_parts.append(arm_row("10k aligned density (no SSL, ep138)", "aligned"))
html_parts.append(arm_row("10k ViT-MAE frozen (ep379, fulltest)", "vit_mae"))
html_parts.append(arm_row("VoxBind reproduction (full data, fulltest)", "repro"))
html_parts.append('</tbody></table>')
html_parts.append('<p class="footnote">Full-arm means are over <i>different</i> target sets; use &sect;2.3 (overlap-only) for head-to-head comparisons. CNN-MAE and ViT-ELECTRA do not yet have a fulltest sampling+docking run - that is the immediate next step.</p>')

html_parts.append('<h2>Appendix E &middot; Open items / known limitations</h2>')
html_parts.append('''<ul>
<li>CNN-MAE pre-training was killed at ep55 to free GPUs - final pretext loss is "best we have" not "best possible". Re-running to 100 ep would tighten the apples-to-apples claim.</li>
<li>ViT-ELECTRA downstream still has ~43 epochs left as of 2026-05-26. Final mIoU may close some gap but trajectory clearly underperforms ViT-MAE.</li>
<li>Only ViT-MAE has a fulltest sampling+docking run (79 targets). Matching runs for CNN-MAE and ViT-ELECTRA are needed before any paper writeup.</li>
<li>40M ViT config is committed but not yet launched.</li>
<li>Vina scoring has a few clash-explosion outliers (vina_score &gt;= 0 on ~2 targets out of 79) - median Vina-dock is more robust than mean. ViT-MAE median Vina-dock = -8.06 vs reproduction -8.12; deltas hold.</li>
</ul>''')

html_parts.append('</div></body></html>')

html = "\n".join(html_parts)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(html)
print(f"wrote: {OUT}  ({len(html)} bytes, {html.count(chr(10))} lines)")
