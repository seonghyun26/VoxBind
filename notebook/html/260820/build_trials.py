#!/usr/bin/env python
"""Build notebook/html/260820/trials.html — Champion vs Trials.

A focused view of the recent CDG-encoder campaign: the champion (0.646) as the reference
line, every trial's honest test/val ρ, the delta vs champion, and a one-line verdict.
Edit CHAMPION / TRIALS and re-run:  python build_trials.py
"""

# (name, recipe, test_rho, val_rho, rmse, note)   test_rho=None → pending/running
# All encoder numbers = 5-seed (0..4) MSE test ρ via the eval_cdg/01c probe (matches the
# baselines' seed count). Champion re-probed on the SAME machinery = 0.643 (the canonical
# 0.646 is the ablation-probe machinery; ~0.003 gap is probe-head noise). Δ uses 0.643.
CHAMP_RHO = 0.643
CHAMPION = ("Champion", "260705 · ChannelViT [7,4,2] · v2 · fixed mask 0.75 · vdW · 100M · 5-seed", 0.643, 0.546, 1.392)

TRIALS = [
    # data-version
    ("v2.4 champion (CASF-clean)", "data", "champion recipe on v2.4 (v2 minus CASF-2016 ID30 homologs, 101K) — first v2.4 encoder", 0.623, 0.558, 1.420,
     "CASF-decontam HURTS like v2.2 (−0.020); fewer samples cost more than the cleanup"),
    ("v2.2 champion (in-vocab)",   "data", "champion recipe on clean v2.2 (v2 minus 5,709 OOV-ligand)", 0.625, 0.551, 1.423,
     "v2.2 HURTS the champion (−0.019); cleaning helps weak configs only"),
    # masking
    ("R2MAE variable mask (v2)",   "masking", "one encoder, mask r~U[0.6,0.9] per batch", 0.622, 0.533, 1.430,
     "single-encoder mask variety ≠ ensemble diversity"),
    ("R2MAE variable mask (v2.2)", "masking", "same, on clean v2.2", 0.635, 0.557, 1.427,
     "v2.2 cleaning lifts it, still < champion"),
    ("per-channel masking (v2.2)", "masking", "per_group: independent block mask per ChannelViT group [7,4,1,1]", 0.623, 0.551, None,
     "per-channel masking HURTS on [7,4,1,1]"),
    ("per-channel × [7,4,2] × var", "masking", "per_group on champion grouping [7,4,2] + variable rate — the fairer cell", 0.633, 0.566, 1.350,
     "≈ regular [7,4,2]-variable 0.635 (tied) — per-channel masking gives NO gain even on champion group"),
    # architecture
    ("channel-sep [7,4,1,1] (v2)",  "arch", "density &amp; gradmag as separate ChannelViT groups", 0.627, 0.558, None,
     "channel separation HURTS (−0.016)"),
    ("channel-sep [7,4,1,1] (v2.2)","arch", "same, clean v2.2", 0.634, 0.571, None,
     "v2.2 narrows the gap, grouped [7,4,2] still wins"),
    ("[7,4,1,1] + variable (v2.2)", "arch", "channel separation + R2MAE", 0.632, 0.553, 1.380,
     "sep + varmask beats neither alone"),
    # modality
    ("no-gradmag CD [7,4,1] (v2.2)","modality", "drop gradmag → 12 channels", 0.625, 0.543, 1.356,
     "gradmag REDUNDANT: CD 0.625 ≈ v2.2-CDG 0.625 (tied) — drop 1 ch free"),
    # readout / ensemble  (analysis-probe machinery, 5-seed)
    ("ranking-loss head",           "head", "mse+rank probe head (directly optimize ordering)", 0.6475, 0.550, 1.391,
     "readout loss not a lever (all heads 0.645–0.648)"),
    ("ensemble (honest)",           "ensemble", "champion + v3_085/090/095, val-selected (5 seeds)", 0.6528, 0.588, 1.511,
     "the honest BEST on record (+0.007 vs 0.646 same-machinery); gain = v3 mask-diversity; RMSE worsens"),
]

GROUP_COL = {"data": "#2f6f8f", "masking": "#8f6f2f", "arch": "#6f4f8f", "modality": "#2f8f6f",
             "head": "#8f2f6f", "ensemble": "#2f8f5b"}


def verdict(rho):
    if rho is None:
        return "pending", "#8a94a6", "running"
    d = rho - CHAMP_RHO
    if d > 0.004:  return "help", "#2f8f5b", f"+{d:.3f}"
    if d < -0.004: return "hurt", "#b3423a", f"{d:.3f}"
    return "neutral", "#c98a12", f"{d:+.3f}"


def bar_chart():
    rows = [CHAMPION] + [(t[0], t[3]) for t in TRIALS]  # (name, rho)
    items = [(CHAMPION[0], CHAMPION[2], "ref")] + [(t[0], t[3], t[1]) for t in TRIALS]
    W, H, ML, MR, MT, MB = 1080, 430, 56, 16, 24, 150
    PW, PH = W - ML - MR, H - MT - MB
    ymin, ymax = 0.44, 0.665
    yof = lambda v: MT + (ymax - v) / (ymax - ymin) * PH
    n = len(items); slot = PW / n; bw = slot * 0.62
    ybase = yof(ymin)
    best = max(v for _, v, _ in items if v is not None)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="champion vs trials">']
    for g in (0.45, 0.50, 0.55, 0.60, 0.65):
        y = yof(g)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eceff3"/>')
        s.append(f'<text x="{ML-8}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="#8a94a6">{g:.2f}</text>')
    yc = yof(CHAMP_RHO)
    s.append(f'<line x1="{ML}" y1="{yc:.1f}" x2="{ML+PW}" y2="{yc:.1f}" stroke="#1c2433" stroke-dasharray="5 3" opacity="0.7"/>')
    s.append(f'<text x="{ML+PW-2}" y="{yc-4:.1f}" text-anchor="end" font-size="10" fill="#1c2433">champion {CHAMP_RHO:.3f}</text>')
    for i, (name, rho, grp) in enumerate(items):
        cx = ML + slot * (i + 0.5)
        if rho is None:
            # pending: hatched placeholder up to champion line
            s.append(f'<rect x="{cx-bw/2:.1f}" y="{yc:.1f}" width="{bw:.1f}" height="{ybase-yc:.1f}" rx="2" '
                     f'fill="#eceff3" stroke="#c9d0da" stroke-dasharray="3 2"/>')
            s.append(f'<text x="{cx:.1f}" y="{(yc+ybase)/2:.1f}" text-anchor="middle" font-size="9" fill="#8a94a6" '
                     f'transform="rotate(-90 {cx:.1f} {(yc+ybase)/2:.1f})">running</text>')
        else:
            col = "#566072" if grp == "ref" else GROUP_COL.get(grp, "#888")
            top = yof(rho); h = ybase - top
            strk = ' stroke="#1c2433" stroke-width="1.5"' if rho == best else ''
            s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{col}"{strk}>'
                     f'<title>{name}: {rho:.3f}</title></rect>')
            s.append(f'<text x="{cx:.1f}" y="{top-4:.1f}" text-anchor="middle" font-size="9.5" fill="#2b3444">{rho:.3f}</text>')
        # x label (rotated)
        s.append(f'<text x="{cx:.1f}" y="{ybase+12:.1f}" text-anchor="end" font-size="9.5" fill="#3a4456" '
                 f'transform="rotate(-40 {cx:.1f} {ybase+12:.1f})">{name}</text>')
    s.append(f'<line x1="{ML}" y1="{ybase:.1f}" x2="{ML+PW}" y2="{ybase:.1f}" stroke="#c9d0da"/>')
    s.append('</svg>')
    return "".join(s)


def table():
    th = ('<tr style="background:#f6f8fa;color:#7a8699;font-size:11px;text-transform:uppercase;letter-spacing:.04em">'
          '<th style="padding:7px 9px;text-align:left">Trial</th>'
          '<th style="padding:7px 9px;text-align:left">Group</th>'
          '<th style="padding:7px 9px;text-align:left">What</th>'
          '<th style="padding:7px 9px;text-align:right">test &#961;</th>'
          '<th style="padding:7px 9px;text-align:right">val &#961;</th>'
          '<th style="padding:7px 9px;text-align:right">RMSE</th>'
          '<th style="padding:7px 9px;text-align:right">&Delta;&#961;</th>'
          '<th style="padding:7px 9px;text-align:center">verdict</th></tr>')
    # champion row first
    rows = [th]
    rows.append(
        '<tr style="border-top:1px solid #edf0f4;background:#f3f6f4;font-weight:700">'
        f'<td style="padding:6px 9px">{CHAMPION[0]}</td>'
        f'<td style="padding:6px 9px;color:#7a8699;font-size:12px">baseline</td>'
        f'<td style="padding:6px 9px;color:#5b6678;font-size:12px">{CHAMPION[1]}</td>'
        f'<td style="padding:6px 9px;text-align:right">{CHAMPION[2]:.3f}</td>'
        f'<td style="padding:6px 9px;text-align:right;color:#5b6678">{CHAMPION[3]:.3f}</td>'
        f'<td style="padding:6px 9px;text-align:right;color:#5b6678">{CHAMPION[4]:.3f}</td>'
        '<td style="padding:6px 9px;text-align:right;color:#8a94a6">&mdash;</td>'
        '<td style="padding:6px 9px;text-align:center"><span style="background:#566072;color:#fff;font-size:10px;'
        'font-weight:700;padding:1px 8px;border-radius:9px">ref</span></td></tr>')
    best = max((t[3] for t in TRIALS if t[3] is not None), default=0)
    for name, grp, what, rho, val, rmse, note in TRIALS:
        vk, col, dtxt = verdict(rho)
        rs = f"{rho:.3f}" if rho is not None else "&mdash;"
        vs = f"{val:.3f}" if val is not None else "&mdash;"
        rm = f"{rmse:.3f}" if rmse is not None else "&mdash;"
        bold = "font-weight:700;" if rho is not None and rho == best else ""
        gcol = GROUP_COL.get(grp, "#888")
        badge = (f'<span style="background:{col};color:#fff;font-size:10px;font-weight:700;'
                 f'padding:1px 8px;border-radius:9px">{vk}</span>')
        rows.append(
            f'<tr style="border-top:1px solid #edf0f4">'
            f'<td style="padding:6px 9px;{bold}">{name}</td>'
            f'<td style="padding:6px 9px"><span style="color:{gcol};font-size:11px;font-weight:600">{grp}</span></td>'
            f'<td style="padding:6px 9px;color:#5b6678;font-size:12px">{what}<br><span style="color:#98a2b3;font-size:11px">{note}</span></td>'
            f'<td style="padding:6px 9px;text-align:right;{bold}">{rs}</td>'
            f'<td style="padding:6px 9px;text-align:right;color:#5b6678">{vs}</td>'
            f'<td style="padding:6px 9px;text-align:right;color:#5b6678">{rm}</td>'
            f'<td style="padding:6px 9px;text-align:right;color:{col};font-weight:600">{dtxt}</td>'
            f'<td style="padding:6px 9px;text-align:center">{badge}</td></tr>')
    return ('<table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff;'
            'border:1px solid #e3e7ee;border-radius:10px;overflow:hidden">' + "".join(rows) + '</table>')


HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Champion vs Trials · 260820</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1c2433;
   background:#f4f6f9;margin:0;padding:28px 20px 60px}}
 .page{{max-width:1120px;margin:0 auto}}
 .eyebrow{{color:#8a94a6;font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
 h1{{font-size:24px;margin:4px 0 6px}}
 .sub{{color:#5b6678;font-size:14px;margin-bottom:18px}}
 .card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:16px 18px;margin-bottom:18px}}
 .title{{font-size:13px;color:#566072;font-weight:600;margin:0 0 10px}}
 .kpis{{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:6px}}
 .kpi b{{font-size:22px}} .kpi span{{color:#8a94a6;font-size:12px;display:block}}
 code{{background:#eef1f5;padding:1px 5px;border-radius:4px;font-size:12px}}
 ul.notes{{color:#5b6678;font-size:13px;line-height:1.6;margin:8px 0 0;padding-left:18px}}
</style></head><body><div class="page">
 <div class="eyebrow">CDG-encoder campaign · 260820</div>
 <h1>Champion vs Trials</h1>
 <div class="sub">Recent trials against the champion CDG encoder on <code>lp_edrscc_v2</code> (MSE probe head, <b>5 seeds</b>, full 1320-complex test, no CL filter). Honest numbers &mdash; val-selected where a choice is made.</div>

 <div class="card">
   <p class="title">Reference</p>
   <div class="kpis">
     <div class="kpi"><b>0.643</b><span>champion test &#961; (5-seed)</span></div>
     <div class="kpi"><b>0.546</b><span>champion val &#961;</span></div>
     <div class="kpi"><b>0.653</b><span>best ensemble (honest, +0.007)</span></div>
   </div>
 </div>

 <div class="card">
   <p class="title">Figure &middot; test &#961; per trial (champion = dashed)</p>
   {bar_chart()}
 </div>

 <div class="card">
   <p class="title">Table &middot; champion vs each trial</p>
   {table()}
 </div>

 <div class="card">
   <p class="title">Takeaways</p>
   <ul class="notes">
     <li><b>Nothing single beats champion.</b> Every single-encoder recipe change (variable mask, channel separation, v2.2, v2.4, per-channel masking, no-gradmag) lands at or below the champion (0.643, 5-seed).</li>
     <li><b>Only the ensemble wins</b> &mdash; champion + 3&times;v3 = <b>0.653</b> (+0.007), driven by v3 mask-diversity; per_group &amp; v2.2-variants do NOT help the ensemble.</li>
     <li><b>gradmag is redundant</b> &mdash; dropping it (CD [7,4,1], 12ch) matches CDG (0.626 ≈ 0.625 on v2.2). One channel saved for free.</li>
     <li><b>v2.2 is not a lever</b> &mdash; helps weak configs, hurts the champion (−0.019). v2.4 (CASF-clean) &amp; per-channel-on-[7,4,2] still running.</li>
     <li>Readout-loss choice (mse / mse+corr / ranking) is within noise (0.645&ndash;0.648) &mdash; the ceiling is the representation.</li>
   </ul>
 </div>
</div></body></html>"""

import os
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials.html")
open(out, "w").write(HTML)
print("wrote", out, f"({len(HTML)} bytes)")
