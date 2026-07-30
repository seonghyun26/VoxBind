#!/usr/bin/env python3
"""build_trials.py — 260806 data/recipe campaign trial tracker (autoresearch-style).

Trial identity split into DATASET / SIZE / MASK / OTHER; val ρ + test ρ/r/RMSE, compared vs the
CHAMPION. val–test gap column = overfitting diagnostic. Per-trial ρ/r/RMSE scatter (red champion
ceiling + gold frontier). Hand-maintained TRIALS; re-run to regenerate trials.html.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
CH = dict(rho=0.644, r=0.660, rmse=1.349)     # champion = best recipe (ceiling)

# dataset, size, mask, other, group, val_rho, test_rho, r, rmse, running, champion
TRIALS = [
    ("v1",   "40M",  "0.5",  "—",                "corpus",   0.548, 0.637, 0.656, 1.353, False, False),
    ("v2",   "40M",  "0.5",  "—",                "corpus",   None,  0.620, 0.634, 1.450, False, False),
    ("v2",   "100M", "0.5",  "—",                "recipe",   None,  0.614, 0.633, 1.391, False, False),
    ("v2",   "100M", "0.75", "—",                "recipe",   0.546, 0.644, 0.660, 1.349, False, True),
    ("v2.1", "40M",  "0.5",  "ema.999",          "clean",    0.533, 0.621, 0.638, 1.426, False, False),
    ("v2.1", "40M",  "0.5",  "ema.9999",         "EMA",      0.515, 0.635, 0.652, 1.401, False, False),
    ("v2.1", "100M", "0.75", "—",                "clean",    0.559, 0.632, 0.650, 1.366, False, False),
    ("v2",   "100M", "0.75", "100ep · ema.999",  "epochs",   None,  0.626, 0.647, 1.358, False, False),
    ("v2",   "100M", "0.75", "100ep · ema.9999", "epochs",   None,  0.622, 0.640, 1.389, False, False),
    ("v2",   "100M", "0.75", "cosine + bs256",   "schedule", None,  0.632, 0.650, 1.359, False, False),
    ("v2",   "100M", "0.75", "cosine-only",      "schedule", None,  0.626, 0.646, 1.414, False, False),
    ("v3",   "100M", "0.75", "HCS.15",           "v3+pose",  0.571, 0.632, 0.646, 1.389, False, False),
    ("v3",   "60M",  "0.75", "HCS.15",           "overfit",  0.563, 0.633, 0.654, 1.423, False, False),
    ("v3",   "60M",  "0.75", "HCS + drop_path.2","overfit",  0.554, 0.639, 0.651, 1.371, False, False),
    ("v3",   "60M",  "0.85", "HCS.15",           "overfit",  0.572, 0.641, 0.661, 1.372, False, False),
    ("v2",   "60M",  "0.75", "size control",     "size",     0.574, 0.630, 0.647, 1.396, False, False),
    ("v2",   "100M", "0.85", "champ+mask",       "mask",     0.570, 0.620, 0.629, 1.463, False, False),
    ("v2",   "100M", "0.90", "champ+mask",       "mask",     0.567, 0.627, 0.645, 1.418, False, False),
    ("v3",   "100M", "0.85", "mask sweep",       "mask",     0.568, 0.640, 0.658, 1.355, False, False),
    ("v3",   "100M", "0.90", "mask sweep",       "mask",     0.571, 0.645, 0.660, 1.383, False, False),
    ("v3",   "100M", "0.95", "mask sweep · best r/RMSE", "mask", 0.584, 0.644, 0.663, 1.334, False, False),
]
DS_COLOR = {"v1":"#1a7f37","v2":"#bc4c00","v2.1":"#2b5fd0","v3":"#8e44ad"}
def VAL(tr): return tr[5]
def R(tr): return tr[6]
def r_(tr): return tr[7]
def RM(tr): return tr[8]
def RUN(tr): return tr[9]
def CHAMP(tr): return tr[10]
def val(tr,key): return RM(tr) if key=="rmse" else (R(tr) if key=="rho" else r_(tr))
def lbl(tr): return f'{tr[0]} · {tr[1]} · m{tr[2]}' + ('' if tr[3]=="—" else f' · {tr[3]}')

def chart(key, ylabel, ymin, ymax, lower_better):
    W,H = 880,300; x0,x1,y0,y1 = 60,858,16,262; n=len(TRIALS)
    xs=[x0+(x1-x0)*i/(n-1) for i in range(n)]
    def y(v): return y1-(y1-y0)*(v-ymin)/(ymax-ymin)
    s=[f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:900px" font-family="-apple-system,Segoe UI,sans-serif">']
    for t in range(7):
        v=ymin+(ymax-ymin)*t/6; yy=y(v)
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#eef1f5"/>')
        s.append(f'<text x="{x0-7}" y="{yy+3:.1f}" font-size="10" fill="#5b6678" text-anchor="end">{v:.3f}</text>')
    cy=y(CH[key])
    s.append(f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="#d64545" stroke-width="1.3" stroke-dasharray="5 4" opacity="0.75"/>')
    s.append(f'<text x="{x1}" y="{cy-4:.1f}" font-size="9.5" fill="#d64545" text-anchor="end" font-weight="700">champion {CH[key]:.3f}</text>')
    for i in range(n):
        s.append(f'<text x="{xs[i]:.1f}" y="{H-16}" font-size="9.5" fill="#1c2433" text-anchor="middle">T{i+1}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-2}" font-size="10" fill="#5b6678" text-anchor="middle">trial (campaign order)</text>')
    best=None; fx=[]; fy=[]
    for i,tr in enumerate(TRIALS):
        v=val(tr,key)
        if RUN(tr) or v is None: continue
        if best is None or (v<best if lower_better else v>best): best=v; fx.append(xs[i]); fy.append(y(v))
    if len(fx)>1:
        s.append('<path d="M '+' L '.join(f'{a:.1f} {b:.1f}' for a,b in zip(fx,fy))+'" fill="none" stroke="#d97706" stroke-width="1.3" stroke-dasharray="2 3" opacity="0.6"/>')
    fset=set(zip(fx,fy))
    for i,tr in enumerate(TRIALS):
        v=val(tr,key); col=DS_COLOR.get(tr[0],"#1d6fd0")
        if RUN(tr) or v is None:
            s.append(f'<circle cx="{xs[i]:.1f}" cy="{H-42}" r="4" fill="none" stroke="#9aa3b2" stroke-width="1.5"><title>{lbl(tr)} — n/a</title></circle>'); continue
        yy=y(v); on_f=(xs[i],yy) in fset; rr=6.5 if on_f else 4.3
        ring=' stroke="#d97706" stroke-width="2.3"' if on_f else ''
        s.append(f'<circle cx="{xs[i]:.1f}" cy="{yy:.1f}" r="{rr}" fill="{col}"{ring} opacity="{1.0 if on_f else 0.75}"><title>{lbl(tr)} — {key} {v:.3f}</title></circle>')
        if on_f: s.append(f'<text x="{xs[i]:.1f}" y="{yy-rr-3:.1f}" font-size="9.5" font-weight="700" fill="#1c2433" text-anchor="middle">{v:.3f}</text>')
    s.append(f'<text x="13" y="{(y0+y1)//2}" font-size="11" fill="#5b6678" transform="rotate(-90 13 {(y0+y1)//2})" text-anchor="middle">{ylabel}</text></svg>')
    return ''.join(s)

def dcell(v, cv, lower):
    if v is None: return '<span class="pm">—</span>'
    d=v-cv; cls='mut' if abs(d)<1e-9 else ('pos' if (d<0 if lower else d>0) else 'neg')
    return f'<span class="num">{v:.3f}</span><span class="dlt {cls}">{d:+.3f}</span>'

rows=[]
for i,tr in enumerate(TRIALS):
    ds,sz,mk,oth,grp,valr,rho,r,rmse,run,champ=tr
    cls='ref' if champ else ('mut' if run else '')
    dsb=f'<span style="color:{DS_COLOR.get(ds)};font-weight:700">{ds}</span>'
    star=' ★' if champ else ''
    # val ρ + gap (val - test); negative gap (test>val) is normal here, positive gap = overfit warning
    if valr is None or rho is None:
        valcell='<span class="pm">—</span>'; gapcell='<span class="pm">—</span>'
    else:
        gap=valr-rho
        gcls='neg' if gap>0.02 else 'mut'   # positive gap (val>test) = overfit flag
        valcell=f'<span class="num">{valr:.3f}</span>'
        gapcell=f'<span class="{gcls}">{gap:+.3f}</span>'
    rows.append(f'<tr class="{cls}"><td>T{i+1}</td><td>{dsb}</td><td>{sz}</td><td>{mk}</td>'
                f'<td style="white-space:normal">{oth}{star}</td>'
                f'<td>{valcell}</td><td>{dcell(rho,CH["rho"],False)}</td><td>{gapcell}</td>'
                f'<td>{dcell(r,CH["r"],False)}</td><td>{dcell(rmse,CH["rmse"],True)}</td></tr>')

html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data & recipe campaign — trials vs best recipe</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1c2433;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.page{{max-width:1060px;margin:0 auto;padding:40px 24px 90px}}
header{{border-bottom:2px solid #1c2433;padding-bottom:14px;margin-bottom:24px}}
header .date{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#5b6678;margin-bottom:6px}}
header h1{{font-size:25px;font-weight:650;margin:0}} header .sub{{font-size:13px;color:#5b6678;margin-top:6px}}
.card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:20px;margin-bottom:26px;box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}}
h2{{font-size:17px;font-weight:640;margin:0 0 4px}} .csub{{font-size:12.5px;color:#5b6678;margin:0 0 14px}}
.best-card{{background:#eaf5ee;border:1px solid #bfe3cd}} .best-card b{{color:#1d5a3a}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 9px;text-align:left;vertical-align:top;border-top:1px solid #e3e7ee;white-space:nowrap}}
thead th{{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.03em;color:#5b6678;border:0;border-bottom:1px solid #e3e7ee}}
td:nth-child(n+6),th:nth-child(n+6){{text-align:right}}
.num{{font-weight:660}}.pm{{color:#9aa3b2;font-weight:500;font-size:11.5px}}
.pos{{color:#1d5a3a;font-weight:700}}.neg{{color:#b4413a;font-weight:700}}.mut td,.mut{{color:#8a93a0}}
.dlt{{display:block;font-size:10px;font-weight:700;margin-top:1px}}
tr.ref td{{background:#eaf5ee;font-weight:600}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.6}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:11.5px;color:#5b6678}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER · 2026-08-06</div>
<h1>Data &amp; recipe campaign — trials vs the best recipe</h1>
<div class="sub">Every data-curation and recipe trial, frozen affinity probe on <code>lp_edrscc_v2</code> (Kd/Ki; val n=817, test n=1320; 3 seeds).
Champion (best recipe) = red ceiling; Δ = vs champion; gold ring = best-so-far. Dot colour = dataset.</div></header>

<div class="card best-card"><h2>★ Best recipe (champion)</h2>
<div class="note"><b>Dataset</b> PLINDER v2 (112K) · <b>Size</b> 100M ChannelViT [7,4,2] C+D+G · <b>Mask</b> 0.75 · <b>Other</b> 50 ep · ema 0.999
&mdash; Pearson r <b>0.660</b> · Spearman ρ <b>0.644</b> · RMSE <b>1.349</b>.</div></div>

<div class="card"><h2>Spearman ρ by trial</h2><p class="csub">Red line = champion ceiling (0.644). Gold ring = best-so-far.</p>{chart('rho','test ρ (Spearman)',0.610,0.660,False)}</div>
<div class="card"><h2>Pearson r by trial</h2>{chart('r','test r (Pearson)',0.626,0.666,False)}</div>
<div class="card"><h2>RMSE (pK) by trial — lower is better</h2>{chart('rmse','test RMSE (pK)',1.330,1.470,True)}
<div class="legend">{''.join(f'<span><span style="width:10px;height:10px;border-radius:50%;background:{c}"></span>{d}</span>' for d,c in DS_COLOR.items())}</div></div>

<div class="card"><h2>Trial table — val + test</h2>
<p class="csub">Δ = vs champion (<span class="pos">green</span> better / <span class="neg">red</span> worse; RMSE lower=better).
<b>val−test gap</b>: negative = test&gt;val (normal — val n=817 is a smaller/harder subset, <b>not</b> overfitting);
<span class="neg">positive &gt;0.02</span> would flag probe overfitting. <span style="background:#eaf5ee;padding:0 4px;border-radius:3px">Champion ★</span>.</p>
<table><thead><tr><th>#</th><th>Dataset</th><th>Size</th><th>Mask</th><th>Other</th><th>val ρ</th><th>test ρ (Δ)</th><th>val−test</th><th>test r (Δ)</th><th>RMSE (Δ)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="note" style="margin-top:12px"><b>Verdict:</b> champion ρ 0.644 not broken, but <b>v3 · 100M · mask0.95 Pareto-beats it</b> (ties ρ 0.644, <b>r 0.663 &gt; 0.660</b>, <b>RMSE 1.334 &lt; 1.349</b>) on a cleaner, 35%-smaller corpus — the best model to adopt.
<b>Masking × cleanliness interact:</b> clean v3 rises to a peak at mask 0.90–0.95 (0.632→0.640→0.645→0.644), while noisy v2 <i>drops</i> above 0.75 (0.644→0.620→0.627) — clean data absorbs aggressive masking, noisy data cannot.
<b>Size × corpus interact:</b> 60M ≈ 100M on small v3, but 100M &gt; 60M (+0.014) on large v2 (data rewards capacity).
All val−test gaps negative (test&gt;val, val n=817 harder subset) → no probe overfitting. Ceiling ρ≈0.644 is downstream (probe on 3850 labels); head trials are summarized in §1.1 of the meeting doc.</div></div>
</div></body></html>"""
out=HERE/"trials.html"; out.write_text(html)
print(f"wrote {out} ({len(html)} bytes, {len(TRIALS)} trials)")
