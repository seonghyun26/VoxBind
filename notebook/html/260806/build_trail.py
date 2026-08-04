#!/usr/bin/env python3
"""build_trail.py — signal / optimizer / architecture campaign tracker (autoresearch-style).

Companion to build_trials.py (data/recipe campaign → sweep.html). This one tracks the trials
that probe whether a NEW signal, optimizer, capacity, corpus, or preprocessing choice beats the
champion recipe. Same frozen affinity probe on lp_edrscc_v2 (Kd/Ki; val 817, test 1320; 3 seeds).
Hand-maintained TRIALS; re-run to regenerate trail.html as each run's probe lands.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
CH = dict(rho=0.644, r=0.660, rmse=1.349)     # champion = best recipe (ceiling)

# theme, variant, group_color_key, val_rho, test_rho, r, rmse, running, champion
TRIALS = [
    ("Champion", "v2 100M mask0.75 C+D+G",       "champ",  0.546, 0.644, 0.660, 1.349, False, True),
    ("Optimizer","Muon · muon_lr 0.02",          "opt",    0.581, 0.632, 0.650, 1.419, False, False),
    ("Optimizer","Muon · muon_lr 0.05",          "opt",    0.415, 0.527, 0.552, 1.522, False, False),
    ("Capacity", "10M (14.4M enc, 256/12/8)",    "cap",    0.553, 0.640, 0.660, 1.346, False, False),
    ("Corpus",   "v4 apo+holo leakage-safe 11.4K","corp",  0.405, 0.561, 0.589, 1.557, False, False),
    ("Signal",   "mFo-DFc diff [7,4,2,2] w0.1",  "sig",    0.523, 0.611, 0.628, 1.431, False, False),
    ("Signal",   "mFo-DFc diff · input-only (w0)","sig",   0.535, 0.618, 0.633, 1.431, False, False),
    ("Signal",   "mFo-DFc diff · recon w0.3",     "sig",   0.502, 0.588, 0.600, 1.477, False, False),
    # ── QUEUE (2026-08-02, running overnight; hollow = pending, priority order) ──
    ("Corpus",   "apo+holo 124K (GOAL, ConcatDataset)", "corp", 0.546, 0.622, 0.642, 1.378, False, False),
    ("Preproc",  "RAW density (unclipped)",       "prep",   0.542, 0.612, 0.631, 1.439, False, False),
    ("Objective","latent / JEPA (data2vec pure)", "obj",    0.358, 0.483, 0.484, 1.613, False, False),
    ("Masking",  "density-visible (mask atoms, keep ρ)", "msk", None, None, None, None, True, False),
    ("Masking",  "holo→apo (mask_strategy=ligand)","msk",   0.543, 0.630, 0.651, 1.379, False, False),
    ("Masking",  "dynamic (atom_biased)",         "msk",    None,  None,  None,  None,  True,  False),
    ("Masking",  "mask-more (block/cluster)",     "msk",    None,  None,  None,  None,  True,  False),
]
GRP_COLOR = {"champ":"#1a7f37","opt":"#bc4c00","cap":"#2b5fd0","corp":"#8e44ad","sig":"#d64545",
             "prep":"#0d9488","obj":"#7c3aed","msk":"#db2777"}
GRP_LABEL = {"champ":"champion","opt":"optimizer","cap":"capacity","corp":"corpus","sig":"signal (mFo-DFc)",
             "prep":"preprocessing","obj":"objective (JEPA)","msk":"masking"}

def R(t): return t[4]
def r_(t): return t[5]
def RM(t): return t[6]
def RUN(t): return t[7]
def CHAMP(t): return t[8]
def val(t,key): return RM(t) if key=="rmse" else (R(t) if key=="rho" else r_(t))
def lbl(t): return f'{t[0]} · {t[1]}'

def chart(key, ylabel, ymin, ymax, lower_better):
    W,H=880,300; x0,x1,y0,y1=60,858,16,262; n=len(TRIALS)
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
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-2}" font-size="10" fill="#5b6678" text-anchor="middle">trial</text>')
    for i,t in enumerate(TRIALS):
        v=val(t,key); col=GRP_COLOR.get(t[2],"#1d6fd0")
        if RUN(t) or v is None:
            s.append(f'<circle cx="{xs[i]:.1f}" cy="{H-42}" r="4" fill="none" stroke="#9aa3b2" stroke-width="1.5"><title>{lbl(t)} — running / n/a</title></circle>'); continue
        yy=y(v); champ=CHAMP(t); rr=6.5 if champ else 4.6
        ring=' stroke="#d97706" stroke-width="2.3"' if champ else ''
        s.append(f'<circle cx="{xs[i]:.1f}" cy="{yy:.1f}" r="{rr}" fill="{col}"{ring} opacity="{1.0 if champ else 0.8}"><title>{lbl(t)} — {key} {v:.3f}</title></circle>')
        if champ: s.append(f'<text x="{xs[i]:.1f}" y="{yy-rr-3:.1f}" font-size="9.5" font-weight="700" fill="#1c2433" text-anchor="middle">{v:.3f}</text>')
    s.append(f'<text x="13" y="{(y0+y1)//2}" font-size="11" fill="#5b6678" transform="rotate(-90 13 {(y0+y1)//2})" text-anchor="middle">{ylabel}</text></svg>')
    return ''.join(s)

def dcell(v, cv, lower):
    if v is None: return '<span class="pm">running</span>'
    d=v-cv; cls='mut' if abs(d)<1e-9 else ('pos' if (d<0 if lower else d>0) else 'neg')
    return f'<span class="num">{v:.3f}</span><span class="dlt {cls}">{d:+.3f}</span>'

rows=[]
for i,t in enumerate(TRIALS):
    theme,var,grp,valr,rho,r,rmse,run,champ=t
    cls='ref' if champ else ('mut' if run else '')
    tb=f'<span style="color:{GRP_COLOR.get(grp)};font-weight:700">{theme}</span>'
    star=' ★' if champ else ''
    if valr is None or rho is None:
        valcell='<span class="pm">—</span>'; gapcell='<span class="pm">—</span>'
    else:
        gap=valr-rho; gcls='neg' if gap>0.02 else 'mut'
        valcell=f'<span class="num">{valr:.3f}</span>'; gapcell=f'<span class="{gcls}">{gap:+.3f}</span>'
    rows.append(f'<tr class="{cls}"><td>T{i+1}</td><td>{tb}</td><td style="white-space:normal">{var}{star}</td>'
                f'<td>{valcell}</td><td>{dcell(rho,CH["rho"],False)}</td><td>{gapcell}</td>'
                f'<td>{dcell(r,CH["r"],False)}</td><td>{dcell(rmse,CH["rmse"],True)}</td></tr>')

html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signal / optimizer / architecture campaign — trials vs champion</title><style>
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
td:nth-child(n+4),th:nth-child(n+4){{text-align:right}}
.num{{font-weight:660}}.pm{{color:#9aa3b2;font-weight:500;font-size:11.5px}}
.pos{{color:#1d5a3a;font-weight:700}}.neg{{color:#b4413a;font-weight:700}}.mut td,.mut{{color:#8a93a0}}
.dlt{{display:block;font-size:10px;font-weight:700;margin-top:1px}}
tr.ref td{{background:#eaf5ee;font-weight:600}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.6}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:11.5px;color:#5b6678}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER · 2026-08-06</div>
<h1>Signal / optimizer / architecture campaign — trials vs champion</h1>
<div class="sub">Does a new <b>signal</b> (mFo-DFc), <b>optimizer</b> (Muon), <b>capacity</b>, <b>corpus</b> (apo), or
<b>preprocessing</b> (raw density) beat the champion recipe? Frozen affinity probe on <code>lp_edrscc_v2</code>
(Kd/Ki; val 817, test 1320; 3 seeds). Red = champion ceiling; Δ = vs champion; hollow = running.</div></header>

<div class="card best-card"><h2>★ Champion (ceiling)</h2>
<div class="note"><b>PLINDER v2 (112K)</b> · 100M ChannelViT [7,4,2] C+D+G · mask 0.75 · 50 ep · ema 0.999
&mdash; Pearson r <b>0.660</b> · Spearman ρ <b>0.644</b> · RMSE <b>1.349</b>. No trial has beaten it yet.</div></div>

<div class="card"><h2>Spearman ρ by trial</h2><p class="csub">Red line = champion ceiling (0.644). Hollow = running.</p>{chart('rho','test ρ (Spearman)',0.52,0.66,False)}</div>
<div class="card"><h2>Pearson r by trial</h2>{chart('r','test r (Pearson)',0.54,0.665,False)}</div>
<div class="card"><h2>RMSE (pK) by trial — lower is better</h2>{chart('rmse','test RMSE (pK)',1.33,1.57,True)}
<div class="legend">{''.join(f'<span><span style="width:10px;height:10px;border-radius:50%;background:{c}"></span>{GRP_LABEL[k]}</span>' for k,c in GRP_COLOR.items())}</div></div>

<div class="card"><h2>Trial table — val + test</h2>
<p class="csub">Δ = vs champion (<span class="pos">green</span> better / <span class="neg">red</span> worse; RMSE lower=better).
<b>val−test gap</b>: negative = test&gt;val (normal — val n=817 harder subset); <span class="neg">positive &gt;0.02</span> flags overfitting.
<span style="background:#eaf5ee;padding:0 4px;border-radius:3px">Champion ★</span>.</p>
<table><thead><tr><th>#</th><th>Theme</th><th>Variant</th><th>val ρ</th><th>test ρ (Δ)</th><th>val−test</th><th>test r (Δ)</th><th>RMSE (Δ)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="note" style="margin-top:12px"><b>Verdict so far:</b> champion ρ 0.644 unbeaten across optimizer, capacity, corpus, and new-signal axes.
<b>Muon</b> peaks at muon_lr 0.02 (0.632) &lt; AdamW; 0.05 overshoots (0.527).
<b>Capacity</b> saturates — 10M ties 100M (0.640).
<b>apo v4</b> (0.561) is corpus-size-limited (11.4K vs 112K), not an apo failure.
<b>mFo-DFc diff</b> hurts (0.611) — aligned + correctly processed (verified), but redundant/refinement-noise for affinity. Direction dropped.</div>
<div class="note" style="margin-top:10px"><b>Queue (overnight, priority):</b> <b>1</b> apo+holo 124K (v2 holo ∪ v4 apo, ConcatDataset, champion recipe, storage-free) = the goal encoder ·
<b>2</b> raw density (unclipped) · <b>3</b> latent/JEPA · then the <b>masking family</b> (single-component, box path):
<b>4</b> density-visible (mask atoms, keep density → reconstruct chemistry from density; cross-modal, MultiMAE-style) ·
<b>5</b> holo→apo (mask_strategy=ligand — reconstruct ligand region from pocket) ·
<b>6</b> dynamic (atom_biased) · <b>7</b> mask-more (block/cluster).
<b>Masking lit:</b> random ≈ best for plain MAE (block degrades &gt;0.75) but <i>block beats random for JEPA</i>; semantic/part (SemMAE) &amp; adaptive (AdaMAE, 95% mask) target informative regions; cross-modal (MultiMAE/M3AE) predicts one modality from another. Adaptive-learned RL sampler deferred.</div></div>
</div></body></html>"""
out=HERE/"trail.html"; out.write_text(html)
print(f"wrote {out} ({len(html)} bytes, {len(TRIALS)} trials)")
