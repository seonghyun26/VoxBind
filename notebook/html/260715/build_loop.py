#!/usr/bin/env python3
"""build_loop.py — render 260715/loop.html, the 100M-on-v2 autoresearch loop tracker.
Same visual language as 260701/autoresearch.html. Hand-maintained trial list below
(sweep numbers are 50ep / probe e49 unless noted; the reference grid is 100ep).
Re-run after each round lands to refresh the page:  python build_loop.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "loop.html")

# ---- reference grid (100ep, probe e99) — context, NOT the same budget as the sweep ----
GRID = [
    ("40M · dim512×12",      "PLINDER v1 (17.5K)", 0.637, True,  "prior campaign ceiling"),
    ("40M · dim512×12",      "PLINDER v2 (112K)",  0.620, False, "more data hurt the small model"),
    ("~85M · dim512×24 (depth)", "PLINDER v2 (112K)", 0.608, False, "depth > width"),
    ("~98M · dim768×12 (width)", "PLINDER v2 (112K)", 0.595, False, "widening regressed"),
]

# ---- the loop: 100M balanced encoder (dim640/L18/h10, 99.47M) on v2, 50ep / e49 ----
# rho=None → pending (running/queued)
TRIALS = [
    dict(t="T1", label="mask 0.50", cfg="d640/L18/h10", knob="mask_ratio 0.50 (campaign default)",
         params="99.5M", rho=0.614, sd=.001, r=0.633, rmse=1.391, status="done"),
    dict(t="T2", label="mask 0.75", cfg="d640/L18/h10", knob="mask_ratio 0.75 (MAE-style)",
         params="99.5M", rho=0.644, sd=.001, r=0.660, rmse=1.349, status="done"),
    dict(t="T3", label="mask 0.85", cfg="d640/L18/h10", knob="mask_ratio 0.85 — overshoots the peak",
         params="99.5M", rho=0.623, sd=.006, r=0.643, rmse=1.419, status="done"),
    dict(t="T4", label="heads 16",  cfg="d640/L18/h16", knob="heads 10→16 @ mask0.75 (head_dim 64→40, free)",
         params="99.5M", rho=0.633, sd=.002, r=0.649, rmse=1.361, status="done"),
    dict(t="T5", label="deeper (h8)", cfg="d512/L28/h8", knob="deeper-narrower (L28), head_dim 64 @ mask0.75 — overshoots depth",
         params="97.1M", rho=0.616, sd=.004, r=0.633, rmse=1.475, status="done"),
    dict(t="T6", label="winner · 100ep", cfg="d640/L18/h10", knob="T2 (0.644) model continued e49→e99 — MORE TRAINING OVERSHOOTS",
         params="99.5M", rho=0.618, sd=.002, r=0.637, rmse=1.370, status="done"),
]

CEIL = 0.637   # prior-campaign ceiling reference line
BUILT = "2026-07-06"

CSS = """*{box-sizing:border-box}
body{margin:0;background:#f5f6f8;color:#1c2433;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.page{max-width:980px;margin:0 auto;padding:40px 24px 90px}
header{border-bottom:2px solid #1c2433;padding-bottom:14px;margin-bottom:24px}
header .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#5b6678;margin-bottom:6px}
header h1{font-size:25px;font-weight:650;margin:0}
header .sub{font-size:13px;color:#5b6678;margin-top:6px}
.card{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:20px;margin-bottom:26px;box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}
h2{font-size:17px;font-weight:640;margin:0 0 4px}
.csub{font-size:12.5px;color:#5b6678;margin:0 0 14px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:12px;color:#5b6678}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:left;vertical-align:top;border-top:1px solid #e3e7ee}
thead th{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#5b6678;border:0;border-bottom:1px solid #e3e7ee}
td:not(:first-child):not(:nth-child(2)):not(:nth-child(3)),th:not(:first-child):not(:nth-child(2)):not(:nth-child(3)){text-align:right}
.num{font-weight:660}.pm{color:#5b6678;font-weight:500;font-size:11.5px}
.pos{color:#1d5a3a;font-weight:700}.neg{color:#b4413a;font-weight:700}.mut td,.mut{color:#5b6678}.small{font-size:12.5px}
.dlt{display:block;font-size:10px;font-weight:700;margin-top:2px}
.ms{white-space:nowrap}
tr.ref td{background:#eef2f7}
.tagref{font-size:10px;font-weight:700;color:#1d4ed8;background:#e8effe;padding:1px 6px;border-radius:999px}
.tagrun{font-size:10px;font-weight:700;color:#b07a17;background:#fcf3e0;padding:1px 6px;border-radius:999px}
.tagq{font-size:10px;font-weight:700;color:#5b6678;background:#eef1f5;padding:1px 6px;border-radius:999px}
a{color:#1d4ed8;text-decoration:none;font-weight:600}a:hover{text-decoration:underline}
.best{font-weight:700;color:#1d5a3a;background:#eaf5ee;border-radius:5px;padding:1px 5px}
tr.hl td{background:#f3f8ff}
.note{font-size:12.5px;color:#5b6678;line-height:1.6}
.note b{color:#1c2433}
.kpi{display:flex;gap:20px;flex-wrap:wrap;margin:2px 0 4px}
.kpi .k{background:#f7f9fb;border:1px solid #e3e7ee;border-radius:10px;padding:10px 16px;min-width:150px}
.kpi .k .v{font-size:22px;font-weight:720;color:#1d5a3a}
.kpi .k .l{font-size:11px;color:#5b6678;text-transform:uppercase;letter-spacing:.04em}
"""


def rho_chart(trials):
    """SVG scatter of sweep ρ by trial, with the 0.637 prior-ceiling reference line."""
    W, H = 820, 300
    x0, x1, y0, y1 = 56, 780, 20, 250
    lo, hi = 0.58, 0.66
    done = [t for t in trials if t["rho"] is not None]
    n = len(trials)

    def yv(v):
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    def xv(i):
        return x0 + (x1 - x0) * (i / max(1, n - 1))

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:840px" '
         f'font-family="-apple-system,Segoe UI,sans-serif">']
    # gridlines
    g = lo
    while g <= hi + 1e-9:
        y = yv(g)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#eef1f5"/>')
        s.append(f'<text x="{x0-7}" y="{y+3:.1f}" font-size="10" fill="#5b6678" text-anchor="end">{g:.2f}</text>')
        g += 0.01
    # ceiling line
    yc = yv(CEIL)
    s.append(f'<line x1="{x0}" y1="{yc:.1f}" x2="{x1}" y2="{yc:.1f}" stroke="#b4413a" stroke-width="1.3" stroke-dasharray="5 4" opacity="0.8"/>')
    s.append(f'<text x="{x1}" y="{yc-5:.1f}" font-size="10.5" font-weight="700" fill="#b4413a" text-anchor="end">prior ceiling 0.637 (v1 40M)</text>')
    # best-so-far frontier over done trials
    best = -1
    front = []
    for i, t in enumerate(trials):
        if t["rho"] is not None:
            if t["rho"] > best:
                best = t["rho"]; front.append((xv(i), yv(t["rho"])))
    if len(front) >= 2:
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in front)
        s.append(f'<path d="{d}" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="2 3" opacity="0.7"/>')
    # x labels + points
    bestv = max((t["rho"] for t in done), default=None)
    for i, t in enumerate(trials):
        x = xv(i)
        s.append(f'<text x="{x:.1f}" y="{H-18}" font-size="10.5" fill="#1c2433" text-anchor="middle">{t["t"]}</text>')
        s.append(f'<text x="{x:.1f}" y="{H-5}" font-size="9" fill="#5b6678" text-anchor="middle">{t["label"]}</text>')
        if t["rho"] is None:
            s.append(f'<text x="{x:.1f}" y="{yv((lo+hi)/2):.1f}" font-size="15" fill="#c4ccd6" text-anchor="middle">?</text>')
            tag = "running" if t["status"] == "running" else "queued"
            s.append(f'<text x="{x:.1f}" y="{yv((lo+hi)/2)+14:.1f}" font-size="8.5" fill="#b07a17" text-anchor="middle">{tag}</text>')
            continue
        isbest = (t["rho"] == bestv)
        y = yv(t["rho"])
        r = 7 if isbest else 5.5
        stroke = ' stroke="#d97706" stroke-width="2.5"' if isbest else ''
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#1d6fd0"{stroke}><title>{t["label"]} — ρ {t["rho"]:.3f}</title></circle>')
        s.append(f'<text x="{x:.1f}" y="{y-10:.1f}" font-size="10.5" font-weight="700" fill="#1c2433" text-anchor="middle">{t["rho"]:.3f}</text>')
    s.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="#5b6678" transform="rotate(-90 14 {(y0+y1)/2:.0f})" text-anchor="middle">test ρ (Spearman)</text>')
    s.append('</svg>')
    return "".join(s)


def grid_rows():
    r = []
    for enc, corp, rho, best, note in GRID:
        cls = ' class="hl"' if best else ''
        val = f'<span class="best">{rho:.3f}</span>' if best else f'<span class="num">{rho:.3f}</span>'
        r.append(f'<tr{cls}><td style="white-space:normal">{enc}</td><td style="white-space:normal">{corp}</td>'
                 f'<td><span class="ms">{val}</span></td><td class="small" style="white-space:normal">{note}</td></tr>')
    return "".join(r)


def trial_rows(trials):
    bestv = max((t["rho"] for t in trials if t["rho"] is not None), default=None)
    base = next((t["rho"] for t in trials if t["rho"] is not None), None)  # T1 = mask0.50 baseline
    r = []
    for t in trials:
        if t["rho"] is None:
            tag = f'<span class="tag{"run" if t["status"]=="running" else "q"}">{t["status"]}</span>'
            r.append(f'<tr class="mut"><td class="num" style="white-space:normal"><span class="pm">{t["t"]}</span> {t["label"]} {tag}</td>'
                     f'<td class="small">{t["cfg"]}</td><td class="small" style="white-space:normal">{t["knob"]}</td>'
                     f'<td class="tbd">—</td><td class="tbd">—</td><td class="tbd">—</td></tr>')
            continue
        isbest = (t["rho"] == bestv)
        rho = f'<span class="best">{t["rho"]:.3f}</span>' if isbest else f'<span class="num">{t["rho"]:.3f}</span>'
        dlt = ""
        if base is not None and t["rho"] != base:
            d = t["rho"] - base
            cls = "pos" if d > 0 else "neg"
            dlt = f'<span class="dlt {cls}">{d:+.3f}</span>'
        r.append(f'<tr><td class="num" style="white-space:normal"><span class="pm">{t["t"]}</span> {t["label"]}</td>'
                 f'<td class="small">{t["cfg"]}</td><td class="small" style="white-space:normal">{t["knob"]}</td>'
                 f'<td><span class="ms">{rho}<span class="pm"> ±{t["sd"]:.3f}</span></span>{dlt}</td>'
                 f'<td><span class="num">{t["r"]:.3f}</span></td><td><span class="num">{t["rmse"]:.3f}</span></td></tr>')
    return "".join(r)


bestv = max((t["rho"] for t in TRIALS if t["rho"] is not None), default=0)
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>100M-on-v2 autoresearch loop — does bigger data+model change the recipe?</title>
<style>{CSS}.tbd{{color:#b07a17;font-weight:600;font-size:12px}}</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER v2 · autoresearch loop</div>
<h1>Bigger data + bigger model: does the recipe change?</h1>
<div class="sub">A brief self-paced autoresearch loop on a <b>balanced ~100M encoder</b> (dim&nbsp;640 &times; depth&nbsp;18 &times; 10&nbsp;heads,
99.47M) pretrained on <b>PLINDER v2</b> (112K, C+D+G, ChannelViT&nbsp;[7,4,2]), sweeping the recipe the v1 campaign had frozen.
Frozen-probe affinity on <code>lp_edrscc_v2</code> (Kd/Ki, 3850/817/1320, 3 seeds). <b>Sweep runs are 50&nbsp;ep / probe&nbsp;e49</b>
for speed — relative comparisons only; the winner gets full-trained to 100&nbsp;ep for the headline table. Built {BUILT}.</div></header>

<div class="card"><h2>Why this loop</h2>
<p class="csub">The v1 campaign (~30 trials) plateaued at ρ&nbsp;0.637 and nothing beat it — capacity, depth, masking <i>strategy</i>,
pretext, finetune all tied-or-hurt. Then the 2&times;2 size&times;corpus grid showed <b>more data + more params both regressed</b> the
frozen probe. Open question: were those recipes (mask&nbsp;0.50, etc.) tuned for the <i>small</i>&nbsp;v1 regime? This loop re-opens the
recipe on the bigger model + bigger corpus.</p>
<div class="kpi">
  <div class="k"><div class="v">0.644</div><div class="l">best so far (T2, 50ep)</div></div>
  <div class="k"><div class="v">+0.007</div><div class="l">vs 0.637 ceiling (at 50ep; 100ep→0.618)</div></div>
  <div class="k"><div class="v">mask 0.75</div><div class="l">the lever that moved</div></div>
</div>
<table><thead><tr><th>Reference grid (100ep)</th><th>Corpus</th><th>ρ</th><th>note</th></tr></thead>
<tbody>{grid_rows()}</tbody></table>
<div class="note" style="margin-top:10px">Context only — the grid is 100&nbsp;ep; the loop below is a 50&nbsp;ep sweep, so numbers aren't directly comparable
until the winner is full-trained.</div></div>

<div class="card"><h2>The loop · test ρ by trial</h2>
<p class="csub">Balanced 100M encoder on v2; each trial changes ONE recipe knob. Gold ring = best-so-far. Red dashed = the prior-campaign
ceiling (0.637, v1 40M). <b>? = running / queued.</b></p>
{rho_chart(TRIALS)}
<div class="legend"><span style="display:inline-flex;align-items:center;gap:5px"><span style="width:11px;height:11px;border-radius:50%;background:#1d6fd0"></span>trial (50ep e49)</span>
<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:13px;height:13px;border-radius:50%;background:#fff;border:2.5px solid #d97706"></span>best-so-far</span>
<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:16px;height:0;border-top:2px dashed #b4413a"></span>prior ceiling 0.637</span></div></div>

<div class="card"><h2>Trial table</h2>
<p class="csub">Δ next to ρ = vs <b>T1 (mask 0.50)</b>, the campaign-default baseline on this 100M model. <span class="pos">green</span>=better.
Sweep = 50&nbsp;ep / probe&nbsp;e49, 3 seeds.</p>
<table><thead><tr><th>Trial</th><th>Arch</th><th>What changed</th><th>test ρ</th><th>test r</th><th>test RMSE</th></tr></thead>
<tbody>{trial_rows(TRIALS)}</tbody></table>
<div class="note" style="margin-top:10px">
<b>Round 1 (mask sweep) — the plateau broke.</b> mask&nbsp;0.75 (ρ&nbsp;0.644) beats the campaign-default 0.50 (0.614) by
<b>+0.030</b>, and at only 50&nbsp;ep it already <b>clears the 0.637</b> that held across the entire v1 campaign. Bigger data+model
wants <b>much heavier masking</b> (MAE-style) — the first real lever after ~30 trials of ties.<br><br>
<b>Round 2 — the recipe converged.</b> Two clean results: (1) <b>mask peaks at 0.75</b> — pushing to 0.85 (T3, 0.623) <i>overshoots</i>,
so the curve is 0.614&nbsp;→&nbsp;<b>0.644</b>&nbsp;→&nbsp;0.623. (2) <b>More heads hurt</b> — h16 (T4, 0.633, head_dim&nbsp;40) &lt; h10
(0.644, head_dim&nbsp;64); the bigger head_dim wins. The leader stays <b>T2: d640/L18/h10 @ mask&nbsp;0.75 = 0.644</b>.<br><br>
<b>Round 3.</b> T5 (depth reshape, dim&nbsp;512&times;28, h8/head_dim&nbsp;64) = ρ&nbsp;<b>0.616</b> &mdash; going <i>even deeper overshoots</i>
(−0.028 vs the balanced d640/L18). So the recipe has a clean optimum on every axis: <b>mask 0.75</b> (0.85 too high), <b>d640×18 balanced</b>
(d768×12 too wide, d512×28 too deep), <b>head_dim 64</b> (h16 too many heads). Winner = <b>d640/L18/h10 @ mask&nbsp;0.75 = 0.644</b>.<br><br>
<b>T6 = the full run (100ep) — training OVERSHOOTS too.</b> Continuing the 0.644 model from e49 to e99 <i>drops</i> it to
ρ&nbsp;<b>0.618</b> (−0.026). Same trajectory, so it's clean: the frozen-probe quality <b>peaks at ~e49 and degrades by e99</b> — the encoder
overfits the reconstruction task. So the <b>epoch budget is another axis with an early optimum</b> (like mask 0.75 and the balanced arch).
Consequence: at <i>matched 100&nbsp;ep</i> this config = 0.618 &lt; the 40M·v1 grid ceiling (0.637); the 0.644 break is a <b>50-ep</b> number.
The grid (all 100&nbsp;ep, mask&nbsp;0.50) may be <i>both</i> under-masked and over-trained — see the overnight hypotheses.</div></div>

</div></body></html>
"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)} bytes) · best-so-far ρ {bestv:.3f}")
