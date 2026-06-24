#!/usr/bin/env python
"""build_results_h200.py — render results_h200.html from the 4-case PLINDER probe CSVs.

Idempotent: reads probe_results_e99_v5_lp_edrscc_v2split_<tag>.csv for the 4 encoders
(density / D+G  x  ViT / ChannelViT), computes mean±std (seeds) of test ρ / r / RMSE,
and writes results_h200.html. Cells with no CSV yet show "running".

Run:  python notebook/html/260625/build_results_h200.py
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home1/irteam/VoxBind/voxbind/dataset/data/pdbbind/results"
OUT = os.path.join(HERE, "results_h200.html")
EPOCH, VV = 99, "v5"

# logical cell -> (input label, arch label, probe --tag)
CELLS = [
    ("density-only", "ViT",        "donly_vit"),
    ("density-only", "ChannelViT", "donly_cvit"),
    ("density+gradmag", "ViT",        "dg_vit"),
    ("density+gradmag", "ChannelViT", "dg_cvit"),
    ("coords+density", "ViT",        "abd_vit"),
    ("coords+density", "ChannelViT", "abd_cvit"),
]


def load(tag):
    p = f"{RES}/probe_results_e{EPOCH}_{VV}_lp_edrscc_v2split_{tag}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return {
        "rho":  (df.test_spearman.mean(), df.test_spearman.std(ddof=0)),
        "r":    (df.test_pearson.mean(),  df.test_pearson.std(ddof=0)),
        "rmse": (df.test_rmse.mean(),     df.test_rmse.std(ddof=0)),
        "seeds": len(df),
    }


rows = [(inp, arch, tag, load(tag)) for inp, arch, tag in CELLS]
done = [r for r in rows if r[3]]
best_rho = max((r[3]["rho"][0] for r in done), default=None)


def fmt(v, d=3):
    return f"{v:.{d}f}"


def cell_rho(res):
    if not res:
        return '<span class="mut">running…</span>'
    m, s = res["rho"]
    cls = "num"
    star = " ★" if best_rho is not None and abs(m - best_rho) < 1e-9 else ""
    return f'<span class="{cls}">{fmt(m)}</span> <span class="pm">±{fmt(s,3)}</span>{star}'


# ── matrix (input x arch) ─────────────────────────────────────────────────────
def matrix_cell(inp, arch):
    for i, a, tag, res in rows:
        if i == inp and a == arch:
            return cell_rho(res)
    return '<span class="mut">—</span>'


matrix = f"""
<table><thead><tr><th>test ρ (Spearman)</th><th>ViT</th><th>ChannelViT</th></tr></thead><tbody>
<tr><td class="lbl">density-only</td><td>{matrix_cell('density-only','ViT')}</td><td>{matrix_cell('density-only','ChannelViT')}</td></tr>
<tr><td class="lbl">density + gradmag</td><td>{matrix_cell('density+gradmag','ViT')}</td><td>{matrix_cell('density+gradmag','ChannelViT')}</td></tr>
<tr><td class="lbl">coords + density</td><td>{matrix_cell('coords+density','ViT')}</td><td>{matrix_cell('coords+density','ChannelViT')}</td></tr>
</tbody></table>"""

# ── detail table ──────────────────────────────────────────────────────────────
detail_rows = ""
for inp, arch, tag, res in rows:
    if res:
        rho = f'<span class="num">{fmt(res["rho"][0])}</span> <span class="pm">±{fmt(res["rho"][1],3)}</span>'
        rr  = f'<span class="num">{fmt(res["r"][0])}</span> <span class="pm">±{fmt(res["r"][1],3)}</span>'
        rmse = f'{fmt(res["rmse"][0],3)}'
        status = f'<span class="pos">done</span> ({res["seeds"]} seeds)'
    else:
        rho = rr = rmse = '<span class="mut">—</span>'
        status = '<span class="mut">running</span>'
    detail_rows += (f'<tr><td class="lbl">{inp}</td><td>{arch}</td>'
                    f'<td>{rho}</td><td>{rr}</td><td>{rmse}</td><td class="small">{status}</td></tr>')

n_done = len(done)
HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLINDER 4-case — probe on lp_edrscc_v2 (H200)</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1c2433;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.page{{max-width:880px;margin:0 auto;padding:40px 24px 90px}}
header{{border-bottom:2px solid #1c2433;padding-bottom:14px;margin-bottom:24px}}
header .date{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#5b6678;margin-bottom:6px}}
header h1{{font-size:25px;font-weight:650;margin:0}}
header .sub{{font-size:13px;color:#5b6678;margin-top:8px;line-height:1.6}}
.card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:20px;margin-bottom:26px;box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}}
h2{{font-size:17px;font-weight:640;margin:0 0 14px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;text-align:left;vertical-align:top;border-top:1px solid #e3e7ee}}
thead th{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#5b6678;border:0;border-bottom:1px solid #e3e7ee}}
.lbl{{font-weight:600}}
.num{{font-weight:660}}.pm{{color:#5b6678;font-weight:500;font-size:12px}}
.pos{{color:#1d5a3a;font-weight:700}}.neg{{color:#b4413a;font-weight:700}}.mut{{color:#5b6678}}.small{{font-size:12.5px}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.65}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER 4-case campaign · H200</div>
<h1>Density-MAE pretrain → frozen probe on lp_edrscc_v2</h1>
<div class="sub">3 inputs × 2 archs, gradmag-only dropped → <b>4 cases</b>: {{density-only, density+gradmag}} × {{ViT, ChannelViT}}.
Encoders pretrained 100 ep on the coverage-gated PLINDER <code>_v2clean</code> set (17,407 crops, pdbbind-only
leakage exclusion). Frozen 512-D mean-pooled features → 2-layer MLP affinity probe on
<b>lp_edrscc_v2</b> (Kd/Ki-only, 3850/817/1320, 3 seeds). {n_done}/4 cases probed.</div></header>

<div class="card"><h2>test ρ (Spearman) — input × architecture</h2>
{matrix}
<div class="note" style="margin-top:10px">★ = best so far. Higher is better.</div></div>

<div class="card"><h2>Per-encoder detail</h2>
<table><thead><tr><th>Input</th><th>Arch</th><th>test ρ</th><th>test r</th><th>test RMSE</th><th>status</th></tr></thead>
<tbody>{detail_rows}</tbody></table></div>

<div class="card"><h2>Reading it</h2>
<div class="note">
These encoders see <b>only density / density+gradmag</b> — <b>no atom channels</b> — so they are far below the
13-channel atom+density+gradmag (C+D+G) probes (~0.57–0.60 on the v1 split): atom positions carry most of the
affinity signal. The point of this grid is the <b>relative</b> effect of (a) adding the gradmag channel and
(b) the ChannelViT patch-embed, on a pure-density encoder.<br><br>
<b>So far (ViT):</b> density-only {fmt(rows[0][3]['rho'][0]) if rows[0][3] else '—'} vs density+gradmag
{fmt(rows[2][3]['rho'][0]) if rows[2][3] else '—'} — adding gradmag did not help (slightly lower).
ChannelViT rows fill in as runs 3–4 finish.<br><br>
<b>Caveat:</b> the v1 baselines elsewhere (260625/autoresearch) are on a different split (5817/1498/2813, all
measurement types); these are <b>lp_edrscc_v2</b> (Kd/Ki-only, ~5,987) — not directly comparable.
</div></div>
</div></body></html>"""

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT}  ({n_done}/4 cases)")
for inp, arch, tag, res in rows:
    print(f"  {inp:16s} {arch:11s} {tag:11s} -> "
          + (f"ρ={fmt(res['rho'][0])}±{fmt(res['rho'][1],3)} r={fmt(res['r'][0])} rmse={fmt(res['rmse'][0],3)}" if res else "running"))
