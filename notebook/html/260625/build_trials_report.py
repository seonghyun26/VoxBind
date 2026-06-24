#!/usr/bin/env python
"""Canonical full-results + autoresearch tracker → autoresearch.html.

Single source of truth: notebook/html/260625/clean_results.csv (folded live by the overnight
monitor). Covers the WHOLE study:
  Phase 1 — input modality (plain ViT: coords vs C+D+G), both splits (lp_edrscc_v2 + lp raw)
  Phase 2 — encoder architecture (plain ViT / ChannelViT / ChA-MAEViT for C+D+G), both splits
  Phase 3 — ChannelViT autoresearch sweep (grouping + knobs) on lp_edrscc_v2, with inline-SVG
            Pareto progression charts + trial table + queued/running list
No JS. Re-run (or let the monitor re-run) to refresh as rounds land.

Run:  python notebook/html/260625/build_trials_report.py
"""
import csv as csvmod, html as htmlmod, math, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # notebook/html/260625/
CSV = HERE / "clean_results.csv"                             # single source of truth
EXPS = HERE.parents[2] / "voxbind" / "exps"
RES = HERE.parents[2] / "voxbind" / "dataset" / "data" / "pdbbind" / "results"
OUT = HERE / "autoresearch.html"

BASE_LABEL = "C+D+G ChannelViT"                              # [7,4,1,1] mask0.5 = the autoresearch base
TRIALS = [
    ("coords ChannelViT g[7,4]", "coords ChannelViT", "Coords (C)",  "matched control, groups [7,4]",      "coords"),
    ("C+D+G ChannelViT",         "[7,4,1,1] base",     "C+D+G",       "base recipe (mask 0.50)",            "base"),
    ("ChannelViT g[7,4,2]",      "[7,4,2]",            "C+D+G",       "group density+gradmag (2-ch)",       "group"),
    ("ChannelViT g[11,2]",       "[11,2]",             "C+D+G",       "atoms-11 / density+gradmag-2",       "group"),
    ("ChannelViT g[11,1,1]",     "[11,1,1]",           "C+D+G",       "merge all 11 atom channels",         "group"),
    ("ChannelViT g[7,4,2] dw0.3","dens-wt 0.3",        "C+D+G",       "density+gradmag recon wt 0.1→0.3",   "knob"),
    ("ChannelViT g[7,4,2] mask0.6","mask 0.6",         "C+D+G",       "mask_ratio 0.50→0.60",               "knob"),
    ("ChannelViT g[7,4,2] mask0.75","mask 0.75",       "C+D+G",       "mask_ratio 0.50→0.75",               "knob"),
    ("ChannelViT g[7,4,2] dw1.0","dens-wt 1.0",        "C+D+G",       "density+gradmag recon wt 0.1→1.0",   "knob"),
    ("ChannelViT g[7,4,2] rope3d","rope3d",            "C+D+G",       "pos_encoding=rope3d (group-tiled)",  "knob"),
]
PENDING_LABELS = {
    "260624_ar_cvit_rope":    ("rope3d ▶ NEXT",   "pos_encoding=rope3d (group-tiled; now wired)"),
    "260624_ar_cvit_lr2e4":   ("lr 2e-4",         "learning rate 1e-4→2e-4"),
    "260624_ar_cvit_mask04":  ("mask 0.4",        "mask_ratio 0.50→0.40"),
    "260624_ar_cvit_atombias":("atom-bias",       "mask_strategy=atom_biased"),
    "260624_ar_cvit_mask03":  ("mask 0.3",        "mask_ratio 0.50→0.30"),
    "260624_ar_cvit_dw005":   ("dens-wt 0.05",    "density+gradmag recon wt 0.1→0.05"),
    "260624_ar_cvit_lr5e5":   ("lr 5e-5",         "learning rate 1e-4→5e-5"),
    "260624_ar_cvit_g13":     ("[13]",            "all channels in one group"),
    "260624_ar_cvit_dw05":    ("dens-wt 0.5",     "density+gradmag recon wt 0.1→0.5"),
    "260624_ar_cvit_block4":  ("block 4",         "mask block_size 8→4"),
    "260624_ar_cvit_wd1e1":   ("wd 0.1",          "weight decay 0.05→0.10"),
    "260624_ar_cvit_headd3":  ("head-d 3",        "recon head_depth 2→3"),
}

DATA = {}                                                    # (label, split) → dict(rho,rho_sd,r,r_sd,n)


def esc(x): return htmlmod.escape(str(x))


def load_csv():
    out = {}
    if not CSV.exists():
        return out
    for r in csvmod.DictReader(open(CSV, encoding="utf-8")):
        try:
            out[(r["label"], r["split"])] = dict(
                rho=float(r["test_rho_mean"]), rho_sd=float(r["test_rho_std"]),
                r=float(r["test_r_mean"]), r_sd=float(r["test_r_std"]), n=int(r["n_test"]))
        except (KeyError, ValueError):
            pass
    return out


def g(label, split="lp_edrscc"):
    return DATA.get((label, split))


def cell(d, kind="rho", best=False):
    if not d:
        return '<span class="mut">—</span>'
    v, sd = (d["rho"], d["rho_sd"]) if kind == "rho" else (d["r"], d["r_sd"])
    c = "best" if best else "num"
    return f'<span class="{c}">{v:.3f}</span><span class="pm"> ±{("%.3f"%sd).lstrip("0")}</span>'


# ── Phase 1 + Phase 2 (full-study baselines, both splits) ───────────────────────
def baseline_tables():
    def row(enc, mod, label, best_v2=False, cls=""):
        v2, lp = g(label, "lp_edrscc"), g(label, "lp")
        return (f'<tr class="{cls}"><td style="white-space:normal">{enc}</td><td style="white-space:normal">{mod}</td>'
                f'<td>{cell(v2,"rho",best_v2)}</td><td>{cell(v2,"r")}</td>'
                f'<td class="div">{cell(lp,"rho")}</td><td>{cell(lp,"r")}</td></tr>')

    # Phase 1 — modality (plain ViT). Δ density row.
    def delta_row():
        out = '<tr class="delta"><td>Δ density (C+D+G − C)</td><td></td>'
        for sp in ("lp_edrscc", "lp"):
            c, d = g("coords-only", sp), g("C+D+G", sp)
            dv = f'{d["rho"]-c["rho"]:+.3f}' if (c and d) else "—"
            klass = "pos" if (c and d and d["rho"]-c["rho"] > 0.0015) else "mut"
            extra = ' class="div"' if sp == "lp" else ''
            out += f'<td{extra}><span class="{klass}">{dv}</span></td><td></td>'
        return out + '</tr>'

    p1 = (f'<table><thead><tr><th>Encoder</th><th>Modality</th>'
          f'<th>ρ (lp_edrscc_v2)</th><th>r</th><th class="div">ρ (lp raw)</th><th>r</th></tr></thead><tbody>'
          + row("plain ViT", "C (coords, 11ch)", "coords-only")
          + row("plain ViT", "C+D+G (13ch)", "C+D+G")
          + delta_row() + '</tbody></table>')

    # Phase 2 — architecture (C+D+G). ChannelViT is best on v2.
    p2 = (f'<table><thead><tr><th>Encoder</th><th>Input</th>'
          f'<th>ρ (lp_edrscc_v2)</th><th>r</th><th class="div">ρ (lp raw)</th><th>r</th></tr></thead><tbody>'
          + row("plain ViT (fused)", "C+D+G", "C+D+G")
          + row("ChannelViT [7,4,1,1]", "C+D+G", "C+D+G ChannelViT", best_v2=True, cls="hl")
          + row("ChA-MAEViT", "C+D+G", "C+D+G ChA-MAEViT", cls="mut")
          + '</tbody></table>')
    return p1, p2


# ── Phase 3 autoresearch progression chart ──────────────────────────────────────
def svg_scatter(rows, axis_label):
    rows = [(i, n, v) for i, n, v in rows if v is not None]
    if not rows:
        return "<p class='mut'>No values yet.</p>"
    n = len(rows)
    vals = [v for _, _, v in rows]
    y0 = math.floor((min(vals) - 0.01) * 100) / 100
    y1 = math.ceil((max(vals) + 0.01) * 100) / 100
    W, H, padL, padR, padT, padB = 880, 320, 56, 20, 16, 40
    def Y(v): return padT + (1 - (v - y0) / (y1 - y0)) * (H - padT - padB)
    def X(i): return padL + (0.5 if n == 1 else i / (n - 1)) * (W - padL - padR)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:900px" font-family="-apple-system,Segoe UI,sans-serif">']
    gg = y0
    while gg <= y1 + 1e-9:
        yy = Y(gg)
        p.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eef1f5"/>')
        p.append(f'<text x="{padL-7}" y="{yy+3:.1f}" font-size="10" fill="#5b6678" text-anchor="end">{gg:.2f}</text>')
        gg += 0.01
    for k, (i, name, _v) in enumerate(rows):
        p.append(f'<text x="{X(k):.1f}" y="{H-padB+18:.1f}" font-size="9.5" fill="#1c2433" text-anchor="middle">{esc(name)}</text>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-4}" font-size="10.5" fill="#5b6678" text-anchor="middle">trial (campaign order)</text>')
    pareto, best, front = set(), -1e9, []
    for k, (i, name, v) in enumerate(rows):
        if v > best + 1e-9:
            best = v; pareto.add(k); front.append((X(k), Y(v)))
    if len(front) > 1:
        d = f'M {front[0][0]:.1f} {front[0][1]:.1f} ' + " ".join(f'L {x:.1f} {y:.1f}' for x, y in front[1:])
        p.append(f'<path d="{d}" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="2 3" opacity="0.7"/>')
    for k, (i, name, v) in enumerate(rows):
        x, y = X(k), Y(v)
        tip = f'{esc(name)} — {axis_label} {v:.3f}'
        if k in pareto:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#1d6fd0" stroke="#d97706" stroke-width="2.5"><title>{tip} (best-so-far)</title></circle>')
            p.append(f'<text x="{x:.1f}" y="{y-11:.1f}" font-size="10.5" font-weight="700" fill="#1c2433" text-anchor="middle">{v:.3f}</text>')
        else:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#1d6fd0" opacity="0.55"><title>{tip}</title></circle>')
    p.append(f'<text x="14" y="{padT+(H-padT-padB)/2:.0f}" font-size="11" fill="#5b6678" transform="rotate(-90 14 {padT+(H-padT-padB)/2:.0f})" text-anchor="middle">{esc(axis_label)}</text>')
    p.append("</svg>")
    return "".join(p)


def pending_rows():
    out = []
    if (EXPS / "260624_ar_cvit_g7_6").exists() and \
       not (RES / "probe_results_e99_v5_lp_edrscc_v2split_260624_ar_cvit_g7_6.csv").exists():
        out.append(("[7,6]", "group poc+density+gradmag (6-ch)", "RUNNING"))
    for tag, (nm, what) in PENDING_LABELS.items():
        if (RES / f"probe_results_e99_v5_lp_edrscc_v2split_{tag}.csv").exists():
            continue
        out.append((nm, what, "NEXT" if "rope" in tag else "QUEUED"))
    return out


def build():
    global DATA
    DATA = load_csv()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    base = (g(BASE_LABEL) or {}).get("rho")
    base_txt = f"{base:.3f}" if base is not None else "—"
    p1, p2 = baseline_tables()

    # Phase-3 sweep
    sp_rows, pe_rows, trows = [], [], ""
    completed = [(lbl, nm, inp, det) for lbl, nm, inp, det, fam in TRIALS if g(lbl)]
    best_rho = max((g(lbl)["rho"] for lbl, *_ in completed), default=None)
    for idx, (lbl, nm, inp, det) in enumerate(completed):
        d = g(lbl)
        sp_rows.append((idx, nm, d["rho"])); pe_rows.append((idx, nm, d["r"]))
        drho = d["rho"] - base if base is not None else 0.0
        dcls = "pos" if drho > 0.0015 else ("neg" if drho < -0.0015 else "mut")
        dtxt = "REF" if lbl == BASE_LABEL else f'{drho:+.3f}'
        star = " ⭐" if d["rho"] == best_rho else ""
        trows += (f'<tr><td class="num" style="white-space:normal">{esc(nm)}{star}</td>'
                  f'<td style="white-space:normal">{esc(inp)}</td>'
                  f'<td class="small" style="white-space:normal">{esc(det)}</td>'
                  f'<td>{cell(d,"rho", d["rho"]==best_rho)}</td><td><span class="{dcls}">{dtxt}</span></td><td>{cell(d,"r")}</td></tr>')
    best_txt = f"{best_rho:.3f}" if best_rho is not None else "—"

    prows = ""
    for name, what, st in pending_rows():
        sty = "background:#e8effe;color:#1d4ed8" if st in ("NEXT", "RUNNING") else "background:#eef1f6;color:#5b6678"
        prows += (f'<tr><td style="white-space:normal">{esc(name)}</td><td class="small" style="white-space:normal">{esc(what)}</td>'
                  f'<td><span style="font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px;{sty}">{st}</span></td></tr>')

    legend = ('<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:11px;height:11px;border-radius:50%;background:#1d6fd0"></span>trial</span>'
              ' <span style="display:inline-flex;align-items:center;gap:5px"><span style="width:13px;height:13px;border-radius:50%;background:#fff;border:2.5px solid #d97706"></span>best-so-far (Pareto frontier)</span>')

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Density modality, architecture & autoresearch — full results</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1c2433;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.page{{max-width:980px;margin:0 auto;padding:40px 24px 90px}}
header{{border-bottom:2px solid #1c2433;padding-bottom:14px;margin-bottom:24px}}
header .date{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#5b6678;margin-bottom:6px}}
header h1{{font-size:25px;font-weight:650;margin:0}}
header .sub{{font-size:13px;color:#5b6678;margin-top:6px}}
.card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:20px;margin-bottom:26px;box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}}
h2{{font-size:17px;font-weight:640;margin:0 0 4px}}
.csub{{font-size:12.5px;color:#5b6678;margin:0 0 14px}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:12px;color:#5b6678}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;text-align:left;vertical-align:top;border-top:1px solid #e3e7ee}}
thead th{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#5b6678;border:0;border-bottom:1px solid #e3e7ee}}
td:not(:first-child):not(:nth-child(2)),th:not(:first-child):not(:nth-child(2)){{text-align:right}}
.div{{border-left:2px solid #e3e7ee}}
.num{{font-weight:660}}.pm{{color:#5b6678;font-weight:500;font-size:11.5px}}
.pos{{color:#1d5a3a;font-weight:700}}.neg{{color:#b4413a;font-weight:700}}.mut td,.mut{{color:#5b6678}}.small{{font-size:12.5px}}
.best{{font-weight:700;color:#1d5a3a;background:#eaf5ee;border-radius:5px;padding:1px 5px}}
tr.hl td{{background:#f3f8ff}}
tr.delta td{{font-size:12.5px;color:#5b6678;background:#fcfdfe;font-style:italic}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.6}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER-OTF · full study</div>
<h1>Electron density: input modality, encoder architecture &amp; autoresearch</h1>
<div class="sub">Frozen-encoder PDBbind affinity probe (pK). All encoders self-supervised on the <b>identical</b>
PLINDER ligand-matched corpus (17,430/100, on-the-fly density resample-aug, 100 ep, eff-batch 128, uniform 50% mask),
differing only in input modality (Ph.1) or architecture (Ph.2/3). Metrics = test Spearman ρ / Pearson r, mean ± std
over 3 probe seeds. Built {ts} from <code>clean_results.csv</code>. <b>lp_edrscc_v2</b> = Kd/Ki-only, 3850/817/1320;
<b>lp raw</b> = LP-PDBBind new_split (coords sees a larger pool than C+D+G).</div></header>

<div class="card"><h2>Phase 1 — Input modality (plain ViT)</h2>
<p class="csub">Does adding electron density + ‖∇ρ‖ to the same plain-ViT encoder help? Same samples, only the channels differ.</p>
{p1}
<div class="note" style="margin-top:10px">Density helps: <b>+0.003</b> ρ on the clean Kd/Ki split (0.605→0.608), <b>+0.021</b> on raw lp (0.592→0.613).</div></div>

<div class="card"><h2>Phase 2 — Encoder architecture (C+D+G)</h2>
<p class="csub">Given the 13-ch C+D+G input, which encoder reads the density channels best?</p>
{p2}
<div class="note" style="margin-top:10px"><b>ChannelViT &gt; plain ViT &gt; ChA-MAEViT</b> on lp_edrscc_v2 (0.630 / 0.608 / 0.573). ChannelViT carries into Phase 3 as the autoresearch base.</div></div>

<div class="card"><h2>Phase 3 — ChannelViT autoresearch · test ρ (Spearman) by trial</h2>
<p class="csub">Each trial varies ONE grouping/knob off the ChannelViT base (ρ {base_txt}); gold = best-so-far. lp_edrscc_v2, 3 seeds. <b>Best so far: ρ {best_txt}</b>.</p>
{svg_scatter(sp_rows, 'test ρ (Spearman)')}</div>

<div class="card"><h2>Phase 3 — test r (Pearson) by trial</h2>
{svg_scatter(pe_rows, 'test r (Pearson)')}
<div class="legend">{legend}</div></div>

<div class="card"><h2>Phase 3 — trial table (completed)</h2>
<table><thead><tr><th>Trial</th><th>Input</th><th>What changed</th><th>test ρ</th><th>Δρ vs base</th><th>test r</th></tr></thead>
<tbody>{trows}</tbody></table></div>

<div class="card"><h2>Phase 3 — queued / running</h2>
<table><thead><tr><th>Trial</th><th>What changes</th><th>status</th></tr></thead>
<tbody>{prows}</tbody></table>
<div class="note" style="margin-top:10px">Two 4-GPU workers (GPU 0-3, 4-7) drain the queue autonomously; roleblob (4-channel) holds one set.
<b>rope3d</b> for the grouped patch-embed is now wired (RoPE table group-tiled) and is running.</div></div>

<div class="card"><h2>Reading it</h2>
<div class="note">
<b>Density</b> lifts ChannelViT <b>+0.033</b> over coords (best {best_txt} vs coords-ChannelViT 0.604).
The winning structure is <b>group density+gradmag together, keep lig/poc separate</b>
([7,4,2] &gt; [11,2] &gt; [7,4,1,1] &gt; [11,1,1]). Every knob tried so far hurts — raising mask ratio (0.6/0.75) or
recon weight (0.3 tied, 1.0 worse) only loses ground. Open-ended until stopped.<br><br>
<b>Reference:</b> TargetDiff EGNN (supervised) scores ρ 0.536 on its own LP-PDBBind split — a different, larger test set, not directly comparable.
</div></div>
</div></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({len(OUT.read_text()):,} bytes, static SVG, no JS)")
