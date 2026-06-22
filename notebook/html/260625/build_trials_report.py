#!/usr/bin/env python
"""Static report: the autoresearch trial campaign and how the frozen-probe test rho changed across
trials. Reads notebook/html/experiment_status.csv (the trial ledger) + the probe_results_*.csv files,
and writes autoresearch.html — inline-SVG scatter charts (no JS) + a delta table whose "what changed"
is split into Architecture / Input / Details. Re-run to refresh as rounds land.
Run:  python notebook/html/260625/build_trials_report.py
"""
import csv as csvmod, statistics, html as htmlmod, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent              # notebook/html/260625/
STATUS = HERE.parent / "experiment_status.csv"      # notebook/html/experiment_status.csv
PDB = HERE.parents[2] / "voxbind" / "dataset" / "data" / "pdbbind"
OUT = HERE / "autoresearch.html"

# Reference baselines (NEW ED+RSCC split, 5817/1498/2813) — value computed per-metric from each
# run's CSV, so both the Spearman and Pearson charts get correct dashed lines. (label, csv, cond, colour).
REF_KEYS = [
    ("coords-only (edrscc C)", "probe_results_e99_v5_plinder_edrscc_C.csv", "atomblob_ligvdw", "#b07a17"),
    ("fused C+D+G (edrscc CDG)", "probe_results_e99_v5_plinder_edrscc_CDG.csv", "atomblob_density_gradmag", "#5b6678"),
    ("Channel-ViT base (edrscc)", "probe_results_e99_v5_edrscc_cvbase.csv", "atomblob_density_gradmag", "#1d6fd0"),
]
# Charts to draw: (column, axis label).
METRICS = [("test_spearman", "test ρ (Spearman)"), ("test_pearson", "test r (Pearson)")]
BASE_CSV = ("probe_results_e99_v5_edrscc_cvbase.csv", "atomblob_density_gradmag")  # Channel-ViT base for Δ


def esc(x): return htmlmod.escape(str(x))


def metric(csv_name, cond, col):
    """(mean, std, k) of `col` for rows matching `cond`, or None."""
    p = PDB / "results" / csv_name
    if not csv_name or not p.exists():
        return None
    try:
        rows = [r for r in csvmod.DictReader(open(p)) if r.get("condition") == cond]
    except Exception:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (KeyError, TypeError, ValueError):
            pass
    if not vals:
        return None
    return (sum(vals) / len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def test_rho(csv_name, cond):
    return metric(csv_name, cond, "test_spearman")


def load_trials():
    out = []
    for r in csvmod.DictReader(open(STATUS, encoding="utf-8")):
        try:
            t = int(r.get("trial") or 0)
        except ValueError:
            t = 0
        if not t:
            continue
        out.append(dict(trial=t, what=(r.get("what") or "").strip(), status=(r.get("status") or "").strip(),
                        architecture=(r.get("architecture") or "").strip(),
                        input=(r.get("input") or "").strip(),
                        details=(r.get("details") or "").strip(),
                        csv=(r.get("result_csv") or "").strip(), cond=(r.get("cond") or "").strip(),
                        note=(r.get("note") or "").strip()))
    # de-dup by trial (keep last), sort
    by = {}
    for r in out:
        by[r["trial"]] = r
    return [by[k] for k in sorted(by)]


# colour a trial by experiment family (from its 'what')
def fam_color(what):
    w = what.lower()
    if "autoresearch" in w: return "#7c3aed"      # purple
    if "channel-vit" in w:  return "#1d6fd0"      # blue
    if "cha-mae" in w:      return "#0e7490"      # teal
    if "coords-only" in w:  return "#b07a17"      # amber
    if "reference" in w:    return "#2f6f4f"      # green
    return "#94a3b8"                               # grey (single-channel ablations)


def svg_scatter(trials, col, axis_label):
    """Scatter: x = trial, y = `col` performance. Dots only (no value/method labels).
    Pareto-optimal dots (best-so-far over trials — no earlier trial reached this value) are
    highlighted (larger + gold ring) and joined by a faint frontier step line."""
    rows = [(t, metric(t["csv"], t["cond"], col)) for t in trials]
    rows = [(t, s) for t, s in rows if s]            # only trials with a landed value
    if not rows:
        return "<p class='mut'>No values yet.</p>"
    maxtrial = max(t["trial"] for t in trials)
    vals = [s[0] for _, s in rows]
    for _lbl, rcsv, rcond, _col in REF_KEYS:
        rv = metric(rcsv, rcond, col)
        if rv:
            vals.append(rv[0])
    vmin, vmax = min(vals), max(vals)
    # nice padded y-range on a 0.05 grid
    import math
    y0 = math.floor((vmin - 0.03) * 20) / 20
    y1 = math.ceil((vmax + 0.03) * 20) / 20
    W, H = 880, 320
    padL, padR, padT, padB = 54, 20, 16, 40
    def Y(v): return padT + (1 - (v - y0) / (y1 - y0)) * (H - padT - padB)
    def X(tr): return padL + (0.5 if maxtrial == 1 else (tr - 1) / (maxtrial - 1)) * (W - padL - padR)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:900px" font-family="-apple-system,Segoe UI,sans-serif">']
    # y gridlines + labels
    g = y0
    while g <= y1 + 1e-9:
        yy = Y(g)
        parts.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eef1f5"/>')
        parts.append(f'<text x="{padL-7}" y="{yy+3:.1f}" font-size="10" fill="#5b6678" text-anchor="end">{g:.2f}</text>')
        g += 0.05
    # x-axis: trial ticks T1..maxtrial
    for tr in range(1, maxtrial + 1):
        xx = X(tr)
        parts.append(f'<text x="{xx:.1f}" y="{H-padB+18:.1f}" font-size="10" fill="#1c2433" text-anchor="middle">T{tr}</text>')
    parts.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-6}" font-size="10.5" fill="#5b6678" text-anchor="middle">trial</text>')
    # Pareto frontier: best-so-far over increasing trial
    srt = sorted(rows, key=lambda r: r[0]["trial"])
    pareto, best = set(), -1e9
    front = []
    for t, s in srt:
        if s[0] > best + 1e-9:
            best = s[0]; pareto.add(t["trial"]); front.append((X(t["trial"]), Y(s[0])))
    if len(front) > 1:
        d = f'M {front[0][0]:.1f} {front[0][1]:.1f} ' + " ".join(f'L {x:.1f} {y:.1f}' for x, y in front[1:])
        parts.append(f'<path d="{d}" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="2 3" opacity="0.7"/>')
    # dots — SAME colour for every method; Pareto-optimal dots get a gold ring + a T{n} label
    DOT = "#1d6fd0"
    for t, s in rows:
        x, y = X(t["trial"]), Y(s[0])
        tip = f'T{t["trial"]}: {esc(t["what"])} — {axis_label} {s[0]:.3f}'
        if t["trial"] in pareto:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{DOT}" stroke="#d97706" stroke-width="2.5"><title>{tip} (Pareto-optimal)</title></circle>')
            parts.append(f'<text x="{x+11:.1f}" y="{y+4:.1f}" font-size="12" font-weight="700" fill="#1c2433" text-anchor="start">T{t["trial"]}</text>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{DOT}" opacity="0.55"><title>{tip}</title></circle>')
    parts.append(f'<text x="14" y="{padT+(H-padT-padB)/2:.0f}" font-size="11" fill="#5b6678" transform="rotate(-90 14 {padT+(H-padT-padB)/2:.0f})" text-anchor="middle">{esc(axis_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build():
    trials = load_trials()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    base_sp = metric(*BASE_CSV, "test_spearman")
    base_pe = metric(*BASE_CSV, "test_pearson")
    base_sp_v = base_sp[0] if base_sp else 0.598
    base_pe_v = base_pe[0] if base_pe else 0.0

    def vd(val, base):  # value cell + delta vs base
        if not val:
            return '<span class="mut">—</span>', '<span class="mut">—</span>'
        m, sd, _ = val
        cell = f'<span class="num">{m:.3f}</span> <span class="pm">±{("%.3f"%sd).lstrip("0")}</span>'
        d = m - base
        dcls = "pos" if d > 0.0015 else ("neg" if d < -0.0015 else "mut")
        return cell, f'<span class="{dcls}">{d:+.3f}</span>'

    trows = ""
    for t in trials:
        sp = metric(t["csv"], t["cond"], "test_spearman")
        pe = metric(t["csv"], t["cond"], "test_pearson")
        if sp or pe:
            sp_c, sp_d = vd(sp, base_sp_v)
            pe_c, _ = vd(pe, base_pe_v)
        else:
            st = t["status"] or "pending"
            sp_c = f'<span class="mut">{esc(st)}</span>'; sp_d = '<span class="mut">—</span>'; pe_c = '<span class="mut">—</span>'
        arch = esc(t["architecture"]) or '<span class="mut">—</span>'
        inp = esc(t["input"]) or '<span class="mut">—</span>'
        det = esc(t["details"]) or '<span class="mut">—</span>'
        trows += (f'<tr><td class="num">{t["trial"]}</td>'
                  f'<td style="white-space:normal">{arch}</td>'
                  f'<td style="white-space:normal">{inp}</td>'
                  f'<td class="small" style="white-space:normal">{det}</td>'
                  f'<td>{sp_c}</td><td>{sp_d}</td><td>{pe_c}</td>'
                  f'<td class="small mut" style="white-space:normal">{esc(t["note"])}</td></tr>')

    chart_sp = svg_scatter(trials, "test_spearman", "test ρ (Spearman)")
    chart_pe = svg_scatter(trials, "test_pearson", "test r (Pearson)")
    legend = ('<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:11px;height:11px;border-radius:50%;background:#1d6fd0"></span>trial</span>'
              ' <span style="display:inline-flex;align-items:center;gap:5px"><span style="width:13px;height:13px;border-radius:50%;background:#fff;border:2.5px solid #d97706"></span>Pareto-optimal (best-so-far, labelled T#)</span>')

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OTF Trials — probe performance progression</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1c2433;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.page{{max-width:980px;margin:0 auto;padding:40px 24px 90px}}
header{{border-bottom:2px solid #1c2433;padding-bottom:14px;margin-bottom:24px}}
header .date{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#5b6678;margin-bottom:6px}}
header h1{{font-size:25px;font-weight:650;margin:0}}
header .sub{{font-size:13px;color:#5b6678;margin-top:6px}}
.card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:20px;margin-bottom:26px;box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}}
h2{{font-size:17px;font-weight:640;margin:0 0 14px}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:12px;color:#5b6678}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;text-align:left;vertical-align:top;border-top:1px solid #e3e7ee}}
thead th{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#5b6678;border:0;border-bottom:1px solid #e3e7ee}}
.num{{font-weight:660}}.pm{{color:#5b6678;font-weight:500;font-size:12px}}
.pos{{color:#1d5a3a;font-weight:700}}.neg{{color:#b4413a;font-weight:700}}.mut{{color:#5b6678}}.small{{font-size:12.5px}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.6}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER OTF campaign</div>
<h1>On-the-fly trials — probe performance progression</h1>
<div class="sub">Frozen-encoder affinity probe (ED+RSCC split, 5817/1498/2813, 3 seeds) by trial. Each dot is a trial;
<b>Pareto-optimal</b> dots (gold ring) are the best-so-far — no earlier trial reached that performance.
Static page, built {ts} from <code>experiment_status.csv</code> + the <code>probe_results_*.csv</code> files.
Δ in the table is vs the Channel-ViT base (Spearman {base_sp_v:.3f}).</div></header>

<div class="card"><h2>test ρ (Spearman) by trial</h2>
{chart_sp}</div>

<div class="card"><h2>test r (Pearson) by trial</h2>
{chart_pe}
<div class="legend">{legend}</div>
<div class="note" style="margin-top:12px">All dots are the <b>ED+RSCC split</b> (5817/1498/2813); the
<b>gold dashed step</b> traces the Pareto frontier (best-so-far) and gold-ringed dots are Pareto-optimal.
Dashed grid only — no horizontal reference lines.</div></div>

<div class="card"><h2>Trial table</h2>
<table><thead><tr><th>Trial</th><th>Architecture</th><th>Input</th><th>Details</th><th>test ρ</th><th>Δ ρ vs base</th><th>test r</th><th>note</th></tr></thead>
<tbody>{trows}</tbody></table></div>

<div class="card"><h2>Reading it</h2>
<div class="note">
<b>New ED+RSCC downstream</b> (LP ∩ ED ∩ lig&amp;pocket RSCC≥0.8 = 5817/1498/2813, ~3.4× the old 839
test set). All numbers here are on this split; the old-839 trial campaign is archived
(<code>experiment_status_839split.csv</code>) and in the 260625 meeting doc.<br><br>
<b>New-split baselines:</b> coords-only <b>0.572</b>, fused C+D+G <b>0.596</b>, Channel-ViT base <b>0.598</b>.
The Channel-ViT edge over fused shrank to ~+0.002 here (was +0.012 on 839) — little arch headroom, so the
autoresearch needs real gains.<br><br>
<b>Autoresearch:</b> base = Channel-ViT (0.598); each round changes one knob, probes on ED+RSCC, and joins
the plot — a new record extends the Pareto frontier. Open-ended until stopped.
</div></div>
</div></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({len(OUT.read_text()):,} bytes, static SVG, no JS)")
