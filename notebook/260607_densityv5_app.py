#!/usr/bin/env python
"""Live dashboard for the PDBbind frozen-encoder probe results.

A version-group selector (?view=) switches the page across density versions:
  v1 · v2 · v3 · v4   per-version probe summaries (probe_results_v{n}.csv)
  v5                  non-filtered v5 runs (probe_results_e99_v5_*.csv)
  v5_filtered         curated CL1-filtered density-v5 sweep (default)

Reads the probe-result CSVs and each run's exps/<run>/cfg.yaml on EVERY request,
so the chart and table always reflect the source files (no hard-coded numbers, no
chart-vs-table drift). Both Spearman ρ and Pearson r (val + test) are shown.

Run:
    python notebook/260607_densityv5_app.py --port 8731 --host 0.0.0.0
Then browse  http://<svr7-ip>:8731/   (LAN)  or tunnel and use localhost.
"""
import argparse, csv, re, statistics, html, datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

VOX = Path(__file__).resolve().parent.parent / "voxbind"
PDB = VOX / "dataset" / "data" / "pdbbind"

LIG = ["lig_C", "lig_O", "lig_N", "lig_S", "lig_F", "lig_Cl", "lig_P"]
POC = ["poc_C", "poc_O", "poc_N", "poc_S"]

# (csv column, short label, bar css class) for the four reported metrics.
COLS = [
    ("best_val_spearman", "val ρ",  "sv"),
    ("test_spearman",     "test ρ", "st"),
    ("val_pearson",       "val r",  "pv"),
    ("test_pearson",      "test r", "pt"),
]
# RMSE (pK units, lower = better) is reported in the tables only — not as a chart
# bar, since its ~1.4-1.8 scale is off the [0,1] correlation axis.
AGG_COLS = [c for c, _, _ in COLS] + ["test_rmse"]

# Run manifest: which CSV + condition holds each encoder's probe result, plus a
# one-line characteristic. Config (weights/channels/status) is read live from cfg.yaml.
RUNS = [
    dict(key="260605", date="2026-06-05",
         exp="260605_atomblob_density_vit_mae_40m_weighted_v5_gradmag_ligvdw_balanced_pretrain",
         csv="probe_results_e99_v5_filtered_260605_balanced.csv", cond="atomblob_density_gradmag",
         note="Sweep-“balanced” full reconstruction of density+gradmag (1.0/1.0). "
              "On the filtered rerun it trails invfreq/w1 by ~0.02 test rho, so treat it "
              "as weaker but not collapsed."),
    dict(key="density-only", date="2026-06-06",
         exp="260606_density_vit_mae_40m_xray_v5_gradmag_pretrain",
         csv="probe_results_e99_v5_filtered_260606density.csv", cond="density_gradmag",
         note="Pure density/gradient encoder — no atom blobs at all. Tests how much the "
              "density field alone carries for affinity."),
    dict(key="baseline", date="2026-06-08",
         exp="260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain",
         csv="probe_results_e99_v5_filtered_260606invfreq.csv", cond="atomblob_density_gradmag",
         note="Down-weighted density/gradmag recon (0.1/0.1) → atoms dominate; density is "
              "light context. Recovers full coords-only transfer. [the given config]"),
    dict(key="w1", date="2026-06-08",
         exp="260608_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_w1_pretrain",
         csv="probe_results_e99_v5_filtered_260608_w1.csv", cond="atomblob_density_gradmag",
         note="Full-weight density+gradmag recon (1.0/1.0) on the inv_freq recipe — the 1.0 end "
              "of the density-weight sweep; isolates weighting vs 260605."),
    dict(key="matched-ctrl", date="2026-06-09",
         exp="260609_atomblob_vit_mae_40m_invfreq_v5_ligvdw_pretrain",
         csv="probe_results_e99_v5_filtered_atomblob_ligvdw.csv", cond="atomblob_ligvdw",
         note="Matched atoms-only control: 11-ch ligvdw atoms (inv_freq), baseline recipe with "
              "density/gradmag dropped. test ρ 0.544 vs baseline+density 0.595 → density adds ~+0.05; "
              "element-wise ligvdw radii rule out blob-size leakage."),
    dict(key="rope3d", date="2026-06-10",
         exp="260609_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_rope3d_pretrain",
         csv="probe_results_e99_v5_filtered_rope3d.csv", cond="atomblob_density_gradmag",
         note="3D RoPE positional encoding (axial, rotate 60/64) on the baseline 13-ch recipe — "
              "RoPE vs learnable absolute PE. test ρ 0.606 vs baseline 0.595 (+0.011 on this matched "
              "839 split; +0.016 on the CL1 split), and RoPE drops a parameter. Current best."),
]


def load_metrics(csv_name, cond):
    p = PDB / csv_name
    if not p.exists():
        return None
    try:
        rows = [r for r in csv.DictReader(open(p)) if r.get("condition") == cond]
    except Exception:
        return None
    if not rows:
        return None
    out = {}
    for col in AGG_COLS:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, "")))
            except (TypeError, ValueError):
                pass
        if vals:
            out[col] = (sum(vals) / len(vals),
                        statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out or None


def load_cfg(exp):
    p = VOX / "exps" / exp / "cfg.yaml"
    if not p.exists():
        return {}
    t = p.read_text(errors="ignore")

    def g(key, cast=str, default=None):
        m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", t, re.M)
        if not m:
            return default
        v = m.group(1).strip().strip('"').strip("'")
        try:
            return cast(v)
        except Exception:
            return v
    return dict(
        input_mode=g("input_mode", default="density"),
        with_gradmag=str(g("with_gradmag", default="false")).lower() == "true",
        pos_encoding=g("pos_encoding", default="learnable"),
        n_in=g("n_in_channels", int),
        density_w=g("density_channel_weight", float),
        gradmag_w=g("gradmag_channel_weight", float),
        channel_weighting=g("channel_weighting"),
        clip=g("channel_weight_clip_ratio", float),
        atom_pos=g("atom_pos_weight", float),
        ligand_radius=g("ligand_radius", float),
        epochs=g("num_epochs", int) or 100,
    )


def channels(cfg):
    im = cfg.get("input_mode", "density") or "density"
    ch = []
    if im == "atomblob_merged_density":
        ch += ["atom_" + e for e in ["C", "O", "N", "S", "F", "Cl", "P"]]
    elif im.startswith("atomblob"):
        ch += LIG + POC
    if "density" in im:
        ch += ["density"]
    if cfg.get("with_gradmag"):
        ch += ["gradmag"]
    return ch


def run_status(exp, epochs):
    d = VOX / "exps" / exp
    if (d / "checkpoint_e0099.pth.tar").exists():
        return ("done", epochs)
    log = VOX / "log" / f"{exp}.log"
    if log.exists():
        txt = log.read_text(errors="ignore").replace("\r", "\n")
        eps = re.findall(r">> epoch:\s+(\d+)\s+\(", txt)
        if eps:
            return ("running", int(eps[-1]) + 1)
        if re.search(r"Traceback|CUDA out of memory|RuntimeError", txt):
            return ("error", None)
    if list(d.glob("checkpoint_*.pth.tar")):
        return ("running", None)
    return ("missing", None)


def esc(x):
    return html.escape(str(x))


CSS = """
:root{--bg:#f5f6f8;--card:#fff;--card2:#f0f2f5;--ink:#1f2430;--mut:#6b7280;--line:#e3e6ea;
--accent:#2563eb;--good:#15a34a;--bad:#dc2626;--warn:#b45309;--best:#ecfbf1;
--sv:#9cc0f7;--st:#2563eb;--pv:#8fd6c4;--pt:#0d9488}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:32px 20px 64px}
.wrap{max-width:1280px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.2px}
.sub{color:var(--mut);margin:0 0 20px;font-size:14px}
h2{font-size:17px;margin:30px 0 12px;border-left:3px solid var(--accent);padding-left:10px}
.ctx{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 18px;color:#3a4252;font-size:14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
code{background:#eef1f6;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12.5px;font-family:ui-monospace,Menlo,Consolas,monospace;color:#1d4ed8}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin-bottom:16px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.crow{margin:0 0 16px}.crow:last-child{margin-bottom:2px}
.rl{font-size:12.5px;font-weight:600;margin-bottom:4px;font-family:ui-monospace,Menlo,monospace;color:#2b3242}
.rl small{color:var(--mut);font-weight:400}
.track{position:relative;height:19px;background:#eef1f5;border-radius:5px;margin:3px 0;overflow:hidden}
.track .bar{position:absolute;left:0;top:0;height:100%;border-radius:5px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar.sv{background:var(--sv);color:#15356e}.bar.st{background:var(--st);color:#fff}
.bar.pv{background:var(--pv);color:#0b3d35}.bar.pt{background:var(--pt);color:#fff}
.gap{height:7px}
.pendrow{font-size:12px;color:var(--mut);font-family:ui-monospace,monospace;padding:6px 0}
.scale{font-size:11px;color:var(--mut);margin-top:10px;border-top:1px dashed var(--line);padding-top:7px}
table{width:100%;border-collapse:separate;border-spacing:0;margin-top:6px;font-size:13.5px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(16,24,40,.04)}
th,td{padding:10px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
thead th{background:var(--card2);color:#4b5260;font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.3px}
th.sp{background:#eaf1fd}th.pe{background:#e6f6f2}
tbody tr:last-child td{border-bottom:none}
tbody tr.best{background:var(--best)}tbody tr:hover{background:#f7f9fc}
.run{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#1d4ed8;word-break:break-word}
.tag{display:inline-block;font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:20px;white-space:nowrap}
.t-done{background:#e7f7ee;color:#15803d;border:1px solid #bbe7cb}
.t-run{background:#fef3e2;color:#b45309;border:1px solid #f6d9ad}
.t-best{background:#dcfce7;color:#15803d;border:1px solid #a7e9c0}
.t-err{background:#fde8e8;color:#b91c1c;border:1px solid #f3c0c0}
.num{font-variant-numeric:tabular-nums;font-weight:700;font-size:14.5px}
.sd{color:var(--mut);font-size:10px}
.pend{color:var(--mut);font-weight:600}
.mut{color:var(--mut)}
.chip{display:inline-block;background:#eef1f6;border:1px solid var(--line);border-radius:5px;padding:1px 6px;margin:1px 2px;font-size:11.5px;color:#3a4252}
ol.ch{margin:6px 0 0;padding-left:20px;font-size:11.5px;line-height:1.5;color:#39414f}
ol.ch li{margin:.5px 0;font-family:ui-monospace,Menlo,Consolas,monospace}
ol.ch li.poc{color:#5b6472}ol.ch li.aux{color:#1d4ed8}
footer{margin-top:28px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
.top{display:flex;justify-content:flex-end;margin:0 0 4px}
.viewsel{font-size:13px;color:var(--mut)}
.viewsel select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:5px 9px;margin-left:6px;cursor:pointer}
.vchart{display:flex;align-items:flex-start;gap:18px;overflow-x:auto;padding:24px 6px 2px;min-height:312px}
.vgroup{display:flex;flex-direction:column;align-items:center;flex:0 0 auto}
.vbars{display:flex;align-items:flex-end;gap:4px;height:260px}
.yax{margin-right:0}
.ylab{height:260px;display:flex;flex-direction:column;justify-content:space-between;align-items:flex-end;font-size:10px;color:var(--mut);font-variant-numeric:tabular-nums;border-right:1px dashed var(--line);padding-right:7px;min-width:30px}
.vbar{width:30px;border-radius:4px 4px 0 0;position:relative;min-height:2px}
.vbar>b{position:absolute;top:-15px;left:50%;transform:translateX(-50%);font-size:9px;font-weight:700;color:#4b5260;font-variant-numeric:tabular-nums}
.vbar.sv{background:var(--sv)}.vbar.st{background:var(--st)}.vbar.pv{background:var(--pv)}.vbar.pt{background:var(--pt)}
.vbar.empty{background:repeating-linear-gradient(45deg,#eef1f5,#eef1f5 4px,#e3e6ea 4px,#e3e6ea 8px)}
.vgroup.vpending{opacity:.5}
.vlabel{margin-top:9px;font-size:11px;font-family:ui-monospace,Menlo,Consolas,monospace;color:#2b3242;max-width:128px;text-align:center;word-break:break-word;line-height:1.35}
.vlabel small{color:var(--mut);font-weight:400}
.vlabel.best{color:var(--good);font-weight:700}
"""

LEGEND = ('<span><i style="background:var(--sv)"></i>val ρ (Spearman)</span>'
          '<span><i style="background:var(--st)"></i>test ρ</span>'
          '<span><i style="background:var(--pv)"></i>val r (Pearson)</span>'
          '<span><i style="background:var(--pt)"></i>test r</span>')

# ── version-group selector ───────────────────────────────────────────────────
VIEW_LABELS = ["v1", "v2", "v3", "v4", "v5", "v5_filtered"]
DEFAULT_VIEW = "v5_filtered"
SUMMARY_CSV = {v: f"probe_results_{v}.csv" for v in ["v1", "v2", "v3", "v4"]}
VER_BLURB = {
    "v1": "per-map z-score + per-crop ±3σ clip (bounded ≈±3).",
    "v2": "pocket-pool z-score (unbounded, cross-crystal comparable).",
    "v3": "pocket-pool max-abs → [−1,+1] (bounded + comparable); best density norm.",
    "v4": "dual-head encoder variant (regressed vs v3).",
    "v5": "non-filtered runs on the full 2704-sample EDS split.",
    "v5_filtered": "CL1-filtered split (n_test≈839); the curated density-v5 sweep.",
}


def list_nonfiltered_v5():
    """Non-filtered v5 per-run CSVs: probe_results_e99_v5_<tag>.csv (excludes _filtered)."""
    out = []
    for p in sorted(PDB.glob("probe_results_e99_v5_*.csv")):
        if "_v5_filtered_" in p.name:
            continue
        tag = p.name[len("probe_results_e99_v5_"):-len(".csv")]
        out.append((tag, p.name))
    return out


def load_table(csv_name):
    """Probe CSV → per (condition, n_train) 3-seed mean±std for the four metrics.

    Returns [{condition, n_train, metrics:{col:(mean,std,n)}, details}], best test ρ first.
    """
    p = PDB / csv_name
    if not p.exists():
        return []
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        return []
    groups = {}
    for r in rows:
        groups.setdefault((r.get("condition", "?"), r.get("n_train", "")), []).append(r)
    out = []
    for (cond, ntr), rs in groups.items():
        met = {}
        for col in AGG_COLS:
            vals = []
            for r in rs:
                try:
                    vals.append(float(r.get(col, "")))
                except (TypeError, ValueError):
                    pass
            if vals:
                met[col] = (sum(vals) / len(vals),
                            statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
        det = next((r.get("details", "") for r in rs if r.get("details")), "")
        try:
            ntr_i = int(ntr)
        except (TypeError, ValueError):
            ntr_i = None
        out.append(dict(condition=cond, n_train=ntr_i, metrics=met, details=det))

    def _ts(d):
        m = d["metrics"].get("test_spearman")
        return m[0] if m else -9.0
    out.sort(key=lambda d: (-_ts(d), d["condition"]))
    return out


def _bars(metrics, scale):
    bars = ""
    for i, (c, lab, cls) in enumerate(COLS):
        if i == 2:
            bars += '<div class="gap"></div>'
        st = metrics.get(c)
        if not st:
            continue
        wpct = f"{min(100, 100 * st[0] / scale):.1f}%"
        bars += f'<div class="track"><span class="bar {cls}" style="width:{wpct}">{lab} {st[0]:.3f}</span></div>'
    return bars


def _yaxis(vals):
    """Zoomed [ymin, ymax] for the bar chart — a non-zero baseline so differences pop."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 0.05
    ymin = max(0.0, vmin - 0.25 * span)
    ymax = min(1.0, vmax + 0.12 * span)
    return ymin, (ymax if ymax - ymin > 0.02 else ymin + 0.02)


def _yaxis_col(ymin, ymax):
    """Left y-axis showing the zoomed top (ymax) and bottom (ymin), aligned to the bars."""
    return (f'<div class="vgroup yax"><div class="ylab"><span>{ymax:.2f}</span>'
            f'<span>{ymin:.2f}</span></div><div class="vlabel">&nbsp;</div></div>')


def _vbar_group(metrics, ymin, ymax, label, sub="", best=False, pending=False):
    """One x-axis group of vertical bars (height ∝ value within [ymin,ymax]) + label."""
    rng = (ymax - ymin) or 1.0
    if pending:
        bars = '<div class="vbar empty" style="height:7px"></div>' * len(COLS)
    else:
        bars = ""
        for c, lab, cls in COLS:
            st = metrics.get(c)
            if not st:
                bars += f'<div class="vbar empty" style="height:3px" title="{lab}: n/a"></div>'
                continue
            h = max(2.0, min(100.0, 100.0 * (st[0] - ymin) / rng))
            bars += (f'<div class="vbar {cls}" style="height:{h:.1f}%" title="{lab} {st[0]:.3f}">'
                     f'<b>{st[0]:.3f}</b></div>')
    sublab = f'<br><small>{esc(sub)}</small>' if sub else ''
    return (f'<div class="vgroup{" vpending" if pending else ""}"><div class="vbars">{bars}</div>'
            f'<div class="vlabel{" best" if best else ""}">{esc(label)}{sublab}</div></div>')


def _rows_for(view):
    """List of (group_label, table_row) for a generic (non-curated) view."""
    if view in SUMMARY_CSV:                       # v1-v4: one summary CSV, all conditions
        return [("", d) for d in load_table(SUMMARY_CSV[view])]
    if view == "v5":                              # v5: 4 non-filtered runs, shared baselines once
        BASE = ("atomblob", "atomblob_weighted")
        base_rows, run_rows, seen = [], [], set()
        for tag, csv_name in list_nonfiltered_v5():
            for d in load_table(csv_name):
                if d["condition"] in BASE:
                    if d["condition"] not in seen:
                        seen.add(d["condition"])
                        base_rows.append(("baseline", d))
                else:
                    run_rows.append((tag, d))
        run_rows.sort(key=lambda t: -((t[1]["metrics"].get("test_spearman") or (-9.0,))[0]))
        base_rows.sort(key=lambda t: t[1]["condition"])
        return base_rows + run_rows
    return []


def _generic_body(view):
    rows = _rows_for(view)
    if not rows:
        exp = SUMMARY_CSV.get(view, "probe_results_e99_v5_*.csv")
        return f'<div class="ctx">No probe CSV found for this view yet (expected <code>{esc(exp)}</code>).</div>'
    allv = [st[0] for _, d in rows for c, _, _ in COLS for st in [d["metrics"].get(c)] if st]
    ymin, ymax = _yaxis(allv)
    has_group = any(lbl for lbl, _ in rows)
    has_details = any(d["details"] for _, d in rows)
    best_i = max((i for i, (_, d) in enumerate(rows) if "test_spearman" in d["metrics"]),
                 key=lambda i: rows[i][1]["metrics"]["test_spearman"][0], default=None)

    vgroups = []
    for i, (lbl, d) in enumerate(rows):
        sp = f"{d['n_train']}" if d["n_train"] and d["n_train"] != 2704 else ""
        name = f'{lbl}·{d["condition"]}' if lbl else d["condition"]
        vgroups.append(_vbar_group(d["metrics"], ymin, ymax, name, sub=sp, best=(i == best_i)))

    hdr = (('<th style="width:13%">Run</th>' if has_group else '')
           + '<th style="width:22%">Condition</th>'
           + ('<th>Notes</th>' if has_details else '')
           + '<th class="sp" style="width:9%">val ρ</th><th class="sp" style="width:9%">test ρ</th>'
             '<th class="pe" style="width:9%">val r</th><th class="pe" style="width:9%">test r</th>'
             '<th style="width:10%">test RMSE ↓</th>')
    trows = ""
    for i, (lbl, d) in enumerate(rows):
        cls = ' class="best"' if i == best_i else ''
        tds = (f'<td><span class="run">{esc(lbl)}</span></td>' if has_group else '')
        sp = f' <span class="mut">({d["n_train"]})</span>' if d["n_train"] and d["n_train"] != 2704 else ''
        tds += f'<td><b>{esc(d["condition"])}</b>{sp}</td>'
        if has_details:
            tds += f'<td><div class="mut" style="font-size:12px">{esc(d["details"])}</div></td>'
        for c, _, _ in COLS:
            st = d["metrics"].get(c)
            tds += ('<td><span class="pend">—</span></td>' if not st
                    else f'<td><span class="num">{st[0]:.3f}</span> <span class="sd">±{st[1]:.3f}</span></td>')
        rm = d["metrics"].get("test_rmse")
        tds += ('<td><span class="pend">—</span></td>' if not rm
                else f'<td><span class="num">{rm[0]:.3f}</span> <span class="sd">±{rm[1]:.3f}</span></td>')
        trows += f'<tr{cls}>{tds}</tr>'

    return (f'<h2>Val &amp; test correlation — Spearman ρ and Pearson r '
            f'<span class="mut" style="font-size:13px;font-weight:400">(3-seed mean)</span></h2>'
            f'<div class="chart"><div class="legend">{LEGEND}</div>'
            f'<div class="vchart">{_yaxis_col(ymin, ymax)}{"".join(vgroups)}</div>'
            f'<div class="scale">y-axis {ymin:.2f}–{ymax:.2f} (zoomed — bars start above 0 to emphasize differences). 3-seed mean.</div></div>'
            f'<h2>Conditions <span class="mut" style="font-size:13px;font-weight:400">— 3-seed mean ± std</span></h2>'
            f'<table><thead><tr>{hdr}</tr></thead><tbody>{trows}</tbody></table>')


def _curated_body():
    data = []
    for r in RUNS:
        cfg = load_cfg(r["exp"])
        data.append((r, load_metrics(r["csv"], r["cond"]), cfg,
                     run_status(r["exp"], cfg.get("epochs", 100))))

    allv = [m[c][0] for _, m, _, _ in data if m for c, _, _ in COLS if m and c in m]
    ymin, ymax = _yaxis(allv)
    done_tests = [(m["test_spearman"][0], r["key"]) for r, m, _, _ in data if m and "test_spearman" in m]
    best_key = max(done_tests)[1] if done_tests else None
    worst_done_test = min(t for t, _ in done_tests) if done_tests else None

    # ---- chart: four vertical bars per run ----
    vgroups = []
    for r, m, cfg, st in data:
        if m and all(c in m for c, _, _ in COLS):
            vgroups.append(_vbar_group(m, ymin, ymax, r["key"], best=(r["key"] == best_key)))
        else:
            lab = "running" if st[0] == "running" else st[0]
            vgroups.append(_vbar_group({}, ymin, ymax, r["key"], sub=f"({lab})", pending=True))

    # ---- table: config + four metric columns ----
    trows = []
    for r, m, cfg, st in data:
        cls = ' class="best"' if r["key"] == best_key else ""
        chli = ""
        for ch in channels(cfg):
            k = "poc" if ch.startswith("poc") else ("aux" if ch in ("density", "gradmag") else "")
            nm = "‖∇ρ‖ gradmag" if ch == "gradmag" else ch
            chli += f'<li class="{k}">{esc(nm)}</li>'
        nin = cfg.get("n_in") or len(channels(cfg)) or "?"
        chips = []
        if cfg.get("channel_weighting"):
            cw = cfg["channel_weighting"] + (f" · clip {cfg['clip']:g}×" if cfg.get("clip") else "")
            chips.append("channel_wt: " + cw)
        has_dens = "density" in (cfg.get("input_mode") or "")
        if has_dens and cfg.get("density_w") is not None:
            chips.append(f"density wt <b>{cfg['density_w']:g}</b>")
        if cfg.get("with_gradmag") and cfg.get("gradmag_w") is not None:
            chips.append(f"gradmag wt <b>{cfg['gradmag_w']:g}</b>")
        if not has_dens and not cfg.get("with_gradmag"):
            chips.append("atoms-only")
        if cfg.get("atom_pos") is not None:
            chips.append(f"atom_pos {cfg['atom_pos']:g}")
        if cfg.get("pos_encoding") and cfg.get("pos_encoding") != "learnable":
            chips.append(f"pos: <b>{cfg['pos_encoding']}</b>")
        chiphtml = "".join(f'<span class="chip">{c}</span>' for c in chips)

        if st[0] == "done":
            stag = '<span class="tag t-done">✓ done · e99</span>'
        elif st[0] == "running":
            stag = f'<span class="tag t-run">● running{(" · e"+str(st[1])+"/"+str(cfg.get("epochs",100))) if st[1] else ""}</span>'
        elif st[0] == "error":
            stag = '<span class="tag t-err">error</span>'
        else:
            stag = '<span class="tag">not started</span>'
        if r["key"] == best_key:
            stag = '<span class="tag t-best">★ best</span> ' + stag

        mcells = ""
        for c, _, _ in COLS:
            stat = m.get(c) if m else None
            if not stat:
                mcells += '<td><span class="pend">—</span></td>'
                continue
            style = ""
            if c == "test_spearman":
                if r["key"] == best_key:
                    style = ' style="color:var(--good)"'
                elif worst_done_test is not None and stat[0] <= worst_done_test + 1e-9:
                    style = ' style="color:var(--bad)"'
            mcells += f'<td><span class="num"{style}>{stat[0]:.3f}</span> <span class="sd">±{stat[1]:.3f}</span></td>'
        rm = m.get("test_rmse") if m else None
        mcells += ('<td><span class="pend">—</span></td>' if not rm
                   else f'<td><span class="num">{rm[0]:.3f}</span> <span class="sd">±{rm[1]:.3f}</span></td>')

        trows.append(
            f'<tr{cls}><td class="mut" style="white-space:nowrap">{esc(r["date"])}</td>'
            f'<td><span class="run">{esc(r["exp"].replace("_pretrain",""))}</span></td>'
            f'<td><b>{nin}-ch</b><ol class="ch">{chli}</ol></td>'
            f'<td>{chiphtml}<div class="mut" style="margin-top:6px">{esc(r["note"])}</div></td>'
            f'<td>{stag}</td>{mcells}</tr>')

    return (f'<h2>Val &amp; test correlation — Spearman ρ and Pearson r '
            f'<span class="mut" style="font-size:13px;font-weight:400">(3-seed mean)</span></h2>'
            f'<div class="chart"><div class="legend">{LEGEND}</div>'
            f'<div class="vchart">{_yaxis_col(ymin, ymax)}{"".join(vgroups)}</div>'
            f'<div class="scale">y-axis {ymin:.2f}–{ymax:.2f} (zoomed — bars start above 0). Source: '
            f'dataset/data/pdbbind/probe_results_e99_v5_filtered_*.csv</div></div>'
            f'<h2>Runs <span class="mut" style="font-size:13px;font-weight:400">— config (cfg.yaml) + metrics (RMSE in pK, lower=better)</span></h2>'
            f'<table><thead><tr><th style="width:8%">Date</th><th style="width:17%">Run name (experiment)</th><th style="width:11%">Input</th>'
            f'<th style="width:20%">Characteristic &amp; key config</th><th style="width:8%">Status</th>'
            f'<th class="sp" style="width:9%">val ρ</th><th class="sp" style="width:9%">test ρ</th>'
            f'<th class="pe" style="width:9%">val r</th><th class="pe" style="width:9%">test r</th>'
            f'<th style="width:10%">test RMSE ↓</th></tr></thead>'
            f'<tbody>{"".join(trows)}</tbody></table>')


CURATED_FOOTER = ("Common: DensityViT-MAE ~40M · patch 8 · 64³ · 100 ep · eff-batch 96 · AdamW lr 1e-4 · "
                  "mask atom_biased 0.6 · v5 density. ρ = Spearman (val = model-selection, test = held-out), "
                  "r = Pearson. Live sources: <code>exps/&lt;run&gt;/cfg.yaml</code>, "
                  "<code>dataset/data/pdbbind/probe_results_e99_v5_filtered_*.csv</code>, <code>log/&lt;run&gt;.log</code>. "
                  "Refresh (or wait 60s) to pick up new probe results.")
GENERIC_FOOTER = ("ρ = Spearman (val = model-selection, test = held-out), r = Pearson · 3-seed mean ± std. "
                  "Switch views with the selector above; refresh (or wait 60s) for new results.")


def _shell(view, h1, sub, intro, body, footer):
    opts = "".join(f'<option value="{v}"{" selected" if v == view else ""}>{v}</option>'
                   for v in VIEW_LABELS)
    selector = (f'<form method="get" class="viewsel">View '
                f'<select name="view" onchange="this.form.submit()">{opts}</select></form>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>PDBbind probe — {esc(view)}</title><style>{CSS}</style></head><body><div class="wrap">
<div class="top">{selector}</div>
<h1>{h1}</h1>
<p class="sub">{sub}</p>
<div class="ctx">{intro}</div>
{body}
<footer>{footer}</footer>
</div></body></html>"""


def render(view):
    if view not in VIEW_LABELS:
        view = DEFAULT_VIEW
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if view == "v5_filtered":
        h1 = "Density&nbsp;v5 Pre-training Runs &mdash; live"
        sub = ("VoxBind · MAE-ViT on CrossDocked X-ray density (v5) → PDBbind affinity probe · "
               f"values read live from CSV · loaded {now} · auto-refresh 60s")
        intro = ("<b>Task.</b> Pre-train a 40M ViT-MAE on CrossDocked X-ray density (self-supervised "
                 "masked reconstruction), then test whether the <b>frozen</b> encoder's features predict "
                 "PDBbind binding affinity (pK) with a 3-seed 2-layer-MLP probe. Each row is one encoder "
                 "variant — we compare input modalities (atoms / +density / +‖∇ρ‖ gradmag), reconstruction-"
                 "loss weightings, and positional encodings to find what transfers best to affinity.<br>"
                 "<b>Readout.</b> 512-D mean-pooled tokens → MLP → pK; <b>Spearman ρ</b> and <b>Pearson r</b> "
                 "(val = model-selection, test = held-out), 3-seed mean, read live from the probe CSVs. "
                 "All rows share one <b>matched eval split (n_test=839)</b> so the encoders are directly comparable.")
        return _shell(view, h1, sub, intro, _curated_body(), CURATED_FOOTER)
    h1 = f"PDBbind affinity probe &mdash; <code>{esc(view)}</code>"
    sub = f"VoxBind · frozen-encoder probe → pK · values read live from CSV · loaded {now} · auto-refresh 60s"
    src = (f"dataset/data/pdbbind/probe_results_{view}.csv" if view in SUMMARY_CSV
           else "dataset/data/pdbbind/probe_results_e99_v5_*.csv")
    intro = (f"<b>{esc(view)} density.</b> {esc(VER_BLURB.get(view, ''))} "
             f"3-seed mean ± std, read live from <code>{esc(src)}</code>.")
    return _shell(view, h1, sub, intro, _generic_body(view), GENERIC_FOOTER)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path not in ("/", "/index.html"):
            self.send_error(404); return
        view = parse_qs(u.query).get("view", [DEFAULT_VIEW])[0]
        try:
            body = render(view).encode("utf-8")
        except Exception as e:
            body = f"<pre>render error: {html.escape(repr(e))}</pre>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"serving density-v5 dashboard on http://{a.host}:{a.port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
