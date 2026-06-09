#!/usr/bin/env python
"""Live dashboard for the density-v5 MAE pre-training probe results.

Reads the probe-result CSVs (dataset/data/pdbbind/probe_results_e99_v5_*.csv) and
each run's exps/<run>/cfg.yaml on EVERY request, so the chart and table always
reflect the source files (no hard-coded numbers, no chart-vs-table drift). Both
Spearman ρ and Pearson r (val + test) are shown together. New runs appear once
their CSV lands.

Run:
    python notebook/260607_densityv5_app.py --port 8731 --host 0.0.0.0
Then browse  http://<svr7-ip>:8731/   (LAN)  or tunnel and use localhost.
"""
import argparse, csv, re, statistics, html, datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

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

# Run manifest: which CSV + condition holds each encoder's probe result, plus a
# one-line characteristic. Config (weights/channels/status) is read live from cfg.yaml.
RUNS = [
    dict(key="260605", date="2026-06-05",
         exp="260605_atomblob_density_vit_mae_40m_weighted_v5_gradmag_ligvdw_balanced_pretrain",
         csv="probe_results_e99_v5_densityonly.csv", cond="atomblob_density_gradmag",
         note="Sweep-“balanced” full reconstruction of density+gradmag (1.0/1.0). "
              "Density objective dominates the loss → corrupts the transferable atom features."),
    dict(key="density-only", date="2026-06-06",
         exp="260606_density_vit_mae_40m_xray_v5_gradmag_pretrain",
         csv="probe_results_e99_v5_densityonly.csv", cond="density_gradmag",
         note="Pure density/gradient encoder — no atom blobs at all. Tests how much the "
              "density field alone carries for affinity."),
    dict(key="baseline", date="2026-06-08",
         exp="260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain",
         csv="probe_results_e99_v5_baseline_invfreq.csv", cond="atomblob_density_gradmag",
         note="Down-weighted density/gradmag recon (0.1/0.1) → atoms dominate; density is "
              "light context. Recovers full coords-only transfer. [the given config]"),
    dict(key="w1", date="2026-06-08",
         exp="260608_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_w1_pretrain",
         csv="probe_results_e99_v5_w1.csv", cond="atomblob_density_gradmag",
         note="Full-weight density+gradmag recon (1.0/1.0) on the inv_freq recipe — the 1.0 end "
              "of the density-weight sweep; isolates weighting vs 260605."),
    dict(key="matched-ctrl", date="2026-06-09",
         exp="260609_atomblob_vit_mae_40m_invfreq_v5_ligvdw_pretrain",
         csv="probe_results_e99_v5_ligvdw_atomsonly.csv", cond="atomblob_ligvdw",
         note="MATCHED CONTROL: 11-ch atoms-only (ligvdw radii, inv_freq) — the baseline recipe "
              "with density/gradmag dropped. Isolates density's net contribution vs baseline 0.473."),
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
    for col, _, _ in COLS:
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
"""


def render():
    data = []
    for r in RUNS:
        cfg = load_cfg(r["exp"])
        data.append((r, load_metrics(r["csv"], r["cond"]), cfg,
                     run_status(r["exp"], cfg.get("epochs", 100))))

    allv = [m[c][0] for _, m, _, _ in data if m for c, _, _ in COLS if m and c in m]
    scale = max(allv) * 1.08 if allv else 0.6
    done_tests = [(m["test_spearman"][0], r["key"]) for r, m, _, _ in data if m and "test_spearman" in m]
    best_key = max(done_tests)[1] if done_tests else None
    worst_done_test = min(t for t, _ in done_tests) if done_tests else None

    def w(x):
        return f"{min(100, 100 * x / scale):.1f}%"

    # ---- chart: four bars per run (val ρ, test ρ | val r, test r) ----
    crows = []
    for r, m, cfg, st in data:
        dw, gw = cfg.get("density_w"), cfg.get("gradmag_w")
        has_dens = "density" in (cfg.get("input_mode") or "")
        cw = cfg.get("channel_weighting", "?")
        if has_dens and dw is not None:
            sub = f"density/gradmag {dw:g}/{gw:g} · {cw}"
        else:
            sub = f"{cfg.get('input_mode','?')} · {cw}"
        star = " — best test" if r["key"] == best_key else ""
        if m and all(c in m for c, _, _ in COLS):
            bars = ""
            for i, (c, lab, cls) in enumerate(COLS):
                if i == 2:
                    bars += '<div class="gap"></div>'
                v = m[c][0]
                bars += f'<div class="track"><span class="bar {cls}" style="width:{w(v)}">{lab} {v:.3f}</span></div>'
            crows.append(f'<div class="crow"><div class="rl">{esc(r["key"])} <small>{esc(sub)}{star}</small></div>{bars}</div>')
        else:
            lab = "running" if st[0] == "running" else st[0]
            crows.append(f'<div class="pendrow">{esc(r["key"])} <small>{esc(sub)}</small> &mdash; {lab} (no probe yet)</div>')

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

        trows.append(
            f'<tr{cls}><td><span class="run">{esc(r["exp"].replace("_pretrain",""))}</span>'
            f'<div class="mut" style="margin-top:4px">{esc(r["date"])}</div></td>'
            f'<td><b>{nin}-ch</b><ol class="ch">{chli}</ol></td>'
            f'<td>{chiphtml}<div class="mut" style="margin-top:6px">{esc(r["note"])}</div></td>'
            f'<td>{stag}</td>{mcells}</tr>')

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    leg = ('<span><i style="background:var(--sv)"></i>val ρ (Spearman)</span>'
           '<span><i style="background:var(--st)"></i>test ρ</span>'
           '<span><i style="background:var(--pv)"></i>val r (Pearson)</span>'
           '<span><i style="background:var(--pt)"></i>test r</span>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Density v5 Pre-training — live</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Density&nbsp;v5 Pre-training Runs &mdash; live</h1>
<p class="sub">VoxBind · MAE-ViT on CrossDocked X-ray density (v5) → PDBbind affinity probe ·
values read live from CSV · loaded {now} · auto-refresh 60s</p>
<div class="ctx"><b>What this is.</b> Frozen-encoder PDBbind affinity probe (512-D mean-pooled tokens → 2-layer MLP → pK),
3-seed mean over 4,827 refined complexes. Chart and table both report <b>Spearman ρ</b> and <b>Pearson r</b>
(val + test), pulled live from the probe CSVs. New runs appear when their CSV lands.</div>
<h2>Val &amp; test correlation — Spearman ρ and Pearson r <span class="mut" style="font-size:13px;font-weight:400">(3-seed mean)</span></h2>
<div class="chart"><div class="legend">{leg}</div>
{''.join(crows)}
<div class="scale">Bars scaled to {scale:.2f}. Source: dataset/data/pdbbind/probe_results_e99_v5_*.csv</div></div>
<h2>Runs <span class="mut" style="font-size:13px;font-weight:400">— config (cfg.yaml) + all four metrics</span></h2>
<table><thead><tr><th style="width:18%">Run / date</th><th style="width:12%">Input</th>
<th style="width:24%">Characteristic &amp; key config</th><th style="width:10%">Status</th>
<th class="sp" style="width:9%">val ρ</th><th class="sp" style="width:9%">test ρ</th>
<th class="pe" style="width:9%">val r</th><th class="pe" style="width:9%">test r</th></tr></thead>
<tbody>{''.join(trows)}</tbody></table>
<footer>Common: DensityViT-MAE ~40M · patch 8 · 64³ · 100 ep · eff-batch 96 · AdamW lr 1e-4 · mask atom_biased 0.6 · v5 density.
ρ = Spearman (val = model-selection, test = held-out), r = Pearson. Live sources:
<code>exps/&lt;run&gt;/cfg.yaml</code>, <code>dataset/data/pdbbind/probe_results_e99_v5_*.csv</code>, <code>log/&lt;run&gt;.log</code>.
Refresh (or wait 60s) to pick up new probe results.</footer>
</div></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path not in ("/", "/index.html"):
            self.send_error(404); return
        try:
            body = render().encode("utf-8")
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
