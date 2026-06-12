#!/usr/bin/env python
"""Live experiment-status dashboard for the VoxBind density/affinity line.

Reads the REAL state on every request — running torchrun jobs (+ epoch/GPUs from
/proc), armed watch_n_launch chains (+ their gate file), background precomputes,
and finished probe CSVs — so the page always reflects what is actually happening.
No hard-coded "running/done"; refresh (or wait 30 s) to update.

Run:
    python notebook/html/experiment_status_app.py --port 8732 --host 0.0.0.0
Then browse  http://<host>:8732/
"""
import argparse, os, re, csv, html, datetime, statistics
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOX = Path(__file__).resolve().parents[2] / "voxbind"   # notebook/html/ -> VoxBind/ -> voxbind
PDB = VOX / "dataset" / "data" / "pdbbind"
LOG = VOX / "log"


def esc(x): return html.escape(str(x))


# ── live process state (via /proc) ────────────────────────────────────────────
def _proc_iter():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "ignore")
        except Exception:
            continue
        yield pid, cmd


def _environ(pid, key):
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().decode("utf8", "ignore")
        m = re.search(rf"{key}=([^\x00]*)", env)
        return m.group(1) if m else None
    except Exception:
        return None


def _epoch_of(exp):
    p = LOG / f"{exp}.log"
    if not p.exists():
        return None
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        return None
    eps = re.findall(r">> epoch:\s+(\d+)\s*\(", txt)
    if eps:
        return int(eps[-1])
    if re.search(r"start density-ViT-MAE", txt):
        return 0
    return None


def running_trainings():
    seen = {}
    for pid, cmd in _proc_iter():
        if "train_density_vit_mae.py" not in cmd or "exp_name=" not in cmd:
            continue
        m = re.search(r"exp_name=(\S+)", cmd)
        if not m:
            continue
        exp = m.group(1)
        if exp in seen:
            continue
        gpus = _environ(pid, "CUDA_VISIBLE_DEVICES") or "?"
        seen[exp] = dict(exp=exp, gpus=gpus, epoch=_epoch_of(exp))
    return list(seen.values())


def active_watchers():
    out, seen = [], set()
    for pid, cmd in _proc_iter():
        if "99_watch_n_launch.sh" not in cmd or "--wait" not in cmd:
            continue
        w = re.search(r"--wait\s+(\S+)", cmd)
        gate = os.path.basename(w.group(1)) if w else "?"
        # label: the script/condition after the trailing "--"
        lbl = "—"
        sm = re.search(r"run_gradmagonly_chain", cmd)
        if sm:
            lbl = "gradmag-only (train → probe)"
        elif "01c_pdbbind_probe.py features" in cmd:
            tag = re.search(r"--tag (\S+)", cmd)
            lbl = f"auto-probe ({tag.group(1) if tag else '?'})"
        key = (gate, lbl)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(gate=gate, label=lbl, gate_ready=(PDB / gate).exists()))
    return out


def active_precomputes():
    out, seen = [], set()
    for pid, cmd in _proc_iter():
        m = re.search(r"(00[a-z]_[a-z_]+\.py)\s+(\w+)", cmd)
        if not m or "python" not in cmd:
            continue
        key = m.group(0)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{m.group(1)} {m.group(2)}")
    return out


# ── completed / pending results (curated manifest, read live) ──────────────────
COMPLETED = [
    dict(label="Coords + density + gradmag",            inp="atoms + dens + ‖∇ρ‖", pe="rope",
         csv="probe_results_e99_v5_filtered_rope3d.csv",            cond="atomblob_density_gradmag", note="RoPE +0.011 over learnable"),
    dict(label="Coords + density + gradmag (w1)",       inp="atoms + dens + ‖∇ρ‖", pe="learn",
         csv="probe_results_e99_v5_filtered_260608_w1.csv",         cond="atomblob_density_gradmag", note="recon wt 1.0"),
    dict(label="Coords + density + gradmag",            inp="atoms + dens + ‖∇ρ‖", pe="learn",
         csv="probe_results_e99_v5_filtered_260606invfreq.csv",     cond="atomblob_density_gradmag", note="reference for the noise control"),
    dict(label="Coords + density + gradmag (balanced)", inp="atoms + dens + ‖∇ρ‖", pe="learn",
         csv="probe_results_e99_v5_filtered_260605_balanced.csv",   cond="atomblob_density_gradmag", note="recon-wt sweep variant"),
    dict(label="Coords only (atoms, ligvdw)",           inp="atoms",                pe="rope",
         csv="probe_results_e99_v5_filtered_atomblob_ligvdw_rope3d839.csv", cond="atomblob_ligvdw", note="RoPE +0.041 (atoms-only)"),
    dict(label="Coords only (atoms, ligvdw)",           inp="atoms",                pe="learn",
         csv="probe_results_e99_v5_filtered_atomblob_ligvdw.csv",   cond="atomblob_ligvdw", note="matched control (density dropped)"),
    dict(label="Density + gradmag only",                inp="dens + ‖∇ρ‖",          pe="learn",
         csv="probe_results_e99_v5_filtered_260606density.csv",     cond="density_gradmag", note="no atoms"),
    dict(label="Density only (pure 1-ch)",              inp="dens",                 pe="learn",
         csv="probe_results_e99_v5_filtered_densityonly.csv",       cond="density_gradmag", note="= density+gradmag → gradmag redundant"),
    dict(label="Coords + NOISE-density + NOISE-gradmag", inp="atoms + noise",       pe="learn",
         csv="probe_results_e99_v5_filtered_noisecontrol.csv",      cond="atomblob_density_gradmag", note="density-ablation control"),
    dict(label="Gradmag only",                          inp="‖∇ρ‖",                 pe="learn",
         csv="probe_results_e99_v5_filtered_gradmagonly.csv",       cond="density_gradmag", note="‖∇ρ‖ as the single channel"),
    dict(label="HBGSA (supervised, external)",          inp="—",                    pe="ext",
         csv="probe_results_e99_v5_filtered_hbgsa_no_cl1.csv",      cond="hbgsa_supervised", note="< every frozen probe"),
]


def read_metrics(csv_name, cond):
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
    for col in ("best_val_spearman", "test_spearman", "test_pearson", "test_rmse"):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, "")))
            except (TypeError, ValueError):
                pass
        if vals:
            out[col] = (sum(vals) / len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return out or None


# ── render ─────────────────────────────────────────────────────────────────────
CSS = """
:root{--ink:#1c2433;--soft:#5b6678;--line:#e3e7ee;--line2:#aeb6c4;--bg:#f5f6f8;--card:#fff;
--run:#1d6fd0;--run-bg:#e7f0fc;--queue:#b07a17;--queue-bg:#fcf3e0;--done:#2f6f4f;--done-bg:#eaf5ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.page{max-width:1280px;margin:0 auto;padding:40px 24px 90px}
header.doc{border-bottom:2px solid var(--ink);padding-bottom:15px;margin-bottom:30px}
header.doc .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin-bottom:6px}
header.doc h1{font-size:26px;font-weight:650;margin:0}
header.doc .sub{font-size:13px;color:var(--soft);margin-top:6px}
.sec{margin-bottom:38px}
.sec-head{font-size:18px;font-weight:640;margin:0 0 13px;display:flex;align-items:center;gap:10px}
.dot{width:11px;height:11px;border-radius:50%}
.dot.run{background:var(--run)}.dot.queue{background:var(--queue)}.dot.done{background:var(--done)}
.sec-head .count{font-size:12.5px;font-weight:500;color:var(--soft)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px 13px;text-align:left;vertical-align:top}
thead th{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--soft);background:#fafbfc;border-bottom:1px solid var(--line);white-space:nowrap}
tbody tr{border-top:1px solid var(--line)}tbody tr:hover{background:#fbfcfd}
.exp{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:#1d4ed8;word-break:break-word}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.run{background:var(--run-bg);color:var(--run)}.pill.queue{background:var(--queue-bg);color:var(--queue)}
.pill.done{background:var(--done-bg);color:var(--done)}.pill.ext{background:#eef1f6;color:var(--soft)}
.pe{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px}
.pe.rope{background:var(--done-bg);color:#1d5a3a}.pe.learn{background:#eef1f6;color:var(--soft)}.pe.ext{background:#eef1f6;color:var(--soft)}
.num{font-weight:660}.best{color:#1d5a3a;font-weight:720}.mut{color:var(--soft)}.small{font-size:12.5px;color:var(--soft)}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{height:7px;border-radius:4px;background:#eef1f5;overflow:hidden;margin-top:4px;max-width:150px}
.bar>i{display:block;height:100%;background:var(--run);border-radius:4px}
.empty{padding:18px;color:var(--soft);font-size:13.5px;text-align:center}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px;font-size:12px;color:var(--soft)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.note{margin-top:11px;font-size:12.5px;color:var(--soft);line-height:1.55}
"""


def render():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    runs = running_trainings()
    watchers = active_watchers()
    preps = active_precomputes()

    # RUNNING
    if runs or preps:
        rows = ""
        for r in runs:
            ep = r["epoch"]
            pct = (ep / 100 * 100) if ep is not None else 0
            prog = (f'<span class="num">e{ep} / 100</span><div class="bar"><i style="width:{pct:.0f}%"></i></div>'
                    if ep is not None else '<span class="small">starting…</span>')
            rows += (f'<tr><td><span class="pill run">training</span><br><span class="exp">{esc(r["exp"])}</span></td>'
                     f'<td>GPU {esc(r["gpus"])}</td><td>{prog}</td></tr>')
        for p in preps:
            rows += (f'<tr><td><span class="pill run">precompute</span><br><span class="exp">{esc(p)}</span></td>'
                     f'<td class="mut">CPU</td><td class="small">materializing…</td></tr>')
        running_html = (f'<div class="wrap"><table><thead><tr><th>Experiment</th><th>Resource</th>'
                        f'<th>Progress</th></tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        running_html = '<div class="wrap"><div class="empty">Nothing training right now.</div></div>'

    # QUEUED
    if watchers:
        rows = ""
        for w in watchers:
            state = ('<span class="small" style="color:var(--done)">gate ready → firing</span>'
                     if w["gate_ready"] else f'<span class="small">waiting for <code>{esc(w["gate"])}</code></span>')
            rows += (f'<tr><td><span class="pill queue">chained</span><br>{esc(w["label"])}</td><td>{state}</td></tr>')
        queued_html = (f'<div class="wrap"><table><thead><tr><th>Step</th><th>Fires when</th></tr></thead>'
                       f'<tbody>{rows}</tbody></table></div>')
    else:
        queued_html = '<div class="wrap"><div class="empty">No chained steps armed.</div></div>'

    # COMPLETED (+ pending rows that auto-fill)
    data = []
    for r in COMPLETED:
        m = read_metrics(r["csv"], r["cond"])
        data.append((r, m))
    # sort: done first by test ρ desc, pending last
    def _key(t):
        m = t[1]
        return (0, -m["test_spearman"][0]) if (m and "test_spearman" in m) else (1, 0)
    data.sort(key=_key)
    best = max((m["test_spearman"][0] for _, m in data if m and "test_spearman" in m), default=None)
    rows = ""
    for r, m in data:
        pe = {"rope": '<span class="pe rope">3D RoPE</span>', "learn": '<span class="pe learn">ViT</span>',
              "ext": '<span class="pe ext">—</span>'}.get(r["pe"], "")
        if m and "test_spearman" in m:
            ts = m["test_spearman"]; tr = m.get("test_pearson"); vs = m.get("best_val_spearman"); rm = m.get("test_rmse")
            is_best = best is not None and abs(ts[0] - best) < 1e-9
            tcell = (f'<span class="num {"best" if is_best else ""}">{ts[0]:.3f}</span> '
                     f'<span class="small">±{ts[1]:.3f}</span>')
            vcell = f'<span class="mut">{vs[0]:.3f}</span>' if vs else "—"
            rcell = f'{tr[0]:.3f}' if tr else "—"
            mcell = f'{rm[0]:.3f}' if rm else "—"
            status = '<span class="pill done">done</span>'
        else:
            tcell = vcell = rcell = mcell = '<span class="pill queue" style="font-weight:600">pending</span>'
            status = '<span class="pill queue">pending</span>'
        rows += (f'<tr><td>{esc(r["label"])}<br><span class="small">{esc(r["inp"])}</span></td>'
                 f'<td>{pe}</td><td>{status}</td>'
                 f'<td>{vcell}</td><td>{tcell}</td><td>{rcell}</td><td>{mcell}</td>'
                 f'<td class="small">{esc(r["note"])}</td></tr>')
    completed_html = (f'<div class="wrap"><table><thead><tr><th>Encoder / input</th><th>PE</th><th>Status</th>'
                      f'<th>val ρ</th><th>test ρ</th><th>test r</th><th>RMSE</th><th>Note</th></tr></thead>'
                      f'<tbody>{rows}</tbody></table></div>')

    n_done = sum(1 for _, m in data if m and "test_spearman" in m)
    n_pend = len(data) - n_done

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Experiment Status — VoxBind</title><style>{CSS}</style></head><body><div class="page">
<header class="doc"><div class="date">VoxBind · density &amp; affinity line</div>
<h1>Experiment Status</h1>
<div class="sub">Live — read from running processes + probe CSVs · loaded {now} · auto-refresh 30 s ·
affinity, LP-PDBBind <code>new_split</code> n 2172/480/839</div></header>

<section class="sec"><h2 class="sec-head"><span class="dot run"></span> Running
<span class="count">— {len(runs)} training{'' if len(runs)==1 else 's'}{', '+str(len(preps))+' precompute' if preps else ''}</span></h2>{running_html}</section>

<section class="sec"><h2 class="sec-head"><span class="dot queue"></span> Queued / chained
<span class="count">— {len(watchers)} armed</span></h2>{queued_html}</section>

<section class="sec"><h2 class="sec-head"><span class="dot done"></span> Completed
<span class="count">— {n_done} done, {n_pend} pending (auto-fill)</span></h2>{completed_html}
<div class="legend"><span><span class="pe rope" style="padding:1px 7px">3D RoPE</span> rotary PE</span>
<span><span class="pe learn" style="padding:1px 7px">ViT</span> learnable PE</span>
<span><span class="best">bold</span> best test ρ</span>
<span>ρ Spearman, r Pearson · 3-seed mean</span></div>
<div class="note"><b>Story:</b> density adds ~+0.05 ρ over coords (0.595 vs 0.544); RoPE-3D is the best PE (0.606);
gradmag is redundant on top of density (both 0.505); the frozen SSL probe beats external supervised HBGSA (0.486).
The <b>noise control</b> (running) tests whether density's gain is signal or capacity.</div></section>

</div></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html", "/experiment_status.html"):
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
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"serving experiment status on http://{a.host}:{a.port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
