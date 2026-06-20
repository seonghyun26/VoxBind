#!/usr/bin/env python
"""Generate experiment_queue.html as a FULLY STATIC page (no JavaScript, no server) from:
  - experiment_status.csv   -> Running / Queued tables (the ledger I keep updated by hand)
  - probe_results_*.csv      -> Done values (test/val Spearman + RMSE, mean +/- std over seeds)

Python reads the CSVs and writes every row and number inline, so the file renders in any
HTML viewer — IDE preview pane, browser, etc. — WITHOUT executing JavaScript.

Run:  python notebook/html/build_queue_html.py     (re-run after editing experiment_status.csv
or when new probe CSVs land).
"""
import csv as csvmod, statistics, html as htmlmod, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_queue.html"
STATUS = HERE / "experiment_status.csv"
PDB = HERE.parents[1] / "voxbind" / "dataset" / "data" / "pdbbind"


def esc(x):
    return htmlmod.escape(str(x))


# ── Done manifest: curated ablation table. (label, csv, cond, desc, tag, best). desc/label are
#    trusted HTML (I author them) and inserted raw; only CSV-derived numbers are dynamic. ──
G = lambda title: {"grp": title}
DONE = [
    G("Affinity — headline (PLINDER, 40M)"),
    dict(label="C + D + G — full model", tag="invfreq", best=True, cond="atomblob_density_gradmag",
         csv="probe_results_e99_plinder_cdg.csv",
         desc="Coords + X-ray density + gradient field, MAE-pretrained on the 17.5k PLINDER ligand-matched corpus, frozen &rarr; affinity. <b>Best encoder on record.</b>"),
    dict(label="Coords only", cond="atomblob_ligvdw", csv="probe_results_e99_plinder_coords.csv",
         desc="Same PLINDER corpus, atoms only (no density). The capacity baseline for C+D+G."),
    dict(label="C + D + G", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_uniform.csv",
         desc="PLINDER C+D+G with fully-uniform MAE loss. On PLINDER invfreq beats uniform."),
    dict(label="Coords only", tag="uniform", cond="atomblob_ligvdw", csv="probe_results_e99_v5_plinder_coords_uniform.csv",
         desc="PLINDER coords-only, uniform loss (n=922 pool)."),
    dict(label="Noise control", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_noisecontrol.csv",
         desc="Density replaced by matched noise. Below real and coords &rarr; on PLINDER density is <b>content, not capacity</b>."),

    G("Affinity — PLINDER single-channel (frozen)"),
    dict(label="Density only", cond="density_gradmag", csv="probe_results_e99_v5_plinder_densityonly.csv",
         desc="Only the X-ray density field, no atoms (1-channel)."),
    dict(label="Gradmag only", cond="density_gradmag", csv="probe_results_e99_v5_plinder_gradmagonly.csv",
         desc="Only the gradient magnitude (edge/surface map), no density values, no atoms."),
    dict(label="Density only (839)", cond="density_gradmag", csv="probe_results_e99_v5_plinder_donly839.csv",
         desc="Density-only re-probed restricted to the canonical 839 split."),

    G("Affinity — PLINDER on-the-fly resample aug"),
    dict(label="OTF C+D+G — 100% (full)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p100.csv",
         desc="C+D+G with density re-cropped from the full 2Fo-Fc map at the augmented pose (no zero-fill). Full corpus."),
    dict(label="OTF C+D+G — seed 0", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_seed0.csv",
         desc="Variance re-draw of the OTF full run."),
    dict(label="OTF C+D+G — 0.4M", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_0p4m.csv",
         desc="OTF C+D+G at 0.4M params (param-scaling point)."),
    dict(label="OTF C+D+G — 4M", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_4m.csv",
         desc="OTF C+D+G at 4M params (param-scaling point)."),
    dict(label="OTF size sweep — 50%", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p50.csv",
         desc="OTF C+D+G on a random 50% of the pool."),
    dict(label="OTF size sweep — 25%", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p25.csv",
         desc="OTF C+D+G on 25% of the pool."),
    dict(label="OTF size sweep — 10%", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p10_clean.csv",
         desc="OTF C+D+G on 10% of the pool."),
    dict(label="OTF size sweep — 5%", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p05.csv",
         desc="OTF C+D+G on 5% of the pool."),
    dict(label="OTF size sweep — 1%", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_p01.csv",
         desc="OTF C+D+G on 1% of the pool."),
    dict(label="OTF density-only (1ch)", cond="density_gradmag", csv="probe_results_e99_v5_plinder_densityonly_otf.csv",
         desc="1-channel density encoder under OTF resample (no atoms, no gradmag)."),
    dict(label="OTF gradmag-only (1ch)", cond="density_gradmag", csv="probe_results_e99_v5_plinder_gradmagonly_otf.csv",
         desc="1-channel gradient-magnitude encoder under OTF resample."),
    dict(label="OTF D+G-only (2ch)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_dg_otf.csv",
         desc="2-channel density+gradmag (no coords) under OTF resample."),

    G("Affinity — architecture ablation"),
    dict(label="Channel-ViT (frozen C+D+G)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_channelvit.csv",
         desc="Grouped patch-embed (per-channel-group tokens [7,4,1,1] + cross-group attention)."),
    dict(label="ChA-MAEViT (frozen C+D+G)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_chamae.csv",
         desc="Token-drop channel-aware MAE (DCP masking + memory tokens + Fourier loss)."),
    dict(label="Channel-ViT (OTF C+D+G)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_channelvit.csv",
         desc="Channel-ViT on the OTF C+D+G config (arch ablation, new)."),
    dict(label="ChA-MAEViT (OTF C+D+G)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_plinder_otf_chamae.csv",
         desc="ChA-MAEViT on the OTF C+D+G config (arch ablation, new)."),
    dict(label="ChA-MAEViT (v5 corpus)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_cha_mae.csv",
         desc="ChA-MAEViT on the CrossDocked v5 corpus."),
    dict(label="ChA-MAEViT (v6)", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_cha_mae_v6.csv",
         desc="ChA-MAEViT on the tiny v6 set &mdash; <b>collapsed</b> (data-hungry; near-chance)."),

    G("Affinity — CrossDocked v5 (older corpus + controls)"),
    dict(label="C + D + G (RoPE-3D)", tag="invfreq", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_filtered_rope3d.csv",
         desc="Full 13-ch encoder, 3D-RoPE PE. The CrossDocked headline before PLINDER."),
    dict(label="Coords only", cond="atomblob_ligvdw", csv="probe_results_e99_v5_filtered_atomblob_ligvdw.csv",
         desc="Atoms only (matched control for &ldquo;does density help?&rdquo;)."),
    dict(label="Density only", cond="density_gradmag", csv="probe_results_e99_v5_filtered_densityonly.csv",
         desc="No atom channels &mdash; density field alone."),
    dict(label="Gradmag only", cond="density_gradmag", csv="probe_results_e99_v5_filtered_gradmagonly.csv",
         desc="Strongest single channel on CrossDocked."),
    dict(label="Noise control", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_filtered_noisecontrol.csv",
         desc="Density+gradmag &rarr; matched noise. Matches real &rarr; capacity, not signal (CrossDocked)."),
    dict(label="Zero control", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_filtered_zerocontrol.csv",
         desc="Density+gradmag zero-filled. Still matches real &rarr; pure patch-embed capacity."),
    dict(label="Coords only", tag="uniform", cond="atomblob_ligvdw", csv="probe_results_e99_v5_uniform_coords.csv",
         desc="Coords-only, uniform loss &mdash; the +0.05 gap was an invfreq artifact."),
    dict(label="C + D + G", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_uniform_dg.csv",
         desc="Full 13-ch, uniform loss. &le; uniform coords-only &rarr; density adds nothing under matched loss."),
    dict(label="Noise control", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_uniform_noise.csv",
         desc="Matched-noise density+gradmag, uniform loss. Lands with coords/real."),
    dict(label="OTF resample", tag="invfreq", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_resample.csv",
         desc="v5 C+D+G, density cropped at the augmented pose. Beats frozen zero-fill crops."),
    dict(label="OTF resample", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_resample_uniform.csv",
         desc="Resample-aug under uniform loss."),

    G("Affinity — matched-density corpora (CrossDocked)"),
    dict(label="v6 ligand-matched", tag="invfreq", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_v6_ligandmatched.csv",
         desc="C+D+G on the 5,270 tt_min subset (density matches pocket+ligand). Inside the capacity cluster."),
    dict(label="v6 ligand-matched", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_v6_uniform.csv",
         desc="v6 under uniform loss &mdash; barely above coords."),
    dict(label="v7 combined", tag="invfreq", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_v7_combined.csv",
         desc="v6 &cup; PDBbind-2020-train matched (7,873). More matched density moves affinity not at all."),
    dict(label="v7 combined", tag="uniform", cond="atomblob_density_gradmag", csv="probe_results_e99_v5_v7_combined_uniform.csv",
         desc="v7 under uniform loss."),

    G("Baselines"),
    dict(label="HBGSA (supervised)", cond="hbgsa_supervised", csv="probe_results_e99_v5_filtered_hbgsa_no_cl1.csv",
         desc="External fully-supervised affinity model, paper-faithful 3.06M params. Loses to the frozen SSL probe."),
]


def stats(csv_name, cond):
    p = PDB / csv_name
    if not csv_name or not p.exists():
        return None
    try:
        rows = [r for r in csvmod.DictReader(open(p)) if r.get("condition") == cond]
    except Exception:
        return None
    if not rows:
        return None
    out = {"n": rows[0].get("n_test", "?"), "k": len(rows)}
    for key, col in (("ts", "test_spearman"), ("bv", "best_val_spearman"), ("rm", "test_rmse")):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (KeyError, TypeError, ValueError):
                pass
        out[key] = (sum(vals) / len(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0) if vals else None
    return out


def cell(pair):
    if not pair:
        return '<span class="mut">—</span>'
    m, s = pair
    pm = f' <span class="pm">±{("%.3f" % s).lstrip("0")}</span>' if s else ""
    return f'<span class="num">{m:.3f}</span>{pm}'


def load_status():
    run, q = [], []
    if not STATUS.exists():
        return run, q
    for r in csvmod.DictReader(open(STATUS, encoding="utf-8")):
        sec = (r.get("section") or "").strip().lower()
        try:
            order = int(r.get("order") or 0)
        except ValueError:
            order = 0
        rec = dict(order=order, exp=(r.get("exp_name") or "").strip(), gpu=(r.get("gpu") or "").strip(),
                   what=(r.get("what") or "").strip(), csv=(r.get("result_csv") or "").strip(),
                   cond=(r.get("cond") or "").strip(), gate=(r.get("gate") or "").strip(),
                   note=(r.get("note") or "").strip())
        if sec == "running":
            run.append(rec)
        elif sec == "queued":
            q.append(rec)
    run.sort(key=lambda x: x["order"])
    q.sort(key=lambda x: x["order"])
    return run, q


def landed(rec):
    s = stats(rec["csv"], rec["cond"]) if rec["csv"] else None
    if s and s.get("ts"):
        return f'<span style="color:var(--done)">✓ landed — test ρ {cell(s["ts"])}</span>'
    return '<span class="mut">result pending</span>'


CSS = """
:root{--ink:#1c2433;--soft:#5b6678;--line:#e3e7ee;--bg:#f5f6f8;--card:#fff;
--run:#1d6fd0;--run-bg:#e7f0fc;--queue:#b07a17;--queue-bg:#fcf3e0;--done:#2f6f4f;--done-bg:#eaf5ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.page{max-width:1280px;margin:0 auto;padding:40px 24px 90px}
header.doc{border-bottom:2px solid var(--ink);padding-bottom:15px;margin-bottom:26px}
header.doc .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin-bottom:6px}
header.doc h1{font-size:26px;font-weight:650;margin:0}
header.doc .sub{font-size:13px;color:var(--soft);margin-top:6px}
.sec{margin-bottom:34px}
.sec-head{font-size:18px;font-weight:640;margin:0 0 12px;display:flex;align-items:center;gap:10px}
.dot{width:11px;height:11px;border-radius:50%}
.dot.run{background:var(--run)}.dot.queue{background:var(--queue)}.dot.done{background:var(--done)}
.sec-head .count{font-size:12.5px;font-weight:500;color:var(--soft)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px 13px;text-align:left;vertical-align:top}
thead th{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--soft);background:#fafbfc;border-bottom:1px solid var(--line);white-space:nowrap}
tbody tr{border-top:1px solid var(--line)}tbody tr:hover{background:#fbfcfd}
tr.grp td{background:#f3f5f8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--soft);padding:6px 13px}
.exp{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:#1d4ed8;word-break:break-word}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.run{background:var(--run-bg);color:var(--run)}.pill.queue{background:var(--queue-bg);color:var(--queue)}
.num{font-weight:660;white-space:nowrap}.pm{color:var(--soft);font-weight:500;font-size:12px}
.best .num{color:#1d5a3a;font-weight:760}
.mut{color:var(--soft)}.small{font-size:12.5px;color:var(--soft)}
.tag{display:inline-block;font-size:10.5px;font-weight:600;padding:1px 7px;border-radius:999px;background:#eef1f6;color:var(--soft);margin-left:6px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.empty{padding:18px;color:var(--soft);font-size:13.5px;text-align:center}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;font-size:12px;color:var(--soft)}
.note{margin-top:11px;font-size:12.5px;color:var(--soft);line-height:1.55}
"""


def build():
    run, q = load_status()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Running
    if run:
        rr = ""
        for r in run:
            rr += (f'<tr><td><span class="pill run">training</span><br><span class="exp">{esc(r["exp"])}</span></td>'
                   f'<td class="small" style="white-space:normal">{r["what"]}</td><td>GPU {esc(r["gpu"])}</td>'
                   f'<td class="small"><code>{esc(r["csv"])}</code><div class="small mut">{landed(r)}</div></td></tr>')
    else:
        rr = '<tr><td colspan="4"><div class="empty">Nothing training right now.</div></td></tr>'

    # Queued
    if q:
        qq = ""
        for i, r in enumerate(q, 1):
            qq += (f'<tr><td>{i}</td><td><span class="pill queue">chained</span><br><span class="exp">{esc(r["exp"])}</span></td>'
                   f'<td class="small" style="white-space:normal">{r["what"]}</td>'
                   f'<td class="small">{esc(r["gate"])}</td></tr>')
    else:
        qq = '<tr><td colspan="4"><div class="empty">No chained steps armed.</div></td></tr>'

    # Done
    dd, ndone = "", 0
    for d in DONE:
        if "grp" in d:
            dd += f'<tr class="grp"><td colspan="5">{esc(d["grp"])}</td></tr>'
            continue
        ndone += 1
        s = stats(d["csv"], d["cond"])
        tag = f'<span class="tag">{esc(d["tag"])}</span>' if d.get("tag") else ""
        tcell = cell(s["ts"]) if s else '<span class="mut">—</span>'
        ninfo = f'<div class="small mut">n={esc(s["n"])}, {s["k"]} seed{"" if s["k"]==1 else "s"}</div>' if s else ""
        vcell = cell(s["bv"]) if s else '<span class="mut">—</span>'
        rcell = cell(s["rm"]) if s else '<span class="mut">—</span>'
        cls = ' class="best"' if d.get("best") else ""
        dd += (f'<tr{cls}><td style="white-space:normal"><b>{d["label"]}</b>{tag}</td>'
               f'<td class="small" style="white-space:normal">{d["desc"]}</td>'
               f'<td>{tcell}{ninfo}</td><td>{vcell}</td><td>{rcell}</td></tr>')

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment Queue — VoxBind</title><style>{CSS}</style></head><body><div class="page">
<header class="doc"><div class="date">VoxBind · density &amp; affinity line</div>
<h1>Experiment Queue</h1>
<div class="sub">Running / Queued / Done — <b>static page</b> rendered by
<code>build_queue_html.py</code> from <code>experiment_status.csv</code> (queue) + the
<code>probe_results_*.csv</code> files (values). No server, no JavaScript — previews anywhere.
Built {ts}. Affinity = LP-PDBBind <code>new_split</code>, canonical 839 test split (unless noted).</div></header>

<section class="sec"><h2 class="sec-head"><span class="dot run"></span> Running
<span class="count">— {len(run)} training{'' if len(run)==1 else 's'}</span></h2>
<div class="wrap"><table><thead><tr><th style="width:30%">Experiment</th><th>What it does</th>
<th style="width:9%">GPU</th><th style="width:24%">Result CSV</th></tr></thead><tbody>{rr}</tbody></table></div></section>

<section class="sec"><h2 class="sec-head"><span class="dot queue"></span> Queued
<span class="count">— {len(q)} armed</span></h2>
<div class="wrap"><table><thead><tr><th style="width:4%">#</th><th style="width:28%">Experiment</th>
<th>What it does</th><th style="width:26%">Fires when</th></tr></thead><tbody>{qq}</tbody></table></div></section>

<section class="sec"><h2 class="sec-head"><span class="dot done"></span> Done
<span class="count">— {ndone} experiments</span></h2>
<div class="wrap"><table><thead><tr><th style="width:21%">Experiment</th><th style="width:40%">What it does</th>
<th style="width:13%">test&nbsp;ρ</th><th style="width:13%">val&nbsp;ρ</th><th style="width:13%">RMSE</th></tr></thead>
<tbody>{dd}</tbody></table></div>
<div class="legend"><span>test/val&nbsp;ρ = Spearman, mean&nbsp;±&nbsp;std over seeds</span>
<span>n = test molecules</span></div></section>
</div></body></html>"""


if __name__ == "__main__":
    html_out = build()
    OUT.write_text(html_out, encoding="utf-8")
    run, q = load_status()
    print(f"wrote {OUT.name}: {len(run)} running, {len(q)} queued, "
          f"{sum(1 for d in DONE if 'grp' not in d)} done rows | {len(html_out):,} bytes (static, no JS)")
