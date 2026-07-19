#!/usr/bin/env python3
"""build_regression_baseline.py — regenerate regression_baseline.html.

Reads the CheapNet result JSONs (base/cheapnet/_edrscc/results_*.json) and parses the
BindNet unicore training logs (base/bindnet/_edrscc/logs/train_seed*.log; test metrics
at the best-valid_pearson step) plus a table of context baselines (from prior meeting
docs / memory), and emits an HTML report styled like 260715_meeting.html. Re-runnable —
pending cells show as "running"/TBA and fill in as jobs complete.

    python notebook/html/260715/build_regression_baseline.py
"""
import os
import re
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CHEAP = os.path.join(REPO, "base", "cheapnet", "_edrscc")
BIND = os.path.join(REPO, "base", "bindnet", "_edrscc")
OUT = os.path.join(HERE, "regression_baseline.html")

# ── read CheapNet 3-seed result json ─────────────────────────────────────────
def read_cheapnet(tag):
    p = os.path.join(CHEAP, f"results_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ns = d.get("per_seed", [])
    n_test = None  # test-set size not stored; derived from meta below
    return {
        "r": (d["pearson"]["mean"], d["pearson"]["std"]),
        "rho": (d["spearman"]["mean"], d["spearman"]["std"]),
        "rmse": (d["rmse"]["mean"], d["rmse"]["std"]),
        "seeds": len(d.get("seeds", ns)),
        "params": d.get("n_params"),
    }

def cheapnet_counts(tag):
    """(train,valid,test) built counts from the data-root meta.json."""
    root = f"data_{tag}" if tag != "lp_edrscc_v2" else "data_lp_edrscc_v2"
    p = os.path.join(CHEAP, root, "meta.json")
    if os.path.exists(p):
        c = json.load(open(p)).get("counts", {})
        return c.get("train"), c.get("valid"), c.get("test")
    return None, None, None

# ── parse a BindNet unicore log: test metrics at best valid_pearson ───────────
_VMET = re.compile(r"valid_pearson (\-?[0-9.]+)")
_TMET = re.compile(r"test_pearson (\-?[0-9.]+).*?")
def parse_bindnet_log(path):
    if not os.path.exists(path):
        return None
    valid_by_upd, test_by_upd = {}, {}
    for line in open(path, errors="ignore"):
        if "| valid |" in line and "valid_pearson" in line:
            m = dict(re.findall(r"(valid_pearson|valid_spearman|valid_RMSE|num_updates) (\-?[0-9.]+)", line))
            if "num_updates" in m and "valid_pearson" in m:
                valid_by_upd[int(float(m["num_updates"]))] = float(m["valid_pearson"])
        elif "| test |" in line and "test_pearson" in line:
            m = dict(re.findall(r"(test_pearson|test_spearman|test_RMSE|num_updates) (\-?[0-9.]+)", line))
            if "num_updates" in m and "test_pearson" in m:
                test_by_upd[int(float(m["num_updates"]))] = (
                    float(m["test_pearson"]), float(m.get("test_spearman", "nan")), float(m.get("test_RMSE", "nan")))
    if not valid_by_upd or not test_by_upd:
        return None
    # best checkpoint = max valid_pearson among steps that also have a test eval
    common = [u for u in valid_by_upd if u in test_by_upd]
    if not common:
        return None
    best_u = max(common, key=lambda u: valid_by_upd[u])
    r, rho, rmse = test_by_upd[best_u]
    return {"r": r, "rho": rho, "rmse": rmse, "best_val": valid_by_upd[best_u], "step": best_u,
            "finished": "ALL_SEEDS" in open(path, errors="ignore").read() or "done training" in open(path, errors="ignore").read()}

def read_bindnet():
    """Aggregate BindNet seeds (seed0 log + seed12 chain log)."""
    logs = {0: os.path.join(BIND, "logs", "train_seed0.log")}
    # seeds 1,2 share one chained log; split on the "### SEED n ###" markers
    chain = os.path.join(BIND, "logs", "train_seed12.log")
    seed_results, running = [], False
    r0 = parse_bindnet_log(logs[0])
    if r0:
        seed_results.append(r0)
    if os.path.exists(chain):
        txt = open(chain, errors="ignore").read()
        parts = re.split(r"### SEED (\d) ###", txt)
        # parts = [pre, '1', body1, '2', body2, ...]
        for i in range(1, len(parts), 2):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            tmpf = os.path.join(BIND, "logs", f"_tmp_seed{parts[i]}.log")
            open(tmpf, "w").write(body)
            rr = parse_bindnet_log(tmpf)
            os.remove(tmpf)
            if rr:
                seed_results.append(rr)
    if not seed_results:
        return None
    import statistics as st
    def agg(k):
        vs = [s[k] for s in seed_results]
        return (sum(vs) / len(vs), (st.pstdev(vs) if len(vs) > 1 else 0.0))
    return {"r": agg("r"), "rho": agg("rho"), "rmse": agg("rmse"), "seeds": len(seed_results)}

# ── context baselines on lp_edrscc_v2 (test, 3-seed) — from prior docs/memory ─
# (r, r_sd), (rho, rho_sd), (rmse, rmse_sd) ; None where a value was not reported.
CONTEXT = [
    ("ProFSA",  "pretrained encoder (frozen) + probe · Uni-Mol pocket, ICLR24",
        (0.626, .005), (0.597, .007), (1.518, None), "pretrained"),
    ("GET",     "E(3)-equivariant transformer, trained · 0.72M · ICML24",
        (0.596, .005), (0.591, .005), (1.428, .008), "trained"),
    ("AEV-PLIG (ens.)", "GATv2 + radial AEV ligand graph · 3-model ensemble · Comms Chem 25",
        (0.584, None), (0.550, None), (1.480, None), "trained"),
    ("EGNN + TargetDiff", "EGNN pretraining feature baseline",
        (None, None), (0.579, None), (None, None), "trained"),
    ("HBGSA",   "hierarchical bipartite graph, 3.06M paper-faithful",
        (None, None), (0.533, None), (None, None), "trained"),
    ("DSMBind", "zero-shot unsupervised energy (size-confounded)",
        (None, None), (0.363, None), (None, None), "zeroshot"),
]
# density-probe reference (the VoxBind encoder), from 260715 meeting Table 1
DENSITY = [
    ("C + D + G density probe", "coords+density+gradmag ChannelViT[7,4,2], 40M·PLINDER v1 — frozen probe (ref)",
        (0.656, .006), (0.637, .006), (1.353, .031), "ref"),
    ("coords-only ViT probe", "coords-only encoder, frozen probe (ref)",
        (None, None), (0.605, None), (None, None), "ref"),
]

# ── HTML helpers ─────────────────────────────────────────────────────────────
CSS = """
  :root{--ink:#1c2433;--ink-soft:#5b6678;--line:#e3e7ee;--line-strong:#aeb6c4;--bg:#f5f6f8;--card:#fff;
    --accent:#2f6f4f;--best-bg:#eaf5ee;--tbd:#b07a17;--tbd-bg:#fcf3e0;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;}
  .page{max-width:1280px;margin:0 auto;padding:48px 24px 96px;}
  header.doc{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:40px;}
  header.doc .date{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:6px;}
  header.doc h1{font-size:28px;font-weight:650;margin:0;letter-spacing:-0.01em;}
  header.doc .lead{font-size:14.5px;color:var(--ink-soft);margin:10px 0 0;max-width:1080px;}
  .doc-section{margin-bottom:64px;}
  .section-head{font-size:22px;font-weight:660;letter-spacing:-0.01em;margin:0 0 18px;padding-bottom:12px;
    border-bottom:2px solid var(--line-strong);display:flex;align-items:center;gap:14px;}
  .section-head .sec-num{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;
    border-radius:8px;background:var(--accent);color:#fff;font-size:15px;font-weight:700;flex:0 0 auto;}
  .section-intro{font-size:13.5px;color:var(--ink-soft);margin:-2px 0 22px;max-width:1080px;line-height:1.6;}
  section.block{margin-bottom:40px;}
  .table-title{font-size:19px;font-weight:620;margin:0 0 4px;}
  .table-sub{font-size:13.5px;color:var(--ink-soft);margin:0 0 18px;max-width:1080px;}
  .table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);
    box-shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05);}
  table.results{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;}
  table.results th,table.results td{padding:11px 9px;text-align:center;white-space:nowrap;}
  table.results thead th{font-weight:600;background:#fafbfc;border-bottom:1px solid var(--line);}
  table.results thead tr.grp th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);font-weight:600;padding-top:13px;padding-bottom:6px;}
  table.results thead tr.sub th{font-size:11px;font-weight:500;color:var(--ink-soft);padding-top:0;padding-bottom:11px;}
  .col-method{text-align:left !important;min-width:230px;}
  td.col-method{font-weight:560;}
  td.col-method .sub{display:block;font-size:11px;color:#7a8699;font-weight:400;white-space:normal;}
  td.col-method .tag,.tag{display:inline-block;font-size:10.5px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:999px;margin-left:7px;vertical-align:middle;}
  .tag.trained{background:#eef1f6;color:#5b6678;}
  .tag.pretrained{background:#e7eefb;color:#38559b;}
  .tag.zeroshot{background:#f3eefb;color:#6b4b9b;}
  .tag.ref{background:var(--best-bg);color:#1d5a3a;}
  .tag.newmodel{background:#fdecdb;color:#b45c17;}
  .tag.inprog{background:var(--tbd-bg);color:var(--tbd);}
  .div-major{border-left:2.5px solid var(--line-strong);}
  table.results tbody tr{border-top:1px solid var(--line);}
  table.results tbody tr.grp-top{border-top:2px solid var(--line-strong);}
  table.results tbody tr.newrow{background:#fffaf4;}
  table.results tbody tr:hover{background:#fbfcfd;}
  .metric{font-size:14.5px;}
  .metric .val{font-weight:560;}
  .metric .sd{color:#9aa3b2;font-size:10.5px;font-weight:400;}
  td.best{background:var(--best-bg);}
  td.best .val{font-weight:720;color:#1d5a3a;}
  .tbd{color:var(--tbd);font-weight:600;font-size:13px;}
  .na{color:#c2c8d2;}
  .notes{margin-top:18px;font-size:13px;color:var(--ink-soft);line-height:1.6;max-width:1080px;}
  .notes b{color:var(--ink);font-weight:600;}
  .notes ul{margin:6px 0 0;padding-left:20px;} .notes li{margin-bottom:5px;}
  .notes code,.table-sub code,.section-intro code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12.5px;}
  .legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;font-size:12.5px;color:var(--ink-soft);}
  .legend span{display:inline-flex;align-items:center;gap:6px;}
"""

def cell(v, best=False, div=False, digits=3):
    cls = "metric" + (" div-major" if div else "") + (" best" if best else "")
    if v is None or v[0] is None:
        return f'<td class="{cls}"><span class="na">&mdash;</span></td>'
    val, sd = v
    sds = f' <span class="sd">±{sd:.3f}</span>' if sd is not None else ""
    return f'<td class="{cls}"><span class="val">{val:.{digits}f}</span>{sds}</td>'

def tba_cell(div=False):
    cls = "metric" + (" div-major" if div else "")
    return f'<td class="{cls}"><span class="tbd">running</span></td>'

def method_cell(name, sub, tag=None):
    t = f' <span class="tag {tag}">{ {"trained":"supervised","pretrained":"pretrained","zeroshot":"zero-shot","ref":"density probe","newmodel":"NEW","inprog":"running"}.get(tag,tag) }</span>' if tag else ""
    return f'<td class="col-method">{name}{t}<span class="sub">{sub}</span></td>'

def row(name, sub, tag, r, rho, rmse, best_r=False, best_rho=False, best_rmse=False, cls=""):
    tr = f'<tr class="{cls}">' if cls else "<tr>"
    return (tr + method_cell(name, sub, tag)
            + cell(r, best_r, div=True) + cell(rho, best_rho) + cell(rmse, best_rmse) + "</tr>")

def build():
    cn_v2 = read_cheapnet("lp_edrscc_v2")
    bind = read_bindnet()

    # ---- Table 1 rows: NEW models first, then context ----
    # best-in-column across the models actually present (test r / rho / rmse)
    present = []
    if cn_v2: present.append(("CheapNet", cn_v2))
    if bind: present.append(("BindNet", bind))
    for nm, sub, r, rho, rmse, tag in CONTEXT:
        present.append((nm, {"r": r, "rho": rho, "rmse": rmse}))
    def best_of(key, hi=True):
        vals = [(nm, m[key][0]) for nm, m in present if m.get(key) and m[key][0] is not None]
        if not vals: return None
        return (max if hi else min)(vals, key=lambda x: x[1])[0]
    best_r_name = best_of("r"); best_rho_name = best_of("rho"); best_rmse_name = best_of("rmse", hi=False)

    rows = []
    # NEW: CheapNet
    if cn_v2:
        rows.append(row("CheapNet", "cross-attn hierarchical GNN (GIGN-based), trained from scratch · ICLR25 · 1.33M",
            "newmodel", cn_v2["r"], cn_v2["rho"], cn_v2["rmse"],
            best_r_name=="CheapNet", best_rho_name=="CheapNet", best_rmse_name=="CheapNet", cls="newrow grp-top"))
    else:
        rows.append('<tr class="newrow grp-top">' + method_cell("CheapNet",
            "cross-attn hierarchical GNN (GIGN-based), trained from scratch · ICLR25", "newmodel")
            + tba_cell(div=True) + tba_cell() + tba_cell() + "</tr>")
    # NEW: BindNet
    bsub = "Uni-Mol complex cross-net, <b>trained from scratch</b> (pretrained ckpt access-restricted) · ICLR24"
    if bind:
        rows.append(row("BindNet", bsub, "newmodel", bind["r"], bind["rho"], bind["rmse"],
            best_r_name=="BindNet", best_rho_name=="BindNet", best_rmse_name=="BindNet", cls="newrow"))
    else:
        rows.append('<tr class="newrow">' + method_cell("BindNet", bsub, "newmodel")
            + tba_cell(div=True) + tba_cell() + tba_cell() + "</tr>")
    # context
    for i, (nm, sub, r, rho, rmse, tag) in enumerate(CONTEXT):
        rows.append(row(nm, sub, tag, r, rho, rmse,
            best_r_name==nm, best_rho_name==nm, best_rmse_name==nm, cls="grp-top" if i == 0 else ""))
    # density reference
    for j, (nm, sub, r, rho, rmse, tag) in enumerate(DENSITY):
        rows.append(row(nm, sub, tag, r, rho, rmse, cls="grp-top" if j == 0 else ""))

    t1 = "\n".join(rows)

    # ---- Table 2: CheapNet across CL diverse-split trials ----
    cl_rows = []
    cl_specs = [("lp_edrscc_v2", "base — LP ∩ ED ∩ RSCC≥0.8 ∩ Kd/Ki", ""),
                ("lp_edrscc_v2_cl1", "+ LP cleaning CL1", "CL1"),
                ("lp_edrscc_v2_cl12", "+ LP cleaning CL1+CL2", "CL1+2"),
                ("lp_edrscc_v2_cl123", "+ LP cleaning CL1+CL2+CL3", "CL1+2+3")]
    for i, (tag, sub, chip) in enumerate(cl_specs):
        d = read_cheapnet(tag); tr, va, te = cheapnet_counts(tag)
        nlab = f"{tr}/{va}/{te}" if tr else "—"
        name = tag.replace("lp_edrscc_v2", "v2")
        mc = (f'<td class="col-method">{name}'
              + (f' <span class="tag inprog">{chip}</span>' if chip else "")
              + f'<span class="sub">{sub} &middot; train/val/test = {nlab}</span></td>')
        if d:
            cl_rows.append('<tr class="' + ("grp-top" if i == 0 else "") + '">' + mc
                + cell(d["r"], div=True) + cell(d["rho"]) + cell(d["rmse"]) + "</tr>")
        else:
            cl_rows.append('<tr class="' + ("grp-top" if i == 0 else "") + '">' + mc
                + tba_cell(div=True) + tba_cell() + tba_cell() + "</tr>")
    t2 = "\n".join(cl_rows)

    bind_status = (f"{bind['seeds']}/3 seeds" if bind else "training")
    cn_status = "done" if cn_v2 else "training"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regression baselines — CheapNet &amp; BindNet on LP-PDBBind</title>
<style>{CSS}</style></head><body><div class="page">

  <header class="doc">
    <div class="date">2026 · 07 · 16 — Baselines</div>
    <h1>Structure-based affinity regression — CheapNet &amp; BindNet</h1>
    <p class="lead">Two new supervised baselines trained on our frozen <code>lp_edrscc_v2</code> split
      (LP-PDBBind &cap; ED &cap; RSCC&nbsp;≥&nbsp;0.8, Kd/Ki; 3850&nbsp;/&nbsp;817&nbsp;/&nbsp;1320), reported against
      the existing baseline suite and the VoxBind density probe. Metrics: Pearson&nbsp;<i>r</i>,
      Spearman&nbsp;&rho;, RMSE on the held-out test set, mean&nbsp;±&nbsp;std over 3 seeds
      (best-validation checkpoint).</p>
  </header>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">1</span> CheapNet &amp; BindNet vs. the baseline suite — <code>lp_edrscc_v2</code></h2>
    <p class="section-intro">
      <b>CheapNet</b> (ICLR&nbsp;2025) — cross-attention over hierarchical (atom&nbsp;&rarr;&nbsp;cluster) protein/ligand
      representations on a GIGN message-passing backbone; trained from scratch. <b>BindNet</b> (ICLR&nbsp;2024) —
      a Uni-Mol complex cross-net; its large pretrained checkpoint is <b>access-restricted</b> on Google Drive, so
      per the brief we train the <b>architecture from scratch</b> on our split (a lower bound on its pretrained
      potential). Rows below the divider are prior baselines and the density probe, for reference.
    </p>
    <section class="block">
      <p class="table-title">Table 1 &nbsp;&middot;&nbsp; Supervised structure-based baselines — <code>lp_edrscc_v2</code> test (3-seed)</p>
      <p class="table-sub">Top two rows (highlighted) are the new models. <b>best</b> = best value in each test column
        across all models shown. CheapNet: {cn_status}. BindNet: {bind_status} (from scratch).</p>
      <div class="table-wrap"><table class="results">
        <thead>
          <tr class="grp"><th class="col-method" rowspan="2">Method</th>
            <th class="div-major" colspan="3">Test — lp_edrscc_v2 (1320)</th></tr>
          <tr class="sub"><th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE</th></tr>
        </thead>
        <tbody>
        {t1}
        </tbody>
      </table></div>
      <div class="legend">
        <span><span class="tag newmodel">NEW</span> trained this run</span>
        <span><span class="tag trained">supervised</span> prior supervised baseline</span>
        <span><span class="tag pretrained">pretrained</span> pretrained encoder + probe</span>
        <span><span class="tag zeroshot">zero-shot</span> unsupervised</span>
        <span><span class="tag ref">density probe</span> VoxBind encoder (reference)</span>
      </div>
      <div class="notes">
        <b>Reading.</b> On this hard novel-target LP split all supervised structure models cluster at
        <i>r</i>&nbsp;≈&nbsp;0.55–0.63. CheapNet lands in that band. BindNet-from-scratch is a
        <b>lower bound</b> — the method is designed to be pretrained on BioLip then finetuned; with the pretrained
        weights it would be expected to move up toward ProFSA (its Uni-Mol sibling). The density probe
        (<b>C&nbsp;+&nbsp;D&nbsp;+&nbsp;G</b>, &rho;&nbsp;0.637) remains the ceiling.
      </div>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">2</span> Diverse-split trials — LP-PDBBind cleaning levels</h2>
    <p class="section-intro">
      Three progressively-cleaner subsets of <code>lp_edrscc_v2</code>, each an exact subset with the LP-PDBBind
      per-complex <b>cleaning level</b> (nested CL3&nbsp;⊆&nbsp;CL2&nbsp;⊆&nbsp;CL1) applied to <b>all</b> splits.
      CheapNet re-trained (3 seeds) on each. <b>Note:</b> the test set differs per row (N shrinks with cleaning),
      so this measures how benchmark difficulty shifts with data quality, not same-test comparison.
    </p>
    <section class="block">
      <p class="table-title">Table 2 &nbsp;&middot;&nbsp; CheapNet across cleaning levels (test, 3-seed)</p>
      <p class="table-sub">Split totals shrink 5987&nbsp;&rarr;&nbsp;2702 as cleaning tightens.</p>
      <div class="table-wrap"><table class="results">
        <thead>
          <tr class="grp"><th class="col-method" rowspan="2">Split variant</th>
            <th class="div-major" colspan="3">Test (per-variant N)</th></tr>
          <tr class="sub"><th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE</th></tr>
        </thead>
        <tbody>
        {t2}
        </tbody>
      </table></div>
      <div class="notes">
        Splits are frozen + hash-gated in <code>voxbind/splits/</code> (schemes
        <code>lp_edrscc_v2_cl1/_cl12/_cl123</code>); BindNet on the CL variants is future work.
      </div>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">3</span> Method &amp; reproducibility</h2>
    <div class="notes" style="max-width:1080px">
      <ul>
        <li><b>CheapNet</b> — <code>base/cheapnet/</code>. GIGN-style graphs (5&nbsp;Å PyMOL pocket, 35-dim atom
          features, intra + inter&nbsp;≤5&nbsp;Å edges) bucketed by the frozen split; 3 HIL blocks &rarr; DiffPool
          clustering ([28,157]) &rarr; ligand/pocket cross-attention. Config = authors' cross_dataset
          (batch&nbsp;128, ≤800&nbsp;epochs, early-stop&nbsp;100, Adam&nbsp;lr&nbsp;5e-4). 5959/5987 complexes built.</li>
        <li><b>BindNet</b> — <code>base/bindnet/</code>. Uni-Mol/unicore <code>affinity_regres</code> (complex_crnet,
          recycling&nbsp;1, seperate-CLS); ATOM3D-LBA <code>.npy</code> data (256-atom pocket + ligand, atomic
          numbers, pK). Trained from scratch on GPU (batch&nbsp;16, lr&nbsp;1e-4, early-stop on
          <code>valid_pearson</code>). <b>Pretrained checkpoint was access-restricted</b> (gdown denied) — to
          finetune the pretrained model, obtain <code>checkpoint_bindnet.pt</code> and pass
          <code>CKPT=…&nbsp;bash&nbsp;_edrscc/train_bindnet.sh</code>.</li>
        <li>Both use the same LP_PDBBind pK labels and structures as the GET / ProFSA baselines, for
          comparability. Repro scripts + READMEs are under each baseline's folder.</li>
      </ul>
    </div>
  </div>

</div></body></html>"""
    open(OUT, "w").write(html)
    print(f"wrote {OUT}")
    print(f"  CheapNet v2: {'ok' if cn_v2 else 'pending'}   BindNet: {bind_status if bind else 'pending'}")


if __name__ == "__main__":
    build()
