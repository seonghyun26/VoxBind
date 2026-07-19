#!/usr/bin/env python3
"""build_results.py — assemble notebook/html/results.html (results only, no analysis).

Section 1 — Binding-affinity regression: method-comparison table with a GROUPED-COLUMN
  layout over four split variants (lp_edrscc_v2, +CL1, +CL1+CL2, +CL1+CL2+CL3 — the
  progressively-cleaner LP-PDBBind no-leak tiers), each reporting Pearson r / Spearman ρ /
  RMSE (mean ± std, 3 seeds). Cells fill in as the per-split campaign completes; not-yet-run
  cells show as "running". Reuses METHODS/colors from 260715/build_appendixB_bar.py for the
  v2 column + the v2 headline bar chart, and reads CheapNet per-split JSONs live.
  Adds HonestAffinity (arXiv 2606.03422 — leak-aware ESM-2 sequence baseline) as a new row.
Section 2 — De novo drug design: Table 2 (+ heavy-atom count column) + the Vina PNG figure,
  extracted verbatim from 260715_meeting.html.

    python notebook/html/build_results.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # notebook/html
REPO = os.path.dirname(os.path.dirname(HERE))                     # VoxBind

# Heavy-atom counts (Avg, Med) per generated molecule, one entry per Table-2 row in order.
# Rows 0-7 (prior methods) are from the VoxBind paper (Table 1, arXiv:2405.03961).
# Rows 8-10 (our reproduced VoxBind σ0.9/σ1.0 + Ours) are None -> rendered "—": the full
# 79-pocket generation output isn't retained in this checkout. To fill: replace the None
# with (avg, med) once the numbers are ready, then re-run this script.
ATOM_COUNTS = [
    (22.8, 21.5),   # Reference
    (17.6, 16.0),   # AR
    (17.7, 15.0),   # Pocket2Mol
    (24.0, 23.0),   # DiffSBDD
    (24.2, 24.0),   # TargetDiff
    (20.9, 21.0),   # DecompDiff
    (23.4, 24.0),   # VoxBind σ=0.9  (paper)
    (21.7, 22.0),   # VoxBind σ=1.0  (paper)
    None,           # VoxBind σ=0.9  (reproduced, 79 density pockets) — TODO fill
    None,           # VoxBind σ=1.0  (reproduced) — TODO fill
    None,           # Ours (frozen C+D+G σ=0.9) — TODO fill
]


def add_atom_column(table):
    """Append a '# Atoms (Avg/Med)' column group to the extracted de novo Table 2."""
    if "# Atoms" in table:                                        # idempotent
        return table
    thead_end = table.index("</thead>")
    head, body = table[:thead_end], table[thead_end:]
    head = re.sub(r'(<tr class="grp">.*?)</tr>',
                  r'\1<th class="div-major" colspan="2"># Atoms<br>(heavy)</th></tr>',
                  head, count=1, flags=re.S)
    head = re.sub(r'(<tr class="sub">.*?)</tr>',
                  r'\1<th class="div-major">Avg</th><th>Med</th></tr>', head, count=1, flags=re.S)
    cnt = [0]

    def proc(m):
        row = m.group(0)
        if 'colspan="17"' in row:                                # section-band rows
            return row.replace('colspan="17"', 'colspan="19"')
        if "col-method" not in row:
            return row
        i = cnt[0]; cnt[0] += 1
        val = ATOM_COUNTS[i] if i < len(ATOM_COUNTS) else None
        if val is not None:
            a, md = val
            cells = (f'<td class="metric div-major"><span class="val">{a:.1f}</span></td>'
                     f'<td class="metric"><span class="val">{md:.1f}</span></td>')
        else:
            cells = ('<td class="metric div-major"><span class="sd">&mdash;</span></td>'
                     '<td class="metric"><span class="sd">&mdash;</span></td>')
        return re.sub(r'</tr>\s*$', cells + '</tr>', row, flags=re.S)

    body = re.sub(r'<tr\b.*?</tr>', proc, body, flags=re.S)
    return head + body


sys.path.insert(0, os.path.join(HERE, "260715"))
import build_appendixB_bar as bar                                 # noqa: E402  (METHODS, svg(), legend())
import build_regression_baseline as reg                           # noqa: E402  (read_cheapnet())

DOC715 = os.path.join(HERE, "260715", "260715_meeting.html")
OUT = os.path.join(HERE, "results.html")

# ── the four split variants (scheme, display label, test-N) ───────────────────
SPLITS = [
    ("lp_edrscc_v2",       "lp_edrscc_v2",   1320),
    ("lp_edrscc_v2_cl1",   "+CL1",           1166),
    ("lp_edrscc_v2_cl12",  "+CL1+CL2",       1149),
    ("lp_edrscc_v2_cl123", "+CL1+CL2+CL3",    733),
]

# ── method metadata (v2 values + colors from bar.METHODS; + HonestAffinity) ───
V2 = {name: {"r": r, "rho": rho, "rmse": rmse}
      for (name, color, flag, r, rho, rmse) in bar.METHODS}
COLOR = {name: color for (name, color, *_ ) in bar.METHODS}
FLAG = {name: flag for (name, color, flag, *_ ) in bar.METHODS}
COLOR["HonestAffinity"] = "#00b894"
FLAG["HonestAffinity"] = "new"

TYPE = {  # display tag per method — three categories, distinctly colored
    "HBGSA": "supervised", "EGNN": "supervised", "EGNN + TargetDiff": "supervised",
    "GET": "supervised", "ProFSA": "pretrained", "CheapNet": "supervised", "BindNet": "supervised",
    "HonestAffinity": "supervised", "AEV-PLIG": "supervised", "DSMBind": "zero-shot",
    "C": "pretrained", "C+D+G": "pretrained",
}
TAGCLASS = {"supervised": "supervised", "pretrained": "pretrained", "zero-shot": "zeroshot"}

# Methods whose CL cells are literature/external numbers we do NOT reproduce per-split
# → render "—" instead of "running". Empty: the whole grid is being reproduced this campaign.
EXTERNAL = set()

# Methods whose v2 value in bar.METHODS is a PAPER number (not run on our ED+RSCC split) —
# suppress it (show "running") until we train them on our splits (base/get, model_type EGNN).
PAPER_ONLY = {"EGNN", "EGNN + TargetDiff"}

# method render order: bar order, with HonestAffinity inserted after BindNet
ORDER = []
for (name, *_ ) in bar.METHODS:
    ORDER.append(name)
    if name == "BindNet":
        ORDER.append("HonestAffinity")

# ── CASF-2016 evaluation columns (all methods trained on lp_edrscc_v2 TRAIN, tested on CASF) ──
# leaky = full CASF-2016 core (214 ED-avail, 90 in v2-train → inflated); nontrain = the 124 held out.
# Results in base/_casf/{key}.json with {"leaky":{pearson:{mean,std},...}, "nontrain":{...}}.
CASF_COLS = [("leaky", "CASF-2016", 214), ("nontrain", "CASF-2016 &minus;train", 124)]
CASF_KEY = {"C": "C", "C+D+G": "CDG", "DSMBind": "DSMBind", "GET": "GET", "EGNN": "EGNN",
            "EGNN + TargetDiff": "EGNN_TD", "CheapNet": "CheapNet", "BindNet": "BindNet",
            "AEV-PLIG": "AEV", "HBGSA": "HBGSA", "ProFSA": "ProFSA", "HonestAffinity": "HonestAffinity"}


def load_casf(method, which):
    """CASF result for a method → {'r','rho','rmse'} or None. which ∈ {'leaky','nontrain'}."""
    key = CASF_KEY.get(method)
    if not key:
        return None
    path = os.path.join(REPO, "base", "_casf", f"{key}.json")
    if not os.path.exists(path):
        return None
    import json as _j
    blk = _j.load(open(path)).get(which)
    if not isinstance(blk, dict):
        return None
    g = lambda k: (blk[k]["mean"], blk[k].get("std")) if isinstance(blk.get(k), dict) else None
    r, rho, rmse = g("pearson"), g("spearman"), g("rmse")
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


_RESULTS_DIR = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "results")

# VoxBind frozen-encoder probe CSVs per (method, split) — C = coords ViT (otf_coords_mask050),
# C+D+G = ChannelViT[7,4,2] (260623_ar_cvit_c1_g742). v2 = canonical CSVs; CL = this campaign.
PROBE_CSV = {
    "C": {
        "lp_edrscc_v2":       "probe_results_e99_v5_lp_edrscc_v2split_otf_coords_mask050.csv",
        "lp_edrscc_v2_cl1":   "probe_results_e99_v5_lp_edrscc_v2_cl1split_c_coords.csv",
        "lp_edrscc_v2_cl12":  "probe_results_e99_v5_lp_edrscc_v2_cl12split_c_coords.csv",
        "lp_edrscc_v2_cl123": "probe_results_e99_v5_lp_edrscc_v2_cl123split_c_coords.csv",
    },
    "C+D+G": {
        "lp_edrscc_v2":       "probe_results_e99_v5_lp_edrscc_v2split_260623_ar_cvit_c1_g742.csv",
        "lp_edrscc_v2_cl1":   "probe_results_e99_v5_lp_edrscc_v2_cl1split_cdg_g742.csv",
        "lp_edrscc_v2_cl12":  "probe_results_e99_v5_lp_edrscc_v2_cl12split_cdg_g742.csv",
        "lp_edrscc_v2_cl123": "probe_results_e99_v5_lp_edrscc_v2_cl123split_cdg_g742.csv",
    },
}


# Per-(method, split) result JSONs written by the CL campaign runs, formatted with the split.
# Short-tag paths use v2/cl1/cl12/cl123; full-tag paths use the scheme name. Checked before the
# v2/paper fallback so cells auto-fill as each baseline's run lands; a missing file → None.
SHORT = {"lp_edrscc_v2": "v2", "lp_edrscc_v2_cl1": "cl1",
         "lp_edrscc_v2_cl12": "cl12", "lp_edrscc_v2_cl123": "cl123"}
METHOD_PATH = {
    "GET":               lambda s: f"base/get/_edrscc/results_GET_{SHORT[s]}.json",
    "EGNN":              lambda s: f"base/get/_edrscc/results_EGNN_{SHORT[s]}.json",
    "EGNN + TargetDiff": lambda s: f"base/get/_edrscc/results_EGNN_TD_{SHORT[s]}.json",
    "BindNet":           lambda s: f"base/bindnet/_edrscc/results_{s}.json",
    "ProFSA":            lambda s: f"base/profsa/results/profsa_{s}.json",
    "AEV-PLIG":          lambda s: f"base/aevplig/results/aevplig_{s}.json",
    "HBGSA":             lambda s: f"base/hbgsa/results/hbgsa_{SHORT[s]}.json",
    "HonestAffinity":    lambda s: f"base/honestaffinity/results/results_{s}.json",
    "DSMBind":           lambda s: f"base/dsmbind/results/dsmbind_probe_{s}.json",
}


def _read_result(relpath):
    """Format-tolerant result reader → {'r':(m,sd),'rho':(m,sd),'rmse':(m,sd)} or None.

    Handles the four schemas the campaign emits:
      A  {"pearson":{"mean","std"}, ...}                       (GET/EGNN, HBGSA, BindNet, Honest)
      B  {"per_seed_mean_std":{"test_pearson":[m,s], ...}}     (AEV-PLIG)
      D  {"mean_std":{"test_pearson":[m,s], ...}}              (ProFSA)
      C  {"mean_pearson","std_pearson", ...}                  (DSMBind probe)
    """
    path = os.path.join(REPO, relpath)
    if not os.path.exists(path):
        return None
    import json as _j
    d = _j.load(open(path))
    if isinstance(d.get("pearson"), dict) and "mean" in d["pearson"]:            # A
        g = lambda k: (d[k]["mean"], d[k].get("std")) if isinstance(d.get(k), dict) and "mean" in d[k] else None
        r, rho, rmse = g("pearson"), g("spearman"), g("rmse")
    elif isinstance(d.get("per_seed_mean_std"), dict) or isinstance(d.get("mean_std"), dict):   # B / D
        m = d.get("per_seed_mean_std") or d.get("mean_std")
        g = lambda k: (m[k][0], m[k][1]) if isinstance(m.get(k), (list, tuple)) and len(m[k]) >= 2 else None
        r, rho, rmse = g("test_pearson"), g("test_spearman"), g("test_rmse")
    elif "mean_pearson" in d:                                                    # C
        g = lambda mk, sk: (d[mk], d.get(sk)) if mk in d else None
        r, rho, rmse = g("mean_pearson", "std_pearson"), g("mean_spearman", "std_spearman"), g("mean_rmse", "std_rmse")
    else:
        return None
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


def _probe_csv(fname):
    """Read a probe_results CSV (per-seed rows) → {'r','rho','rmse'} mean±std over seeds."""
    import csv as _csv
    import statistics as _st
    path = os.path.join(_RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    rows = list(_csv.DictReader(open(path)))

    def col(c):
        vs = [float(r[c]) for r in rows if r.get(c) not in (None, "", "nan")]
        if not vs:
            return None
        return (sum(vs) / len(vs), _st.pstdev(vs) if len(vs) > 1 else 0.0)

    r, rho, rmse = col("test_pearson"), col("test_spearman"), col("test_rmse")
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


def load(method, split):
    """Per-method-per-split result loader → {'r':(m,sd),'rho':(m,sd),'rmse':(m,sd)} or None.

    Live campaign JSONs/CSVs fill as each run lands; else fall back to the v2 column from
    bar.METHODS (our runs), or None (→ "running" placeholder). Paper v2 (EGNN rows) suppressed.
    """
    if method == "CheapNet":
        return reg.read_cheapnet(split)                       # results_{split}.json (all 4 done)
    if method in PROBE_CSV:                                    # VoxBind C / C+D+G frozen-encoder probe
        fn = PROBE_CSV[method].get(split)
        return _probe_csv(fn) if fn else None
    if method in METHOD_PATH:                                  # campaign JSONs fill as they land
        d = _read_result(METHOD_PATH[method](split))
        if d:
            return d
    if method in PAPER_ONLY:
        return None                                           # suppress paper v2 until run on our split
    if split == "lp_edrscc_v2":
        return V2.get(method)                                 # v2 column from bar.METHODS (our runs)
    return None                                               # CL cells: pending this campaign


def best_per_col():
    """best method per column among present values (r/ρ max, RMSE min); LP splits + CASF."""
    best = {}
    for split, _, _ in SPLITS:
        for metric, hi in (("r", True), ("rho", True), ("rmse", False)):
            vals = [(m, load(m, split)[metric][0]) for m in ORDER
                    if load(m, split) and load(m, split).get(metric) and load(m, split)[metric][0] is not None]
            if vals:
                best[(split, metric)] = (max if hi else min)(vals, key=lambda x: x[1])[0]
    for which, _, _ in CASF_COLS:
        for metric, hi in (("r", True), ("rho", True), ("rmse", False)):
            vals = [(m, load_casf(m, which)[metric][0]) for m in ORDER
                    if load_casf(m, which) and load_casf(m, which).get(metric) and load_casf(m, which)[metric][0] is not None]
            if vals:
                best[("casf", which, metric)] = (max if hi else min)(vals, key=lambda x: x[1])[0]
    return best


def slice_between(text, start, end, frm=0, include_end=True):
    i = text.index(start, frm)
    j = text.index(end, i) + (len(end) if include_end else 0)
    return text[i:j], j


def extract_from_715():
    doc = open(DOC715, encoding="utf-8").read()
    css = doc[doc.index("<style>"):doc.index("</style>") + len("</style>")]
    # De novo Table 2 — the <table class="results"> after the "Table 2" title
    k = doc.index("Table 2 &nbsp;")
    table2, _ = slice_between(doc, '<table class="results">', "</table>", frm=k)
    sub2, _ = slice_between(doc, '<p class="table-sub">', "</p>", frm=k)
    # De novo Figure 1 — the base64 Vina PNG (find the <img ...> after the "Figure 1" title)
    f = doc.index("Figure 1 &nbsp;")
    img, _ = slice_between(doc, "<img ", ">", frm=f)
    return css, table2, sub2, img


# ── Section 1: grouped affinity table over the four split variants ────────────
def cell(v, best=False, div=False):
    cls = "metric" + (" div-major" if div else "") + (" best" if best else "")
    if v is None or v[0] is None:
        return f'<td class="{cls}"><span class="na">&mdash;</span></td>'
    m, sd = v
    sds = f' <span class="sd">±{sd:.3f}</span>' if sd is not None else ""
    return f'<td class="{cls}"><span class="val">{m:.3f}</span>{sds}</td>'


def pending_cell(external, div=False):
    cls = "metric" + (" div-major" if div else "")
    inner = '<span class="na">&mdash;</span>' if external else '<span class="tbd">running</span>'
    return f'<td class="{cls}">{inner}</td>'


def metric_cell(v, best, divcls):
    """One metric td with an explicit divider class ('' | 'div-major' | 'div-block')."""
    cls = "metric" + (f" {divcls}" if divcls else "") + (" best" if best else "")
    if v is None or v[0] is None:
        return f'<td class="{cls}"><span class="na">&mdash;</span></td>'
    m, sd = v
    sds = f' <span class="sd">±{sd:.3f}</span>' if sd is not None else ""
    return f'<td class="{cls}"><span class="val">{m:.3f}</span>{sds}</td>'


def pending2(divcls):
    cls = "metric" + (f" {divcls}" if divcls else "")
    return f'<td class="{cls}"><span class="tbd">running</span></td>'


def method_cell(name):
    typ = TYPE.get(name, "supervised")
    tagc = TAGCLASS[typ]
    color = COLOR.get(name, "#888")
    flag = FLAG.get(name, False)
    border = ";border:2px solid #000" if flag is True else ""   # best marker only; new rows not highlighted
    sw = (f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
          f'background:{color};margin-right:7px;vertical-align:middle{border}"></span>')
    return f'<td class="col-method">{sw}{name}<span class="tag {tagc}">{typ}</span></td>'


def affinity_table_head():
    grp = ['<tr class="grp"><th class="col-method" rowspan="2">Method</th>']
    for _, label, n in SPLITS:
        grp.append(f'<th class="div-major" colspan="3">{label}<span class="nsub">N&nbsp;=&nbsp;{n}</span></th>')
    grp.append("</tr>")
    sub = ['<tr class="sub">']
    for _ in SPLITS:
        sub.append('<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>')
    sub.append("</tr>")
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def affinity_rows():
    best = best_per_col()
    rows = []
    for name in ORDER:
        external = name in EXTERNAL
        tds = [method_cell(name)]
        for split, _, _ in SPLITS:
            d = load(name, split)
            for k, metric in enumerate(("r", "rho", "rmse")):
                divcls = "div-major" if k == 0 else ""
                if d is None or d.get(metric) is None or d[metric][0] is None:
                    tds.append(pending2(divcls) if not external else metric_cell(None, False, divcls))
                else:
                    tds.append(metric_cell(d[metric], best.get((split, metric)) == name, divcls))
        rowcls = ' class="grp-top"' if name == "C" else ""     # divider before probe group; no new-row highlight
        rows.append(f"<tr{rowcls}>" + "".join(tds) + "</tr>")
    return "\n          ".join(rows)


def casf_table_head():
    grp = ['<tr class="grp"><th class="col-method" rowspan="2">Method</th>']
    for _, label, n in CASF_COLS:
        grp.append(f'<th class="div-major" colspan="3">{label}<span class="nsub">N&nbsp;=&nbsp;{n}</span></th>')
    grp.append("</tr>")
    sub = ['<tr class="sub">']
    for _ in CASF_COLS:
        sub.append('<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>')
    sub.append("</tr>")
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def casf_rows():
    """CASF appendix table — methods sorted by non-train ρ (honest), leaky + non-train × r/ρ/RMSE."""
    best = best_per_col()
    present = [m for m in ORDER if m in CASF_KEY and load_casf(m, "nontrain")]
    present.sort(key=lambda m: -(load_casf(m, "nontrain")["rho"][0] or -1))
    rows = []
    for name in present:
        tds = [method_cell(name)]
        for which, _, _ in CASF_COLS:
            d = load_casf(name, which)
            for k, metric in enumerate(("r", "rho", "rmse")):
                divcls = "div-major" if k == 0 else ""
                if d is None or d.get(metric) is None or d[metric][0] is None:
                    tds.append(pending2(divcls))
                else:
                    tds.append(metric_cell(d[metric], best.get(("casf", which, metric)) == name, divcls))
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return "\n          ".join(rows)


def casf_leakage_svg():
    """Dumbbell chart: per method, non-train ρ (honest, filled dot) → leaky ρ (inflated, open dot),
    sorted by leakage gap. Red = big leak (≥0.10, memorizers); blue = moderate; grey = negligible."""
    data = []
    for m in ORDER:
        lk, nt = load_casf(m, "leaky"), load_casf(m, "nontrain")
        if lk and nt and lk.get("rho") and nt.get("rho") and lk["rho"][0] is not None and nt["rho"][0] is not None:
            data.append((m, nt["rho"][0], lk["rho"][0]))
    data.sort(key=lambda t: t[2] - t[1], reverse=True)
    X0, X1, TOP, RH = 205, 855, 46, 26
    H = TOP + len(data) * RH + 52
    vmin, vmax = 0.40, 0.86
    def x(r): return X0 + (r - vmin) / (vmax - vmin) * (X1 - X0)
    o = [f'<svg viewBox="0 0 910 {H:.0f}" width="100%" style="max-width:910px;display:block;margin:0 auto" '
         'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    o.append('<text x="455" y="20" font-size="12.5" font-weight="700" fill="#1c2433" text-anchor="middle">'
             'CASF-2016 leakage gap &mdash; non-train (honest) &rarr; leaky (train-overlap), Spearman &rho;</text>')
    ybot = TOP + len(data) * RH
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        xt = x(t)
        o.append(f'<line x1="{xt:.1f}" y1="{TOP-6:.0f}" x2="{xt:.1f}" y2="{ybot:.0f}" stroke="#eceff3" stroke-width="1"/>')
        o.append(f'<text x="{xt:.1f}" y="{ybot+15:.0f}" font-size="9" fill="#9aa3b2" text-anchor="middle">{t:.1f}</text>')
    for i, (m, nt, lk) in enumerate(data):
        y = TOP + i * RH + RH / 2
        gap = lk - nt
        col = "#d64545" if gap >= 0.10 else ("#9aa3b2" if gap < 0.03 else "#3f7fc4")
        o.append(f'<text x="{X0-10:.0f}" y="{y+3:.0f}" font-size="10.5" fill="#1c2433" text-anchor="end">{m}</text>')
        o.append(f'<line x1="{x(nt):.1f}" y1="{y:.1f}" x2="{x(lk):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2.6" opacity=".85"/>')
        o.append(f'<circle cx="{x(nt):.1f}" cy="{y:.1f}" r="4.3" fill="{col}"/>')
        o.append(f'<circle cx="{x(lk):.1f}" cy="{y:.1f}" r="4.3" fill="#fff" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{x(max(lk,nt))+8:.1f}" y="{y+3:.0f}" font-size="9" fill="{col}" font-weight="600">{gap:+.3f}</text>')
    ly = H - 10
    o.append(f'<circle cx="{X0}" cy="{ly-3}" r="4.3" fill="#5b6678"/>'
             f'<text x="{X0+9}" y="{ly}" font-size="9.5" fill="#5b6678">non-train (honest / OOD)</text>')
    o.append(f'<circle cx="{X0+175}" cy="{ly-3}" r="4.3" fill="#fff" stroke="#5b6678" stroke-width="2"/>'
             f'<text x="{X0+184}" y="{ly}" font-size="9.5" fill="#5b6678">leaky (incl. 90 train-overlap)</text>')
    o.append(f'<text x="{X0+400}" y="{ly}" font-size="9.5" fill="#d64545" font-weight="600">red = big leak (&ge;0.10)</text>')
    o.append("</svg>")
    return "\n      ".join(o)


def build():
    css, table2, sub2, vina_img = extract_from_715()
    table2 = add_atom_column(table2)
    casf_chart = casf_leakage_svg()
    chart = bar.svg()
    legend = bar.legend()

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxBind — Results</title>
{css}
<style>
  /* three distinctly-coloured method tags */
  table.results td.col-method .tag.supervised{{background:#eef1f6;color:#5b6678;}}
  table.results td.col-method .tag.pretrained{{background:#e7eefb;color:#38559b;}}
  table.results td.col-method .tag.zeroshot{{background:#f3eefb;color:#6b4b9b;}}
  .lead{{font-size:14px;color:var(--ink-soft);margin:10px 0 0;max-width:1000px;}}
  table.results thead tr.grp th .nsub{{display:block;font-size:9.5px;font-weight:500;letter-spacing:.02em;
    text-transform:none;color:#9aa3b2;margin-top:2px;}}
  .tbd{{color:#b07a17;font-weight:600;font-size:12px;}}
  .na{{color:#c2c8d2;}}
  table.results .div-block{{border-left:3px solid #1c2433;}}
  table.results thead .div-block{{background:#f4f7fb;}}
  table.results td.metric,table.results th{{padding-left:7px;padding-right:7px;}}
  .metric .val{{font-weight:560;}} .metric .sd{{color:#9aa3b2;font-size:10px;font-weight:400;}}
</style>
</head><body><div class="page">

  <header class="doc">
    <div class="date">2026 · 07 · 16 — Results</div>
    <h1>VoxBind &mdash; Results</h1>
    <p class="lead">Binding-affinity regression and de novo drug design &mdash; headline tables and charts only.</p>
  </header>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">1</span> Binding-affinity regression</h2>
    <p class="section-intro">Method comparison across four progressively-cleaner LP-PDBBind splits
      (<code>lp_edrscc_v2</code> &rarr; <code>+CL1</code> &rarr; <code>+CL1+CL2</code> &rarr; <code>+CL1+CL2+CL3</code>),
      the nested no-leak cleaning tiers (CL3&nbsp;&sub;&nbsp;CL2&nbsp;&sub;&nbsp;CL1) applied to all of train/val/test
      &mdash; each an exact subset of <code>lp_edrscc_v2</code> (Kd/Ki, 3 seeds). Supervised structure baselines +
      <b>CheapNet</b>/<b>BindNet</b> + <b>HonestAffinity</b> (new; ESM-2 leak-aware sequence baseline, arXiv&nbsp;2606.03422)
      vs. the frozen density probe (<b>C</b> coords, <b>C+D+G</b> coords+density+gradmag). 3 seeds, all trained &amp; tested
      per tier on our ED+RSCC data.</p>

    <section class="block">
      <p class="table-title">Figure 1 &nbsp;&middot;&nbsp; Test Pearson r / Spearman &rho; / RMSE &mdash; <code>lp_edrscc_v2</code></p>
      <div class="table-wrap" style="padding:16px 18px 10px;">
      {chart}
      {legend}
      <p class="figure-cap">Headline <code>lp_edrscc_v2</code> chart. Whiskers&nbsp;=&nbsp;&plusmn;1&nbsp;std (3 seeds).
        Per-split (CL) breakdown in Table&nbsp;1. <b>C+D+G</b> (green, bold border) = best.</p>
      </div>
    </section>

    <section class="block">
      <p class="table-title">Table 1 &nbsp;&middot;&nbsp; Test metrics across cleaning tiers &mdash; mean &plusmn; std (3 seeds)</p>
      <p class="table-sub">The four <b>LP-PDBBind</b> groups are the nested no-leak cleaning tiers (CL3&nbsp;&sub;&nbsp;CL2&nbsp;&sub;&nbsp;CL1),
        each an exact subset of <code>lp_edrscc_v2</code> <b>trained &amp; tested on that tier</b> (test N 1320&nbsp;&rarr;&nbsp;733).
        These are sequence-clustered novel-target splits, so this reads generalization to unseen targets &mdash; <b>not</b> a
        same-test comparison across columns. <b>best</b> = best value in each (split&nbsp;&times;&nbsp;metric) column.
        A held-out <b>CASF-2016</b> evaluation of the same methods (canonical vs. leak-controlled) is in <b>Appendix&nbsp;A</b>.</p>
      <div class="table-wrap"><table class="results">
        <thead>
          {affinity_table_head()}
        </thead>
        <tbody>
          {affinity_rows()}
        </tbody>
      </table></div>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">2</span> De novo drug design</h2>
    <p class="section-intro">Conditional de novo ligand generation on the CrossDocked test pockets, scored by
      AutoDock&nbsp;Vina + drug-likeness. <b>Ours</b> = frozen-encoder generator; &sigma;&nbsp;0.9&nbsp;/&nbsp;1.0 =
      reproduced VoxBind; prior-SBDD rows = VoxBind paper (100 pockets).</p>

    <section class="block">
      <p class="table-title">Figure 2 &nbsp;&middot;&nbsp; Vina Score / Min / Dock &mdash; average &amp; median</p>
      <div class="table-wrap" style="padding:16px">{vina_img}</div>
      <p class="figure-cap">AutoDock Vina (kcal/mol, lower better). Source: <code>260715_meeting.html</code> &sect;2.</p>
    </section>

    <section class="block">
      <p class="table-title">Table 2 &nbsp;&middot;&nbsp; De novo generation &mdash; CrossDocked benchmark</p>
      {sub2}
      <div class="table-wrap">{table2}</div>
      <p class="figure-cap"><b># Atoms</b> = heavy-atom count of generated molecules (Avg&nbsp;/&nbsp;Med).
        Baselines from the VoxBind paper (<a href="https://arxiv.org/abs/2405.03961">arXiv:2405.03961</a>, Table&nbsp;1);
        our reproduced rows show &mdash; (the full 79-pocket generation output isn't retained in this checkout &mdash; to be filled).</p>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num" style="background:#566072">A</span> Appendix &mdash; CASF-2016 held-out evaluation</h2>
    <p class="section-intro">The <b>same 12 methods, all trained on the <code>lp_edrscc_v2</code> train set</b>, evaluated on the
      CASF-2016 core set (214 ED-available complexes, all Kd/Ki) &mdash; the HonestAffinity canonical-vs-leak-controlled
      protocol. <b>CASF-2016</b> = all 214 (90 sit in the train &rarr; leaked/inflated); <b>CASF-2016&nbsp;&minus;train</b> =
      the 124 not in the train. Their gap isolates the train-leakage effect.</p>

    <section class="block">
      <p class="table-title">Table A1 &nbsp;&middot;&nbsp; CASF-2016 &mdash; leaky (all 214) vs. non-train (124), sorted by non-train &rho;</p>
      <p class="table-sub">Same trained models as &sect;1; each method predicts all 214, scored twice (all 214 = leaky;
        the 124 held-out = non-train). <b>best</b> = best in each column. <i>BindNet is a single-seed estimate
        (its 3-seed retrain was OOM-unstable); all other cells are 3&nbsp;seeds.</i></p>
      <div class="table-wrap"><table class="results">
        <thead>
          {casf_table_head()}
        </thead>
        <tbody>
          {casf_rows()}
        </tbody>
      </table></div>
    </section>

    <section class="block">
      <p class="table-title">Figure A1 &nbsp;&middot;&nbsp; CASF-2016 leakage gap per method</p>
      <div class="table-wrap" style="padding:16px 18px 12px;">
      {casf_chart}
      <p class="figure-cap">Each method: honest <b>non-train</b> &rho; (filled dot) &rarr; <b>leaky</b> &rho; (open dot, includes
        the 90 train-overlap complexes). Line length = leakage inflation, sorted top-to-bottom.
        <span style="color:#d64545;font-weight:600">Memorization-prone methods</span> (HonestAffinity, CheapNet, AEV-PLIG)
        inflate ~0.15; structure-geometry methods (C, C+D+G, GET, EGNN, ProFSA, DSMBind) stay ~0.07.</p>
      </div>
      <div class="notes">
        <b>Caveat &mdash; "non-train" is not the same as "out-of-distribution."</b> CASF-2016&nbsp;&minus;train removes only
        exact training <i>members</i> (by PDB id), not homologous <i>families</i>: CASF's targets (thrombin, HIV protease,
        carbonic anhydrase, kinases&hellip;) are, by construction, well-represented in PDBbind training, and it has a wider,
        curated affinity range (pKd std&nbsp;1.98 vs&nbsp;1.77). That is why every method scores <i>higher</i> here than on the
        sequence-clustered LP no-leak tiers in &sect;1 &mdash; those are the stricter novel-target test. Difficulty ordering:
        CASF-leaky&nbsp;&gt;&nbsp;CASF-&minus;train&nbsp;&gt;&nbsp;LP no-leak tiers.
      </div>
    </section>
  </div>

</div></body></html>"""
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT}  ({len(html)} bytes)")
    done = sum(1 for m in ORDER for s, _, _ in SPLITS if load(m, s))
    casf = sum(1 for m in ORDER for which, _, _ in CASF_COLS if load_casf(m, which))
    print(f"  filled LP {done}/{len(ORDER) * len(SPLITS)} | CASF {casf}/{len(ORDER) * len(CASF_COLS)}")


if __name__ == "__main__":
    build()
