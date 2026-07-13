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
    ("ChannelViT g[7,4,2] wd0.1","wd 0.1",             "C+D+G",       "weight decay 0.05→0.10",            "knob"),
    ("ChannelViT g[7,4,2] hcs0.15","HCS 0.15",         "C+D+G",       "channel-group dropout p=0.15 (HCS)", "knob"),
    ("ChannelViT g[7,4,2] hcs0.30","HCS 0.30",         "C+D+G",       "channel-group dropout p=0.30 (HCS)", "knob"),
    ("ChannelViT g[7,4,2] dim768","dim768 2×",         "C+D+G",       "dim 512→768, heads 8→12 (97.9M, 2.1×)", "capacity"),
    ("ChannelViT g[7,4,2] atombias","atom-bias",       "C+D+G",       "mask_strategy=atom_biased",          "knob"),
    ("ChannelViT g[7,4,2] mask0.4","mask 0.4",         "C+D+G",       "mask_ratio 0.50→0.40",              "knob"),
    ("ChannelViT g[7,4,2] block4","block 4",           "C+D+G",       "mask block_size 8→4 (1Å holes)",    "knob"),
]
# Reordered 2026-06-25 by expected impact (▶ NEXT = top). Rationale: every completed
# scalar/optimizer/PE knob ties-or-hurts (incl. rope3d 0.610, wd0.1 0.624) — only the
# [7,4,2] grouping beat base. So levers that change the SSL PROBLEM (mask distribution,
# amount, granularity, recon weighting) rank above pure optimizer knobs and the
# predictable-negative [13] control.
PENDING_LABELS = {
    "260624_ar_cvit_atombias":("atom-bias ▶ NEXT", "mask_strategy=atom_biased (mask chemistry-rich voxels)"),
    "260624_ar_cvit_mask04":  ("mask 0.4",         "mask_ratio 0.50→0.40 (bracket optimum low side)"),
    "260624_ar_cvit_dw005":   ("dens-wt 0.05",     "density+gradmag recon wt 0.1→0.05 (trend: less is better)"),
    "260624_ar_cvit_block4":  ("block 4",          "mask block_size 8→4 (finer masking)"),
    "260624_ar_cvit_mask03":  ("mask 0.3",         "mask_ratio 0.50→0.30 (low-side bracket pair)"),
    "260624_ar_cvit_lr2e4":   ("lr 2e-4",          "learning rate 1e-4→2e-4 (prior: neutral)"),
    "260624_ar_cvit_headd3":  ("head-d 3",         "recon head_depth 2→3 (head discarded at probe)"),
    "260624_ar_cvit_lr5e5":   ("lr 5e-5",          "learning rate 1e-4→5e-5 (may undertrain)"),
    "260624_ar_cvit_dw05":    ("dens-wt 0.5",      "recon wt 0.1→0.5 (between two worse points)"),
    "260624_ar_cvit_g13":     ("[13]",             "all channels one group (control; collapses grouping)"),
}

DATA = {}                                                    # (label, split) → dict(rho,rho_sd,r,r_sd,n)


def esc(x): return htmlmod.escape(str(x))


def load_csv():
    out = {}
    if not CSV.exists():
        return out
    for r in csvmod.DictReader(open(CSV, encoding="utf-8")):
        try:
            d = dict(
                rho=float(r["test_rho_mean"]), rho_sd=float(r["test_rho_std"]),
                r=float(r["test_r_mean"]), r_sd=float(r["test_r_std"]), n=int(r["n_test"]))
        except (KeyError, ValueError):
            continue
        try:                                            # RMSE is optional — never drop ρ/r if absent
            d["rmse"], d["rmse_sd"] = float(r["test_rmse_mean"]), float(r["test_rmse_std"])
        except (KeyError, ValueError):
            d["rmse"], d["rmse_sd"] = None, None
        out[(r["label"], r["split"])] = d
    return out


def g(label, split="lp_edrscc"):
    return DATA.get((label, split))


def cell(d, kind="rho", best=False):
    if not d:
        return '<span class="mut">—</span>'
    v, sd = {"rho": ("rho", "rho_sd"), "r": ("r", "r_sd"), "rmse": ("rmse", "rmse_sd")}[kind]
    v, sd = d.get(v), d.get(sd)
    if v is None:
        return '<span class="mut">—</span>'
    c = "best" if best else "num"
    return f'<span class="ms"><span class="{c}">{v:.3f}</span><span class="pm"> ±{("%.3f"%sd).lstrip("0")}</span></span>'


def mcell(d, kind, base_val, best=False, lower_better=False, is_base=False):
    """value ±sd  +  small inline Δ vs the [7,4,1,1] base. `lower_better` flips the
    good/bad coloring (RMSE: a drop is good). base row → 'REF'."""
    html = cell(d, kind, best)
    key = {"rho": "rho", "r": "r", "rmse": "rmse"}[kind]
    v = (d or {}).get(key)
    if v is None:
        return html
    if is_base:
        return html                                  # base row is highlighted instead of showing REF
    if base_val is None:
        return html
    dv = v - base_val
    if abs(dv) <= 0.0015:
        cls = "mut"
    else:
        cls = "pos" if ((dv < 0) if lower_better else (dv > 0)) else "neg"
    dtxt = ("%+.3f" % dv).replace("0.", ".", 1)
    return f'{html}<span class="dlt {cls}">{dtxt}</span>'


# ── Phase 1 + Phase 2 (full-study baselines, both splits) ───────────────────────
def baseline_tables():
    def row(enc, mod, label, best_v2=False, cls=""):
        v2 = g(label, "lp_edrscc")                       # lp_edrscc_v2 only (LP-raw columns removed)
        return (f'<tr class="{cls}"><td style="white-space:normal">{enc}</td><td style="white-space:normal">{mod}</td>'
                f'<td>{cell(v2,"rho",best_v2)}</td><td>{cell(v2,"r")}</td></tr>')

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
          f'<th>ρ (lp_edrscc_v2)</th><th>r</th></tr></thead><tbody>'
          + row("plain ViT (fused)", "C+D+G", "C+D+G")
          + row("ChannelViT [7,4,1,1]", "C+D+G", "C+D+G ChannelViT", best_v2=True, cls="hl")
          + row("ChA-MAEViT", "C+D+G", "C+D+G ChA-MAEViT", cls="mut")
          + '</tbody></table>')
    return p1, p2


# ── Phase 4 — atom-type representation (diverse PLINDER role-split) ──────────────
def phase4_table():
    """Element-channel atomblob vs role-split, and the gain from keeping ALL atom types
    (diverse PLINDER-v2). Rows read by label from clean_results.csv (phase-4 + the
    [7,4,2] reference). atomblob [7,4,2] highlighted as the reference."""
    rows = [
        # (label, representation, atom encoding, groups, ref?, running?)
        ("ChannelViT g[7,4,2]",       "atomblob (element)",          "11 element 1-hot channels",        "[7,4,2]",  True,  False),
        ("roleblob [1,1,1,1]",        "roleblob",                    "2 role ch · 7/4-vocab summed",     "[1,1,1,1]",False, False),
        ("roleblob-div [1,1,1,1]",    "roleblob + diverse atoms",    "2 role ch · ALL atoms (vdW size)", "[1,1,1,1]",False, False),
        ("roleblob-div [1,1,2]",      "roleblob-diverse + grp D+G",  "2 role ch · ALL atoms (vdW size)", "[1,1,2]",  False, False),
        ("roleblob-div [1,1,2] atombias","roleblob-div [1,1,2] + atom-bias","2 role ch · ALL atoms · atom_biased mask","[1,1,2]",False, False),
        ("roleblob-div [1,1,2] v2-113K","roleblob-div [1,1,2] · 6.5× DATA","2 role ch · 113K corpus (vs 17K) · 50ep","[1,1,2]",False, False),
    ]
    body = ""
    for lbl, rep, enc, grp, ref, running in rows:
        d = g(lbl)
        cls = ' class="hl"' if ref else ''
        tag = ' <span class="tagref">ref</span>' if ref else (' <span class="tagrun">running</span>' if running and not d else '')
        body += (f'<tr{cls}><td style="white-space:normal"><b>{esc(rep)}</b>{tag}</td>'
                 f'<td class="small" style="white-space:normal">{esc(enc)}</td>'
                 f'<td class="small">{esc(grp)}</td>'
                 f'<td>{cell(d,"rho", ref)}</td><td>{cell(d,"r")}</td><td>{cell(d,"rmse")}</td></tr>')
    return (f'<table><thead><tr><th>Representation</th><th>Atom encoding</th><th>Groups</th>'
            f'<th>ρ (lp_edrscc_v2)</th><th>r</th><th>RMSE</th></tr></thead><tbody>{body}</tbody></table>')


# ── Phase 5 — pretext objective (SSL signal; frontier follow-up on [7,4,2]) ──────
def phase5_table():
    rows = [
        ("ChannelViT g[7,4,2]",        "masked reconstruction (MAE)", "mask 50% holes → inpaint",                       True,  False),
        ("ChannelViT g[7,4,2] denoise","density denoising AE",        "no mask; density+∇ρ +N(0,1)·0.5 → reconstruct clean", False, False),
    ]
    body = ""
    for lbl, obj, how, ref, running in rows:
        d = g(lbl)
        cls = ' class="hl"' if ref else ''
        tag = ' <span class="tagref">ref</span>' if ref else (' <span class="tagrun">running</span>' if running and not d else '')
        body += (f'<tr{cls}><td style="white-space:normal"><b>{esc(obj)}</b>{tag}</td>'
                 f'<td class="small" style="white-space:normal">{esc(how)}</td>'
                 f'<td>{cell(d,"rho", ref)}</td><td>{cell(d,"r")}</td><td>{cell(d,"rmse")}</td></tr>')
    return (f'<table><thead><tr><th>Objective</th><th>What it reconstructs</th>'
            f'<th>ρ (lp_edrscc_v2)</th><th>r</th><th>RMSE</th></tr></thead><tbody>{body}</tbody></table>')


# ── Phase 6 — seed robustness (is any knob effect above the pretrain-seed noise floor?) ──
def seed_robustness_table():
    """Two pretrain seeds × {base [7,4,2], atom-bias}. Each ρ is already a 3-probe-seed mean;
    this adds a 2nd PRETRAIN seed to expose the pretrain-seed variance — the real noise floor."""
    rows = [
        ("base [7,4,2]",      "ChannelViT g[7,4,2]",          "ChannelViT g[7,4,2] s1base"),
        ("atom-bias [7,4,2]", "ChannelViT g[7,4,2] atombias", "ChannelViT g[7,4,2] s1ab"),
    ]
    body = ""
    for name, lbl42, lbl1 in rows:
        d42, d1 = g(lbl42), g(lbl1)
        spread = (f'<span class="num">{abs(d42["rho"]-d1["rho"]):.3f}</span>'
                  if (d42 and d1) else '<span class="mut">—</span>')
        body += (f'<tr><td style="white-space:normal"><b>{esc(name)}</b></td>'
                 f'<td>{cell(d42,"rho")}</td><td>{cell(d1,"rho")}</td><td>{spread}</td></tr>')
    return (f'<table><thead><tr><th>Condition</th><th>ρ · pretrain seed 42</th>'
            f'<th>ρ · pretrain seed 1</th><th>|spread|</th></tr></thead><tbody>{body}</tbody></table>')


# ── Phase 7 — eval protocol (frozen probe vs end-to-end finetune) ───────────────
def phase7_table():
    rows = [
        ("Frozen probe (head-only)", "encoder frozen → train MLP head",      "ft g742 frozen",     True),
        ("End-to-end full-FT",       "unfreeze encoder, encoder_lr 1e-5",    "ft g742 end-to-end", False),
    ]
    body = ""
    for name, how, lbl, ref in rows:
        d = g(lbl); cls = ' class="hl"' if ref else ''
        tag = ' <span class="tagref">ref</span>' if ref else ''
        body += (f'<tr{cls}><td style="white-space:normal"><b>{esc(name)}</b>{tag}</td>'
                 f'<td class="small" style="white-space:normal">{esc(how)}</td>'
                 f'<td>{cell(d,"rho", ref)}</td><td>{cell(d,"r")}</td><td>{cell(d,"rmse")}</td></tr>')
    return (f'<table><thead><tr><th>Protocol</th><th>What trains</th>'
            f'<th>ρ (lp_edrscc_v2)</th><th>r</th><th>RMSE</th></tr></thead><tbody>{body}</tbody></table>')


# ── Phase 8 — dramatic algorithm trials (new code, each w/ a trial{N}.html report) ──
def phase8_table():
    # (n, name, what changed, clean_results label, report html)
    rows = [
        (1, "Cluster masking",     "few LARGE contiguous atom-anchored holes (mask_strategy=cluster)",      "cluster mask [7,4,2]", "trial1.html"),
        (2, "Cross-modal masking", "drop ALL density+gradmag, predict from atoms (modal_mask_prob=0.5)",    "crossmodal [7,4,2]",   "trial2.html"),
        (3, "Ligand masking",      "mask the LIGAND, reconstruct from pocket (mask_strategy=ligand)",       "ligand mask [7,4,2]",  "trial3.html"),
        (4, "Interface masking",   "mask the ligand-pocket CONTACT, reconstruct it (mask_strategy=interface)", "interface mask [7,4,2]", "trial4.html"),
        (5, "Finer patches (1 Å)",  "patch_size 8→4: 2× finer patches, 12K ChannelViT tokens (spatial-resolution axis)", "patch4 [7,4,2]", "trial5.html"),
        (6, "Contrastive auxiliary","SimCLR InfoNCE on two MAE-masked views + MAE (contrastive_weight=0.05)", "contrastive [7,4,2]",  "trial6.html"),
        (7, "Radius-embed atoms",   "learned φ(vdW-radius)→k per role replaces one-hot atom channels (radius_embed_k=4)", "radius-embed [7,4,2]", "trial7.html"),
        (8, "Distance-cond. atoms", "radius-embed k=8 with a LEARNED radial profile (multi-scale blur mixture, scales [0,2,4])", "radius-embed-dist [7,4,2]", "trial8.html"),
        (9, "Stochastic Depth",     "CV/DeiT: drop whole residual branches in training, ramped 0→0.1 over depth (drop_path=0.1)", "droppath [7,4,2]", "trial9.html"),
        (10, "data2vec latent MAE", "NLP/CV: + predict EMA-teacher per-patch latent at masked patches (data2vec_weight=0.1)", "data2vec [7,4,2]", "trial10.html"),
    ]
    body = ""
    for n, name, what, lbl, html in rows:
        d = g(lbl)
        if d:
            dv = d["rho"] - 0.637
            cls = "pos" if dv > 0.0015 else ("neg" if dv < -0.0015 else "mut")
            rho = cell(d, "rho"); delta = f'<span class="{cls}">{dv:+.3f}</span>'
        else:
            rho = '<span class="tagrun">running</span>'; delta = '<span class="mut">—</span>'
        body += (f'<tr><td style="white-space:nowrap"><b>trial {n}</b> · <a href="{html}">report&nbsp;↗</a></td>'
                 f'<td style="white-space:normal"><b>{esc(name)}</b></td>'
                 f'<td class="small" style="white-space:normal">{esc(what)}</td>'
                 f'<td>{rho}</td><td>{delta}</td></tr>')
    return (f'<table><thead><tr><th>Trial</th><th>Algorithm</th><th>What changed</th>'
            f'<th>ρ (lp_edrscc_v2)</th><th>Δ vs 0.637</th></tr></thead><tbody>{body}</tbody></table>')


# ── Phase 3 autoresearch progression chart ──────────────────────────────────────
def svg_scatter(rows, axis_label, lower_better=False):
    rows = [(i, n, v) for i, n, v in rows if v is not None]
    if not rows:
        return "<p class='mut'>No values yet.</p>"
    n = len(rows)
    vals = [v for _, _, v in rows]
    y0 = math.floor((min(vals) - 0.01) * 100) / 100
    y1 = math.ceil((max(vals) + 0.01) * 100) / 100
    tick = next((t for t in (0.01, 0.02, 0.05, 0.1, 0.2) if (y1 - y0) / t <= 12), 0.2)
    W, H, padL, padR, padT, padB = 880, 320, 56, 20, 16, 40
    def Y(v): return padT + (1 - (v - y0) / (y1 - y0)) * (H - padT - padB)
    def X(i): return padL + (0.5 if n == 1 else i / (n - 1)) * (W - padL - padR)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:900px" font-family="-apple-system,Segoe UI,sans-serif">']
    gg = y0
    while gg <= y1 + 1e-9:
        yy = Y(gg)
        p.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eef1f5"/>')
        p.append(f'<text x="{padL-7}" y="{yy+3:.1f}" font-size="10" fill="#5b6678" text-anchor="end">{gg:.2f}</text>')
        gg += tick
    for k, (i, name, _v) in enumerate(rows):
        p.append(f'<text x="{X(k):.1f}" y="{H-padB+18:.1f}" font-size="10.5" fill="#1c2433" text-anchor="middle">T{k+1}</text>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-4}" font-size="10.5" fill="#5b6678" text-anchor="middle">trial (campaign order)</text>')
    pareto, best, front = set(), (1e9 if lower_better else -1e9), []
    for k, (i, name, v) in enumerate(rows):
        improved = (v < best - 1e-9) if lower_better else (v > best + 1e-9)
        if improved:
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
    base_d = g(BASE_LABEL) or {}
    base, base_r, base_rmse = base_d.get("rho"), base_d.get("r"), base_d.get("rmse")
    base_txt = f"{base:.3f}" if base is not None else "—"
    _, p2 = baseline_tables()

    # Phase-3 sweep
    sp_rows, pe_rows, rmse_rows, trows = [], [], [], ""
    completed = [(lbl, nm, inp, det) for lbl, nm, inp, det, fam in TRIALS if g(lbl)]
    best_rho = max((g(lbl)["rho"] for lbl, *_ in completed), default=None)
    best_r = max((g(lbl)["r"] for lbl, *_ in completed), default=None)
    rmse_vals = [g(lbl)["rmse"] for lbl, *_ in completed if g(lbl).get("rmse") is not None]
    best_rmse = min(rmse_vals) if rmse_vals else None
    for idx, (lbl, nm, inp, det) in enumerate(completed):
        d = g(lbl)
        sp_rows.append((idx, nm, d["rho"])); pe_rows.append((idx, nm, d["r"]))
        rmse_rows.append((idx, nm, d.get("rmse")))
        star = " ⭐" if d["rho"] == best_rho else ""
        is_base = lbl == BASE_LABEL
        rowcls = ' class="ref"' if is_base else ''
        is_best_rmse = best_rmse is not None and d.get("rmse") == best_rmse
        trows += (f'<tr{rowcls}><td class="num" style="white-space:normal"><span class="pm">T{idx+1}</span> {esc(nm)}{star}</td>'
                  f'<td style="white-space:normal">{esc(inp)}</td>'
                  f'<td class="small" style="white-space:normal">{esc(det)}</td>'
                  f'<td>{mcell(d,"rho",base,best=d["rho"]==best_rho,is_base=is_base)}</td>'
                  f'<td>{mcell(d,"r",base_r,best=best_r is not None and d.get("r")==best_r,is_base=is_base)}</td>'
                  f'<td>{mcell(d,"rmse",base_rmse,best=is_best_rmse,lower_better=True,is_base=is_base)}</td></tr>')
    best_txt = f"{best_rho:.3f}" if best_rho is not None else "—"
    best_rmse_txt = f"{best_rmse:.3f}" if best_rmse is not None else "—"

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
.dlt{{display:block;font-size:10px;font-weight:700;margin-top:2px}}
.ms{{white-space:nowrap}}
tr.ref td{{background:#eef2f7}}
.tagref{{font-size:10px;font-weight:700;color:#1d4ed8;background:#e8effe;padding:1px 6px;border-radius:999px}}
.tagrun{{font-size:10px;font-weight:700;color:#b07a17;background:#fcf3e0;padding:1px 6px;border-radius:999px}}
a{{color:#1d4ed8;text-decoration:none;font-weight:600}}a:hover{{text-decoration:underline}}
.best{{font-weight:700;color:#1d5a3a;background:#eaf5ee;border-radius:5px;padding:1px 5px}}
tr.hl td{{background:#f3f8ff}}
tr.delta td{{font-size:12.5px;color:#5b6678;background:#fcfdfe;font-style:italic}}
.note{{font-size:12.5px;color:#5b6678;line-height:1.6}}
</style></head><body><div class="page">
<header><div class="date">VoxBind · PLINDER-OTF · full study</div>
<h1>Electron density: input modality, encoder architecture &amp; autoresearch</h1>
<div class="sub">Frozen-encoder PDBbind affinity probe (pK). All encoders self-supervised on the <b>identical</b>
PLINDER ligand-matched corpus (17,430/100, on-the-fly density resample-aug, 100 ep, eff-batch 128, uniform 50% mask),
differing only in encoder architecture (Ph.2), grouping/knob (Ph.3), or atom-type representation (Ph.4). Metrics =
test Spearman ρ / Pearson r, mean ± std over 3 probe seeds. Built {ts} from <code>clean_results.csv</code>.
Eval split <b>lp_edrscc_v2</b> = LP-PDBBind ∩ ED ∩ RSCC≥0.8, Kd/Ki-only, 3850/817/1320.</div></header>

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

<div class="card"><h2>Phase 3 — test RMSE (pK) by trial</h2>
<p class="csub">Affinity-error on lp_edrscc_v2 in pK units; <b>lower is better</b>, so gold = best-so-far running <i>minimum</i>. 3 probe seeds. <b>Best so far: RMSE {best_rmse_txt}</b>.</p>
{svg_scatter(rmse_rows, 'test RMSE (pK)', lower_better=True)}
<div class="legend">{legend}</div></div>

<div class="card"><h2>Phase 3 — trial table (completed)</h2>
<p class="csub">Small Δ next to each value = vs the <b>[7,4,1,1] base</b> (T2, <span style="background:#eef2f7;padding:0 4px;border-radius:3px">highlighted</span>); <span class="pos">green</span> = better, <span class="neg">red</span> = worse (RMSE: lower is better).</p>
<table><thead><tr><th>Trial</th><th>Input</th><th>What changed</th><th>test ρ</th><th>test r</th><th>test RMSE</th></tr></thead>
<tbody>{trows}</tbody></table>
<div class="note" style="margin-top:10px"><b>Phase 3 concluded</b> — the ChannelViT knob/grouping sweep is closed: nothing beat [7,4,2] 0.637
(atom-bias 0.641 within noise), and the masking-distribution axis (mask 0.4, block 4) hurt. Remaining knobs
(lr/wd/head-d/[13]/dw0.5) deprioritized. <b>A 2nd-pretrain-seed re-run of atom-bias is in flight</b> to test
whether the 0.641 replicates or was seed luck. The loop also explored representation (Phase 4) and pretext (Phase 5).</div></div>

<div class="card"><h2>Phase 4 — atom-type representation (diverse PLINDER-v2)</h2>
<p class="csub">Does keeping <b>all</b> heavy-atom types (metals/Se/halogens that the 7-lig/4-poc element vocab drops) help?
<b>roleblob</b> = 2 role channels [lig, poc] with element identity carried by vdW blob <i>size</i>; <b>diverse</b> =
the plinder-v2 build that admits every element. Reference = the element-channel atomblob winner [7,4,2].</p>
{phase4_table()}
<div class="note" style="margin-top:10px"><b>Diverse atoms help the role rep</b> (+0.031: 0.554→0.585) — the dropped metals/halogens carry real
signal. <b>The D+G-grouping win transfers</b>: [1,1,2] = 0.592 (+0.007 over [1,1,1,1] — the <i>same</i> +0.007 it gave
the atomblob), so grouping density+gradmag is a representation-agnostic improvement. But <b>atom-bias does NOT
transfer</b> — it <i>hurt</i> the role rep (−0.031 → 0.561): concentrating the mask on the few atom blobs destroys
too much in a 2-channel rep (it only helped the redundant 11-channel atomblob). <b>Arm concluded:</b> role-split
caps at <b>[1,1,2] 0.592 &lt; atomblob 0.637</b> — the gap is element-identity that role+size can't recover, not a
tuning deficit. The on-theme frontier test (diverse atoms as <i>expanded element channels</i>) needs a vocab
rebuild (element vocab is hardcoded 7-lig/4-poc) — flagged for approval, not a config override.
<br><br><b>Data-scale verdict: 6.5× more pretrain data does NOT help.</b> Re-running the best role config [1,1,2] on
the <b>full PLINDER-v2 corpus (113K complexes, 6.5× the 17.5K)</b> gives ρ <b>0.586</b> — flat vs the 17K run's 0.592
(Δ −0.006, within seed noise), despite ~3.3× more total training samples. <b>Data quantity is not the bottleneck</b> —
the plateau is representation/capacity-bound, not data-starved. This predicts the pending element-channel [8,4,2]-on-v2
run will win (if at all) from its <i>per-element representation</i>, not from the larger corpus.</div></div>

<div class="card"><h2>Phase 5 — pretext objective (SSL signal)</h2>
<p class="csub">The whole campaign used masked-reconstruction MAE. Does a different self-supervised <b>objective</b> read
density better? <b>Denoising</b> = no masking; corrupt the density+‖∇ρ‖ channels with Gaussian noise and reconstruct
the <i>clean</i> field — a density-denoising autoencoder that directly targets density-reading. Frontier follow-up
on the [7,4,2] winner.</p>
{phase5_table()}
<div class="note" style="margin-top:10px"><b>Denoising HURT</b> (0.587, −0.050). Same lesson as masking: the
masked-reconstruction task forces the encoder to <i>infer missing structure from context</i> (harder, more
semantic → better features), whereas denoise-in-place is a local low-level task. <b>MAE wins; the pretext axis is
settled.</b> (electra — the other pretext — isn't applicable to 13-channel C+D+G.)</div></div>

<div class="card"><h2>Phase 6 — seed robustness (the noise floor)</h2>
<p class="csub">Every ρ above is a 3-<i>probe</i>-seed mean from a single <i>pretrain</i> run. But how much does the
<b>pretrain</b> seed itself move ρ? This re-runs two conditions with a 2nd pretrain seed.</p>
{seed_robustness_table()}
<div class="note" style="margin-top:10px"><b>The apparent winner was seed luck.</b> Across two pretrain seeds the
<b>two-seed means are base 0.633 ≥ atom-bias 0.630</b> — atom-bias is <i>not</i> better, and it's far less stable
(spread 0.022 vs the base's 0.007). Its 0.641 was a single-seed high; seed 1 (0.619) is <i>below</i> the base. So
<b>nothing reliably beats [7,4,2] ≈ 0.63</b>. The honest reading of the whole campaign: most Phase-3 trial-to-trial
differences sit <i>within</i> this pretrain-seed noise; only the large effects exceed it — grouping over coords
(real gain), and the clear regressions (capacity, denoise, block 4, role-rep). <b>Config-overridable design space
exhausted.</b></div></div>

<div class="card"><h2>Phase 7 — eval protocol (frozen probe vs end-to-end finetune)</h2>
<p class="csub">Is ρ≈0.63 just the <i>frozen-probe</i> ceiling? Unfreeze the [7,4,2] encoder and finetune it end-to-end
on lp_edrscc_v2 (<code>ft_scope=full</code>, encoder_lr 1e-5), vs the frozen head-only probe. Both 3 seeds.</p>
{phase7_table()}
<div class="note" style="margin-top:10px"><b>End-to-end finetuning HURTS</b> (0.605, −0.032, and worst RMSE) — and it's
rock-stable (±.001), so not noise. Unfreezing a 46.7M encoder on only 3,850 train complexes <b>overfits</b>; the
frozen pretrained features are <i>better</i>. So <b>0.637 is a genuine representation-quality ceiling, not a
frozen-probe artifact</b> — finetuning does not unlock more. (A gentler protocol — partial <code>lastk</code> FT or
lower encoder_lr — might avoid the overfit, but naive full-FT clearly loses.)</div></div>

<div class="card"><h2>Phase 8 — dramatic algorithm trials</h2>
<p class="csub">Pivot from minor-knob tweaks to <b>new algorithms</b>, each implemented as clean opt-in code (no-op default,
non-destructive, like atom-bias) with its own implementation report. Off the [7,4,2] winner; <b>↗ links open the per-trial page</b>.</p>
{phase8_table()}
<div class="note" style="margin-top:10px"><b>trial 1</b> cluster masking <b>hurt</b> (0.609, −0.028 — large contiguous holes
remove too much sparse local signal). <b>trial 2</b> cross-modal masking <b>tied</b> (0.635 — predicting density from
atoms adds no affinity signal). <b>trial 3</b> ligand masking (mask the ligand, predict it from the pocket — the
de-novo task itself) <b>running</b>. So far dramatic masking/objective changes confirm the plateau; the most
on-theme binding-aware ideas <b>trial 3 (ligand 0.621) and trial 4 (interface 0.610) both HURT</b>. <b>Verdict:
every targeted/concentrated masking hurts</b> (cluster −0.028, interface −0.027, ligand −0.016); uniform scattered
masking is optimal. <b>Masking strategy is not the lever</b> — the plateau is robust to it.
<br><br><b>trial 6 (contrastive auxiliary) badly HURT — ρ 0.499, the WORST result in the campaign</b> (−0.138). A SimCLR
InfoNCE loss alongside MAE (two MAE-masked views pulled together, different complexes pushed apart) degraded val too
(best_val 0.36 vs 0.53) — the features are worse, not mis-calibrated. <b>Instance discrimination is anti-affinity</b>:
it pushes every different complex apart <i>equally</i>, discarding the graded similarity structure ρ needs (similar
binders should sit close, not far). This was the last genuinely-different <i>objective</i>; both non-MAE objectives
tried regress (denoise −0.050, contrastive −0.138), so <b>reconstruction is actively protected</b> and the plateau is
firmly <b>data/probe-ceiling bound, not objective-bound</b> (the flat 6.5×-data result, Phase 4, agrees).
<br><br><b>trial 7 (learnable radius-embed atoms) — the best dramatic trial, ρ 0.624.</b> Replacing one-hot element
channels with a learned <code>φ(vdW-radius)→4</code> per role (a continuous, compressed atom featurization)
<b>beats roleblob 0.592 by +0.032</b> (it recovers the element separation role-occupancy sums away) and comes within
<b>−0.013 of one-hot atomblob 0.637</b> — a near-lossless substitute, not an improvement. The small gap is exactly the
predicted cost: ligand S and P share vdW 1.8 Å, so radius cannot separate them. <b>So even a learned atom featurization
sits at the plateau</b>; representation richness isn't the missing lever either.</div></div>

<div class="card"><h2>Reading it</h2>
<div class="note">
<b>Density</b> lifts ChannelViT over coords (best {best_txt} vs coords-ChannelViT 0.604). The winning structure is
<b>group density+gradmag together, keep lig/poc separate</b> ([7,4,2] &gt; [11,2] &gt; [7,4,1,1] &gt; [11,1,1]).<br><br>
<b>The knob space is essentially flat.</b> Across every trial, only grouping ever moved ρ. <b>atom-bias</b> (mask
chemistry-rich voxels) is the lone nominal beat at ρ 0.641, but its ±.006 overlaps the [7,4,2] base (0.637) and
its RMSE is <i>worse</i> (1.385 vs 1.353) — better ranking, worse calibration, i.e. within noise. Mask ratio
(0.6/0.75), recon weight (0.3 tied, 1.0 worse), rope3d (0.610), wd0.1 (0.624) and HCS channel-group dropout
(0.637/0.633) all tie-or-hurt.<br><br>
<b>Capacity is NOT the bottleneck.</b> Doubling the encoder (dim768, 97.9M, 2.1×) <i>regressed</i> to ρ 0.626 (−0.011,
worst RMSE) — a bigger model just overfits the frozen probe. So the plateau is <b>data scale / probe-ceiling</b>
bound, not model size or pretext recipe. The masking-distribution axis is now closed too (mask 0.4 / block 4 hurt).<br><br>
<b>Open thread — atom-type representation (Phase 4).</b> Diverse PLINDER-v2 atoms help the role rep (+0.031) but
role-split trails the element atomblob; testing whether the D+G-grouping win lifts it.<br><br>
<b>Reference:</b> TargetDiff EGNN (supervised) scores ρ 0.536 on its own LP-PDBBind split — a different, larger test set, not directly comparable.
</div></div>
</div></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({len(OUT.read_text()):,} bytes, static SVG, no JS)")
