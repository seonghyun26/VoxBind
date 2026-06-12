#!/usr/bin/env python
"""Render the ViT implementation-level space/time autoresearch into a single HTML report.

Reads the per-component JSONs written by voxbind/test/benchmark_vit_impl.py from
voxbind/log/ and emits a self-contained report (inline CSS + SVG, no external deps) to
notebook/html/260611_vit_impl_autoresearch.html.

    python notebook/make_vit_impl_report.py
"""

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "voxbind/log"
OUT = REPO / "notebook/html/260611_vit_impl_autoresearch.html"
COMPONENTS = ["profile", "roundtrip", "checkpoint", "attn", "patch_embed"]


def load(comp):
    p = LOG / f"260611_vit_impl_{comp}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def esc(s):
    return html.escape(str(s))


# ── tiny inline-SVG helpers ──────────────────────────────────────────────────────
def hbars(rows, *, width=560, rowh=26, pad=4, maxval=None, unit="", fmt="{:.1f}",
          colors=None, labw=150):
    """rows: list of (label, value[, color]). Horizontal bar chart as inline SVG."""
    barw = width - labw - 70
    vals = [r[1] for r in rows]
    mx = maxval if maxval else (max(vals) if vals else 1) or 1
    h = len(rows) * (rowh + pad) + pad
    out = [f'<svg width="{width}" height="{h}" role="img" '
           f'style="font:12px ui-monospace,monospace">']
    for i, r in enumerate(rows):
        lab, val = r[0], r[1]
        col = (r[2] if len(r) > 2 else None) or (colors[i] if colors else "#4f8cff")
        y = pad + i * (rowh + pad)
        w = max(1, int(barw * val / mx))
        out.append(f'<text x="0" y="{y+rowh*0.68}" fill="#cdd6e4">{esc(lab)}</text>')
        out.append(f'<rect x="{labw}" y="{y}" width="{w}" height="{rowh-6}" rx="3" fill="{col}"/>')
        out.append(f'<text x="{labw+w+6}" y="{y+rowh*0.62}" fill="#9fb0c8">'
                   f'{esc(fmt.format(val))}{esc(unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def badge(text, kind):
    c = {"adopt": "#1f9d6b", "free": "#2f7fe0", "space": "#9a6bd4",
         "skip": "#888", "warn": "#c2870a"}.get(kind, "#888")
    return f'<span class="badge" style="background:{c}">{esc(text)}</span>'


def fmt_pct(x, plus=True):
    return f"{x:+.1f}%" if plus else f"{x:.1f}%"


# ── section builders ─────────────────────────────────────────────────────────────
def sec_summary(data):
    rows = []
    pe = data.get("patch_embed")
    if pe:
        r = pe["result"]
        sp = r["fp32_conv"]["ms"] / r["fp32_linear"]["ms"]
        rows.append(("Linear patch-embed", "time", badge("ADOPT", "adopt"),
                     f"{sp:.2f}× fp32 / 2.9× amp; survives compile", "equivalent (gemm ≡ conv)"))
    ck = data.get("checkpoint")
    if ck:
        r = ck["result"]
        rows.append(("Activation checkpointing", "space", badge("ADOPT (opt-in)", "space"),
                     f"−{r['fp32']['mem_cut_pct']:.0f}% peak for +{r['fp32']['time_cost_pct']:.0f}% time",
                     "bit-exact" if r["same_results"] else "CHECK"))
    rt = data.get("roundtrip")
    if rt:
        r = rt["result"]
        rows.append(("Round-trip elimination", "time+space", badge("ADOPT (free)", "free"),
                     f"{r['fp32']['speedup_pct']:+.1f}% step, {r['fp32']['dmem_gb']:+.2f} GB",
                     "bit-exact" if r["same_results"] else "CHECK"))
    at = data.get("attn")
    if at:
        r = at["result"]
        sp = (1 - r["amp_unbind"]["ms"] / r["amp_permute"]["ms"]) * 100
        rows.append(("Attention <code>unbind</code> layout", "time", badge("eager-only", "warn"),
                     f"{sp:.0f}% amp eager; ~0 under compile (§6)", "bit-exact (Δ=0)"))
    trs = "".join(
        f"<tr><td><b>{lab}</b></td><td>{esc(lev)}</td><td>{verd}</td>"
        f"<td>{esc(res)}</td><td>{esc(corr)}</td></tr>"
        for lab, lev, verd, res, corr in rows)
    return f"""
    <h2>Executive summary</h2>
    <p>Four implementation-level levers, each measured in isolation against a shared-init
    reference and gated for correctness. All four preserve results; none touches the
    systems knobs (amp / channels_last / fused AdamW / compile) already tuned in the
    <code>benchmark_vit_opts</code> work.</p>
    <table class="tbl">
      <thead><tr><th>Lever</th><th>Axis</th><th>Verdict</th><th>Effect</th><th>Correctness</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>"""


def sec_profile(data):
    pr = data.get("profile")
    if not pr:
        return "<h2>1 · Profile</h2><p class='muted'>(profile JSON not found)</p>"
    r = pr["result"]
    st = r["stage_ms"]
    order = ["patch_embed", "attn(sum)", "mlp(sum)", "decoder_proj", "tokens->vox",
             "patchify", "recon_mlp", "unpatchify", "norm"]
    col = {"patch_embed": "#e06c4f", "attn(sum)": "#4f8cff", "mlp(sum)": "#6fb0ff",
           "decoder_proj": "#9a6bd4", "tokens->vox": "#c2870a", "patchify": "#c2870a",
           "recon_mlp": "#1f9d6b", "unpatchify": "#888", "norm": "#888"}
    rows = [(k, st[k], col.get(k, "#4f8cff")) for k in order if k in st]
    bars = hbars(rows, unit=" ms", fmt="{:.2f}", width=600)
    act = r["activations_mb"]
    short = {"tokens (B,N,D)": "tokens (B,N,D)",
             "decoder_proj out (B,N,c_half*p^3)": "decoder_proj out",
             "voxel map (B,c_half,G^3)": "voxel map"}
    arows = [(short.get(k, k), v, "#9a6bd4" if v > 100 else "#4f8cff") for k, v in act.items()]
    par = r["params"]
    prow = "".join(f"<tr><td>{esc(k)}</td><td>{v/1e6:.2f} M</td></tr>" for k, v in par.items())
    rt = r["roundtrip_tax_ms"]
    return f"""
    <h2>1 · Where the time &amp; memory go (profile)</h2>
    <p>Single forward, B={pr['meta']['batch_size']}, instrumented per module with CUDA events.
    The transformer blocks dominate (≈68%), but the <b>patch-embed Conv3d is ≈23% of the
    forward on its own</b> — a red flag for a single non-overlapping convolution. Two
    intermediates each weigh <b>537 MB</b> in fp32.</p>
    <div class="grid2">
      <div class="card"><h3>Forward stage time</h3>{bars}</div>
      <div class="card"><h3>Key activation tensors (fp32)</h3>{hbars(arows, unit=' MB', fmt='{:.0f}', width=560, labw=130)}
        <table class="tbl mini"><thead><tr><th>module</th><th>params</th></tr></thead>
        <tbody>{prow}</tbody></table>
      </div>
    </div>
    <p class="note">↳ The <b>tokens→voxel + patchify round-trip tax</b> is {rt:.2f} ms/fwd and a
    537 MB activation that the <code>patch_mlp</code> head immediately undoes (see §3).</p>"""


def sec_patch_embed(data):
    pe = data.get("patch_embed")
    if not pe:
        return ""
    r = pe["result"]
    def row(tag):
        c, cl, l = r[f"{tag}_conv"], r[f"{tag}_conv_cl"], r[f"{tag}_linear"]
        sp = c["ms"] / l["ms"]
        return (f"<tr><td>{tag}</td><td>{c['ms']:.2f}</td><td>{cl['ms']:.2f}</td>"
                f"<td><b>{l['ms']:.2f}</b></td><td><b>{sp:.2f}×</b></td>"
                f"<td>{l['out_maxabs_vs_conv']:.1e}</td></tr>")
    bars = hbars([("Conv3d", r["fp32_conv"]["ms"], "#e06c4f"),
                  ("Conv3d (chan-last)", r["fp32_conv_cl"]["ms"], "#e0905f"),
                  ("Linear(unfold)", r["fp32_linear"]["ms"], "#1f9d6b")],
                 unit=" ms", fmt="{:.1f}", width=560)
    return f"""
    <h2>2 · Patch-embed: Conv3d → Linear(unfold) &nbsp;{badge('biggest time win', 'adopt')}</h2>
    <p>A non-overlapping <code>Conv3d(13→512, k=s=8)</code> is mathematically a single gemm on
    flattened 8³ patches. cuDNN's Conv3d path is ~3× slower than that gemm here. Initializing a
    <code>Linear(13·8³ → 512)</code> from the conv weight (<code>W.reshape(512, 13·512)</code>)
    reproduces the conv output (fp32 Δ=7.6e-5, a reduction-order difference; amp Δ=1.6e-2 is the
    bf16 band).</p>
    <div class="card"><h3>fwd+bwd of the patch-embed alone</h3>{bars}</div>
    <table class="tbl"><thead><tr><th>prec</th><th>Conv3d ms</th><th>Conv chan-last</th>
      <th>Linear ms</th><th>speedup</th><th>Δ vs conv</th></tr></thead>
      <tbody>{row('fp32')}{row('amp')}</tbody></table>
    <p class="note">Trade-off: the Linear path materializes the unfolded patches
    <code>(B, 512, 13·8³)</code> ≈ 436 MB fp32 / 218 MB bf16 (transient, freed after the matmul).
    Net peak impact on the ~11 GB step is small; validate end-to-end before adopting.</p>"""


def sec_roundtrip(data):
    rt = data.get("roundtrip")
    if not rt:
        return ""
    r = rt["result"]
    def row(tag):
        d = r[tag]
        return (f"<tr><td>{tag}</td><td>{d['baseline']['step_ms_mean']:.1f}</td>"
                f"<td>{d['variant']['step_ms_mean']:.1f}</td><td>{d['speedup_pct']:+.1f}%</td>"
                f"<td>{d['baseline']['peak_gb']:.2f}</td><td>{d['variant']['peak_gb']:.2f}</td>"
                f"<td>{d['dmem_gb']:+.2f}</td></tr>")
    c = r["correctness"]
    return f"""
    <h2>3 · Round-trip elimination &nbsp;{badge('free, bit-exact', 'free')}</h2>
    <p>In <code>patch_mlp</code> mode the encoder runs <code>decoder_proj → _tokens_to_voxels</code>
    to build the (B,16,64³) voxel map, then the head immediately calls <code>_patchify</code> to turn
    it back into tokens. <code>_patchify ∘ _tokens_to_voxels</code> is provably the identity
    (verified max|Δ|=0), so both 8D-permute+<code>contiguous()</code> copies (forward and backward) are
    dead work. Feeding <code>decoder_proj</code> tokens straight to <code>recon_mlp</code> removes them.</p>
    <table class="tbl"><thead><tr><th>prec</th><th>base step ms</th><th>var step ms</th><th>Δtime</th>
      <th>base GB</th><th>var GB</th><th>Δmem</th></tr></thead>
      <tbody>{row('fp32')}{row('amp')}</tbody></table>
    <p class="note">Correctness: forward Δ={c['out_maxabs']:.0e}, gradient Δ={c['grad_maxabs']:.0e}
    → <b>bit-exact</b>. Small but truly free (only relevant to the patch_mlp head; structure head
    still needs the voxel map).</p>"""


def sec_checkpoint(data):
    ck = data.get("checkpoint")
    if not ck:
        return ""
    r = ck["result"]
    def row(tag):
        d = r[tag]
        return (f"<tr><td>{tag}</td><td>{d['baseline']['step_ms_mean']:.1f}</td>"
                f"<td>{d['variant']['step_ms_mean']:.1f}</td><td>+{d['time_cost_pct']:.0f}%</td>"
                f"<td>{d['baseline']['peak_gb']:.2f}</td><td>{d['variant']['peak_gb']:.2f}</td>"
                f"<td>−{d['mem_cut_pct']:.0f}%</td></tr>")
    sc = r.get("scaling", {})
    schead = "".join(f"<th>B={b}</th>" for b in sc)
    def scrow(which, col):
        cells = "".join(
            f"<td class='{ 'oom' if sc[b][which]=='OOM' else '' }'>{esc(sc[b][which])}</td>"
            for b in sc)
        return f"<tr><td>{which}</td>{cells}</tr>"
    c = r["correctness"]
    return f"""
    <h2>4 · Activation checkpointing &nbsp;{badge('space lever', 'space')}</h2>
    <p>Recompute each transformer block in the backward pass instead of stashing its internal
    activations (<code>torch.utils.checkpoint</code>, <code>use_reentrant=False</code>). Standard
    compute-for-memory trade; bit-exact with dropout off / RNG preserved.</p>
    <table class="tbl"><thead><tr><th>prec</th><th>base step ms</th><th>ckpt step ms</th><th>Δtime</th>
      <th>base GB</th><th>ckpt GB</th><th>Δpeak</th></tr></thead>
      <tbody>{row('fp32')}{row('amp')}</tbody></table>
    <h3>Batch-size scaling (fp32 peak GB; OOM = did not fit on 24 GB)</h3>
    <table class="tbl"><thead><tr><th></th>{schead}</tr></thead>
      <tbody>{scrow('baseline','#e06c4f')}{scrow('ckpt','#1f9d6b')}</tbody></table>
    <p class="note">Correctness: forward Δ={c['out_maxabs']:.0e}, gradient Δ={c['grad_maxabs']:.0e}
    → bit-exact. Use when memory-bound (larger batch / wider model); skip when compute-bound.</p>"""


def sec_attn(data):
    at = data.get("attn")
    if not at:
        return ""
    r = at["result"]
    def row(tag):
        p_, u_, c_ = r[f"{tag}_permute"], r[f"{tag}_unbind"], r[f"{tag}_chunk"]
        sp = (1 - u_["ms"] / p_["ms"]) * 100
        return (f"<tr><td>{tag}</td><td>{p_['ms']:.2f}</td><td><b>{u_['ms']:.2f}</b></td>"
                f"<td>{c_['ms']:.2f}</td><td><b>{sp:.0f}%</b></td>"
                f"<td>{u_['out_maxabs_vs_permute']:.0e}</td></tr>")
    return f"""
    <h2>5 · Attention q/k/v layout &nbsp;{badge('free, bit-exact', 'free')}</h2>
    <p>The current attention builds q,k,v with <code>reshape(B,N,3,H,hd).permute(2,0,3,1,4)</code>.
    Replacing it with <code>.unbind(2)</code> + <code>transpose(1,2)</code> produces identical q,k,v
    (Δ=0) but a friendlier layout into <code>scaled_dot_product_attention</code>.</p>
    <table class="tbl"><thead><tr><th>prec</th><th>permute ms</th><th>unbind ms</th><th>chunk ms</th>
      <th>unbind speedup</th><th>Δ vs permute</th></tr></thead>
      <tbody>{row('fp32')}{row('amp')}</tbody></table>
    <p class="note">Peak memory identical across layouts — a pure time micro-opt (one line).</p>"""


def sec_compile(data):
    pe_e, pe_c = data.get("patch_embed"), data.get("patch_embed_compiled")
    at_e, at_c = data.get("attn"), data.get("attn_compiled")
    if not (pe_c or at_c):
        return ""
    blocks = []
    if pe_e and pe_c:
        re_, rc = pe_e["result"], pe_c["result"]
        def prow(tag):
            return (f"<tr><td>{tag}</td>"
                    f"<td>{re_[f'{tag}_conv']['ms']:.1f}</td><td>{rc[f'{tag}_conv']['ms']:.1f}</td>"
                    f"<td>{re_[f'{tag}_linear']['ms']:.1f}</td><td><b>{rc[f'{tag}_linear']['ms']:.1f}</b></td>"
                    f"<td><b>{rc[f'{tag}_conv']['ms']/rc[f'{tag}_linear']['ms']:.2f}×</b></td></tr>")
        blocks.append("<h3>patch-embed — win survives (inductor keeps the slow Conv3d)</h3>"
                      "<table class='tbl'><thead><tr><th>prec</th><th>conv eager</th><th>conv comp</th>"
                      "<th>linear eager</th><th>linear comp</th><th>comp speedup</th></tr></thead>"
                      f"<tbody>{prow('fp32')}{prow('amp')}</tbody></table>")
    if at_e and at_c:
        re_, rc = at_e["result"], at_c["result"]
        def arow(tag):
            return (f"<tr><td>{tag}</td>"
                    f"<td>{re_[f'{tag}_permute']['ms']:.2f}</td><td>{re_[f'{tag}_unbind']['ms']:.2f}</td>"
                    f"<td>{rc[f'{tag}_permute']['ms']:.2f}</td><td>{rc[f'{tag}_unbind']['ms']:.2f}</td></tr>")
        blocks.append("<h3>attention — win evaporates (compile normalizes the layout)</h3>"
                      "<table class='tbl'><thead><tr><th>prec</th><th>permute eager</th><th>unbind eager</th>"
                      "<th>permute comp</th><th>unbind comp</th></tr></thead>"
                      f"<tbody>{arow('fp32')}{arow('amp')}</tbody></table>")
    return f"""
    <h2>6 · Does <code>torch.compile</code> change the verdict?</h2>
    <p>Every number above is <b>eager</b> — production runs with <code>compile.enabled: false</code>, so
    eager is today's real baseline. Re-running the two time micro-benches under <code>torch.compile</code>
    (inductor) splits them cleanly:</p>
    {''.join(blocks)}
    <p class="note">Takeaway: the <b>Linear patch-embed win is real code work compile won't do for you</b> —
    inductor leaves the cuDNN Conv3d untouched, so the gap holds (3.4–3.9×). The <b>attention layout
    micro-opt is moot once you compile</b>: all layouts converge and compiled <code>permute</code> already
    matches eager <code>unbind</code>. Adopt patch-embed either way; adopt <code>unbind</code> only if you
    stay eager.</p>"""


def sec_reco(data):
    return f"""
    <h2>7 · Recommendations</h2>
    <ol>
      <li><b>Linear patch-embed</b> — biggest single time win (~3× on a 23%-of-forward stage).
      Swap <code>Conv3d(k=s=p)</code> for <code>_patchify</code>+<code>Linear</code>; the conv weight
      reshapes 1:1 to the linear weight (<code>(512,13,8,8,8) → (512, 13·8³)</code>), so a one-line
      loader shim migrates existing checkpoints. Validate the full-model step time &amp; peak (the
      unfold adds a transient ~436 MB fp32 activation).</li>
      <li><b>Attention <code>unbind</code> layout</b> — one-line, bit-exact, free 5–10% <i>in eager</i>;
      worthless once you <code>torch.compile</code> (§6), so only bother if the run stays eager.</li>
      <li><b>Round-trip elimination</b> — bit-exact, free; add an <code>encode_tokens()</code> path so the
      <code>patch_mlp</code> head skips the voxel round-trip.</li>
      <li><b>Activation checkpointing</b> — opt-in flag for memory-bound runs: −50–60% peak for ~+23%
      time, ~doubles the batch that fits.</li>
    </ol>
    <p>Time levers 1–3 are independent of the already-landed systems knobs and stack with amp /
    channels_last / fused AdamW. Estimated combined forward reduction (isolated, to be confirmed
    end-to-end): patch-embed dominates at ~−15–18%, +~2% each from attn &amp; round-trip.</p>
    <p class="muted">Reproduce: <code>CUDA_VISIBLE_DEVICES=4 python voxbind/test/benchmark_vit_impl.py
    --component profile,roundtrip</code> (and <code>checkpoint</code> / <code>attn</code> /
    <code>patch_embed</code> on GPU 5/6/7).</p>"""


def raw_transcripts(data):
    blocks = []
    for c in COMPONENTS:
        d = data.get(c)
        if not d:
            continue
        txt = esc("\n".join(d.get("report", [])))
        blocks.append(f"<details><summary>{c}</summary><pre>{txt}</pre></details>")
    return "<h2>Raw transcripts</h2>" + "".join(blocks)


def main():
    data = {c: load(c) for c in COMPONENTS}
    for c in ("attn_compiled", "patch_embed_compiled"):   # optional torch.compile runs (§6)
        data[c] = load(c)
    have = [c for c in COMPONENTS if data[c]]
    meta = next((data[c]["meta"] for c in have), {})
    gpu = meta.get("gpu", "?")
    cfg = meta.get("cfg", {})
    cfgline = (f"dim={cfg.get('dim')} depth={cfg.get('depth')} heads={cfg.get('n_heads')} "
               f"n_in={cfg.get('n_in_channels')} patch={cfg.get('patch_size')} "
               f"grid={cfg.get('grid_dim')} head={cfg.get('head_style')}")
    body = "\n".join([
        sec_summary(data), sec_profile(data), sec_patch_embed(data),
        sec_roundtrip(data), sec_checkpoint(data), sec_attn(data),
        sec_compile(data), sec_reco(data), raw_transcripts(data),
    ])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxBind ViT — implementation space/time autoresearch (2026-06-11)</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#cdd6e4;
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:40px 22px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; color:#e9eef6; }}
  h2 {{ font-size:20px; margin:38px 0 10px; color:#e9eef6;
        border-bottom:1px solid #232c3a; padding-bottom:6px; }}
  h3 {{ font-size:14px; margin:6px 0 8px; color:#9fb0c8; font-weight:600; }}
  p {{ margin:8px 0; }}
  code {{ background:#171e29; padding:1px 5px; border-radius:4px;
          font:13px ui-monospace,monospace; color:#e6b673; }}
  pre {{ background:#0a0e14; border:1px solid #1d2633; border-radius:6px; padding:12px;
         overflow:auto; font:12px ui-monospace,monospace; color:#b9c4d4; }}
  .tbl {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13.5px; }}
  .tbl th, .tbl td {{ border:1px solid #232c3a; padding:6px 10px; text-align:right; }}
  .tbl th:first-child, .tbl td:first-child {{ text-align:left; }}
  .tbl thead th {{ background:#151c27; color:#9fb0c8; font-weight:600; }}
  .tbl.mini {{ font-size:12px; margin-top:8px; }}
  td.oom {{ color:#e06c4f; font-weight:600; }}
  .badge {{ display:inline-block; color:#fff; font-size:11px; font-weight:700;
            padding:2px 8px; border-radius:10px; vertical-align:middle; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card {{ background:#11161f; border:1px solid #1d2633; border-radius:8px; padding:14px; }}
  .note {{ background:#10151d; border-left:3px solid #2f7fe0; padding:8px 12px;
           border-radius:4px; font-size:13.5px; color:#aebbcd; }}
  .muted {{ color:#7e8ba0; font-size:13px; }}
  .meta {{ color:#7e8ba0; font-size:13px; margin-bottom:18px; }}
  details {{ margin:8px 0; }} summary {{ cursor:pointer; color:#9fb0c8; }}
  @media (max-width:680px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">
  <h1>VoxBind ViT — implementation-level space/time autoresearch</h1>
  <div class="meta">2026-06-11 · {esc(gpu)} (GPU 4–7) · DensityViTMAE {esc(cfgline)} · B={meta.get('batch_size')}
   · data-free, fwd/bwd/opt isolated from the dataloader</div>
  {body}
</div></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"[saved] {OUT}  ({len(doc)//1024} KB; components: {', '.join(have)})")


if __name__ == "__main__":
    main()
