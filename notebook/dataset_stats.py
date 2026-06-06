"""dataset_stats.py — generate notebook/html/dataset_stats.html

Self-contained report of the PDBbind density normalisation variants v1–v5:
characteristics table + density-distribution plots (rendered from the actual
on-disk crops in voxbind/dataset/data/pdbbind/voxels{,_v2,_v3,_v4,_v5}).

    cd voxbind && python ../notebook/dataset_stats.py
"""
import base64
import io
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PB = Path("/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data/pdbbind")
OUT = Path("/home/shpark/prj-denovo/VoxBind/notebook/html/dataset_stats.html")
VERS = ["v1", "v2", "v3", "v4", "v5"]
VOX = {"v1": PB / "voxels", "v2": PB / "voxels_v2", "v3": PB / "voxels_v3",
       "v4": PB / "voxels_v4", "v5": PB / "voxels_v5"}
COLOR = {"v1": "#3498db", "v2": "#e67e22", "v3": "#2ecc71", "v4": "#9b59b6", "v5": "#e74c3c"}
LABEL = {
    "v1": "v1 · per-map z + per-crop ±3σ clip",
    "v2": "v2 · pocket-pool z-score",
    "v3": "v3 · pocket-pool max-abs",
    "v4": "v4 · pocket-pool clip + z-score",
    "v5": "v5 · pocket-pool arcsinh + z-score",
}
N_SAMPLE = 200

# ── load sample ──────────────────────────────────────────────────────────────
rng = random.Random(0)
pidsets = {v: {p.stem for p in (d / "density").glob("*.npy")} for v, d in VOX.items()}
counts = {v: len(s) for v, s in pidsets.items()}
shared = sorted(set.intersection(*pidsets.values()))
sample = rng.sample(shared, min(N_SAMPLE, len(shared)))

def load(v, pid):
    return np.load(VOX[v] / "density" / f"{pid}.npy").astype(np.float32)

pool = {v: np.concatenate([load(v, p).ravel() for p in sample]) for v in VERS}
pstat = {v: dict(min=float(pool[v].min()), max=float(pool[v].max()),
                 mean=float(pool[v].mean()), std=float(pool[v].std())) for v in VERS}

stats = {}
for v in ("v2", "v3", "v4", "v5"):
    sj = VOX[v] / "stats.json"
    stats[v] = json.loads(sj.read_text()) if sj.exists() else {}

def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── Fig 1: pool histograms ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 3.8))
for ax, v in zip(axes, VERS):
    vals = pool[v]
    ax.hist(vals, bins=180, color=COLOR[v], edgecolor="black", linewidth=0.15, alpha=0.9)
    ax.set_yscale("log"); ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"{LABEL[v]}\n[{vals.min():+.2f}, {vals.max():+.2f}]  μ={vals.mean():+.3f}  σ={vals.std():.3f}",
                 fontsize=9)
    ax.set_xlabel("density value"); ax.grid(alpha=0.25)
axes[0].set_ylabel("voxel count (log)")
fig.tight_layout()
img_pool = b64(fig)

# ── Fig 2: per-crop summary stats ────────────────────────────────────────────
pcs = {v: {k: [] for k in ("mean", "std", "min", "max")} for v in VERS}
for v in VERS:
    for p in sample:
        a = load(v, p)
        pcs[v]["mean"].append(float(a.mean())); pcs[v]["std"].append(float(a.std()))
        pcs[v]["min"].append(float(a.min())); pcs[v]["max"].append(float(a.max()))
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, key in zip(axes, ("mean", "std", "min", "max")):
    for v in VERS:
        ax.hist(pcs[v][key], bins=36, alpha=0.5, color=COLOR[v], label=v,
                edgecolor="black", linewidth=0.2)
    ax.set_title(f"per-crop {key}", fontsize=11); ax.set_xlabel(key)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
axes[0].set_ylabel("# crops")
fig.tight_layout()
img_pcs = b64(fig)

# ── Fig 3: same-pocket slices ────────────────────────────────────────────────
PID = "10gs" if "10gs" in shared else sample[0]
crops = {v: load(v, PID) for v in VERS}
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), constrained_layout=True)
for ax, v in zip(axes, VERS):
    a = crops[v]; sl = a[:, :, a.shape[-1] // 2]
    vmax = max(abs(a.min()), abs(a.max())) or 1.0
    im = ax.imshow(sl.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{LABEL[v]}\n[{a.min():+.2f}, {a.max():+.2f}]", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
img_slice = b64(fig)

# ── derived numbers ──────────────────────────────────────────────────────────
v2_sig = stats["v2"].get("sigma", float("nan"))
v3_max = stats["v3"].get("max_abs", float("nan"))
ratio = v2_sig / v3_max if v3_max else float("nan")
v4_lo = stats["v4"].get("clip_lo_raw", -0.578); v4_hi = stats["v4"].get("clip_hi_raw", 1.647)
v4_mu = stats["v4"].get("mu_clip", float("nan")); v4_sg = stats["v4"].get("sigma_clip", float("nan"))
v5_s = stats["v5"].get("arcsinh_scale", 0.5)
v5_mua = stats["v5"].get("mu_a", float("nan")); v5_sga = stats["v5"].get("sigma_a", float("nan"))

# ── characteristics table rows ───────────────────────────────────────────────
CHARS = {
    "v1": dict(formula="z-score per map → clip ±3σ → re-z-score", scope="per-crop", linear="no (clip)",
               rng="[−3, +3] hard", pcstd="forced ≈ 1",
               note="Only non-global scheme. Same raw value maps differently per pocket → destroys cross-pocket magnitude; peaks pile up at ±3."),
    "v2": dict(formula="(x − μ_pool) / σ_pool", scope="pool-global", linear="yes",
               rng=f"[{pstat['v2']['min']:+.1f}, {pstat['v2']['max']:+.1f}] (tail)", pcstd="varies (real signal)",
               note=f"μ_pool≈0, σ_pool={v2_sig:.4f}. Unit-scale; keeps the long positive heavy-atom/metal tail."),
    "v3": dict(formula="x / max_abs_pool", scope="pool-global", linear="yes",
               rng="[−1, +1] strict", pcstd="varies, ≈ 0.011",
               note=f"max_abs={v3_max:.2f}. <b>Identical to v2 up to a global ×{ratio:.4f} scalar</b> — same shape, ~92× smaller."),
    "v4": dict(formula="(clip(x,[{:.3f},{:.3f}]) − μ_clip) / σ_clip".format(v4_lo, v4_hi),
               scope="pool-global", linear="no (clip)",
               rng=f"[{pstat['v4']['min']:+.2f}, {pstat['v4']['max']:+.2f}]", pcstd="varies (real signal)",
               note=f"μ_clip={v4_mu:.4f}, σ_clip={v4_sg:.3f}. Changes the global <i>shape</i>: truncates the tail, then unit-scales."),
    "v5": dict(formula="(arcsinh(x / {:g}) − μ_a) / σ_a".format(v5_s),
               scope="pool-global", linear="no (arcsinh)",
               rng=f"[{pstat['v5']['min']:+.2f}, {pstat['v5']['max']:+.2f}]", pcstd="varies (real signal)",
               note=f"Soft-squash, <b>no clip</b>: arcsinh(x/{v5_s:g}) compresses the tail smoothly — the +30 raw peak maps to ≈+{pstat['v5']['max']:.0f} (vs +92 for v2), with no pile-up spike and every voxel kept. Bulk preserved; unit-ish scale."),
}

PROBE = [  # condition, ρ, note
    ("atomblob_weighted (atoms only)", "0.473", "best overall — no density"),
    ("atomblob (atoms only)", "0.421", "strong atom baseline"),
    ("merged_density · v3 (density muted)", "0.422", "≈ atom baseline"),
    ("merged_density · v2 (density loud)", "0.342", "fitting density hurts"),
    ("merged_density · v1", "0.296", "per-crop info loss"),
    ("merged_density · v4 dual-head (loudest)", "0.261", "dedicated density head → worst"),
    ("merged_density · v5 (arcsinh soft-squash)", "— not trained", "predicted ≈ v2 (~0.34)"),
]

def row_cells(*xs):
    return "".join(f"<td>{x}</td>" for x in xs)

char_rows = ""
for v in VERS:
    c = CHARS[v]
    char_rows += (
        f'<tr style="border-left:6px solid {COLOR[v]}">'
        f'<td><b>{v}</b></td><td class="mono">{c["formula"]}</td><td>{c["scope"]}</td>'
        f'<td>{c["linear"]}</td><td class="mono">{c["rng"]}</td><td>{c["pcstd"]}</td>'
        f'<td class="note">{c["note"]}</td></tr>\n'
    )

pool_rows = "".join(
    f'<tr style="border-left:6px solid {COLOR[v]}"><td><b>{v}</b></td>'
    f'<td class="mono">[{pstat[v]["min"]:+.3f}, {pstat[v]["max"]:+.3f}]</td>'
    f'<td class="mono">{pstat[v]["mean"]:+.4f}</td><td class="mono">{pstat[v]["std"]:.4f}</td></tr>'
    for v in VERS
)

probe_rows = "".join(
    f'<tr><td>{c}</td><td class="mono"><b>{r}</b></td><td class="note">{n}</td></tr>'
    for c, r, n in PROBE
)

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDBbind Density Normalisation — Dataset Stats (v1–v5)</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#666; --line:#e3e3e3; --bg:#fafafa; --card:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); margin:0; line-height:1.55; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:36px 28px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }}
  h2 {{ font-size:18px; margin:40px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
  .sub {{ color:var(--muted); font-size:14px; margin-bottom:8px; }}
  .tldr {{ background:#fff8e6; border:1px solid #f0d98a; border-radius:10px; padding:16px 20px; margin:22px 0; font-size:14.5px; }}
  .tldr b {{ color:#8a6d00; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:13.5px;
           border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ background:#f2f3f5; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#555; }}
  tr:last-child td {{ border-bottom:none; }}
  .mono {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:12.5px; }}
  .note {{ color:var(--muted); font-size:12.5px; }}
  figure {{ margin:18px 0 6px; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }}
  figcaption {{ color:var(--muted); font-size:12.5px; margin-top:6px; }}
  .chips {{ display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 0; font-size:12.5px; }}
  .chip {{ display:inline-flex; align-items:center; gap:6px; }}
  .sw {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  footer {{ margin-top:50px; color:#999; font-size:12px; border-top:1px solid var(--line); padding-top:14px; }}
  code {{ background:#f0f0f3; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style></head><body><div class="wrap">

<h1>PDBbind Density Normalisation — Dataset Statistics</h1>
<div class="sub">Five normalisation variants of the experimental 2Fo-Fc pocket density crops · 64³ @ 0.25 Å ·
<b>{len(shared):,} data points</b> (complexes; all five versions = {counts['v1']:,} crops each). Distribution
plots below are computed over a random <b>{N_SAMPLE}-crop sample</b> = {N_SAMPLE * 64**3:,} voxels per version.</div>
<div class="chips">
  {''.join(f'<span class="chip"><span class="sw" style="background:{COLOR[v]}"></span>{LABEL[v]}</span>' for v in VERS)}
</div>

<div class="tldr">
<b>Key point.</b> v1 is the only <i>per-crop</i> scheme — it discards cross-pocket density magnitude and
clips peaks. v2–v5 are <i>pool-global</i>; <b>v2 and v3 are the same linear map up to a global ×{ratio:.4f}
scalar</b> (v3 is just v2 shrunk ~{1/ratio:.0f}×), v4 hard-clips the tail then z-scores, and v5 instead
<b>soft-squashes</b> it with arcsinh (no clip → no pile-up spike). Because the MAE reconstructs the density
channel with plain MSE and the on-disk scale is fed unchanged, <b>each scheme's scale silently sets how heavily
density is weighted in the loss</b> (∝ scale²) — which drives the downstream ranking below.
</div>

<h2>1 · The five schemes</h2>
<table>
<tr><th>ver</th><th>formula</th><th>scope</th><th>linear</th><th>output range</th><th>per-crop std</th><th>characteristic</th></tr>
{char_rows}
</table>

<h2>2 · Pool-wide density distribution</h2>
<figure><img src="data:image/png;base64,{img_pool}">
<figcaption>Concatenated voxel histogram over {N_SAMPLE} random crops = {N_SAMPLE * 64**3:,} voxels (log y). v1 is roughly symmetric, hard-clipped
at ±3 (note the spike at +3). v2/v3 share the <i>same</i> right-skewed shape — v3 is v2 rescaled into [−1,+1], so its
mass sits near ±0.01. v4 folds the tail onto a +{(v4_hi - v4_mu)/v4_sg:.1f}σ ceiling (clip → spike at the edge);
v5 instead <i>soft-squashes</i> with arcsinh — the +30 peak compresses smoothly to ≈+{pstat['v5']['max']:.0f}, no clip, no edge spike.</figcaption></figure>
<table>
<tr><th>ver</th><th>pool range</th><th>pool mean</th><th>pool std</th></tr>
{pool_rows}
</table>

<h2>3 · Per-crop summary statistics (cross-pocket variation)</h2>
<figure><img src="data:image/png;base64,{img_pcs}">
<figcaption>Distribution of each crop's own mean/std/min/max over the {N_SAMPLE}-crop sample (each point = one crop). <b>per-crop std</b> is the tell: v1 is pinned at ≈1
(every pocket forced to the same scale — magnitude information destroyed), whereas v2/v3/v4/v5 vary across pockets
(denser pockets stay denser). <b>per-crop max</b>: v2 ranges 0→60 (unbounded), v1/v4 pile at their clip ceilings,
v5's max varies smoothly (arcsinh — no ceiling), v3 sits near 0.</figcaption></figure>

<h2>4 · Same pocket under each normalisation</h2>
<figure><img src="data:image/png;base64,{img_slice}">
<figcaption>Central z-slice of PDB <b>{PID}</b> (per-panel symmetric colour scale). The spatial pattern is
<i>identical</i> across all five — only the value scale/contrast differs. v1's vivid red/blue is a per-crop
contrast stretch + global-vs-local zero-point, not extra information.</figcaption></figure>

<h2>5 · Why it matters for training</h2>
<p style="font-size:14px">The density channel is reconstructed by the MAE pretext with plain per-element MSE and
<b>no per-channel renormalisation</b> (<code>compute_losses</code>), and the crop is fed <i>as-is</i>
(<code>normalize:false</code>, <code>sigma_noise:0</code>). So a scheme's value-scale ∝ its reconstruction-loss
weight. v2 = "density loud" (unit scale), v3 = "density muted" (~10⁴× smaller). The frozen-encoder PDBbind
affinity probe (Spearman ρ, merged_density encoder) is monotonic in that loudness:</p>
<table>
<tr><th>condition</th><th>test ρ</th><th>reading</th></tr>
{probe_rows}
</table>
<p class="note" style="margin-top:10px">→ v3 most likely leads by nearly <i>disabling</i> density (reverting to the strong atom-only baseline,
0.422 ≈ 0.421), not by representing it well. Forcing experimental density into the representation (v2, dual-head v4)
<i>degrades</i> affinity here. Clean isolating test still pending: the plain-v4 encoder (unit scale) is predicted to
land near v2 (~0.34), below v3.</p>

<footer>
Generated by <code>notebook/dataset_stats.py</code> from on-disk crops in
<code>voxbind/dataset/data/pdbbind/voxels{{,_v2,_v3,_v4,_v5}}</code> · {N_SAMPLE}-crop sample, seed 0.
Probe ρ from <code>probe_results_e99_v4.csv</code> (3 seeds, epoch 99). Re-run the script to regenerate.
</footer>
</div></body></html>"""

OUT.write_text(HTML)
kb = OUT.stat().st_size / 1024
print(f"wrote {OUT}  ({kb:.0f} kB)")
print(f"sample={len(sample)}  shared={len(shared)}  counts={counts}")
for v in VERS:
    print(f"  {v}: pool range [{pstat[v]['min']:+.3f}, {pstat[v]['max']:+.3f}]  μ={pstat[v]['mean']:+.4f}  σ={pstat[v]['std']:.4f}")
print(f"  v2σ/v3max_abs = {v2_sig:.4f}/{v3_max:.2f} = {ratio:.5f}  (v2 = v3 × {1/ratio:.1f})")
