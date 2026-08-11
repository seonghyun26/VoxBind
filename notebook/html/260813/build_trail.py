#!/usr/bin/env python
"""Build notebook/html/260813/trail.html — the CDG-encoder experiment trail.
Each experiment gets a stable #NN number + an adjective-noun codename
(deterministic md5 of the run name, so re-runs keep the same code).
Add/edit rows in EXP and re-run:  python build_trail.py
"""
import hashlib

ADJ = ("sweet cool warm bold calm brave keen soft bright swift mild ripe fresh "
       "neat wild pure tart crisp lush snug spry bold merry glossy jolly").split()
NOUN = ("lemon grape mango peach melon berry plum lime cherry olive apple pear "
        "fig kiwi guava date apricot papaya lychee quince pecan cocoa maple").split()

# section, name, what-it-checks, result, css-class-for-result
EXP = [
  ("run",  "260813_cdg_100m_v2_g7411", "arch",
   "<b>Channel separation</b>: density &amp; gradmag as separate ChannelViT groups <b>[7,4,1,1]</b> vs champion's grouped [7,4,2].",
   "GPU 0-3 · training (chain)", "na"),
  ("done", "260810_cdg_100m_v2_atommask050", "masking",
   "Does atom-biased masking (ratio 0.5) beat champion at <b>full 100M</b>? (with 46M controls, isolates the idea.)",
   "0.624 / 0.524 — atom-mask HURTS at 100M too", "flat"),
  ("run",  "260810_cdg_100m_v2_uniflig_vdwpoc", "radius",
   "<b>Original VoxBind representation</b>: uniform ligand (0.5) + vdW pocket (−1). Isolates the 260603 ligand→vdW gain (vs champion vdW/vdW 0.644 and full-uniform 0.468). Probes on v5 (matched, no build).",
   "GPU 4-7 · training", "na"),
  ("done", "vdw_vs_uniform_matched", "radius",
   "<b>vdW vs uniform atom radius</b>, each probed on <b>matched</b> voxels. vdW keeps physical steric size; uniform blobs lose it.",
   "vdW 0.644 vs uniform 0.468 — <b>vdW ESSENTIAL (+0.176)</b>", "win"),
  ("done", "260810_cdg_45m_v2_unifmask050", "masking",
   "46M UNIFORM-mask control for the atom-mask run (both 46M, mask 0.5). Isolates masking strategy.",
   "0.629 / 0.538 — &gt; atom_biased 0.612 (atom-mask HURTS −0.017)", "flat"),
  ("done", "ENSEMBLE_x3_mse+corr", "ensemble",
   "Concat champion + v3_m090 + v3_m095 features → one head. Diverse encoders (v2/v3, different masks) are complementary.",
   "0.656 / 0.591 ✓ BEST", "win"),
  ("done", "champion_mse+corr_head", "head",
   "Add a Pearson-correlation aux term to the probe MLP loss (GenScore-style).",
   "0.647 / 0.559 ✓", "win"),
  ("done", "260705_champion_v2_m075_g742", "baseline",
   "The reference: 100M ChannelViT, PLINDER-v2, mask 0.75, groups [7,4,2].",
   "0.644 / 0.546 (ref)", "na"),
  ("done", "260806_cdg_100m_v2_ep100", "schedule",
   "Undertraining? Train 100 epochs instead of 50.",
   "e25 0.653 ≈ e50 0.636 — SATURATED", "flat"),
  ("done", "260808_cdg_100m_v3_m075", "data×mask",
   "Clean v3 data at champion's mask 0.75 (v3 previously only 0.85–0.95).",
   "0.632 / 0.562 — v3 wants high mask", "flat"),
  ("done", "260809_cdg_100m_v2_m075_rope3d", "pos-enc",
   "Swap learnable position encoding → axial 3D RoPE.",
   "0.622 / 0.542 — HURTS", "fail"),
  ("done", "260809_cdg_45m_v2_atommask050", "masking",
   "User idea: mask near atoms (empty space wasteful). 46M + atom_biased mask 0.5 together.",
   "0.612 / 0.528 — size-confounded → controls running", "flat"),
  ("done", "260806_cdg_100m_v2_d2vaux05", "objective",
   "New pretraining objective: data2vec latent-prediction aux on top of MAE.",
   "0.572 / 0.507 — HURTS (−0.07)", "fail"),
  ("done", "260808_cdg_100m_v2_m075_uniformrad", "radius",
   "Uniform atom radius (0.5), probed on matched uniform voxels.",
   "0.468 / 0.375 — uniform radius MUCH worse than vdW", "fail"),
  ("done", "GET_baseline_lba30_lba60", "baseline",
   "External E(3)-equiv baseline on protein-novelty splits.",
   "0.517 / 0.562 (&lt; our 0.569/0.613)", "na"),
]


def codename(name, used):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    for k in range(200):
        a = ADJ[(h + k * 7) % len(ADJ)]
        n = NOUN[((h // len(ADJ)) + k * 13) % len(NOUN)]
        c = f"{a}-{n}"
        if c not in used:
            used.add(c); return c
    return f"exp-{h % 9999:04d}"

used = set()
codes = [(f"{i+1:02d}", codename(e[1], used)) for i, e in enumerate(EXP)]

SEC = {"run": ("Running", "run"), "q": ("Queued", "q"), "done": ("Done", "done")}
ORDER = ["run", "q", "done"]

def rows(sec):
    out = []
    for (num, code), e in zip(codes, EXP):
        if e[0] != sec: continue
        out.append(
          f'<tr><td class="code"><span class="num">#{num}</span><span class="cn">{code}</span></td>'
          f'<td class="name">{e[1]}</td>'
          f'<td class="what"><span class="tag">{e[2]}</span>{e[3]}</td>'
          f'<td class="res {e[5]}">{e[4]}</td></tr>')
    return "\n".join(out)

def section(sec):
    title, cls = SEC[sec]
    n = sum(1 for e in EXP if e[0] == sec)
    return (f'<h2><span class="pill {cls}">{title}</span><span class="cnt">{n}</span></h2>\n'
            f'<table><thead><tr><th>Code</th><th>Experiment</th><th>What it checks</th><th>Result (test ρ / val ρ)</th></tr></thead>\n'
            f'<tbody>\n{rows(sec)}\n</tbody></table>')

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment trail · 260813</title>
<style>
 :root{{--ink:#1c2433;--ink-soft:#5b6678;--ink-faint:#7a8699;--line:#e3e7ee;--bg:#f5f6f8;--card:#fff;
  --green:#2f6f4f;--green-bg:#eaf5ee;--blue:#38559b;--blue-bg:#e7eefb;--warn:#9a6112;--warn-bg:#fcf3e0;
  --danger:#a33f39;--shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.05);}}
 *{{box-sizing:border-box;}} body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5;}}
 .page{{max-width:1120px;margin:0 auto;padding:44px 22px 90px;}}
 header.doc{{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:14px;}}
 .eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:700;color:var(--green);}}
 h1{{font-size:27px;margin:8px 0 4px;letter-spacing:-.01em;}} .sub{{color:var(--ink-soft);font-size:14px;}}
 .goal{{margin:14px 0 6px;padding:12px 16px;background:var(--green-bg);border:1px solid #cfe6d8;border-radius:10px;font-size:14px;}}
 .goal b{{color:var(--green);}}
 h2{{display:flex;align-items:center;gap:10px;font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:32px 0 11px;}}
 .pill{{font-size:11px;font-weight:800;padding:3px 10px;border-radius:999px;letter-spacing:.04em;}}
 .pill.run{{color:var(--blue);background:var(--blue-bg);}} .pill.q{{color:var(--warn);background:var(--warn-bg);}} .pill.done{{color:var(--green);background:var(--green-bg);}}
 .cnt{{color:var(--ink-faint);font-weight:600;font-size:12px;}}
 table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:var(--shadow);margin-bottom:6px;}}
 th,td{{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;}}
 th{{background:#f6f8fa;color:var(--ink-faint);font-size:11px;text-transform:uppercase;letter-spacing:.05em;}}
 tbody tr:last-child td{{border-bottom:0;}}
 td.code{{white-space:nowrap;}} .num{{display:inline-block;color:var(--ink-faint);font-weight:700;font-size:11.5px;font-family:ui-monospace,Menlo,monospace;margin-right:7px;}}
 .cn{{display:inline-block;font-weight:700;font-size:12.5px;color:var(--blue);background:var(--blue-bg);padding:1px 8px;border-radius:6px;font-family:ui-monospace,Menlo,monospace;}}
 td.name{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--ink-soft);white-space:nowrap;}}
 td.what{{color:var(--ink-soft);font-size:13.5px;}} td.res{{font-weight:700;white-space:nowrap;font-size:13px;}}
 .win{{color:var(--green);}} .fail{{color:var(--danger);}} .flat{{color:var(--warn);}} .na{{color:var(--ink-faint);font-weight:500;}}
 .tag{{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:5px;margin-right:5px;background:#eef1f5;color:var(--ink-soft);}}
 footer{{margin-top:36px;color:var(--ink-faint);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;}}
 code{{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:.9em;}}
</style></head><body><div class="page">
 <header class="doc"><div class="eyebrow">Experiment trail · 260813</div>
 <h1>CDG encoder — beat the champion</h1>
 <div class="sub">Affinity regression on <code>lp_edrscc_v2</code> (test ρ / val ρ, 3 seeds). Each run has a <b>#num</b> + codename.</div></header>
 <div class="goal"><b>Goal:</b> beat the champion CDG encoder (<b>test ρ 0.644 / val ρ 0.546</b>). Only the <b>3-encoder ensemble (0.656)</b> and the <b>mse+corr probe head (0.647)</b> have beaten it — single-encoder recipe changes keep hitting the plateau.</div>
 {section("run")}
 {section("q")}
 {section("done")}
 <footer>260813 · <code>notebook/html/260813/trail.html</code> · built by <code>build_trail.py</code> (codes = md5(name), stable). Baselines below champion: ProFSA 0.597 · GET 0.591 · AEV 0.550 · HBGSA 0.533 · DSMBind 0.363.</footer>
</div></body></html>"""

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trail.html")
open(out, "w").write(HTML)
print("wrote", out)
for (num, code), e in zip(codes, EXP):
    print(f"  #{num}  {code:16s}  [{e[0]}] {e[1]}")
