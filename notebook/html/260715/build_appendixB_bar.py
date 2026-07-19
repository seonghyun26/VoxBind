#!/usr/bin/env python3
"""build_appendixB_bar.py — insert/refresh Appendix B (baselines bar chart) in 260715_meeting.html.

Replicates the 3-panel (Pearson r / Spearman ρ / RMSE) grouped bar chart from
260701_meeting.html Table 1 and adds the two new supervised baselines trained this
run — CheapNet and BindNet. Whiskers = ±1 std over 3 seeds. Idempotent: re-running
replaces the block between the APPENDIX B markers.

    python notebook/html/260715/build_appendixB_bar.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "260715_meeting.html")

# (name, color, best, r(mean,sd), rho(mean,sd), rmse(mean,sd)) on lp_edrscc_v2 test, 3 seeds.
# The 9 prior rows are read off 260701_meeting.html Table 1; CheapNet/BindNet are new (this run).
METHODS = [
    ("HBGSA",             "#9b59b6", False, (0.565, 0.007), (0.546, 0.003), (1.568, 0.066)),
    ("EGNN",              "#e67e22", False, (0.539, 0.012), (0.533, 0.016), (1.531, 0.012)),
    ("EGNN + TargetDiff", "#e74c3c", False, (0.597, 0.009), (0.579, 0.019), (1.447, 0.016)),
    ("GET",               "#2c3e50", False, (0.596, 0.005), (0.591, 0.005), (1.428, 0.008)),
    ("ProFSA",            "#e84393", False, (0.626, 0.005), (0.597, 0.007), (1.518, 0.016)),
    ("CheapNet",          "#f39c12", "new", (0.607, 0.012), (0.581, 0.013), (1.460, 0.032)),
    ("BindNet",           "#6c5ce7", "new", (0.529, 0.020), (0.512, 0.012), (1.604, 0.122)),
    ("AEV-PLIG",          "#1abc9c", False, (0.522, 0.019), (0.492, 0.019), (1.617, 0.004)),
    ("DSMBind",           "#8395a7", False, (0.541, 0.007), (0.507, 0.010), (1.494, 0.011)),
    ("C",                 "#3498db", False, (0.632, 0.007), (0.605, 0.002), (1.519, 0.117)),
    ("C+D+G",             "#2ecc71", True,  (0.656, 0.006), (0.637, 0.006), (1.353, 0.031)),
]

PANELS = [
    dict(idx=0, key=0, title="Test Pearson r",       vmin=0.50, vmax=0.70, ticks=[0.50, 0.55, 0.60, 0.65, 0.70]),
    dict(idx=1, key=1, title="Test Spearman &rho;",  vmin=0.45, vmax=0.65, ticks=[0.45, 0.50, 0.55, 0.60, 0.65]),
    dict(idx=2, key=2, title="Test RMSE",            vmin=1.30, vmax=1.75, ticks=[1.30, 1.40, 1.50, 1.60, 1.70]),
]
PLOTW, TOP, BASE = 300.0, 36.0, 220.0     # taller plot (was BASE 176)
PANEL_X0 = [45.0, 425.0, 805.0]        # plot-left of each panel
BARW = 22.0


def svg():
    n = len(METHODS)
    pitch = PLOTW / n
    out = ['<svg viewBox="0 0 1130 256" width="100%" style="max-width:1130px;display:block;margin:0 auto" '
           'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    for p in PANELS:
        x0 = PANEL_X0[p["idx"]]
        x1 = x0 + PLOTW
        vmin, vmax = p["vmin"], p["vmax"]
        def y(v):
            return BASE - (v - vmin) / (vmax - vmin) * (BASE - TOP)
        # title
        out.append(f'<text x="{x0 + PLOTW/2:.1f}" y="15" font-size="11.5" font-weight="600" '
                   f'fill="#1c2433" text-anchor="middle">{p["title"]}</text>')
        # gridlines + tick labels
        for t in p["ticks"]:
            yy = y(t)
            out.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#e6e9ef" stroke-width="1"/>')
            out.append(f'<text x="{x0-5:.1f}" y="{yy+3:.1f}" font-size="8.5" fill="#9aa3b2" text-anchor="end">{t:.2f}</text>')
        # bars
        for i, (name, color, best, *metrics) in enumerate(METHODS):
            m, sd = metrics[p["key"]]
            cx = x0 + pitch * (i + 0.5)
            bx = cx - BARW / 2
            by = y(m)
            h = max(0.0, BASE - by)
            stroke = ' stroke="#000" stroke-width="2.5"' if best is True else (
                     ' stroke="#1c2433" stroke-width="1.4" stroke-dasharray="3 2"' if best == "new" else '')
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BARW}" height="{h:.1f}" rx="2" fill="{color}"{stroke}/>')
            # whisker (±sd)
            yhi, ylo = y(m + sd), y(m - sd)
            out.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" stroke="#566072" stroke-width="1"/>')
            out.append(f'<line x1="{cx-3:.1f}" y1="{yhi:.1f}" x2="{cx+3:.1f}" y2="{yhi:.1f}" stroke="#566072" stroke-width="1"/>')
            out.append(f'<line x1="{cx-3:.1f}" y1="{ylo:.1f}" x2="{cx+3:.1f}" y2="{ylo:.1f}" stroke="#566072" stroke-width="1"/>')
            lab = "#1d5a3a" if best is True else ("#7a4a17" if best == "new" else "#5b6678")
            out.append(f'<text x="{cx:.1f}" y="{yhi-4:.1f}" font-size="8" fill="{lab}" text-anchor="middle" '
                       f'font-weight="600">{m:.3f}</text>')
        # baseline
        out.append(f'<line x1="{x0:.1f}" y1="{BASE:.1f}" x2="{x1:.1f}" y2="{BASE:.1f}" stroke="#aeb6c4" stroke-width="1.5"/>')
        if p["key"] == 2:
            out.append(f'<text x="{x0 + PLOTW/2:.1f}" y="{BASE+28:.0f}" font-size="9" fill="#b04a2f" text-anchor="middle">&#8595; lower is better</text>')
    out.append("</svg>")
    return "\n      ".join(out)


def legend():
    items = []
    for name, color, best, *_ in METHODS:
        bd = ";border:2px solid #000" if best is True else (";border:1.4px dashed #1c2433" if best == "new" else "")
        nm = f"<b>{name}</b>" if best == "new" else name
        items.append(f'<span><span style="width:12px;height:12px;border-radius:3px;background:{color};'
                     f'display:inline-block{bd}"></span> {nm}</span>')
    return ('<div class="legend" style="justify-content:center;margin-top:8px;gap:13px;">'
            + " ".join(items) + "</div>")


def appendix_b():
    return f"""<!-- APPENDIX B START (generated by build_appendixB_bar.py) -->
  <!-- ============================================================ -->
  <!-- APPENDIX B — SUPERVISED BASELINES BAR CHART                  -->
  <!-- ============================================================ -->
  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num" style="background:#566072">B</span> Supervised structure-based baselines &mdash; <code>lp_edrscc_v2</code></h2>
    <p class="section-intro">
      The full baseline suite on our frozen <code>lp_edrscc_v2</code> test split (Kd/Ki, 1320; 3 seeds),
      reproduced from <b>260701</b> Table&nbsp;1 with the two baselines trained this run added:
      <b style="color:#b8860b">CheapNet</b> (ICLR&nbsp;2025, GIGN cross-attention, from scratch) and
      <b style="color:#6c5ce7">BindNet</b> (ICLR&nbsp;2024, Uni-Mol complex cross-net, <b>from scratch</b> &mdash;
      its pretrained checkpoint is access-restricted). <b>C</b>&nbsp;/&nbsp;<b>C+D+G</b> are the frozen density
      probe (coords vs. coords+density+gradmag). Whiskers&nbsp;=&nbsp;&plusmn;1&nbsp;std.
    </p>
    <section class="block">
      <p class="table-title">Appendix&nbsp;B &nbsp;&middot;&nbsp; Baselines + CheapNet/BindNet vs. the density probe</p>
      <div class="table-wrap" style="padding:16px 18px 10px;">
      {svg()}
      {legend()}
      <p class="figure-cap">Frozen-probe / trained test metrics &middot; whiskers&nbsp;=&nbsp;&plusmn;1&nbsp;std (3 seeds).
        <b>C+D+G</b> (green, bold border) = best; <b>CheapNet</b>&nbsp;/&nbsp;<b>BindNet</b> (dashed border) = new this run.</p>
      </div>
      <div class="notes">
        <b>Where the new models land.</b> On this hard novel-target split everything clusters at
        <i>r</i>&nbsp;≈&nbsp;0.53&ndash;0.66. <b>CheapNet</b> (r&nbsp;0.607) sits with the supervised structure
        pack &mdash; ≈&nbsp;GET (0.596), &lt;&nbsp;ProFSA (0.626) and the density probe C+D+G (0.656 / &rho;&nbsp;0.637).
        <b>BindNet from scratch</b> (r&nbsp;0.529) is the weakest supervised row and a <b>lower bound</b>: it is a
        pretrain&rarr;finetune method, and the ≈0.10&nbsp;<i>r</i> gap to its pretrained Uni-Mol sibling ProFSA is
        essentially the value of the BioLip pretraining we could not download.
      </div>
    </section>
  </div>
<!-- APPENDIX B END -->"""


def main():
    html = open(DOC, encoding="utf-8").read()
    block = appendix_b()
    start, end = "<!-- APPENDIX B START", "<!-- APPENDIX B END -->"
    if start in html and end in html:
        i, j = html.index(start), html.index(end) + len(end)
        html = html[:i] + block + html[j:]
    else:
        anchor = "\n</div>\n</body>"          # the .page-closing </div> before </body>
        k = html.rfind(anchor)
        if k < 0:
            raise SystemExit("could not find .page close anchor")
        html = html[:k] + "\n" + block + html[k:]
    open(DOC, "w", encoding="utf-8").write(html)
    print(f"patched {DOC}  ({len(METHODS)} methods, Appendix B)")


if __name__ == "__main__":
    main()
