#!/usr/bin/env python3
"""Regenerate the Table-1 bar chart SVG (3 panels x 6 bars) in 260701_meeting.html.

Panels: Test Pearson r, Test Spearman rho, Test RMSE. Bars carry mean +/- std over
3 seeds (whiskers). Re-ranges per panel so every bar (incl. AEV-PLIG) fits.
Splices the new <svg> + legend in place of the existing one (kept idempotent by
markers on the table-wrap padding div).
"""
import re, sys

HTML = "/home/shpark/prj-denovo/VoxBind/notebook/html/260701/260701_meeting.html"

# bar order + styling (matches existing legend)
BARS = [
    ("C",                 "#3498db", False),
    ("C+D+G",             "#2ecc71", True),   # best per panel -> black stroke
    ("HBGSA",             "#9b59b6", False),
    ("EGNN",              "#e67e22", False),
    ("EGNN + TargetDiff", "#e74c3c", False),
    ("AEV-PLIG",          "#1abc9c", False),
]

# (mean, std) per bar, same order as BARS
PANELS = [
    {"title": "Test Pearson r", "lo": 0.50, "hi": 0.70, "step": 0.05, "dec": 3,
     "vals": [(0.632, 0.007), (0.656, 0.006), (0.565, 0.007),
              (0.539, 0.012), (0.597, 0.009), (0.522, 0.019)]},
    {"title": "Test Spearman &rho;", "lo": 0.45, "hi": 0.65, "step": 0.05, "dec": 3,
     "vals": [(0.605, 0.002), (0.637, 0.006), (0.546, 0.003),
              (0.533, 0.016), (0.579, 0.019), (0.492, 0.019)]},
    {"title": "Test RMSE", "lo": 1.30, "hi": 1.70, "step": 0.10, "dec": 3,
     "vals": [(1.519, 0.117), (1.353, 0.031), (1.568, 0.066),
              (1.531, 0.012), (1.447, 0.016), (1.617, 0.004)]},
]

PANEL_W = 300.0      # each panel occupies 300 user-units horizontally
PLOT_L  = 40.0       # left inset within panel (axis labels live left of this)
PLOT_R  = 288.0      # right inset within panel
BASE_Y  = 176.0      # y of the value-axis baseline (== panel "lo")
TOP_Y   = 36.0       # y of the panel "hi"
N = len(BARS)
BARW = 24.0


def yfor(v, lo, hi):
    return BASE_Y - (v - lo) / (hi - lo) * (BASE_Y - TOP_Y)


def fmt(v, dec):
    return f"{v:.{dec}f}"


def frange(lo, hi, step):
    out, x = [], lo
    # avoid float drift
    n = round((hi - lo) / step)
    return [round(lo + i * step, 6) for i in range(n + 1)]


def panel_svg(p, px):
    """px = x offset of this panel."""
    s = []
    x0, x1 = px + PLOT_L, px + PLOT_L + (PLOT_R - PLOT_L)
    cx_title = px + (PLOT_L + PLOT_R) / 2.0
    s.append(f'<text x="{cx_title:.1f}" y="15" font-size="11.5" font-weight="600" fill="#1c2433" text-anchor="middle">{p["title"]}</text>')
    # gridlines + tick labels
    for t in frange(p["lo"], p["hi"], p["step"]):
        gy = yfor(t, p["lo"], p["hi"])
        lbl = f'{t:.2f}'
        s.append(f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x1:.1f}" y2="{gy:.1f}" stroke="#e6e9ef" stroke-width="1"/>')
        s.append(f'<text x="{x0-5:.1f}" y="{gy+3:.1f}" font-size="8.5" fill="#9aa3b2" text-anchor="end">{lbl}</text>')
    # bars
    plot_w = x1 - x0
    slot = plot_w / N
    for i, (label, color, best) in enumerate(BARS):
        mean, std = p["vals"][i]
        left = x0 + slot * i + (slot - BARW) / 2.0
        cx = left + BARW / 2.0
        by = yfor(mean, p["lo"], p["hi"])
        h = BASE_Y - by
        stroke = ' stroke="#000" stroke-width="2.5"' if best else ''
        s.append(f'<rect x="{left:.1f}" y="{by:.1f}" width="{BARW:.0f}" height="{h:.1f}" rx="2" fill="{color}"{stroke}/>')
        # whiskers (mean +/- std), clamped into plot
        y_hi = yfor(min(mean + std, p["hi"]), p["lo"], p["hi"])
        y_lo = yfor(max(mean - std, p["lo"]), p["lo"], p["hi"])
        s.append(f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#566072" stroke-width="1"/>')
        s.append(f'<line x1="{cx-3:.1f}" y1="{y_hi:.1f}" x2="{cx+3:.1f}" y2="{y_hi:.1f}" stroke="#566072" stroke-width="1"/>')
        s.append(f'<line x1="{cx-3:.1f}" y1="{y_lo:.1f}" x2="{cx+3:.1f}" y2="{y_lo:.1f}" stroke="#566072" stroke-width="1"/>')
        tcolor = '#1d5a3a' if best else '#5b6678'
        s.append(f'<text x="{cx:.1f}" y="{y_hi-4:.1f}" font-size="8" fill="{tcolor}" text-anchor="middle" font-weight="600">{fmt(mean, p["dec"])}</text>')
    # solid baseline
    s.append(f'<line x1="{x0:.1f}" y1="{BASE_Y:.1f}" x2="{x1:.1f}" y2="{BASE_Y:.1f}" stroke="#aeb6c4" stroke-width="1.5"/>')
    return "\n      ".join(s)


def build_svg():
    parts = ['<svg viewBox="0 0 900 210" width="100%" style="max-width:900px;display:block;margin:0 auto" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    for k, p in enumerate(PANELS):
        parts.append(panel_svg(p, k * PANEL_W))
    # "lower is better" under the RMSE panel
    rmse_cx = 2 * PANEL_W + (PLOT_L + PLOT_R) / 2.0
    parts.append(f'<text x="{rmse_cx:.1f}" y="203.0" font-size="9" fill="#b04a2f" text-anchor="middle">&#8595; lower is better</text>')
    parts.append('</svg>')
    svg = "\n      ".join(parts)
    legend = ('<div class="legend" style="justify-content:center;margin-top:8px;gap:14px;">'
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#3498db;display:inline-block"></span> C</span> '
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#2ecc71;display:inline-block;border:2px solid #000"></span> C+D+G</span> '
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#9b59b6;display:inline-block"></span> HBGSA</span> '
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#e67e22;display:inline-block"></span> EGNN</span> '
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#e74c3c;display:inline-block"></span> EGNN + TargetDiff</span> '
              '<span><span style="width:12px;height:12px;border-radius:3px;background:#1abc9c;display:inline-block"></span> AEV-PLIG</span></div>')
    return svg, legend


def main():
    with open(HTML) as f:
        html = f.read()
    svg, legend = build_svg()
    new_block = "      " + svg + "\n      " + legend + "\n"
    # replace from the first <svg ...> up to and including the existing legend </div>,
    # within the chart table-wrap (padding:16px ...). Anchor on the figure-cap that follows.
    pat = re.compile(r'      <svg viewBox="0 0 900 [0-9]+".*?</div>\n(?=      <p class="figure-cap")', re.DOTALL)
    m = pat.search(html)
    if not m:
        print("ERROR: chart block not found", file=sys.stderr); sys.exit(1)
    html = html[:m.start()] + new_block + html[m.end():]
    with open(HTML, "w") as f:
        f.write(html)
    print("chart updated (6 bars, re-ranged Spearman lo=0.45 / RMSE hi=1.70)")


if __name__ == "__main__":
    main()
