#!/usr/bin/env python
"""Re-render Fig 2 (dock box/strip) and Fig 3 (dock vs n_atoms) with the y-axis
in STANDARD orientation (more-negative / stronger-binding at the BOTTOM).

The original per-molecule metrics.json (voxbind/exps/260518_voxbind_10k_noise/
samples/res_ep99_test) no longer exists on disk, so the point data is recovered
from the committed SVGs and pixel->data is self-calibrated from the known group
means (printed in the meeting note), then validated against the known Pearson
correlation before anything is written.

Ground-truth anchors (from macrocycle Tables 2 & 3):
  dock mean : macro -9.98 (n=10) , non -7.32 (n=608)
  n_atoms   : macro  36.5        , non  22.3
  corr(dock, heavy atoms) Pearson = -0.656
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ACC, ACC2, SOFT = "#2f6f4f", "#b07a17", "#5b6678"
MACRO = 12

# anchors
DOCK_MACRO, DOCK_NON = -9.98, -7.32
NAT_MACRO, NAT_NON = 36.5, 22.3
CORR_TARGET = -0.656


def group_substr(s, gid):
    """Return the balanced <g id="gid"> ... </g> substring."""
    start = s.index('<g id="%s">' % gid)
    i, depth = start, 0
    for m in re.finditer(r'<g\b[^>]*>|</g>', s[start:]):
        if m.group(0).startswith('</g'):
            depth -= 1
            if depth == 0:
                return s[start:start + m.end()]
        else:
            depth += 1
    raise RuntimeError("unbalanced group " + gid)


def uses_xy(sub):
    out = []
    for u in re.findall(r'<use\b[^>]*?/>', sub):
        mx = re.search(r'\bx="([-\d.]+)"', u)
        my = re.search(r'\by="([-\d.]+)"', u)
        if mx and my:
            out.append((float(mx.group(1)), float(my.group(1))))
    return np.array(out, float)


def linmap(px_a, val_a, px_b, val_b):
    """Linear map f(px)=k*px+c through (px_a,val_a) and (px_b,val_b)."""
    k = (val_a - val_b) / (px_a - px_b)
    c = val_a - k * px_a
    return k, c


# ─────────────────────────── recover points ────────────────────────────────
f3 = open(os.path.join(HERE, "fig_dock_natoms.svg")).read()
non3 = uses_xy(group_substr(f3, "PathCollection_1"))   # non-macro scatter
mac3 = uses_xy(group_substr(f3, "PathCollection_2"))   # macro scatter
print("fig3 recovered points: non=%d macro=%d" % (len(non3), len(mac3)))

# x-axis (n_atoms) calibration from group means; y-axis (dock) from group means
kx, cx = linmap(mac3[:, 0].mean(), NAT_MACRO, non3[:, 0].mean(), NAT_NON)
ky, cy = linmap(mac3[:, 1].mean(), DOCK_MACRO, non3[:, 1].mean(), DOCK_NON)

na_non = np.rint(kx * non3[:, 0] + cx).astype(int)
na_mac = np.rint(kx * mac3[:, 0] + cx).astype(int)
d_non = ky * non3[:, 1] + cy
d_mac = ky * mac3[:, 1] + cy

na_all = np.concatenate([na_non, na_mac]).astype(float)
d_all = np.concatenate([d_non, d_mac])
corr = np.corrcoef(na_all, d_all)[0, 1]
print("  validate: macro n_atoms mean=%.1f dock=%.2f | non n_atoms=%.1f dock=%.2f | corr=%.3f (target %.3f)"
      % (na_mac.mean(), d_mac.mean(), na_non.mean(), d_non.mean(), corr, CORR_TARGET))
assert abs(corr - CORR_TARGET) < 0.03, "correlation mismatch — extraction wrong"

# ─────────────────────────── Fig 3 re-render ────────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=110)
ax.scatter(na_non, d_non, s=16, color=SOFT, alpha=.30, edgecolors="none",
           label="non-macrocycle (n=%d)" % len(na_non), zorder=2)
ax.scatter(na_mac, d_mac, s=46, color=ACC, alpha=.95, edgecolors="white",
           linewidths=.6, label="macrocycle (n=%d)" % len(na_mac), zorder=4)
coef = np.polyfit(na_all, d_all, 1)
xr = np.array([na_all.min(), na_all.max()])
ax.plot(xr, coef[0] * xr + coef[1], color=ACC2, lw=1.6, ls="--", label="size trend", zorder=3)
ax.set_xlabel("heavy-atom count", fontsize=10.5)
ax.set_ylabel("Vina dock score (kcal/mol)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e3e7ee", lw=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")  # strong-dock now bottom-right
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_dock_natoms.svg"))
plt.close(fig)
print("wrote fig_dock_natoms.svg (flipped)")

# ─────────────────────────── Fig 2 re-render ────────────────────────────────
f2 = open(os.path.join(HERE, "fig_dock_macro.svg")).read()
non2 = uses_xy(group_substr(f2, "PathCollection_1"))   # non-macro strip
mac2 = uses_xy(group_substr(f2, "PathCollection_2"))   # macro strip
print("fig2 recovered points: non=%d macro=%d" % (len(non2), len(mac2)))
ky2, cy2 = linmap(mac2[:, 1].mean(), DOCK_MACRO, non2[:, 1].mean(), DOCK_NON)
v_non = ky2 * non2[:, 1] + cy2
v_mac = ky2 * mac2[:, 1] + cy2
print("  validate: macro dock mean=%.2f (n=%d) | non dock mean=%.2f (n=%d)"
      % (v_mac.mean(), len(v_mac), v_non.mean(), len(v_non)))
assert abs(v_mac.mean() - DOCK_MACRO) < 0.05 and abs(v_non.mean() - DOCK_NON) < 0.05

fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=110)
groups = [("non-macrocycle", v_non, SOFT), ("macrocycle\n(ring ≥ %d)" % MACRO, v_mac, ACC)]
rng = np.random.default_rng(0)
for i, (lab, vals, col) in enumerate(groups):
    vals = np.asarray(vals)
    x = np.full(len(vals), i, float) + rng.uniform(-.12, .12, len(vals))
    ax.scatter(x, vals, s=14, color=col, alpha=.35, edgecolors="none", zorder=2)
    bp = ax.boxplot(vals, positions=[i], widths=.42, showfliers=False, patch_artist=True, zorder=3)
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor=col, alpha=.9, lw=1.6)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=col, lw=1.4)
    for md in bp["medians"]:
        md.set(color=col, lw=2.2)
    # standard axis: strongest (most negative) at bottom; put n= above the box
    ax.text(i, float(vals.max()) + 0.7, "n=%d" % len(vals), ha="center", fontsize=9,
            color=col, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
ax.set_ylabel("Vina dock score (kcal/mol) · lower = better", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e3e7ee", lw=.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_dock_macro.svg"))
plt.close(fig)
print("wrote fig_dock_macro.svg (flipped)")
