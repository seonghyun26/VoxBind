#!/usr/bin/env python3
"""Visualize the mFo-DFc DIFFERENCE density in the model's own 64³ crop frame,
overlaid on the ligand/pocket atoms and the 2Fo-Fc density — same idea as
visualize_plinder_apo_pockets.ipynb, but for the diff channel the encoder ingests.

All three crops (atoms / 2Fo-Fc / mFo-DFc) share ONE 64³ × 0.25 Å ligand-centroid
grid, so if the diff lobes sit on the atoms the crop is aligned in the exact frame
the encoder sees. Produces:
  * diff_density_alignment.html   interactive 3D (cropped to the ligand region)
  * diff_density_slices.png       2D orthogonal-slice montage (atom contour +
                                  2Fo-Fc + mFo-DFc ±), the crispest alignment proof

    python notebook/html/260806/build_diff_density_viz.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
VOX = ROOT / "voxbind" / "dataset" / "data" / "pdbbind"
ADIR, DDIR, FDIR = VOX / "voxels_v5" / "atoms", VOX / "voxels_v5" / "density", VOX / "voxels_v5_diff" / "density"
HTML = Path(__file__).resolve().parent / "diff_density_alignment.html"
PNG = Path(__file__).resolve().parent / "diff_density_slices.png"
PIDS = ["2brb", "3iub", "2pgz"]
G, VOX_A = 64, 0.25


def load(pid):
    a = np.load(ADIR / f"{pid}.npy").astype(np.float32)
    return a[:7].sum(0), a[7:11].sum(0), \
        np.load(DDIR / f"{pid}.npy").astype(np.float32), \
        np.load(FDIR / f"{pid}.npy").astype(np.float32)


def bbox(mask, margin=10):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, G)
    return tuple(slice(lo[i], hi[i]) for i in range(3))


def frac_on(lig, poc, diff, f_p, f_n):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    strong = (diff > f_p) | (diff < f_n)
    near = binary_dilation((lig + poc) > 0.4, generate_binary_structure(3, 1),
                           iterations=int(round(2.0 / VOX_A)))
    return float((strong & near).sum() / max(strong.sum(), 1))


def iso(vol, ax, isomin, color, name, opacity):
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=vol.ravel(),
                         isomin=isomin, isomax=float(vol.max()), surface_count=1,
                         colorscale=[[0, color], [1, color]], showscale=False,
                         opacity=opacity, name=name,
                         caps=dict(x_show=False, y_show=False, z_show=False))


def build_html():
    parts = ["<html><head><meta charset='utf-8'><title>mFo-DFc diff alignment</title></head>",
             "<body style='font-family:system-ui;margin:24px;max-width:1100px'>",
             "<h1>mFo-DFc difference density — alignment in the encoder's crop frame</h1>",
             "<p>Cropped to the ligand region of the model's 64³ × 0.25 Å crop. "
             "<b>Dark</b>=ligand, <b>faint</b>=pocket, <b>blue</b>=2Fo-Fc, "
             "<b>green</b>=mFo-DFc + (unmodeled), <b>red</b>=mFo-DFc − (over-modeled). "
             "Green/red lobes on the atoms ⇒ diff aligned with atoms/density.</p>"]
    for i, pid in enumerate(PIDS):
        lig, poc, dens, diff = load(pid)
        sl = bbox(lig > 0.4, margin=8)                    # ligand region only (compact)
        s = 2                                             # 2× downsample → small HTML
        ax = (np.arange(G)[sl[0]] * VOX_A)[::s]
        L, P, D, F = lig[sl][::s, ::s, ::s], poc[sl][::s, ::s, ::s], \
            dens[sl][::s, ::s, ::s], diff[sl][::s, ::s, ::s]
        f_p, f_n = np.percentile(diff, 99.3), np.percentile(diff, 0.7)
        fr = frac_on(lig, poc, diff, f_p, f_n)
        fig = go.Figure([
            iso(P, ax, 0.4, "#b8c0cc", "pocket", 0.10),
            iso(L, ax, 0.4, "#111827", "ligand", 0.55),
            iso(D, ax, float(np.percentile(dens, 99)), "#2563eb", "2Fo-Fc", 0.13),
            iso(F, ax, float(f_p), "#16a34a", "mFo-DFc +", 0.6),
            iso(-F, ax, float(-f_n), "#dc2626", "mFo-DFc −", 0.6),
        ])
        fig.update_layout(title=f"{pid} · {fr*100:.0f}% of strong diff lobes within 2Å of an atom",
                          scene=dict(aspectmode="data"), height=620,
                          margin=dict(l=0, r=0, t=40, b=0))
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
    parts.append("</body></html>")
    HTML.write_text("\n".join(parts))
    print("wrote", HTML, f"({HTML.stat().st_size/1e6:.1f} MB)")


def build_png():
    """Row per complex: 2Fo-Fc slice and mFo-DFc slice through the ligand centre,
    both with the ligand-atom contour overlaid → alignment visible in 2D."""
    fig, axes = plt.subplots(len(PIDS), 2, figsize=(8, 4 * len(PIDS)))
    for r, pid in enumerate(PIDS):
        lig, poc, dens, diff = load(pid)
        z = int(np.argmax(lig.sum((0, 1))))              # slice through ligand centre
        sl = bbox((lig + poc) > 0.4)[:2]
        Ls, Ds, Fs = lig[sl][:, :, z], dens[sl][:, :, z], diff[sl][:, :, z]
        Ps = poc[sl][:, :, z]
        for c, (title, img, cmap, vlim) in enumerate([
                ("2Fo-Fc density", Ds, "gray", None),
                ("mFo-DFc difference", Fs, "RdBu_r", np.percentile(np.abs(diff), 99.3))]):
            ax = axes[r, c] if len(PIDS) > 1 else axes[c]
            if vlim:
                ax.imshow(img.T, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim)
            else:
                ax.imshow(img.T, origin="lower", cmap=cmap)
            ax.contour(Ls.T, levels=[0.4], colors="#00d000", linewidths=1.4)   # ligand
            ax.contour(Ps.T, levels=[0.4], colors="#ff9900", linewidths=0.6, alpha=0.6)  # pocket
            ax.set_title(f"{pid}  ·  {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("mFo-DFc diff vs 2Fo-Fc, ligand-centre slice · green=ligand contour, orange=pocket",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(PNG, dpi=130, bbox_inches="tight")
    print("wrote", PNG)


if __name__ == "__main__":
    build_html()
    build_png()
