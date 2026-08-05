#!/usr/bin/env python
"""Before/after visualisation of protein-only attenuation masking (alpha field).

Builds alpha(x) in [0,1] from POCKET ATOM POSITIONS ONLY — ligand coordinates never
enter construction — then applies the soft filter

    rho' = alpha * rho + (1 - alpha) * noise

and renders density-before / alpha / density-after for real v5 crops, with the ligand
overlaid for VERIFICATION ONLY (the sanity check from the design brief: most ligand
atoms should land in low-alpha territory).

Two details that matter and are easy to get wrong:
  * the fill noise is BLURRED to the map's intrinsic resolution before blending. White
    N(0,1) noise has enormous high-frequency gradient, and VoxBind derives a gradmag
    channel from the density in-model — white noise there would be wildly OOD.
  * alpha is built from a Gaussian-blurred pocket occupancy, i.e. a smooth monotone
    proxy for distance-to-nearest-protein-atom. Transition width is set by sigma_vox
    (1 vox = 0.25 A), so sigma 6-8 gives the ~1.5-2 A falloff the brief requires — at
    or above the map's own blur, so the boundary is not physically detectable.

Run (CPU, niced — training is using the GPUs):
    nice -n 15 /opt/conda/envs/voxbind/bin/python notebook/html/260806/build_alpha_viz.py
"""
import base64, io, os, sys
sys.path.insert(0, "/home1/irteam/VoxBind")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray
from voxbind.voxelizer import Voxelizer
from voxbind.models.mae_ops import gaussian_blur3d

ROOT = "/home1/irteam/VoxBind/voxbind"
OUT = os.path.join(ROOT, "notebook/html/260806")
N_SHOW = 3                                  # complexes to render
SIGMA_ALPHA = 7.0                           # vox; 7 * 0.25 A = 1.75 A transition width
SIGMA_NOISE = 7.0                           # vox; correlate fill noise to map resolution
STRENGTHS = [0.0, 0.5, 0.8, 1.0]            # dose-response sweep (0 = holo, 1 = full)
QUANTILE = 0.90                             # proximity value mapped to alpha = 1


def build_alpha(poc_vox: torch.Tensor, sigma_vox: float, q: float) -> torch.Tensor:
    """alpha in [0,1] from POCKET ATOMS ONLY. High near protein, low in open space."""
    occ = poc_vox.sum(dim=1, keepdim=True)                     # (B,1,G,G,G) protein occupancy
    prox = gaussian_blur3d(occ, sigma_vox)                     # smooth monotone-in-distance proxy
    # normalise by a high quantile of the *protein-adjacent* values rather than the max,
    # so dense protein cores don't compress everything else toward 0.
    ref = torch.quantile(prox[occ > 0], q) if (occ > 0).any() else prox.max()
    return (prox / ref.clamp(min=1e-8)).clamp(0.0, 1.0)


def correlated_noise(like: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """N(0,1) blurred to the map's resolution, renormalised to unit variance."""
    n = torch.randn_like(like)
    n = gaussian_blur3d(n, sigma_vox)
    return n / n.std().clamp(min=1e-8)


def apply_filter(rho, alpha, strength, sigma_noise):
    """rho' = a*rho + (1-a)*noise, with a = 1 - strength*(1-alpha)."""
    a = 1.0 - strength * (1.0 - alpha)
    return a * rho + (1.0 - a) * correlated_noise(rho, sigma_noise), a


def png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    # same dataset settings as the running protein_first job
    ds = DatasetCrossDockedXray(
        data_dir=os.path.join(ROOT, "dataset/data"),
        crops_dir=os.path.join(ROOT, "dataset/data/xray_crops_aligned_v5"),
        split="train", use_xray=True, aug=False, normalize=False,
        ligand_radius=0.5, pocket_radius=-1,
        subset_xray_only=True, subset_n=78512, cache_size=8,
    )
    vox = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device="cpu")
    print(f"dataset: {len(ds)} complexes", flush=True)

    cards, stats = [], []
    shown = 0
    for idx in range(len(ds)):
        if shown >= N_SHOW:
            break
        s = ds[idx]
        if s.get("xray_density") is None:
            continue
        rho = s["xray_density"].unsqueeze(0).unsqueeze(0).float()      # (1,1,G,G,G)
        # Voxelizer.mol2vox indexes batch["coords"] etc., i.e. it wants a COLLATED dict
        # of batched tensors (as a DataLoader would produce), not a list of samples.
        batched = lambda d: {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in d.items()}
        lig = vox.forward(batched(s["ligand"]), num_channels=7)        # (1,7,G,G,G)
        poc = vox.forward(batched(s["pocket"]), num_channels=4)        # (1,4,G,G,G)

        alpha = build_alpha(poc, SIGMA_ALPHA, QUANTILE)
        lig_occ, poc_occ = lig.sum(1, keepdim=True), poc.sum(1, keepdim=True)
        # SANITY CHECK: ligand used for verification only, never for construction.
        a_lig = float((alpha * lig_occ).sum() / lig_occ.sum().clamp(min=1e-8))
        a_poc = float((alpha * poc_occ).sum() / poc_occ.sum().clamp(min=1e-8))
        stats.append((idx, a_lig, a_poc))
        print(f"[{idx}] alpha@ligand={a_lig:.3f}  alpha@protein={a_poc:.3f}", flush=True)

        z = rho.shape[-1] // 2
        fig, axes = plt.subplots(1, 3 + len(STRENGTHS) - 1, figsize=(3.0 * (2 + len(STRENGTHS)), 3.2))
        vmin, vmax = np.percentile(rho.numpy(), [2, 98])

        axes[0].imshow(rho[0, 0, :, :, z], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("density (holo)", fontsize=9)
        im = axes[1].imshow(alpha[0, 0, :, :, z], cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"alpha (protein only)\nsigma={SIGMA_ALPHA:.0f}vox={SIGMA_ALPHA*0.25:.2f}A", fontsize=9)
        axes[1].contour(lig_occ[0, 0, :, :, z], levels=[0.2], colors="cyan", linewidths=0.9)
        for j, st in enumerate([s_ for s_ in STRENGTHS if s_ > 0]):
            out, _ = apply_filter(rho, alpha, st, SIGMA_NOISE)
            axes[2 + j].imshow(out[0, 0, :, :, z], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[2 + j].set_title(f"attenuated, strength={st}", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"complex idx {idx}   (cyan = ligand footprint, verification only)", fontsize=10)
        cards.append((idx, a_lig, a_poc, png_b64(fig)))
        shown += 1

    ok = sum(1 for _, al, ap, _ in cards if al < ap)
    rows = "\n".join(
        f"<tr><td>{i}</td><td>{al:.3f}</td><td>{ap:.3f}</td>"
        f"<td class='{'good' if al < ap else 'bad'}'>{'ligand attenuated' if al < ap else 'CHECK: ligand preserved'}</td></tr>"
        for i, al, ap, _ in cards)
    imgs = "\n".join(
        f"<figure><img src='data:image/png;base64,{b}'/>"
        f"<figcaption>idx {i} &mdash; alpha@ligand {al:.3f} vs alpha@protein {ap:.3f}</figcaption></figure>"
        for i, al, ap, b in cards)

    html = f"""<h1>Protein-only attenuation masking &mdash; before / after</h1>
<p class=sub>alpha(x) built from <b>pocket atom positions only</b>; ligand coordinates never enter
construction and are drawn (cyan) purely to verify the field. Soft filter
<code>rho' = alpha&middot;rho + (1&minus;alpha)&middot;noise</code>, noise blurred to
sigma={SIGMA_NOISE:.0f} vox ({SIGMA_NOISE*0.25:.2f} A) so its gradient statistics match real
density &mdash; VoxBind derives a gradmag channel from this map in-model, and white noise there
would be immediately out of distribution.</p>
<h2>Sanity check</h2>
<p>Mean alpha weighted by ligand occupancy vs by protein occupancy. The mask is doing its job
when <b>alpha@ligand &lt; alpha@protein</b>. Passing: <b>{ok}/{len(cards)}</b>.</p>
<table><tr><th>idx</th><th>alpha@ligand</th><th>alpha@protein</th><th>verdict</th></tr>
{rows}</table>
<h2>Slices (central z-plane)</h2>
{imgs}
<h2>Parameters</h2>
<pre>sigma_alpha = {SIGMA_ALPHA} vox ({SIGMA_ALPHA*0.25:.2f} A transition width)
sigma_noise = {SIGMA_NOISE} vox ({SIGMA_NOISE*0.25:.2f} A)
quantile    = {QUANTILE} (proximity value mapped to alpha=1)
strengths   = {STRENGTHS}
crops       = xray_crops_aligned_v5 (arcsinh+z), normalize=false, pocket_radius=-1</pre>"""

    style = """<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1400px;
margin:2rem auto;padding:0 1.5rem;line-height:1.6;color:#1a1a1a;background:#fff}
h1{font-size:1.6rem;border-bottom:2px solid #e5e5e5;padding-bottom:.4rem}
h2{font-size:1.15rem;margin-top:2rem}
.sub{color:#555;font-size:.95rem}
table{border-collapse:collapse;margin:1rem 0}
th,td{border:1px solid #ddd;padding:.4rem .8rem;text-align:left;font-size:.9rem}
th{background:#f6f6f6}
.good{color:#0a7d33;font-weight:600}.bad{color:#c0392b;font-weight:600}
figure{margin:1.5rem 0}img{max-width:100%;border:1px solid #e0e0e0;border-radius:4px}
figcaption{font-size:.85rem;color:#666;margin-top:.4rem}
pre{background:#f6f6f6;padding:.8rem;border-radius:4px;font-size:.85rem;overflow-x:auto}
code{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px}
@media(prefers-color-scheme:dark){body{background:#151515;color:#e8e8e8}
h1{border-color:#333}th{background:#222}th,td{border-color:#333}.sub,figcaption{color:#aaa}
pre,code{background:#1e1e1e}img{border-color:#333}}
</style>"""
    path = os.path.join(OUT, "260806_alpha_masking.html")
    open(path, "w").write(f"<!doctype html><meta charset=utf-8><title>Alpha masking</title>{style}{html}")
    print(f"\nwrote {path}\nsanity: {ok}/{len(cards)} complexes have alpha@ligand < alpha@protein")


if __name__ == "__main__":
    main()
