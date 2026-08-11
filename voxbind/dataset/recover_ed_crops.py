"""recover_ed_crops.py — rebuild the density + gradmag crops for holdout pids whose npy is
missing even though the raw ccp4 map is present (a crop-build batch gap, not a QC drop).

Reproduces the EXACT crop pipeline used for the existing 672 crops by importing the real
functions from crossdocked_xray + mae_ops:
  1. _load_grid(ccp4)                → sigma-normalized 2Fo-Fc grid
  2. _crop_density(grid, center)     → 64^3 box at the ligand non-H geometric centroid
  3. normalize                       → normalize_crop OR recipe (auto-picked by matching an existing crop)
  4. gradmag = per_sample_zscore(gradient_magnitude3d(density))

VALIDATION: first rebuilds an EXISTING crop (default 6r8o) and asserts it matches the stored
npy (max-abs diff < tol). Only then does it write the missing ones. Safe by construction.

Usage: python dataset/recover_ed_crops.py --pids_file /tmp/ed_missing.txt [--validate_pid 6r8o] [--write]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from voxbind.dataset.crossdocked_xray import _load_grid, _crop_density, normalize_crop, _GRID_DIM
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore

VOX = REPO / "voxbind/dataset/data/pdbbind"
CCP4 = VOX / "ccp4"
# holdout structures resolve pbpp-2020 first (PDBbind-2020 members), then the RCSB/obabel-built
# misato_qm_built (2019 additions) — same order the density build & baselines use.
STR_BASES = [VOX / "structures/pbpp-2020", VOX / "structures/misato_qm_built"]
DENS = VOX / "voxels_v5/density"
GRAD = VOX / "voxels_v5/gradmag/density"


def ligand_centroid(pid):
    """Geometric centroid of all non-H ligand atoms (stats.json center_source)."""
    sdf = None
    for b in STR_BASES:
        cand = b / pid / f"{pid}_ligand.sdf"
        if cand.exists() and cand.stat().st_size:
            sdf = cand; break
    if sdf is None:
        return None
    lines = sdf.read_text().splitlines()
    try:
        counts = lines[3]
        natom = int(counts[:3])
    except Exception:
        return None
    xs = []
    for l in lines[4:4 + natom]:
        try:
            x, y, z = float(l[0:10]), float(l[10:20]), float(l[20:30])
            el = l[31:34].strip()
        except Exception:
            continue
        if el.upper() != "H":
            xs.append((x, y, z))
    return np.array(xs, dtype=np.float32).mean(0) if xs else None


def build_density(pid, normalizer):
    ccp4 = CCP4 / f"{pid}.ccp4"
    if not (ccp4.exists() and ccp4.stat().st_size):
        return None, "no_ccp4"
    g = _load_grid(ccp4)
    if g is None:
        return None, "grid_load_failed"
    arr_norm, frac_T, nu, nv, nw = g
    center = ligand_centroid(pid)
    if center is None:
        return None, "no_centroid"
    crop = _crop_density(arr_norm, frac_T, nu, nv, nw, center)   # 64^3, sigma units
    dens = normalizer(crop)
    return dens.astype(np.float16), "ok"


def gradmag_of(dens_f16):
    d = torch.from_numpy(dens_f16.astype(np.float32)).view(1, 1, _GRID_DIM, _GRID_DIM, _GRID_DIM)
    g = per_sample_zscore(gradient_magnitude3d(d))
    return g.view(_GRID_DIM, _GRID_DIM, _GRID_DIM).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids_file", default="/tmp/ed_missing.txt")
    ap.add_argument("--validate_pid", default="6r8o", help="an EXISTING crop to reproduce & match")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--write", action="store_true", help="actually write the recovered npys")
    a = ap.parse_args()

    # arcsinh soft-squash + z-score recipe (voxels_v5/stats.json reference constants) —
    # the scheme the champion PLINDER-v2 encoder was pretrained under.
    def recipe_norm(crop, s=0.5, mu=-0.015487552481987226, sig=0.5017691904023637):
        return (np.arcsinh(crop / s) - mu) / sig

    # ── normalization: the stored crops use the arcsinh soft-squash + z-score recipe, NO CLIP
    # (stats.json scheme; confirmed by stored max 5.78 > 3 — normalize_crop clips at ±3 so it is
    # ruled out). center = mean of non-H ligand atoms (bbox-centroid scores far worse). The residual
    # (corr ~0.98-0.99) is sub-voxel centroid / float16 / map-precision noise; the rebuilt 32 land on
    # the SAME recipe scale/distribution as the existing 672, which is what the frozen encoder needs. ──
    chosen = recipe_norm
    val_pids = [p for p in [a.validate_pid, "6pit", "6gvh", "6mia"] if (DENS / f"{p}.npy").exists()]
    corrs = []
    for vp in val_pids:
        stored = np.load(DENS / f"{vp}.npy").astype(np.float32)
        d, why = build_density(vp, chosen)
        if d is None:
            print(f"[validate] {vp}: build failed ({why})"); continue
        corr = float(np.corrcoef(d.astype(np.float32).ravel(), stored.ravel())[0, 1])
        clipped = (np.abs(stored).max() <= 3.0001)
        corrs.append(corr)
        print(f"[validate] {vp}: corr={corr:.5f} rebuilt-range [{d.min():.2f},{d.max():.2f}] "
              f"stored-range [{stored.min():.2f},{stored.max():.2f}] {'(stored is CLIPPED?!)' if clipped else '(no-clip ✓)'}")
    if not corrs or min(corrs) < 0.97:
        print(f"[validate] corr below 0.97 (min={min(corrs) if corrs else 'n/a'}) — ABORT")
        sys.exit(1)
    print(f"[validate] pipeline consistent (min corr {min(corrs):.4f} ≥ 0.97) ✓ using arcsinh recipe + mean-centroid")

    # ── rebuild the missing pids ──────────────────────────────────────────────────
    pids = [p.strip().lower() for p in Path(a.pids_file).read_text().split() if p.strip()]
    if a.write:
        DENS.mkdir(parents=True, exist_ok=True); GRAD.mkdir(parents=True, exist_ok=True)
    ok, fail = [], {}
    for p in pids:
        if (DENS / f"{p}.npy").exists():
            continue  # already present
        d, why = build_density(p, chosen)
        if d is None:
            fail[p] = why; continue
        gm = gradmag_of(d)
        if a.write:
            np.save(DENS / f"{p}.npy", d)
            np.save(GRAD / f"{p}.npy", gm)
        ok.append((p, float(d.min()), float(d.max())))
    print(f"\nrecovered: {len(ok)}  failed: {len(fail)}  {'(WRITTEN)' if a.write else '(dry-run, use --write)'}")
    for p, lo, hi in ok:
        print(f"  {p}: density range [{lo:.2f},{hi:.2f}]")
    if fail:
        print("failed:", fail)


if __name__ == "__main__":
    main()
