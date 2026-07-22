#!/usr/bin/env python
"""00h_make_noise_graddens.py — ‖∇(noise ρ)‖ crops (gradient-of-noise-density).

The 3×3 density×gradmag grid needs a NOISE gradmag defined consistently with the
real one: gradmag = ‖∇density‖. For the noise arm that means the gradient of the
NOISE density crops (00e), NOT 00e's i.i.d. gradmag (which is drawn from the real
gradmag histogram, a different object). Mirrors 00f but over the noise crops.

Outputs (per-sample z-scored ‖∇·‖, drop-in 'density' source, shuffle order preserved
because the source noise crops already are):
  xray_crops_aligned_v5_noise/graddens/train/{idx:06d}.npy   (1ch 64^3 = ‖∇(noise ρ)‖)
  pdbbind/voxels_v5_noise/graddens/density/{pid}.npy         (probe)

Usage:  python dataset/00h_make_noise_graddens.py [train|probe|all]
"""
import sys, glob, time
from pathlib import Path
import numpy as np
import torch
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore

HERE  = Path(__file__).resolve().parent
DATA  = HERE / "data"
NOISE = DATA / "xray_crops_aligned_v5_noise"
GRAD  = NOISE / "graddens"
PNOISE = DATA / "pdbbind" / "voxels_v5_noise"
PGRAD  = PNOISE / "graddens"
GRID = 64


def _gradmag(density_np: np.ndarray) -> np.ndarray:
    g = gradient_magnitude3d(torch.from_numpy(density_np.astype(np.float32)).view(1, 1, GRID, GRID, GRID))
    g = per_sample_zscore(g)
    return g.view(GRID, GRID, GRID).numpy().astype(np.float16)


def materialize_train():
    crops = sorted(glob.glob(str(NOISE / "density" / "train" / "[0-9]*.npy")))
    out = GRAD / "train"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(crops):
        idx = int(Path(c).stem)
        np.save(out / f"{idx:06d}.npy", _gradmag(np.load(c)))
        if (i + 1) % 2000 == 0:
            print(f"  train {i+1}/{len(crops)}  ({time.time()-t0:.0f}s)", flush=True)
    # copy availability + stats so crops_dir=graddens is a self-contained drop-in
    for f in ("train_available.npy", "test_available.npy", "stats.json"):
        src = NOISE / f
        if src.exists():
            (GRAD / f).write_bytes(src.read_bytes())
    print(f"[train] done {len(crops)} -> {GRAD}")


def materialize_probe():
    dens = sorted(glob.glob(str(PNOISE / "density" / "*.npy")))
    out = PGRAD / "density"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(dens):
        pid = Path(c).stem
        np.save(out / f"{pid}.npy", _gradmag(np.load(c)))
        if (i + 1) % 500 == 0:
            print(f"  probe {i+1}/{len(dens)}", flush=True)
    print(f"[probe] done {len(dens)} -> {PGRAD}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("train", "all"):
        materialize_train()
    if stage in ("probe", "all"):
        materialize_probe()
