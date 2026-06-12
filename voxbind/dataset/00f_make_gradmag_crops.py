#!/usr/bin/env python
"""Materialize REAL gradmag (‖∇ρ‖) crops as a drop-in "density" source → gradmag-only ablation.

Gradmag-only is implemented WITHOUT touching the training pipeline: precompute the
per-sample-z-scored gradient magnitude of each density crop and store it exactly
where a density crop lives, so `input_mode=density` (the verified 1-channel path)
trained on this dir == a gradmag-only encoder.

Correctness w.r.t. augmentation: ‖∇·‖ is rotation-invariant, so rotating the stored
gradmag (the density-only aug path) equals ‖∇(rotated ρ)‖ — voxel-aligned, same as the
on-the-fly gradmag the full model derives post-augmentation.

Outputs (float16, mirror the real layout so each is a drop-in density source):
  xray_crops_aligned_v5_gradmag/train/{idx:06d}.npy   (1ch 64^3 = ‖∇ρ‖)
  pdbbind/voxels_v5_gradmag/density/{pid}.npy          (probe: gradmag-as-density)

Usage:  python dataset/00f_make_gradmag_crops.py [train|probe|all]
"""
import sys, glob, time
from pathlib import Path
import numpy as np
import torch
from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore

HERE  = Path(__file__).resolve().parent
DATA  = HERE / "data"
CROPS = DATA / "xray_crops_aligned_v5"
GMAG  = DATA / "xray_crops_aligned_v5_gradmag"
PVOX  = DATA / "pdbbind" / "voxels_v5"
PGMAG = DATA / "pdbbind" / "voxels_v5_gradmag"
GRID  = 64


def _gradmag(density_np: np.ndarray) -> np.ndarray:
    g = gradient_magnitude3d(torch.from_numpy(density_np.astype(np.float32)).view(1, 1, GRID, GRID, GRID))
    return per_sample_zscore(g).view(GRID, GRID, GRID).numpy().astype(np.float16)


def materialize_train():
    crops = sorted(glob.glob(str(CROPS / "train" / "[0-9]*.npy")))
    out = GMAG / "train"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(crops):
        idx = int(Path(c).stem)
        np.save(out / f"{idx:06d}.npy", _gradmag(np.load(c)))
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  train {i+1}/{len(crops)}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    for f in ("train_available.npy", "test_available.npy", "stats.json"):
        src = CROPS / f
        if src.exists():
            (GMAG / f).write_bytes(src.read_bytes())
    print(f"[train] done {len(crops)} crops in {time.time()-t0:.0f}s -> {GMAG}")


def materialize_probe():
    dens = sorted(glob.glob(str(PVOX / "density" / "*.npy")))
    out = PGMAG / "density"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(dens):
        pid = Path(c).stem
        np.save(out / f"{pid}.npy", _gradmag(np.load(c)))
        if (i + 1) % 500 == 0:
            print(f"  probe {i+1}/{len(dens)}", flush=True)
    for f in ("stats.json",):
        src = PVOX / f
        if src.exists():
            (PGMAG / f).write_bytes(src.read_bytes())
    print(f"[probe] done {len(dens)} complexes in {time.time()-t0:.0f}s -> {PGMAG}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("train", "all"):
        materialize_train()
    if stage in ("probe", "all"):
        materialize_probe()
    print("DONE")
