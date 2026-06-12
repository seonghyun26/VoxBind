#!/usr/bin/env python
"""Materialize global-empirical NOISE density+gradmag crops for the density-ablation control.

Negative control for "does density carry real signal": replace the density and
gradmag channels with i.i.d. noise drawn from the GLOBAL empirical value
distribution of the real channels (same marginal histogram, zero spatial signal).
Noise is materialized IN ADVANCE (fixed per complex), not generated on the fly.

Pools are built from the real v5 X-ray crops; the same pools seed both the
training-crop noise and the PDBbind probe-voxel noise so the encoder sees a
consistent noise distribution at train and probe time.

Outputs (float16, mirrors the real layout so each is a drop-in source):
  xray_crops_aligned_v5_noise/density/train/{idx:06d}.npy   (1ch 64^3)
  xray_crops_aligned_v5_noise/gradmag/train/{idx:06d}.npy   (1ch 64^3)
  pdbbind/voxels_v5_noise/density/{pid}.npy                  (probe density noise)
  pdbbind/voxels_v5_noise/gradmag/{pid}.npy                  (probe gradmag noise)

Usage:
  python dataset/00e_make_noise_crops.py pools     # build + save the global pools, print stats
  python dataset/00e_make_noise_crops.py train     # materialize training-crop noise
  python dataset/00e_make_noise_crops.py probe      # materialize PDBbind probe-voxel noise
  python dataset/00e_make_noise_crops.py all        # pools + train + probe
"""
import sys, glob, os, time
from pathlib import Path
import numpy as np
import torch
from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore

HERE   = Path(__file__).resolve().parent
DATA   = HERE / "data"
CROPS  = DATA / "xray_crops_aligned_v5"
NOISE  = DATA / "xray_crops_aligned_v5_noise"
PDBB   = DATA / "pdbbind"
PVOX   = PDBB / "voxels_v5"
PNOISE = PDBB / "voxels_v5_noise"
POOLS  = NOISE / "pools.npz"
GRID   = 64
SEED0  = 20260612          # base seed; per-item seed = SEED0 + index/pid-hash (deterministic, fixed)
POOL_CROPS = 600           # # real crops sampled to build the global pools
POOL_KEEP  = 12_000_000    # # values kept per pool (drawn from with replacement)


def _gradmag_of(density_np: np.ndarray) -> np.ndarray:
    g = gradient_magnitude3d(torch.from_numpy(density_np.astype(np.float32)).view(1, 1, GRID, GRID, GRID))
    g = per_sample_zscore(g)
    return g.view(GRID, GRID, GRID).numpy()


def build_pools() -> tuple[np.ndarray, np.ndarray]:
    crops = sorted(glob.glob(str(CROPS / "train" / "[0-9]*.npy")))
    rng = np.random.default_rng(SEED0)
    pick = rng.choice(len(crops), size=min(POOL_CROPS, len(crops)), replace=False)
    dvals, gvals = [], []
    for j, i in enumerate(pick):
        d = np.load(crops[i]).astype(np.float32)
        g = _gradmag_of(d)
        dvals.append(d.ravel()); gvals.append(g.ravel())
        if (j + 1) % 100 == 0:
            print(f"  pooled {j+1}/{len(pick)} crops", flush=True)
    dvals = np.concatenate(dvals); gvals = np.concatenate(gvals)
    # subsample to POOL_KEEP values each (the empirical distribution to draw from)
    d_pool = rng.choice(dvals, size=min(POOL_KEEP, dvals.size), replace=False).astype(np.float16)
    g_pool = rng.choice(gvals, size=min(POOL_KEEP, gvals.size), replace=False).astype(np.float16)
    NOISE.mkdir(parents=True, exist_ok=True)
    np.savez(POOLS, density=d_pool, gradmag=g_pool)
    print(f"[pools] density n={d_pool.size:,} range=[{d_pool.min():.3f},{d_pool.max():.3f}] "
          f"mean={d_pool.mean():.4f} std={d_pool.std():.4f}")
    print(f"[pools] gradmag n={g_pool.size:,} range=[{g_pool.min():.3f},{g_pool.max():.3f}] "
          f"mean={g_pool.mean():.4f} std={g_pool.std():.4f}")
    print(f"[pools] saved -> {POOLS}")
    return d_pool, g_pool


def _load_pools() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(POOLS)
    return z["density"], z["gradmag"]


def _draw(pool: np.ndarray, rng: np.random.Generator, shape) -> np.ndarray:
    return rng.choice(pool, size=int(np.prod(shape)), replace=True).reshape(shape).astype(np.float16)


def materialize_train(d_pool, g_pool):
    crops = sorted(glob.glob(str(CROPS / "train" / "[0-9]*.npy")))
    out_d = NOISE / "density" / "train"; out_g = NOISE / "gradmag" / "train"
    out_d.mkdir(parents=True, exist_ok=True); out_g.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(crops):
        idx = int(Path(c).stem)
        rng = np.random.default_rng(SEED0 + idx)            # fixed-per-complex
        np.save(out_d / f"{idx:06d}.npy", _draw(d_pool, rng, (GRID, GRID, GRID)))
        np.save(out_g / f"{idx:06d}.npy", _draw(g_pool, rng, (GRID, GRID, GRID)))
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  train {i+1}/{len(crops)}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    # mirror the availability file so the loader's subset logic matches the real crops
    for f in ("train_available.npy", "test_available.npy", "stats.json"):
        src = CROPS / f
        if src.exists():
            (NOISE / f).write_bytes(src.read_bytes())
    print(f"[train] done {len(crops)} crops in {time.time()-t0:.0f}s -> {NOISE}")


def materialize_probe(d_pool, g_pool):
    dens = sorted(glob.glob(str(PVOX / "density" / "*.npy")))
    out_d = PNOISE / "density"; out_g = PNOISE / "gradmag"
    out_d.mkdir(parents=True, exist_ok=True); out_g.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(dens):
        pid = Path(c).stem
        shape = np.load(c).shape
        rng = np.random.default_rng(SEED0 + (abs(hash(pid)) % 10_000_019))
        np.save(out_d / f"{pid}.npy", _draw(d_pool, rng, shape))
        np.save(out_g / f"{pid}.npy", _draw(g_pool, rng, shape))
        if (i + 1) % 500 == 0:
            print(f"  probe {i+1}/{len(dens)}", flush=True)
    print(f"[probe] done {len(dens)} complexes in {time.time()-t0:.0f}s -> {PNOISE}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("pools", "all"):
        d_pool, g_pool = build_pools()
    else:
        d_pool, g_pool = _load_pools()
    if stage in ("train", "all"):
        materialize_train(d_pool, g_pool)
    if stage in ("probe", "all"):
        materialize_probe(d_pool, g_pool)
    print("DONE")
