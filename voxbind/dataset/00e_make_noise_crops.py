#!/usr/bin/env python
"""Materialize global-empirical NOISE density+gradmag crops for the density-ablation control.

Negative control for "does density carry real signal": replace the density (and
gradmag) channels with i.i.d. noise drawn from the GLOBAL empirical value
distribution of the real density channel (same marginal histogram, zero spatial
signal). Noise is materialized IN ADVANCE (fixed per complex), not on the fly.

Pools are built from a set of real density crops (--crops_dir); the same density
pool seeds both the training-crop noise and the PDBbind probe-voxel noise so the
encoder sees a consistent noise distribution at train and probe time.

Gradmag channel (--gradmag_mode):
  pool       draw gradmag i.i.d. from the real-gradmag distribution (marginal-matched,
             independent of the noise density). The original v5 control.
  from_noise gradmag = per_sample_zscore(‖∇(noise density)‖) — derived FROM the noise
             density (self-consistent noise density+gradient pair; marginal NOT matched).

Outputs (float16, mirrors the real layout so each is a drop-in source):
  {out_dir}/density/{split}/{idx:06d}.npy   (1ch 64^3)   + {out_dir}/density/{split}_available.npy
  {out_dir}/gradmag/{split}/{idx:06d}.npy   (1ch 64^3)
  {probe_out_dir}/density/{pid}.npy          (probe density noise)  + {probe_out_dir}/stats.json
  {probe_out_dir}/gradmag/{pid}.npy          (probe gradmag noise)

Usage (v5 — original behaviour, all defaults):
  python dataset/00e_make_noise_crops.py all

Usage (PLINDER density pool, gradmag derived from the noise):
  python dataset/00e_make_noise_crops.py all \
      --crops_dir dataset/data/pretrain/xray_crops_aligned_plinder \
      --out_dir   dataset/data/pretrain/xray_crops_aligned_plinder_noise \
      --probe_out_dir dataset/data/pdbbind/voxels_v5_plinder_noise \
      --gradmag_mode from_noise
"""
import argparse
import glob
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root → `import voxbind`
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GRID = 64
SEED0 = 20260612          # base seed; per-item seed = SEED0 + index / pid-hash (deterministic, fixed)
POOL_CROPS = 600          # # real crops sampled to build the global pool(s)
POOL_KEEP = 12_000_000    # # values kept per pool (drawn from with replacement)


def _gradmag_of(density_np: np.ndarray) -> np.ndarray:
    g = gradient_magnitude3d(torch.from_numpy(density_np.astype(np.float32)).view(1, 1, GRID, GRID, GRID))
    g = per_sample_zscore(g)
    return g.view(GRID, GRID, GRID).numpy()


def build_pools(crops_dir: Path, pools_path: Path, gradmag_mode: str, seed: int):
    """Pool real density (and, for gradmag_mode='pool', real gradmag) voxel values."""
    crops = sorted(glob.glob(str(crops_dir / "train" / "[0-9]*.npy")))
    if not crops:
        sys.exit(f"[error] no crops at {crops_dir}/train/*.npy")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(crops), size=min(POOL_CROPS, len(crops)), replace=False)
    dvals, gvals = [], []
    for j, i in enumerate(pick):
        d = np.load(crops[i]).astype(np.float32)
        dvals.append(d.ravel())
        if gradmag_mode == "pool":
            gvals.append(_gradmag_of(d).ravel())
        if (j + 1) % 100 == 0:
            print(f"  pooled {j+1}/{len(pick)} crops", flush=True)
    dvals = np.concatenate(dvals)
    d_pool = rng.choice(dvals, size=min(POOL_KEEP, dvals.size), replace=False).astype(np.float16)
    save = {"density": d_pool}
    print(f"[pools] density n={d_pool.size:,} range=[{d_pool.min():.3f},{d_pool.max():.3f}] "
          f"mean={d_pool.mean():.4f} std={d_pool.std():.4f}")
    g_pool = None
    if gradmag_mode == "pool":
        gvals = np.concatenate(gvals)
        g_pool = rng.choice(gvals, size=min(POOL_KEEP, gvals.size), replace=False).astype(np.float16)
        save["gradmag"] = g_pool
        print(f"[pools] gradmag n={g_pool.size:,} range=[{g_pool.min():.3f},{g_pool.max():.3f}] "
              f"mean={g_pool.mean():.4f} std={g_pool.std():.4f}")
    else:
        print("[pools] gradmag_mode=from_noise → no gradmag pool (derived per crop)")
    pools_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pools_path, **save)
    print(f"[pools] saved -> {pools_path}")
    return d_pool, g_pool


def _load_pools(pools_path: Path):
    z = np.load(pools_path)
    return z["density"], (z["gradmag"] if "gradmag" in z.files else None)


def _draw(pool: np.ndarray, rng: np.random.Generator, shape) -> np.ndarray:
    return rng.choice(pool, size=int(np.prod(shape)), replace=True).reshape(shape).astype(np.float16)


def _noise_gradmag(noise_d: np.ndarray, g_pool, rng, gradmag_mode: str) -> np.ndarray:
    if gradmag_mode == "from_noise":
        return _gradmag_of(noise_d.astype(np.float32)).astype(np.float16)
    return _draw(g_pool, rng, noise_d.shape)


# ---------------------------------------------------------------------------
# parallel materialization (ProcessPoolExecutor). Each crop is independent: the
# density noise is drawn from the shared pool seeded per-complex, and (for
# gradmag_mode=from_noise) the gradmag is a single 64^3 conv on that noise. We
# fan out across processes, each pinned to 1 torch thread to avoid the
# many-procs × many-threads oversubscription that made the serial run crawl.
# Pools are reloaded from disk in each worker (cheap, ~24MB) so nothing large
# is pickled per task.
# ---------------------------------------------------------------------------
_WORKER = {}


def _init_worker(pools_path, gradmag_mode, seed):
    torch.set_num_threads(1)
    d_pool, g_pool = _load_pools(Path(pools_path))
    _WORKER.update(d_pool=d_pool, g_pool=g_pool, gradmag_mode=gradmag_mode, seed=seed)


def _train_one(item):
    idx, out_d_str, out_g_str = item
    seed = _WORKER["seed"]
    rng = np.random.default_rng(seed + idx)                  # fixed-per-complex
    nd = _draw(_WORKER["d_pool"], rng, (GRID, GRID, GRID))
    np.save(Path(out_d_str) / f"{idx:06d}.npy", nd)
    np.save(Path(out_g_str) / f"{idx:06d}.npy",
            _noise_gradmag(nd, _WORKER["g_pool"], rng, _WORKER["gradmag_mode"]))
    return idx


def _probe_one(item):
    pid, shape, out_d_str, out_g_str = item
    seed = _WORKER["seed"]
    # stable (cross-process, reproducible) per-pid seed — NOT Python's randomized hash().
    rng = np.random.default_rng(seed + (zlib.crc32(pid.encode()) % 10_000_019))
    nd = _draw(_WORKER["d_pool"], rng, tuple(shape))
    np.save(Path(out_d_str) / f"{pid}.npy", nd)
    np.save(Path(out_g_str) / f"{pid}.npy",
            _noise_gradmag(nd, _WORKER["g_pool"], rng, _WORKER["gradmag_mode"]))
    return pid


def _run_pool(fn, items, pools_path, gradmag_mode, seed, workers, label, log_every):
    """Map fn over items across `workers` procs (or serially if workers<=1)."""
    t0 = time.time()
    n = len(items)
    if workers <= 1:
        _init_worker(str(pools_path), gradmag_mode, seed)
        for i, it in enumerate(items):
            fn(it)
            if (i + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  {label} {i+1}/{n}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(pools_path), gradmag_mode, seed)) as ex:
            for i, _ in enumerate(ex.map(fn, items, chunksize=32)):
                if (i + 1) % log_every == 0:
                    el = time.time() - t0
                    print(f"  {label} {i+1}/{n}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    print(f"[{label}] done {n} in {time.time()-t0:.0f}s ({n/max(time.time()-t0,1e-9):.0f}/s)")


def materialize_train(crops_dir, out_dir, pools_path, gradmag_mode, seed, workers):
    crops = sorted(glob.glob(str(crops_dir / "train" / "[0-9]*.npy")))
    out_d = out_dir / "density" / "train"; out_g = out_dir / "gradmag" / "train"
    out_d.mkdir(parents=True, exist_ok=True); out_g.mkdir(parents=True, exist_ok=True)
    items = [(int(Path(c).stem), str(out_d), str(out_g)) for c in crops]
    _run_pool(_train_one, items, pools_path, gradmag_mode, seed, workers, "train", 4000)
    # mirror availability/stats so the loader's subset logic matches the real crops.
    # crops_dir=out_dir/density at train time → put the files in BOTH out_dir/ and out_dir/density/.
    for f in ("train_available.npy", "test_available.npy", "stats.json"):
        src = crops_dir / f
        if src.exists():
            (out_dir / f).write_bytes(src.read_bytes())
            (out_dir / "density" / f).write_bytes(src.read_bytes())
    print(f"[train] mirrored availability/stats -> {out_dir}")


def materialize_probe(probe_vox_dir, probe_out_dir, pools_path, gradmag_mode, seed, workers):
    dens = sorted(glob.glob(str(probe_vox_dir / "density" / "*.npy")))
    if not dens:
        sys.exit(f"[error] no probe voxels at {probe_vox_dir}/density/*.npy")
    out_d = probe_out_dir / "density"; out_g = probe_out_dir / "gradmag"
    out_d.mkdir(parents=True, exist_ok=True); out_g.mkdir(parents=True, exist_ok=True)
    # the probe reads {probe_out_dir}/stats.json (density_stats_metadata: dens_dir.parent/stats.json)
    src_stats = probe_vox_dir / "stats.json"
    if src_stats.exists():
        shutil.copyfile(src_stats, probe_out_dir / "stats.json")
    items = [(Path(c).stem, np.load(c, mmap_mode="r").shape, str(out_d), str(out_g)) for c in dens]
    _run_pool(_probe_one, items, pools_path, gradmag_mode, seed, workers, "probe", 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all", choices=["pools", "train", "probe", "all"])
    ap.add_argument("--crops_dir", default=str(DATA / "pretrain/xray_crops_aligned_v5"),
                    help="real density crops to pool from + mirror (train/{idx}.npy layout)")
    ap.add_argument("--out_dir", default=str(DATA / "pretrain/xray_crops_aligned_v5_noise"),
                    help="output noise dir (density/ + gradmag/ subdirs)")
    ap.add_argument("--probe_vox_dir", default=str(DATA / "pdbbind" / "voxels_v5"),
                    help="PDBbind probe voxels to mirror shapes from (has density/ + stats.json)")
    ap.add_argument("--probe_out_dir", default=str(DATA / "pdbbind" / "voxels_v5_noise"),
                    help="output PDBbind probe-noise dir (for the probe's --noise_voxels_dir)")
    ap.add_argument("--gradmag_mode", choices=["pool", "from_noise"], default="pool",
                    help="pool=draw gradmag i.i.d. from real-gradmag dist (v5 control); "
                         "from_noise=derive ‖∇(noise density)‖ (self-consistent noise pair)")
    ap.add_argument("--seed", type=int, default=SEED0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 8),
                    help="parallel processes for train/probe materialization (1=serial)")
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir); out_dir = Path(args.out_dir)
    probe_vox_dir = Path(args.probe_vox_dir); probe_out_dir = Path(args.probe_out_dir)
    pools_path = out_dir / "pools.npz"

    print(f"=== noise crops ===  gradmag_mode={args.gradmag_mode}  workers={args.workers}")
    print(f"  crops_dir : {crops_dir}\n  out_dir   : {out_dir}")
    if args.stage in ("probe", "all"):
        print(f"  probe_vox : {probe_vox_dir}\n  probe_out : {probe_out_dir}")

    if args.stage in ("pools", "all"):
        build_pools(crops_dir, pools_path, args.gradmag_mode, args.seed)
    if args.stage in ("train", "all"):
        materialize_train(crops_dir, out_dir, pools_path, args.gradmag_mode, args.seed, args.workers)
    if args.stage in ("probe", "all"):
        materialize_probe(probe_vox_dir, probe_out_dir, pools_path, args.gradmag_mode, args.seed, args.workers)
    print("DONE")


if __name__ == "__main__":
    main()
