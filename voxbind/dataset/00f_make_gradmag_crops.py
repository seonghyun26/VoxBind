#!/usr/bin/env python
"""Materialize REAL gradmag (‖∇ρ‖) crops as a drop-in "density" source → gradmag-only ablation.

Gradmag-only is implemented WITHOUT touching the training pipeline: precompute the
per-sample-z-scored gradient magnitude of each density crop and store it exactly
where a density crop lives, so `input_mode=density` (the verified 1-channel path)
trained on this dir == a gradmag-only encoder.

Correctness w.r.t. augmentation: ‖∇·‖ is rotation-invariant, so rotating the stored
gradmag (the density-only aug path) equals ‖∇(rotated ρ)‖ — voxel-aligned, same as the
on-the-fly gradmag the full model derives post-augmentation.

Outputs (float16, mirror the real layout so each is a drop-in density source).
Co-located INSIDE the v5 dataset version (same source crops), matching the
nesting 00e uses for the noise control:
  xray_crops_aligned_v5/gradmag/train/{idx:06d}.npy    (1ch 64^3 = ‖∇ρ‖)
  pdbbind/voxels_v5/gradmag/density/{pid}.npy          (probe: gradmag-as-density)

Usage:  python dataset/00f_make_gradmag_crops.py [train|probe|all]
"""
import argparse, glob, time
from pathlib import Path
import numpy as np
import torch
from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore

HERE  = Path(__file__).resolve().parent
DATA  = HERE / "data"
CROPS = DATA / "xray_crops_aligned_v5"          # default v5 train crops
PVOX  = DATA / "pdbbind" / "voxels_v5"          # default PDBbind probe voxels (corpus-independent)
GRID  = 64


def _gradmag(density_np: np.ndarray) -> np.ndarray:
    g = gradient_magnitude3d(torch.from_numpy(density_np.astype(np.float32)).view(1, 1, GRID, GRID, GRID))
    return per_sample_zscore(g).view(GRID, GRID, GRID).numpy().astype(np.float16)


def materialize_train(crops_dir: Path, out_dir: Path):
    crops = sorted(glob.glob(str(crops_dir / "train" / "[0-9]*.npy")))
    out = out_dir / "train"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(crops):
        idx = int(Path(c).stem)
        np.save(out / f"{idx:06d}.npy", _gradmag(np.load(c)))
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  train {i+1}/{len(crops)}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    for f in ("train_available.npy", "test_available.npy", "stats.json"):
        src = crops_dir / f
        if src.exists():
            (out_dir / f).write_bytes(src.read_bytes())
    print(f"[train] done {len(crops)} crops in {time.time()-t0:.0f}s -> {out_dir}")


def materialize_probe(probe_vox_dir: Path, probe_out: Path):
    dens = sorted(glob.glob(str(probe_vox_dir / "density" / "*.npy")))
    out = probe_out / "density"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(dens):
        pid = Path(c).stem
        np.save(out / f"{pid}.npy", _gradmag(np.load(c)))
        if (i + 1) % 500 == 0:
            print(f"  probe {i+1}/{len(dens)}", flush=True)
    for f in ("stats.json",):
        src = probe_vox_dir / f
        if src.exists():
            (probe_out / f).write_bytes(src.read_bytes())
    print(f"[probe] done {len(dens)} complexes in {time.time()-t0:.0f}s -> {probe_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all", choices=["train", "probe", "all"])
    ap.add_argument("--crops_dir", default=str(CROPS), help="train density crops (train/{idx}.npy)")
    ap.add_argument("--out", default="", help="train gradmag output dir (default: <crops_dir>/gradmag)")
    ap.add_argument("--probe_vox_dir", default=str(PVOX), help="PDBbind probe voxels (has density/)")
    ap.add_argument("--probe_out", default="", help="probe gradmag output (default: <probe_vox_dir>/gradmag)")
    ap.add_argument("--threads", type=int, default=0, help="cap torch CPU threads (0 = leave default)")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    crops_dir = Path(args.crops_dir); out_dir = Path(args.out) if args.out else crops_dir / "gradmag"
    probe_vox_dir = Path(args.probe_vox_dir); probe_out = Path(args.probe_out) if args.probe_out else probe_vox_dir / "gradmag"
    print(f"=== gradmag crops ===  stage={args.stage}  threads={args.threads or 'default'}")
    if args.stage in ("train", "all"):
        print(f"  train: {crops_dir} -> {out_dir}")
        materialize_train(crops_dir, out_dir)
    if args.stage in ("probe", "all"):
        print(f"  probe: {probe_vox_dir} -> {probe_out}")
        materialize_probe(probe_vox_dir, probe_out)
    print("DONE")
