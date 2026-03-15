"""05b_renormalize_crops.py — Apply local normalization to precomputed X-ray crops.

Reads existing globally z-scored crops from xray_crops/ (produced by
05_process_xray.py), applies local ±3σ clipping + z-score normalization,
and writes the result to xray_crops_norm/.

The output directory can be used with DatasetCrossDockedXray(normalize=False)
to skip per-sample normalization at training time (slightly faster __getitem__).
With the default normalize=True, normalization always happens at load time, so
this script is optional — it trades preprocessing time for faster data loading.

Normalization pipeline (per crop):
  1. Compute local mean and std of the 64³ crop.
  2. Clip values at ±3σ (local) to suppress outlier density peaks.
  3. Re-z-score using pre-clip statistics → output mean ≈ 0, std ≈ 1.

Usage
-----
    cd voxbind
    python scripts/05b_renormalize_crops.py \
        --in_dir  dataset/data/xray_crops \
        --out_dir dataset/data/xray_crops_norm

    # then use with:
    #   DatasetCrossDockedXray(crops_dir='.../xray_crops_norm', normalize=False)
    # or in YAML:
    #   crops_dir: dataset/data/xray_crops_norm
    #   normalize: false

Outputs
-------
    dataset/data/xray_crops_norm/
        train/{:06d}.npy     float16 (64, 64, 64)  locally normalized
        test/{:06d}.npy      float16 (64, 64, 64)  locally normalized
        train_available.npy  copied unchanged from in_dir
        test_available.npy   copied unchanged from in_dir
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from voxbind.dataset.crossdocked_xray import normalize_crop


def process_split(
    in_dir: Path,
    out_dir: Path,
    split: str,
    overwrite: bool,
) -> None:
    in_split  = in_dir  / split
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    crop_files = sorted(in_split.glob("*.npy"))
    if not crop_files:
        print(f"[{split}] No .npy files found in {in_split} — skipping.")
        return

    n_written = n_skipped = 0
    for crop_path in tqdm(crop_files, desc=split, unit="crop"):
        out_path = out_split / crop_path.name
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue

        crop = np.load(str(crop_path)).astype(np.float32)   # (64, 64, 64)
        crop_norm = normalize_crop(crop)
        np.save(str(out_path), crop_norm.astype(np.float16))
        n_written += 1

    # Copy availability mask unchanged (records which indices have a crop file)
    avail_src = in_dir / f"{split}_available.npy"
    avail_dst = out_dir / f"{split}_available.npy"
    if avail_src.exists():
        shutil.copy2(str(avail_src), str(avail_dst))

    total = len(crop_files)
    print(f"  [{split}]  written={n_written:,}  skipped={n_skipped:,}  "
          f"total={total:,}  →  {out_split}")
    if avail_src.exists():
        print(f"           availability mask copied  →  {avail_dst}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply local normalization to precomputed X-ray density crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in_dir",    default="dataset/data/xray_crops",
                   help="Input directory (output of 05_process_xray.py)")
    p.add_argument("--out_dir",   default="dataset/data/xray_crops_norm",
                   help="Output directory for locally normalized crops")
    p.add_argument("--splits",    nargs="+", default=["train", "test"],
                   choices=["train", "val", "test"],
                   help="Splits to process")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute and overwrite existing output files")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {in_dir}\n"
            "Run scripts/05_process_xray.py first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Local normalization of X-ray density crops ===")
    print(f"  in_dir   : {in_dir.resolve()}")
    print(f"  out_dir  : {out_dir.resolve()}")
    print(f"  splits   : {args.splits}")
    print(f"  overwrite: {args.overwrite}")
    print()

    for split in args.splits:
        process_split(in_dir, out_dir, split, args.overwrite)

    print("\nDone.")
    print("To use pre-normalized crops (skips runtime normalization):")
    print(f"  DatasetCrossDockedXray(crops_dir='{out_dir}', normalize=False)")
    print("Or in YAML:")
    print(f"  crops_dir: {out_dir}")
    print( "  normalize: false")


if __name__ == "__main__":
    main()
