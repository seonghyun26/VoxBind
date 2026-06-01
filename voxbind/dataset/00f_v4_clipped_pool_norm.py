"""00f_v4_clipped_pool_norm.py — V4 density crops for CrossDocked.

Pool-z-score after asymmetric percentile clip:

    Step 1: invert v2's normalization to recover raw voxel values
            raw = v2 * σ_v2 + μ_v2          (σ_v2=0.29722, μ_v2=0.00196)
    Step 2: clip raw to [CLIP_LO, CLIP_HI]
            (CLIP_LO = P_0.1%  ≈ -0.578)
            (CLIP_HI = P_99.9% ≈ +1.647)
    Step 3: compute pool stats (μ_clipped, σ_clipped) over the clipped voxels
    Step 4: x' = (clip(raw) − μ_clipped) / σ_clipped

Why this beats v3 (max-abs): v3 divided everything by a single outlier voxel
(+31.34), collapsing typical signal to ~0.01. v4 clips the outlier tail first
so the z-score divisor reflects real signal scale.

Why asymmetric clip: positive tail = atom-peak signal, negative = noise floor.
Symmetric ±3σ clipping would chop legitimate atom density. The 99.9%-ile
threshold preserves virtually every atom peak (atoms typically peak at
+1 to +1.5 raw) while removing the long-tail single-voxel outliers.

Outputs
-------
    dataset/data/xray_crops_aligned_v4/train/{:06d}.npy     float16 (64,64,64)
    dataset/data/xray_crops_aligned_v4/test/{:06d}.npy      float16
    dataset/data/xray_crops_aligned_v4/{split}_available.npy
    dataset/data/xray_crops_aligned_v4/stats.json
"""

import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# v2 normalization constants (from xray_crops_aligned_v2/stats.json)
V2_SIGMA = 0.29721657782649036
V2_MU    = 0.001964637491336175

# Clip thresholds (raw units), from 200-sample voxel-pool quantile estimate
CLIP_LO = -0.578     # P_0.1% in raw scale
CLIP_HI = +1.647     # P_99.9% in raw scale

V2_DIR = Path("/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data/xray_crops_aligned_v2")
V4_DIR = Path("/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data/xray_crops_aligned_v4")


def recover_raw(arr_v2_fp16: np.ndarray) -> np.ndarray:
    return arr_v2_fp16.astype(np.float32) * V2_SIGMA + V2_MU


def main():
    V4_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: clip + accumulate pool stats ──────────────────────────────────
    print(f"Pass 1: scanning v2 crops, clipping to [{CLIP_LO}, {CLIP_HI}], "
          f"accumulating pool stats …")
    sum_x   = np.float64(0.0)
    sum_xx  = np.float64(0.0)
    n       = 0
    raw_min, raw_max = np.inf, -np.inf
    per_split_stats  = {}

    for split in ["train", "test"]:
        src = V2_DIR / split
        files = sorted(src.glob("*.npy"))
        s_sum, s_sum_sq, s_n = 0.0, 0.0, 0
        for f in tqdm(files, desc=f"  pass1 {split}"):
            arr_v2  = np.load(f)
            arr_raw = recover_raw(arr_v2)
            raw_min = min(raw_min, float(arr_raw.min()))
            raw_max = max(raw_max, float(arr_raw.max()))
            arr_clip = np.clip(arr_raw, CLIP_LO, CLIP_HI)
            s = arr_clip.astype(np.float64).sum()
            ss = (arr_clip.astype(np.float64) ** 2).sum()
            sum_x  += s
            sum_xx += ss
            s_sum  += s
            s_sum_sq += ss
            n      += arr_clip.size
            s_n    += arr_clip.size
        per_split_stats[split] = {
            "n_files":  len(files),
            "n_voxels": int(s_n),
            "sum":      float(s_sum),
            "sum_sq":   float(s_sum_sq),
        }

    mu_clipped    = float(sum_x  / n)
    sigma_clipped = float(np.sqrt(sum_xx / n - mu_clipped ** 2))
    print(f"\nPool stats (after clip):")
    print(f"  n_voxels      = {n:,}")
    print(f"  μ_clipped     = {mu_clipped:+.6f}")
    print(f"  σ_clipped     = {sigma_clipped:.6f}")
    print(f"  raw range observed (pre-clip): [{raw_min:+.4f}, {raw_max:+.4f}]")

    # ── Pass 2: apply z-score, save v4 ────────────────────────────────────────
    print(f"\nPass 2: applying z-score, writing v4 to {V4_DIR} …")
    final_min, final_max = np.inf, -np.inf
    for split in ["train", "test"]:
        src = V2_DIR / split
        dst = V4_DIR / split
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.npy"))
        for f in tqdm(files, desc=f"  pass2 {split}"):
            arr_v2  = np.load(f)
            arr_raw = recover_raw(arr_v2)
            arr_clip = np.clip(arr_raw, CLIP_LO, CLIP_HI)
            arr_v4 = ((arr_clip - mu_clipped) / sigma_clipped).astype(np.float16)
            final_min = min(final_min, float(arr_v4.min()))
            final_max = max(final_max, float(arr_v4.max()))
            np.save(dst / f.name, arr_v4)

    # ── Copy {split}_available.npy ───────────────────────────────────────────
    for split in ["train", "test"]:
        src_avail = V2_DIR / f"{split}_available.npy"
        dst_avail = V4_DIR / f"{split}_available.npy"
        shutil.copy(src_avail, dst_avail)

    # ── stats.json ───────────────────────────────────────────────────────────
    stats = {
        "scheme": "pocket-pool clip + z-score (CrossDocked v4)",
        "formula": "x' = (clip(x_raw, [CLIP_LO, CLIP_HI]) - mu_clipped) / sigma_clipped",
        "source": "derived from xray_crops_aligned_v2 by inverting its z-score",
        "clip_lo_raw":   float(CLIP_LO),
        "clip_hi_raw":   float(CLIP_HI),
        "mu_clipped":    mu_clipped,
        "sigma_clipped": sigma_clipped,
        "raw_range_observed_preclip":  [raw_min, raw_max],
        "final_range_observed_postnorm": [final_min, final_max],
        "n_voxels": int(n),
        "per_split": per_split_stats,
        "v2_recovery_constants": {"v2_sigma": V2_SIGMA, "v2_mu": V2_MU},
    }
    with open(V4_DIR / "stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)
    print(f"\nWrote stats.json — final z-scored range observed: "
          f"[{final_min:+.3f}, {final_max:+.3f}]")
    print("done.")


if __name__ == "__main__":
    main()
