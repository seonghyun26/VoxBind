"""00c_make_noise_density.py — Smoothed-Gaussian-noise density crops (A/B control)

Generates a drop-in replacement for dataset/data/xray_crops/ in which every
crop is *structureless* Gaussian noise, low-pass filtered to reproduce the
spatial autocorrelation length of the real 2Fo-Fc electron-density crops.

Purpose
-------
Control arm for the X-ray density conditioning experiment. The noise crops
carry NO information about the ligand/pocket — only the density-like
*texture* (smoothness). A model trained on these vs. the real-density model
(exps/260515_voxbind_100ep_density) isolates whether the density's *content*
matters, beyond merely having a smooth per-sample conditioning channel.

Design
------
* Only the crops used by the 100-train + 100-val subset are generated: the
  first n_subset x-ray-available indices, exactly matching
  DatasetCrossDockedXray's subset_n / subset_val_n selection.
  {train,test}_available.npy are copied verbatim so subset selection picks
  the identical 200 pockets.
* The Gaussian blur sigma is calibrated FROM the real crops: the radially
  averaged spatial autocorrelation of the real density is measured and the
  blur is chosen to reproduce its 1/e correlation length.
* One fixed noise volume per sample (static) — mirroring the real density,
  which is one fixed map per PDB. Rotation augmentation is still applied
  on-the-fly by the dataset, identically to the real run.

Usage
-----
    cd voxbind
    python dataset/00c_make_noise_density.py \
        --real_crops_dir dataset/data/xray_crops \
        --out_dir        dataset/data/xray_crops_noise \
        --n_subset 200 --seed 42
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.ndimage

# normalize_crop is what the dataset applies at load time (normalize=true);
# measuring on normalize_crop'd volumes matches exactly what the model sees.
from voxbind.dataset.crossdocked_xray import normalize_crop

_G = 64                       # grid dimension (matches voxbind/configs/vox/default.yaml)
_RES = 0.25                   # Å per voxel
_LEVEL = float(np.exp(-1))    # 1/e autocorrelation level → "correlation length"


# ── Texture measurement ─────────────────────────────────────────────────────────

def _radial_acf(vol: np.ndarray):
    """Radially-averaged normalized autocorrelation of a 3D volume.

    Returns (radii, profile) with profile[0] == 1.0 at zero lag.
    """
    v = vol.astype(np.float64)
    v = v - v.mean()
    power = np.abs(np.fft.fftn(v)) ** 2          # |FFT|² → real, Hermitian
    acf = np.fft.fftshift(np.fft.ifftn(power).real)
    G = v.shape[0]
    c = G // 2
    acf /= acf[c, c, c]                          # zero lag → 1.0

    ax = np.arange(G) - c
    rr = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    r_int = np.minimum(rr.round().astype(int), c)
    nb = c + 1
    s = np.bincount(r_int.ravel(), weights=acf.ravel(), minlength=nb)[:nb]
    n = np.bincount(r_int.ravel(), minlength=nb)[:nb]
    return np.arange(nb), s / np.maximum(n, 1)


def _corr_length(radii: np.ndarray, prof: np.ndarray, level: float = _LEVEL) -> float:
    """First lag (voxels) where the radial ACF crosses `level`, linearly interpolated."""
    below = np.flatnonzero(prof <= level)
    if below.size == 0:
        return float(radii[-1])
    i = int(below[0])
    if i == 0:
        return 0.0
    p0, p1 = prof[i - 1], prof[i]
    return float(radii[i - 1] + (level - p0) * (radii[i] - radii[i - 1]) / (p1 - p0))


def _mean_acf(crops):
    """Mean radial ACF profile over an iterable (generator-safe) of 3D volumes."""
    acc, radii, n = None, None, 0
    for v in crops:
        radii, prof = _radial_acf(v)
        acc = prof if acc is None else acc + prof
        n += 1
    return radii, acc / n


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--real_crops_dir", default="dataset/data/xray_crops",
                    help="Real density crops dir (e.g. xray_crops_aligned/, from dataset/00b_density_preprocess.py)")
    ap.add_argument("--out_dir", default="dataset/data/xray_crops_noise",
                    help="Output dir for the smoothed-noise crops")
    ap.add_argument("--n_subset", type=int, default=200,
                    help="Crops to generate = subset_n + subset_val_n (first N x-ray-available)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the noise (one fixed volume per sample)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Override blur sigma in voxels; default = calibrated from real crops")
    ap.add_argument("--also_test", action="store_true",
                    help="Also generate content-free noise crops for every x-ray-available "
                         "test pocket (out_dir/test/), reusing the train-calibrated sigma — "
                         "lets the noise-control model sample the held-out test split.")
    args = ap.parse_args()

    real_dir = Path(args.real_crops_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)

    print("=== Smoothed-Gaussian-noise density crops ===")
    print(f"  real crops : {real_dir.resolve()}")
    print(f"  out dir    : {out_dir.resolve()}")

    # ── used indices: first n_subset x-ray-available samples (matches dataset) ──
    avail = np.load(real_dir / "train_available.npy")
    pool = np.flatnonzero(avail)
    if pool.size < args.n_subset:
        raise SystemExit(f"only {pool.size} x-ray-available samples, need {args.n_subset}")
    used = pool[: args.n_subset]
    print(f"  used idx   : {args.n_subset}  (range {used[0]}..{used[-1]})")

    # ── measure real density texture ───────────────────────────────────────────
    real = (normalize_crop(np.load(real_dir / "train" / f"{i:06d}.npy")) for i in used)
    radii, prof_real = _mean_acf(real)
    L_real = _corr_length(radii, prof_real)
    print(f"  real density corr length : {L_real:.3f} vox  ({L_real * _RES:.2f} Å)")

    # ── calibrate blur sigma (analytic guess + one linear correction step) ─────
    if args.sigma is not None:
        sigma = args.sigma
        print(f"  blur sigma : {sigma:.3f} vox  (user override)")
    else:
        cal_rng = np.random.default_rng(args.seed + 1)   # separate stream — keeps crops deterministic
        sigma0 = max(L_real / 2.0, 1e-3)
        test = (scipy.ndimage.gaussian_filter(cal_rng.standard_normal((_G,) * 3), sigma0)
                for _ in range(16))
        _, prof_t = _mean_acf(test)
        L0 = _corr_length(radii, prof_t)
        sigma = sigma0 * (L_real / max(L0, 1e-3))
        print(f"  calibration: sigma0={sigma0:.3f} → L0={L0:.3f} → sigma={sigma:.3f} vox")

    # ── generate + save noise crops (one fixed volume per sample) ──────────────
    rng = np.random.default_rng(args.seed)
    for i in used:
        white = rng.standard_normal((_G, _G, _G))
        blurred = scipy.ndimage.gaussian_filter(white, sigma)
        blurred = (blurred - blurred.mean()) / (blurred.std() + 1e-8)   # O(1) for float16 headroom
        np.save(out_dir / "train" / f"{int(i):06d}.npy", blurred.astype(np.float16))

    # ── optional: content-free noise crops for the test split ──────────────────
    # The noise carries no information, so one fixed smoothed-noise volume per
    # x-ray-available test pocket (reusing the calibrated train sigma) is enough
    # to let the noise-control model sample the held-out test split, which
    # otherwise has no xray_crops_noise/test entries.
    if args.also_test:
        (out_dir / "test").mkdir(parents=True, exist_ok=True)
        test_idx = np.flatnonzero(np.load(real_dir / "test_available.npy"))
        rng_test = np.random.default_rng(args.seed + 100)
        for i in test_idx:
            white = rng_test.standard_normal((_G, _G, _G))
            blurred = scipy.ndimage.gaussian_filter(white, sigma)
            blurred = (blurred - blurred.mean()) / (blurred.std() + 1e-8)
            np.save(out_dir / "test" / f"{int(i):06d}.npy", blurred.astype(np.float16))
        print(f"  wrote {len(test_idx)} test crops → {out_dir / 'test'}")

    # ── copy availability masks verbatim → identical subset selection ──────────
    for f in ("train_available.npy", "test_available.npy"):
        if (real_dir / f).exists():
            shutil.copy2(real_dir / f, out_dir / f)

    # ── verify generated texture matches the real density ──────────────────────
    gen = (normalize_crop(np.load(out_dir / "train" / f"{int(i):06d}.npy")) for i in used)
    _, prof_gen = _mean_acf(gen)
    L_gen = _corr_length(radii, prof_gen)
    print(f"  noise crop corr length   : {L_gen:.3f} vox  ({L_gen * _RES:.2f} Å)")
    print(f"  radial ACF r=1..5  real  : {np.array2string(prof_real[1:6], precision=3)}")
    print(f"  radial ACF r=1..5  noise : {np.array2string(prof_gen[1:6], precision=3)}")

    meta = {
        "source_crops": str(real_dir.resolve()),
        "n_crops": int(args.n_subset),
        "used_index_range": [int(used[0]), int(used[-1])],
        "seed": args.seed,
        "blur_sigma_vox": round(float(sigma), 4),
        "corr_length_real_vox": round(L_real, 4),
        "corr_length_noise_vox": round(L_gen, 4),
        "resolution_A": _RES,
        "note": "structureless smoothed-Gaussian-noise control for the X-ray density A/B test",
    }
    (out_dir / "noise_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  wrote {args.n_subset} crops → {out_dir / 'train'}")
    print(f"  meta → {out_dir / 'noise_meta.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
