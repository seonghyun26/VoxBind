"""smoke_resample_reader.py — does the resample-manifest actually drive a train-time reader?

Consumes ONLY the artifacts produced by `00b --mode resample` (dataset/data/xray_resample_v5/):
  {split}_manifest.npz  (pdb_id, centroid, Kabsch R/t, ok)  +  resample.json (norm recipe)  +  the CCP4 maps.
Builds a minimal train-time Dataset that, per sample, samples the RAW map at the AUGMENTED pose
    cart = (centroid + R_aug.T·(o − t_aug)) @ R_kabschᵀ + t_kabsch
applies the recipe's v5 normalization, and returns a 64³ density. Atom coords come from data_test.pt
(positionally aligned with the manifest), exactly as the real dataset would pair them.

Checks:
  1. DataLoader (workers) yields batches with no error — finite, right shape.
  2. aug-OFF reader density == precomputed v5 crop  (corr ~1.0)  → manifest+recipe reproduce v5.
  3. aug-ON: 0% zero-fill, finite, and POCKET co-registration preserved (≈ aug-off) → geometry correct.

Run:
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python test/smoke_resample_reader.py
"""

import gemmi  # before torch

import importlib.util
import json
import os
import sys

import numpy as np
import scipy.ndimage

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402
from voxbind.utils.dataset_utils import random_rot_matrix  # noqa: E402

# load_raw_grid lives in the numbered 00b module
_spec = importlib.util.spec_from_file_location("p00b", os.path.join(REPO, "voxbind/dataset/00b_density_preprocess.py"))
m00b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m00b)

D = os.path.join(REPO, "voxbind/dataset/data")
RDIR = os.path.join(D, "xray_resample_v5")
G, RES, DELTA, SPLIT = 64, 0.25, 1.0, "test"
_OFFS = (np.arange(G) - G * 0.5) * RES


def load_split_test():
    data = torch.load(os.path.join(D, "data_test.pt"), weights_only=False)
    return [(p, l) for p, l in data if l["max_len"] <= 30]   # test = no shuffle (matches 00b)


def resample_box(arr, fT, nu, nv, nw, center, Rk, tk, Raug, taug):
    """Sample raw map onto a 64³ box at the augmented pose. Raug=I,taug=0 → plain v5 crop."""
    gx, gy, gz = np.meshgrid(_OFFS, _OFFS, _OFFS, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float64)
    o_pre = (o - taug.reshape(1, 3)) @ Raug                       # row form of R_augᵀ·(o − t)
    cart = (center.reshape(1, 3) + o_pre) @ Rk.T + tk.reshape(1, 3)
    frac = cart @ fT
    ci = (frac[:, 0] * nu) % nu; cj = (frac[:, 1] * nv) % nv; ck = (frac[:, 2] * nw) % nw
    d = scipy.ndimage.map_coordinates(arr, [ci, cj, ck], order=1, mode="wrap", prefilter=False)
    return d.reshape(G, G, G)


class ResampleReader(Dataset):
    """Train-time reader driven entirely by the resample manifest + recipe + maps."""
    def __init__(self, split, aug=True):
        self.recipe = json.load(open(os.path.join(RDIR, "resample.json")))
        n = self.recipe["normalization"]
        self.s, self.mu, self.sig = n["arcsinh_scale"], n["mu_a"], n["sigma_a"]
        self.ccp4 = self.recipe["ccp4_dir"]
        mf = np.load(os.path.join(RDIR, f"{split}_manifest.npz"))
        self.pdb_id, self.centroid, self.R, self.t, self.ok = (
            mf["pdb_id"], mf["centroid"], mf["R"], mf["t"], mf["ok"])
        self.idxs = np.where(self.ok)[0]
        self.aug = aug
        self._cache = {}

    def __len__(self): return len(self.idxs)

    def _grid(self, pid):
        if pid not in self._cache:
            self._cache[pid] = m00b.load_raw_grid(os.path.join(self.ccp4, f"{pid}.ccp4"))
        return self._cache[pid]

    def density(self, i, Raug, taug):
        g = self._grid(self.pdb_id[i])
        if g is None:
            return None
        arr, fT, nu, nv, nw = g
        raw = resample_box(arr, fT, nu, nv, nw, self.centroid[i].astype(float),
                           self.R[i].astype(float), self.t[i].astype(float), Raug, taug)
        return ((np.arcsinh(raw / self.s) - self.mu) / self.sig).astype(np.float32)   # v5 norm

    def __getitem__(self, k):
        i = self.idxs[k]
        if self.aug:
            Raug = random_rot_matrix().numpy().astype(np.float64)
            taug = ((np.random.rand(3) - 0.5) * 2 * DELTA)
        else:
            Raug, taug = np.eye(3), np.zeros(3)
        d = self.density(i, Raug, taug)
        return torch.zeros(G, G, G) if d is None else torch.from_numpy(d)


def density_at_atoms(crop, offsets):
    idx = np.round(offsets / RES + G * 0.5).astype(int)
    inb = np.all((idx >= 0) & (idx < G), axis=1); idx = idx[inb]
    return float(crop[idx[:, 0], idx[:, 1], idx[:, 2]].mean()) if len(idx) else float("nan")


def main():
    samples = load_split_test()
    ds = ResampleReader(SPLIT, aug=True)
    print(f"reader: {len(ds)} available / {len(samples)} samples  | recipe v5 "
          f"s={ds.s} mu_a={ds.mu:.4f} sig_a={ds.sig:.4f}\n")

    # 1. DataLoader (workers) yields batches, no error
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True)
    b = next(iter(dl))
    print(f"[1] DataLoader batch: shape {tuple(b.shape)}  finite={bool(torch.isfinite(b).all())}  "
          f"mean {b.mean():.3f}  std {b.std():.3f}")

    # 2/3. correctness on a handful of available samples
    rng = np.random.RandomState(0)
    corrs, zfracs, zoff, zon = [], [], [], []
    checked = 0
    for k in range(len(ds)):
        i = ds.idxs[k]
        cf = os.path.join(D, f"xray_crops_aligned_v5/{SPLIT}/{i:06d}.npy")
        if not os.path.exists(cf):
            continue
        # aug-OFF vs precomputed v5 crop
        d_off = ds.density(i, np.eye(3), np.zeros(3))
        if d_off is None:
            continue
        ref = np.load(cf).astype(np.float32)
        corrs.append(float(np.corrcoef(d_off.ravel(), ref.ravel())[0, 1]))

        # aug-ON: zero-fill + pocket co-registration (preserved vs aug-off)
        Raug = random_rot_matrix().numpy().astype(np.float64)
        taug = (rng.rand(3) - 0.5) * 2 * DELTA
        d_on = ds.density(i, Raug, taug)
        zfracs.append(float((d_on == 0).mean()))
        poc = samples[i][0]["coords"][samples[i][0]["atoms_channel"] < 4].numpy().astype(float)
        pp = poc - ds.centroid[i].astype(float)
        zoff.append(density_at_atoms(d_off, pp))
        zon.append(density_at_atoms(d_on, pp @ Raug.T + taug))      # atoms aug: p' = p @ Raugᵀ + t
        checked += 1
        if checked >= 12:
            break

    corr_m, zf_m = float(np.mean(corrs)), float(np.mean(zfracs))
    zoff_m, zon_m = float(np.nanmean(zoff)), float(np.nanmean(zon))
    print(f"[2] aug-OFF vs precomputed v5 crop : mean corr {corr_m:.6f}  (min {np.min(corrs):.6f}, n={len(corrs)})")
    print(f"[3] aug-ON  zero-fill frac          : {100*zf_m:.3f}%")
    print(f"    pocket density z  aug-off {zoff_m:.3f} → aug-on {zon_m:.3f}  (co-registration preserved)")

    checks = {
        "DataLoader batch finite":          bool(torch.isfinite(b).all()),
        "aug-off reproduces v5 (corr>.999)": np.min(corrs) > 0.999,
        "aug-on no zero-fill (<0.5%)":        zf_m < 0.005,
        "pocket co-reg preserved (≈ off)":    zon_m > 0.7 * zoff_m,
    }
    print("\nchecks:")
    for name, okk in checks.items():
        print(f"  [{'PASS' if okk else 'FAIL'}] {name}")
    ok = all(checks.values())
    print(f"\n{'== RESAMPLE READER SMOKE PASSED ==' if ok else '== FAILED =='}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
