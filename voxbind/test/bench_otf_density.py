"""bench_otf_density.py — is on-the-fly density cropping affordable in the dataloader?

Compares the density branch of two dataset paths on CrossDocked, isolating the cost
that differs (coordinate ops are identical for both, so excluded):

  PRECOMP (current): np.load(64³ .npy) -> rotate -> translate          [+zero-fill]
  OTF (proposed)   : _load_grid(ccp4)[LRU] -> crop_density_aug(R,t) -> normalize_crop

Reports (a) single-process per-op breakdown (cold gemmi load is the suspect cost),
and (b) realistic DataLoader throughput (samples/sec) at num_workers=8 with a
per-worker LRU map cache at two sizes (cache-miss rate is the crux under shuffling).

Run:
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python test/bench_otf_density.py
"""

import gemmi  # noqa: F401  — before torch (CXXABI)

import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

from voxbind.dataset.crossdocked_xray import (  # noqa: E402
    _load_grid, _crop_density, _rotate_density, _translate_density, normalize_crop,
)
from voxbind.utils.dataset_utils import random_rot_matrix  # noqa: E402

G, RES, DELTA = 64, 0.25, 1.0
DATA = os.path.join(REPO, "voxbind/dataset/data/data_train.pt")
CCP4_DIR = os.path.join(REPO, "voxbind/dataset/data/ccp4")
CROPS = os.path.join(REPO, "voxbind/dataset/data/xray_crops_aligned_v5/train")
AVAIL = os.path.join(REPO, "voxbind/dataset/data/xray_crops_aligned_v5/train_available.npy")
WORKERS, BSZ, N_BATCHES, WORKSET = 8, 8, 200, 3000


def pdbid(pid): return pid.split("/")[1].split("_")[0].lower()


def crop_density_aug(arr_norm, frac_T, nu, nv, nw, center, R, t):
    offs = (np.arange(G, dtype=np.float32) - G * 0.5) * RES
    gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    src = (o - t.reshape(1, 3)) @ R + center.reshape(1, 3)
    frac = src @ frac_T
    ci = (frac[:, 0] * nu) % nu; cj = (frac[:, 1] * nv) % nv; ck = (frac[:, 2] * nw) % nw
    d = scipy.ndimage.map_coordinates(arr_norm, [ci, cj, ck], order=1, mode="wrap", prefilter=False)
    return d.reshape(G, G, G).astype(np.float32)


def build_workset():
    """Positions (in the filtered train order) that have BOTH a precomputed crop and a map."""
    data = torch.load(DATA, weights_only=False)
    random.Random(1234).shuffle(data)
    data = data[: len(data) - 100]
    data = [(p, l) for p, l in data if l["max_len"] <= 30]            # filtered order == crop layout
    avail = np.load(AVAIL)
    ws = []
    for i, (p, l) in enumerate(data):
        if len(ws) >= WORKSET:
            break
        if not avail[i]:
            continue
        pid = pdbid(p["id"])
        if not os.path.exists(os.path.join(CCP4_DIR, f"{pid}.ccp4")):
            continue
        if not os.path.exists(os.path.join(CROPS, f"{i:06d}.npy")):
            continue
        m = l["atoms_channel"] < 7
        center = l["coords"][m].numpy().astype(np.float32).mean(axis=0)
        ws.append((i, pid, center))
    return ws


class PrecompDS(Dataset):
    def __init__(self, ws): self.ws = ws
    def __len__(self): return len(self.ws)
    def __getitem__(self, k):
        i, _, _ = self.ws[k]
        d = np.load(os.path.join(CROPS, f"{i:06d}.npy")).astype(np.float32)
        R = random_rot_matrix(); t = (torch.rand((1, 3)) - 0.5) * 2 * DELTA
        d = _translate_density(_rotate_density(d, R), t)
        return torch.from_numpy(d)


class OTFDS(Dataset):
    def __init__(self, ws, cache_size): self.ws = ws; self.cache_size = cache_size; self.cache = OrderedDict()
    def __len__(self): return len(self.ws)
    def _grid(self, pid):
        if pid in self.cache:
            self.cache.move_to_end(pid); return self.cache[pid]
        g = _load_grid(Path(CCP4_DIR) / f"{pid}.ccp4")
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[pid] = g
        return g
    def __getitem__(self, k):
        i, pid, center = self.ws[k]
        g = self._grid(pid)
        R = random_rot_matrix().numpy().astype(np.float32)
        t = ((torch.rand((1, 3)) - 0.5) * 2 * DELTA).numpy().astype(np.float32)
        if g is None:
            return torch.zeros(G, G, G)
        arr, frac_T, nu, nv, nw = g
        d = normalize_crop(crop_density_aug(arr, frac_T, nu, nv, nw, center, R.T, t))
        return torch.from_numpy(d)


def micro(ws):
    print("── single-process per-op (ms/sample, mean of 40) ──")
    sub = ws[:40]
    # precomp
    tl = tr = 0.0
    for i, pid, c in sub:
        t0 = time.perf_counter(); d = np.load(os.path.join(CROPS, f"{i:06d}.npy")).astype(np.float32); tl += time.perf_counter()-t0
        R = random_rot_matrix(); t = (torch.rand((1,3))-0.5)*2*DELTA
        t0 = time.perf_counter(); _translate_density(_rotate_density(d, R), t); tr += time.perf_counter()-t0
    n = len(sub)
    print(f"  PRECOMP : np.load {1e3*tl/n:6.2f}  + rotate+translate {1e3*tr/n:6.2f}  = {1e3*(tl+tr)/n:6.2f}")
    # otf (cold every time — no cache)
    tg = tc = 0.0; ok = 0
    for i, pid, c in sub:
        t0 = time.perf_counter(); g = _load_grid(Path(CCP4_DIR)/f"{pid}.ccp4"); tg += time.perf_counter()-t0
        if g is None: continue
        ok += 1; arr, fT, nu, nv, nw = g
        R = random_rot_matrix().numpy(); t = ((torch.rand((1,3))-0.5)*2*DELTA).numpy()
        t0 = time.perf_counter(); normalize_crop(crop_density_aug(arr, fT, nu, nv, nw, c, R.T, t)); tc += time.perf_counter()-t0
    print(f"  OTF cold: _load_grid {1e3*tg/n:6.2f}  + crop+normalize {1e3*tc/max(ok,1):6.2f}  = {1e3*(tg+tc)/n:6.2f}  (interp dominates, NOT the load → caching barely helps)")
    print(f"  OTF warm: (cache hit) crop+normalize only        = {1e3*tc/max(ok,1):6.2f}")


def throughput(name, ds):
    dl = DataLoader(ds, batch_size=BSZ, num_workers=WORKERS, shuffle=True,
                    persistent_workers=True, prefetch_factor=2)
    it = iter(dl)
    next(it)                                   # warmup (worker spinup) — not timed
    t0 = time.perf_counter(); n = 0
    for _ in range(N_BATCHES):
        try: b = next(it)
        except StopIteration: break
        n += b.shape[0]
    dt = time.perf_counter() - t0
    del it, dl
    print(f"  {name:<28} {n/dt:7.1f} samples/s   ({n} samples in {dt:5.1f}s, {WORKERS} workers)")
    return n / dt


def main():
    print(f"building work set (≤{WORKSET} CrossDocked samples with crop+map)…")
    ws = build_workset()
    print(f"  work set: {len(ws)} samples\n")
    micro(ws)
    print(f"\n── DataLoader throughput (bsz={BSZ}, {N_BATCHES} batches) ──")
    r_pre = throughput("PRECOMP (current)", PrecompDS(ws))
    r_o32 = throughput("OTF (cache=32)", OTFDS(ws, 32))
    r_o512 = throughput("OTF (cache=512)", OTFDS(ws, 512))

    # training consumption target: ~97s/epoch, ~17.3k samples, 4 ranks → per-rank loader rate
    per_rank = 17330 / 97 / 4
    print(f"\n── verdict ──")
    print(f"  per-rank training consumption ≈ {per_rank:.0f} samples/s (4-GPU PLINDER C+D+G, ~97s/epoch)")
    print(f"  PRECOMP {r_pre:.0f}/s | OTF cache32 {r_o32:.0f}/s | OTF cache512 {r_o512:.0f}/s")
    best = max(r_o32, r_o512)
    print(f"  OTF/PRECOMP ratio: {best/r_pre:.2f}×")
    print(f"  affordable (OTF ≥ training need)? {'YES' if best >= per_rank else 'NO'}"
          f"   |  hides behind GPU (OTF ≥ PRECOMP)? {'YES' if best >= r_pre else 'NO (slower, but maybe still ≥ need)'}")


if __name__ == "__main__":
    main()
