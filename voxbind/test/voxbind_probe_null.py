"""voxbind_probe_null.py — the trivial baselines every frozen-encoder probe rho sits on top of.

The L0 result (one conv layer, spatially mean-pooled -> rho 0.578) is too high to be
"representation quality": pK correlates strongly with ligand size, and any spatially
mean-pooled feature encodes occupancy mass. Before reading 0.596 (coords) vs 0.644 (CDG)
vs the U-Net numbers as representation differences, we need the floor they share.

Three nulls, same split / head / seeds as every other probe:
  atomsum11 : mean-pooled RAW input voxels (7 ligand-element + 4 pocket-element) -> 11 dims.
              No network at all. Pure per-element occupancy mass.
  ligmass1  : summed ligand occupancy only -> 1 dim. A heavy-atom-count proxy.
  pocmass1  : summed pocket occupancy only -> 1 dim. Should be near zero if the probe is
              measuring ligand chemistry rather than pocket/box size.

Run:
    python test/voxbind_probe_null.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
VOX = REPO / "voxbind"
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

ATOM_DIR = VOX / "dataset" / "data" / "pdbbind" / "voxels_v5" / "atoms"
CACHE = VOX / "dataset" / "data" / "pdbbind" / "features" / "voxbind_null_atomsum.pt"
SPLIT = "lp_edrscc_v2"
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)


class _Sums(Dataset):
    def __init__(self, pids):
        self.pids = list(pids)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        try:
            a = np.load(ATOM_DIR / f"{pid}.npy").astype(np.float32)   # (11, G, G, G)
            return pid, torch.from_numpy(a.mean(axis=(1, 2, 3))), ""
        except Exception as e:
            return pid, None, repr(e)[:120]


def _collate(b):
    good = [(p, x) for p, x, e in b if x is not None]
    if not good:
        return [], None
    p, x = zip(*good)
    return list(p), torch.stack(x, 0)


def build_cache(pids):
    loader = DataLoader(_Sums(pids), batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=_collate)
    out, seen = {}, 0
    for bp, x in loader:
        if x is None:
            continue
        for pid, v in zip(bp, x):
            out[pid] = v.clone()
        seen += len(bp)
        if seen % 960 == 0:
            print(f"    {seen}/{len(pids)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": out}, CACHE)
    return out


def run(feats, lp_df, split_map, label, dev, seeds=1):
    data = pr.build_dataset(feats, lp_df, drop_covalent=True, cl1_only=False,
                            split_map=split_map)
    mu = data["train"]["X"].mean(0, keepdims=True)
    sd = data["train"]["X"].std(0, keepdims=True) + 1e-6
    for k in ("train", "val", "test"):
        data[k]["X"] = (data[k]["X"] - mu) / sd
    ms = [pr.train_one(data, seed=s, device=dev, **HP) for s in range(seeds)]
    g = lambda k: np.array([m[k] for m in ms])
    r = dict(feature=label, dim=data["train"]["X"].shape[1],
             n_train=ms[0]["n_train"], n_test=ms[0]["n_test"],
             rho=g("test_spearman").mean(), rho_sd=g("test_spearman").std(),
             rmse=g("test_rmse").mean())
    print(f"  {label:<12} dim{r['dim']:>3}  rho {r['rho']:.3f}+-{r['rho_sd']:.3f}  "
          f"rmse {r['rmse']:.3f}", flush=True)
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp_df = pr.load_lp_index(pr.LP_CSV)
    split_map, scheme = pr.load_frozen_split_map(SPLIT)
    have = {p.stem for p in ATOM_DIR.glob("*.npy")}
    pids = sorted(p for p in split_map if p in have)
    print(f"  split {scheme}: {len(pids):,} complexes\n")

    if CACHE.exists():
        sums = torch.load(CACHE, map_location="cpu", weights_only=False)["features"]
        print(f"  cache hit: {CACHE.name} ({len(sums):,})")
    else:
        sums = build_cache(pids)

    rows = [
        run(sums, lp_df, split_map, "atomsum11", dev),
        run({p: v[:7].sum(dim=0, keepdim=True) for p, v in sums.items()},
            lp_df, split_map, "ligmass1", dev),
        run({p: v[7:11].sum(dim=0, keepdim=True) for p, v in sums.items()},
            lp_df, split_map, "pocmass1", dev),
        # matched null for the ligand-free ("noise") U-Net condition: per-element POCKET
        # composition only. The right floor to judge a pocket-only representation against.
        run({p: v[7:11].clone() for p, v in sums.items()},
            lp_df, split_map, "pocsum4", dev),
        run({p: v[:7].clone() for p, v in sums.items()},
            lp_df, split_map, "ligsum7", dev),
    ]
    df = pd.DataFrame(rows)
    csv = VOX / "test" / "results" / "voxbind_probe_null.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"\n  reference (same split): CDG champion 0.644 | coords-only 0.596")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
