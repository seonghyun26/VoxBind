"""verify_plinder_big_alignment.py — prove crop[i] ↔ tuple[i] for the big PLINDER build.

The dataset loads density crops BY POSITION ({i:06d}.npy) after replaying
Random(1234).shuffle → drop last 100 → filter max_len<=30. 03c lays the crops in
that SAME order. If the two ever disagree, atoms and density are SILENTLY scrambled
(garbage pretraining, no error) — the documented hazard in memory. This script
re-derives the train order from data_train_plinder_big.pt, re-crops the RAW 2Fo-Fc
map at each sampled position's ligand centroid, and checks it equals the on-disk crop.

    python dataset/verify_plinder_big_alignment.py            # 12 sampled positions
    python dataset/verify_plinder_big_alignment.py --n 40
"""
import gemmi  # noqa: F401  before torch (shared-lib load order)
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

VOX_ROOT = Path(__file__).resolve().parents[1]            # .../voxbind
VOXDATA = VOX_ROOT / "dataset" / "data"
import sys
sys.path.insert(0, str(VOX_ROOT.parent))
from voxbind.dataset.crossdocked_xray import _crop_density  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


PDBB = load_module(VOX_ROOT / "dataset" / "legacy" / "01b_pdbbind_preprocess.py", "pdbb01b")
GRID_DIM, RESOLUTION, VAL_SZ, MAX_LEN = 64, 0.25, 100, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_big")
    ap.add_argument("--n", type=int, default=12, help="sampled positions to verify")
    ap.add_argument("--v5_stats", default=str(VOXDATA / "xray_crops_aligned_v5" / "stats.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops_dir = VOXDATA / f"xray_crops_aligned_plinder{args.tag}"
    data_file = VOXDATA / f"data_train_plinder{args.tag}.pt"
    ccp4_dir = VOXDATA / "plinder" / "ccp4"

    v5 = json.loads(Path(args.v5_stats).read_text())
    s5, mu_a, sigma_a = float(v5["arcsinh_scale"]), float(v5["mu_a"]), float(v5["sigma_a"])
    def norm(raw):
        return ((np.arcsinh(raw.astype(np.float64) / s5) - mu_a) / sigma_a).astype(np.float16)

    # ── replay the EXACT dataset train laydown (== 03c crop order) ──
    data = torch.load(str(data_file), weights_only=False)
    random.Random(1234).shuffle(data)
    train = data[: len(data) - VAL_SZ]                       # drop last 100 (val)
    train = [(p, l) for p, l in train if l["max_len"] <= MAX_LEN]
    n_total = len(train)

    avail = np.load(str(crops_dir / "train_available.npy"))
    stats = json.loads((crops_dir / "stats.json").read_text())
    print(f"data_file tuples: {len(data)}  → train(after shuffle/drop/filter): {n_total}")
    print(f"train_available.npy: {len(avail)}   stats.n_train: {stats['n_train']}")
    ok_len = (len(avail) == n_total == stats["n_train"])
    print(f"  length alignment: {'OK' if ok_len else 'MISMATCH !!'}")

    n_files = len(list((crops_dir / "train").glob("*.npy")))
    print(f"crop files on disk: {n_files}")

    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(n_total), min(args.n, n_total)))
    print(f"\nverifying {len(idxs)} positions (re-crop raw map vs on-disk crop):")
    n_ok = 0
    for i in idxs:
        pocket_, ligand_ = train[i]
        key = str(ligand_["id"]).split("/")[-1]             # "{pid}_{asym}"
        pid = key.split("_")[0].lower()
        center = ligand_["coords"].to(torch.float32).mean(dim=0).numpy().astype(np.float64)
        disk = np.load(str(crops_dir / "train" / f"{i:06d}.npy")).astype(np.float32)

        grid = PDBB.load_raw_grid(ccp4_dir / f"{pid}.ccp4")
        if grid is None:
            print(f"  [{i:6d}] {key:14s} RAW MAP MISSING — skip"); continue
        arr, frac_T, nu, nv, nw = grid
        recrop = norm(_crop_density(arr, frac_T, nu, nv, nw, center,
                                    G=GRID_DIM, res=RESOLUTION, transform=None)).astype(np.float32)
        corr = float(np.corrcoef(disk.ravel(), recrop.ravel())[0, 1])
        mad = float(np.abs(disk - recrop).max())
        good = corr > 0.999 and mad < 0.05
        n_ok += good
        print(f"  [{i:6d}] {key:14s} corr={corr:.5f} maxabsdiff={mad:.4f} {'OK' if good else 'FAIL <<<'}")

    print(f"\n{n_ok}/{len(idxs)} positions match exactly. "
          + ("ALIGNMENT OK ✓" if (n_ok == len(idxs) and ok_len) else "ALIGNMENT PROBLEM ✗"))


if __name__ == "__main__":
    main()
