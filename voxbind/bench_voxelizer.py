"""Benchmark: PyUUL vs PyTorch voxelizer backends.

Runs on a small number of val batches so it doesn't compete too
heavily with an ongoing training job.

Run from the voxbind/ directory:
    python bench_voxelizer.py [--bsz 16] [--n_batches 5]
"""
import argparse
import time
import torch
import numpy as np

from voxbind.voxelizer import Voxelizer
from voxbind.dataset.crossdocked import DatasetCrossdocked


# ── helpers ──────────────────────────────────────────────────────────────────

def load_batches(bsz: int, n: int):
    """Load n+2 val batches (aug=False → deterministic coords)."""
    dset = DatasetCrossdocked(data_dir="dataset/data", split="val", aug=False)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=bsz, shuffle=False, drop_last=True, num_workers=2,
    )
    batches = []
    for i, batch in enumerate(loader):
        if i >= n + 2:   # extra 2 are used as warmup
            break
        batches.append(batch)
    return batches


def run_batches(vox, batches, key, num_channels, n_warmup=2):
    """Return per-batch latencies (ms) after warming up."""
    outputs = []
    with torch.no_grad():
        for batch in batches[:n_warmup]:
            vox(batch[key], num_channels=num_channels)
        torch.cuda.synchronize()

        for batch in batches[n_warmup:]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = vox(batch[key], num_channels=num_channels)
            torch.cuda.synchronize()
            outputs.append((out, time.perf_counter() - t0))

    voxels  = [o for o, _ in outputs]
    latency = [t * 1000 for _, t in outputs]
    return voxels, latency


def similarity(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Scalar similarity metrics between two voxel grids."""
    a, b = a.float(), b.float()
    mse  = torch.nn.functional.mse_loss(a, b).item()
    cos  = torch.nn.functional.cosine_similarity(
        a.flatten(1), b.flatten(1), dim=1
    ).mean().item()
    return {"mse": mse, "cosine": cos}


def report(label, latency, sim=None):
    mean, std = np.mean(latency), np.std(latency)
    line = f"  {label:<10} {mean:6.1f} ± {std:4.1f} ms"
    if sim:
        line += f"   (vs pyuul — MSE {sim['mse']:.2e}, cosine {sim['cosine']:.4f})"
    print(line)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz",      type=int, default=16,
                        help="Batch size (default 16 — keep small during training)")
    parser.add_argument("--n_batches", type=int, default=5,
                        help="Number of timed batches (default 5)")
    args = parser.parse_args()

    device = "cuda"
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Batch  : {args.bsz}   |   Timed batches: {args.n_batches}\n")

    batches = load_batches(args.bsz, args.n_batches)
    print(f"Loaded {len(batches)} batches from val set\n")

    vox_pyuul = Voxelizer(backend="pyuul", device=device)
    vox_torch = Voxelizer(backend="torch", device=device)

    for key, n_channels, label in [
        ("ligand", 7, "Ligand (7ch)"),
        ("pocket", 4, "Pocket (4ch)"),
    ]:
        print(f"─── {label} ───────────────────────────────────")
        vox_p, lat_p = run_batches(vox_pyuul, batches, key, n_channels)
        vox_t, lat_t = run_batches(vox_torch, batches, key, n_channels)

        # per-batch similarity (pyuul vs torch on the same input)
        sims = [similarity(a, b) for a, b in zip(vox_p, vox_t)]
        avg_sim = {k: np.mean([s[k] for s in sims]) for k in sims[0]}

        report("pyuul",  lat_p)
        report("torch",  lat_t, sim=avg_sim)
        speedup = np.mean(lat_p) / np.mean(lat_t)
        print(f"  Speedup : {speedup:.2f}x  {'(torch faster)' if speedup > 1 else '(pyuul faster)'}\n")


if __name__ == "__main__":
    main()
