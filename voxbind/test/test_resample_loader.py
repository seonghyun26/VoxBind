"""Isolation smoke-test for the on-the-fly RESAMPLE dataloader.

WHY: the resample path (density cropped from the full CCP4 map at the augmented
pose) parses big maps from a 54GB pool. With too many workers x a big per-worker
CCP4 cache the box thrashes and the c10d rendezvous times out BEFORE batch 0 even
lands (killed the 260616 resample run + the first channelvit-uniform-resample launch).
This script measures the REAL cost cheaply (single process, then a few workers) so we
never again discover a stall by burning an 11-minute 4-GPU rendezvous timeout.

Run (once the voxbind env is back):
    /opt/conda/envs/voxbind/bin/python test/test_resample_loader.py [config_name]
Pass/launch gate: a worker-backed batch should arrive in well under the c10d timeout
(~60s) — aim for < ~20s/batch warm.
"""
import sys, time
from hydra import compose, initialize
import torch
from torch.utils.data import DataLoader
from voxbind.dataset import _make_dataset

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "config_train_atomblob_density_gradmag_channelvit_uniform_resample_v5"

with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name=CFG)
print(f"cfg={CFG}  resample_dir={cfg.dset.get('resample_dir','')}  "
      f"cache_size={cfg.dset.get('cache_size')}  num_workers(cfg)={cfg.num_workers}  "
      f"prefetch={cfg.get('prefetch_factor')}")

ds = _make_dataset(cfg, split="train", aug=cfg.aug)
print(f"dataset={type(ds).__name__}  len={len(ds)}")

# 1) single-process per-sample cost (cold cache, distinct maps)
N = 6
t0 = time.time()
for i in range(N):
    ti = time.time(); _ = ds[i]
    print(f"  sample {i}: {time.time()-ti:.2f}s")
print(f"single-proc: {(time.time()-t0)/N:.2f}s/sample over {N} cold samples")

# 2) worker-backed batch latency at the config's batch size (the real DDP cost)
for nw in (4, int(cfg.num_workers)):
    dl = DataLoader(ds, batch_size=cfg.bsz, shuffle=True, drop_last=True,
                    num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
                    prefetch_factor=int(cfg.get("prefetch_factor", 4)) if nw > 0 else None)
    it = iter(dl)
    t0 = time.time(); b0 = next(it); t_first = time.time() - t0
    t0 = time.time(); _ = next(it); t_warm = time.time() - t0
    keys = list(b0.keys()) if isinstance(b0, dict) else type(b0).__name__
    print(f"  num_workers={nw}: first batch {t_first:.1f}s  warm batch {t_warm:.1f}s  bsz={cfg.bsz}")
    del dl, it
print("OK")
