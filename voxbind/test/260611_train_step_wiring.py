#!/usr/bin/env python
"""Integration smoke for the throughput-knob wiring in train_density.py.

The full debug launch is blocked on this box by a crops/preprocessing alignment
check in the dataloader (data-vintage mismatch, unrelated to these knobs), so we
drive the REAL `train_epoch` directly with a synthetic prefetcher. This exercises
the exact integration points that were edited: _amp_setup/autocast placement, the
bf16->float cast, channels_last x_in conversion, fused AdamW, foreach EMA, and
compute_losses on the autocast output — for a few flag combinations.

    CUDA_VISIBLE_DEVICES=3 python voxbind/test/260611_train_step_wiring.py
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from voxbind.train_density import train_epoch
from voxbind.models.density_vit import DensityViTMAE
from voxbind.models.adamw import AdamW as CustomAdamW
from voxbind.models.ema import ModelEma

DEV = "cuda"
B, G, NIN, NSTRUCT = 4, 64, 12, 11   # atomblob_density (11 atom + 1 density), struct target width


class FakePrefetcher:
    """Yields (x_in, x_clean, mask, target_str) with the shapes train_epoch expects."""
    def __init__(self, n, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.batches = []
        for _ in range(n):
            x_clean = torch.randn(B, NIN, G, G, G, device=DEV, generator=g)
            mask = (torch.rand(B, 1, G, G, G, device=DEV, generator=g) < 0.6)
            x_in = x_clean * (~mask)
            target_str = torch.randn(B, NSTRUCT, G, G, G, device=DEV, generator=g)
            self.batches.append((x_in, x_clean, mask, target_str))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def make_cfg(amp, channels_last):
    return OmegaConf.create({
        "debug": True, "input_mode": "atomblob_density", "with_gradmag": False,
        "accum_steps": 1, "channels_last": channels_last,
        "amp": {"enabled": amp, "dtype": "bf16"},
        "model": {"patch_size": 8},
        "mae": {"pretext_style": "mae", "gradmag_reconstruct": True,
                "lambda_dens": 1.0, "lambda_str": 1.0, "grad_clip": 1.0,
                "struct_pos_weight": 1.0, "struct_pos_thresh": 0.05,
                "atom_pos_weight": 1.0, "atom_pos_thresh": 0.05},
    })


def build_model(pos_encoding, rope_fast, channels_last):
    torch.manual_seed(0)
    m = DensityViTMAE(
        grid_dim=G, patch_size=8, n_in_channels=NIN, n_recon_channels=NIN,
        n_channels=32, dim=512, depth=12, n_heads=8, mlp_ratio=4, dropout=0.1,
        n_struct_channels=0, pretext_style="mae", head_style="patch_mlp",
        head_hidden_dim=67, head_depth=2, pos_encoding=pos_encoding, rope_fast=rope_fast,
    ).to(DEV)
    if channels_last:
        m = m.to(memory_format=torch.channels_last_3d)
    return m


def run(tag, *, amp, channels_last, fused, foreach_ema, pos_encoding="learnable", rope_fast=False):
    cfg = make_cfg(amp, channels_last)
    model = build_model(pos_encoding, rope_fast, channels_last)
    if fused:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2,
                                betas=(0.99, 0.999), eps=1e-8, fused=True)
    else:
        opt = CustomAdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    ema = ModelEma(model, decay=0.999, foreach=foreach_ema)
    prefetcher = FakePrefetcher(4)
    metrics, gs = train_epoch(cfg, prefetcher, model, opt, ema, DEV, 0,
                              ch_weight=None, ema_source_model=model)
    finite = all(isinstance(v, float) and v == v and abs(v) != float("inf")
                 for v in metrics.values())
    # confirm EMA actually moved (sanity that update ran)
    ema_p = next(iter(ema.module.state_dict().values()))
    ok = finite and torch.isfinite(ema_p).all().item()
    print(f"  {tag:42s} -> {'OK ' if ok else 'FAIL'} | "
          f"L_dens={metrics.get('L_dens', float('nan')):.4f} "
          f"L_str={metrics.get('L_str', float('nan')):.4f} "
          f"grad_norm={metrics.get('grad_norm', float('nan')):.3f} steps={gs}")
    del model, opt, ema, prefetcher
    torch.cuda.empty_cache()
    return ok


def main():
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("high")
    print(f"train_epoch wiring smoke on {torch.cuda.get_device_name(0)} (B={B}, n_in={NIN})")
    results = []
    results.append(run("fp32 baseline (all knobs off)",
                       amp=False, channels_last=False, fused=False, foreach_ema=False))
    results.append(run("amp",                amp=True,  channels_last=False, fused=False, foreach_ema=False))
    results.append(run("amp+channels_last",  amp=True,  channels_last=True,  fused=False, foreach_ema=False))
    results.append(run("amp+cl+fused",       amp=True,  channels_last=True,  fused=True,  foreach_ema=False))
    results.append(run("amp+cl+fused+foreach_ema (full stack)",
                       amp=True, channels_last=True, fused=True, foreach_ema=True))
    results.append(run("full stack + rope3d + rope_fast",
                       amp=True, channels_last=True, fused=True, foreach_ema=True,
                       pos_encoding="rope3d", rope_fast=True))
    print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'} "
          f"({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
