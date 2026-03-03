"""
test_density_poc.py — End-to-end smoke test for the density-conditioning PoC
=============================================================================
Runs through the full pipeline without any training:

  1. Load real CrossDocked samples from data_test.pt
  2. Voxelize pocket + ligand (GPU)
  3. Generate pdb2vol density from pocket coords
  4. Forward pass through baseline VoxBind (density=None)
  5. Forward pass through density-conditioned VoxBind (density provided)
  6. Load the pretrained checkpoint into the density model (strict=False)
     and confirm the frozen baseline weights are identical
  7. Print a brief summary table

Usage
-----
    conda run -n voxbind python test/test_density_poc.py
    conda run -n voxbind python test/test_density_poc.py --pretrained exps/exp_sig0.9
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO / "voxbind")   # Hydra / relative-path convention

from voxbind.dataset.crossdocked import DatasetCrossdocked
from voxbind.dataset.pdb2vol import coords_to_density
from voxbind.models.voxbind import VoxBind
from voxbind.voxelizer import Voxelizer


# ── Config ────────────────────────────────────────────────────────────────────

PRETRAINED_DEFAULT = "exps/exp_sig0.9"
DATA_DIR           = "dataset/data"
BSZ                = 8       # small for smoke test
SIGMA              = 1.5     # pdb2vol sigma in Å
SMOOTH_SIGMA       = 0.9
GRID_DIM           = 64
RESOLUTION         = 0.25


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def check(label: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        raise AssertionError(label)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pdb2vol(batch: dict, device: torch.device) -> torch.Tensor:
    section("1. pdb2vol density generation")

    coords   = batch["pocket"]["coords"].to(device)    # (B, 2000, 3)
    atoms_ch = batch["pocket"]["atoms_channel"].to(device)  # (B, 2000)

    t0 = time.time()
    density = coords_to_density(
        coords, atoms_ch,
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        sigma=SIGMA,
    )
    elapsed = time.time() - t0

    check("output shape (B,1,64,64,64)", density.shape == (BSZ, 1, GRID_DIM, GRID_DIM, GRID_DIM))
    check("dtype float32", density.dtype == torch.float32)
    check("no NaN", not torch.isnan(density).any())
    check("z-scored (mean ≈ 0, std ≈ 1)", all(
        abs(density[b].mean().item()) < 0.05 and abs(density[b].std().item() - 1.0) < 0.05
        for b in range(BSZ)
    ))

    nonzero_frac = (density.abs() > 0.01).float().mean().item()
    print(f"  Non-trivial voxel fraction : {100 * nonzero_frac:.1f}%")
    print(f"  Peak value                 : {density.max().item():.2f}")
    print(f"  Time for B={BSZ}           : {1000 * elapsed:.1f} ms")

    return density


def test_model_forward(voxels_lig, voxels_poc, density, device):
    section("2. VoxBind forward passes")

    smooth = voxels_lig + torch.randn_like(voxels_lig) * SMOOTH_SIGMA

    model_kwargs = dict(n_channels=32, n_groups=16, smooth_sigma=SMOOTH_SIGMA)

    # Baseline (no density)
    model_base = VoxBind(with_density=False, **model_kwargs).to(device).eval()
    with torch.no_grad():
        out_base = model_base(smooth, voxels_poc)
    check("baseline output shape", out_base.shape == voxels_lig.shape)
    check("baseline no NaN", not torch.isnan(out_base).any())

    # Density-conditioned
    model_dens = VoxBind(with_density=True, **model_kwargs).to(device).eval()
    with torch.no_grad():
        out_dens = model_dens(smooth, voxels_poc, density=density)
    check("density output shape", out_dens.shape == voxels_lig.shape)
    check("density no NaN", not torch.isnan(out_dens).any())

    # density=None on density model should behave like baseline architecture
    with torch.no_grad():
        out_no_den = model_dens(smooth, voxels_poc, density=None)
    check("density=None fallback shape", out_no_den.shape == voxels_lig.shape)

    print(f"  Baseline   MSE vs lig : {torch.nn.functional.mse_loss(out_base, voxels_lig):.4f}")
    print(f"  + density  MSE vs lig : {torch.nn.functional.mse_loss(out_dens, voxels_lig):.4f}")
    print("  (random init — MSE values are not meaningful yet)")

    return model_dens


def test_checkpoint_load(pretrained_path: str, device: torch.device):
    section("3. Pretrained checkpoint → density model (strict=False)")

    ckpt_path = os.path.join(pretrained_path, "checkpoint.pth.tar")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("state_dict_ema", ckpt.get("state_dict", {}))
    sd   = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    # Infer n_channels from checkpoint weights
    enc_w = sd.get("ligand_encoder.conv2.weight",
                   sd.get("ligand_encoder.block2.weight", None))
    n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
    n_groups   = min(16, n_channels // 2)
    model_kwargs = dict(n_channels=n_channels, n_groups=n_groups, smooth_sigma=SMOOTH_SIGMA)
    print(f"  Detected n_channels={n_channels}, n_groups={n_groups}")

    model = VoxBind(with_density=True, **model_kwargs).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    check("only density_encoder keys are missing",
          all("density_encoder" in k for k in missing))
    check("no unexpected keys", len(unexpected) == 0)
    print(f"  Missing (will train) : {missing}")

    # Confirm pocket_encoder weights transferred correctly
    model_base = VoxBind(with_density=False, **model_kwargs).to(device)
    model_base.load_state_dict(sd, strict=True)

    for (n1, p1), (n2, p2) in zip(
        model_base.pocket_encoder.named_parameters(),
        model.pocket_encoder.named_parameters(),
    ):
        if not torch.allclose(p1, p2):
            raise AssertionError(f"Weight mismatch in pocket_encoder.{n1}")
    check("pocket_encoder weights match baseline", True)

    n_dens = sum(p.numel() for p in model.density_encoder.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  density_encoder params : {n_dens:,}  ({100*n_dens/n_total:.3f}% of total)")

    return model


def test_frozen_finetune(model: VoxBind, voxels_lig, voxels_poc, density, device):
    section("4. Simulate one finetune step (density_encoder only)")

    if model is None:
        print("  [SKIP] no model from checkpoint test")
        return

    # Freeze all except density_encoder
    for name, p in model.named_parameters():
        p.requires_grad_("density_encoder" in name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("trainable params > 0", n_trainable > 0)

    # Save a pocket_encoder param before the step
    poc_w_before = next(model.pocket_encoder.parameters()).detach().clone()
    den_w_before = next(model.density_encoder.parameters()).detach().clone()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    criterion = torch.nn.MSELoss(reduction="sum")

    model.train()
    model.density_encoder.train()
    smooth = voxels_lig + torch.randn_like(voxels_lig) * SMOOTH_SIGMA

    pred = model(smooth, voxels_poc, density=density)
    loss = criterion(pred, voxels_lig)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    poc_w_after = next(model.pocket_encoder.parameters()).detach()
    den_w_after = next(model.density_encoder.parameters()).detach()

    check("pocket_encoder NOT updated", torch.allclose(poc_w_before, poc_w_after))
    check("density_encoder IS  updated", not torch.allclose(den_w_before, den_w_after))
    print(f"  Loss on random init : {loss.item():.2f}")
    print(f"  density_encoder weight delta : {(den_w_after - den_w_before).abs().max():.2e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=PRETRAINED_DEFAULT)
    p.add_argument("--data_dir",   default=DATA_DIR)
    p.add_argument("--bsz",        type=int, default=BSZ)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    # ── Load a small batch from the test set ──────────────────────────────────
    section("0. Loading data")
    dset = DatasetCrossdocked(data_dir=args.data_dir, split="test", aug=False, small=True)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.bsz, shuffle=False, num_workers=0, drop_last=True
    )
    batch = next(iter(loader))
    print(f"  Loaded {args.bsz} test samples")

    voxelizer = Voxelizer(grid_dim=GRID_DIM, resolution=RESOLUTION, cubes_around=8, device=device)
    with torch.no_grad():
        voxels_lig = voxelizer.forward(batch["ligand"], num_channels=7)
        voxels_poc = voxelizer.forward(batch["pocket"], num_channels=4)
    print(f"  voxels_lig : {voxels_lig.shape}  voxels_poc : {voxels_poc.shape}")

    # ── Run tests ─────────────────────────────────────────────────────────────
    density = test_pdb2vol(batch, device)
    test_model_forward(voxels_lig, voxels_poc, density, device)
    model   = test_checkpoint_load(args.pretrained, device)
    test_frozen_finetune(model, voxels_lig, voxels_poc, density, device)

    print("\n" + "=" * 55)
    print("  All tests PASSED")
    print("=" * 55)
    print(f"\nNext: finetune with")
    print(f"  cd voxbind")
    print(f"  python train_density.py --pretrained_path {args.pretrained} \\")
    print(f"      --sigma {SIGMA} --epochs 30 --bsz 32")


if __name__ == "__main__":
    main()
