"""density_mae.py — Voxel-MAE pre-training for the X-ray density encoder.

Pretext task
------------
Inputs : synthetic density volumes (Gaussian-blurred sum of ligand+pocket atom
         voxels, per-sample z-scored, with optional additive noise), partially
         masked with 3D block masks.
Targets: (a) the un-masked clean density (reconstruction on masked voxels),
         (b) the raw 11-channel atom voxels — 7 ligand + 4 pocket.

The encoder is shape-compatible with VoxBind's existing `density_encoder`
(Conv3d(1→C/2) + SiLU + N × ResidualBlock(C/2→C/2)). After pre-training, the
encoder state_dict can be loaded into a deepened VoxBind density branch
(Phase 3) — fusion via the existing zero-init `density_proj` stays identical.

Forward
-------
    d_hat, a_hat = model(density_input)   # both at (B, *, G, G, G)

Targets / loss helpers live in `train_density_mae.py`.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxbind.models.unet3d import ResidualBlock


# ── Synthetic-density helpers ─────────────────────────────────────────────────

def gaussian_blur3d(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """Separable 3D Gaussian blur. sigma_vox is in voxel units (1 vox = 0.25 Å)."""
    half = max(1, int(3.0 * sigma_vox + 0.5))
    coords = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (coords / sigma_vox) ** 2)
    g = g / g.sum()
    kx = g.view(1, 1, -1, 1, 1)
    ky = g.view(1, 1, 1, -1, 1)
    kz = g.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, kx, padding=(half, 0, 0))
    x = F.conv3d(x, ky, padding=(0, half, 0))
    x = F.conv3d(x, kz, padding=(0, 0, half))
    return x


def make_block_mask(
    bsz: int, grid_dim: int, block_size: int, ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample a (B, 1, G, G, G) bool mask where masked voxels = True.

    Mask is generated at (G/block)³ resolution then upsampled by repetition,
    so masked regions are cubes of `block_size`³ voxels.
    """
    assert grid_dim % block_size == 0, f"grid_dim {grid_dim} % block_size {block_size} != 0"
    gb = grid_dim // block_size
    blocks = torch.rand(bsz, 1, gb, gb, gb, device=device) < ratio   # (B,1,gb,gb,gb)
    mask = blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)
    return mask  # (B, 1, G, G, G) bool


def per_sample_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score each (C, G, G, G) volume independently along spatial dims."""
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    std = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)
    return (x - mu) / std


def synth_density(
    voxels_lig: torch.Tensor,
    voxels_poc: torch.Tensor,
    sigma_blur_vox_range: Tuple[float, float],
    sigma_noise: float,
) -> torch.Tensor:
    """Build a synthetic 2Fo-Fc-like volume from voxelized ligand+pocket atoms.

    Steps: sum-over-channels → random-σ Gaussian blur → per-sample z-score →
    additive Gaussian noise. Returns (B, 1, G, G, G).
    """
    atoms = voxels_lig.sum(dim=1, keepdim=True) + voxels_poc.sum(dim=1, keepdim=True)
    sigma_lo, sigma_hi = sigma_blur_vox_range
    sigma = float(torch.empty(1).uniform_(sigma_lo, sigma_hi).item())
    d = gaussian_blur3d(atoms, sigma)
    d = per_sample_zscore(d)
    if sigma_noise > 0:
        d = d + torch.randn_like(d) * sigma_noise
    return d


# ── ELECTRA-style corruption ops ──────────────────────────────────────────────
# Each op takes the clean z-scored density and the (B,1,G,G,G) bool corruption
# mask, and returns a (B,1,G,G,G) density with masked voxels replaced by the
# op's "fake" content (and clean voxels untouched). Used by
# `train_density_vit_mae.py` when `mae.pretext_style == 'electra'`.

def corrupt_swap(d_clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked blocks with same-region content from another sample.

    Cyclically rolls the batch by 1 so each sample's corrupted regions come
    from a different sample's clean density. Cheap and distribution-matched.
    Edge case: B == 1 → no-op (returns d_clean unchanged).
    """
    if d_clean.shape[0] <= 1:
        return d_clean
    d_other = torch.roll(d_clean, shifts=1, dims=0)
    return torch.where(mask, d_other, d_clean)


def corrupt_reblur(
    atoms_sum: torch.Tensor,
    d_clean: torch.Tensor,
    mask: torch.Tensor,
    sigma_alt_vox: float,
) -> torch.Tensor:
    """Replace masked blocks with the same atoms re-blurred at a different σ.

    Captures resolution-mismatch failure modes (e.g. distinguishing real EM
    density at the modeled resolution vs. an over-smoothed or under-smoothed
    alternative). Computes a fresh per-sample z-scored volume at `sigma_alt_vox`,
    then composites into the mask.
    """
    d_alt = gaussian_blur3d(atoms_sum, sigma_alt_vox)
    d_alt = per_sample_zscore(d_alt)
    return torch.where(mask, d_alt, d_clean)


def corrupt_noise(d_clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked blocks with N(0, 1) noise (matched to z-scored scale)."""
    noise = torch.randn_like(d_clean)
    return torch.where(mask, noise, d_clean)


def corrupt_generator(
    d_clean: torch.Tensor, mask: torch.Tensor, generator_net,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a learned generator on the partially-masked density.

    The generator sees `d_clean * (~mask)` (zero-masked input) and produces a
    full (B,1,G,G,G) density `d_gen`. Returns:
      d_corrupted : (B,1,G,G,G) — d_clean with masked positions replaced by
                    d_gen.detach() (so disc-side gradient does NOT flow into gen).
      d_gen       : (B,1,G,G,G) — the raw generator output WITH grad, for the
                    generator's MLM-style MSE loss on masked positions.
    """
    d_in_masked = d_clean * (~mask).to(d_clean.dtype)
    d_gen = generator_net(d_in_masked)
    d_corrupted = torch.where(mask, d_gen.detach(), d_clean)
    return d_corrupted, d_gen


def voxel_mask_to_patch_target(
    mask: torch.Tensor, patch_size: int,
) -> torch.Tensor:
    """Pool a (B,1,G,G,G) bool voxel-level mask to (B,1,G_p,G_p,G_p) float.

    Output is 1.0 if any voxel in the patch is corrupted (max-pool), else 0.0.
    When block_size == patch_size and the mask is block-aligned, every patch is
    either fully corrupted or fully clean — but using max_pool keeps things
    safe if the two ever diverge.
    """
    m = mask.to(dtype=torch.float32)
    return F.max_pool3d(m, kernel_size=patch_size, stride=patch_size)


# ── DensityMAE ────────────────────────────────────────────────────────────────

class DensityMAE(nn.Module):
    """Voxel-MAE: encoder → (density head, structure head).

    The encoder mirrors VoxBind's `density_encoder` shape so the pre-trained
    weights can drop into a deeper VoxBind density branch later. With
    `n_blocks=1` it is byte-for-byte identical to the current VoxBind branch
    (modulo init values); the default `n_blocks=4` adds capacity for SSL to
    actually use — Phase 3 will deepen VoxBind's branch to match.
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_blocks: int = 4,
        n_groups: int = 16,
        dropout: float = 0.1,
        n_struct_channels: int = 11,  # 7 ligand + 4 pocket
    ):
        super().__init__()
        c_half = n_channels // 2
        g = min(n_groups, c_half) if n_groups > 0 else 0

        # Encoder — same structure as voxbind.density_encoder, just deeper.
        # Module names: encoder.0 = Conv3d, encoder.1 = SiLU,
        # encoder.2..(2+n_blocks-1) = ResidualBlocks.
        layers = [nn.Conv3d(1, c_half, kernel_size=3, padding=1), nn.SiLU()]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(c_half, c_half, n_groups=g, dropout=dropout))
        self.encoder = nn.Sequential(*layers)

        self.head_density = nn.Sequential(
            nn.Conv3d(c_half, c_half, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(c_half, 1, kernel_size=3, padding=1),
        )
        self.head_structure = nn.Sequential(
            nn.Conv3d(c_half, c_half, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(c_half, n_struct_channels, kernel_size=3, padding=1),
        )

        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.n_struct_channels = n_struct_channels

    def encode(self, density: torch.Tensor) -> torch.Tensor:
        return self.encoder(density)

    def forward(self, density: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(density)
        d_hat = self.head_density(z)
        a_hat = self.head_structure(z)
        return d_hat, a_hat
