"""mae_ops.py — shared voxel/density ops for the density-MAE pretexts.

Pure functions used across pre-training (train_density.py) and the dataset
density pipeline (crossdocked_*, 00b/00e/00f, the probe): synthetic-density
synthesis, 3D block masking, per-sample z-scoring, gradient magnitude, and the
ELECTRA-style corruption ops. No model state — just tensor transforms.

(Extracted from the former density_mae.py, whose DensityMAE CNN model — superseded
by the ViT encoder in density_vit.py — was removed along with its deleted trainer.)
"""

from typing import Tuple

import torch
import torch.nn.functional as F


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


def gradient_magnitude3d(x: torch.Tensor, spacing: float = 1.0) -> torch.Tensor:
    """Per-voxel gradient magnitude ‖∇x‖ via central finite differences.

    An edge / iso-surface detector for a density field: large where the density
    changes fastest (atom-blob shells, 2Fo-Fc iso-surfaces), complementing the
    raw density magnitude. Replicate padding preserves the G³ shape and avoids
    spurious large gradients at the boundary faces.

    `spacing` (voxel size) only rescales the output globally, so when the result
    is subsequently per-sample z-scored it is irrelevant (default 1.0 = voxel
    units, matching the rest of the density pipeline).

    Args
    ----
    x : (B, 1, G, G, G) tensor (CPU or CUDA).

    Returns
    -------
    (B, 1, G, G, G) non-negative tensor, same dtype/device as `x`.
    """
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    gx = (xp[:, :, 2:, 1:-1, 1:-1] - xp[:, :, :-2, 1:-1, 1:-1]) / (2.0 * spacing)
    gy = (xp[:, :, 1:-1, 2:, 1:-1] - xp[:, :, 1:-1, :-2, 1:-1]) / (2.0 * spacing)
    gz = (xp[:, :, 1:-1, 1:-1, 2:] - xp[:, :, 1:-1, 1:-1, :-2]) / (2.0 * spacing)
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def protein_attenuation_field(
    poc_occ: torch.Tensor,
    sigma_vox: float = 7.0,
    quantile: float = 0.90,
    strength: float = 1.0,
) -> torch.Tensor:
    """alpha(x) in [0,1] over the density grid, built from PROTEIN ATOMS ONLY.

    High alpha near protein atoms (preserve density), low in open space (attenuate) —
    i.e. we remove *the space a ligand could occupy*, never "the ligand". Ligand
    coordinates never enter construction, which is what makes this (a) free of any
    leakage argument (the mask shape cannot encode ligand shape, unlike
    `VoxBind.ligand_occupancy_mask`) and (b) computable at GENERATION time, where no
    ligand exists — the diagnostic condition and the deployment condition are identical.

    A Gaussian-blurred occupancy is a smooth, monotone-in-distance proxy for
    distance-to-nearest-protein-atom, and `sigma_vox` sets the transition width
    (1 vox = 0.25 A, so sigma 6-8 gives the ~1.5-2 A falloff of the map's own blur).
    A boundary sharper than the map's intrinsic resolution is physically impossible and
    the network would read it as a feature.

    Normalisation uses a high quantile of the values *at occupied voxels* rather than the
    max, so dense protein cores don't compress the rest of the field toward 0.

    Args:
        poc_occ:   (B,1,G,G,G) protein/pocket occupancy (sum over pocket atom channels).
        sigma_vox: transition width in voxels.
        quantile:  proximity value mapped to alpha = 1.
        strength:  global dose. alpha_eff = 1 - strength*(1-alpha); 0 = no-op, 1 = full.
                   Sweeping this is the performance-vs-alpha dose-response axis.

    Returns:
        (B,1,G,G,G) float in [0,1].
    """
    prox = gaussian_blur3d(poc_occ, sigma_vox)
    occupied = poc_occ > 0
    if occupied.any():
        ref = torch.quantile(prox[occupied].float(), quantile)
    else:                                        # degenerate crop: no pocket atoms
        ref = prox.amax()
    alpha = (prox / ref.clamp(min=1e-8)).clamp(0.0, 1.0)
    if strength != 1.0:
        alpha = 1.0 - float(strength) * (1.0 - alpha)
    return alpha


def attenuate_density(
    density: torch.Tensor,
    alpha: torch.Tensor,
    noise_sigma_vox: float = 7.0,
) -> torch.Tensor:
    """rho' = alpha*rho + (1-alpha)*noise — the "nothing here but water" fill.

    The noise is BLURRED to `noise_sigma_vox` and renormalised to unit variance, so its
    gradient statistics resemble real density. This is not cosmetic: VoxBind derives a
    gradmag channel from this map, and white N(0,1) noise has enormous high-frequency
    gradient — it would make that channel wildly out-of-distribution and turn "density
    removed" into "gradmag broken". Unit variance also keeps the z-scored scale the
    encoder was pretrained on.

    Prefer this over plain `alpha*rho`, which drives the region to the normalisation mean
    and reads as "unknown"; a perfectly flat region is itself a tell.
    """
    noise = torch.randn_like(density)
    if noise_sigma_vox and noise_sigma_vox > 0:
        noise = gaussian_blur3d(noise, noise_sigma_vox)
        noise = noise / noise.std().clamp(min=1e-6)
    return alpha * density + (1.0 - alpha) * noise


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


def make_atom_biased_block_mask(
    atoms_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    tau: float = 1.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B, 1, G, G, G) bool mask biased toward atom-occupied blocks.

    `atoms_sum` is (B, 1, G, G, G) — sum of all atom-voxel channels. For each
    sample we score each block by its atom-mass via avg_pool3d, add Gaussian
    noise scaled by `tau · std(scores)`, and take the top floor(ratio · gb³)
    blocks. Mask cardinality is exact per sample.

      tau → 0   : deterministic top-K (every most-atom-occupied block masked first)
      tau → ∞   : equivalent to uniform random masking (noise dominates)
      tau = 1   : moderate bias — atom-occupied blocks dominate but training
                  still sees some empty-space variety.
    """
    B, _, G, _, _ = atoms_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = int(round(ratio * n_blocks))
    # Per-block atom mass: avg_pool3d gives mean; × block³ = sum over the block.
    score = F.avg_pool3d(atoms_sum, block_size) * (block_size ** 3)  # (B, 1, gb, gb, gb)
    score = score.flatten(2)                                        # (B, 1, gb³)
    std = score.std(dim=2, keepdim=True).clamp(min=1e-6)
    if generator is not None:
        noise = torch.randn(score.shape, device=score.device, generator=generator)
    else:
        noise = torch.randn_like(score)
    priority = score + noise * std * tau
    _, top_idx = priority.topk(n_mask, dim=2)
    blocks_flat = torch.zeros_like(score, dtype=torch.bool)
    blocks_flat.scatter_(2, top_idx, True)
    blocks = blocks_flat.view(B, 1, gb, gb, gb)
    mask = blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)
    return mask  # (B, 1, G, G, G) bool


def make_cluster_mask(
    atoms_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    n_seeds: int = 4,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B,1,G,G,G) bool mask as a few LARGE CONTIGUOUS atom-anchored clusters.

    A dramatic alternative to scattered block masking (make_block_mask / atom_biased,
    which both drop many small independent blocks). For each sample we pick `n_seeds`
    seed blocks (∝ atom mass, like atom_biased), then mask the floor(ratio·gb³) blocks
    that are CLOSEST (min Euclidean block-grid distance) to any seed. That yields
    `n_seeds` large contiguous holes centred on chemistry, so the encoder must inpaint
    whole substructures from long-range context instead of filling many tiny gaps —
    the MAE "large contiguous masks force semantic inference" regime, but atom-anchored.
    Mask cardinality is exact per sample. Reduces to a single big ball when n_seeds=1.
    """
    B, _, G, _, _ = atoms_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = max(1, int(round(ratio * n_blocks)))
    k = max(1, min(int(n_seeds), n_blocks))
    dev = atoms_sum.device
    # per-block atom mass (≥0) → seed-sampling weights, biased toward atoms
    w = (F.avg_pool3d(atoms_sum, block_size) * (block_size ** 3)).reshape(B, n_blocks)
    w = w.clamp_min(0) + 1e-6
    # integer block-grid coords (n_blocks, 3)
    rng = torch.arange(gb, device=dev)
    coords = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1)\
                  .reshape(-1, 3).float()
    mask_flat = torch.zeros(B, n_blocks, dtype=torch.bool, device=dev)
    for b in range(B):
        seeds = torch.multinomial(w[b], k, replacement=False, generator=generator)
        dist = torch.cdist(coords, coords[seeds]).amin(dim=1)        # (n_blocks,) → nearest seed
        sel = dist.topk(n_mask, largest=False).indices
        mask_flat[b, sel] = True
    mask = mask_flat.view(B, 1, gb, gb, gb)
    return mask.repeat_interleave(block_size, dim=2)\
               .repeat_interleave(block_size, dim=3)\
               .repeat_interleave(block_size, dim=4)  # (B, 1, G, G, G) bool


def make_interface_mask(
    lig_sum: torch.Tensor,
    poc_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    tau: float = 0.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B,1,G,G,G) bool mask over the ligand–pocket INTERFACE (contact region).

    The interface is where binding affinity originates: voxels where ligand and pocket
    atoms are adjacent. We score each block by how much it has BOTH ligand and pocket
    mass nearby — block-level product of (dilated) ligand and pocket occupancy — and mask
    the floor(ratio·gb³) highest-scoring blocks. The encoder then reconstructs the contact
    interface from its surroundings → learns binding-interaction structure. `tau`>0 adds
    Gaussian tie-break noise (0 = deterministic top-K interface). Exact mask cardinality.
    """
    B, _, G, _, _ = lig_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = max(1, int(round(ratio * n_blocks)))
    # block-resolution occupancy, dilated by 1 block so adjacency (not overlap) counts
    lb = F.max_pool3d(F.avg_pool3d(lig_sum, block_size), kernel_size=3, stride=1, padding=1)
    pb = F.max_pool3d(F.avg_pool3d(poc_sum, block_size), kernel_size=3, stride=1, padding=1)
    score = (lb * pb).flatten(2)                                    # (B,1,gb³) high at the interface
    if tau > 0:
        std = score.std(dim=2, keepdim=True).clamp(min=1e-6)
        noise = (torch.randn(score.shape, device=score.device, generator=generator)
                 if generator is not None else torch.randn_like(score))
        score = score + noise * std * tau
    _, top_idx = score.topk(n_mask, dim=2)
    blocks_flat = torch.zeros_like(score, dtype=torch.bool)
    blocks_flat.scatter_(2, top_idx, True)
    blocks = blocks_flat.view(B, 1, gb, gb, gb)
    return blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)  # (B, 1, G, G, G) bool


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
# `train_density.py` when `mae.pretext_style == 'electra'`.

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
