"""
pdb2vol — Gaussian density generation from atom coordinates
============================================================
Pure-PyTorch implementation of the Situs pdb2vol algorithm (Wriggers 2010).

Given a batch of (possibly padded) pocket atom coordinates, produces a
single-channel Gaussian density map on a voxel grid that matches the
VoxBind 64³ / 0.25 Å grid.

Algorithm (equivalent to Situs pdb2vol):
  1. Map each heavy atom to its nearest voxel (scatter / splat).
  2. Convolve the resulting delta-function map with a 3D isotropic Gaussian
     kernel whose sigma corresponds to the target EM resolution.

Default sigma = 1.5 Å  (FWHM ≈ 3.5 Å, approximating 3–4 Å cryo-EM data).

Usage
-----
    from voxbind.dataset.pdb2vol import coords_to_density

    # coords  : (B, N, 3) float32, in Å, centered at ligand CoM
    # atom_ch : (B, N)    float32, padding atoms have channel 999
    density = coords_to_density(coords, atom_ch)  # → (B, 1, 64, 64, 64)
"""

import math

import torch
import torch.nn.functional as F


# ── Gaussian kernel ───────────────────────────────────────────────────────────

def gaussian_kernel_3d(sigma_vox: float, trunc: float = 3.0) -> torch.Tensor:
    """Build a normalised 3-D isotropic Gaussian kernel.

    Parameters
    ----------
    sigma_vox : float
        Standard deviation in voxels.
    trunc : float
        Half-width of the kernel in multiples of sigma (default 3 → 6σ total).

    Returns
    -------
    torch.Tensor, shape (2r+1, 2r+1, 2r+1), float32, sums to 1.
    """
    r = max(1, int(math.ceil(trunc * sigma_vox)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    g1d = torch.exp(-0.5 * (xs / sigma_vox) ** 2)
    g1d = g1d / g1d.sum()
    # Outer product along all three axes
    kernel = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    return kernel  # (2r+1, 2r+1, 2r+1)


# ── Main API ──────────────────────────────────────────────────────────────────

def coords_to_density(
    coords: torch.Tensor,
    atoms_channel: torch.Tensor,
    grid_dim: int = 64,
    resolution: float = 0.25,
    sigma: float = 1.5,
    max_channel: int = 4,
) -> torch.Tensor:
    """Convert padded pocket atom coordinates to a Gaussian density map.

    Parameters
    ----------
    coords : Tensor, (B, N, 3), float32
        Atom positions in Å, centred at the ligand CoM.
        Padding atoms should have coords ≥ 999 (VoxBind convention).
    atoms_channel : Tensor, (B, N), float32
        Atom type channel indices.  Padding uses value 999.
    grid_dim : int
        Number of voxels per axis (default 64).
    resolution : float
        Voxel size in Å (default 0.25).
    sigma : float
        Gaussian sigma in Å (default 1.5).
    max_channel : int
        Atoms with channel index < max_channel are treated as valid pocket atoms;
        use 4 for pocket (C/O/N/S), 7 for ligand (default 4).

    Returns
    -------
    Tensor, (B, 1, grid_dim, grid_dim, grid_dim), float32
        Gaussian density map, z-scored per sample.
    """
    B, N, _ = coords.shape
    half = grid_dim // 2
    device = coords.device

    # ── valid atom mask ───────────────────────────────────────────────────────
    valid_ch = atoms_channel < max_channel                      # (B, N) not padding
    vox_f = coords / resolution + half                         # fractional voxel index
    vox_i = vox_f.round().long()                               # (B, N, 3)

    in_bounds = (
        (vox_i[..., 0] >= 0) & (vox_i[..., 0] < grid_dim) &
        (vox_i[..., 1] >= 0) & (vox_i[..., 1] < grid_dim) &
        (vox_i[..., 2] >= 0) & (vox_i[..., 2] < grid_dim)
    )                                                           # (B, N)
    mask = valid_ch & in_bounds                                # (B, N)

    # ── splat atoms onto grid (scatter accumulate) ────────────────────────────
    density = torch.zeros(B, 1, grid_dim, grid_dim, grid_dim,
                          device=device, dtype=torch.float32)

    for b in range(B):
        m = mask[b]                                            # (N,)
        if not m.any():
            continue
        vx = vox_i[b, m, 0]
        vy = vox_i[b, m, 1]
        vz = vox_i[b, m, 2]
        ones = torch.ones(int(m.sum()), device=device, dtype=torch.float32)
        density[b, 0].index_put_((vx, vy, vz), ones, accumulate=True)

    # ── Gaussian smoothing via 3-D convolution ────────────────────────────────
    sigma_vox = sigma / resolution
    kernel = gaussian_kernel_3d(sigma_vox).to(device)          # (k, k, k)
    pad = kernel.shape[0] // 2
    weight = kernel.unsqueeze(0).unsqueeze(0)                  # (1, 1, k, k, k)
    density = F.conv3d(density, weight, padding=pad)

    # ── per-sample z-score normalisation ─────────────────────────────────────
    for b in range(B):
        d = density[b, 0]
        mu  = d.mean()
        std = d.std()
        if std > 1e-8:
            density[b, 0] = (d - mu) / std
        else:
            density[b, 0] = torch.zeros_like(d)

    return density
