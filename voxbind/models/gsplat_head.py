"""Gaussian-splatting output head for the VoxBind denoiser.

WHY THIS CAN WORK AT ALL: the grid the denoiser regresses is itself a sum of isotropic
Gaussians. `voxelizer._torch_voxelize` writes, per atom, `exp(-d^2 / 2 sigma^2)` with
`sigma = radius / resolution` (0.5 A / 0.25 A = 2 voxels) scatter-added over a 17^3
stencil. The stock head is a 3x3x3 conv that paints those blobs channel-by-channel with
no notion of an atom; this head instead predicts a sparse set of Gaussians and rasterizes
them additively, so the decoder can only express sums-of-blobs -- the shape the data has.

HOW THE RASTERIZER IS CHEAP: evaluating every Gaussian at every voxel is ~10^8 exp() per
sample (32^3 anchors x 7 channels x a 9^3 neighbourhood). Splatting a Gaussian is instead
"scatter an impulse, then blur": with a FIXED sigma, a sum of Gaussians is exactly the
convolution of a weighted impulse grid with one Gaussian kernel, and a 3D isotropic
Gaussian convolution is separable into three 1D passes. So each Gaussian costs 8 trilinear
scatter writes (its sub-voxel centre) plus a shared O(3 * (2r+1)) blur over the grid.
Variable width is recovered by mixing a few FIXED sigma bins per Gaussian (softmax
weights), which keeps the rasterizer convolutional while letting the network choose widths.
Centres, amplitudes and the sigma mixture are all differentiable; the bin sigmas are not.

Parameterization, per anchor (stride `stride` grid over the 64^3 output), per ligand
channel, K Gaussians:
    offset (3)      tanh-bounded to +/- offset_bound voxels around the anchor centre
    amplitude (1)   softplus, >= 0, initialised small
    sigma mix (n)   softmax over the fixed `sigmas` bins
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-x.pow(2) / (2.0 * sigma * sigma))
    return k


class GaussianSplatHead(nn.Module):
    """Predict anchor-grid Gaussians and rasterize them into the ligand voxel grid.

    Args:
        in_channels: width of the U-Net feature map this head consumes.
        n_out_channels: ligand atom-type channels to render (7).
        grid_dim: output grid edge (64).
        stride: anchor spacing in voxels. 4 -> a 16^3 anchor grid (4,096 anchors),
            which against ~25-atom ligands is already 160x over-complete.
        gaussians_per_anchor: K.
        hidden: width of the 1x1x1 parameter head.
        sigmas: FIXED bin widths in voxels. The voxelizer's own sigma is 2.0, so the
            default brackets it; a mixture can also fake a wider blob than any bin.
        offset_bound: how far a Gaussian may move from its anchor, in voxels.
        cutoff_sigma: blur kernel radius, in units of the widest bin.
        init_amplitude: softplus target for the initial amplitude. Small but non-zero:
            the head starts near-silent and grows blobs, rather than starting with a
            saturated grid that the first optimizer steps have to undo.
    """

    def __init__(
        self,
        in_channels: int,
        n_out_channels: int,
        grid_dim: int,
        *,
        stride: int = 4,
        gaussians_per_anchor: int = 2,
        hidden: int = 256,
        sigmas: Sequence[float] = (1.2, 2.0, 3.0),
        offset_bound: float = 3.0,
        cutoff_sigma: float = 3.0,
        init_amplitude: float = 0.05,
    ) -> None:
        super().__init__()
        if grid_dim % stride != 0:
            raise ValueError(f"grid_dim {grid_dim} must be divisible by stride {stride}")
        if gaussians_per_anchor < 1:
            raise ValueError("gaussians_per_anchor must be >= 1")
        if not sigmas:
            raise ValueError("at least one sigma bin is required")

        self.n_out_channels = int(n_out_channels)
        self.grid_dim = int(grid_dim)
        self.stride = int(stride)
        self.k = int(gaussians_per_anchor)
        self.offset_bound = float(offset_bound)
        self.n_bins = len(sigmas)
        self.register_buffer("sigmas", torch.tensor([float(s) for s in sigmas]),
                             persistent=False)

        # 4 + n_bins numbers per (channel, gaussian): 3 offsets, 1 amplitude, the mixture.
        self.per_gaussian = 4 + self.n_bins
        n_out = self.n_out_channels * self.k * self.per_gaussian

        # One strided conv reduces 64^3 -> the anchor grid and gives each anchor a real
        # receptive field; the rest is 1x1x1 so width is cheap on the small grid.
        self.trunk = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.out = nn.Conv3d(hidden, n_out, kernel_size=1)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            bias = self.out.bias.view(self.n_out_channels, self.k, self.per_gaussian)
            # softplus(b) = init_amplitude  ->  b = log(exp(a) - 1)
            bias[..., 3].fill_(math.log(math.expm1(max(init_amplitude, 1e-6))))

        radius = max(1, int(math.ceil(cutoff_sigma * max(sigmas))))
        self.radius = radius
        kernels = torch.stack([_gaussian_kernel1d(float(s), radius) for s in sigmas])
        # Normalisation is deliberately NOT applied: the voxelizer's blobs peak at 1.0 per
        # atom, so the kernel must keep exp(0) = 1 at the centre for an amplitude of 1 to
        # mean "one atom here".
        self.register_buffer("kernels1d", kernels, persistent=False)

        anchor = (torch.arange(grid_dim // stride, dtype=torch.float32) + 0.5) * stride - 0.5
        self.register_buffer("anchor_coords", anchor, persistent=False)

    def extra_repr(self) -> str:
        return (f"stride={self.stride}, k={self.k}, bins={self.sigmas.tolist()}, "
                f"radius={self.radius}")

    # ------------------------------------------------------------------ parameters
    def decode_parameters(self, x: torch.Tensor) -> dict:
        """Feature map -> bounded Gaussian parameters in VOXEL coordinates."""
        raw = self.out(self.trunk(x))
        b, _, a0, a1, a2 = raw.shape
        raw = raw.view(b, self.n_out_channels, self.k, self.per_gaussian, a0, a1, a2)

        ax = self.anchor_coords[:a0].view(1, 1, 1, a0, 1, 1)
        ay = self.anchor_coords[:a1].view(1, 1, 1, 1, a1, 1)
        az = self.anchor_coords[:a2].view(1, 1, 1, 1, 1, a2)
        off = torch.tanh(raw[:, :, :, 0:3]) * self.offset_bound
        cx = ax + off[:, :, :, 0]
        cy = ay + off[:, :, :, 1]
        cz = az + off[:, :, :, 2]
        centers = torch.stack([cx, cy, cz], dim=3)                # (B,C,K,3,A,A,A)
        amplitude = F.softplus(raw[:, :, :, 3])                   # (B,C,K,A,A,A)
        mixture = torch.softmax(raw[:, :, :, 4:], dim=3)          # (B,C,K,n_bins,A,A,A)
        return {"centers": centers, "amplitude": amplitude, "mixture": mixture}

    # ------------------------------------------------------------------ rasterizer
    def _scatter(self, params: dict) -> torch.Tensor:
        """Trilinear-scatter each Gaussian's weighted impulse into its sigma bin's grid."""
        centers = params["centers"]
        amplitude = params["amplitude"]
        mixture = params["mixture"]
        b = centers.shape[0]
        g = self.grid_dim
        c, k, n_bins = self.n_out_channels, self.k, self.n_bins

        # (B, C, K, 3, A,A,A) -> (B, C*K*A^3, 3); the same flattening for the weights.
        centers = centers.permute(0, 1, 2, 4, 5, 6, 3).reshape(b, -1, 3)
        weights = (amplitude.unsqueeze(3) * mixture)              # (B,C,K,n_bins,A,A,A)
        weights = weights.permute(0, 1, 2, 4, 5, 6, 3).reshape(b, c, -1, n_bins)
        n_per_channel = weights.shape[2]

        base = centers.floor()
        frac = centers - base
        base = base.long()

        grid = centers.new_zeros(b, c * n_bins, g * g * g)
        for dx in (0, 1):
            wx = frac[..., 0] if dx else (1.0 - frac[..., 0])
            ix = base[..., 0] + dx
            for dy in (0, 1):
                wy = frac[..., 1] if dy else (1.0 - frac[..., 1])
                iy = base[..., 1] + dy
                for dz in (0, 1):
                    wz = frac[..., 2] if dz else (1.0 - frac[..., 2])
                    iz = base[..., 2] + dz
                    inside = ((ix >= 0) & (ix < g) & (iy >= 0) & (iy < g)
                              & (iz >= 0) & (iz < g))
                    corner = (wx * wy * wz * inside).view(b, c, n_per_channel, 1)
                    lin = (ix.clamp(0, g - 1) * g * g
                           + iy.clamp(0, g - 1) * g
                           + iz.clamp(0, g - 1)).view(b, c, n_per_channel)
                    for bin_i in range(n_bins):
                        grid[:, bin_i * c:(bin_i + 1) * c].scatter_add_(
                            2, lin, corner[..., 0] * weights[..., bin_i]
                        )
        return grid.view(b, n_bins * c, g, g, g)

    def _blur(self, impulses: torch.Tensor) -> torch.Tensor:
        """Separable per-bin Gaussian convolution; bins are summed into the output."""
        b = impulses.shape[0]
        c, n_bins, g = self.n_out_channels, self.n_bins, self.grid_dim
        out = impulses.new_zeros(b, c, g, g, g)
        r = self.radius
        for bin_i in range(n_bins):
            x = impulses[:, bin_i * c:(bin_i + 1) * c]
            k1 = self.kernels1d[bin_i].to(x.dtype)
            for axis in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[2 + axis] = 2 * r + 1
                pad = [0, 0, 0, 0, 0, 0]
                pad[2 * (2 - axis)] = r
                pad[2 * (2 - axis) + 1] = r
                x = F.conv3d(F.pad(x, pad), k1.view(shape).expand(c, 1, *shape[2:]),
                             groups=c)
            out = out + x
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._blur(self._scatter(self.decode_parameters(x)))
