"""density_vit.py — Pure 3D Vision Transformer density encoder + MAE/ELECTRA wrapper.

`DensityViT` is shape-compatible with VoxBind's CNN `density_encoder`:
    in:  (B, 1,   G, G, G)
    out: (B, C/2, G, G, G)
so it drops into `VoxBind` in place of the existing Conv-stem + ResidualBlocks.

`DensityViTMAE` wraps `DensityViT` as `self.encoder` and adds pretext heads.
Two pretext styles are supported via `pretext_style`:

  - "mae"     : block-mask + reconstruct masked voxels. `head_density` is a
                Conv3d stack producing (B, 1, G, G, G); 11-ch atom-structure
                auxiliary head also active.

  - "electra" : ELECTRA-style replaced-token detection. Inputs are CORRUPTED
                (rule-based: swap-from-other-sample / re-blur / noise / or an
                optional learned generator) rather than zero-masked. `head_rtd`
                emits one logit per ViT patch (B, 1, G_p, G_p, G_p) for binary
                "was this patch corrupted?" classification. The 11-ch atom-
                structure auxiliary head is retained.

In either style, the `encoder.*` slice of the EMA checkpoint loads drop-in into
`VoxBind.density_encoder` via the existing loader in `models/__init__.py`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch <-> voxel reshape helpers (for the patch-MLP recon head) ────────────
# Inverses of each other; `_unpatchify` matches DensityViT._tokens_to_voxels, so
# a reconstruction laid out this way is voxel-aligned with the patch-embed input.

def _patchify(x: torch.Tensor, gp: int, p: int) -> torch.Tensor:
    """(B, c, G, G, G) → (B, gp³, c·p³): gather each p³ patch's voxels per token."""
    B, c = x.shape[0], x.shape[1]
    x = x.reshape(B, c, gp, p, gp, p, gp, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()   # (B, gp,gp,gp, c, p,p,p)
    return x.reshape(B, gp * gp * gp, c * p * p * p)


def _unpatchify(t: torch.Tensor, gp: int, p: int, c: int) -> torch.Tensor:
    """(B, gp³, c·p³) → (B, c, G, G, G): inverse of `_patchify` (matches _tokens_to_voxels)."""
    B = t.shape[0]
    x = t.reshape(B, gp, gp, gp, c, p, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()   # (B, c, gp,p, gp,p, gp,p)
    return x.reshape(B, c, gp * p, gp * p, gp * p)


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Unit-quaternion (..., 4) → rotation matrix (..., 3, 3). Normalizes q first."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# ── Transformer building blocks ───────────────────────────────────────────────

def _rope_axis_dim(head_dim: int) -> int:
    """Per-axis rotary width for axial 3D RoPE: the largest multiple of 6 that
    fits in head_dim, split evenly across x/y/z; trailing dims are left
    un-rotated. e.g. head_dim 64 → rotate 60 (20 per axis), 4 pass through."""
    return ((head_dim // 6) * 6) // 3            # even per-axis width


class RoPE3D(nn.Module):
    """Axial 3D rotary position embedding (Primus, arXiv:2503.01835).

    The rotary block is split evenly across the (x, y, z) patch axes; standard
    1D RoPE is applied within each axis from that axis's integer patch
    coordinate, so attention sees relative 3D offsets instead of absolute slots.
    If head_dim is not a multiple of 6 the largest multiple-of-6 prefix is
    rotated (e.g. 60 of 64) and the trailing dims pass through, keeping the three
    axes equal and even. cos/sin are precomputed for the fixed g_p³ patch grid
    (the encoder always sees all tokens) and applied to q/k inside attention.
    No learnable parameters.
    """

    def __init__(self, g_p: int, head_dim: int, base: float = 10000.0, fast: bool = False):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim={head_dim} must be even for RoPE"
        self.head_dim = head_dim
        self.fast = fast                              # fused q/k rotate path (numerically identical)
        self.d_axis = _rope_axis_dim(head_dim)        # even per-axis rotary width
        self.rope_dim = 3 * self.d_axis               # total rotated dims (<= head_dim)
        # patch coords (N, 3) in flatten(2) order: token n = i·g_p² + j·g_p + k.
        coords = torch.stack(torch.meshgrid(
            torch.arange(g_p), torch.arange(g_p), torch.arange(g_p), indexing="ij",
        ), dim=-1).reshape(-1, 3).float()                       # (N, 3)
        inv = base ** (-torch.arange(0, self.d_axis, 2).float() / self.d_axis)   # (d_axis/2,)
        ang = torch.cat([coords[:, a:a + 1] * inv[None, :] for a in range(3)], dim=1)  # (N, rope_dim/2)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def _cos_sin(self, N: int, dtype):
        """cos/sin tables for N query/key tokens. fused: N == g_p³ → one copy.
        channel_group: N == nG·g_p³ → tile the per-patch table nG times so every
        group's patch reuses the same spatial rotation (group identity is carried by
        group_embed, not RoPE). Bit-exact to the old `self.cos[:N]` when N == g_p³."""
        P = self.cos.shape[0]
        if N != P and P > 0 and N % P == 0:
            reps = N // P
            return self.cos.repeat(reps, 1).to(dtype), self.sin.repeat(reps, 1).to(dtype)
        return self.cos[:N].to(dtype), self.sin[:N].to(dtype)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the first rope_dim dims of q/k (B,H,N,head_dim) by the per-token 3D
        angles (interleaved pairs); the trailing head_dim-rope_dim dims pass through."""
        B, H, N, D = x.shape
        rd = self.rope_dim
        x_rot, x_pass = x[..., :rd], x[..., rd:]
        xr = x_rot.reshape(B, H, N, rd // 2, 2)
        x0, x1 = xr[..., 0], xr[..., 1]                        # (B, H, N, rd/2)
        cos, sin = self._cos_sin(N, x.dtype)                   # (N, rd/2) → broadcast over B,H
        r0 = x0 * cos - x1 * sin
        r1 = x0 * sin + x1 * cos
        x_rot = torch.stack((r0, r1), dim=-1).reshape(B, H, N, rd)
        return torch.cat((x_rot, x_pass), dim=-1)

    def _rotate_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Leading-dim-agnostic rotate (same math as `rotate`, fewer ops).

        Uses strided even/odd slicing instead of a reshape→unbind→stack→reshape,
        and broadcasts cos/sin over ANY number of leading dims — so q and k can be
        stacked and rotated in a single call (see `rotate_qk`). Bitwise-equivalent
        to `rotate`: x0/x1 are the same even/odd lanes and the interleave is identical.
        """
        rd = self.rope_dim
        x_rot, x_pass = x[..., :rd], x[..., rd:]
        x0 = x_rot[..., 0::2]                                  # (..., rd/2) even lanes
        x1 = x_rot[..., 1::2]                                  # (..., rd/2) odd lanes
        cos, sin = self._cos_sin(x.shape[-2], x.dtype)        # (N, rd/2) → broadcasts over leading dims
        r0 = x0 * cos - x1 * sin
        r1 = x0 * sin + x1 * cos
        out = torch.stack((r0, r1), dim=-1).flatten(-2)       # interleave back → (..., rd)
        if x_pass.shape[-1] == 0:
            return out
        return torch.cat((out, x_pass), dim=-1)

    def rotate_qk(self, q: torch.Tensor, k: torch.Tensor):
        """Rotate q and k together. fast=True fuses them into one kernel pass by
        stacking on a new leading dim; otherwise falls back to two `rotate` calls."""
        if self.fast:
            qk = self._rotate_fast(torch.stack((q, k), dim=0))   # (2, B, H, N, D)
            return qk[0], qk[1]
        return self.rotate(q), self.rotate(k)


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA via `F.scaled_dot_product_attention` (flash on Ampere+).

    Optional axial 3D RoPE is applied to q/k when a `RoPE3D` is passed to forward.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout_p = dropout
        # analysis-only: when capture_attn is set, take an explicit-softmax path and
        # stash the (B,H,N,N) weights in attn_weights (training stays on flash SDPA).
        self.capture_attn = False
        self.attn_weights = None

    def forward(self, x: torch.Tensor, rope: "RoPE3D" = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:
            q, k = rope.rotate_qk(q, k)
        if self.capture_attn:                            # explicit softmax (analysis only)
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            self.attn_weights = attn.detach()
            out = attn @ v                               # (B, H, N, head_dim)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
            )                                            # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN ViT block: x + DropPath(MHSA(LN(x))) → x + DropPath(MLP(LN(x)))."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, rope: "RoPE3D" = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope=rope))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (Huang 2016 / DeiT): per-SAMPLE, drop the whole residual branch
    with prob `p` during training (surviving branches rescaled by 1/(1−p) to keep the
    expected value). Eval / p=0 → identity (frozen probe + checkpoints bit-exact). Drops
    activations, not parameters, so every param still gets grad (DDP-safe)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)         # per-sample, broadcast over (T,D)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


# ── RadiusAtomEmbed (learnable radius-conditioned atom embedding, opt-in) ─────

class RadiusAtomEmbed(nn.Module):
    """Replace the per-element one-hot atom channels with a learned k-dim feature
    that is a NONLINEAR function of each element's vdW radius.

    Two small MLPs φ_lig, φ_poc map a scalar radius → ℝ^k, evaluated at the (fixed)
    per-element radii to give embedding matrices W_lig (n_lig×k), W_poc (n_poc×k).
    The k-channel ligand/pocket embedding grids are then `einsum`-contractions of the
    one-hot occupancy grids with these matrices — which, for the type-determined radii
    we use, is *exactly* the per-atom splat ``Σ_a φ(role_a, r_a)·g_a`` (a zeroed/masked
    voxel maps to a zero embedding, so MAE masking commutes with the embedding).

    Because φ sees only the *continuous* radius (nonlinear), this is NOT absorbable into
    the linear patch-embed when k < n_elements (a learned soft vocabulary), and it
    recovers the per-element separation that role-occupancy (roleblob) sums away. Note:
    elements that share a vdW radius (e.g. ligand S and P are both 1.8 Å) necessarily
    map to the same embedding — radius cannot tell them apart, by construction.

    DISTANCE CONDITIONING (opt-in, `scales` = list of extra-blur σ in voxels): the radial
    profile is learned as a MIXTURE over B fixed Gaussian scales. φ now outputs (B·k); for
    each scale b the occupancy is depthwise-blurred by σ_b (σ=0 → identity) and contracted
    with that scale's W slice, then summed: ``e = Σ_b W[:,b,:] · blur_{σ_b}(occupancy)``. So
    near an atom the deposited feature is a learned mixture of Gaussians at increasing widths
    — a learned (widening) radial profile instead of the single fixed Gaussian. scales=[0.0]
    (default) → exactly the single-scale version above (φ outputs k, no blur).

    Input grid layout (channel order): [ n_lig ligand | n_poc pocket | n_pt passthrough ]
    where passthrough = density (+ gradmag) is forwarded unchanged. Output width = 2k+n_pt.
    """

    def __init__(self, lig_radii, poc_radii, k: int, hidden: int = 16, n_passthrough: int = 2,
                 scales=None):
        super().__init__()
        self.n_lig = len(lig_radii)
        self.n_poc = len(poc_radii)
        self.k = int(k)
        self.n_pt = int(n_passthrough)
        self.scales = [float(s) for s in (scales if scales is not None else [0.0])]
        self.n_scales = len(self.scales)
        self.register_buffer("lig_r", torch.tensor(lig_radii, dtype=torch.float32).view(-1, 1))
        self.register_buffer("poc_r", torch.tensor(poc_radii, dtype=torch.float32).view(-1, 1))
        # fixed separable 1-D Gaussian kernels per non-zero scale (depthwise blur basis).
        for b, s in enumerate(self.scales):
            if s > 0:
                r = int(3 * s) + 1
                xs = torch.arange(-r, r + 1, dtype=torch.float32)
                ker = torch.exp(-(xs ** 2) / (2 * s * s)); ker = ker / ker.sum()
                self.register_buffer(f"gk_{b}", ker)            # (2r+1,)
        def _mlp():  # outputs B·k → reshaped (·, B, k) so each scale gets its own weights
            return nn.Sequential(nn.Linear(1, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.n_scales * self.k))
        self.phi_lig, self.phi_poc = _mlp(), _mlp()

    @property
    def out_channels(self) -> int:
        return 2 * self.k + self.n_pt

    @staticmethod
    def _sep_blur(x: torch.Tensor, ker: torch.Tensor) -> torch.Tensor:
        """Depthwise separable 3-D Gaussian blur (same 1-D kernel along each axis)."""
        C, K = x.shape[1], ker.numel()
        pad = K // 2
        kz = ker.view(1, 1, K, 1, 1).expand(C, 1, K, 1, 1)
        ky = ker.view(1, 1, 1, K, 1).expand(C, 1, 1, K, 1)
        kx = ker.view(1, 1, 1, 1, K).expand(C, 1, 1, 1, K)
        x = F.conv3d(x, kz, padding=(pad, 0, 0), groups=C)
        x = F.conv3d(x, ky, padding=(0, pad, 0), groups=C)
        x = F.conv3d(x, kx, padding=(0, 0, pad), groups=C)
        return x

    def _embed(self, v, W):
        """v: (B, n_elem, G³); W: (n_elem, B_scales, k) → (B, k, G³) summed over scales."""
        out = 0.0
        for b, s in enumerate(self.scales):
            vb = v if s == 0 else self._sep_blur(v, getattr(self, f"gk_{b}"))
            out = out + torch.einsum("bcxyz,ck->bkxyz", vb, W[:, b, :])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_lig = self.phi_lig(self.lig_r).view(self.n_lig, self.n_scales, self.k)
        W_poc = self.phi_poc(self.poc_r).view(self.n_poc, self.n_scales, self.k)
        v_lig = x[:, : self.n_lig]                  # (B, n_lig, G,G,G)
        v_poc = x[:, self.n_lig : self.n_lig + self.n_poc]
        passth = x[:, self.n_lig + self.n_poc :]    # density (+ gradmag)
        e_lig = self._embed(v_lig, W_lig)           # (B, k, G,G,G), learned radial profile
        e_poc = self._embed(v_poc, W_poc)
        return torch.cat([e_lig, e_poc, passth], dim=1)   # (B, 2k+n_pt, G,G,G)


# ── DensityViT (encoder w/ to-spatial head, drop-in for CNN density_encoder) ──

class DensityViT(nn.Module):
    """Pure 3D ViT producing a full-resolution `(B, C/2, G, G, G)` feature.

    Designed as a 1:1 replacement for VoxBind's `density_encoder`. Block-masking
    for MAE pretext happens upstream (zeroing voxels in input space), so the
    encoder always sees a full G³ volume — this keeps the drop-in interface
    with `VoxBind.density_proj` byte-clean and matches the existing CNN-encoder
    MAE pipeline.

    `n_in_channels` controls the number of input channels at the patch embed:
        1  → density-only (default; matches the original density_encoder slot)
        11 → atom-blob only (7 ligand + 4 pocket)
        12 → atom-blob + density (joint multi-channel input)
    """

    def __init__(
        self,
        grid_dim: int = 64,
        patch_size: int = 8,
        n_in_channels: int = 1,     # input channels: 1=density, 11=atomblob, 12=atomblob+density
        c_out: int = 16,            # = n_channels // 2 for VoxBind (default n_channels=32)
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        pos_embed_std: float = 0.02,
        pos_encoding: str = "learnable",   # "learnable" | "rope3d" (axial 3D RoPE)
        rope_fast: bool = False,           # fused q/k rotate path (rope3d only; numerically identical)
        patch_embed_mode: str = "fused",   # "fused" (1 Conv mixes all channels) | "channel_group" (ChannelViT)
        channel_groups: Optional[Tuple[int, ...]] = None,  # channel_group: per-group channel counts (sum=n_in)
        n_memory_tokens: int = 0,          # ChA-MAE: l learnable cross-channel memory tokens prepended pre-trunk
        channel_group_dropout: float = 0.0,  # ChannelViT HCS: per-forward prob of dropping each whole channel-group (train-only)
        radius_embed: Optional[dict] = None,  # opt-in learnable radius→k atom embedding front-end (see RadiusAtomEmbed)
        drop_path: float = 0.0,            # stochastic depth: max DropPath rate, linearly ramped 0→drop_path over depth
    ):
        super().__init__()
        assert grid_dim % patch_size == 0, (
            f"grid_dim={grid_dim} must be divisible by patch_size={patch_size}"
        )
        # Learnable radius-conditioned atom embedding (opt-in): transforms the raw
        # [n_lig|n_poc|passthrough] input into 2k+n_pt channels BEFORE patch_embed. When
        # set, n_in_channels / channel_groups passed in must already describe the POST-embed
        # width (the caller, DensityViTMAE, computes them). frozen probe inherits it (it lives
        # in the encoder), so feeding the raw atomblob grid just works.
        self.atom_embed = RadiusAtomEmbed(**radius_embed) if radius_embed else None
        self.grid_dim = grid_dim
        self.patch_size = patch_size
        self.n_in_channels = n_in_channels
        self.c_out = c_out
        self.dim = dim
        self.depth = depth
        # ChannelViT Hierarchical Channel Sampling (Bao et al. 2023): per-forward,
        # drop each whole channel-GROUP with this probability (train-only, ≥1 kept).
        # No-op at eval and for fused mode. Forces attention to fuse variable channel
        # subsets and turns the MAE into a cross-modal completion task.
        self.channel_group_dropout = float(channel_group_dropout)

        self.g_p = grid_dim // patch_size
        self.n_tokens = self.g_p ** 3
        self.patch_volume = patch_size ** 3

        assert patch_embed_mode in ("fused", "channel_group"), (
            f"unknown patch_embed_mode={patch_embed_mode!r} (expected 'fused' or 'channel_group')"
        )
        self.patch_embed_mode = patch_embed_mode
        # With the radius-embed front-end, patch_embed/grouping operate on the POST-embed
        # width (2k+n_pt), regrouped (k,k,n_pt); n_in_channels above stays the RAW width the
        # encoder receives (atom_embed bridges raw→pe in _tokenize). Without it, pe == raw.
        pe_in, pe_groups = n_in_channels, channel_groups
        if self.atom_embed is not None:
            pe_in = self.atom_embed.out_channels
            if patch_embed_mode == "channel_group":
                pe_groups = (self.atom_embed.k, self.atom_embed.k, self.atom_embed.n_pt)
        if patch_embed_mode == "fused":
            # one Conv3d fuses ALL input channels into each patch token (original ViT).
            self.patch_embed = nn.Conv3d(
                pe_in, dim, kernel_size=patch_size, stride=patch_size, padding=0
            )
            self.group_proj = None
            self.group_embed = None
            self.channel_groups = None
            self.n_groups = 1
        else:
            # ChannelViT-style: tokenize per channel-GROUP (one token-set per group) so
            # self-attention spans (group × patch) tokens and can attend across groups.
            # Each group keeps its own projection (groups differ in channel count + meaning);
            # a learnable group embedding tags every patch token of that group.
            assert pe_groups is not None and sum(pe_groups) == pe_in, (
                f"channel_groups={pe_groups} must sum to (post-embed) n_in_channels={pe_in}"
            )
            # pos_encoding='rope3d' IS supported here: the (group, patch) token axis is
            # group-tiled, so RoPE3D tiles its per-patch table nG times (see _cos_sin) —
            # every group's patch reuses the same spatial rotation; group identity is
            # carried by group_embed, not by position. (learnable PE also fine.)
            self.channel_groups = tuple(int(c) for c in pe_groups)
            self.n_groups = len(self.channel_groups)
            self.patch_embed = None
            self.group_proj = nn.ModuleList([
                nn.Conv3d(cg, dim, kernel_size=patch_size, stride=patch_size, padding=0)
                for cg in self.channel_groups
            ])
            self.group_embed = nn.Parameter(torch.zeros(1, self.n_groups, 1, dim))
        assert pos_encoding in ("learnable", "rope3d"), (
            f"unknown pos_encoding={pos_encoding!r} (expected 'learnable' or 'rope3d')"
        )
        self.pos_encoding = pos_encoding
        if pos_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
            self.rope = None
        else:                                    # rope3d → no absolute table; rotate q/k in attn
            self.pos_embed = None
            self.rope = RoPE3D(self.g_p, dim // n_heads, fast=rope_fast)

        # ChA-MAE memory tokens (arXiv:2503.19331): l learnable tokens prepended
        # ahead of the patch tokens before the trunk, acting as a cross-channel /
        # cross-patch scratchpad. They live inside DensityViT so they also fire on
        # the dense downstream path (train/inference consistency); conditional so
        # n_memory_tokens=0 (default) leaves the state_dict byte-identical to
        # pre-ChA checkpoints (strict-load clean in models/__init__.py).
        self.n_memory_tokens = int(n_memory_tokens)
        if self.n_memory_tokens > 0:
            assert pos_encoding == "learnable", (
                "n_memory_tokens>0 requires pos_encoding='learnable' (RoPE over "
                "prepended memory tokens is not supported)"
            )
            self.memory_tokens = nn.Parameter(torch.zeros(1, self.n_memory_tokens, dim))
        else:
            self.memory_tokens = None

        # Stochastic depth: linearly ramp the DropPath rate 0 → drop_path across blocks
        # (DeiT schedule — shallow blocks barely dropped, deepest dropped at `drop_path`).
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Patch-output decoder: each token → (c_out, p, p, p) cube.
        self.decoder_proj = nn.Linear(dim, c_out * self.patch_volume)

        self._init_weights(pos_embed_std)

    def _init_weights(self, pos_embed_std: float) -> None:
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)
        if self.group_embed is not None:
            nn.init.trunc_normal_(self.group_embed, std=pos_embed_std)
        if self.memory_tokens is not None:
            nn.init.trunc_normal_(self.memory_tokens, std=pos_embed_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tokens_to_voxels(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, c_out·p³) → (B, c_out, G, G, G)."""
        B = tokens.shape[0]
        g_p, p, c_out = self.g_p, self.patch_size, self.c_out
        # (B, N, c_out·p³) → (B, g_p, g_p, g_p, c_out, p, p, p)
        x = tokens.reshape(B, g_p, g_p, g_p, c_out, p, p, p)
        # permute spatial-blocks so they interleave correctly when reshaped:
        # (B, c_out, g_p, p, g_p, p, g_p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        # collapse (g_p, p) pairs → G per axis
        x = x.reshape(B, c_out, g_p * p, g_p * p, g_p * p)
        return x

    def embed_groups(self, density: torch.Tensor) -> torch.Tensor:
        """channel_group only: (B, C, G³) → (B, nG, N, D) per-group patch tokens with
        per-group embedding + shared spatial PE, BEFORE flattening the (group, patch)
        axes. The ChA-MAE token-drop path selects visible tokens from this; `_tokenize`
        just flattens it for the dense path."""
        chunks = torch.split(density, self.channel_groups, dim=1)           # per-group (B, c_g, G³)
        z = torch.stack([proj(c).flatten(2).transpose(1, 2)
                         for proj, c in zip(self.group_proj, chunks)], dim=1)  # (B, nG, N, D)
        z = z + self.group_embed                                            # per-group embedding
        if self.pos_embed is not None:
            z = z + self.pos_embed.unsqueeze(1)                             # patch PE, shared over groups
        return z                                                            # (B, nG, N, D)

    def _hcs_drop_groups(self, z: torch.Tensor) -> torch.Tensor:
        """ChannelViT HCS: per-forward, drop whole channel-groups (Bernoulli, prob
        `channel_group_dropout`) so the encoder sees a random non-empty subset of the
        nG groups, shared across the batch. Train-only; eval / p=0 / nG≤1 → identity
        (bit-exact, so frozen-probe + checkpoints are unchanged). True token-drop
        (removed from attention), so it needs DDP find_unused_parameters=True — the
        train loop flips that on when channel_group_dropout>0."""
        if not self.training or self.channel_group_dropout <= 0.0 or self.n_groups <= 1:
            return z
        keep = torch.rand(self.n_groups, device=z.device) >= self.channel_group_dropout
        if not bool(keep.any()):                      # guard: never drop every group
            keep[torch.randint(self.n_groups, (1,), device=z.device)] = True
        return z[:, keep]                             # (B, nG_keep, N, D)

    def _tokenize(self, density: torch.Tensor) -> torch.Tensor:
        """(B, C, G,G,G) → (B, T, D) patch tokens. fused: T=N patches. channel_group:
        T=n_groups·N — one token-set per group, +learnable group embedding, +shared patch PE.
        channel_group + HCS (train-only): T=nG_keep·N over a random non-empty group subset."""
        if self.atom_embed is not None:
            density = self.atom_embed(density)     # (B, raw) → (B, 2k+n_pt) learnable radius embed
        if self.patch_embed_mode == "fused":
            z = self.patch_embed(density).flatten(2).transpose(1, 2)        # (B, N, D)
            if self.pos_embed is not None:
                z = z + self.pos_embed
            return z
        z = self._hcs_drop_groups(self.embed_groups(density))              # (B, nG_keep, N, D)
        return z.flatten(1, 2)                                             # (B, nG_keep·N, D)

    def _pool_groups(self, z: torch.Tensor) -> torch.Tensor:
        """channel_group: mean-pool per-group tokens back to one feature/patch
        (B, nG·N, D) → (B, N, D) so the spatial output interface is unchanged. fused: no-op.
        nG is inferred from the token count so HCS (fewer live groups) pools correctly;
        bit-exact to nG=self.n_groups when no group is dropped (eval / p=0)."""
        if self.patch_embed_mode == "fused":
            return z
        B = z.shape[0]
        n_live = z.shape[1] // self.n_tokens
        return z.reshape(B, n_live, self.n_tokens, self.dim).mean(dim=1)

    def _prepend_memory(self, z: torch.Tensor) -> torch.Tensor:
        """Prepend the l learnable memory tokens ahead of the token axis (no-op when
        memory tokens are disabled)."""
        if self.memory_tokens is None:
            return z
        return torch.cat([self.memory_tokens.expand(z.shape[0], -1, -1), z], dim=1)

    def run_trunk(self, tokens: torch.Tensor, prepend_memory: bool = True) -> torch.Tensor:
        """[optional memory-prepend →] transformer blocks → final LayerNorm, applied to
        an arbitrary token set (B, T, D) → (B, [l+]T, D). The ChA-MAE token-drop path
        feeds only the visible (group, patch) tokens here; the dense `forward` feeds the
        full set. RoPE (fused mode only) is applied inside the blocks as usual."""
        if prepend_memory:
            tokens = self._prepend_memory(tokens)
        for blk in self.blocks:
            tokens = blk(tokens, rope=self.rope)
        return self.norm(tokens)

    def forward(self, density: torch.Tensor, return_tokens: bool = False):
        """density: (B, n_in_channels, G, G, G) → (B, c_out, G, G, G).

        `return_tokens=True` additionally returns the post-norm patch tokens
        (B, T_patch, D) — the SAME representation `forward_features` exposes —
        so a contrastive head can tap the pooled feature from this exact pass
        (no extra trunk forward for view-a). The voxel output is unchanged."""
        z = self._tokenize(density)                      # (B, T, D)
        z = self.run_trunk(z, prepend_memory=True)       # (B, l+T, D) post-norm
        if self.memory_tokens is not None:
            z = z[:, self.n_memory_tokens:]              # drop memory tokens before to-voxel
        tokens = z                                       # (B, T_patch, D) — probe representation
        z = self._pool_groups(z)                         # (B, N, D)
        z = self.decoder_proj(z)                         # (B, N, c_out·p³)
        vox = self._tokens_to_voxels(z)                  # (B, c_out, G, G, G)
        return (vox, tokens) if return_tokens else vox

    def forward_features(self, density: torch.Tensor) -> torch.Tensor:
        """Post-norm patch tokens (B, T_patch, D), memory tokens excluded and groups NOT
        pooled — the representation the frozen affinity probe mean-pools (it stops before
        `decoder_proj`). fused: T_patch=N; channel_group: T_patch=nG·N."""
        z = self._tokenize(density)
        z = self.run_trunk(z, prepend_memory=True)
        if self.memory_tokens is not None:
            z = z[:, self.n_memory_tokens:]
        return z

    def forward_features_memory(self, density: torch.Tensor) -> torch.Tensor:
        """Post-norm MEMORY tokens (B, l, D) — the complement of `forward_features`.

        A ChA-MAE encoder prepends `l` learnable cross-channel memory tokens through the
        trunk; `forward_features` drops them and returns the patch tokens. This returns
        the l memory tokens instead, so a probe can read off the CLS-like aggregate the
        encoder learned rather than mean-pooling the patches. Errors if the encoder was
        built without memory tokens (n_memory_tokens=0)."""
        if self.memory_tokens is None:
            raise ValueError("forward_features_memory requires n_memory_tokens>0 "
                             "(this encoder has none)")
        z = self._tokenize(density)
        z = self.run_trunk(z, prepend_memory=True)
        return z[:, : self.n_memory_tokens]

    @torch.no_grad()
    def group_attention(self, density: torch.Tensor):
        """channel_group only — the ChannelViT payoff. Returns per-layer:
            gg   (depth, nG, nG)               group→group mean attention (rows=query group)
            recv (depth, nG, g_p, g_p, g_p)    attention each (group, patch) RECEIVES
        Averaged over batch, heads, and the partner patch/group axes. Uses the explicit
        -softmax path, so run with a small batch (full (B,H,T,T) weights are held per layer)."""
        assert self.patch_embed_mode == "channel_group", "group_attention needs channel_group mode"
        mhsa = [m for m in self.modules() if isinstance(m, MultiHeadSelfAttention)]
        was_training = self.training
        self.eval()
        for m in mhsa:
            m.capture_attn = True
        try:
            self.forward(density)
            nG, N, gp = self.n_groups, self.n_tokens, self.g_p
            gg, recv = [], []
            for m in mhsa:
                A = m.attn_weights.reshape(*m.attn_weights.shape[:2], nG, N, nG, N)  # (B,H,qG,qN,kG,kN)
                # group→group: sum attention over key-patches in a group, mean over queries
                # → rows sum to 1 ("fraction of attention query-group sends to each key-group").
                gg.append(A.sum(dim=5).mean(dim=(0, 1, 3)))         # (nG, nG)
                # received: mean attention each (key-group, key-patch) gets over all queries.
                recv.append(A.mean(dim=(0, 1, 2, 3)).reshape(nG, gp, gp, gp))
            return torch.stack(gg), torch.stack(recv)
        finally:
            for m in mhsa:
                m.capture_attn = False
                m.attn_weights = None
            if was_training:
                self.train()


# ── DensityViTMAE (encoder + density head + structure head) ───────────────────

class DensityViTMAE(nn.Module):
    """Voxel-MAE / ELECTRA wrapper around `DensityViT`.

    The encoder is exposed as `self.encoder` (a `DensityViT`) so its weights
    load drop-in into `VoxBind.density_encoder` via the existing `encoder.*`
    strip in `models/__init__.py::create_model`. Which pretext head is active
    depends on `pretext_style`:

      - "mae"     : `head_density` (B, n_in_channels, G, G, G)
                    + optional `head_structure` (B, n_struct_channels, G, G, G)
      - "electra" : `head_rtd`     (B, 1, Gp, Gp, Gp)
                    + optional `head_structure` (B, n_struct_channels, G, G, G)

    `n_in_channels` widens both the patch embed and the MAE reconstruction head
    so the same wrapper handles density-only (=1), atom-blob-only (=11), and
    atom-blob+density (=12) pretraining variants.

    `n_struct_channels=0` skips the auxiliary structure head — useful when the
    atom channels are already part of the input (atomblob / atomblob_density),
    making cross-modal structure prediction redundant.

    Forward returns `(out_pretext, out_structure_or_None)`. `out_pretext` is the
    masked-voxel reconstruction in mae mode and per-patch RTD logits in electra
    mode — interpretation lives in `train_density.py::compute_losses`.
    """

    def __init__(
        self,
        grid_dim: int = 64,
        patch_size: int = 8,
        n_in_channels: int = 1,       # patch-embed input width (atoms + density + gradmag)
        n_recon_channels: Optional[int] = None,  # MAE recon / loss-target width; None → n_in_channels.
                                      # < n_in → trailing channels (e.g. input-only gradmag) encoded
                                      # but NOT reconstructed. > n_in → target-only channels the
                                      # encoder must PREDICT from a narrower input (e.g. coords-only
                                      # encoder reconstructing density+gradmag).
        n_channels: int = 32,         # backbone width; c_out = n_channels // 2
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        n_struct_channels: int = 11,  # 7 ligand + 4 pocket; 0 disables the head
        pretext_style: str = "mae",   # "mae" | "electra"
        dual_head: bool = False,      # split the MAE recon head into atoms vs density branches
        head_hidden_dim: int = 0,     # MLP-head hidden width; 0 → defaults to c_half (=n_channels//2)
        head_depth: int = 2,          # total Conv3d layers in each MAE recon head (>=1)
        head_style: str = "conv",     # "conv" = full-res Conv3d head; "patch_mlp" = per-patch token MLP; "splat" = Gaussian-splat head
        splat_k: int = 4,             # head_style="splat": # Gaussians predicted per patch (rendered analytically)
        splat_aniso: bool = False,    # splat: True → anisotropic Gaussians (per-axis scale + quaternion rotation) vs isotropic σ
        pos_encoding: str = "learnable",  # "learnable" | "rope3d" → forwarded to the DensityViT encoder
        rope_fast: bool = False,       # fused q/k rotate (rope3d only) → forwarded to the DensityViT encoder
        patch_embed_mode: str = "fused",  # "fused" | "channel_group" (ChannelViT) → forwarded to encoder
        channel_groups: Optional[Tuple[int, ...]] = None,  # channel_group: per-group channel counts
        channel_group_dropout: float = 0.0,  # ChannelViT HCS group-drop prob → forwarded to encoder
        contrastive_dim: int = 0,     # >0 builds a SimCLR projector on the pooled tokens (opt-in aux)
        radius_embed: Optional[dict] = None,  # opt-in learnable radius→k atom embedding (RadiusAtomEmbed kwargs)
        drop_path: float = 0.0,            # stochastic depth max rate (ramped over depth) → forwarded to encoder
        data2vec: bool = False,            # opt-in: build the data2vec latent-prediction predictor head
    ):
        super().__init__()
        assert pretext_style in ("mae", "electra"), (
            f"unknown pretext_style: {pretext_style!r} (expected 'mae' or 'electra')"
        )
        if dual_head:
            assert pretext_style == "mae", "dual_head only valid with pretext_style='mae'"
            assert n_in_channels >= 2, (
                f"dual_head requires n_in_channels >= 2 (atoms + density); "
                f"got n_in_channels={n_in_channels}"
            )
            assert n_recon_channels is None or n_recon_channels == n_in_channels, (
                "dual_head assumes the trailing channel is the lone density channel; "
                "it is incompatible with an input-only (n_recon < n_in) layout."
            )
        assert head_depth >= 1, f"head_depth must be >= 1, got {head_depth}"
        assert head_style in ("conv", "patch_mlp", "splat"), f"unknown head_style: {head_style!r}"
        assert not (dual_head and head_style in ("patch_mlp", "splat")), (
            f"head_style={head_style!r} is incompatible with dual_head"
        )
        n_recon = n_recon_channels if n_recon_channels is not None else n_in_channels
        # n_recon may be < n_in (trailing input-only channels, e.g. gradmag encoded but
        # not predicted) OR > n_in (target-only channels the encoder must PREDICT from a
        # narrower input, e.g. a coords-only encoder reconstructing density+gradmag). Both
        # are valid; the single MAE / patch_mlp head sizes its output to n_recon. dual_head
        # separately requires n_recon == n_in (asserted above).
        assert n_recon >= 1, f"n_recon_channels={n_recon} must be >= 1"
        self.pretext_style = pretext_style
        self.dual_head = dual_head
        c_half = n_channels // 2
        head_h = head_hidden_dim if head_hidden_dim > 0 else c_half

        def _build_head(c_in: int, c_out: int) -> nn.Module:
            """Conv3d MLP head: `head_depth` total convs, hidden width = head_h.

            depth=1: single Conv3d(c_in→c_out)               — linear projection only
            depth=2: Conv(c_in→head_h)+SiLU+Conv(head_h→c_out) — current default behavior
            depth>=3: extra SiLU+Conv(head_h→head_h) blocks inserted in the middle.
            """
            if head_depth == 1:
                return nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)
            layers = [nn.Conv3d(c_in, head_h, kernel_size=3, padding=1), nn.SiLU()]
            for _ in range(head_depth - 2):
                layers += [nn.Conv3d(head_h, head_h, kernel_size=3, padding=1), nn.SiLU()]
            layers.append(nn.Conv3d(head_h, c_out, kernel_size=3, padding=1))
            return nn.Sequential(*layers)

        # Radius-embed front-end (opt-in): DensityViT internally widens its patch_embed to
        # the post-embed width (2k+n_pt) and regroups (k,k,n_pt) while keeping n_in_channels
        # = the RAW atom+density width (what it receives & the probe must feed). The MAE still
        # reconstructs the RAW n_recon channels, so pass the raw n_in_channels / channel_groups.
        self.encoder = DensityViT(
            grid_dim=grid_dim,
            patch_size=patch_size,
            n_in_channels=n_in_channels,
            c_out=c_half,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_encoding=pos_encoding,
            rope_fast=rope_fast,
            patch_embed_mode=patch_embed_mode,
            channel_groups=channel_groups,
            channel_group_dropout=channel_group_dropout,
            radius_embed=radius_embed,
            drop_path=drop_path,
        )

        if pretext_style == "mae":
            # Two head modes:
            #   single-head (default): one MLP head that outputs ALL n_in_channels
            #     jointly. Hidden representation is shared across all output channels.
            #   dual_head: two independent heads — `head_atoms` for the leading
            #     (n_in_channels - 1) atom channels and `head_density` for the
            #     trailing 1 density channel. Hidden representations decouple,
            #     so each modality can specialize.
            #
            # In dual_head mode `head_density` outputs ONLY the 1 density channel,
            # not the full recon — concat with `head_atoms` output happens in forward.
            self.recon_mlp = None
            if dual_head:
                n_atom = n_in_channels - 1   # density is the trailing channel
                self.head_atoms = _build_head(c_half, n_atom)
                self.head_density = _build_head(c_half, 1)
            elif head_style == "patch_mlp":
                # Per-patch token MLP recon head: each patch's c_half·p³ encoder
                # features → 2-layer MLP → n_recon·p³ → unpatchify. Runs at token
                # resolution (g_p³ patches) so it scales to ~1M params cheaply
                # (vs a full-res Conv3d head, where FLOPs ≈ params × G³). Reads the
                # encoder OUTPUT, so the whole encoder (incl. its to-voxel
                # projection) is trained by the reconstruction.
                self.head_atoms = None
                self.head_density = None
                pv = patch_size ** 3
                h = head_hidden_dim if head_hidden_dim > 0 else c_half
                self.recon_mlp = nn.Sequential(
                    nn.Linear(c_half * pv, h), nn.GELU(), nn.Linear(h, n_recon * pv),
                )
            elif head_style == "splat":
                # Gaussian-splatting recon head (initial version): each patch token predicts
                # K free Gaussians — position μ∈[0,p]³ (sub-voxel), isotropic σ, and per-channel
                # amplitude — which are SPLATTED (analytically evaluated) onto the patch's p³ voxel
                # grid and summed. So the head reconstructs density as a sum of atom-like Gaussians
                # (the physical prior) rather than free voxel regression. Pretrain-only (not in
                # encoder.*) → frozen probe bit-exact. per-Gaussian params = 3 (μ) + 1 (logσ) + n_recon (amp).
                self.head_atoms = None
                self.head_density = None
                self.splat_k = int(splat_k)
                self.splat_aniso = bool(splat_aniso)
                # per-Gaussian params: 3 (μ) + amp(n_recon) + width.
                #   isotropic : 1 (log σ)
                #   anisotropic: 3 (log per-axis scale) + 4 (rotation quaternion)
                self.splat_per_g = 3 + (7 if self.splat_aniso else 1) + n_recon
                pv = patch_size ** 3
                h = head_hidden_dim if head_hidden_dim > 0 else c_half
                self.recon_mlp = nn.Sequential(
                    nn.Linear(c_half * pv, h), nn.GELU(), nn.Linear(h, self.splat_k * self.splat_per_g),
                )
                # patch voxel-center grid (p³, 3), ordered (px,py,pz) with px slowest to match
                # _unpatchify's reshape(c,p,p,p); +0.5 → voxel centers.
                _p = patch_size
                _ix = torch.arange(_p)
                _gx, _gy, _gz = torch.meshgrid(_ix, _ix, _ix, indexing="ij")
                _ctr = torch.stack([_gx, _gy, _gz], dim=-1).reshape(-1, 3).float() + 0.5  # (p³,3)
                self.register_buffer("splat_centers", _ctr, persistent=False)
            else:
                self.head_atoms = None
                self.head_density = _build_head(c_half, n_recon)
        else:
            # Per-patch RTD logits via a strided Conv3d that pools the full-res
            # encoder feature map (B, c_half, G, G, G) down to the patch grid
            # (B, 1, g_p, g_p, g_p). Mirrors `head_density`'s 2-layer pattern.
            self.recon_mlp = None
            self.head_rtd = nn.Sequential(
                nn.Conv3d(c_half, c_half, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(c_half, 1,
                          kernel_size=patch_size, stride=patch_size, padding=0),
            )

        if n_struct_channels > 0:
            self.head_structure = nn.Sequential(
                nn.Conv3d(c_half, c_half, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(c_half, n_struct_channels, kernel_size=3, padding=1),
            )
        else:
            self.head_structure = None

        # Contrastive projector (SimCLR-style, opt-in): pooled tokens (D) → MLP → proj_dim.
        # Pretrain-only — NOT part of the `encoder.*` strip, so the frozen probe never
        # sees it and downstream eval is bit-exact. None ⇒ pure MAE, no projector built.
        self.contrastive_dim = int(contrastive_dim)
        if self.contrastive_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.contrastive_dim),
            )
        else:
            self.projector = None

        # data2vec predictor (opt-in): 2-layer MLP on the per-patch pooled tokens that
        # predicts the EMA-teacher's contextualized latent at masked patches. Pretrain-only
        # (NOT in the `encoder.*` strip) → frozen probe bit-exact.
        self.data2vec = bool(data2vec)
        self.d2v_predictor = (
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            if self.data2vec else None
        )

        self.n_in_channels = n_in_channels
        self.n_recon_channels = n_recon
        self.n_channels = n_channels
        self.n_struct_channels = n_struct_channels
        self.head_style = head_style
        self.g_p = grid_dim // patch_size
        self.patch_size = patch_size
        self.n_recon = n_recon

    def encode(self, density: torch.Tensor) -> torch.Tensor:
        return self.encoder(density)

    def _splat_render(self, params: torch.Tensor) -> torch.Tensor:
        """Render K per-patch Gaussians onto the p³ voxel grid (differentiable analytic splat).
        params: (B, P, K·per_g) → recon (B, P, n_recon·p³) laid out for _unpatchify.
        Weight = exp(-½ (v-μ)ᵀ Σ⁻¹ (v-μ)); isotropic Σ=σ²I, or anisotropic Σ=R·diag(scale²)·Rᵀ."""
        B, P, _ = params.shape
        K, p, C = self.splat_k, self.patch_size, self.n_recon
        g = params.reshape(B, P, K, self.splat_per_g)
        mu = torch.sigmoid(g[..., :3]) * p                 # (B,P,K,3) sub-voxel position in [0,p]
        ctr = self.splat_centers                           # (p³, 3) voxel centers
        d = ctr.view(1, 1, 1, -1, 3) - mu.unsqueeze(3)     # (B,P,K,p³,3) voxel−center offsets
        if self.splat_aniso:
            scale = F.softplus(g[..., 3:6]) + 0.5          # (B,P,K,3) per-axis widths ≥~0.5
            R = _quat_to_rot(g[..., 6:10])                 # (B,P,K,3,3) rotation
            # local-frame offset d' = Rᵀ d, then Mahalanobis quad = Σ_i (d'_i/scale_i)²
            d_local = torch.einsum("bpkji,bpkvj->bpkvi", R, d)   # Rᵀ·d  (R indexed [.,j,i] = Rᵀ[i,j])
            quad = ((d_local / scale.unsqueeze(3)) ** 2).sum(-1)  # (B,P,K,p³)
            amp = g[..., 10:10 + C]                        # (B,P,K,C)
        else:
            sigma = F.softplus(g[..., 3]) + 0.5            # (B,P,K) isotropic width
            quad = (d ** 2).sum(-1) / sigma.unsqueeze(-1) ** 2   # (B,P,K,p³)
            amp = g[..., 4:4 + C]                          # (B,P,K,C)
        w = torch.exp(-0.5 * quad)                         # (B,P,K,p³)
        recon = torch.einsum("bpkc,bpkv->bpcv", amp, w)    # (B,P,C,p³)
        return recon.reshape(B, P, C * p ** 3)             # (B,P, n_recon·p³)

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pooled patch tokens (B, T_patch, D) → L2-normalized projection (B, proj_dim)."""
        return F.normalize(self.projector(tokens.mean(dim=1)), dim=-1)

    @torch.no_grad()
    def data2vec_target(self, density: torch.Tensor) -> torch.Tensor:
        """EMA-teacher path: per-patch pooled latent of the FULL (unmasked) input, LayerNorm'd
        → (B, N, D). Used as the regression target the student predicts at masked patches."""
        z = self.encoder._pool_groups(self.encoder.forward_features(density))  # (B, N, D)
        return F.layer_norm(z, (z.shape[-1],))

    def forward(self, density: torch.Tensor, density_b: Optional[torch.Tensor] = None,
                return_d2v: bool = False):
        """Returns (out_pretext, out_structure). When `density_b` is given (a second masked
        view) AND a projector exists, ALSO returns contrastive projections (z_a, z_b). When
        `return_d2v`, ALSO returns the data2vec student prediction (B, N, D) from the predictor
        on the pooled tokens. All extra heads run inside this one forward so DDP sees every param."""
        want_tok = (density_b is not None) or return_d2v
        want_con = density_b is not None
        enc = self.encoder(density, return_tokens=want_tok)
        z, tokens_a = enc if want_tok else (enc, None)
        if self.pretext_style == "electra":
            out_pretext = self.head_rtd(z)            # (B, 1, g_p, g_p, g_p)
        elif self.dual_head:
            out_atoms   = self.head_atoms(z)          # (B, n_in-1, G, G, G)
            out_density = self.head_density(z)        # (B, 1,      G, G, G)
            out_pretext = torch.cat([out_atoms, out_density], dim=1)  # (B, n_in, G³)
        elif self.head_style == "patch_mlp":
            tok = _patchify(z, self.g_p, self.patch_size)      # (B, g_p³, c_half·p³)
            tok = self.recon_mlp(tok)                          # (B, g_p³, n_recon·p³)
            out_pretext = _unpatchify(tok, self.g_p, self.patch_size, self.n_recon)
        elif self.head_style == "splat":
            tok = _patchify(z, self.g_p, self.patch_size)      # (B, g_p³, c_half·p³)
            params = self.recon_mlp(tok)                       # (B, g_p³, K·per_g)
            patch = self._splat_render(params)                 # (B, g_p³, n_recon·p³)
            out_pretext = _unpatchify(patch, self.g_p, self.patch_size, self.n_recon)
        else:
            out_pretext = self.head_density(z)        # (B, n_recon_channels, G, G, G)
        out_structure = (
            self.head_structure(z) if self.head_structure is not None else None
        )
        if return_d2v:
            assert self.d2v_predictor is not None, "return_d2v needs data2vec=True"
            d2v_pred = self.d2v_predictor(self.encoder._pool_groups(tokens_a))  # (B, N, D)
            return out_pretext, out_structure, d2v_pred
        if want_con:
            assert self.projector is not None, "contrastive forward needs contrastive_dim>0"
            z_a = self._project(tokens_a)                          # view-a (free, MAE pass)
            z_b = self._project(self.encoder.forward_features(density_b))  # view-b (+1 pass)
            return out_pretext, out_structure, z_a, z_b
        return out_pretext, out_structure


# ── DensityGenerator (optional learned corruption for ELECTRA) ────────────────

class DensityGenerator(nn.Module):
    """Tiny 3D CNN that fills corrupted blocks with plausible density values.

    Used only when "generator" is included in cfg.electra.corruption_ops AND
    cfg.electra.generator.enabled is True. Trained jointly with the
    discriminator via MLM-style MSE on the corrupted-voxel positions; its
    detached samples then replace the original values for those positions
    before the discriminator forward.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        return self.net(density)
