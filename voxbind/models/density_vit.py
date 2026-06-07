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


# ── Transformer building blocks ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA via `F.scaled_dot_product_attention` (flash on Ampere+)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN ViT block: x + MHSA(LN(x)) → x + MLP(LN(x))."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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
    ):
        super().__init__()
        assert grid_dim % patch_size == 0, (
            f"grid_dim={grid_dim} must be divisible by patch_size={patch_size}"
        )
        self.grid_dim = grid_dim
        self.patch_size = patch_size
        self.n_in_channels = n_in_channels
        self.c_out = c_out
        self.dim = dim
        self.depth = depth

        self.g_p = grid_dim // patch_size
        self.n_tokens = self.g_p ** 3
        self.patch_volume = patch_size ** 3

        self.patch_embed = nn.Conv3d(
            n_in_channels, dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Patch-output decoder: each token → (c_out, p, p, p) cube.
        self.decoder_proj = nn.Linear(dim, c_out * self.patch_volume)

        self._init_weights(pos_embed_std)

    def _init_weights(self, pos_embed_std: float) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)
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

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """density: (B, n_in_channels, G, G, G) → (B, c_out, G, G, G)."""
        z = self.patch_embed(density)                    # (B, D, g_p, g_p, g_p)
        z = z.flatten(2).transpose(1, 2)                 # (B, N, D)
        z = z + self.pos_embed
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        z = self.decoder_proj(z)                         # (B, N, c_out·p³)
        return self._tokens_to_voxels(z)                 # (B, c_out, G, G, G)


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
    mode — interpretation lives in `train_density_vit_mae.py::compute_losses`.
    """

    def __init__(
        self,
        grid_dim: int = 64,
        patch_size: int = 8,
        n_in_channels: int = 1,       # patch-embed input width (atoms + density + gradmag)
        n_recon_channels: Optional[int] = None,  # MAE recon / loss-target width; None → n_in_channels.
                                      # Set < n_in to make trailing channels (e.g. an input-only
                                      # gradmag) encoded but NOT reconstructed.
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
        head_style: str = "conv",     # "conv" = full-res Conv3d head; "patch_mlp" = per-patch token MLP
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
        assert head_style in ("conv", "patch_mlp"), f"unknown head_style: {head_style!r}"
        assert not (dual_head and head_style == "patch_mlp"), (
            "head_style='patch_mlp' is incompatible with dual_head"
        )
        n_recon = n_recon_channels if n_recon_channels is not None else n_in_channels
        assert 1 <= n_recon <= n_in_channels, (
            f"n_recon_channels={n_recon} must be in [1, n_in_channels={n_in_channels}]"
        )
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

    def forward(
        self, density: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.encoder(density)
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
        else:
            out_pretext = self.head_density(z)        # (B, n_recon_channels, G, G, G)
        out_structure = (
            self.head_structure(z) if self.head_structure is not None else None
        )
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
