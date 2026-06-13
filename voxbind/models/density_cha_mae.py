"""density_cha_mae.py — Faithful ChA-MAEViT (arXiv:2503.19331) for 3D voxel grids.

Token-dropping multi-channel MAE on top of the grouped `DensityViT` encoder. The
2D paper targets channel-varying microscopy / satellite imagery; this is the 3D
voxel adaptation for VoxBind's multi-channel inputs (atoms + density + gradmag),
with the paper's "channel" generalized to a "channel GROUP" so cross-modal
completion (reconstruct the density group from the atom groups, etc.) is the unit
of whole-channel masking.

Pieces (paper §3):

  • per-GROUP patch tokenization + group embeddings — the ChannelViT backbone,
    reused verbatim from `DensityViT(patch_embed_mode="channel_group")`;
  • l learnable MEMORY tokens carried through the encoder trunk (they live inside
    the DensityViT, so they also fire on the dense downstream path);
  • Dynamic Channel-Patch (DCP) masking: per ITERATION either (a) random per-group
    patch masking — keep ~(1−r_p) of each group's patches, positions independent
    per group — or (b) whole-group ("channel") masking — drop k entire groups. The
    encoder sees ONLY the visible (group, patch) tokens (true MAE token-drop);
  • a lightweight Channel-Aware transformer DECODER: visible encoded tokens are
    scattered back to their slots, masked slots filled with a shared mask token,
    every slot re-tagged with its group + spatial embedding, the encoded memory
    tokens prepended, then per-group linear heads reconstruct each group's voxel
    patches;
  • reconstruction loss (paper Eq. 4) = (1−λ_f)·pixel-MSE + λ_f·Fourier-magnitude
    L1, over the MASKED patches only.

The CLS token + Hybrid Token Fusion classification head is intentionally omitted
(VoxBind's pretext has no class target). The encoder is exposed as `self.encoder`
(a `DensityViT`) so its EMA `encoder.*` slice still drops into
`VoxBind.density_encoder` and into the frozen affinity probe (which mean-pools the
post-norm patch tokens via `DensityViT.forward_features`). The downstream
DensityViT must be built with the SAME `n_memory_tokens` for the strict load in
`models/__init__.py` to match.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from voxbind.models.density_vit import DensityViT, TransformerBlock, _patchify


class DensityChaMAE(nn.Module):
    """ChA-MAEViT wrapper around a grouped `DensityViT` encoder (see module docstring)."""

    def __init__(
        self,
        grid_dim: int = 64,
        patch_size: int = 8,
        channel_groups: Tuple[int, ...] = (7, 4, 1, 1),  # e.g. 7 lig + 4 poc + density + gradmag
        n_in_channels: Optional[int] = None,             # default = sum(channel_groups)
        n_channels: int = 32,                            # backbone c_out = n_channels // 2 (dense path)
        # ── encoder (DensityViT) ──
        dim: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        n_memory_tokens: int = 4,
        # ── channel-aware decoder (lightweight) ──
        dec_dim: int = 256,
        dec_depth: int = 2,
        dec_heads: int = 8,
        dec_mlp_ratio: int = 4,
        # ── DCP masking ──
        patch_mask_ratio: float = 0.75,                  # r_p — fraction of patches masked in patch mode
        dcp_strategy: str = "alternate",                 # "alternate" | "patch" | "channel"
        channel_mask_min: int = 1,                       # min # groups masked in channel mode
        channel_mask_max: Optional[int] = None,          # max; default nG-1 (keep ≥1 group)
        # ── loss ──
        lambda_fourier: float = 0.01,
    ):
        super().__init__()
        groups = tuple(int(c) for c in channel_groups)
        n_in = int(n_in_channels) if n_in_channels is not None else sum(groups)
        assert sum(groups) == n_in, (
            f"channel_groups {groups} must sum to n_in_channels {n_in}"
        )
        nG = len(groups)
        assert nG >= 2, f"ChA-MAE needs ≥2 channel groups for cross-channel masking, got {groups}"
        assert dcp_strategy in ("alternate", "patch", "channel"), (
            f"unknown dcp_strategy={dcp_strategy!r}"
        )
        assert 0.0 < patch_mask_ratio < 1.0, f"patch_mask_ratio must be in (0,1), got {patch_mask_ratio}"
        cmax = (nG - 1) if channel_mask_max is None else int(channel_mask_max)
        assert 1 <= channel_mask_min <= cmax <= nG - 1, (
            f"channel mask range [{channel_mask_min}, {cmax}] invalid for nG={nG} "
            f"(must keep ≥1 group and mask ≥1)"
        )
        self.channel_groups = groups
        self.n_groups = nG
        self.n_in_channels = n_in
        self.patch_size = patch_size
        self.g_p = grid_dim // patch_size
        self.n_tokens = self.g_p ** 3
        self.patch_volume = patch_size ** 3
        self.n_memory_tokens = int(n_memory_tokens)
        self.patch_mask_ratio = float(patch_mask_ratio)
        self.dcp_strategy = dcp_strategy
        self.channel_mask_min = int(channel_mask_min)
        self.channel_mask_max = cmax
        self.lambda_fourier = float(lambda_fourier)
        # per-group channel offsets into the n_in-channel input: group g = [offs[g]:offs[g+1])
        offs = [0]
        for c in groups:
            offs.append(offs[-1] + c)
        self.group_offsets = offs                        # len nG+1

        # ── Encoder: grouped DensityViT with memory tokens (learnable PE) ──
        self.encoder = DensityViT(
            grid_dim=grid_dim,
            patch_size=patch_size,
            n_in_channels=n_in,
            c_out=n_channels // 2,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_encoding="learnable",
            patch_embed_mode="channel_group",
            channel_groups=groups,
            n_memory_tokens=n_memory_tokens,
        )

        # ── Channel-aware decoder ──
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        # spatial PE (shared across groups) + per-group (channel) embedding
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dec_dim))
        self.dec_group_embed = nn.Parameter(torch.zeros(1, nG, 1, dec_dim))
        self.dec_memory_embed = nn.Linear(dim, dec_dim) if self.n_memory_tokens > 0 else None
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dec_dim, dec_heads, mlp_ratio=dec_mlp_ratio, dropout=dropout)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        # per-group reconstruction heads: decoder token → c_g · p³ voxel values
        self.decoder_heads = nn.ModuleList([
            nn.Linear(dec_dim, c * self.patch_volume) for c in groups
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_group_embed, std=0.02)
        for m in self.decoder_blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        heads = [self.decoder_embed, *self.decoder_heads]
        if self.dec_memory_embed is not None:
            heads.append(self.dec_memory_embed)
        for m in heads:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ── DCP masking ───────────────────────────────────────────────────────────

    @staticmethod
    def _sample_int(lo: int, hi: int, device, generator=None) -> int:
        if hi <= lo:
            return int(lo)
        return int(torch.randint(lo, hi + 1, (1,), device=device, generator=generator).item())

    def _dcp_mask(self, B: int, device, generator=None):
        """Build the per-iteration keep/drop layout over the T = nG·N tokens (token
        index t = g·N + n, group-major, matching `DensityViT.embed_groups().flatten`).

        Returns:
            ids_keep    (B, V)   original indices of the kept (visible) tokens
            ids_restore (B, T)   inverse permutation of [keep; mask] → original order
            mask        (B, T)   float, 1 = masked / reconstructed, 0 = visible
            mode        str      "patch" | "channel" (which DCP branch fired)
        V is constant across the batch (same per-sample keep count), so gather/scatter
        stay rectangular.
        """
        nG, N = self.n_groups, self.n_tokens
        T = nG * N
        if self.dcp_strategy == "patch":
            mode = "patch"
        elif self.dcp_strategy == "channel":
            mode = "channel"
        else:  # alternate: coin-flip per iteration
            coin = torch.rand((), device=device, generator=generator)
            mode = "patch" if float(coin) < 0.5 else "channel"

        goff = (torch.arange(nG, device=device).view(1, nG, 1)) * N    # (1, nG, 1)
        patch_ar = torch.arange(N, device=device)

        if mode == "patch":
            keep_n = max(1, round(N * (1.0 - self.patch_mask_ratio)))
            noise = torch.rand(B, nG, N, device=device, generator=generator)
            ids_sort = torch.argsort(noise, dim=2)                     # (B, nG, N)
            keep_idx = ids_sort[:, :, :keep_n]                         # (B, nG, keep_n)
            mask_idx = ids_sort[:, :, keep_n:]                         # (B, nG, N-keep_n)
            ids_keep = (keep_idx + goff).reshape(B, -1)                # (B, nG·keep_n)
            ids_mask = (mask_idx + goff).reshape(B, -1)
            mask_gn = torch.ones(B, nG, N, device=device)
            mask_gn.scatter_(2, keep_idx, 0.0)                         # kept → 0
            mask = mask_gn.reshape(B, T)
        else:  # channel: drop k whole groups, keep the rest fully
            k = self._sample_int(self.channel_mask_min, self.channel_mask_max, device, generator)
            gperm = torch.argsort(torch.rand(B, nG, device=device, generator=generator), dim=1)
            keep_groups = gperm[:, : nG - k]                           # (B, nG-k)
            mask_groups = gperm[:, nG - k :]                           # (B, k)
            ids_keep = (keep_groups.unsqueeze(-1) * N + patch_ar.view(1, 1, N)).reshape(B, -1)
            ids_mask = (mask_groups.unsqueeze(-1) * N + patch_ar.view(1, 1, N)).reshape(B, -1)
            mask_g = torch.zeros(B, nG, device=device)
            mask_g.scatter_(1, mask_groups, 1.0)                       # dropped groups → 1
            mask = mask_g.unsqueeze(-1).expand(B, nG, N).reshape(B, T).contiguous()

        ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)          # (B, T) permutation
        ids_restore = torch.argsort(ids_shuffle, dim=1)               # (B, T)
        return ids_keep, ids_restore, mask, mode

    # ── forward / loss ──────────────────────────────────────────────────────────

    def forward(self, x_clean: torch.Tensor, generator=None) -> Dict[str, torch.Tensor]:
        """x_clean: (B, n_in_channels, G, G, G) — clean multi-channel voxel input.

        Runs DCP token-drop MAE and returns a dict of losses (`L_recon`, `L_pixel`,
        `L_fourier`) plus `mask_ratio` (scalar tensor) for logging. Masking and the
        decoder live here; the encoder only ever sees the visible tokens.
        """
        B, device = x_clean.shape[0], x_clean.device
        nG, N = self.n_groups, self.n_tokens
        l = self.n_memory_tokens

        tokens = self.encoder.embed_groups(x_clean).flatten(1, 2)     # (B, nG·N, D)
        D = tokens.shape[-1]
        ids_keep, ids_restore, mask, mode = self._dcp_mask(B, device, generator)
        V = ids_keep.shape[1]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))   # (B, V, D)

        enc = self.encoder.run_trunk(visible, prepend_memory=True)    # (B, l+V, D) post-norm
        mem_enc, vis_enc = enc[:, :l], enc[:, l:]                     # (B, l, D), (B, V, D)

        # decoder: scatter visible tokens back, fill holes with the mask token
        vis_dec = self.decoder_embed(vis_enc)                         # (B, V, dec)
        dec_dim = vis_dec.shape[-1]
        T = nG * N
        pad = self.mask_token.expand(B, T - V, dec_dim)
        x_ = torch.cat([vis_dec, pad], dim=1)                        # (B, T, dec) in [keep; mask] order
        x_full = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, dec_dim))  # original order
        # channel-aware tagging: per-group embedding + shared spatial PE
        x_full = x_full.view(B, nG, N, dec_dim)
        x_full = x_full + self.dec_group_embed + self.dec_pos_embed.unsqueeze(1)
        x_full = x_full.reshape(B, T, dec_dim)
        if l > 0:
            seq = torch.cat([self.dec_memory_embed(mem_enc), x_full], dim=1)   # prepend encoded memory
        else:
            seq = x_full
        for blk in self.decoder_blocks:
            seq = blk(seq)                                           # no RoPE in the decoder
        seq = self.decoder_norm(seq)
        dec_tokens = seq[:, l:]                                      # (B, T, dec); drop memory

        return self._recon_loss(dec_tokens, x_clean, mask)

    def _recon_loss(self, dec_tokens, x_clean, mask) -> Dict[str, torch.Tensor]:
        """Per-group reconstruction → masked pixel-MSE + Fourier-magnitude L1.

        Loss is averaged over masked (group, patch) tokens (paper's 1/P, with the
        group as the channel unit): each masked token's loss is the mean over its
        c_g·p³ voxels, so a 7-channel ligand token and a 1-channel density token are
        weighted equally per token.
        """
        B = x_clean.shape[0]
        nG, N, p, gp = self.n_groups, self.n_tokens, self.patch_size, self.g_p
        mask_gn = mask.view(B, nG, N)                                # 1 = masked
        n_masked = mask.sum().clamp(min=1.0)
        pix_num = x_clean.new_zeros(())
        fou_num = x_clean.new_zeros(())
        use_fourier = self.lambda_fourier > 0.0
        for g in range(nG):
            c0, c1 = self.group_offsets[g], self.group_offsets[g + 1]
            cg = c1 - c0
            tok_g = dec_tokens[:, g * N : (g + 1) * N]               # (B, N, dec)
            pred = self.decoder_heads[g](tok_g).float()             # (B, N, cg·p³)
            target = _patchify(x_clean[:, c0:c1], gp, p).float()    # (B, N, cg·p³)
            m_g = mask_gn[:, g]                                     # (B, N)
            pix = ((pred - target) ** 2).mean(dim=-1)              # (B, N) per-patch MSE
            pix_num = pix_num + (pix * m_g).sum()
            if use_fourier:
                pr = pred.reshape(B, N, cg, p, p, p)
                tg = target.reshape(B, N, cg, p, p, p)
                Pf = torch.fft.rfftn(pr, dim=(-3, -2, -1)).abs()
                Tf = torch.fft.rfftn(tg, dim=(-3, -2, -1)).abs()
                fou = (Pf - Tf).abs().mean(dim=(-4, -3, -2, -1))   # (B, N) over channels + freq
                fou_num = fou_num + (fou * m_g).sum()
        L_pixel = pix_num / n_masked
        L_fourier = fou_num / n_masked
        L_recon = (1.0 - self.lambda_fourier) * L_pixel + self.lambda_fourier * L_fourier
        return {
            "L_recon": L_recon,
            "L_pixel": L_pixel.detach(),
            "L_fourier": L_fourier.detach(),
            "mask_ratio": (mask.sum() / mask.numel()).detach(),
        }

    @torch.no_grad()
    def encode_dense(self, density: torch.Tensor) -> torch.Tensor:
        """Convenience: the dense (B, c_out, G, G, G) feature from the underlying
        DensityViT — the drop-in `density_encoder` output. Not used in pretraining."""
        return self.encoder(density)
