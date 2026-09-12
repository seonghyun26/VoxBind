"""density_multimae.py — MultiMAE (Bachmann et al., ECCV 2022, arXiv:2204.01678) for
3-modality voxel grids.

The original MultiMAE pretrains a SINGLE shared ViT on a jointly-masked set of tokens
drawn from several spatially-aligned image modalities (RGB / depth / segmentation), and
reconstructs each modality with its OWN shallow decoder that cross-attends to the full set
of encoded tokens (so a masked depth patch is inpainted from the visible RGB patches, etc.).
That cross-modal completion objective is exactly what we want for deployability: at real
inference no crystal / X-ray density exists, so the encoder must reconstruct — and represent
— the density modality from the atom modalities alone. Ref. arXiv:2509.11442 (MultiMAE for
brain MRIs with MISSING modalities) is the same missing-modality-robustness use case.

3D voxel adaptation for VoxBind's grouped input (channel_groups sum = n_in_channels):

  • MODALITIES = the channel groups. Default [7, 4, 2] = 7 ligand atom types /
    4 pocket atom types / (density + gradmag). All three share ONE voxel grid, so unlike
    the 2D paper the token grids are perfectly aligned by construction.
  • per-modality patch tokenization + group/positional embeddings — reused verbatim from
    `DensityViT(patch_embed_mode="channel_group")` (the ChannelViT backbone), identical to
    the CDG-v2 champion encoder so a frozen affinity probe is apples-to-apples.
  • MULTI-MODAL masking (paper §3.1): keep a fixed budget V = keep_ratio · (nG·N) of all
    tokens; the per-modality split of that budget is drawn from a symmetric
    Dirichlet(alpha) (alpha=1 ⇒ uniform over the simplex, so some samples see a modality
    fully masked). The encoder sees ONLY the visible tokens (true MAE token-drop).
  • MULTI-TASK decoders (paper §3.2): one shallow decoder per modality. Each = a single
    CROSS-attention layer (queries = that modality's N slots — real encoded tokens where
    visible, a shared mask token where masked; keys/values = ALL encoded tokens across every
    modality) + MLP, then `dec_depth` self-attention blocks, then a per-modality linear head
    to c_g·p³ voxel values. Cross-attention is where the cross-modal signal enters.
  • loss = mean over modalities of the masked-token MSE (equal task weighting, paper default),
    with per-modality terms surfaced for logging. Optional Fourier-magnitude L1 (off by default).

Interface matches `DensityChaMAE`: `forward(x_clean, generator=None) -> dict` of scalar-tensor
losses, so it plugs into the shared MAE registry (`train_epoch_multimae`/`val_epoch_multimae`).
The encoder is exposed as `self.encoder` (a `DensityViT`) so its EMA `encoder.*` slice drops
straight into `VoxBind.density_encoder` and the frozen probe. Built with n_memory_tokens=0 by
default so the downstream probe path is byte-identical to the champion CDG encoder.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from voxbind.models.density_vit import DensityViT, TransformerBlock, _patchify


class CrossAttentionBlock(nn.Module):
    """One MultiMAE decoder step: cross-attention (queries attend to a separate key/value
    context) + MLP, pre-norm residual. Plain `nn.MultiheadAttention` (batch_first) — the
    MultiMAE decoder carries no RoPE."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_n = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv_n, kv_n, need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


class DensityMultiMAE(nn.Module):
    """MultiMAE over grouped voxel modalities (see module docstring)."""

    def __init__(
        self,
        grid_dim: int = 64,
        patch_size: int = 8,
        channel_groups: Tuple[int, ...] = (7, 4, 2),   # ligand / pocket / (density+gradmag)
        n_in_channels: Optional[int] = None,           # default = sum(channel_groups)
        n_channels: int = 32,                          # backbone c_out = n_channels // 2 (dense path)
        # ── encoder (DensityViT) ──
        dim: int = 640,
        depth: int = 18,
        n_heads: int = 10,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        n_memory_tokens: int = 0,                      # 0 ⇒ identical frozen-probe path to champion
        # ── per-modality decoders ──
        dec_dim: int = 256,
        dec_depth: int = 2,
        dec_heads: int = 8,
        dec_mlp_ratio: int = 4,
        # ── MultiMAE Dirichlet masking ──
        keep_ratio: float = 1.0 / 6.0,                 # fraction of all tokens kept visible
        dirichlet_alpha: float = 1.0,                  # symmetric Dirichlet concentration
        # ── loss ──
        lambda_fourier: float = 0.0,                   # Fourier-magnitude L1 weight (0 = pure MSE)
        modality_weights: Optional[List[float]] = None,  # per-modality loss weights (default equal)
        # ── audit-fix opt-ins (paper-faithful; default off = current behaviour) ──
        norm_pix: bool = False,                        # per-patch target standardization (MAE norm_pix_loss)
        force_mask_groups: Tuple[int, ...] = (),       # modalities ALWAYS masked at encoder (recon-only);
                                                       # e.g. (2,) = density never seen by encoder → pure-atoms
    ):
        super().__init__()
        groups = tuple(int(c) for c in channel_groups)
        n_in = int(n_in_channels) if n_in_channels is not None else sum(groups)
        assert sum(groups) == n_in, f"channel_groups {groups} must sum to n_in_channels {n_in}"
        nG = len(groups)
        assert nG >= 2, f"MultiMAE needs >=2 modalities, got {groups}"
        assert 0.0 < keep_ratio < 1.0, f"keep_ratio must be in (0,1), got {keep_ratio}"
        self.channel_groups = groups
        self.n_groups = nG
        self.n_in_channels = n_in
        self.patch_size = patch_size
        self.g_p = grid_dim // patch_size
        self.n_tokens = self.g_p ** 3                  # N patches per modality
        self.patch_volume = patch_size ** 3
        self.n_memory_tokens = int(n_memory_tokens)
        self.keep_ratio = float(keep_ratio)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.lambda_fourier = float(lambda_fourier)
        self.norm_pix = bool(norm_pix)
        self.force_mask_groups = tuple(sorted({int(g) for g in force_mask_groups}))
        assert all(0 <= g < nG for g in self.force_mask_groups), \
            f"force_mask_groups {self.force_mask_groups} out of range for nG={nG}"
        assert len(self.force_mask_groups) < nG, "force_mask_groups must leave >=1 visible modality"
        T = nG * self.n_tokens
        self.n_visible = max(nG, int(round(self.keep_ratio * T)))   # V, constant across batch
        # per-group channel offsets into the input: group g = [offs[g]:offs[g+1])
        offs = [0]
        for c in groups:
            offs.append(offs[-1] + c)
        self.group_offsets = offs
        # per-modality loss weights (equal by default ⇒ mean over modalities)
        if modality_weights is None:
            w = [1.0 / nG] * nG
        else:
            assert len(modality_weights) == nG, "modality_weights must have one entry per group"
            s = float(sum(modality_weights))
            w = [float(x) / s for x in modality_weights]
        self.register_buffer("modality_weights", torch.tensor(w), persistent=False)
        # friendly per-modality names for logging
        if nG == 3:
            self._mod_names = ["L_lig", "L_poc", "L_dens"]
        else:
            self._mod_names = [f"L_g{g}" for g in range(nG)]

        # ── Encoder: grouped DensityViT (same as champion CDG) ──
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

        # ── Per-modality decoders ──
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dec_dim))   # spatial PE (shared)
        self.dec_mod_embed = nn.Parameter(torch.zeros(1, nG, dec_dim))              # per-modality embed
        if self.n_memory_tokens > 0:
            self.dec_memory_embed = nn.Linear(dim, dec_dim)
            self.mem_kv_embed = nn.Parameter(torch.zeros(1, 1, dec_dim))            # tags memory in K/V
        else:
            self.dec_memory_embed = None
            self.mem_kv_embed = None
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(dec_dim, dec_heads, mlp_ratio=dec_mlp_ratio, dropout=dropout)
            for _ in groups
        ])
        self.self_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(dec_dim, dec_heads, mlp_ratio=dec_mlp_ratio, dropout=dropout)
                for _ in range(dec_depth)
            ])
            for _ in groups
        ])
        self.decoder_norms = nn.ModuleList([nn.LayerNorm(dec_dim) for _ in groups])
        self.decoder_heads = nn.ModuleList([
            nn.Linear(dec_dim, c * self.patch_volume) for c in groups
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_mod_embed, std=0.02)
        if self.mem_kv_embed is not None:
            nn.init.trunc_normal_(self.mem_kv_embed, std=0.02)
        for mod in (self.cross_blocks, self.self_blocks, self.decoder_norms):
            for m in mod.modules():
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

    # ── MultiMAE Dirichlet masking ───────────────────────────────────────────────

    def _multimodal_mask(self, B: int, device, generator=None):
        """Sample the MultiMAE joint mask over T = nG·N tokens (token t = g·N + n,
        group-major, matching `embed_groups().flatten(1, 2)`).

        Budget V = self.n_visible is fixed; the per-modality split (v_0, …, v_{nG-1}),
        sum = V, is drawn per sample from a symmetric Dirichlet(alpha). Returns:
            ids_keep    (B, V)      original indices of visible tokens
            ids_restore (B, T)      inverse of [keep; mask] → original order
            mask        (B, T)      float, 1 = masked / reconstructed, 0 = visible
            vis_map     (B, nG, N)  bool, True = visible
        """
        nG, N = self.n_groups, self.n_tokens
        T = nG * N
        V = self.n_visible
        # symmetric Dirichlet(alpha) over the nG modalities (Gamma(alpha,1) normalised).
        # alpha == 1 ⇒ Gamma(1) = Exponential = -log U, generator-friendly (used in val).
        if abs(self.dirichlet_alpha - 1.0) < 1e-6:
            u = torch.rand(B, nG, device=device, generator=generator).clamp_(min=1e-9)
            gamma = -torch.log(u)
        else:
            gamma = torch._standard_gamma(
                torch.full((B, nG), self.dirichlet_alpha, device=device))
        if self.force_mask_groups:                               # forced modalities → 0 gamma → 0 visible
            gamma[:, list(self.force_mask_groups)] = 0.0         # (encoder never sees them; recon-only)
        p = gamma / gamma.sum(dim=1, keepdim=True)               # (B, nG) Dirichlet sample
        # integer per-modality budgets summing to V: floor + distribute remainder by frac part
        scaled = p * V
        v = torch.floor(scaled).long()                           # (B, nG)
        rem = V - v.sum(dim=1)                                   # (B,) tokens still to assign (>=0)
        frac_order = torch.argsort(scaled - v.float(), dim=1, descending=True)   # (B, nG)
        add_slot = torch.arange(nG, device=device).unsqueeze(0) < rem.unsqueeze(1)  # (B, nG)
        add = torch.zeros_like(v)
        add.scatter_(1, frac_order, add_slot.long())
        v = (v + add).clamp_(min=0, max=N)                       # (B, nG), sum == V (N large ⇒ no clamp)
        # per-modality: keep the v_g lowest-noise positions
        noise = torch.rand(B, nG, N, device=device, generator=generator)
        rank = noise.argsort(dim=2).argsort(dim=2)               # (B, nG, N) rank 0..N-1
        vis_map = rank < v.unsqueeze(-1)                         # (B, nG, N) bool, exactly v_g True
        flat = vis_map.reshape(B, T)                             # exactly V True per row
        mask = 1.0 - flat.float()                               # 1 = masked
        # visible indices first (stable ⇒ ascending original index), then masked
        _, order = torch.sort(flat.to(torch.float32), dim=1, descending=True, stable=True)
        ids_keep = order[:, :V].contiguous()
        ids_mask = order[:, V:].contiguous()
        ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        return ids_keep, ids_restore, mask, vis_map

    # ── forward / loss ───────────────────────────────────────────────────────────

    def forward(self, x_clean: torch.Tensor, generator=None) -> Dict[str, torch.Tensor]:
        """x_clean: (B, n_in_channels, G, G, G). Runs MultiMAE token-drop + per-modality
        cross-attention decoders; returns a dict of scalar-tensor losses for logging."""
        B, device = x_clean.shape[0], x_clean.device
        nG, N = self.n_groups, self.n_tokens
        l = self.n_memory_tokens
        T = nG * N

        tokens = self.encoder.embed_groups(x_clean).flatten(1, 2)      # (B, T, D)
        D = tokens.shape[-1]
        ids_keep, ids_restore, mask, _vis_map = self._multimodal_mask(B, device, generator)
        V = ids_keep.shape[1]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))   # (B, V, D)

        enc = self.encoder.run_trunk(visible, prepend_memory=(l > 0))  # (B, l+V, D) post-norm
        mem_enc, vis_enc = enc[:, :l], enc[:, l:]                      # (B, l, D), (B, V, D)

        # scatter decoder-embedded visible tokens back to the full nG·N grid; fill holes with
        # the shared mask token; tag every slot with its (modality, spatial) embedding.
        vis_dec = self.decoder_embed(vis_enc)                          # (B, V, dec)
        dec_dim = vis_dec.shape[-1]
        pad = self.mask_token.expand(B, T - V, dec_dim)
        x_ = torch.cat([vis_dec, pad], dim=1)                          # [keep; mask] order
        x_full = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, dec_dim))  # original order
        x_full = x_full.view(B, nG, N, dec_dim)
        x_full = x_full + self.dec_mod_embed.unsqueeze(2) + self.dec_pos_embed.unsqueeze(1)
        x_full_flat = x_full.reshape(B, T, dec_dim)

        # shared cross-attention context (keys/values) = memory (if any) + ALL visible tokens,
        # each carrying its (modality, spatial) embedding so the attention is position-aware.
        vis_ctx = torch.gather(x_full_flat, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dec_dim))  # (B, V, dec)
        if l > 0:
            mem_ctx = self.dec_memory_embed(mem_enc) + self.mem_kv_embed           # (B, l, dec)
            context = torch.cat([mem_ctx, vis_ctx], dim=1)                          # (B, l+V, dec)
        else:
            context = vis_ctx

        # per-modality decoder: cross-attend its N slots to the full context, then self-attend
        preds: List[torch.Tensor] = []
        for g in range(nG):
            q = x_full[:, g]                                           # (B, N, dec)
            q = self.cross_blocks[g](q, context)
            for blk in self.self_blocks[g]:
                q = blk(q)
            q = self.decoder_norms[g](q)
            preds.append(self.decoder_heads[g](q))                    # (B, N, c_g·p³)

        return self._recon_loss(preds, x_clean, mask)

    def _recon_loss(self, preds: List[torch.Tensor], x_clean: torch.Tensor,
                    mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-modality masked-token reconstruction. L_recon = Σ_g w_g · L_g with equal task
        weights by default (paper). Each L_g is the masked-token mean of the per-token voxel
        MSE, so modalities are balanced regardless of channel count / masked-token count."""
        B = x_clean.shape[0]
        nG, N, p, gp = self.n_groups, self.n_tokens, self.patch_size, self.g_p
        mask_gn = mask.view(B, nG, N)                                 # 1 = masked
        use_fourier = self.lambda_fourier > 0.0
        w = self.modality_weights
        L_pixel = x_clean.new_zeros(())
        L_fourier = x_clean.new_zeros(())
        out: Dict[str, torch.Tensor] = {}
        for g in range(nG):
            c0, c1 = self.group_offsets[g], self.group_offsets[g + 1]
            cg = c1 - c0
            pred = preds[g].float()                                   # (B, N, cg·p³)
            target = _patchify(x_clean[:, c0:c1], gp, p).float()      # (B, N, cg·p³)
            if self.norm_pix:                                         # MAE norm_pix_loss: per-patch standardize
                mu = target.mean(dim=-1, keepdim=True)
                sd = target.std(dim=-1, keepdim=True) + 1e-6
                target = (target - mu) / sd
            m_g = mask_gn[:, g]                                       # (B, N)
            denom = m_g.sum().clamp(min=1.0)
            pix = ((pred - target) ** 2).mean(dim=-1)                # (B, N) per-token MSE
            L_g = (pix * m_g).sum() / denom
            out[self._mod_names[g]] = L_g.detach()
            L_pixel = L_pixel + w[g] * L_g
            if use_fourier:
                pr = pred.reshape(B, N, cg, p, p, p)
                tg = target.reshape(B, N, cg, p, p, p)
                Pf = torch.fft.rfftn(pr, dim=(-3, -2, -1)).abs()
                Tf = torch.fft.rfftn(tg, dim=(-3, -2, -1)).abs()
                fou = (Pf - Tf).abs().mean(dim=(-4, -3, -2, -1))     # (B, N)
                L_fg = (fou * m_g).sum() / denom
                L_fourier = L_fourier + w[g] * L_fg
        L_recon = (1.0 - self.lambda_fourier) * L_pixel + self.lambda_fourier * L_fourier
        out["L_recon"] = L_recon
        out["L_pixel"] = L_pixel.detach()
        out["L_fourier"] = L_fourier.detach()
        out["mask_ratio"] = (mask.sum() / mask.numel()).detach()
        return out

    @torch.no_grad()
    def encode_dense(self, density: torch.Tensor) -> torch.Tensor:
        """Convenience: the dense (B, c_out, G, G, G) feature from the underlying DensityViT —
        the drop-in `density_encoder` output. Not used in pretraining."""
        return self.encoder(density)
