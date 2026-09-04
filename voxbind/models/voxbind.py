import math
import random
import torch
from typing import Tuple, Union, List

from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS
from voxbind.models.density_vit import DensityViT
from voxbind.models.mae_ops import (attenuate_density, gaussian_blur3d,
                                    gradient_magnitude3d, per_sample_zscore,
                                    protein_attenuation_field, region_keep_masks)
from voxbind.models.unet3d import UNet3D, ResidualBlock


class PocketAdapter(torch.nn.Module):
    """Trainable spatial adapter that turns the FROZEN pretrained pocket encoder's post-norm
    patch tokens into a residual VoxBind conditioning volume.

    tokens (B, N=g_p³, D)  →  reshape to the g_p³ patch grid (B, D, g_p, g_p, g_p)
                           →  1×1 project D→hidden (cheap, at patch res)
                           →  trilinear upsample ×patch_size to full (G, G, G)
                           →  3×3 refine  →  3×3 zero-init out (→ c_out)

    The final conv is zero-initialised so the adapter contributes exactly 0 at step 0 (the frozen
    VoxBind baseline is reproduced) and its influence is learned from there — a stable residual.
    Spatial structure is preserved throughout (no global pooling)."""

    def __init__(self, dim: int, g_p: int, patch_size: int, c_out: int, hidden: int = None):
        super().__init__()
        self.g_p = g_p
        h = int(hidden) if hidden else c_out
        self.proj = torch.nn.Conv3d(dim, h, kernel_size=1)
        self.up = torch.nn.Upsample(scale_factor=patch_size, mode="trilinear", align_corners=False)
        self.refine = torch.nn.Sequential(
            torch.nn.Conv3d(h, h, kernel_size=3, padding=1), torch.nn.SiLU())
        self.out = torch.nn.Conv3d(h, c_out, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = pooled_tokens.shape
        gp = self.g_p
        x = pooled_tokens.transpose(1, 2).reshape(B, D, gp, gp, gp)   # (B, D, g_p³ grid)
        x = self.proj(x)                                              # (B, h, g_p³)
        x = self.up(x)                                               # (B, h, G, G, G)
        x = self.refine(x)
        return self.out(x)                                          # (B, c_out, G, G, G), starts at 0


def _intra_patch_offsets(grid_dim: int, patch: int) -> torch.Tensor:
    """(1, 3, G, G, G) — each voxel's position WITHIN its own patch, normalised to [-1, 1].

    A ViT token is constant across the p³ voxels of its patch, so a per-voxel MLP fed the
    broadcast token alone cannot tell those voxels apart on the density side. One axis per
    channel restores a position-dependent readout without reusing the frozen MAE basis.

    Grid-index space, which is the augmented frame both branches already share: the rotation
    augmentation is 90° axis swaps/flips, so it maps grid indices to grid indices and patch
    boundaries stay aligned. Deterministic ⇒ registered non-persistent (never in a checkpoint).
    """
    off = (torch.arange(grid_dim) % patch).float() / max(patch - 1, 1) * 2.0 - 1.0   # (G,)
    shape = (1, 1, grid_dim, grid_dim, grid_dim)
    return torch.cat([
        off.view(1, 1, -1, 1, 1).expand(shape),
        off.view(1, 1, 1, -1, 1).expand(shape),
        off.view(1, 1, 1, 1, -1).expand(shape),
    ], dim=1).contiguous()


class VoxBind(torch.nn.Module):
    def __init__(
        self,
        n_channels_ligand: int = N_LIGAND_ELEMENTS,
        n_channels_pocket: int = N_POCKET_ELEMENTS,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        with_density: bool = False,
        with_gradmag: bool = False,
        density_encoder_type: str = "cnn",
        density_encoder_blocks: int = 1,
        density_grid_dim: int = 64,
        density_vit_patch: int = 8,
        density_vit_dim: int = 192,
        density_vit_depth: int = 6,
        density_vit_heads: int = 6,
        density_vit_mlp_ratio: int = 4,
        density_vit_dropout: float = 0.1,
        density_vit_patch_embed_mode: str = "fused",
        density_vit_channel_groups=None,
        density_vit_n_memory_tokens: int = 0,
        density_vit_n_in_channels: int = None,
        density_mask_ligand: bool = False,
        density_mask_threshold: float = 0.2,
        density_mask_dilate: int = 2,
        density_encoder_sees_ligand: bool = False,
        density_encoder_amp: bool = False,
        density_noise_alpha: float = 1.0,
        density_noise_sigma: float = 0.0,
        density_drop_ligand_tokens: bool = True,
        density_attenuate: bool = False,
        density_attenuate_sigma: float = 7.0,
        density_attenuate_quantile: float = 0.90,
        density_attenuate_strength: float = 1.0,
        density_attenuate_noise_sigma: float = 7.0,
        fusion: str = "default",
        density_cond_dropout: float = 0.0,
        density_proj_hidden: int = None,
        density_proj_kernel: int = 1,
        density_token_spatial_norm: bool = True,
        density_token_norm_gamma: float = 1.0,
        density_token_group_dim: int = 256,
        density_token_c: int = 64,
        density_token_kernel: int = 3,
        density_token_hidden: int = 64,
        density_token_offsets: bool = True,
        adapter_dim: int = 512,
        adapter_depth: int = 12,
        adapter_heads: int = 8,
        adapter_mlp_ratio: int = 4,
        adapter_n_in: int = 7,
        adapter_channel_groups=(4, 2, 1),
        adapter_patch: int = 8,
        adapter_grid_dim: int = 64,
        adapter_hidden: int = None,
        adapter_mask_basis: str = "protein_vdw",
        adapter_mask_thresh: float = 0.2,
        verbose: bool = False
    ):
        """
        VoxBind is a class that represents the VoxBind model.

        Args:
            n_channels_ligand (int): Number of channels for the ligand input. Defaults to N_LIGAND_ELEMENTS.
            n_channels_pocket (int): Number of channels for the pocket input. Defaults to N_POCKET_ELEMENTS.
            n_channels (int): Number of channels in the model. Defaults to 64.
            ch_mults (Union[Tuple[int, ...], List[int]]): Channel multipliers for each block in the UNet3D.
            Defaults to (1, 2, 2, 4).
            is_attn (Union[Tuple[bool, ...], List[int]]): Attention flag for each block in the UNet3D.
            Defaults to (False, False, True, True).
            n_blocks (int): Number of blocks in the UNet3D. Defaults to 2.
            n_groups (int): Number of groups in the ResidualBlock. Defaults to 32.
            dropout (float): Dropout rate. Defaults to 0.1.
            smooth_sigma (float): Sigma value for smoothing. Defaults to 0.0.
            with_density (bool): If True, add a density_encoder branch that accepts a
                single-channel Gaussian density map (e.g. an atom-blob or X-ray crop). The branch is
                summed with the ligand and pocket encodings. Checkpoint-compatible: old
                checkpoints load fine with with_density=False (default). Defaults to False.
            density_encoder_type (str): "cnn" (default; Conv3d + N×ResidualBlock, matches
                the 260521 density-MAE pretrained encoder) or "vit" (pure 3D ViT, matches
                the 260522 density-ViT-MAE pretrained encoder; see `models/density_vit.py`).
                Output shape is `(B, n_channels//2, G, G, G)` in both cases, so the
                downstream zero-init `density_proj` fusion is unchanged.
            density_encoder_blocks (int): cnn-only — ResidualBlocks in the density_encoder.
                Set to 4 to match the 260521 cnn-MAE pretrained encoder.
            density_grid_dim (int): vit-only — input grid dim (G). Must divide
                `density_vit_patch`. Defaults to 64 (matches voxbind default).
            density_vit_patch, density_vit_dim, density_vit_depth, density_vit_heads,
            density_vit_mlp_ratio, density_vit_dropout: vit-only hyperparameters.
            verbose (bool): Flag to print the number of parameters in the model. Defaults to False.
        """
        super().__init__()

        # cross_attn: the UNet's attention levels cross-attend to the frozen density ViT's
        # tokens, so it needs the token width. Any other fusion -> ctx_dim None -> no
        # DensityCrossAttn modules are built at all (unused params are a DDP hazard).
        _ctx_dim = int(density_vit_dim) if str(fusion) == "cross_attn" else None
        self.unet3d = UNet3D(
            n_channels // 2,
            n_channels,
            n_channels,
            ch_mults,
            is_attn,
            n_blocks,
            n_groups,
            dropout,
            smooth_sigma,
            ctx_dim=_ctx_dim,
            verbose=False
        )

        self.smooth_sigma = smooth_sigma
        self.n_channels_ligand = n_channels_ligand
        self.n_channels_pocket = n_channels_pocket
        self.with_density = with_density
        # with_gradmag appends a ‖∇ρ‖ channel to the density input, widening the
        # density-branch encoder from 1→2 channels (density + gradmag). Matches a
        # 2-ch (input_mode=density, with_gradmag) density-ViT-MAE pretrained encoder.
        self.with_gradmag = with_gradmag
        # Leak-removal ablation (full-voxel Path B only): blank the density + gradmag
        # channels inside the CLEAN ligand's footprint so the frozen encoder cannot
        # read the co-crystal ligand's own electron density (the crop is holo). The
        # mask region is derived from the ligand coordinates (occupancy of the clean
        # ligand voxel grid), dilated by `density_mask_dilate` voxels to cover the
        # x-ray envelope that spills past the atom cores.
        self.density_mask_ligand = bool(density_mask_ligand)
        self.density_mask_threshold = float(density_mask_threshold)
        self.density_mask_dilate = int(density_mask_dilate)
        # Feed the REAL ligand's atoms to the frozen density encoder (train: the clean
        # voxels; sampling: the reference ligand) instead of zeros. Note this is separate
        # from the density itself — the holo pocket/map is intrinsic to the CrossDocked SBDD
        # setting that every baseline shares, whereas these 7 channels are the target
        # molecule fed in directly. Judge the consequence by diversity / novelty-vs-reference
        # of the samples, not by assumption. Default off.
        self.density_encoder_sees_ligand = bool(density_encoder_sees_ligand)
        self.density_cond_dropout = float(density_cond_dropout)
        # Global density->noise blend (see _blend_density_noise). 1.0 = exact no-op, so
        # every existing config/checkpoint is unaffected. Test-time dose-response knob.
        self.density_noise_alpha = float(density_noise_alpha)
        self.density_noise_sigma = float(density_noise_sigma)
        # Protein-only attenuation field alpha(x) (see models/mae_ops.protein_attenuation_field).
        # Built from pocket atoms alone -> no ligand needed, so the training condition and the
        # generation-time condition are identical. Default off: existing configs unaffected.
        # cross_attn: skip the always-zero ligand tokens as cross-attention keys (33% of them
        # for groups [7,4,2]). Pure waste otherwise -- they are a constant.
        self.density_drop_ligand_tokens = bool(density_drop_ligand_tokens)
        self.density_attenuate = bool(density_attenuate)
        self.density_attenuate_sigma = float(density_attenuate_sigma)
        self.density_attenuate_quantile = float(density_attenuate_quantile)
        self.density_attenuate_strength = float(density_attenuate_strength)
        self.density_attenuate_noise_sigma = float(density_attenuate_noise_sigma)
        # Fusion variant. "default": ligand_encoder + pocket_encoder, with the frozen density
        # branch added via the zero-init density_proj. "v3": the frozen encoder ENTIRELY replaces
        # pocket_encoder — pocket + (apo) density are carried by the frozen ViT and fused into the
        # ligand features via a normal-init context_proj (early fusion). Ligand stream unchanged.
        # "protein_first": as "default", but the density correction is fused into the POCKET
        # representation alone and only then added to the ligand features, so density_proj never
        # sees ligand features. Same modules/shapes/zero-init as "default".
        self.fusion = str(fusion)
        # Validate explicitly: forward() dispatches with an `else` fallback, so an unknown value
        # (a typo) would otherwise run "default" silently and quietly train the wrong experiment.
        _valid_fusions = ("default", "v3", "protein_first", "cross_attn", "v4")
        if self.fusion not in _valid_fusions:
            raise ValueError(
                f"fusion must be one of {_valid_fusions}, got {fusion!r}"
            )

        self.ligand_encoder = ResidualBlock(
            n_channels_ligand, n_channels // 2, n_groups=0, dropout=0
        )
        # v3 removes pocket_encoder entirely (the frozen encoder carries the pocket). Don't even
        # build it — otherwise its params get no gradient and DDP errors on unused parameters.
        if self.fusion != "v3":
            self.pocket_encoder = ResidualBlock(
                n_channels_pocket, n_channels // 2, n_groups=0, dropout=0
            )
        if with_density:
            # density-branch input channels: density (+ gradmag when enabled).
            #   default              : 1 (density) or 2 (density+gradmag)  → encodes the FIELD only.
            #   density_vit_n_in_channels=13 (Path B): the FULL pretrained voxel encoder
            #     [ n_lig lig-atom, n_poc poc-atom, density, gradmag ] is reused frozen; the
            #     ligand-atom channels are masked (zeros) at runtime since the ligand is the
            #     generation target and is unavailable when conditioning. Lets a combined
            #     atomblob_density_gradmag ViT-MAE (e.g. exps/260616_best) load drop-in.
            n_dens_in = (int(density_vit_n_in_channels)
                         if density_vit_n_in_channels is not None
                         else (2 if with_gradmag else 1))
            self.density_n_in = n_dens_in
            # Path B: full-voxel conditioning with the ligand masked.
            self.density_full_voxel = (n_dens_in == n_channels_ligand + n_channels_pocket + 2)
            # COORDS-ONLY encoder (model_zoo C_v2: input_mode=atomblob, with_gradmag=false,
            # n_in=11, groups [7,4]). Same atom-blob voxel layout as full-voxel minus the
            # rho / ||grad rho|| channels, so it is the matched control that isolates what the
            # DENSITY channels contribute while holding the fusion path and the frozen
            # encoder's capacity fixed. Without this branch `_density_encoder_input` returns
            # the bare 1-channel map and the channel-group patch embed dies with
            # "split_with_sizes expects split_sizes to sum exactly to 1 ... got [7, 4]".
            self.density_coords_only = (n_dens_in == n_channels_ligand + n_channels_pocket)
            # Both build the [lig, poc, ...] atom-blob stack, so everything keyed off "the
            # frozen encoder consumes atom-blob voxels" must test this, not full_voxel alone.
            self.density_atomblob_in = self.density_full_voxel or self.density_coords_only
            if self.density_coords_only and density_encoder_type != "vit":
                raise ValueError(
                    "density_vit_n_in_channels=n_lig+n_poc (coords-only conditioning) "
                    f"requires density_encoder_type='vit', got {density_encoder_type!r}"
                )
            if self.density_full_voxel and density_encoder_type != "vit":
                raise ValueError(
                    "density_vit_n_in_channels=13 (full-voxel conditioning) requires "
                    f"density_encoder_type='vit', got {density_encoder_type!r}"
                )
            if density_encoder_type == "vit":
                # Pure 3D ViT — patch_embed (Conv3d k=p s=p) → L pre-LN MHSA+MLP
                # → Linear(D → C/2·p³) + pixel-shuffle3D back to full resolution.
                # Output shape (B, C/2, G, G, G) matches the CNN branch so the
                # `density_proj` fusion below is unchanged. State_dict layout
                # matches DensityViTMAE.encoder.* so the 260522 pretrained
                # encoder loads drop-in via `density_pretrained_path` (n_in_channels
                # must match the pretrained patch_embed: 1 = density, 2 = +gradmag).
                self.density_encoder = DensityViT(
                    grid_dim=density_grid_dim,
                    patch_size=density_vit_patch,
                    n_in_channels=n_dens_in,
                    c_out=n_channels // 2,
                    dim=density_vit_dim,
                    depth=density_vit_depth,
                    n_heads=density_vit_heads,
                    mlp_ratio=density_vit_mlp_ratio,
                    dropout=density_vit_dropout,
                    patch_embed_mode=density_vit_patch_embed_mode,
                    channel_groups=density_vit_channel_groups,
                    n_memory_tokens=density_vit_n_memory_tokens,
                )
            elif density_encoder_type == "cnn":
                # Lift density (+gradmag) to C/2, then N ResidualBlocks with GroupNorm.
                # GroupNorm can't be applied to 1 input channel, so we use a plain conv first.
                # N controlled by `density_encoder_blocks` so the 260521 pretrained encoder
                # (N=4) loads drop-in via `density_pretrained_path` in create_model.
                _dens_groups = min(n_groups, n_channels // 2) if n_groups > 0 else 0
                _enc_layers = [
                    torch.nn.Conv3d(n_dens_in, n_channels // 2, kernel_size=3, padding=1),
                    torch.nn.SiLU(),
                ]
                for _ in range(density_encoder_blocks):
                    _enc_layers.append(
                        ResidualBlock(n_channels // 2, n_channels // 2,
                                      n_groups=_dens_groups, dropout=dropout)
                    )
                self.density_encoder = torch.nn.Sequential(*_enc_layers)
            else:
                raise ValueError(
                    f"density_encoder_type must be 'cnn' or 'vit', got {density_encoder_type!r}"
                )
            self.density_encoder_type = density_encoder_type
            # bf16 autocast for the FROZEN encoder's forward only (the denoiser itself stays
            # fp32, as in the paper's train.py). The pretrained encoders were themselves
            # trained under bf16 autocast (train_density.py `amp.enabled: true`), so this
            # restores their pretraining precision rather than departing from it. Frozen +
            # no gradient path ⇒ trained weights are unaffected beyond the conditioning
            # signal's own rounding. Default off so existing configs reproduce exactly.
            self.density_encoder_amp = bool(density_encoder_amp)
            # Projects cat([ligand+pocket, density], dim=1) → C/2.
            # Zero-init so initial output is 0: x = x_backbone + 0 = x_backbone.
            # Training grows the density correction from zero, keeping early
            # iterations stable and identical to the baseline. Identical for
            # cnn and vit branches since both output (B, C/2, G, G, G).
            # v3 uses context_proj, v4 uses token_proj — don't build density_proj for either
            # (unused params are a DDP error).
            if self.fusion not in ("v3", "cross_attn", "v4"):
                proj_hidden = (int(density_proj_hidden)
                               if density_proj_hidden is not None else None)
                proj_kernel = int(density_proj_kernel)
                if proj_kernel not in (1, 3):
                    raise ValueError(
                        f"density_proj_kernel must be 1 or 3, got {proj_kernel}"
                    )
                if proj_hidden is None and proj_kernel == 1:
                    # Historical path: preserve the exact one-voxel 1x1 projection
                    # and checkpoint layout for existing experiments.
                    self.density_proj = torch.nn.Conv3d(
                        n_channels, n_channels // 2, kernel_size=1
                    )
                    torch.nn.init.zeros_(self.density_proj.weight)
                    torch.nn.init.zeros_(self.density_proj.bias)
                else:
                    # Spatial per-voxel MLP: the first layer can see a local 3x3x3
                    # neighborhood, while the final layer remains zero-initialized.
                    # Thus step 0 is still exactly the density-free VoxBind model.
                    proj_hidden = proj_hidden or n_channels
                    self.density_proj = torch.nn.Sequential(
                        torch.nn.Conv3d(
                            n_channels, proj_hidden,
                            kernel_size=proj_kernel, padding=proj_kernel // 2,
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv3d(proj_hidden, n_channels // 2, kernel_size=1),
                    )
                    torch.nn.init.zeros_(self.density_proj[-1].weight)
                    torch.nn.init.zeros_(self.density_proj[-1].bias)

        # v3 fusion: the frozen encoder replaces pocket_encoder entirely. context_proj is a
        # NORMAL-init ResidualBlock (in==out → identity skip), so the frozen pocket+density
        # context is present from step 0 — unlike the zero-init density_proj, which would leave
        # the model blind to the pocket early once pocket_encoder is gone. Requires the full-voxel
        # (n_in=13) encoder so the pocket channels are actually inside the frozen input.
        if self.fusion == "v3":
            if not (self.with_density and getattr(self, "density_full_voxel", False)):
                raise ValueError(
                    "fusion='v3' requires with_density=True and full-voxel density encoding "
                    "(density_vit.n_in_channels=13) so the frozen encoder carries the pocket."
                )
            self.context_proj = ResidualBlock(
                n_channels // 2, n_channels // 2, n_groups=0, dropout=0
            )

        # v4 fusion: merge the frozen encoder's TOKENS, not its unpatchified voxel output.
        # The voxel path (default/protein_first/v3) loses two things on the way out of the ViT:
        #   * `_pool_groups` MEAN-POOLS the channel-group tokens (nG → 1) before decoding, so
        #     with groups [7,4,1,1] the density and gradmag tokens arrive diluted to 1/4 each.
        #     Here the groups are kept and concatenated on the channel axis.
        #   * `decoder_proj` is a FROZEN Linear(D → c_out·p³) that was trained as an intermediate
        #     of the MAE reconstruction head, so the within-patch readout is a fixed basis tuned
        #     for reproducing the map — not for conditioning a denoiser. Here every voxel simply
        #     receives its own patch's token and a TRAINABLE per-voxel MLP does the readout,
        #     jointly with the local ligand+pocket features it is being fused into.
        # This is the representation the affinity probe is scored on (`forward_features` stops
        # before both of the above), so what the generator sees is what was measured.
        # Channel reduction happens on the g_p³ token grid, BEFORE the broadcast: with a 1×1×1
        # first layer that is mathematically identical to broadcasting the full nG·D width, and
        # ~500x cheaper (g_p³ = 512 positions instead of G³ = 262,144).
        # NOT fixed by this: the tokens themselves are patch-resolution (p·voxel_size), so no
        # sub-patch density detail exists to recover — that is the encoder's granularity.
        if self.fusion == "v4":
            if not (self.with_density and density_encoder_type == "vit"):
                raise ValueError(
                    "fusion='v4' requires with_density=True and density_encoder_type='vit' "
                    "(it consumes the encoder's patch tokens directly)."
                )
            n_tok_groups = 1
            if density_vit_patch_embed_mode == "channel_group" and density_vit_channel_groups:
                n_tok_groups = len(density_vit_channel_groups)
                if self.density_drop_ligand_tokens and self.density_atomblob_in:
                    n_tok_groups -= 1          # the ligand group is all zeros -> constant keys
            self.density_token_groups = int(n_tok_groups)
            self.density_token_c = int(density_token_c)
            self.density_token_offsets = bool(density_token_offsets)
            # Token trunk, on the g_p³ grid where capacity is ~500x cheaper than at G³:
            #   (1) GROUPED 1x1x1: each channel group is compressed on its own, D -> t_g. The
            #       groups are distinct modalities (pocket atoms / rho / grad-rho, once the
            #       always-zero ligand group is dropped) and they sit contiguously on the
            #       channel axis, so `groups=nG` keeps them apart until mixing actually buys
            #       something -- and costs nG x fewer parameters than a dense layer of the
            #       same output width.
            #   (2) DENSE k_t: mixes the groups with each other and, at k_t=3, each patch with
            #       its 26 neighbours (a 6 A receptive field at 8³ x 2 A). This is the only
            #       place the density branch gets ANY spatial mixing -- and it is BETWEEN
            #       patches; sub-patch detail does not exist in the tokens to recover.
            # Spatial normalisation of the frozen representation, over the PATCH axis
            # (iREPA, ICLR 2026): z-score each feature channel across the g_p³ patches of a
            # sample, which deliberately discards the component shared by every patch and
            # keeps only how patches DIFFER from each other. Two reasons to want it here:
            #   * post-LayerNorm ViT tokens are mu + delta with ||mu|| >> ||delta||, so without
            #     centering the informative part is a small perturbation on a large constant.
            #     This repo already hit exactly that in the U-REPA manifold loss, where a
            #     missing centering made the projector converge to a uniform target.
            #   * iREPA measures spatial-structure metrics correlating |r| ~ 0.85-0.89 with
            #     generation quality while linear-probe accuracy correlates |r| = 0.26 --
            #     and mixing global information INTO the patches raised probe accuracy while
            #     making generation worse. The affinity probe used to pick encoders here is a
            #     mean-pooled (i.e. global) quantity, so the same caveat applies.
            # Channels are group-major, so per-channel statistics are automatically per-group.
            # No learnable affine: the grouped conv immediately after can absorb one.
            self.density_token_spatial_norm = bool(density_token_spatial_norm)
            self.density_token_norm_gamma = float(density_token_norm_gamma)
            _tok_w = self.density_token_groups * int(density_vit_dim)
            _tg = int(density_token_group_dim or 0)
            _kt = int(density_token_kernel)
            if _kt not in (1, 3):
                raise ValueError(f"density_token_kernel must be 1 or 3, got {_kt}")
            _trunk = []
            if _tg > 0:
                _trunk += [torch.nn.Conv3d(_tok_w, self.density_token_groups * _tg,
                                           kernel_size=1, groups=self.density_token_groups),
                           torch.nn.SiLU()]
                _tok_w = self.density_token_groups * _tg
            _trunk += [torch.nn.Conv3d(_tok_w, self.density_token_c,
                                       kernel_size=_kt, padding=_kt // 2),
                       torch.nn.SiLU()]
            self.token_trunk = torch.nn.Sequential(*_trunk)
            if self.density_token_offsets:
                self.register_buffer(
                    "_patch_offsets",
                    _intra_patch_offsets(int(density_grid_dim), int(density_vit_patch)),
                    persistent=False,
                )
            _tok_in = (n_channels // 2 + self.density_token_c
                       + (3 if self.density_token_offsets else 0))
            self.token_proj = torch.nn.Sequential(
                torch.nn.Conv3d(_tok_in, int(density_token_hidden), kernel_size=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(int(density_token_hidden), n_channels // 2, kernel_size=1),
            )
            # Zero-init the last layer, exactly as density_proj: step 0 is the density-free model.
            torch.nn.init.zeros_(self.token_proj[-1].weight)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        # fusion='adapter': transfer the FROZEN two-tower pocket encoder into a FROZEN VoxBind and
        # inject its spatial pocket features through a trainable residual PocketAdapter. Keeps the
        # original pocket_encoder (pocket coord channels preserved); does NOT touch the with_density
        # branch. The frozen encoder input is [pocket atoms, M_P⊙ρ, ‖∇ρ‖, M_P] (the pocket-tower
        # layout); create_model loads its pretrained trunk and freezes everything but the adapter.
        if self.fusion == "adapter":
            self.adapter_encoder = DensityViT(
                grid_dim=int(adapter_grid_dim),
                patch_size=int(adapter_patch),
                n_in_channels=int(adapter_n_in),
                c_out=n_channels // 2,             # unused (we call forward_features), any value
                dim=int(adapter_dim),
                depth=int(adapter_depth),
                n_heads=int(adapter_heads),
                mlp_ratio=int(adapter_mlp_ratio),
                dropout=0.0,
                patch_embed_mode=("channel_group" if adapter_channel_groups else "fused"),
                channel_groups=(tuple(int(c) for c in adapter_channel_groups)
                                if adapter_channel_groups else None),
            )
            self.pocket_adapter = PocketAdapter(
                dim=int(adapter_dim),
                g_p=int(adapter_grid_dim) // int(adapter_patch),
                patch_size=int(adapter_patch),
                c_out=n_channels // 2,
                hidden=adapter_hidden,
            )
            self.adapter_mask_basis = str(adapter_mask_basis)
            self.adapter_mask_thresh = float(adapter_mask_thresh)

        self.final_ligand = torch.nn.Conv3d(
            n_channels, n_channels_ligand, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )

        if verbose:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f">> model has {(n_params/1e6):.02f}M parameters")

    @staticmethod
    def ligand_occupancy_mask(ligand_clean: torch.Tensor,
                              threshold: float = 0.2,
                              dilate: int = 2) -> torch.Tensor:
        """Binary occupancy mask (B, 1, G, G, G) of the CLEAN ligand's footprint.

        Sum the ligand atom-type channels, threshold, then optionally dilate by
        `dilate` voxels (max-pool) so the mask covers the x-ray density envelope that
        spills a little beyond the atom-blob cores. Derived from the clean ligand voxel
        grid (i.e. from the ligand coordinates), NOT the noisy input — the caller must
        pass the ground-truth/reference ligand.
        """
        occ = ligand_clean.sum(dim=1, keepdim=True)            # (B, 1, G, G, G)
        m = (occ > threshold).float()
        if dilate and dilate > 0:
            k = 2 * dilate + 1
            m = torch.nn.functional.max_pool3d(m, kernel_size=k, stride=1, padding=dilate)
        return m

    def _pocket_adapter_input(self, pocket: torch.Tensor,
                              density: torch.Tensor) -> torch.Tensor:
        """Build the 7-channel input for the FROZEN pocket encoder, matching the two-tower
        pocket-tower layout EXACTLY: [ pocket atoms (n_poc), M_P⊙ρ, ‖∇(M_P⊙ρ)‖, M_P ].

        M_P is the region keep-mask for the pocket tower (protein_vdw ⇒ M_P = pocket-atom vdW
        occupancy > thresh). Masking ρ by M_P removes non-protein density — including any
        co-crystal ligand density in a holo training crop (no ED leak), and it is a no-op at
        generation time where no ligand is present. Channel order = [atoms, density, gradmag,
        mask], the same trailing-mask layout the tower was pretrained with.
        """
        # pocket occupancy; ligand not needed for protein_vdw / none bases (lig_occ ⇒ zeros).
        poc_occ = pocket.sum(dim=1, keepdim=True)
        lig_occ = torch.zeros_like(poc_occ)
        keep_p, _ = region_keep_masks(lig_occ, poc_occ,
                                      self.adapter_mask_basis, self.adapter_mask_thresh)
        d_masked = density * keep_p
        gradmag = per_sample_zscore(gradient_magnitude3d(d_masked))
        return torch.cat([pocket, d_masked, gradmag, keep_p.to(density.dtype)], dim=1)

    def _blend_density_noise(self, density: torch.Tensor) -> torch.Tensor:
        """d = a*d + (1-a)*noise with a = `density_noise_alpha` (global scalar).

        a=1 returns `density` untouched (identity, not a numerically-equal copy), so every
        existing config and checkpoint reproduces bit-for-bit. Intended as a TEST-TIME
        dose-response knob: sweeping a from 1 -> 0 measures how much the trained model
        relies on the density channel at all, with a=0 the random-density lower bound.
        """
        a = float(getattr(self, "density_noise_alpha", 1.0))
        if density is None or a >= 1.0:
            return density
        noise = torch.randn_like(density)
        sig = float(getattr(self, "density_noise_sigma", 0.0))
        if sig > 0:
            # correlate the noise to the map's resolution, then restore unit variance so
            # the blend keeps the z-scored scale the encoder was pretrained on.
            noise = gaussian_blur3d(noise, sig)
            noise = noise / noise.std().clamp(min=1e-6)
        return a * density + (1.0 - a) * noise

    def _density_encoder_input(self, pocket: torch.Tensor,
                               density: torch.Tensor,
                               dens_mask: torch.Tensor = None,
                               enc_ligand: torch.Tensor = None) -> torch.Tensor:
        """Build the tensor fed to the (frozen) density_encoder.

        Default field mode: the encoder is a density-only / density+gradmag ViT-MAE,
        so the input IS the density field (passed straight through; the caller already
        supplies 1 or 2 channels).

        Full-voxel mode (Path B, n_in=13): the encoder is a combined
        atomblob_density_gradmag ViT-MAE that was pretrained on the full voxel
        [ lig-atom, poc-atom, density, gradmag ]. We reconstruct that channel layout
        here — pocket atoms + density + on-the-fly ‖∇ρ‖ (z-scored, exactly as in
        pretraining), with the LIGAND-atom channels zeroed (masked): the ligand is the
        generation target and is not available to condition on. The frozen MAE encoder
        was trained with masked inputs, so an all-zero ligand block is in-distribution.

        Leak-removal (density_mask_ligand): when `dens_mask` (B,1,G,G,G, the clean
        ligand footprint) is supplied, the density AND gradmag are additionally zeroed
        inside that region so the encoder cannot read the co-crystal ligand's own
        electron density. gradmag is computed from the UNMASKED density first (to avoid
        a spurious sharp edge at the mask boundary) and then blanked in the region.

        Global noise blend (density_noise_alpha): d = a*d + (1-a)*noise, a GLOBAL scalar
        (no spatial structure — that is the separate soft alpha-field idea). a=1 is an
        exact no-op; a=0 replaces the map with pure noise, i.e. the "random density"
        lower-bound condition. Applied BEFORE gradmag is derived so channel 12 is the
        gradient of the map the encoder actually sees — deriving it first would leave the
        real density's edges in gradmag and defeat the whole point.
        `density_noise_sigma`>0 blurs the noise to that voxel scale first (1 vox = 0.25 A)
        so its gradient statistics resemble real density; 0 = literal white randn, which
        makes gradmag strongly OOD and conflates "density removed" with "gradmag broken".
        """
        density = self._blend_density_noise(density)
        # Protein-only attenuation: alpha(x) from the POCKET channels we already receive,
        # so it needs no ligand and is computable at generation time. Applied here, BEFORE
        # gradmag, so channel 12 is the gradient of the attenuated map — deriving gradmag
        # first would leave the real density's edges in it and defeat the masking (the
        # density x gradmag ablation found the signal lives in the EDGES).
        if self.density_attenuate and density is not None:
            alpha = protein_attenuation_field(
                pocket.sum(dim=1, keepdim=True),
                sigma_vox=self.density_attenuate_sigma,
                quantile=self.density_attenuate_quantile,
                strength=self.density_attenuate_strength,
            )
            density = attenuate_density(density, alpha, self.density_attenuate_noise_sigma)
        if not self.density_atomblob_in:
            return density
        # density: (B, 1, G, G, G) — the v5 (arcsinh+z) pocket density crop.
        B, _, G, _, _ = density.shape
        # Ligand channels. Default: ZEROED — the ligand is the generation target and is not
        # available to condition on, and the MAE encoder saw masked inputs in pretraining.
        # `density_encoder_sees_ligand=true` instead feeds the available REFERENCE ligand (train: clean
        # voxels; sampling: the reference ligand). Whether that collapses generation into
        # reconstruction is measurable (sample diversity, novelty vs the reference), so report
        # those with the Vina numbers rather than labelling the run up front.
        if self.density_encoder_sees_ligand and enc_ligand is not None:
            lig0 = enc_ligand.to(density.dtype)
        else:
            lig0 = density.new_zeros(B, self.n_channels_ligand, G, G, G)
        # Coords-only encoder: it was pretrained on [lig, poc] alone, so the density and
        # gradmag channels must NOT be appended — the experimental map reaches this arm
        # only through what the pocket atoms already encode, which is exactly the point.
        if self.density_coords_only:
            return torch.cat([lig0, pocket], dim=1)                    # (B, n_lig+n_poc, G,G,G)
        gradmag = per_sample_zscore(gradient_magnitude3d(density))     # (B, 1, G, G, G)
        if self.density_mask_ligand and dens_mask is not None:
            keep = 1.0 - dens_mask                                     # 0 inside ligand region
            density = density * keep
            gradmag = gradmag * keep
        # channel order MUST match pretraining: [ lig, poc, density, gradmag ]
        return torch.cat([lig0, pocket, density, gradmag], dim=1)

    def _encode_density(self, enc_in: torch.Tensor, return_tokens: bool = False,
                        tokens_only: bool = False):
        """Run the density encoder, optionally in bf16 (see `density_encoder_amp`).

        Only the encoder forward is autocast — `_density_encoder_input` (gradmag +
        z-score) stays fp32, and the result is cast back to fp32 so the downstream
        fusion conv and the whole denoiser see exactly the dtype they always did.
        """
        def _run(z):
            # tokens_only: `forward_features` stops before _pool_groups and decoder_proj, so
            # neither the group mean-pool nor the frozen MAE voxel decode is computed at all.
            if tokens_only:
                return self.density_encoder.forward_features(z)
            return self.density_encoder(z, return_tokens=return_tokens)

        if not (self.density_encoder_amp and enc_in.is_cuda):
            return _run(enc_in)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _run(enc_in)
        if return_tokens and not tokens_only:
            vox, tok = out
            return vox.float(), tok.float()
        return out.float()

    def _density_token_grid(self, enc_in: "torch.Tensor") -> "torch.Tensor":
        """v4: frozen-encoder patch tokens → (B, density_token_c, G, G, G).

        Every voxel receives the (channel-reduced) token of the patch it belongs to. The
        whole trunk runs on the g_p³ token grid and only then broadcasts -- 512 positions
        instead of G³ = 262,144, so capacity there is ~500x cheaper than capacity after the
        broadcast, which is why the trunk is wide and the per-voxel MLP is thin.
        """
        tok = self._encode_density(enc_in, tokens_only=True)          # (B, nG·N, D)
        g_p = self.density_encoder.g_p
        n_patch = g_p ** 3
        if (self.density_drop_ligand_tokens and self.density_atomblob_in
                and tok.shape[1] > n_patch):
            tok = tok[:, n_patch:]            # groups are token-major, ligand group first
        if tok.shape[1] != self.density_token_groups * n_patch:
            raise ValueError(
                f"v4: expected {self.density_token_groups} token group(s) x {n_patch} patches, "
                f"got {tok.shape[1]} tokens — check density_vit.channel_groups."
            )
        B, D = tok.shape[0], tok.shape[2]
        nG = self.density_token_groups
        # (B, nG·N, D) → (B, nG, N, D) → (B, nG, D, N) → (B, nG·D, g_p, g_p, g_p).
        # N is laid out to match _tokens_to_voxels' reshape, so the last reshape puts each
        # token at its own patch cell.
        tok = (tok.reshape(B, nG, n_patch, D)
                  .permute(0, 1, 3, 2)
                  .reshape(B, nG * D, g_p, g_p, g_p))
        if self.density_token_spatial_norm:
            # z-score each channel across the g_p³ patch positions of its own sample
            mu = tok.mean(dim=(2, 3, 4), keepdim=True)
            var = tok.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
            tok = (tok - self.density_token_norm_gamma * mu) / torch.sqrt(var + 1e-6)
        tok = self.token_trunk(tok)                                   # (B, c, g_p³)
        p = self.density_encoder.patch_size
        return (tok.repeat_interleave(p, dim=2)
                   .repeat_interleave(p, dim=3)
                   .repeat_interleave(p, dim=4))                      # (B, c, G, G, G)

    def _drop_density_residual(self, delta: "torch.Tensor") -> "torch.Tensor":
        """Zero the density residual on a random subset of the batch, training only.

        The frozen encoder gives the density branch no regularization of its own: it is
        held in eval() and its train() is overridden, so `density_vit.dropout` never
        fires (models/__init__.py). Without this the model sees the density on every
        single example and has no reason not to lean on it.

        The residual is masked rather than `x_dens`, because zeroing the encoder output
        does not produce a density-free example: density_proj still contributes its bias
        and the x half of its input. Killing the residual leaves `x` untouched, which is
        exactly the density-free baseline the zero-init was chosen to reproduce -- so a
        dropped sample trains the same path the model must fall back to.

        That path is also what makes classifier-free guidance possible at sampling time:
        with a density-free branch the model has actually trained on, predictions can be
        extrapolated as pred_free + w * (pred_dens - pred_free) to dial the condition
        up or down without retraining.

        density_cond_dropout=0.0 (default) returns delta untouched, so existing runs and
        every checkpoint trained before this are bit-for-bit unaffected.
        """
        if not self.training or self.density_cond_dropout <= 0:
            return delta
        keep = torch.rand(delta.shape[0], device=delta.device) >= self.density_cond_dropout
        return delta * keep.view(-1, *([1] * (delta.dim() - 1))).to(delta.dtype)

    def forward(
        self,
        ligand: torch.Tensor,
        pocket: torch.Tensor,
        density: torch.Tensor = None,
        dens_mask: torch.Tensor = None,
        enc_ligand: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the VoxBind model.

        Args:
            ligand (torch.Tensor): Input tensor for the ligand. Shape (B, 7, G, G, G).
            pocket (torch.Tensor): Input tensor for the pocket. Shape (B, 4, G, G, G).
            density (torch.Tensor, optional): density map (atom-blob or X-ray crop),
                shape (B, 1, G, G, G), or (B, 2, G, G, G) = [density, gradmag] when
                with_gradmag=True. Only used when with_density=True.
                Defaults to None (standard baseline behaviour).

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        if self.fusion == "adapter":
            # Frozen VoxBind baseline (original ligand + pocket coord channels), then a trainable
            # residual from the FROZEN pretrained pocket encoder via the PocketAdapter.
            if density is None:
                raise ValueError("fusion='adapter' requires the pocket density (density=...).")
            x = self.ligand_encoder(ligand) + self.pocket_encoder(pocket)
            enc_in = self._pocket_adapter_input(pocket, density)
            tokens = self.adapter_encoder.forward_features(enc_in)          # (B, nG·N, D) frozen
            pooled = self.adapter_encoder._pool_groups(tokens)             # (B, N, D)
            x = x + self.pocket_adapter(pooled)                            # zero-init residual
        elif self.fusion == "v3":
            # frozen encoder carries pocket + (apo) density; pocket_encoder is gone. Ligand
            # stream identical. Early fusion: add the projected frozen context to ligand features.
            enc_in = self._density_encoder_input(pocket, density, dens_mask=dens_mask,
                                                 enc_ligand=enc_ligand)
            x_dens = self._encode_density(enc_in)
            x = self.ligand_encoder(ligand) + self.context_proj(x_dens)
        elif self.fusion == "cross_attn":
            # Density enters INSIDE the denoiser, not at its input: the UNet's attention
            # levels query the frozen ViT's patch tokens. No density_proj -- the only path
            # is the (zero-init) cross-attention, so this isolates the fusion mechanism
            # against protein_first/default rather than stacking two pathways.
            x = self.ligand_encoder(ligand) + self.pocket_encoder(pocket)
            if self.with_density and density is not None:
                enc_in = self._density_encoder_input(pocket, density, dens_mask=dens_mask)
                _, dens_ctx = self._encode_density(enc_in, return_tokens=True)
                # Drop the LIGAND channel-group's tokens. In full-voxel mode the 7 ligand
                # channels are always zero (the ligand is the generation target), so with
                # channel_group tokenisation a full 1/nG of the keys carry no information --
                # attending to them is pure cost. Tokens are (B, nG*N, D) with the groups in
                # channel order [lig, poc, dens+gradmag], so the first N are the ligand's.
                if self.density_drop_ligand_tokens and self.density_atomblob_in:
                    n_per_group = self.density_encoder.g_p ** 3
                    if dens_ctx.shape[1] > n_per_group:
                        dens_ctx = dens_ctx[:, n_per_group:]
            else:
                dens_ctx = None
            x = self.unet3d(x, None, ctx=dens_ctx)
            x = self.unet3d.act(x)
            return self.final_ligand(x)
        elif self.fusion == "v4":
            # Token fusion at FULL resolution: each voxel is concatenated with its own patch's
            # token (channel-reduced) plus its position inside that patch, and a trainable
            # per-voxel MLP maps the result back to the feature width. Same residual shape and
            # same zero-init discipline as density_proj, so step 0 == density-free baseline.
            x = self.ligand_encoder(ligand) + self.pocket_encoder(pocket)
            if self.with_density and density is not None:
                enc_in = self._density_encoder_input(pocket, density, dens_mask=dens_mask,
                                                     enc_ligand=enc_ligand)
                tok = self._density_token_grid(enc_in)
                if tok.shape[-3:] != x.shape[-3:]:
                    raise ValueError(
                        f"v4: token grid {tuple(tok.shape[-3:])} != voxel grid "
                        f"{tuple(x.shape[-3:])}; density_grid_dim must equal the ligand grid."
                    )
                parts = [x, tok]
                if self.density_token_offsets:
                    parts.append(self._patch_offsets.to(x.dtype)
                                 .expand(x.shape[0], -1, -1, -1, -1))
                x = x + self._drop_density_residual(
                    self.token_proj(torch.cat(parts, dim=1)))
        elif self.fusion == "protein_first":
            # Density is fused into the PROTEIN representation on its own, and only the
            # result is added to the ligand features. density_proj therefore conditions on
            # pocket features alone — under "default" it also sees the (noisy) ligand, which
            # lets the density correction depend on the generation target. Modules, shapes
            # and the zero-init are identical to "default", so step 0 is still exactly the
            # density-free baseline.
            x_poc = self.pocket_encoder(pocket)
            if self.with_density and density is not None:
                enc_in = self._density_encoder_input(pocket, density, dens_mask=dens_mask,
                                                 enc_ligand=enc_ligand)
                x_dens = self._encode_density(enc_in)
                x_poc = x_poc + self._drop_density_residual(
                    self.density_proj(torch.cat([x_poc, x_dens], dim=1)))
            x = self.ligand_encoder(ligand) + x_poc
        else:
            x = self.ligand_encoder(ligand) + self.pocket_encoder(pocket)
            if self.with_density and density is not None:
                enc_in = self._density_encoder_input(pocket, density, dens_mask=dens_mask,
                                                 enc_ligand=enc_ligand)
                x_dens = self._encode_density(enc_in)
                x = x + self._drop_density_residual(
                    self.density_proj(torch.cat([x, x_dens], dim=1)))

        x = self.unet3d(x, None)
        x = self.unet3d.act(x)
        x = self.final_ligand(x)

        return x

    def score(self, y: torch.Tensor, pocket: torch.Tensor,
              density: torch.Tensor = None,
              dens_mask: torch.Tensor = None,
              enc_ligand: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the score function.

        Args:
            y (torch.Tensor): The y tensor.
            pocket (torch.Tensor): The pocket tensor.
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.
            dens_mask (torch.Tensor, optional): clean-ligand footprint (B,1,G,G,G) used to
                blank density+gradmag in the ligand region. Defaults to None.

        Returns:
            torch.Tensor: The calculated base score tensor.
        """
        xhat = self.forward(y, pocket, density=density, dens_mask=dens_mask,
                            enc_ligand=enc_ligand)
        return (xhat - y) / (self.smooth_sigma ** 2)

    ####################################################################################
    # conditional walk-jump sampling methods
    def initialize_y_v(
        self,
        vox_pockets: torch.Tensor,
        ligand_gt: torch.Tensor,
        smooth_sigma: float,
        chain_init: str = "denovo"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the y and v tensors for the walk-jump sampling.

        Args:
            vox_pockets (torch.Tensor): The vox_pockets tensor.
            ligand_gt (torch.Tensor): The ligand_gt tensor.
            smooth_sigma (float): The smooth_sigma value.
            chain_init (str, optional): The chain initialization method. Defaults to "denovo".

        Returns:
            torch.Tensor: The initialized y tensor.
            torch.Tensor: The initialized v tensor.
        """
        n_channels = N_LIGAND_ELEMENTS
        grid_dim = vox_pockets.shape[-1]
        n_chains = vox_pockets.shape[0]

        # gaussian noise
        y = torch.cuda.FloatTensor(n_chains, n_channels, grid_dim, grid_dim, grid_dim)
        y.normal_(0, smooth_sigma)

        if chain_init == "ligand":
            y += ligand_gt
        elif chain_init == "denovo":
            mask_pocket = get_pocket_mask(vox_pockets, n_channels)
            noise = torch.cuda.FloatTensor(y.shape).uniform_(0, 1)
            noise[mask_pocket] = 0
            y += noise

        return y, torch.zeros_like(y)

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor, pocket: torch.Tensor,
                      density: torch.Tensor = None,
                      dens_mask: torch.Tensor = None,
                      enc_ligand: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the jump step of the walk-jump sampling.

        Args:
            y (torch.Tensor): The y tensor.
            pocket (torch.Tensor): The pocket tensor.
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.
            dens_mask (torch.Tensor, optional): clean-ligand footprint (B,1,G,G,G). Defaults to None.
            enc_ligand (torch.Tensor, optional): reference-ligand grid handed to the frozen
                density encoder when density_encoder_sees_ligand=true. Ignored otherwise, so
                the density-free and non-reference paths are unaffected. Defaults to None.

        Returns:
            torch.Tensor: The estimated "clean" samples xhats.
        """
        return self.forward(y, pocket, density=density, dens_mask=dens_mask,
                            enc_ligand=(enc_ligand if self.density_encoder_sees_ligand
                                        else None))

    @torch.no_grad()
    def wjs_walk_steps(
        self,
        y: torch.Tensor,
        v: torch.Tensor,
        pocket: torch.Tensor,
        mask: torch.Tensor = None,
        n_steps: int = 100,
        friction: float = 1.,
        lipschitz: float = 1.,
        density: torch.Tensor = None,
        dens_mask: torch.Tensor = None,
        enc_ligand: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs `n_steps` walk steps of the walk-jump sampling.

        Args:
            y (torch.Tensor): The y tensor.
            v (torch.Tensor): The v tensor.
            pocket (torch.Tensor): The pocket tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
            n_steps (int, optional): The number of steps. Defaults to 100.
            friction (float, optional): The friction value. Defaults to 1..
            lipschitz (float, optional): The lipschitz value. Defaults to 1..
            density (torch.Tensor, optional): X-ray density map (B,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The updated y tensor.
            torch.Tensor: The updated v tensor.
        """
        gamma = friction
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(-gamma)  # gamma = "effective friction"
        zeta2 = math.exp(-2 * gamma)
        delta = self.smooth_sigma / 2

        for _ in range(n_steps):
            with torch.no_grad():
                y += delta * v / 2
            psi = self.score(y, pocket, density=density, dens_mask=dens_mask,
                             enc_ligand=(enc_ligand if self.density_encoder_sees_ligand
                                         else None))
            with torch.no_grad():
                noise = torch.randn_like(y)
                if mask is not None:
                    noise[mask] = 0.
                    psi[mask] = 0.
                v += u * delta * psi / 2
                v = zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * noise  # v_{t+1}
                y += delta * v / 2
        torch.cuda.empty_cache()
        return y, v

    def sample(
            self,
            pocket: torch.Tensor,
            ligand: torch.Tensor,
            warmup_wjs: int = 400,
            steps: int = 100,
            max_steps: int = 100,
            chain_init: str = "denovo",
            mask_pocket: bool = True,
            n_chains: int = 48,
            threshold: float = .2,
            density: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs the walk-jump sampling.

        Args:
            pocket (torch.Tensor): The pocket tensor.
            ligand (torch.Tensor): The ligand tensor.
            warmup_wjs (int, optional): The number of warmup wjs steps. Defaults to 0.
            steps (int, optional): The number of steps per iteration. Defaults to 100.
            max_steps (int, optional): The maximum number of steps. Defaults to 100.
            chain_init (str, optional): The chain initialization method. Defaults to "denovo".
            mask_pocket (bool, optional): Whether to mask the pocket. Defaults to True.
            threshold (float, optional): The threshold value. Defaults to .2.
            density (torch.Tensor, optional): X-ray density map (1,1,G,G,G). Defaults to None.

        Returns:
            torch.Tensor: The generated voxels.
        """
        self.eval()
        if ligand is None:
            chain_init = "denovo"

        # rotate pocket and ligand
        N = n_chains if n_chains < 10 else 10
        assert n_chains % N == 0, "n_chains must be divisible by N"
        pocket = pocket.repeat(N, 1, 1, 1, 1)
        ligand = ligand.repeat(N, 1, 1, 1, 1)
        if density is not None:
            density = density.repeat(N, 1, 1, 1, 1)
        rand_rots = [
            [random.choice([[2, 3], [3, 4], [2, 4], [3, 2], [4, 3], [4, 2], None]) for _ in range(N)],
            [random.randint(1, 4) for _ in range(N)]
        ]
        pocket = rotate_batch_voxel_grids(pocket, rand_rots)
        ligand = rotate_batch_voxel_grids(ligand, rand_rots)
        if density is not None:
            # density must follow the same per-chain rotation as the pocket,
            # else the conditioning is spatially misaligned during sampling.
            density = rotate_batch_voxel_grids(density, rand_rots)

        y, v = self.initialize_y_v(pocket, ligand, self.smooth_sigma, chain_init)

        # density-leak mask: blank density+gradmag inside the (rotated) reference-ligand
        # footprint. Derived from the clean `ligand` grid, which is rotated/repeated in
        # lockstep with `density`, so it stays spatially aligned. None → no masking.
        def _dmask(lig_grid):
            if not (self.density_mask_ligand and density is not None and lig_grid is not None):
                return None
            return self.ligand_occupancy_mask(
                lig_grid, self.density_mask_threshold, self.density_mask_dilate)

        # warm up
        if warmup_wjs > 0:
            mask_warmup = get_pocket_mask(pocket, n_channels=N_LIGAND_ELEMENTS)
            y, v = self.wjs_walk_steps(y, v, pocket, mask_warmup, warmup_wjs,
                                       density=density, dens_mask=_dmask(ligand),
                                       enc_ligand=ligand)
        y, v = y.repeat(n_chains // N, 1, 1, 1, 1), v.repeat(n_chains // N, 1, 1, 1, 1)
        pocket, ligand = pocket.repeat(n_chains // N, 1, 1, 1, 1), ligand.repeat(n_chains // N, 1, 1, 1, 1)
        if density is not None:
            density = density.repeat(n_chains // N, 1, 1, 1, 1)
        rand_rots[0] = rand_rots[0] * (n_chains // N)
        rand_rots[1] = rand_rots[1] * (n_chains // N)

        # mask
        mask = None
        if mask_pocket:
            mask = get_pocket_mask(pocket, n_channels=N_LIGAND_ELEMENTS)

        # density-leak mask at the full n_chains scale (ligand now repeated to n_chains)
        dens_mask_main = _dmask(ligand)

        # sample
        voxels = []
        for _ in range(0, max_steps, steps):
            # walk `steps` steps
            y, v = self.wjs_walk_steps(y, v, pocket, mask, steps, density=density,
                                       dens_mask=dens_mask_main, enc_ligand=ligand)

            # jump step
            xhats = self.wjs_jump_step(y, pocket, density=density, dens_mask=dens_mask_main,
                                       enc_ligand=ligand)

            nz = (xhats > 0).float().mean().item()
            # quantile() rejects tensors > ~16M elements; subsample for stats.
            flat = xhats.flatten()
            if flat.numel() > 1_000_000:
                idx = torch.randint(flat.numel(), (1_000_000,), device=flat.device)
                stat_sample = flat[idx]
            else:
                stat_sample = flat
            print(
                f"[sample] xhats stats before threshold={threshold:.2f}: "
                f"min={xhats.min():.4f}  max={xhats.max():.4f}  "
                f"mean={xhats.mean():.4f}  "
                f"p25={stat_sample.quantile(0.25):.4f}  p50={stat_sample.quantile(0.50):.4f}  "
                f"p75={stat_sample.quantile(0.75):.4f}  p90={stat_sample.quantile(0.90):.4f}  "
                f"p95={stat_sample.quantile(0.95):.4f}  p99={stat_sample.quantile(0.99):.4f}  "
                f"frac>0={nz:.4f}  "
                f"frac>{threshold:.2f}={(xhats > threshold).float().mean().item():.4f}"
            )
            xhats[xhats < threshold] = 0
            xhats = unrotate_voxel_grids(xhats, rand_rots)
            voxels.append(xhats)

        voxels = torch.concat(voxels, axis=0)

        return voxels


def rotate_batch_voxel_grids(batch: torch.Tensor, rand_rots: list):
    """
    Rotate a batch of voxel grids based on random rotations.

    Args:
        batch (torch.Tensor): The input batch of voxel grids.
        rand_rots (list): A list of random rotations for each voxel grid in the batch.

    Returns:
        torch.Tensor: The rotated batch of voxel grids.
    """
    batch_sz = batch.shape[0]
    rot_batch = []
    for i in range(batch_sz):
        rand_rot, n_rots = rand_rots[0][i], rand_rots[1][i]
        if rand_rot is not None:
            # rot_i = torch.rot90(batch[i:i + 1], k=1, dims=rand_rot)
            rot_i = batch[i:i + 1]
            for j in range(n_rots):
                rot_i = torch.rot90(rot_i, k=1, dims=rand_rot)
        else:
            rot_i = batch[i:i + 1].clone()
        rot_batch.append(rot_i)

    return torch.cat(rot_batch, 0)


def unrotate_voxel_grids(xhats: torch.Tensor, rand_rots: list):
    """
    Unrotates the voxel grids based on the given random rotations.

    Args:
        xhats (torch.Tensor): The input voxel grids.
        rand_rots (list): The list of random rotations.

    Returns:
        torch.Tensor: The unrotated voxel grids.
    """
    unrot_gen_vox_ligands = []
    for i, xhat in enumerate(xhats):
        rand_rot = rand_rots[0][i]
        if rand_rot is None:
            unrot_i = xhat.unsqueeze(0)
        else:
            n_rots = rand_rots[1][i]
            unrot_i = xhat.unsqueeze(0)
            for j in range(n_rots):
                unrot_i = torch.rot90(unrot_i, k=1, dims=rand_rot[::-1])
        unrot_gen_vox_ligands.append(unrot_i)
    unrot_gen_vox_ligands = torch.concat(unrot_gen_vox_ligands)
    return unrot_gen_vox_ligands


def get_pocket_mask(pocket: torch.Tensor, n_channels: int = 7):
    """
    Generate a mask for the given pocket tensor.

    Args:
        pocket (torch.Tensor): The input pocket tensor.
        n_channels (int, optional): The number of channels in the mask. Defaults to 7.

    Returns:
        torch.Tensor: The generated mask tensor.
    """
    mask = ((pocket > 0).float().sum(1) > 0)
    # mask = ndimage.binary_dilation(mask.cpu())
    mask = torch.Tensor(mask).unsqueeze(1).repeat(1, n_channels, 1, 1, 1).cuda()
    return mask.bool()
