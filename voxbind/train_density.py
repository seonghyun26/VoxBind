"""train_density.py — DDP MAE/ELECTRA pre-training for the ViT density encoder.

Two pretexts are supported via `mae.pretext_style`:

  - "mae"     : synthetic Gaussian-blur density + 3D block masking (zero
                masked voxels) + density-reconstruction + 11-channel atom-
                structure auxiliary head. Apples-to-apples vs cnn-MAE.

  - "electra" : same synthetic density, but masked blocks are CORRUPTED
                (replaced) rather than zeroed. Corruption op is sampled per
                batch from `cfg.electra.corruption_ops` (any combination of
                `swap` / `reblur` / `noise`, weighted by
                `cfg.electra.corruption_op_weights`). The discriminator emits
                one logit per ViT patch and is trained with BCE on the binary
                "was this patch corrupted?" target; the 11-channel atom-
                structure auxiliary head is retained.

Checkpoint layout:
  state_dict_ema          full DensityViTMAE EMA
  encoder_state_dict_ema  encoder.* slice — drops into VoxBind.density_encoder
                          (with `density_encoder_type=vit`) via the existing
                          loader in `models/__init__.py`.

Launch
------
    cd voxbind
    CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --standalone --nproc_per_node=5 \\
        train_density.py --config-name=config_train_density_vit_electra \\
        exp_name=260524_density_vit_electra_pretrain \\
        output_dir=exps/260524_density_vit_electra_pretrain
"""

import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from voxbind.dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.mae_ops import (
    corrupt_noise, corrupt_reblur, corrupt_swap,
    gaussian_blur3d, gradient_magnitude3d, make_atom_biased_block_mask,
    make_block_mask, make_cluster_mask, make_interface_mask, per_sample_zscore,
    region_keep_masks, voxel_mask_to_patch_target,
)
from voxbind.models.density_cha_mae import DensityChaMAE
from voxbind.models.density_vit import DensityViTMAE
from voxbind.train_common import (  # shared pretext-independent engine (extracted)
    AsyncCheckpointSaver, _amp_setup, _build_merged_atoms, _build_role_atoms, _channel_layout,
    _ch_freq_cache_path, _cleanup_ddp, _compile_options, _reconcile_input_keys,
    _setup_ddp, _unwrap, _val_cache_path, _CH_FREQ_DEFAULT_N_SAMPLES,
    _CH_FREQ_POS_THRESH, _VALID_INPUT_MODES, log_metrics, maybe_compile_model,
    precompute_channel_weights, precompute_val, set_atom_channels, set_density_sources,
)
from voxbind.models.ema import ModelEma
from voxbind.utils.base_utils import create_exp_dir, makedir, seed_everything
from voxbind.voxelizer import Voxelizer

logger = logging.getLogger("train-density-vit-mae")
















# ── Prefetcher (MAE zero-masking OR ELECTRA rule-based corruption) ────────────

_RULE_BASED_OPS = ("swap", "reblur", "noise")


def _sample_op(
    ops: tuple, weights: tuple, generator: torch.Generator = None,
) -> str:
    """Sample one corruption op name from `ops` with given `weights`."""
    if len(ops) == 1:
        return ops[0]
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp(min=1e-8)
    if generator is not None:
        idx = int(torch.multinomial(w, num_samples=1, generator=generator).item())
    else:
        idx = int(torch.multinomial(w, num_samples=1).item())
    return ops[idx]


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temp: float) -> torch.Tensor:
    """SimCLR NT-Xent contrastive loss for paired views.

    z_a, z_b : (B, d) L2-normalized projections of two augmented views (here, two
    independent MAE maskings) of the SAME B complexes. Positives = (a_i, b_i);
    negatives = every other view in the local batch (2B-2 per anchor). Symmetric
    over both view→view directions. Local in-batch negatives only (no cross-GPU
    gather) — keeps the DDP backward a single clean reducer pass.
    """
    B = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)                 # (2B, d)
    sim = (z @ z.t()) / temp                         # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))                # drop self-similarity
    targets = torch.cat([
        torch.arange(B, 2 * B, device=z.device),     # a_i's positive is b_i
        torch.arange(0, B, device=z.device),         # b_i's positive is a_i
    ])
    return F.cross_entropy(sim, targets)






class MAEPrefetcher:
    """Voxelize → build per-mode input → mask/corrupt, overlapped with compute.

    The `input_mode` selects which voxel tensor the ViT actually sees:
      - density         : (B,  1, G, G, G) — z-scored density (synthetic or xray)
      - atomblob        : (B, 11, G, G, G) — atom voxels (7 ligand + 4 pocket)
      - atomblob_density: (B, 12, G, G, G) — cat(atom voxels, density)

    Yields per batch:
        x_in        (B, C, G, G, G)  — masked (mae) OR corrupted (electra) input
        x_clean     (B, C, G, G, G)  — un-masked, noiseless target for MAE recon
        mask        (B, 1, G, G, G)  — bool, True = masked/corrupted
        target_str  (B, 11, G, G, G) — voxelized ligand (7) + pocket (4),
                                        used only when the model's struct head
                                        is enabled (n_struct_channels > 0)
    where C = 1 / 11 / 12 according to input_mode.
    """

    def __init__(
        self,
        loader,
        voxelizer: Voxelizer,
        sigma_blur_vox_range: tuple,
        sigma_noise: float,
        block_size: int,
        mask_ratio: float,
        generator: torch.Generator = None,
        pretext_style: str = "mae",
        corruption_ops: tuple = ("swap",),
        corruption_op_weights: tuple = (1.0,),
        density_source: str = "synthetic",
        input_mode: str = "density",
        mask_strategy: str = "uniform",
        mask_atom_tau: float = 1.0,
        mask_n_seeds: int = 4,
        mask_ratio_min: float = None,
        mask_ratio_max: float = None,
        modal_mask_prob: float = 0.0,
        density_visible: bool = False,
        lig_mask_thresh: float = 0.1,
        pocket_ed_mask: str = "ligand_footprint",
        mask_as_channel: bool = False,
        apo_prob: float = 0.0,
        contrastive: bool = False,
        with_gradmag: bool = False,
        gradmag_reconstruct: bool = True,
        gradmag_noise: bool = False,
        density_input: bool = True,
        n_lig_ch: int = 7,
        n_poc_ch: int = 4,
    ):
        assert pretext_style in ("mae", "electra", "denoise")
        assert density_source in ("synthetic", "xray"), (
            f"density_source={density_source!r}; expected 'synthetic' or 'xray'"
        )
        assert input_mode in _VALID_INPUT_MODES, (
            f"input_mode={input_mode!r}; expected one of {_VALID_INPUT_MODES}"
        )
        assert mask_strategy in ("uniform", "atom_biased", "cluster", "ligand", "interface"), (
            f"mask_strategy={mask_strategy!r}; expected 'uniform', 'atom_biased', 'cluster', 'ligand' or 'interface'"
        )
        for op in corruption_ops:
            if op not in _RULE_BASED_OPS:
                raise NotImplementedError(
                    f"corruption op {op!r} not supported in v1; expected one of "
                    f"{_RULE_BASED_OPS} (learned-generator path is not wired)."
                )
        assert len(corruption_ops) == len(corruption_op_weights), (
            f"len(corruption_ops)={len(corruption_ops)} != "
            f"len(corruption_op_weights)={len(corruption_op_weights)}"
        )
        self.loader = loader
        self.voxelizer = voxelizer
        self.sigma_range = sigma_blur_vox_range
        self.sigma_noise = sigma_noise
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        # Optional variable mask-ratio (R2MAE): if both bounds are set and max>min, each
        # training batch draws r ~ U[min,max] instead of the fixed mask_ratio (see forward).
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self._var_ratio = (
            mask_ratio_min is not None and mask_ratio_max is not None
            and float(mask_ratio_max) > float(mask_ratio_min)
        )
        if self._var_ratio:
            print(f">> [MAEPrefetcher] VARIABLE mask_ratio ~ U[{mask_ratio_min}, {mask_ratio_max}] "
                  f"(per-batch, train-only); fixed fallback = {mask_ratio}", flush=True)
        self.generator = generator
        self.pretext_style = pretext_style
        self.corruption_ops = tuple(corruption_ops)
        self.corruption_op_weights = tuple(corruption_op_weights)
        self.density_source = density_source
        self.input_mode = input_mode
        self.mask_strategy = mask_strategy
        self.mask_atom_tau = mask_atom_tau
        self.mask_n_seeds = mask_n_seeds
        self.modal_mask_prob = modal_mask_prob
        # density_visible: keep the density(+gradmag) channels FULLY VISIBLE (never spatially
        # masked) while the atom channels are masked as usual → the encoder reconstructs the
        # masked ATOMS from the visible density (cross-modal, MultiMAE-style). Since density
        # magnitude encodes atom identity (electron count), this trains the density→chemistry
        # decoding directly. Reverse of modal_mask_prob (which drops density).
        self.density_visible = bool(density_visible)
        # Two-tower density gating. `_lig_mask_thresh` is the vdW-occupancy threshold used to
        # draw the pocket/ligand boundary. `pocket_ed_mask` selects the boundary basis
        # (ligand_footprint = legacy apo-like pocket; protein_vdw = M_P = protein-atom vdW;
        # none = holo/unmasked). `mask_as_channel` additionally feeds the binary keep-mask as an
        # explicit input+recon channel (folded into the atom block). See mae_ops.region_keep_masks.
        self._lig_mask_thresh = float(lig_mask_thresh)
        self.pocket_ed_mask = str(pocket_ed_mask)
        self.mask_as_channel = bool(mask_as_channel)
        # apo augmentation: for a random `apo_prob` fraction of TRAIN samples, delete the ligand
        # from the ENCODER INPUT (ligand atoms + footprint density → 0) but keep the ligand-region
        # density as a PREDICTION TARGET (force-masked recon of the real holo density) — the
        # encoder learns to predict the ligand's electron-density field from the apo pocket alone,
        # matching VoxBind's pocket→ligand generation setup. 0.0 = off (holo only). See _step.
        self.apo_prob = float(apo_prob)
        self.contrastive = bool(contrastive)
        self.with_gradmag = with_gradmag
        self.gradmag_reconstruct = gradmag_reconstruct
        self.gradmag_noise = gradmag_noise
        self.density_input = bool(density_input)
        self.n_lig_ch = int(n_lig_ch)
        self.n_poc_ch = int(n_poc_ch)
        self.layout = _channel_layout(
            input_mode, with_gradmag, gradmag_reconstruct, density_input=self.density_input,
            mask_as_channel=self.mask_as_channel,
        )
        # 2 density sources → the dataset also emits the mFo-DFc (density, gradmag) pair,
        # appended as trailing channels in _step. False (single 2Fo-Fc) = unchanged path.
        self.with_diff = self.layout.get("n_density_src", 1) > 1
        self.stream = torch.cuda.Stream()

    def _step(self, batch):
        device = self.voxelizer.device
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                v_lig = self.voxelizer.forward(batch["ligand"], num_channels=self.n_lig_ch)
                v_poc = self.voxelizer.forward(batch["pocket"], num_channels=self.n_poc_ch)

                # Density is only synthesized / loaded when the input actually
                # needs it. atomblob* modes skip this entirely — no I/O via the
                # xray path, no blur on the synthetic path.
                need_density = self.input_mode in (
                    "density", "atomblob_density", "atomblob_merged_density",
                    "roleblob_density", "pocket_density", "ligand_density",
                )
                # atom-biased / cluster masks need the all-channel atom sum even in atomblob mode
                need_atoms_sum = need_density or self.mask_strategy in ("atom_biased", "cluster")
                d_clean = None
                atoms_sum = None
                sigma_lo, sigma_hi = self.sigma_range  # always defined for reblur
                if need_atoms_sum:
                    atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
                if need_density:
                    if self.density_source == "xray":
                        # Real 2Fo-Fc crop, already locally ±3σ-clipped + z-scored
                        # by DatasetCrossDockedXray.normalize_crop. Lift to (B,1,G³).
                        d_clean = batch["xray_density"].to(device, non_blocking=True).unsqueeze(1)
                    else:
                        if self.generator is not None:
                            u = torch.rand(1, generator=self.generator).item()
                        else:
                            u = torch.rand(1).item()
                        sigma_vox = sigma_lo + u * (sigma_hi - sigma_lo)
                        d_clean = gaussian_blur3d(atoms_sum, sigma_vox)
                        d_clean = per_sample_zscore(d_clean)

                # Two-tower density gating: split the shared 2Fo-Fc map into the pocket region
                # (rho_P) and the ligand region (rho_L) per the pocket_ed_mask basis. The pocket
                # tower sees rho_P (ligand-free / protein-interior); the ligand tower sees rho_L
                # (the non-protein / ligand-accessible region). `keep_ch` is stashed for the
                # optional mask-as-channel input. gradmag is re-derived from the MASKED density
                # below so both channels stay consistent.
                twotower = self.input_mode in ("pocket_density", "ligand_density")
                keep_ch = None
                if twotower and d_clean is not None:
                    lig_occ = v_lig.sum(dim=1, keepdim=True)
                    poc_occ = v_poc.sum(dim=1, keepdim=True)
                    keep_p, keep_l = region_keep_masks(
                        lig_occ, poc_occ, self.pocket_ed_mask, self._lig_mask_thresh)
                    keep_ch = keep_p if self.input_mode == "pocket_density" else keep_l
                    d_clean = d_clean * keep_ch

                # Gradient-magnitude channel ‖∇ρ‖. xray: precomputed in the
                # dataset from the aligned crop (post-augmentation). synthetic:
                # derived on-GPU from the clean density. Appended as trailing ch.
                g_clean = None
                if self.with_gradmag:
                    if twotower:
                        # derive ‖∇ρ‖ from the region-MASKED density (consistency)
                        g_clean = per_sample_zscore(gradient_magnitude3d(d_clean))
                    elif self.density_source == "xray":
                        g_clean = batch["xray_gradmag"].to(device, non_blocking=True).unsqueeze(1)
                    else:
                        g_clean = per_sample_zscore(gradient_magnitude3d(d_clean))

                # Assemble the multi-channel input per mode.
                if self.input_mode == "density":
                    x_clean = d_clean                                       # (B, 1, G³)
                elif self.input_mode == "atomblob":
                    x_clean = torch.cat([v_lig, v_poc], dim=1)              # (B, 11, G³)
                elif self.input_mode == "atomblob_density":
                    x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)     # (B, 12, G³)
                elif self.input_mode == "atomblob_merged":
                    x_clean = _build_merged_atoms(v_lig, v_poc)             # (B, 7, G³)
                elif self.input_mode == "atomblob_merged_density":
                    v_merged = _build_merged_atoms(v_lig, v_poc)
                    x_clean = torch.cat([v_merged, d_clean], dim=1)         # (B, 8, G³)
                elif self.input_mode == "roleblob":
                    x_clean = _build_role_atoms(v_lig, v_poc)               # (B, 2, G³)
                elif self.input_mode == "roleblob_density":
                    x_clean = torch.cat(
                        [_build_role_atoms(v_lig, v_poc), d_clean], dim=1)  # (B, 3, G³)
                elif self.input_mode == "ligand":
                    x_clean = v_lig                                        # (B, 7, G³) ligand atoms only
                elif self.input_mode == "pocket":
                    x_clean = v_poc                                        # (B, 4, G³) pocket atoms only (coords-only)
                elif self.input_mode in ("pocket_density", "ligand_density"):
                    atoms = v_poc if self.input_mode == "pocket_density" else v_lig
                    x_clean = torch.cat([atoms, d_clean], dim=1)           # [ atoms, density ]
                else:
                    raise RuntimeError(f"unexpected input_mode={self.input_mode!r}")

                # Append gradmag as the trailing channel: [ …atoms…, density, gradmag ].
                if self.with_gradmag:
                    x_clean = torch.cat([x_clean, g_clean], dim=1)

                # Trailing input-only region-mask channel (two-tower, mask_as_channel): appended
                # LAST so it lands past n_recon and is excluded from the recon target.
                if self.mask_as_channel and keep_ch is not None:
                    x_clean = torch.cat([x_clean, keep_ch.to(x_clean.dtype)], dim=1)

                # Second density SOURCE (mFo-DFc difference map): append its (density, gradmag)
                # pair from the dataset → [ …atoms…, dens0, grad0, dens1, grad1 ], matching
                # channel_groups=[…,2,2]. Only the xray path carries a real second map.
                if self.with_diff and self.density_source == "xray":
                    d_diff = batch["xray_diff_density"].to(device, non_blocking=True).unsqueeze(1)
                    x_clean = torch.cat([x_clean, d_diff], dim=1)
                    if self.with_gradmag:
                        g_diff = batch["xray_diff_gradmag"].to(device, non_blocking=True).unsqueeze(1)
                        x_clean = torch.cat([x_clean, g_diff], dim=1)

                # Noise is a density-domain augmentation applied to the density
                # channel BY INDEX (gradmag may trail it). Atom-only modes skip it;
                # gradmag is noised only when gradmag_noise is set.
                atom_only = self.input_mode in ("atomblob", "atomblob_merged", "roleblob")
                if self.sigma_noise > 0 and not atom_only:
                    def _randn_channel():
                        shape = (x_clean.shape[0], 1, *x_clean.shape[2:])
                        if self.generator is not None:
                            return torch.randn(shape, device=device, generator=self.generator)
                        return torch.randn(shape, device=device)
                    x_noisy = x_clean.clone()
                    d_idx = self.layout["density_idx"]
                    x_noisy[:, d_idx:d_idx + 1] = (
                        x_noisy[:, d_idx:d_idx + 1] + _randn_channel() * self.sigma_noise
                    )
                    if self.with_gradmag and self.gradmag_noise:
                        g_idx = self.layout["gradmag_idx"]
                        x_noisy[:, g_idx:g_idx + 1] = (
                            x_noisy[:, g_idx:g_idx + 1] + _randn_channel() * self.sigma_noise
                        )
                else:
                    x_noisy = x_clean

                B = x_clean.shape[0]
                G = x_clean.shape[-1]
                # Variable mask-ratio (R2MAE-style): when a [min,max] range is configured,
                # draw a fresh r ~ U[min,max] PER BATCH so one encoder is trained across a
                # spread of masking difficulty — a cheap single-model analog of a mask-ratio
                # ensemble. TRAIN-ONLY: val uses a fixed ratio in evaluate(), so val loss
                # stays comparable across epochs. Falls back to the fixed self.mask_ratio.
                if self._var_ratio:
                    _gdev = self.generator.device if self.generator is not None else device
                    _u = torch.rand((), device=_gdev, generator=self.generator).item()
                    ratio = self.mask_ratio_min + (self.mask_ratio_max - self.mask_ratio_min) * _u
                else:
                    ratio = self.mask_ratio
                if self.mask_strategy == "atom_biased":
                    mask = make_atom_biased_block_mask(
                        atoms_sum, self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "cluster":
                    mask = make_cluster_mask(
                        atoms_sum, self.block_size, ratio,
                        n_seeds=self.mask_n_seeds, generator=self.generator,
                    )
                elif self.mask_strategy == "ligand":
                    # Pocket-conditioned ligand prediction: mask the blocks most occupied
                    # by LIGAND atoms (reuse the atom-biased top-K, weighted by ligand mass
                    # only) so the encoder reconstructs the ligand region from the pocket —
                    # mirrors the de-novo placement task. Small mask_ratio ≈ ligand footprint.
                    lig_sum = v_lig.sum(dim=1, keepdim=True)
                    mask = make_atom_biased_block_mask(
                        lig_sum, self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "interface":
                    # Mask the ligand-pocket CONTACT region (where binding affinity lives),
                    # reconstruct it from its surroundings → learn interaction structure.
                    mask = make_interface_mask(
                        v_lig.sum(dim=1, keepdim=True), v_poc.sum(dim=1, keepdim=True),
                        self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                else:
                    mask = make_block_mask(B, G, self.block_size, ratio, device)
                # mask: (B, 1, G, G, G) — broadcasts across all input channels

                # apo augmentation (opt-in): for a random `apo_prob` fraction of samples, DELETE
                # the ligand from the ENCODER INPUT but keep its density as a PREDICTION TARGET —
                # the encoder must reconstruct the ligand's electron density from the apo pocket
                # alone (exactly VoxBind's pocket→ligand task; makes the encoder apo-robust).
                #   input  : ligand atom channels → 0 everywhere; density+gradmag → 0 inside the
                #            ligand vdW footprint (pocket atoms + pocket density stay intact).
                #   target : ligand atom channels → 0 (not predicted); the ligand-footprint
                #            density(+gradmag) keep their REAL holo values and the footprint is
                #            FORCE-MASKED, so the recon head predicts the ligand density field.
                # Random uniform masking still covers the remaining (pocket) region as usual.
                if self.apo_prob > 0.0:
                    lig_occ = v_lig.sum(dim=1, keepdim=True)                      # (B,1,G³)
                    fp = (lig_occ > self._lig_mask_thresh).to(x_noisy.dtype)      # ligand footprint
                    apo_f = (torch.rand(B, device=device, generator=self.generator)
                             < self.apo_prob).view(B, 1, 1, 1, 1).to(x_noisy.dtype)
                    lig_keep = 1.0 - apo_f              # (B,1,1,1,1): 0 → drop the whole ligand
                    keep = 1.0 - fp * apo_f             # (B,1,G³): 0 → drop footprint (input side)
                    d_idx = self.layout["density_idx"]
                    g_idx = self.layout.get("gradmag_idx", None)
                    # input and target diverge in the ligand region → ensure separate tensors
                    if x_noisy is x_clean:
                        x_noisy = x_noisy.clone()
                    # INPUT (apo): ligand atoms removed everywhere; density/gradmag removed in fp.
                    x_noisy[:, :self.n_lig_ch] = x_noisy[:, :self.n_lig_ch] * lig_keep
                    x_noisy[:, d_idx:d_idx + 1] = x_noisy[:, d_idx:d_idx + 1] * keep
                    if g_idx is not None:
                        x_noisy[:, g_idx:g_idx + 1] = x_noisy[:, g_idx:g_idx + 1] * keep
                    # TARGET: don't predict ligand ATOMS (blank), but DO predict the ligand
                    # DENSITY(+gradmag) — leave x_clean's density channels at real holo values.
                    x_clean[:, :self.n_lig_ch] = x_clean[:, :self.n_lig_ch] * lig_keep
                    # force the footprint to be reconstructed (input hidden, density predicted)
                    mask = mask | ((fp * apo_f) > 0.5)

                x_in_b = None
                # Encoder input width: full channels normally; atoms-only when density
                # is a reconstruction target only (density_input=False → n_in < C). The
                # trailing density/gradmag channels stay in x_clean (the target) but are
                # sliced off the encoder input here. No-op when n_in == C.
                n_in_enc = self.layout["n_in"]
                if self.pretext_style == "mae":
                    x_in = x_noisy[:, :n_in_enc] * (~mask).to(x_noisy.dtype)
                    # Cross-modal masking (opt-in): per-sample, drop the ENTIRE density
                    # (+gradmag) modality from the input so the encoder must predict it
                    # from the atom channels alone — a cross-modal objective on top of the
                    # spatial MAE. Target (x_clean) is unchanged, so the recon loss scores
                    # the atoms→density prediction. No-op when modal_mask_prob=0.
                    if (self.modal_mask_prob > 0.0 and self.density_input
                            and self.layout.get("n_density", 0) > 0):
                        drop = torch.rand(B, device=device, generator=self.generator) < self.modal_mask_prob
                        if bool(drop.any()):
                            x_in[drop, self.layout["density_idx"]] = 0.0
                            g_idx = self.layout.get("gradmag_idx", None)
                            if g_idx is not None:
                                x_in[drop, g_idx] = 0.0
                    # density_visible (opt-in): restore density(+gradmag) to their unmasked
                    # values so ONLY the atom channels are spatially masked → reconstruct atoms
                    # from fully-visible density. No-op when density_visible=False.
                    if (self.density_visible and self.density_input
                            and self.layout.get("n_density", 0) > 0):
                        d_idx = self.layout["density_idx"]
                        x_in[:, d_idx] = x_noisy[:, d_idx]
                        g_idx = self.layout.get("gradmag_idx", None)
                        if g_idx is not None:
                            x_in[:, g_idx] = x_noisy[:, g_idx]
                    # Contrastive 2nd view (opt-in): a SECOND independent uniform masking of
                    # the SAME complex → the augmentation pair for the InfoNCE aux loss. The
                    # encoder is pulled to map both maskings of a complex to the same vector.
                    if self.contrastive:
                        mask_b = make_block_mask(B, G, self.block_size, ratio, device)
                        x_in_b = x_noisy[:, :n_in_enc] * (~mask_b).to(x_noisy.dtype)
                elif self.pretext_style == "denoise":
                    # Recovery-from-noise: the FULL noised input (no masking); the recon
                    # loss covers every voxel (mask=all), so the model denoises the signal.
                    x_in = x_noisy[:, :n_in_enc]
                    mask = torch.ones_like(mask)
                else:
                    # ELECTRA corruption ops are currently single-channel only;
                    # the new multi-channel input modes go through MAE pretext.
                    if self.input_mode != "density" or self.with_gradmag:
                        raise NotImplementedError(
                            f"pretext_style='electra' supports single-channel density "
                            f"only (input_mode={self.input_mode!r}, "
                            f"with_gradmag={self.with_gradmag}); use pretext_style='mae'."
                        )
                    op = _sample_op(
                        self.corruption_ops, self.corruption_op_weights, self.generator
                    )
                    if op == "swap":
                        x_in = corrupt_swap(x_noisy, mask)
                    elif op == "noise":
                        x_in = corrupt_noise(x_noisy, mask)
                    elif op == "reblur":
                        # Re-blur the raw atoms at a different σ drawn from a wider
                        # range so the alt density is materially different from
                        # the clean one (otherwise corruption is trivial).
                        sigma_alt_lo = max(0.4 * sigma_lo, 0.5)
                        sigma_alt_hi = 2.0 * sigma_hi
                        if self.generator is not None:
                            u2 = torch.rand(1, generator=self.generator).item()
                        else:
                            u2 = torch.rand(1).item()
                        sigma_alt = sigma_alt_lo + u2 * (sigma_alt_hi - sigma_alt_lo)
                        x_in = corrupt_reblur(atoms_sum, x_noisy, mask, sigma_alt)
                    else:
                        raise RuntimeError(f"unhandled corruption op: {op}")

                target_str = torch.cat([v_lig, v_poc], dim=1)
        return x_in, x_clean, mask, target_str, x_in_b

    def __iter__(self):
        loader_it = iter(self.loader)
        try:
            next_batch = next(loader_it)
        except StopIteration:
            return
        next_out = self._step(next_batch)
        for batch in loader_it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur = next_out
            next_out = self._step(batch)
            yield cur
        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_out

    def __len__(self):
        return len(self.loader)


# ── Losses (branches on pretext_style; both keep the structure aux head) ─────

def compute_losses(
    out_pretext, a_hat, x_clean, target_str, mask,
    struct_pos_weight: float = 1.0,
    struct_pos_thresh: float = 0.05,
    pretext_style: str = "mae",
    patch_size: int = 8,
    input_mode: str = "density",
    ch_weight: torch.Tensor = None,
    atom_pos_weight: float = 1.0,
    atom_pos_thresh: float = 0.05,
    with_gradmag: bool = False,
    gradmag_reconstruct: bool = True,
):
    """Returns a dict of named losses.

    mae path:
      L_dens : masked-voxel MSE between out_pretext (B,C,G,G,G) and x_clean,
               where C = 1 / 7 / 8 / 11 / 12 per input_mode.
      L_str  : 11-ch atom structure MSE (pos-weighted), only if a_hat is given.

      For *_density modes also returns the per-modality split:
        L_dens_atom    — MSE on the atom channels in masked positions
        L_dens_density — MSE on the trailing density channel in masked positions

      Two complementary weighting axes (composable):
        `ch_weight`        : 1-D tensor of length n_channels, applied per-channel.
                             Used by inv-sqrt-freq to balance across atom-type
                             channels (rare F/Cl/P up-weighted vs common C).
        `atom_pos_weight`  : scalar > 1, applied per-voxel within atom channels.
                             At voxels where target > `atom_pos_thresh`, the
                             squared error is multiplied by atom_pos_weight.
                             Density and atom-off voxels are unchanged. Targets
                             the SECOND sparsity axis: within-channel atom-on
                             voxels are still rare even after channel weighting.

    electra path:
      L_rtd  : per-patch BCE-with-logits between out_pretext (B,1,Gp,Gp,Gp)
               and patch-level corruption target derived from `mask`.
      L_str  : 11-ch atom structure MSE (pos-weighted), only if a_hat is given.
    """
    losses = {}
    if pretext_style == "electra":
        # Pool voxel mask to patch grid (max-pool).
        patch_target = voxel_mask_to_patch_target(mask, patch_size)
        # Per-patch BCE; mean reduction over (B, 1, Gp, Gp, Gp).
        L_pretext = F.binary_cross_entropy_with_logits(out_pretext, patch_target)
        out_key = "L_rtd"
    else:
        m = mask.to(out_pretext.dtype)
        # mask broadcasts across channels: total masked (B,c,pos) entries =
        # n_masked_spatial × n_channels — keep per-element MSE scale.
        n_masked_spatial = m.sum().clamp(min=1.0)
        n_channels = out_pretext.shape[1]                     # = n_recon
        # When gradmag is input-only the head emits n_recon < n_in channels;
        # the recon target is the leading n_recon channels of x_clean (gradmag
        # trails and is excluded). Prefix slice → atom/density indices unchanged.
        target = x_clean if n_channels == x_clean.shape[1] else x_clean[:, :n_channels]
        diff_sq = (out_pretext - target) ** 2 * m

        # ── atom_pos_weight: up-weight error² at atom-on voxels in atom channels.
        lay = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
        n_atom = lay["n_atom"]                                # leading atom channels
        if atom_pos_weight > 1.0 and n_atom > 0:
            # Mask of atom-on positions for the leading atom channels.
            pos_atom = (x_clean[:, :n_atom] > atom_pos_thresh).to(diff_sq.dtype)
            # Multiplier: atom-on voxels get atom_pos_weight, atom-off get 1.
            atom_mult = 1.0 + (atom_pos_weight - 1.0) * pos_atom              # (B, n_atom, G³)
            if n_atom < n_channels:
                # Density (and any trailing non-atom channels) get 1× (no boost).
                pad = torch.ones(
                    (atom_mult.shape[0], n_channels - n_atom, *atom_mult.shape[2:]),
                    device=atom_mult.device, dtype=atom_mult.dtype,
                )
                full_mult = torch.cat([atom_mult, pad], dim=1)
            else:
                full_mult = atom_mult
            diff_sq = diff_sq * full_mult

        if ch_weight is not None:
            assert ch_weight.shape[0] == n_channels, (
                f"ch_weight has {ch_weight.shape[0]} entries but out_pretext has "
                f"{n_channels} channels"
            )
            w_full = ch_weight.to(diff_sq.device, diff_sq.dtype).view(1, n_channels, 1, 1, 1)
            diff_sq_weighted = diff_sq * w_full
            # Weights are normalized to sum=n_channels, so total mass is preserved.
            L_pretext = diff_sq_weighted.sum() / (n_masked_spatial * n_channels)
        else:
            L_pretext = diff_sq.sum() / (n_masked_spatial * n_channels)
        out_key = "L_dens"

        # Per-modality split over the RECONSTRUCTED channels — reported whenever
        # the target mixes ≥2 modalities. Channel order: [atoms, density, gradmag].
        # `diff_sq` already reflects atom_pos_weight if active.
        n_density = lay["n_density"]
        recon_gradmag = with_gradmag and gradmag_reconstruct
        n_modalities = (n_atom > 0) + (n_density > 0) + (1 if recon_gradmag else 0)
        if n_modalities >= 2:
            def _masked_mean(c0, c1):
                if ch_weight is not None:
                    num = (diff_sq[:, c0:c1] * w_full[:, c0:c1]).sum()
                else:
                    num = diff_sq[:, c0:c1].sum()
                return num / (n_masked_spatial * (c1 - c0))
            if n_atom > 0:
                losses["L_dens_atom"] = _masked_mean(0, n_atom)
            if n_density > 0:
                losses["L_dens_density"] = _masked_mean(n_atom, n_atom + n_density)
            if recon_gradmag:
                g0 = n_atom + n_density
                losses["L_dens_gradmag"] = _masked_mean(g0, g0 + 1)
    losses[out_key] = L_pretext

    if a_hat is not None:
        if struct_pos_weight > 1.0:
            pos = (target_str > struct_pos_thresh).to(a_hat.dtype)
            w = 1.0 + (struct_pos_weight - 1.0) * pos
            L_str = ((a_hat - target_str) ** 2 * w).mean()
        else:
            L_str = F.mse_loss(a_hat, target_str)
        losses["L_str"] = L_str
    else:
        # Struct head disabled — keep the key for downstream code that
        # always expects it, but with zero contribution.
        losses["L_str"] = torch.zeros((), device=out_pretext.device)
    return losses


# ── Val cache ─────────────────────────────────────────────────────────────────





# ── Channel-frequency cache + inv-sqrt-freq weights ──────────────────────────







def val_epoch(cfg, model, val_cache, device, ch_weight=None) -> dict:
    """Deterministic val pass — fixed-seed σ_blur / noise / mask per sample.

    In electra mode, cycles deterministically through `cfg.electra.corruption_ops`
    by batch index so val loss is comparable across epochs.
    """
    model.eval()
    amp_enabled, amp_ctx_fn = _amp_setup(cfg)
    channels_last = bool(cfg.get("channels_last", False))
    sigma_lo = cfg.mae.sigma_blur_a_lo / cfg.vox.resolution
    sigma_hi = cfg.mae.sigma_blur_a_hi / cfg.vox.resolution
    block = cfg.mae.block_size
    ratio = cfg.mae.mask_ratio
    sigma_noise = cfg.mae.sigma_noise
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    density_source = str(cfg.mae.get("density_source", "synthetic"))
    input_mode = str(cfg.get("input_mode", "density"))
    mask_strategy = str(cfg.mae.get("mask_strategy", "uniform"))
    mask_atom_tau = float(cfg.mae.get("mask_atom_tau", 1.0))
    mask_n_seeds = int(cfg.mae.get("mask_n_seeds", 4))
    patch_size = int(cfg.model.patch_size)
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    gradmag_noise = bool(cfg.mae.get("gradmag_noise", False))
    density_input = bool(cfg.mae.get("density_input", True))
    mask_as_channel = bool(cfg.mae.get("mask_as_channel", False))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             density_input=density_input, mask_as_channel=mask_as_channel)
    need_density = input_mode in (
        "density", "atomblob_density", "atomblob_merged_density",
        "roleblob_density", "pocket_density", "ligand_density",
    )
    need_atoms_sum = need_density or mask_strategy in ("atom_biased", "cluster")
    lig_mask_thresh = float(cfg.mae.get("lig_mask_thresh", 0.1))
    pocket_ed_mask = str(cfg.mae.get("pocket_ed_mask", "ligand_footprint"))

    if pretext_style == "electra":
        corruption_ops = tuple(cfg.electra.corruption_ops)
        pretext_key = "L_rtd"
        lambda_pretext = float(cfg.mae.get("lambda_rtd", 1.0))
    else:
        corruption_ops = ()
        pretext_key = "L_dens"
        lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))

    voxels_lig = val_cache["voxels_lig"]
    voxels_poc = val_cache["voxels_poc"]
    cached_xray = val_cache.get("xray_density", None)
    cached_gradmag = val_cache.get("xray_gradmag", None)
    # 2nd density source (mFo-DFc): cached (density, gradmag) appended below to match training.
    with_diff = layout.get("n_density_src", 1) > 1
    cached_diff = val_cache.get("xray_diff_density", None)
    cached_diff_gradmag = val_cache.get("xray_diff_gradmag", None)
    n = voxels_lig.shape[0]

    L_pretext_sum, L_str_sum, n_batches = 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum, L_dens_gradmag_sum = 0.0, 0.0, 0.0
    gen = torch.Generator(device=device)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n, cfg.bsz)):
            gen.manual_seed(batch_idx + 1)
            v_lig = voxels_lig[start:start + cfg.bsz].to(device, non_blocking=True)
            v_poc = voxels_poc[start:start + cfg.bsz].to(device, non_blocking=True)
            d_clean = None
            atoms_sum = None
            if need_atoms_sum:
                atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
            if need_density:
                if density_source == "xray":
                    d_clean = cached_xray[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
                else:
                    sigma_vox = sigma_lo + (sigma_hi - sigma_lo) * 0.5
                    d_clean = gaussian_blur3d(atoms_sum, sigma_vox)
                    d_clean = per_sample_zscore(d_clean)
            twotower = input_mode in ("pocket_density", "ligand_density")
            keep_ch = None
            if twotower and d_clean is not None:                   # split shared map into rho_P / rho_L
                lig_occ = v_lig.sum(dim=1, keepdim=True)
                poc_occ = v_poc.sum(dim=1, keepdim=True)
                keep_p, keep_l = region_keep_masks(
                    lig_occ, poc_occ, pocket_ed_mask, lig_mask_thresh)
                keep_ch = keep_p if input_mode == "pocket_density" else keep_l
                d_clean = d_clean * keep_ch

            g_clean = None
            if with_gradmag:
                if twotower:
                    g_clean = per_sample_zscore(gradient_magnitude3d(d_clean))
                elif density_source == "xray":
                    g_clean = cached_gradmag[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
                else:
                    g_clean = per_sample_zscore(gradient_magnitude3d(d_clean))

            if input_mode == "density":
                x_clean = d_clean
            elif input_mode == "atomblob":
                x_clean = torch.cat([v_lig, v_poc], dim=1)
            elif input_mode == "atomblob_density":
                x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)
            elif input_mode == "atomblob_merged":
                x_clean = _build_merged_atoms(v_lig, v_poc)
            elif input_mode == "atomblob_merged_density":
                v_merged = _build_merged_atoms(v_lig, v_poc)
                x_clean = torch.cat([v_merged, d_clean], dim=1)
            elif input_mode == "roleblob":
                x_clean = _build_role_atoms(v_lig, v_poc)
            elif input_mode == "roleblob_density":
                x_clean = torch.cat([_build_role_atoms(v_lig, v_poc), d_clean], dim=1)
            elif input_mode == "ligand":
                x_clean = v_lig
            elif input_mode == "pocket":
                x_clean = v_poc                                       # pocket atoms only (coords-only)
            elif input_mode in ("pocket_density", "ligand_density"):
                atoms = v_poc if input_mode == "pocket_density" else v_lig
                x_clean = torch.cat([atoms, d_clean], dim=1)          # [ atoms, density ]
            else:
                raise RuntimeError(f"unexpected input_mode={input_mode!r}")

            if with_gradmag:
                x_clean = torch.cat([x_clean, g_clean], dim=1)

            # Trailing input-only region-mask channel (two-tower, mask_as_channel).
            if mask_as_channel and keep_ch is not None:
                x_clean = torch.cat([x_clean, keep_ch.to(x_clean.dtype)], dim=1)

            # 2nd density source (mFo-DFc): append its (density, gradmag) → matches training's
            # [ …atoms…, dens0, grad0, dens1, grad1 ]. Single-source → no-op.
            if with_diff and density_source == "xray":
                d_diff = cached_diff[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
                x_clean = torch.cat([x_clean, d_diff], dim=1)
                if with_gradmag:
                    g_diff = cached_diff_gradmag[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
                    x_clean = torch.cat([x_clean, g_diff], dim=1)

            atom_only = input_mode in ("atomblob", "atomblob_merged", "roleblob")
            if sigma_noise > 0 and not atom_only:
                x_noisy = x_clean.clone()
                d_idx = layout["density_idx"]
                x_noisy[:, d_idx:d_idx + 1] = (
                    x_noisy[:, d_idx:d_idx + 1]
                    + torch.randn((x_clean.shape[0], 1, *x_clean.shape[2:]),
                                  device=device, generator=gen) * sigma_noise
                )
                if with_gradmag and gradmag_noise:
                    g_idx = layout["gradmag_idx"]
                    x_noisy[:, g_idx:g_idx + 1] = (
                        x_noisy[:, g_idx:g_idx + 1]
                        + torch.randn((x_clean.shape[0], 1, *x_clean.shape[2:]),
                                      device=device, generator=gen) * sigma_noise
                    )
            else:
                x_noisy = x_clean

            B, G = x_clean.shape[0], x_clean.shape[-1]
            if mask_strategy == "atom_biased":
                mask = make_atom_biased_block_mask(
                    atoms_sum, block, ratio, tau=mask_atom_tau, generator=gen,
                )
            elif mask_strategy == "cluster":
                mask = make_cluster_mask(
                    atoms_sum, block, ratio, n_seeds=mask_n_seeds, generator=gen,
                )
            else:
                blocks = torch.rand(B, 1, G // block, G // block, G // block,
                                    device=device, generator=gen) < ratio
                mask = blocks.repeat_interleave(block, 2)\
                             .repeat_interleave(block, 3)\
                             .repeat_interleave(block, 4)

            # Encoder input width: atoms-only when density is a target-only channel
            # (density_input=False → n_in < C); no-op otherwise. x_clean keeps all C
            # channels as the recon target.
            n_in_enc = layout["n_in"]
            if pretext_style == "mae":
                x_in = x_noisy[:, :n_in_enc] * (~mask).to(x_noisy.dtype)
            elif pretext_style == "denoise":
                x_in = x_noisy[:, :n_in_enc]
                mask = torch.ones_like(mask)
            else:
                if input_mode != "density":
                    raise NotImplementedError(
                        f"val_epoch: pretext_style='electra' not supported for input_mode={input_mode!r}"
                    )
                op = corruption_ops[batch_idx % len(corruption_ops)]
                if op == "swap":
                    x_in = corrupt_swap(x_noisy, mask)
                elif op == "noise":
                    x_in = corrupt_noise(x_noisy, mask)
                elif op == "reblur":
                    sigma_alt = max(0.4 * sigma_lo, 0.5) * 1.5
                    x_in = corrupt_reblur(atoms_sum, x_noisy, mask, sigma_alt)
                else:
                    raise RuntimeError(f"unsupported val corruption op: {op}")

            target_str = torch.cat([v_lig, v_poc], dim=1)
            if channels_last:
                x_in = x_in.to(memory_format=torch.channels_last_3d)
            with amp_ctx_fn():
                out_pretext, a_hat = model(x_in)
            out_pretext = out_pretext.float()
            if a_hat is not None:
                a_hat = a_hat.float()
            losses = compute_losses(
                out_pretext, a_hat, x_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
                pretext_style=pretext_style,
                patch_size=patch_size,
                input_mode=input_mode,
                ch_weight=ch_weight,
                atom_pos_weight=float(cfg.mae.get("atom_pos_weight", 1.0)),
                atom_pos_thresh=float(cfg.mae.get("atom_pos_thresh", 0.05)),
                with_gradmag=with_gradmag,
                gradmag_reconstruct=gradmag_reconstruct,
            )
            L_pretext_sum += losses[pretext_key].item()
            L_str_sum += losses["L_str"].item()
            if "L_dens_atom" in losses:
                L_dens_atom_sum += losses["L_dens_atom"].item()
            if "L_dens_density" in losses:
                L_dens_density_sum += losses["L_dens_density"].item()
            if "L_dens_gradmag" in losses:
                L_dens_gradmag_sum += losses["L_dens_gradmag"].item()
            n_batches += 1
            if cfg.debug and n_batches >= 5:
                break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    recon_gradmag = with_gradmag and gradmag_reconstruct
    n_modalities = (layout["n_atom"] > 0) + (layout["n_density"] > 0) + (1 if recon_gradmag else 0)
    if n_modalities >= 2:
        if layout["n_atom"] > 0:
            metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        if layout["n_density"] > 0:
            metrics["L_dens_density"] = L_dens_density_sum / n_b
        if recon_gradmag:
            metrics["L_dens_gradmag"] = L_dens_gradmag_sum / n_b
    return metrics


# ── Train loop ────────────────────────────────────────────────────────────────

def train_epoch(cfg, prefetcher, model, optimizer, model_ema, device, global_step,
                ch_weight=None, ema_source_model=None) -> tuple:
    model.train()
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    input_mode = str(cfg.get("input_mode", "density"))
    patch_size = int(cfg.model.patch_size)
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             density_input=bool(cfg.mae.get("density_input", True)),
                             mask_as_channel=bool(cfg.mae.get("mask_as_channel", False)))
    if pretext_style == "electra":
        pretext_key = "L_rtd"
        lambda_pretext = float(cfg.mae.get("lambda_rtd", 1.0))
    else:
        pretext_key = "L_dens"
        lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))
    # Contrastive auxiliary (opt-in): InfoNCE on two MAE-masked views per complex.
    lambda_con = float(cfg.mae.get("contrastive_weight", 0.0))
    con_temp = float(cfg.mae.get("contrastive_temp", 0.2))
    contrastive_on = lambda_con > 0.0 and pretext_style == "mae"
    # data2vec auxiliary (opt-in): predict the EMA-teacher's per-patch latent at masked patches.
    lambda_d2v = float(cfg.mae.get("data2vec_weight", 0.0))
    d2v_on = lambda_d2v > 0.0 and pretext_style == "mae"
    if d2v_on:
        model_ema.module.eval()                 # teacher = EMA; eval so DropPath/dropout are off

    L_pretext_sum, L_str_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum, L_dens_gradmag_sum, L_con_sum = 0.0, 0.0, 0.0, 0.0
    L_d2v_sum = 0.0
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    import contextlib
    amp_enabled, amp_ctx_fn = _amp_setup(cfg)
    channels_last = bool(cfg.get("channels_last", False))
    for i, (x_in, x_clean, mask, target_str, x_in_b) in enumerate(prefetcher):
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_iter)
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        if channels_last:
            x_in = x_in.to(memory_format=torch.channels_last_3d)
        do_con = contrastive_on and x_in_b is not None
        with sync_ctx:
            with amp_ctx_fn():
                if do_con:
                    if channels_last:
                        x_in_b = x_in_b.to(memory_format=torch.channels_last_3d)
                    out_pretext, a_hat, z_a, z_b = model(x_in, x_in_b)
                elif d2v_on:
                    out_pretext, a_hat, d2v_pred = model(x_in, return_d2v=True)
                else:
                    out_pretext, a_hat = model(x_in)
            out_pretext = out_pretext.float()        # loss math in fp32 (bf16 under autocast)
            if a_hat is not None:
                a_hat = a_hat.float()
            losses = compute_losses(
                out_pretext, a_hat, x_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
                pretext_style=pretext_style,
                patch_size=patch_size,
                input_mode=input_mode,
                ch_weight=ch_weight,
                atom_pos_weight=float(cfg.mae.get("atom_pos_weight", 1.0)),
                atom_pos_thresh=float(cfg.mae.get("atom_pos_thresh", 0.05)),
                with_gradmag=with_gradmag,
                gradmag_reconstruct=gradmag_reconstruct,
            )
            L_pretext = losses[pretext_key]
            L_str = losses["L_str"]
            loss = lambda_pretext * L_pretext + lambda_str * L_str
            if do_con:
                L_con = nt_xent(z_a.float(), z_b.float(), con_temp)
                loss = loss + lambda_con * L_con
                L_con_sum += L_con.item()
            if d2v_on:
                with torch.no_grad():
                    target = _unwrap(model_ema.module).data2vec_target(x_clean)   # (B, N, D)
                pm = voxel_mask_to_patch_target(
                    mask.to(out_pretext.dtype), patch_size).reshape(mask.shape[0], -1) > 0.5
                L_d2v = F.smooth_l1_loss(d2v_pred.float()[pm], target.float()[pm])
                loss = loss + lambda_d2v * L_d2v
                L_d2v_sum += L_d2v.item()
            loss.backward()
        if is_step:
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    _unwrap(model).parameters(), max_norm=grad_clip
                )
                grad_norm_sum += float(gn)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(ema_source_model if ema_source_model is not None else _unwrap(model))
            global_step += 1
        L_pretext_sum += L_pretext.item()
        L_str_sum += L_str.item()
        if "L_dens_atom" in losses:
            L_dens_atom_sum += losses["L_dens_atom"].item()
        if "L_dens_density" in losses:
            L_dens_density_sum += losses["L_dens_density"].item()
        if "L_dens_gradmag" in losses:
            L_dens_gradmag_sum += losses["L_dens_gradmag"].item()
        n_batches += 1
        if cfg.debug and i == 10:
            break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    recon_gradmag = with_gradmag and gradmag_reconstruct
    n_modalities = (layout["n_atom"] > 0) + (layout["n_density"] > 0) + (1 if recon_gradmag else 0)
    if n_modalities >= 2:
        if layout["n_atom"] > 0:
            metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        if layout["n_density"] > 0:
            metrics["L_dens_density"] = L_dens_density_sum / n_b
        if recon_gradmag:
            metrics["L_dens_gradmag"] = L_dens_gradmag_sum / n_b
    if grad_clip > 0:
        n_steps = max(1, n_batches // accum_steps)
        metrics["grad_norm"] = grad_norm_sum / n_steps
    if contrastive_on:
        metrics["L_con"] = L_con_sum / n_b
    if d2v_on:
        metrics["L_d2v"] = L_d2v_sum / n_b
    return metrics, global_step




# ── Main ──────────────────────────────────────────────────────────────────────


_METRIC_KEYS = ("L_recon", "L_pixel", "L_fourier", "mask_ratio")


def _build_chamae(cfg: DictConfig, device, layout=None, is_main: bool = True) -> DensityChaMAE:
    groups = tuple(int(c) for c in cfg.model.channel_groups)
    n_in = int(cfg.model.n_in_channels)
    assert sum(groups) == n_in, (
        f"channel_groups {groups} must sum to n_in_channels {n_in}"
    )
    cmax = cfg.model.get("channel_mask_max", None)
    model = DensityChaMAE(
        grid_dim=int(cfg.vox.grid_dim),
        patch_size=int(cfg.model.patch_size),
        channel_groups=groups,
        n_in_channels=n_in,
        n_channels=int(cfg.model.n_channels),
        dim=int(cfg.model.dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.heads),
        mlp_ratio=int(cfg.model.mlp_ratio),
        dropout=float(cfg.model.dropout),
        n_memory_tokens=int(cfg.model.get("n_memory_tokens", 4)),
        dec_dim=int(cfg.model.get("dec_dim", 256)),
        dec_depth=int(cfg.model.get("dec_depth", 2)),
        dec_heads=int(cfg.model.get("dec_heads", 8)),
        dec_mlp_ratio=int(cfg.model.get("dec_mlp_ratio", 4)),
        patch_mask_ratio=float(cfg.model.get("patch_mask_ratio", 0.75)),
        dcp_strategy=str(cfg.model.get("dcp_strategy", "alternate")),
        channel_mask_min=int(cfg.model.get("channel_mask_min", 1)),
        channel_mask_max=(int(cmax) if cmax is not None else None),
        lambda_fourier=float(cfg.model.get("lambda_fourier", 0.01)),
    ).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_enc = sum(p.numel() for p in model.encoder.parameters())
        logger.info(
            f"DensityChaMAE {(n_params/1e6):.02f}M params (encoder {(n_enc/1e6):.02f}M) — "
            f"groups={model.channel_groups}, n_memory_tokens={model.n_memory_tokens}, "
            f"dcp={model.dcp_strategy}, patch_mask_ratio={model.patch_mask_ratio}, "
            f"lambda_fourier={model.lambda_fourier}")
    return model


def train_epoch_cha(cfg, prefetcher, model, optimizer, model_ema, device,
                    global_step, ch_weight=None, ema_source_model=None):
    model.train()
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    channels_last = bool(cfg.get("channels_last", False))
    _, amp_ctx_fn = _amp_setup(cfg)
    sums = {k: 0.0 for k in _METRIC_KEYS}
    grad_norm_sum, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    # MAEPrefetcher yields (x_in, x_clean, mask, target_str, x_in_b) — ChA uses x_clean only.
    for i, (_x_in, x_clean, _mask, _target_str, _x_in_b) in enumerate(prefetcher):
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_iter)
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        if channels_last:
            x_clean = x_clean.to(memory_format=torch.channels_last_3d)
        with sync_ctx:
            with amp_ctx_fn():
                out = model(x_clean)
            loss = out["L_recon"]                  # already fp32 (recon loss casts internally)
            loss.backward()
        if is_step:
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    _unwrap(model).parameters(), max_norm=grad_clip)
                grad_norm_sum += float(gn)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(ema_source_model if ema_source_model is not None else _unwrap(model))
            global_step += 1
        for k in _METRIC_KEYS:
            sums[k] += float(out[k])
        n_batches += 1
        if cfg.debug and i == 10:
            break

    n_b = max(1, n_batches)
    metrics = {k: sums[k] / n_b for k in _METRIC_KEYS}
    metrics["loss"] = metrics["L_recon"]
    if grad_clip > 0:
        metrics["grad_norm"] = grad_norm_sum / max(1, n_batches // accum_steps)
    return metrics, global_step


def _assemble_x_clean(cfg, val_cache, start, bsz, device, input_mode, with_gradmag):
    """Assemble the clean multi-channel input from the precomputed val voxel cache
    (mirrors the MAEPrefetcher assembly, minus noise/masking)."""
    v_lig = val_cache["voxels_lig"][start:start + bsz].to(device, non_blocking=True)
    v_poc = val_cache["voxels_poc"][start:start + bsz].to(device, non_blocking=True)
    density_source = str(cfg.mae.get("density_source", "synthetic"))
    need_density = input_mode in (
        "density", "atomblob_density", "atomblob_merged_density", "roleblob_density",
    )
    d_clean = None
    if need_density:
        if density_source == "xray":
            d_clean = val_cache["xray_density"][start:start + bsz].to(device, non_blocking=True).unsqueeze(1)
        else:
            sigma_lo = cfg.mae.sigma_blur_a_lo / cfg.vox.resolution
            sigma_hi = cfg.mae.sigma_blur_a_hi / cfg.vox.resolution
            atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
            d_clean = per_sample_zscore(gaussian_blur3d(atoms_sum, (sigma_lo + sigma_hi) / 2))
    g_clean = None
    if with_gradmag:
        if density_source == "xray":
            g_clean = val_cache["xray_gradmag"][start:start + bsz].to(device, non_blocking=True).unsqueeze(1)
        else:
            g_clean = per_sample_zscore(gradient_magnitude3d(d_clean))

    if input_mode == "density":
        x = d_clean
    elif input_mode == "atomblob":
        x = torch.cat([v_lig, v_poc], dim=1)
    elif input_mode == "atomblob_density":
        x = torch.cat([v_lig, v_poc, d_clean], dim=1)
    elif input_mode == "atomblob_merged":
        x = _build_merged_atoms(v_lig, v_poc)
    elif input_mode == "atomblob_merged_density":
        x = torch.cat([_build_merged_atoms(v_lig, v_poc), d_clean], dim=1)
    elif input_mode == "roleblob":
        x = _build_role_atoms(v_lig, v_poc)
    elif input_mode == "roleblob_density":
        x = torch.cat([_build_role_atoms(v_lig, v_poc), d_clean], dim=1)
    else:
        raise RuntimeError(f"unexpected input_mode={input_mode!r}")
    if with_gradmag:
        x = torch.cat([x, g_clean], dim=1)
    return x


def val_epoch_cha(cfg, model, val_cache, device, ch_weight=None):
    """Deterministic recon-loss pass — DCP masking is fixed per batch via a
    re-seeded generator so val loss is comparable across epochs."""
    model.eval()
    _, amp_ctx_fn = _amp_setup(cfg)
    channels_last = bool(cfg.get("channels_last", False))
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    n = val_cache["voxels_lig"].shape[0]
    sums = {k: 0.0 for k in _METRIC_KEYS}
    n_b = 0
    gen = torch.Generator(device=device)
    with torch.no_grad():
        for bi, start in enumerate(range(0, n, cfg.bsz)):
            gen.manual_seed(bi + 1)
            x = _assemble_x_clean(cfg, val_cache, start, cfg.bsz, device, input_mode, with_gradmag)
            if channels_last:
                x = x.to(memory_format=torch.channels_last_3d)
            with amp_ctx_fn():
                out = model(x, generator=gen)
            for k in _METRIC_KEYS:
                sums[k] += float(out[k])
            n_b += 1
            if cfg.debug and n_b >= 5:
                break
    metrics = {k: sums[k] / max(1, n_b) for k in _METRIC_KEYS}
    metrics["loss"] = metrics["L_recon"]
    return metrics


def _build_vit(cfg: DictConfig, device, layout: dict, is_main: bool):
    """Build DensityViTMAE for the reconstruction/detection pretexts (mae | denoise | electra)."""
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    dual_head = bool(cfg.model.get("dual_head", False))
    if with_gradmag and dual_head:
        raise ValueError(
            "dual_head is not yet supported with with_gradmag (the dual head "
            "assumes a lone trailing density channel); use the single MAE head."
        )
    head_hidden_dim = int(cfg.model.get("head_hidden_dim", 0))
    head_depth = int(cfg.model.get("head_depth", 2))
    head_style = str(cfg.model.get("head_style", "conv"))
    splat_k = int(cfg.model.get("splat_k", 4))
    splat_aniso = bool(cfg.model.get("splat_aniso", False))
    pos_encoding = str(cfg.model.get("pos_encoding", "learnable"))
    n_in_cfg = int(cfg.model.get("n_in_channels", 1))
    if n_in_cfg != layout["n_in"]:
        raise RuntimeError(
            f"input_mode={input_mode!r} with_gradmag={with_gradmag} expects "
            f"model.n_in_channels={layout['n_in']}, got {n_in_cfg}"
        )
    # Learnable radius-conditioned atom embedding (opt-in, model.radius_embed_k>0): replaces
    # the one-hot atom channels with a learned φ(vdW-radius)→k feature per role. The encoder
    # reconstructs the RAW atom+density channels (n_recon unchanged); only its patch_embed
    # width/grouping change. Requires per-element atom channels (atomblob*), not roleblob.
    radius_embed = None
    rek = int(cfg.model.get("radius_embed_k", 0))
    if rek > 0:
        from voxbind.constants import vdw_radius
        if layout["n_atom"] <= 0:
            raise RuntimeError("radius_embed_k>0 requires per-element atom channels "
                               f"(atomblob* input_mode); got input_mode={input_mode!r}")
        LIG_EL = ["C", "O", "N", "S", "F", "Cl", "P"]
        POC_EL = ["C", "O", "N", "S"]
        n_lig = int(cfg.model.get("n_channels_ligand", 7))
        n_poc = int(cfg.model.get("n_channels_pocket", 4))
        def _radii(elems, n):
            return [vdw_radius(elems[i]) if i < len(elems) else vdw_radius("C")
                    for i in range(n)]
        n_pt = layout["n_in"] - layout["n_atom"]    # density (+ gradmag) passthrough
        scales_cfg = cfg.model.get("radius_embed_scales", None)
        scales = [float(s) for s in scales_cfg] if scales_cfg is not None else [0.0]
        radius_embed = {
            "lig_radii": _radii(LIG_EL, n_lig),
            "poc_radii": _radii(POC_EL, n_poc),
            "k": rek,
            "hidden": int(cfg.model.get("radius_embed_hidden", 16)),
            "n_passthrough": n_pt,
            "scales": scales,
        }
        if is_main:
            logger.info(f"radius-embed front-end: k={rek} scales={scales} "
                        f"lig_radii={radius_embed['lig_radii']} poc_radii={radius_embed['poc_radii']} "
                        f"n_pt={n_pt} → encoder patch_embed={2*rek+n_pt}ch groups=({rek},{rek},{n_pt}), "
                        f"recon={layout['n_recon']}ch")
    model = DensityViTMAE(
        grid_dim=int(cfg.vox.grid_dim),
        patch_size=int(cfg.model.patch_size),
        n_in_channels=n_in_cfg,
        n_recon_channels=layout["n_recon"],
        n_channels=int(cfg.model.n_channels),
        dim=int(cfg.model.dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.heads),
        mlp_ratio=int(cfg.model.mlp_ratio),
        dropout=float(cfg.model.dropout),
        n_struct_channels=int(cfg.model.n_struct_channels),
        # denoise reconstructs the clean signal -> same recon head as mae (the prefetcher
        # + loss still branch on the real pretext_style="denoise").
        pretext_style=("mae" if pretext_style == "denoise" else pretext_style),
        dual_head=dual_head,
        head_hidden_dim=head_hidden_dim,
        head_depth=head_depth,
        head_style=head_style,
        splat_k=splat_k,
        splat_aniso=splat_aniso,
        pos_encoding=pos_encoding,
        rope_fast=bool(cfg.model.get("rope_fast", False)),
        patch_embed_mode=str(cfg.model.get("patch_embed_mode", "fused")),
        channel_groups=(tuple(int(c) for c in cfg.model.channel_groups)
                        if cfg.model.get("channel_groups", None) else None),
        channel_group_dropout=float(cfg.model.get("channel_group_dropout", 0.0)),
        contrastive_dim=(int(cfg.mae.get("contrastive_dim", 128))
                         if float(cfg.mae.get("contrastive_weight", 0.0)) > 0.0 else 0),
        radius_embed=radius_embed,
        drop_path=float(cfg.model.get("drop_path", 0.0)),
        data2vec=(float(cfg.mae.get("data2vec_weight", 0.0)) > 0.0),
    ).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"DensityViTMAE has {(n_params/1e6):.02f}M parameters "
                    f"(pretext_style={pretext_style}, input_mode={input_mode}, "
                    f"with_gradmag={with_gradmag} (recon={gradmag_reconstruct}), "
                    f"n_in={layout['n_in']} n_recon={layout['n_recon']}, "
                    f"dual_head={dual_head}, head_style={head_style}, "
                    f"head_hidden_dim={head_hidden_dim or '(c_half)'}, head_depth={head_depth})")
    return model


def run(cfg: DictConfig, method: str) -> None:
    """Unified density pre-training loop for every MAE method (see MAE_METHODS).
    The method spec supplies the model builder, the train/val epoch fns, and a few
    flags; everything else (DDP, AMP, resume, EMA, val cache, checkpointing) is shared."""
    spec = MAE_METHODS[method]
    assert torch.cuda.is_available(), "GPU required."
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    start_epoch = 0
    # P1b/#2: normalize before create_exp_dir() writes cfg.yaml so downstream
    # tools that still consume the legacy top-level keys read the right layout.
    _reconcile_input_keys(cfg)
    # Pin ligand/pocket element-channel counts for _channel_layout (default 7/4; 8/4 for the
    # per-element "+other" corpus). Must precede any _channel_layout / model build below.
    set_atom_channels(int(cfg.model.get("n_channels_ligand", 7)), int(cfg.model.get("n_channels_pocket", 4)))
    # Pin the # of density sources (1 = 2Fo-Fc only; 2 = + mFo-DFc difference map). Read from
    # dset.density_sources (default 1 → all prior runs unaffected). Must precede _channel_layout.
    set_density_sources(int(cfg.dset.get("density_sources", 1)))
    create_exp_dir(cfg, write=is_main)
    if world_size > 1:
        dist.barrier()

    logger.info(f"DDP: world_size={world_size} rank={rank} local_rank={local_rank}")
    if is_main:
        logger.info(f"saving experiments in: {cfg.output_dir}")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed + rank)

    if cfg.resume is not None and os.path.isdir(cfg.resume):
        if is_main:
            logger.info(f"resuming from: {cfg.resume}")
        resume = cfg.resume
        wandb_override = cfg.wandb
        resume_epoch_override = cfg.resume_epoch
        num_epochs_override = cfg.num_epochs
        cfg = OmegaConf.load(os.path.join(cfg.resume, "cfg.yaml"))
        cfg.output_dir, cfg.resume = resume, resume
        cfg.wandb = wandb_override
        cfg.resume_epoch = resume_epoch_override
        cfg.num_epochs = num_epochs_override

    # P1b/#2: run again after resume reload so pre-P1b resumes are normalized too.
    _reconcile_input_keys(cfg)

    if is_main:
        logger.info("cfg:\n" + OmegaConf.to_yaml(cfg))

    if is_main and cfg.wandb:
        try:
            _tags = cfg.get("wandb_tags", None)
            # OmegaConf parses bare numeric values in CLI list overrides as ints,
            # while W&B requires every run tag to be a string.
            _tags = [str(tag) for tag in _tags] if _tags else None
            # Strip leading YYMMDD_ from the exp_name for the wandb display name.
            # exp_name on disk keeps the date (per the exp-dir naming convention);
            # only the wandb run name shows the bare topic.
            import re as _re
            _wandb_name = _re.sub(r"^\d{6}_", "", cfg.exp_name)
            wandb.init(
                project=cfg.get("wandb_project", "binding-affinity"),
                entity="eddy26",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=_wandb_name,
                tags=_tags,
                dir=cfg.output_dir,
                resume="allow",
                settings=wandb.Settings(code_dir=".", init_timeout=300),
            )
        except Exception as e:
            logger.warning(f"wandb.init failed (continuing without): {e}")

    # Data
    loader_train, loader_val, _ = create_dataloaders(
        cfg, distributed=True, rank=rank, world_size=world_size,
    )
    n_train, n_val = len(loader_train.dataset), len(loader_val.dataset)
    if is_main:
        accum = max(1, int(cfg.get("accum_steps", 1)))
        logger.info(f"train/val set size: {n_train}/{n_val}")
        logger.info(f"train batches per rank: {len(loader_train)} "
                    f"(effective batch = {cfg.bsz * world_size * accum})")

    # Model — pure 3D ViT encoder + pretext head (density/RTD) + optional structure head.
    # `n_in_channels` widens the patch embed + MAE recon head based on input_mode:
    #   density=1, atomblob=11, atomblob_density=12.
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    gradmag_noise = bool(cfg.mae.get("gradmag_noise", False))
    gradmag_channel_weight = float(cfg.mae.get("gradmag_channel_weight", 1.0))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             density_input=bool(cfg.mae.get("density_input", True)),
                             mask_as_channel=bool(cfg.mae.get("mask_as_channel", False)))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    model = spec.build_model(cfg, device, layout, is_main)
    # channels_last_3d: better cuDNN Conv3d kernels for the patch-embed/heads.
    # Applied before optimizer/compile/DDP/EMA so all downstream copies inherit it.
    if bool(cfg.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last_3d)
        if is_main:
            logger.info("channels_last_3d memory format enabled")

    # optimizer.type=muon → Muon (Newton-Schulz orthogonalized momentum) on the transformer
    # BLOCK weight matrices only (2D params under '.blocks.': attn qkv/proj + mlp), with AdamW
    # as the auxiliary optimizer for everything else (patch-embed conv, pos/group/memory embeds,
    # every norm/bias, and the reconstruction head). Correct under DDP: each rank all-reduces
    # grads in backward, so identical NS on identical grads → identical updates → params stay
    # in sync (no distributed-Muon code path, which is the buggy part). Constant LR is enforced
    # for muon (the two groups keep their own base LR; the epoch scheduler would clobber them).
    _opt_type = str(cfg.get("optimizer", {}).get("type", "adamw")).lower()
    using_muon = _opt_type == "muon"
    if using_muon:
        from voxbind.models.muon import SingleDeviceMuonWithAuxAdam
        muon_lr  = float(cfg.get("optimizer", {}).get("muon_lr", 0.02))
        muon_mom = float(cfg.get("optimizer", {}).get("muon_momentum", 0.95))
        muon_p, adam_p = [], []
        for _n, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            # transformer block matrices only: 2D weights whose name contains 'blocks.'
            # (robust to prefix, e.g. 'encoder.blocks.0.attn.qkv.weight'). Head Linears
            # (decoder_proj/recon_mlp), embeds, norms, biases, patch-embed conv → AdamW.
            (muon_p if (_p.ndim == 2 and "blocks." in _n) else adam_p).append(_p)
        # Fail loud rather than silently training the bulk on AdamW (a classic Muon-integration bug).
        assert len(muon_p) > 0, ("Muon selected 0 matrices — no 2D param name contains 'blocks.'; "
                                 "check the model's parameter names before running.")
        optimizer = SingleDeviceMuonWithAuxAdam([
            dict(params=muon_p, use_muon=True,  lr=muon_lr, momentum=muon_mom, weight_decay=cfg.wd),
            dict(params=adam_p, use_muon=False, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.wd),
        ])
        if is_main:
            _nm = sum(p.numel() for p in muon_p) / 1e6
            _na = sum(p.numel() for p in adam_p) / 1e6
            logger.info(f"Muon+AuxAdam: {len(muon_p)} block matrices ({_nm:.1f}M) @ muon_lr={muon_lr} "
                        f"mom={muon_mom}; {len(adam_p)} aux tensors ({_na:.1f}M) @ AdamW lr={cfg.lr}; "
                        f"wd={cfg.wd}. Constant LR (epoch scheduler skipped for muon).")
    elif bool(cfg.get("optimizer", {}).get("fused", False)):
        # CUDA-fused multi-tensor AdamW (one kernel for the whole update).
        # Faithful drop-in for the vendored AdamW: keep its non-standard betas=(0.99,0.999),
        # but use eps=1e-8 instead of its eps=0 — torch's FUSED kernel returns nan at eps=0,
        # and since v_hat >> 1e-8 the change is numerically negligible (Δw ~ 4e-4 over 4 steps,
        # below the conv/attn backward nondeterminism floor).
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
                                      betas=(0.99, 0.999), eps=1e-8, fused=True)
        if is_main:
            logger.info("fused AdamW enabled (betas=(0.99,0.999), eps=1e-8 ~ vendored AdamW)")
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()

    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    # Resume model+optimizer state (before DDP wrap)
    global_step = 0
    if cfg.resume is not None:
        ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pth.tar")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict_ema", ckpt.get("state_dict"))
            sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            if is_main:
                logger.info(f"resumed from epoch {start_epoch - 1} (global_step={global_step})")
            if cfg.resume_epoch is not None:
                start_epoch = cfg.resume_epoch

    model_train = maybe_compile_model(model, cfg, is_main=is_main)
    if world_size > 1:
        # NB ChannelViT HCS drops whole groups, but `embed_groups` runs every group's
        # patch-embed conv BEFORE the drop-slice, so every param still gets a (zero)
        # grad each step → no unused params, find_unused_parameters stays False.
        model_train = torch.nn.parallel.DistributedDataParallel(
            model_train,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=spec.ddp_find_unused,
            gradient_as_bucket_view=True,
        )
    model_ema = ModelEma(model, decay=cfg.mae.ema_decay,
                         foreach=bool(cfg.get("ema", {}).get("foreach", False)))

    # Val cache
    if is_main:
        val_cache = precompute_val(loader_val, voxelizer, cfg)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        val_cache = precompute_val(loader_val, voxelizer, cfg)

    # Per-channel loss weighting.
    # Two orthogonal knobs that compose into a single 11- or 12-vector:
    #   mae.channel_weighting : 'uniform' | 'inv_sqrt_freq'    (atom-channel side)
    #   mae.density_channel_weight : float, default 1.0        (density-channel side,
    #                                                           only used when
    #                                                           input_mode='atomblob_density')
    # When both are at defaults (`uniform` + `1.0`) and no warm-up is set,
    # `build_ch_weight` stays None and compute_losses takes the unweighted code
    # path — preserving baseline behavior. When anything is non-default for
    # atomblob_density, the final 12-vec is renormalized to sum=n_channels so the
    # overall L_pretext scale matches the unweighted case (gradient share shifts;
    # total mass preserved). A non-zero mae.density_weight_warmup_epochs makes the
    # density+gradmag entries time-varying (cosine 0 -> target), rebuilt each epoch.
    # `build_ch_weight(epoch)` is set below: either a closure returning the (possibly
    # warm-up-scaled) weight vector, or None to take the unweighted code path.
    build_ch_weight = None
    dens_warmup_epochs = 0
    if spec.uses_ch_weight:
        ch_weighting = str(cfg.mae.get("channel_weighting", "uniform"))
        if ch_weighting not in ("uniform", "inv_sqrt_freq", "inv_freq"):
            raise ValueError(
                f"unknown channel_weighting={ch_weighting!r}; expected "
                f"'uniform', 'inv_sqrt_freq', or 'inv_freq'"
            )
        # Exponent on (1/freq): inv_sqrt_freq → 0.5 (gentle), inv_freq → 1.0 (strong).
        freq_power = 1.0 if ch_weighting == "inv_freq" else 0.5
        # Optional cap on the rare/common atom-weight ratio (0 = off); floors crushed
        # channels so 1/freq doesn't collapse voxel-dense pocket atoms to ~0.01.
        channel_weight_clip = float(cfg.mae.get("channel_weight_clip_ratio", 0.0))
        density_channel_weight = float(cfg.mae.get("density_channel_weight", 1.0))
        needs_atom_weights = ch_weighting in ("inv_sqrt_freq", "inv_freq")
        _atomblob_modes = ("atomblob", "atomblob_density", "atomblob_merged", "atomblob_merged_density")
        # roleblob_density carries a density channel → density_channel_weight must apply.
        # (roleblob atom channels use uniform weighting; inv_freq on 2 role channels is
        # intentionally not in _atomblob_modes so it raises rather than mis-counting 7/11.)
        _density_modes = ("atomblob_density", "atomblob_merged_density", "roleblob_density")
        _merged_modes  = ("atomblob_merged", "atomblob_merged_density")
        needs_dens_downweight = (
            input_mode in _density_modes and density_channel_weight != 1.0
        )
        # gradmag weight only enters the loss when gradmag is a reconstruction target.
        needs_gradmag_weight = (
            with_gradmag and gradmag_reconstruct and gradmag_channel_weight != 1.0
        )
        # Optional cosine warm-up on the density+gradmag reconstruction supervision:
        # their channel weights ramp 0 -> their configured targets over the first
        # `density_weight_warmup_epochs` epochs (cosine), then hold. 0 = off (default).
        # Because the weight vector is renormalized to sum=n_recon, scale=0 puts the
        # full loss mass on the atom channels -> "pure atomblob MAE first, then fold
        # in density/gradmag". Anchored at absolute epoch 0 so resume is consistent.
        dens_warmup_epochs = int(cfg.mae.get("density_weight_warmup_epochs", 0))
        has_aux_recon = (
            (input_mode in _density_modes) or (with_gradmag and gradmag_reconstruct)
        )
        needs_warmup = dens_warmup_epochs > 0 and has_aux_recon
        if needs_atom_weights or needs_dens_downweight or needs_gradmag_weight or needs_warmup:
            if needs_atom_weights and input_mode not in _atomblob_modes:
                raise ValueError(
                    f"channel_weighting={ch_weighting!r} requires an atomblob* input_mode; "
                    f"got input_mode={input_mode!r}"
                )
            if needs_dens_downweight and input_mode not in _density_modes:
                raise ValueError(
                    f"density_channel_weight={density_channel_weight} is only meaningful "
                    f"with a *_density input_mode; got input_mode={input_mode!r}"
                )

            merge_lig_poc = input_mode in _merged_modes
            n_atom = layout["n_atom"]          # 0 (density-only) / 7 (merged) / 11
            n_density = layout["n_density"]

            # Atom-side weights: inv-sqrt-freq (rank 0 computes + caches; all ranks
            # load from disk) OR uniform ones (length-0 when there are no atom channels).
            if needs_atom_weights:
                if is_main:
                    atom_w = precompute_channel_weights(
                        loader_train.dataset, voxelizer, cfg, device=device,
                        merge_lig_poc=merge_lig_poc,
                        power=freq_power, clip_ratio=channel_weight_clip,
                    )
                if world_size > 1:
                    dist.barrier()
                if not is_main:
                    atom_w = precompute_channel_weights(
                        loader_train.dataset, voxelizer, cfg, device=device,
                        merge_lig_poc=merge_lig_poc,
                        power=freq_power, clip_ratio=channel_weight_clip,
                    )
            else:
                atom_w = torch.ones(n_atom)

            # Reconstructed-channel weight vector [atoms, density?, gradmag?]; length
            # = n_recon (gradmag excluded when input-only). Renormalized to sum =
            # n_recon so L_pretext magnitude matches the unweighted case; inter-channel
            # gradient ratios are preserved. The density+gradmag entries are scaled by
            # `aux_scale` (1.0 unless the warm-up is active) so the vector can be rebuilt
            # cheaply each epoch without recomputing the (cached) atom-side weights.
            def _aux_scale(epoch: int) -> float:
                if dens_warmup_epochs <= 0:
                    return 1.0
                p = min(1.0, max(0.0, float(epoch) / float(dens_warmup_epochs)))
                return 0.5 * (1.0 - math.cos(math.pi * p))      # cosine 0 -> 1

            def build_ch_weight(epoch: int):
                scale = _aux_scale(epoch)
                parts = [atom_w]
                # One (density, gradmag) weight pair per density SOURCE, interleaved to match
                # the [ …atoms…, dens0, grad0, dens1, grad1 ] channel order. n_density_src=1 →
                # single pair (identical to the pre-diff single-2Fo-Fc vector).
                for _src in range(layout["n_density_src"]):
                    if n_density > 0:
                        parts.append(torch.tensor([density_channel_weight * scale]))
                    if with_gradmag and gradmag_reconstruct:
                        parts.append(torch.tensor([gradmag_channel_weight * scale]))
                raw = torch.cat(parts, dim=0)
                assert raw.shape[0] == layout["n_recon"], (
                    f"weight vector length {raw.shape[0]} != n_recon {layout['n_recon']}"
                )
                denom = float(raw.sum())
                if denom <= 0.0:        # degenerate: all channels at 0 (e.g. density-only + scale 0)
                    denom = float(raw.shape[0])
                    raw = torch.ones_like(raw)
                return (raw * (float(layout["n_recon"]) / denom)).to(device), scale

            if is_main:
                logger.info(
                    f"using channel weights on {layout['n_recon']} channels  "
                    f"channel_weighting={ch_weighting}  channel_weight_clip_ratio={channel_weight_clip or 'off'}  "
                    f"density_channel_weight={density_channel_weight}  "
                    f"gradmag_channel_weight={gradmag_channel_weight}  merge_lig_poc={merge_lig_poc}  "
                    f"density_weight_warmup_epochs={dens_warmup_epochs or 'off'}"
                )

    sigma_blur_vox_range = (
        cfg.mae.sigma_blur_a_lo / cfg.vox.resolution,
        cfg.mae.sigma_blur_a_hi / cfg.vox.resolution,
    )

    if is_main:
        logger.info(f"start {spec.label} pre-training...")

    # ── Opt-in optimizer-LR schedule (warmup + cosine decay). Default 'constant' keeps
    #    the constant cfg.lr behavior BIT-IDENTICAL (all prior runs unaffected). Applied
    #    per-epoch to every optimizer param group. Knobs (all via Hydra, no yaml edit):
    #      cfg.lr_schedule : 'constant' (default) | 'cosine'
    #      cfg.lr_warmup_epochs : linear 0 -> cfg.lr over N epochs (default 0)
    #      cfg.lr_min : cosine floor (default 0.0);  peak = cfg.lr
    #    On resume the cosine spans [start_epoch, start_epoch+num_epochs) so extend-runs
    #    decay over their own window.
    _lr_sched = str(cfg.get("lr_schedule", "constant"))
    _lr_warmup = int(cfg.get("lr_warmup_epochs", 0))
    _lr_min = float(cfg.get("lr_min", 0.0))
    _lr_base = float(cfg.lr)
    _lr_total = start_epoch + int(cfg.num_epochs)

    def _lr_at_epoch(ep):
        if _lr_sched == "constant":
            return _lr_base
        if _lr_warmup > 0 and ep < start_epoch + _lr_warmup:
            return _lr_base * float(ep - start_epoch + 1) / float(_lr_warmup)   # linear warmup
        prog = float(ep - start_epoch - _lr_warmup) / float(max(1, _lr_total - start_epoch - _lr_warmup))
        prog = min(1.0, max(0.0, prog))
        return _lr_min + 0.5 * (_lr_base - _lr_min) * (1.0 + math.cos(math.pi * prog))  # cosine

    epochs_iter = tqdm(
        range(start_epoch, start_epoch + cfg.num_epochs),
        desc="Epochs", disable=not is_main,
    )
    ckpt_saver = AsyncCheckpointSaver() if is_main else None

    for epoch in epochs_iter:
        t0 = time.time()
        if _lr_sched != "constant" and not using_muon:
            _cur_lr = _lr_at_epoch(epoch)
            for _pg in optimizer.param_groups:
                _pg["lr"] = _cur_lr
            if is_main:
                logger.info(f"[epoch {epoch}] lr={_cur_lr:.3e} "
                            f"(schedule={_lr_sched}, warmup={_lr_warmup}ep, "
                            f"peak={_lr_base:.2e}, min={_lr_min:.1e})")
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # (re)build per-channel loss weights for this epoch; aux_scale < 1 only
        # while the density/gradmag warm-up is ramping (constant otherwise).
        if build_ch_weight is not None:
            ch_weight, aux_scale = build_ch_weight(epoch)
            if is_main and dens_warmup_epochs > 0 and epoch <= dens_warmup_epochs:
                n_atom_ = layout["n_atom"]
                msg = f"[epoch {epoch}] density/gradmag warm-up scale={aux_scale:.4f}"
                if layout["n_density"] > 0:
                    msg += f"  eff density={float(ch_weight[n_atom_]):.4f}"
                if with_gradmag and gradmag_reconstruct:
                    msg += f"  eff gradmag={float(ch_weight[n_atom_ + layout['n_density']]):.4f}"
                logger.info(msg)
        else:
            ch_weight, aux_scale = None, 1.0

        if pretext_style == "electra":
            corruption_ops = tuple(cfg.electra.corruption_ops)
            corruption_op_weights = tuple(
                cfg.electra.get("corruption_op_weights", [1.0] * len(corruption_ops))
            )
        else:
            corruption_ops = ("swap",)        # unused in mae mode
            corruption_op_weights = (1.0,)
        prefetch_pretext = "mae" if method == "chamae" else pretext_style
        prefetcher = MAEPrefetcher(
            loader_train, voxelizer,
            sigma_blur_vox_range=sigma_blur_vox_range,
            sigma_noise=cfg.mae.sigma_noise,
            block_size=cfg.mae.block_size,
            mask_ratio=cfg.mae.mask_ratio,
            pretext_style=prefetch_pretext,
            corruption_ops=corruption_ops,
            corruption_op_weights=corruption_op_weights,
            density_source=str(cfg.mae.get("density_source", "synthetic")),
            input_mode=input_mode,
            mask_strategy=str(cfg.mae.get("mask_strategy", "uniform")),
            mask_atom_tau=float(cfg.mae.get("mask_atom_tau", 1.0)),
            mask_n_seeds=int(cfg.mae.get("mask_n_seeds", 4)),
            mask_ratio_min=cfg.mae.get("mask_ratio_min", None),
            mask_ratio_max=cfg.mae.get("mask_ratio_max", None),
            modal_mask_prob=float(cfg.mae.get("modal_mask_prob", 0.0)),
            density_visible=bool(cfg.mae.get("density_visible", False)),
            lig_mask_thresh=float(cfg.mae.get("lig_mask_thresh", 0.1)),
            pocket_ed_mask=str(cfg.mae.get("pocket_ed_mask", "ligand_footprint")),
            mask_as_channel=bool(cfg.mae.get("mask_as_channel", False)),
            apo_prob=float(cfg.mae.get("apo_prob", 0.0)),
            contrastive=(float(cfg.mae.get("contrastive_weight", 0.0)) > 0.0
                         and prefetch_pretext == "mae"),
            with_gradmag=with_gradmag,
            gradmag_reconstruct=gradmag_reconstruct,
            gradmag_noise=gradmag_noise,
            density_input=bool(cfg.mae.get("density_input", True)),
            n_lig_ch=int(cfg.model.get("n_channels_ligand", 7)),
            n_poc_ch=int(cfg.model.get("n_channels_pocket", 4)),
        )
        train_metrics, global_step = spec.train_epoch(
            cfg, prefetcher, model_train, optimizer, model_ema, device, global_step,
            ch_weight=ch_weight, ema_source_model=model,
        )

        val_metrics = None
        if epoch % max(1, int(cfg.mae.val_every)) == 0:
            val_metrics = spec.val_epoch(cfg, model_ema.module, val_cache, device,
                                    ch_weight=ch_weight)

        if is_main:
            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb and wandb.run is not None:
                log_payload = {"train": train_metrics, "epoch": epoch}
                if val_metrics is not None:
                    log_payload["val"] = val_metrics
                if dens_warmup_epochs > 0:
                    log_payload["aux_weight_scale"] = aux_scale
                try:
                    wandb.log(log_payload, step=global_step)
                except Exception as e:
                    logger.warning(f"wandb.log failed: {e}")

            encoder_sd = {
                k: v for k, v in model_ema.module.state_dict().items()
                if k.startswith("encoder.")
            }
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "metrics": {"train": train_metrics, "val": val_metrics},
                "cfg": cfg,
                "state_dict_ema": model_ema.module.state_dict(),
                "encoder_state_dict_ema": encoder_sd,
                "optimizer": optimizer.state_dict(),
            }
            ckpt_saver.save(state, save_dir=cfg.output_dir)
            ckpt_every = max(1, int(cfg.mae.ckpt_every))
            # skip the epoch-0 tagged checkpoint (epoch-0 init weights are never used;
            # the per-epoch "latest" checkpoint.pth.tar above still covers resume).
            if epoch > 0 and ((epoch % ckpt_every == 0) or (epoch == cfg.num_epochs - 1)):
                ckpt_saver.save(state, save_dir=cfg.output_dir,
                                chkp_name=f"checkpoint_e{epoch:04d}.pth.tar")

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        torch.cuda.empty_cache()

    if is_main and ckpt_saver is not None:
        ckpt_saver.wait()
    if world_size > 1:
        dist.barrier()
    _cleanup_ddp()


# ── Density-MAE method registry ───────────────────────────────────────────────
# A "method" = one self-supervised pretext for the density encoder. Each entry binds a
# model builder + train/val epoch fns + a couple of flags; run() is the one shared loop.
#   mae      block-mask -> reconstruct masked density
#   denoise  add noise  -> recover the clean density (recovery-from-noise)
#   electra  replaced-token detection (kept; not used currently)
#   chamae   ChannelViT cross-modal completion (ChA-MAEViT; token-drop, its own epoch fns)
@dataclass(frozen=True)
class MaeMethod:
    label: str
    build_model: Callable
    train_epoch: Callable
    val_epoch: Callable
    ddp_find_unused: bool = False
    uses_ch_weight: bool = False


MAE_METHODS = {
    "mae":     MaeMethod("density-ViT-MAE",     _build_vit,    train_epoch,     val_epoch,     uses_ch_weight=True),
    "denoise": MaeMethod("density-ViT-denoise", _build_vit,    train_epoch,     val_epoch,     uses_ch_weight=True),
    "electra": MaeMethod("density-ViT-ELECTRA", _build_vit,    train_epoch,     val_epoch,     uses_ch_weight=True),
    "chamae":  MaeMethod("ChA-MAEViT",          _build_chamae, train_epoch_cha, val_epoch_cha, ddp_find_unused=True),
}


def resolve_method(cfg) -> str:
    """Explicit cfg.mae.method wins; else fall back to legacy arch / pretext_style."""
    if cfg.mae.get("method", None):
        return str(cfg.mae.method)
    if str(cfg.model.get("arch", "vit_mae")) == "cha_mae":
        return "chamae"
    return str(cfg.mae.get("pretext_style", "mae"))


@hydra.main(config_path="configs", config_name="config_train_density_vit_mae", version_base=None)
def main(cfg: DictConfig) -> None:
    method = resolve_method(cfg)
    if method not in MAE_METHODS:
        raise ValueError(f"unknown MAE method {method!r}; available: {sorted(MAE_METHODS)}")
    run(cfg, method)


if __name__ == "__main__":
    main()
