"""train_density_vit_mae.py — DDP MAE/ELECTRA pre-training for the ViT density encoder.

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

Checkpoint layout matches `train_density_mae.py` exactly:
  state_dict_ema          full DensityViTMAE EMA
  encoder_state_dict_ema  encoder.* slice — drops into VoxBind.density_encoder
                          (with `density_encoder_type=vit`) via the existing
                          loader in `models/__init__.py`.

Launch
------
    cd voxbind
    CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --standalone --nproc_per_node=5 \\
        train_density_vit_mae.py --config-name=config_train_density_vit_electra \\
        exp_name=260524_density_vit_electra_pretrain \\
        output_dir=exps/260524_density_vit_electra_pretrain
"""

import copy
import hashlib
import json
import logging
import os
import threading
import time
from datetime import timedelta

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from voxbind.dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.density_mae import (
    corrupt_noise, corrupt_reblur, corrupt_swap,
    gaussian_blur3d, gradient_magnitude3d, make_atom_biased_block_mask,
    make_block_mask, per_sample_zscore, voxel_mask_to_patch_target,
)
from voxbind.models.density_vit import DensityViTMAE
from voxbind.models.ema import ModelEma
from voxbind.utils.base_utils import create_exp_dir, makedir, seed_everything
from voxbind.voxelizer import Voxelizer

logger = logging.getLogger("train-density-vit-mae")


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _setup_ddp() -> tuple:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=2)
    )
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class AsyncCheckpointSaver:
    """Atomic + threaded checkpoint writer (rank 0 only). Mirrors train_ddp.py."""

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()

    @staticmethod
    def _write(state: dict, save_dir: str, chkp_name: str) -> None:
        tmp = os.path.join(save_dir, chkp_name + ".tmp")
        final = os.path.join(save_dir, chkp_name)
        try:
            torch.save(state, tmp)
            os.replace(tmp, final)
        except Exception as e:
            logger.warning(f"checkpoint write failed: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def save(self, state: dict, save_dir: str, chkp_name: str = "checkpoint.pth.tar") -> None:
        self.wait()
        frozen = copy.deepcopy(state)
        with self._lock:
            self._thread = threading.Thread(
                target=self._write,
                args=(frozen, save_dir, chkp_name),
                daemon=False,
            )
            self._thread.start()

    def wait(self) -> None:
        with self._lock:
            t = self._thread
        if t is not None:
            t.join()


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


_VALID_INPUT_MODES = (
    "density",
    "atomblob",
    "atomblob_density",
    "atomblob_merged",            # 7 atom ch: pocket C/O/N/S folded into ligand C/O/N/S
    "atomblob_merged_density",    # 8 ch: merged-7 atoms + 1 density
)


def _build_merged_atoms(v_lig: torch.Tensor, v_poc: torch.Tensor) -> torch.Tensor:
    """Fold pocket atoms into the first 4 channels of the ligand 7-channel tensor.

    CrossDocked element vocab is C/O/N/S/F/Cl/P for ligand and C/O/N/S for
    pocket (pocket has no F/Cl/P). After this op every channel encodes
    'atoms-of-element-X' regardless of source — the lig-vs-pocket
    disambiguation drops out of the pretext.
    """
    # v_lig (B, 7, G³)  +  v_poc (B, 4, G³) summed into channels [0..4)
    v_merged = v_lig.clone()
    v_merged[:, :4] = v_merged[:, :4] + v_poc
    return v_merged                                                 # (B, 7, G³)


def _channel_layout(
    input_mode: str,
    with_gradmag: bool = False,
    gradmag_reconstruct: bool = True,
) -> dict:
    """Single source of truth for the multi-channel input / reconstruction layout.

    Channel order is always  [ …atoms (n_atom)…, density (n_density), gradmag (n_gradmag) ].

      n_in     : encoder patch-embed input width  (atoms + density + gradmag)
      n_recon  : MAE head / loss-target width.  Equals n_in when gradmag is a
                 reconstruction target; n_in − n_gradmag when gradmag is
                 input-only (encoded as context but not predicted).

    `density_idx` / `gradmag_idx` are the channel offsets of those trailing
    channels (gradmag_idx is None when with_gradmag is False).
    """
    if input_mode == "density":
        n_atom = 0
    elif input_mode in ("atomblob", "atomblob_density"):
        n_atom = 11
    elif input_mode in ("atomblob_merged", "atomblob_merged_density"):
        n_atom = 7
    else:
        raise RuntimeError(f"unexpected input_mode={input_mode!r}")

    has_density = input_mode in ("density", "atomblob_density", "atomblob_merged_density")
    n_density = 1 if has_density else 0
    if with_gradmag and not has_density:
        raise ValueError(
            f"with_gradmag=True requires a density-bearing input_mode "
            f"(density / *_density); got input_mode={input_mode!r}"
        )
    n_gradmag = 1 if with_gradmag else 0
    n_in = n_atom + n_density + n_gradmag
    n_recon = n_in - (0 if gradmag_reconstruct else n_gradmag)
    return {
        "n_atom": n_atom,
        "n_density": n_density,
        "n_gradmag": n_gradmag,
        "n_in": n_in,
        "n_recon": n_recon,
        "density_idx": n_atom,
        "gradmag_idx": (n_atom + n_density) if with_gradmag else None,
    }


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
        with_gradmag: bool = False,
        gradmag_reconstruct: bool = True,
        gradmag_noise: bool = False,
    ):
        assert pretext_style in ("mae", "electra")
        assert density_source in ("synthetic", "xray"), (
            f"density_source={density_source!r}; expected 'synthetic' or 'xray'"
        )
        assert input_mode in _VALID_INPUT_MODES, (
            f"input_mode={input_mode!r}; expected one of {_VALID_INPUT_MODES}"
        )
        assert mask_strategy in ("uniform", "atom_biased"), (
            f"mask_strategy={mask_strategy!r}; expected 'uniform' or 'atom_biased'"
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
        self.generator = generator
        self.pretext_style = pretext_style
        self.corruption_ops = tuple(corruption_ops)
        self.corruption_op_weights = tuple(corruption_op_weights)
        self.density_source = density_source
        self.input_mode = input_mode
        self.mask_strategy = mask_strategy
        self.mask_atom_tau = mask_atom_tau
        self.with_gradmag = with_gradmag
        self.gradmag_reconstruct = gradmag_reconstruct
        self.gradmag_noise = gradmag_noise
        self.layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
        self.stream = torch.cuda.Stream()

    def _step(self, batch):
        device = self.voxelizer.device
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                v_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                v_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)

                # Density is only synthesized / loaded when the input actually
                # needs it. atomblob* modes skip this entirely — no I/O via the
                # xray path, no blur on the synthetic path.
                need_density = self.input_mode in (
                    "density", "atomblob_density", "atomblob_merged_density",
                )
                # atom-biased mask needs the all-channel atom sum even in atomblob mode
                need_atoms_sum = need_density or self.mask_strategy == "atom_biased"
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

                # Gradient-magnitude channel ‖∇ρ‖. xray: precomputed in the
                # dataset from the aligned crop (post-augmentation). synthetic:
                # derived on-GPU from the clean density. Appended as trailing ch.
                g_clean = None
                if self.with_gradmag:
                    if self.density_source == "xray":
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
                else:
                    raise RuntimeError(f"unexpected input_mode={self.input_mode!r}")

                # Append gradmag as the trailing channel: [ …atoms…, density, gradmag ].
                if self.with_gradmag:
                    x_clean = torch.cat([x_clean, g_clean], dim=1)

                # Noise is a density-domain augmentation applied to the density
                # channel BY INDEX (gradmag may trail it). Atom-only modes skip it;
                # gradmag is noised only when gradmag_noise is set.
                atom_only = self.input_mode in ("atomblob", "atomblob_merged")
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
                if self.mask_strategy == "atom_biased":
                    mask = make_atom_biased_block_mask(
                        atoms_sum, self.block_size, self.mask_ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                else:
                    mask = make_block_mask(B, G, self.block_size, self.mask_ratio, device)
                # mask: (B, 1, G, G, G) — broadcasts across all input channels

                if self.pretext_style == "mae":
                    x_in = x_noisy * (~mask).to(x_noisy.dtype)
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
        return x_in, x_clean, mask, target_str

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

def _val_cache_path(cfg) -> str:
    key = {
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "cubes_around": cfg.vox.cubes_around,
        "ligand_radius": cfg.dset.ligand_radius,
        "pocket_radius": cfg.dset.pocket_radius,
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        "density_source": cfg.mae.get("density_source", "synthetic"),
        "input_mode": cfg.get("input_mode", "density"),
        # with_gradmag adds an xray_gradmag tensor to the cache (xray source),
        # so it must key the cache to avoid reusing a gradmag-less blob.
        "with_gradmag": cfg.get("with_gradmag", False),
        # `density_mae_v1` is kept identical to cnn-MAE — the cached tensors are
        # just voxelized ligand+pocket (and optionally xray), no encoder-specific
        # content; input_mode only enters the assembly at val_epoch time.
        "task": "density_mae_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_mae_{h}.pt")


def precompute_val(loader_val, voxelizer, cfg) -> dict:
    path = _val_cache_path(cfg)
    if os.path.isfile(path):
        logger.info(f"loading pre-computed val voxels from {path}")
        return torch.load(path, weights_only=True)

    density_source = str(cfg.mae.get("density_source", "synthetic"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    cache_gradmag = with_gradmag and density_source == "xray"
    logger.info(
        f"pre-computing val voxels (density_source={density_source}, "
        f"with_gradmag={with_gradmag})..."
    )
    all_lig, all_poc, all_xray, all_gradmag = [], [], [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=7).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=4).cpu())
            if density_source == "xray":
                all_xray.append(batch["xray_density"].cpu())
            if cache_gradmag:
                all_gradmag.append(batch["xray_gradmag"].cpu())
    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    if density_source == "xray":
        cache["xray_density"] = torch.cat(all_xray, 0)
    if cache_gradmag:
        cache["xray_gradmag"] = torch.cat(all_gradmag, 0)
    torch.save(cache, path)
    logger.info(f"saved val voxels to {path}")
    return cache


# ── Channel-frequency cache + inv-sqrt-freq weights ──────────────────────────

_CH_FREQ_POS_THRESH = 0.05            # voxel "atom-present" threshold (matches struct head)
_CH_FREQ_DEFAULT_N_SAMPLES = 2000     # samples scanned for the one-time precompute


def _ch_freq_cache_path(cfg, n_samples: int, merge_lig_poc: bool = False) -> str:
    key = {
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        # Blob radii set per-channel atom-positive-voxel frequencies, so they
        # must key the cache — else an element-wise-ligand run would reuse a
        # uniform-radius freq blob and get wrong inv-sqrt-freq weights.
        "ligand_radius": cfg.dset.get("ligand_radius", 0.5),
        "pocket_radius": cfg.dset.get("pocket_radius", -1),
        "n_samples": n_samples,
        "pos_thresh": _CH_FREQ_POS_THRESH,
        "merge_lig_poc": merge_lig_poc,
        "task": "channel_freq_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"channel_freq_{h}.pt")


def precompute_channel_weights(
    train_dataset, voxelizer, cfg, device,
    n_samples: int = _CH_FREQ_DEFAULT_N_SAMPLES,
    merge_lig_poc: bool = False,
    power: float = 0.5,
    clip_ratio: float = 0.0,
) -> torch.Tensor:
    """Per-channel atom-positive-voxel frequencies → inverse-frequency weights
    `w = (1 / freq) ** power`, normalized so sum = n_atom_channels (= 11 when
    merge_lig_poc=False; = 7 when True) so the overall loss scale matches the
    unweighted case.

        power = 0.5 → 1/sqrt(freq)  (gentle rebalancing)
        power = 1.0 → 1/freq        (strong: rare atoms much heavier)

    `clip_ratio > 0` caps the max/min weight ratio by raising the floor to
    `w.max() / clip_ratio` BEFORE normalization, so the rarest atom can't
    out-weight the most common one by more than `clip_ratio`×. Useful with
    1/freq, where voxel-dense pocket atoms would otherwise collapse to ~0.01.

    The per-channel frequencies are cached (keyed on dataset / grid / radius /
    merge — NOT on power or clip), and the weight transform is applied on top,
    so changing power or clip reuses the one freq scan.

    With `merge_lig_poc=True`, pocket atoms are folded into the first 4 channels
    of the ligand 7-vec before counting.
    """
    n_atom_channels = 7 if merge_lig_poc else 11
    path = _ch_freq_cache_path(cfg, n_samples, merge_lig_poc=merge_lig_poc)
    if os.path.isfile(path):
        logger.info(f"loading channel-frequency cache from {path}")
        freq = torch.load(path, weights_only=True)["freq"].to(torch.float64)
    else:
        n = min(n_samples, len(train_dataset))
        logger.info(
            f"computing per-channel atom-voxel frequencies "
            f"(merge_lig_poc={merge_lig_poc}, n_channels={n_atom_channels}) "
            f"over {n} training samples..."
        )
        sum_per_ch = torch.zeros(n_atom_channels, dtype=torch.float64)
        n_voxels_total = 0
        with torch.no_grad():
            for i in range(n):
                sample = train_dataset[i]
                lig = {k: sample["ligand"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                poc = {k: sample["pocket"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                v_lig = voxelizer.forward(lig, num_channels=7)
                v_poc = voxelizer.forward(poc, num_channels=4)
                if merge_lig_poc:
                    atoms = _build_merged_atoms(v_lig, v_poc)          # (1, 7, G, G, G)
                else:
                    atoms = torch.cat([v_lig, v_poc], dim=1)           # (1, 11, G, G, G)
                pos = (atoms > _CH_FREQ_POS_THRESH).float()
                sum_per_ch += pos.sum(dim=(0, 2, 3, 4)).cpu().to(torch.float64)
                n_voxels_total += atoms.shape[2] * atoms.shape[3] * atoms.shape[4]
        freq = (sum_per_ch / max(1, n_voxels_total)).clamp(min=1e-8)   # pos-voxel frac
        torch.save({"freq": freq.to(torch.float32), "n_samples": n,
                    "pos_thresh": _CH_FREQ_POS_THRESH, "merge_lig_poc": merge_lig_poc}, path)
        logger.info(f"saved channel-frequency cache to {path}")

    w = (1.0 / freq).pow(power)
    if clip_ratio and clip_ratio > 0:
        # Raise the floor so max/min ≤ clip_ratio (keeps rare-atom emphasis,
        # gives crushed common/pocket channels a meaningful weight).
        w = w.clamp(min=float(w.max()) / float(clip_ratio))
    w = (w / w.sum() * float(n_atom_channels)).to(torch.float32)

    elem_lig = ["C", "O", "N", "S", "F", "Cl", "P"]
    elem_poc = ["C", "O", "N", "S"]
    names = ([f"merged_{e}" for e in elem_lig] if merge_lig_poc
             else [f"lig_{e}" for e in elem_lig] + [f"poc_{e}" for e in elem_poc])
    scheme = "1/freq" if power == 1.0 else ("1/sqrt(freq)" if power == 0.5 else f"1/freq^{power}")
    logger.info(
        f"per-channel atom weights ({scheme}, clip_ratio={clip_ratio or 'off'}, "
        f"norm to sum={n_atom_channels}; max/min={float(w.max()/w.min()):.1f}x):"
    )
    for i, name in enumerate(names):
        logger.info(f"  {name:>9}: freq={float(freq[i]):.3e}  weight={float(w[i]):.3f}")
    return w


def val_epoch(cfg, model, val_cache, device, ch_weight=None) -> dict:
    """Deterministic val pass — fixed-seed σ_blur / noise / mask per sample.

    In electra mode, cycles deterministically through `cfg.electra.corruption_ops`
    by batch index so val loss is comparable across epochs.
    """
    model.eval()
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
    patch_size = int(cfg.model.patch_size)
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    gradmag_noise = bool(cfg.mae.get("gradmag_noise", False))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
    need_density = input_mode in (
        "density", "atomblob_density", "atomblob_merged_density",
    )
    need_atoms_sum = need_density or mask_strategy == "atom_biased"

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

            g_clean = None
            if with_gradmag:
                if density_source == "xray":
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
            else:
                raise RuntimeError(f"unexpected input_mode={input_mode!r}")

            if with_gradmag:
                x_clean = torch.cat([x_clean, g_clean], dim=1)

            atom_only = input_mode in ("atomblob", "atomblob_merged")
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
            else:
                blocks = torch.rand(B, 1, G // block, G // block, G // block,
                                    device=device, generator=gen) < ratio
                mask = blocks.repeat_interleave(block, 2)\
                             .repeat_interleave(block, 3)\
                             .repeat_interleave(block, 4)

            if pretext_style == "mae":
                x_in = x_noisy * (~mask).to(x_noisy.dtype)
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
            out_pretext, a_hat = model(x_in)
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

def train_epoch(cfg, prefetcher, model, optimizer, model_ema, device, global_step, ch_weight=None) -> tuple:
    model.train()
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    input_mode = str(cfg.get("input_mode", "density"))
    patch_size = int(cfg.model.patch_size)
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
    if pretext_style == "electra":
        pretext_key = "L_rtd"
        lambda_pretext = float(cfg.mae.get("lambda_rtd", 1.0))
    else:
        pretext_key = "L_dens"
        lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))

    L_pretext_sum, L_str_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum, L_dens_gradmag_sum = 0.0, 0.0, 0.0
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    import contextlib
    for i, (x_in, x_clean, mask, target_str) in enumerate(prefetcher):
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_iter)
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        with sync_ctx:
            out_pretext, a_hat = model(x_in)
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
            loss.backward()
        if is_step:
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    _unwrap(model).parameters(), max_norm=grad_clip
                )
                grad_norm_sum += float(gn)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(_unwrap(model))
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
    return metrics, global_step


def log_metrics(epoch, train_metrics, val_metrics, dt):
    logger.info(f"epoch: {epoch} ({dt:.2f}s)")
    for split, m in zip(["train", "val"], [train_metrics, val_metrics]):
        if m is None:
            continue
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
        logger.info(f"[{split}] {parts}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config_train_density_vit_mae", version_base=None)
def main(cfg: DictConfig) -> None:
    assert torch.cuda.is_available(), "GPU required."
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    start_epoch = 0
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

    if is_main:
        logger.info("cfg:\n" + OmegaConf.to_yaml(cfg))

    if is_main and cfg.wandb:
        try:
            _tags = cfg.get("wandb_tags", None)
            _tags = list(_tags) if _tags else None
            # Strip leading YYMMDD_ from the exp_name for the wandb display name.
            # exp_name on disk keeps the date (per the exp-dir naming convention);
            # only the wandb run name shows the bare topic.
            import re as _re
            _wandb_name = _re.sub(r"^\d{6}_", "", cfg.exp_name)
            wandb.init(
                project="voxbind",
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
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    dual_head       = bool(cfg.model.get("dual_head", False))
    if with_gradmag and dual_head:
        raise ValueError(
            "dual_head is not yet supported with with_gradmag (the dual head "
            "assumes a lone trailing density channel); use the single MAE head."
        )
    head_hidden_dim = int(cfg.model.get("head_hidden_dim", 0))   # 0 → defaults to c_half
    head_depth      = int(cfg.model.get("head_depth", 2))
    # n_in must match the assembled layout (atoms + density + gradmag).
    n_in_cfg = int(cfg.model.get("n_in_channels", 1))
    if n_in_cfg != layout["n_in"]:
        raise RuntimeError(
            f"input_mode={input_mode!r} with_gradmag={with_gradmag} expects "
            f"model.n_in_channels={layout['n_in']}, got {n_in_cfg}"
        )
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
        pretext_style=pretext_style,
        dual_head=dual_head,
        head_hidden_dim=head_hidden_dim,
        head_depth=head_depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"DensityViTMAE has {(n_params/1e6):.02f}M parameters "
                    f"(pretext_style={pretext_style}, input_mode={input_mode}, "
                    f"with_gradmag={with_gradmag} (recon={gradmag_reconstruct}), "
                    f"n_in={layout['n_in']} n_recon={layout['n_recon']}, "
                    f"dual_head={dual_head}, head_hidden_dim={head_hidden_dim or '(c_half)'}, head_depth={head_depth})")

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

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    model_ema = ModelEma(_unwrap(model), decay=cfg.mae.ema_decay)

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
    # When both are at defaults (`uniform` + `1.0`), `ch_weight` stays None
    # and compute_losses takes the unweighted code path — preserving baseline behavior.
    # When either is non-default for atomblob_density, the final 12-vec is
    # renormalized to sum=n_channels so the overall L_pretext scale matches the
    # unweighted case (gradient share shifts; total mass preserved).
    ch_weight = None
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
    _density_modes = ("atomblob_density", "atomblob_merged_density")
    _merged_modes  = ("atomblob_merged", "atomblob_merged_density")
    needs_dens_downweight = (
        input_mode in _density_modes and density_channel_weight != 1.0
    )
    # gradmag weight only enters the loss when gradmag is a reconstruction target.
    needs_gradmag_weight = (
        with_gradmag and gradmag_reconstruct and gradmag_channel_weight != 1.0
    )
    if needs_atom_weights or needs_dens_downweight or needs_gradmag_weight:
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
        # gradient ratios are preserved.
        parts = [atom_w]
        if n_density > 0:
            parts.append(torch.tensor([density_channel_weight]))
        if with_gradmag and gradmag_reconstruct:
            parts.append(torch.tensor([gradmag_channel_weight]))
        raw = torch.cat(parts, dim=0)
        assert raw.shape[0] == layout["n_recon"], (
            f"weight vector length {raw.shape[0]} != n_recon {layout['n_recon']}"
        )
        ch_weight = (raw * (float(layout["n_recon"]) / float(raw.sum()))).to(device)

        if is_main:
            logger.info(
                f"using channel weights on {ch_weight.shape[0]} channels "
                f"(sum={float(ch_weight.sum()):.3f})  "
                f"channel_weighting={ch_weighting}  channel_weight_clip_ratio={channel_weight_clip or 'off'}  "
                f"density_channel_weight={density_channel_weight}  "
                f"gradmag_channel_weight={gradmag_channel_weight}  merge_lig_poc={merge_lig_poc}"
            )
            if n_density > 0:
                logger.info(f"  effective density weight = {float(ch_weight[n_atom]):.4f}")
            if with_gradmag and gradmag_reconstruct:
                logger.info(f"  effective gradmag weight = {float(ch_weight[n_atom + n_density]):.4f}")

    sigma_blur_vox_range = (
        cfg.mae.sigma_blur_a_lo / cfg.vox.resolution,
        cfg.mae.sigma_blur_a_hi / cfg.vox.resolution,
    )

    if is_main:
        logger.info("start density-ViT-MAE pre-training...")

    epochs_iter = tqdm(
        range(start_epoch, start_epoch + cfg.num_epochs),
        desc="Epochs", disable=not is_main,
    )
    ckpt_saver = AsyncCheckpointSaver() if is_main else None

    for epoch in epochs_iter:
        t0 = time.time()
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        if pretext_style == "electra":
            corruption_ops = tuple(cfg.electra.corruption_ops)
            corruption_op_weights = tuple(
                cfg.electra.get("corruption_op_weights", [1.0] * len(corruption_ops))
            )
        else:
            corruption_ops = ("swap",)        # unused in mae mode
            corruption_op_weights = (1.0,)
        prefetcher = MAEPrefetcher(
            loader_train, voxelizer,
            sigma_blur_vox_range=sigma_blur_vox_range,
            sigma_noise=cfg.mae.sigma_noise,
            block_size=cfg.mae.block_size,
            mask_ratio=cfg.mae.mask_ratio,
            pretext_style=pretext_style,
            corruption_ops=corruption_ops,
            corruption_op_weights=corruption_op_weights,
            density_source=str(cfg.mae.get("density_source", "synthetic")),
            input_mode=input_mode,
            mask_strategy=str(cfg.mae.get("mask_strategy", "uniform")),
            mask_atom_tau=float(cfg.mae.get("mask_atom_tau", 1.0)),
            with_gradmag=with_gradmag,
            gradmag_reconstruct=gradmag_reconstruct,
            gradmag_noise=gradmag_noise,
        )
        train_metrics, global_step = train_epoch(
            cfg, prefetcher, model, optimizer, model_ema, device, global_step,
            ch_weight=ch_weight,
        )

        val_metrics = None
        if epoch % max(1, int(cfg.mae.val_every)) == 0:
            val_metrics = val_epoch(cfg, model_ema.module, val_cache, device,
                                    ch_weight=ch_weight)

        if is_main:
            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb and wandb.run is not None:
                log_payload = {"train": train_metrics, "epoch": epoch}
                if val_metrics is not None:
                    log_payload["val"] = val_metrics
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
            if (epoch % ckpt_every == 0) or (epoch == cfg.num_epochs - 1):
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


if __name__ == "__main__":
    main()
