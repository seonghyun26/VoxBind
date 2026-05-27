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
    gaussian_blur3d, make_block_mask, per_sample_zscore,
    voxel_mask_to_patch_target,
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


_VALID_INPUT_MODES = ("density", "atomblob", "atomblob_density")


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
    ):
        assert pretext_style in ("mae", "electra")
        assert density_source in ("synthetic", "xray"), (
            f"density_source={density_source!r}; expected 'synthetic' or 'xray'"
        )
        assert input_mode in _VALID_INPUT_MODES, (
            f"input_mode={input_mode!r}; expected one of {_VALID_INPUT_MODES}"
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
        self.stream = torch.cuda.Stream()

    def _step(self, batch):
        device = self.voxelizer.device
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                v_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                v_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)

                # Density is only synthesized / loaded when the input actually
                # needs it. atomblob skips this entirely — no I/O via the
                # xray path, no blur on the synthetic path.
                need_density = self.input_mode in ("density", "atomblob_density")
                d_clean = None
                atoms_sum = None
                sigma_lo, sigma_hi = self.sigma_range  # always defined for reblur
                if need_density:
                    atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
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

                # Assemble the multi-channel input per mode.
                if self.input_mode == "density":
                    x_clean = d_clean                                       # (B, 1, G³)
                elif self.input_mode == "atomblob":
                    x_clean = torch.cat([v_lig, v_poc], dim=1)              # (B, 11, G³)
                else:                                                       # atomblob_density
                    x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)     # (B, 12, G³)

                # Noise is a density-only augmentation. atomblob skips it;
                # atomblob_density noises only the trailing density channel.
                if self.sigma_noise > 0 and self.input_mode != "atomblob":
                    if self.input_mode == "density":
                        if self.generator is not None:
                            noise = torch.randn(x_clean.shape, device=device, generator=self.generator)
                        else:
                            noise = torch.randn_like(x_clean)
                        x_noisy = x_clean + noise * self.sigma_noise
                    else:  # atomblob_density: noise only on the density channel
                        if self.generator is not None:
                            n_dens = torch.randn(
                                (x_clean.shape[0], 1, *x_clean.shape[2:]),
                                device=device, generator=self.generator,
                            )
                        else:
                            n_dens = torch.randn(
                                (x_clean.shape[0], 1, *x_clean.shape[2:]), device=device,
                            )
                        x_noisy = x_clean.clone()
                        x_noisy[:, -1:] = x_noisy[:, -1:] + n_dens * self.sigma_noise
                else:
                    x_noisy = x_clean

                B = x_clean.shape[0]
                G = x_clean.shape[-1]
                mask = make_block_mask(B, G, self.block_size, self.mask_ratio, device)
                # mask: (B, 1, G, G, G) — broadcasts across all input channels

                if self.pretext_style == "mae":
                    x_in = x_noisy * (~mask).to(x_noisy.dtype)
                else:
                    # ELECTRA corruption ops are currently single-channel only;
                    # the new multi-channel input modes go through MAE pretext.
                    if self.input_mode != "density":
                        raise NotImplementedError(
                            f"pretext_style='electra' is not yet supported for "
                            f"input_mode={self.input_mode!r}; use pretext_style='mae'."
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
):
    """Returns a dict of named losses.

    mae path:
      L_dens : masked-voxel MSE between out_pretext (B,C,G,G,G) and x_clean,
               where C = 1 / 11 / 12 per input_mode.
      L_str  : 11-ch atom structure MSE (pos-weighted), only if a_hat is given.

      For input_mode='atomblob_density' also returns the per-modality split:
        L_dens_atom    — MSE on the 11 atom channels in masked positions
        L_dens_density — MSE on the trailing density channel in masked positions

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
        diff_sq = (out_pretext - x_clean) ** 2 * m
        n_channels = out_pretext.shape[1]
        L_pretext = diff_sq.sum() / (n_masked_spatial * n_channels)
        out_key = "L_dens"

        if input_mode == "atomblob_density":
            # First 11 channels = atom voxels; last 1 = density.
            atom_diff = diff_sq[:, :11].sum() / (n_masked_spatial * 11)
            dens_diff = diff_sq[:, 11:].sum() / (n_masked_spatial * 1)
            losses["L_dens_atom"] = atom_diff
            losses["L_dens_density"] = dens_diff
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
    logger.info(f"pre-computing val voxels (density_source={density_source})...")
    all_lig, all_poc, all_xray = [], [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=7).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=4).cpu())
            if density_source == "xray":
                all_xray.append(batch["xray_density"].cpu())
    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    if density_source == "xray":
        cache["xray_density"] = torch.cat(all_xray, 0)
    torch.save(cache, path)
    logger.info(f"saved val voxels to {path}")
    return cache


def val_epoch(cfg, model, val_cache, device) -> dict:
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
    patch_size = int(cfg.model.patch_size)
    need_density = input_mode in ("density", "atomblob_density")

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
    n = voxels_lig.shape[0]

    L_pretext_sum, L_str_sum, n_batches = 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum = 0.0, 0.0
    gen = torch.Generator(device=device)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n, cfg.bsz)):
            gen.manual_seed(batch_idx + 1)
            v_lig = voxels_lig[start:start + cfg.bsz].to(device, non_blocking=True)
            v_poc = voxels_poc[start:start + cfg.bsz].to(device, non_blocking=True)
            d_clean = None
            atoms_sum = None
            if need_density:
                atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
                if density_source == "xray":
                    d_clean = cached_xray[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
                else:
                    sigma_vox = sigma_lo + (sigma_hi - sigma_lo) * 0.5
                    d_clean = gaussian_blur3d(atoms_sum, sigma_vox)
                    d_clean = per_sample_zscore(d_clean)

            if input_mode == "density":
                x_clean = d_clean
            elif input_mode == "atomblob":
                x_clean = torch.cat([v_lig, v_poc], dim=1)
            else:  # atomblob_density
                x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)

            if sigma_noise > 0 and input_mode != "atomblob":
                if input_mode == "density":
                    x_noisy = x_clean + torch.randn(x_clean.shape, device=device, generator=gen) * sigma_noise
                else:  # atomblob_density: noise on density channel only
                    n_dens = torch.randn(
                        (x_clean.shape[0], 1, *x_clean.shape[2:]),
                        device=device, generator=gen,
                    )
                    x_noisy = x_clean.clone()
                    x_noisy[:, -1:] = x_noisy[:, -1:] + n_dens * sigma_noise
            else:
                x_noisy = x_clean

            B, G = x_clean.shape[0], x_clean.shape[-1]
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
            )
            L_pretext_sum += losses[pretext_key].item()
            L_str_sum += losses["L_str"].item()
            if "L_dens_atom" in losses:
                L_dens_atom_sum += losses["L_dens_atom"].item()
                L_dens_density_sum += losses["L_dens_density"].item()
            n_batches += 1
            if cfg.debug and n_batches >= 5:
                break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    if input_mode == "atomblob_density":
        metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        metrics["L_dens_density"] = L_dens_density_sum / n_b
    return metrics


# ── Train loop ────────────────────────────────────────────────────────────────

def train_epoch(cfg, prefetcher, model, optimizer, model_ema, device, global_step) -> tuple:
    model.train()
    pretext_style = str(cfg.mae.get("pretext_style", "mae"))
    input_mode = str(cfg.get("input_mode", "density"))
    patch_size = int(cfg.model.patch_size)
    if pretext_style == "electra":
        pretext_key = "L_rtd"
        lambda_pretext = float(cfg.mae.get("lambda_rtd", 1.0))
    else:
        pretext_key = "L_dens"
        lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))

    L_pretext_sum, L_str_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum = 0.0, 0.0
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
            L_dens_density_sum += losses["L_dens_density"].item()
        n_batches += 1
        if cfg.debug and i == 10:
            break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    if input_mode == "atomblob_density":
        metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        metrics["L_dens_density"] = L_dens_density_sum / n_b
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
            wandb.init(
                project="voxbind",
                entity="eddy26",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=cfg.exp_name,
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
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    model = DensityViTMAE(
        grid_dim=int(cfg.vox.grid_dim),
        patch_size=int(cfg.model.patch_size),
        n_in_channels=int(cfg.model.get("n_in_channels", 1)),
        n_channels=int(cfg.model.n_channels),
        dim=int(cfg.model.dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.heads),
        mlp_ratio=int(cfg.model.mlp_ratio),
        dropout=float(cfg.model.dropout),
        n_struct_channels=int(cfg.model.n_struct_channels),
        pretext_style=pretext_style,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"DensityViTMAE has {(n_params/1e6):.02f}M parameters "
                    f"(pretext_style={pretext_style}, input_mode={input_mode})")

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
        )
        train_metrics, global_step = train_epoch(
            cfg, prefetcher, model, optimizer, model_ema, device, global_step,
        )

        val_metrics = None
        if epoch % max(1, int(cfg.mae.val_every)) == 0:
            val_metrics = val_epoch(cfg, model_ema.module, val_cache, device)

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
