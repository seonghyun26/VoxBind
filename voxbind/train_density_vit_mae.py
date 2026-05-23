"""train_density_vit_mae.py — DDP voxel-MAE pre-training for the ViT density encoder.

Same pretext as `train_density_mae.py` (synthetic Gaussian-blur density + 3D
block masking + density-reconstruction + 11-channel atom-structure heads), but
the encoder is `DensityViTMAE` (pure 3D ViT) instead of the CNN-stack
`DensityMAE`. Apples-to-apples ablation: change `model:` only.

Checkpoint layout matches `train_density_mae.py` exactly:
  state_dict_ema          full DensityViTMAE EMA
  encoder_state_dict_ema  encoder.* slice — drops into VoxBind.density_encoder
                          (with `density_encoder_type=vit`) via the existing
                          loader in `models/__init__.py`.

Launch
------
    cd voxbind
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \\
        train_density_vit_mae.py \\
        exp_name=260522_density_vit_mae_pretrain \\
        output_dir=exps/260522_density_vit_mae_pretrain
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
    gaussian_blur3d, make_block_mask, per_sample_zscore,
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


# ── MAE prefetcher (same as train_density_mae.py) ─────────────────────────────

class MAEPrefetcher:
    """Voxelize → synthesize density → mask, all overlapped with model compute.

    Yields per batch:
        density_in   (B, 1, G, G, G)  — masked, noisy synthetic density
        density_clean(B, 1, G, G, G)  — un-masked, noiseless z-scored target
        mask         (B, 1, G, G, G)  — bool, True = masked
        target_str   (B, 11, G, G, G) — voxelized ligand (7) + pocket (4)
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
    ):
        self.loader = loader
        self.voxelizer = voxelizer
        self.sigma_range = sigma_blur_vox_range
        self.sigma_noise = sigma_noise
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        self.generator = generator
        self.stream = torch.cuda.Stream()

    def _step(self, batch):
        device = self.voxelizer.device
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                v_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                v_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)

                atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
                sigma_lo, sigma_hi = self.sigma_range
                if self.generator is not None:
                    u = torch.rand(1, generator=self.generator).item()
                else:
                    u = torch.rand(1).item()
                sigma_vox = sigma_lo + u * (sigma_hi - sigma_lo)
                d_clean = gaussian_blur3d(atoms_sum, sigma_vox)
                d_clean = per_sample_zscore(d_clean)
                noise = torch.randn(d_clean.shape, device=device,
                                    generator=self.generator) if self.generator is not None \
                    else torch.randn_like(d_clean)
                d_noisy = d_clean + noise * self.sigma_noise

                B = d_clean.shape[0]
                G = d_clean.shape[-1]
                mask = make_block_mask(B, G, self.block_size, self.mask_ratio, device)
                d_in = d_noisy * (~mask).to(d_noisy.dtype)

                target_str = torch.cat([v_lig, v_poc], dim=1)
        return d_in, d_clean, mask, target_str

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


# ── Losses (identical to cnn-MAE so the ViT vs CNN ablation is apples-to-apples) ──

def compute_losses(
    d_hat, a_hat, d_clean, target_str, mask,
    struct_pos_weight: float = 1.0,
    struct_pos_thresh: float = 0.05,
):
    m = mask.to(d_hat.dtype)
    n_masked = m.sum().clamp(min=1.0)
    L_dens = ((d_hat - d_clean) ** 2 * m).sum() / n_masked

    if struct_pos_weight > 1.0:
        pos = (target_str > struct_pos_thresh).to(a_hat.dtype)
        w = 1.0 + (struct_pos_weight - 1.0) * pos
        L_str = ((a_hat - target_str) ** 2 * w).mean()
    else:
        L_str = F.mse_loss(a_hat, target_str)
    return L_dens, L_str


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
        # `density_mae_v1` is kept identical to cnn-MAE — the cached tensors are
        # just voxelized ligand+pocket, no encoder-specific content.
        "task": "density_mae_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_mae_{h}.pt")


def precompute_val(loader_val, voxelizer, cfg) -> dict:
    path = _val_cache_path(cfg)
    if os.path.isfile(path):
        logger.info(f"loading pre-computed val voxels from {path}")
        return torch.load(path, weights_only=True)

    logger.info("pre-computing val voxels...")
    all_lig, all_poc = [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=7).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=4).cpu())
    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    torch.save(cache, path)
    logger.info(f"saved val voxels to {path}")
    return cache


def val_epoch(cfg, model, val_cache, device) -> dict:
    """Deterministic val pass — fixed-seed σ_blur / noise / mask per sample."""
    model.eval()
    sigma_lo = cfg.mae.sigma_blur_a_lo / cfg.vox.resolution
    sigma_hi = cfg.mae.sigma_blur_a_hi / cfg.vox.resolution
    block = cfg.mae.block_size
    ratio = cfg.mae.mask_ratio
    sigma_noise = cfg.mae.sigma_noise

    voxels_lig = val_cache["voxels_lig"]
    voxels_poc = val_cache["voxels_poc"]
    n = voxels_lig.shape[0]

    L_dens_sum, L_str_sum, n_batches = 0.0, 0.0, 0
    gen = torch.Generator(device=device)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n, cfg.bsz)):
            gen.manual_seed(batch_idx + 1)
            v_lig = voxels_lig[start:start + cfg.bsz].to(device, non_blocking=True)
            v_poc = voxels_poc[start:start + cfg.bsz].to(device, non_blocking=True)
            atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
            sigma_vox = sigma_lo + (sigma_hi - sigma_lo) * 0.5
            d_clean = gaussian_blur3d(atoms_sum, sigma_vox)
            d_clean = per_sample_zscore(d_clean)
            d_noisy = d_clean + torch.randn(d_clean.shape, device=device, generator=gen) * sigma_noise
            B, G = d_clean.shape[0], d_clean.shape[-1]
            blocks = torch.rand(B, 1, G // block, G // block, G // block,
                                device=device, generator=gen) < ratio
            mask = blocks.repeat_interleave(block, 2)\
                         .repeat_interleave(block, 3)\
                         .repeat_interleave(block, 4)
            d_in = d_noisy * (~mask).to(d_noisy.dtype)
            target_str = torch.cat([v_lig, v_poc], dim=1)
            d_hat, a_hat = model(d_in)
            L_dens, L_str = compute_losses(
                d_hat, a_hat, d_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
            )
            L_dens_sum += L_dens.item()
            L_str_sum += L_str.item()
            n_batches += 1
            if cfg.debug and n_batches >= 5:
                break

    return {
        "L_dens": L_dens_sum / max(1, n_batches),
        "L_str": L_str_sum / max(1, n_batches),
        "loss": (cfg.mae.lambda_dens * L_dens_sum + cfg.mae.lambda_str * L_str_sum)
                / max(1, n_batches),
    }


# ── Train loop ────────────────────────────────────────────────────────────────

def train_epoch(cfg, prefetcher, model, optimizer, model_ema, device, global_step) -> tuple:
    model.train()
    L_dens_sum, L_str_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0.0, 0
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    import contextlib
    for i, (d_in, d_clean, mask, target_str) in enumerate(prefetcher):
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_iter)
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        with sync_ctx:
            d_hat, a_hat = model(d_in)
            L_dens, L_str = compute_losses(
                d_hat, a_hat, d_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
            )
            loss = cfg.mae.lambda_dens * L_dens + cfg.mae.lambda_str * L_str
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
        L_dens_sum += L_dens.item()
        L_str_sum += L_str.item()
        n_batches += 1
        if cfg.debug and i == 10:
            break

    metrics = {
        "L_dens": L_dens_sum / max(1, n_batches),
        "L_str": L_str_sum / max(1, n_batches),
        "loss": (cfg.mae.lambda_dens * L_dens_sum + cfg.mae.lambda_str * L_str_sum)
                / max(1, n_batches),
    }
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

    # Model — pure 3D ViT density encoder + density head + 11-ch atom-structure head
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    model = DensityViTMAE(
        grid_dim=int(cfg.vox.grid_dim),
        patch_size=int(cfg.model.patch_size),
        n_channels=int(cfg.model.n_channels),
        dim=int(cfg.model.dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.heads),
        mlp_ratio=int(cfg.model.mlp_ratio),
        dropout=float(cfg.model.dropout),
        n_struct_channels=int(cfg.model.n_struct_channels),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"DensityViTMAE has {(n_params/1e6):.02f}M parameters")

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

        prefetcher = MAEPrefetcher(
            loader_train, voxelizer,
            sigma_blur_vox_range=sigma_blur_vox_range,
            sigma_noise=cfg.mae.sigma_noise,
            block_size=cfg.mae.block_size,
            mask_ratio=cfg.mae.mask_ratio,
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
