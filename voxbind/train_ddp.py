"""DDP variant of train.py.

Differences vs train.py (all additive — math is preserved):
  * torch.distributed.init_process_group(backend="nccl") from torchrun env vars
  * DataParallel -> DistributedDataParallel
  * create_dataloaders(..., distributed=True, rank, world_size)
  * train_sampler.set_epoch(epoch) each epoch (shuffle correctness)
  * Rank-0 only: wandb.init, create_exp_dir write, save_checkpoint, sample(),
    val_cache file write (others barrier+load).
  * cfg.bsz is interpreted PER-RANK in DDP (matches dataset/__init__.py contract
    and the comment in scripts/01a_train_xray100.sh). Effective batch
    size across the world is cfg.bsz * world_size.

Launch:
    torchrun --nproc_per_node=<W> train_ddp.py <hydra overrides>
"""
import hashlib
import json
import logging
import math
import os
import time

import hydra
import torch
import torch.distributed as dist
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from voxbind.dataset import create_dataloaders
from voxbind.metrics import create_metrics_for_training
from voxbind.models import create_model
from voxbind.models.adamw import AdamW
from voxbind.models.ema import ModelEma
from voxbind.utils.base_utils import (
    create_exp_dir, load_checkpoint, makedir, save_checkpoint, seed_everything,
)
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.voxelizer import Voxelizer

logger = logging.getLogger("training-ddp")


def _unwrap(model):
    """Return the underlying module if model is wrapped in DP/DDP."""
    return model.module if hasattr(model, "module") else model


def _setup_ddp() -> tuple:
    """Initialize the default process group from torchrun env vars.

    Returns:
        (rank, local_rank, world_size)
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class VoxelPrefetcher:
    """Overlaps GPU voxelization of batch N+1 with model compute on batch N.

    Identical to train.py.VoxelPrefetcher — duplicated here only to keep this
    file self-contained for the smoke test (no cross-imports of private
    helpers from train.py).
    """

    def __init__(self, loader, voxelizer, smooth_sigma, training=True, with_density=False):
        self.loader = loader
        self.voxelizer = voxelizer
        self.smooth_sigma = smooth_sigma
        self.training = training
        self.with_density = with_density
        self.stream = torch.cuda.Stream()

    def _voxelize(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                voxels_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                smooth_voxels_lig = add_noise_vox(voxels_lig, self.smooth_sigma)
                voxels_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)
                if self.training:
                    voxels_poc[:math.ceil(.2 * voxels_poc.shape[0])].zero_()
                density = None
                if self.with_density and "xray_density" in batch:
                    density = batch["xray_density"].to(
                        self.voxelizer.device, non_blocking=True
                    ).unsqueeze(1)
                    if "xray_available" in batch:
                        avail = batch["xray_available"].to(
                            self.voxelizer.device, non_blocking=True
                        )
                        density = density * avail.view(-1, 1, 1, 1, 1).float()
        return voxels_lig, smooth_voxels_lig, voxels_poc, density

    def __iter__(self):
        loader_it = iter(self.loader)
        try:
            next_batch = next(loader_it)
        except StopIteration:
            return

        next_out = self._voxelize(next_batch)

        for batch in loader_it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur_out = next_out
            next_out = self._voxelize(batch)
            yield cur_out

        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_out

    def __len__(self):
        return len(self.loader)


@hydra.main(config_path="configs", config_name="config_train", version_base=None)
def main(cfg: DictConfig) -> None:
    # -----------------------------------------------------------
    # DDP init
    assert torch.cuda.is_available(), "not a good idea to train on cpu..."
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    start_epoch = 0
    create_exp_dir(cfg, write=is_main)
    # Sync after create_exp_dir so non-main ranks don't race ahead and read
    # cfg.yaml from disk while rank 0 is mid-write. This bites the resume
    # path specifically: cfg.resume and cfg.output_dir resolve to the same
    # directory, so OmegaConf.load(cfg.resume/cfg.yaml) below races with
    # rank 0's create_exp_dir write. Without this barrier, ranks 1..W-1
    # see a half-written YAML (e.g. missing top-level keys like `aug`) and
    # crash at the first cfg attribute access.
    if world_size > 1:
        dist.barrier()
    # Every rank logs its DDP identity once — lets downstream tooling verify
    # that all expected ranks actually came up (rank-0-only log gives a false
    # negative for "all ranks initialized?").
    logger.info(f"DDP: world_size={world_size} rank={rank} local_rank={local_rank}")
    if is_main:
        logger.info(f"n gpus visible to this rank: {torch.cuda.device_count()}")
        logger.info(f"saving experiments in: {cfg.output_dir}")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    # Aug differs per rank (different shard + worker init); model init seed is
    # re-set below so all ranks build identical weights.
    seed_everything(cfg.seed + rank)

    # resume?
    if cfg.resume is not None and os.path.isdir(cfg.resume):
        if is_main:
            logger.info(f"resuming from: {cfg.resume}")
        resume = cfg.resume
        wjs_override = cfg.wjs
        wandb_override = cfg.wandb
        resume_epoch_override = cfg.resume_epoch
        num_epochs_override = cfg.num_epochs   # let CLI extend training
        cfg = OmegaConf.load(os.path.join(cfg.resume, "cfg.yaml"))
        cfg.output_dir, cfg.resume = resume, resume
        cfg.wandb = wandb_override
        cfg.wjs = wjs_override
        cfg.resume_epoch = resume_epoch_override
        cfg.num_epochs = num_epochs_override

    if is_main:
        logger.info("cfg:\n" + OmegaConf.to_yaml(cfg))

    # wandb (rank 0 only)
    if is_main and cfg.wandb:
        wandb.init(
            project="voxbind",
            entity="eddy26",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.exp_name,
            dir=cfg.output_dir,
            resume="allow",
            settings=wandb.Settings(code_dir=".", init_timeout=300),
        )

    # -----------------------------------------------------------
    # data loaders (train uses DistributedSampler; val/sampling are not sharded)
    loader_train, loader_val, loader_sampling = create_dataloaders(
        cfg, distributed=True, rank=rank, world_size=world_size
    )
    n_train, n_val = len(loader_train.dataset), len(loader_val.dataset)
    if is_main:
        logger.info(f"training/val set size: {n_train}/{n_val}")
        logger.info(f"train batches per rank: {len(loader_train)} "
                    f"(effective batch = {cfg.bsz * world_size})")

    # model, criterion, optimizer — identical init across ranks (DDP also
    # broadcasts rank-0 params on wrap, but matching seeds keeps the pre-wrap
    # state identical too).
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    model = create_model(cfg, device=device)
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"model has {(n_params/1e6):.02f}M parameters")

    # voxelizer (per-rank, on local device)
    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    # resume model+optimizer state (before DDP wrap)
    if cfg.resume is not None:
        if is_main:
            logger.info("reloading states of model, optimizer")
        model, optimizer, start_epoch = load_checkpoint(
            model, cfg.output_dir, optimizer, best_model=False
        )
        if is_main:
            os.system(
                f"cp {os.path.join(cfg.output_dir, 'checkpoint.pth.tar')} "
                f"{os.path.join(cfg.output_dir, f'checkpoint_{start_epoch}.pth.tar')}"
            )
            logger.info(f"model trained for {start_epoch} epochs")
        if cfg.resume_epoch is not None:
            if is_main:
                logger.info(
                    f"overriding start_epoch {start_epoch} -> {cfg.resume_epoch} (resume_epoch)"
                )
            start_epoch = cfg.resume_epoch

    # DDP wrap. find_unused_parameters=True because VoxBind has params that
    # don't receive grad on every forward (the UNet3D time-embedding MLP is
    # called with t=None, plus a few attn-block biases). Cleaner long-term
    # fix is to gate the time-embed MLP creation on whether t is ever passed,
    # but for the smoke test we just enable the runtime check.
    model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    model_ema = ModelEma(_unwrap(model), decay=.999)

    # metrics (torchmetrics auto-syncs across ranks on .compute())
    metrics_denoise, _ = create_metrics_for_training(device=device)

    # pre-compute val voxels — rank 0 writes the disk cache, others wait+load
    if is_main:
        val_cache = precompute_val_voxels(loader_val, voxelizer, cfg)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        val_cache = precompute_val_voxels(loader_val, voxelizer, cfg)

    # -----------------------------------------------------------
    # start training
    if is_main:
        logger.info("start training...")
    epochs_iter = tqdm(
        range(start_epoch, start_epoch + cfg.num_epochs),
        desc="Epochs", disable=not is_main,
    )
    for epoch in epochs_iter:
        t0 = time.time()

        # DistributedSampler needs set_epoch for proper shuffling
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # train (all ranks)
        train_metrics = train(
            cfg, loader_train, voxelizer, model, criterion, optimizer,
            metrics_denoise, model_ema,
        )

        # val (all ranks — val_cache is identical, metrics auto-sync. Wasteful
        # but avoids the rank-0-only / sync_dist plumbing for a smoke test.)
        val_metrics = val(
            cfg, loader_val, voxelizer, model_ema.module, criterion, metrics_denoise,
            val_cache=val_cache,
        )

        # sample (rank 0 only — wjs is not parallelized)
        if is_main and epoch > 0 and (epoch % 50 == 0 or epoch == cfg.num_epochs - 1):
            sample(cfg, loader_sampling, voxelizer, model_ema.module, epoch)

        # log + save (rank 0)
        if is_main:
            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb:
                wandb.log({"train": train_metrics, "val": val_metrics})
            save_checkpoint({
                "epoch": epoch,
                "metrics": {"train": train_metrics, "val": val_metrics},
                "cfg": cfg,
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_dir=cfg.output_dir)

        # Wait for rank 0 to finish save_checkpoint before any rank advances.
        # Without this, ranks 1..W-1 race past the save into the next
        # iteration (or, on the last epoch, into _cleanup_ddp), and on the
        # last epoch destroy_process_group ends up waiting on rank 0 for
        # potentially minutes of disk I/O. If that exceeds the NCCL watchdog
        # timeout (default 600s) the watchdog SIGABRTs the ranks — exactly
        # the T4 hang we saw at 600s post-training.
        if world_size > 1:
            # device_ids tells NCCL which CUDA device to issue the barrier
            # collective on; without it the first-ever barrier emits a noisy
            # "using GPU N to perform barrier as devices used by this
            # process are currently unknown" W-level warning per rank.
            dist.barrier(device_ids=[local_rank])

        torch.cuda.empty_cache()

    # Defense in depth: explicit final barrier so all ranks reach the same
    # point before tearing down the process group.
    if world_size > 1:
        dist.barrier()
    _cleanup_ddp()


def train(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    criterion: torch.nn,
    optimizer: torch.optim,
    metrics: torchmetrics.MetricCollection,
    model_ema: torch.nn,
) -> dict:
    """Train one epoch — body identical to train.py.train()."""
    metrics.reset()
    model.train()
    train_miou_interval = max(1, int(cfg.get("train_miou_interval", 8)))

    with_density = bool(cfg.model.get("with_density", False))
    prefetcher = VoxelPrefetcher(
        loader, voxelizer, cfg.smooth_sigma,
        training=True, with_density=with_density,
    )
    for i, (voxels_lig, smooth_voxels_lig, voxels_poc, density) in enumerate(prefetcher):
        pred = model(smooth_voxels_lig, voxels_poc, density=density)
        loss = criterion(pred, voxels_lig)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model_ema.update(_unwrap(model))

        metrics.update(
            loss.detach(), pred.detach(), voxels_lig,
            update_miou=(i % train_miou_interval == 0)
        )

        if cfg.debug and i == 10:
            break

    return metrics.compute()


def val(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    criterion: torch.nn,
    metrics: torchmetrics.MetricCollection,
    val_cache: dict = None,
) -> dict:
    """Evaluate — body identical to train.py.val()."""
    metrics.reset()
    model.eval()
    device = next(model.parameters()).device
    with_density = bool(cfg.model.get("with_density", False))

    with torch.no_grad():
        if val_cache is not None:
            voxels_lig_all = val_cache["voxels_lig"]
            voxels_poc_all = val_cache["voxels_poc"]
            density_all = val_cache.get("density") if with_density else None
            n = voxels_lig_all.shape[0]
            for i, start in enumerate(range(0, n, cfg.bsz)):
                voxels_lig = voxels_lig_all[start:start + cfg.bsz].to(device)
                voxels_poc = voxels_poc_all[start:start + cfg.bsz].to(device)
                density = density_all[start:start + cfg.bsz].to(device) if density_all is not None else None
                smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
                pred = model(smooth_voxels_lig, voxels_poc, density=density)
                loss = criterion(pred, voxels_lig)
                metrics.update(loss, pred, voxels_lig)
                if cfg.debug and i == 10:
                    break
        else:
            prefetcher = VoxelPrefetcher(
                loader, voxelizer, cfg.smooth_sigma,
                training=False, with_density=with_density,
            )
            for i, (voxels_lig, smooth_voxels_lig, voxels_poc, density) in enumerate(prefetcher):
                pred = model(smooth_voxels_lig, voxels_poc, density=density)
                loss = criterion(pred, voxels_lig)
                metrics.update(loss, pred, voxels_lig)
                if cfg.debug and i == 10:
                    break

    return metrics.compute()


def sample(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    epoch: int,
) -> None:
    """Rank-0-only WJS sampling."""
    model = _unwrap(model)
    model.eval()

    dirname = os.path.join(cfg.output_dir, f"samples_training/{epoch:02d}")
    makedir(dirname)

    for pocket_id, batch in enumerate(loader):
        if pocket_id == cfg.wjs.n_targets:
            break
        logger.info(f"| sampling pocket {pocket_id}")
        target_dirname = os.path.join(dirname, f"target_{pocket_id:02d}")
        pocket, ligand_gt = batch["pocket"], batch["ligand"]
        sample_molecules(model, pocket, ligand_gt, voxelizer, target_dirname, cfg)


def _val_cache_path(cfg) -> str:
    """Cache path for val voxels — keyed by voxelizer + dataset params."""
    key = {
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "cubes_around": cfg.vox.cubes_around,
        "ligand_radius": cfg.dset.ligand_radius,
        "pocket_radius": cfg.dset.pocket_radius,
        "dset_name": cfg.dset.dset_name,
        "with_density": bool(cfg.model.get("with_density", False)),
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_{h}.pt")


def precompute_val_voxels(
    loader_val: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    cfg: DictConfig,
) -> dict:
    """Voxelize the val set once and cache to disk."""
    cache_path = _val_cache_path(cfg)
    if os.path.isfile(cache_path):
        logger.info(f"loading pre-computed val voxels from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    with_density = bool(cfg.model.get("with_density", False))
    logger.info(
        f"pre-computing val voxels (with_density={with_density}); will cache for future runs..."
    )
    all_lig, all_poc, all_dens = [], [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=7).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=4).cpu())
            if with_density and "xray_density" in batch:
                d = batch["xray_density"].unsqueeze(1)
                if "xray_available" in batch:
                    d = d * batch["xray_available"].view(-1, 1, 1, 1, 1).float()
                all_dens.append(d.cpu())

    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    if all_dens:
        cache["density"] = torch.cat(all_dens, 0)
    torch.save(cache, cache_path)
    logger.info(f"saved val voxels to {cache_path}")
    return cache


def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma > 0:
        return voxels + torch.empty_like(voxels).normal_(0, sigma)
    return voxels


def log_metrics(epoch: int, train_metrics: dict, val_metrics: dict, time: float) -> None:
    all_metrics = [train_metrics, val_metrics]
    metrics_names = ["train", "val"]
    logger.info(f"epoch: {epoch} ({time:.2f}s)")
    for split, metric in zip(metrics_names, all_metrics):
        if metric is None:
            continue
        str_ = f"[{split}]"
        for k, v in metric.items():
            str_ += f" | {k}: {v:.2f}" if k == "loss" else f" | {k}: {v:.4f}"
        logger.info(str_)


if __name__ == "__main__":
    main()
