import hydra
import logging
import math
from tqdm import tqdm
import os
from omegaconf import DictConfig, OmegaConf
import time
import torch
import torchmetrics
import wandb
import hashlib
import json

from voxbind.voxelizer import Voxelizer
from voxbind.models import create_model
from voxbind.utils.base_utils import (
    create_exp_dir, makedir, seed_everything, save_checkpoint, load_checkpoint
)
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.ema import ModelEma
from voxbind.metrics import create_metrics_for_training

logger = logging.getLogger("training")


class VoxelPrefetcher:
    """Overlaps GPU voxelization of batch N+1 with model compute on batch N.

    Voxelization runs on a secondary CUDA stream so the GPU is always busy
    with either model forward/backward or the next voxelization, eliminating
    the idle-GPU oscillation seen when voxelization and compute are sequential.
    Augmentation (rotation/translation) still happens in DataLoader workers.
    """

    def __init__(self, loader, voxelizer, smooth_sigma, training=True):
        self.loader = loader
        self.voxelizer = voxelizer
        self.smooth_sigma = smooth_sigma
        self.training = training
        self.stream = torch.cuda.Stream()

    def _voxelize(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                voxels_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                smooth_voxels_lig = add_noise_vox(voxels_lig, self.smooth_sigma)
                voxels_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)
                if self.training:
                    voxels_poc[:math.ceil(.2 * voxels_poc.shape[0])].zero_()
        return voxels_lig, smooth_voxels_lig, voxels_poc

    def __iter__(self):
        loader_it = iter(self.loader)
        try:
            next_batch = next(loader_it)
        except StopIteration:
            return

        next_voxels = self._voxelize(next_batch)

        for batch in loader_it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur_voxels = next_voxels
            # kick off next voxelization while model trains on current batch
            next_voxels = self._voxelize(batch)
            yield cur_voxels

        # last batch
        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_voxels

    def __len__(self):
        return len(self.loader)


@hydra.main(config_path="configs", config_name="config_train", version_base=None)
def main(cfg: DictConfig) -> None:
    # -----------------------------------------------------------
    # basic inits
    assert torch.cuda.is_available(), "not a good idea to train on cpu..."
    start_epoch = 0
    create_exp_dir(cfg)
    logger.info(f"n gpus available: {torch.cuda.device_count()}")
    logger.info(f"saving experiments in: {cfg.output_dir}")
    torch.set_default_dtype(torch.float32)
    logger.info("set matmul precision to high")
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)

    # resume?
    if cfg.resume is not None and os.path.isdir(cfg.resume):
        logger.info(f"resuming from: {cfg.resume}")
        resume = cfg.resume
        wjs_override = cfg.wjs
        wandb_override = cfg.wandb
        resume_epoch_override = cfg.resume_epoch
        cfg = OmegaConf.load(os.path.join(cfg.resume, "cfg.yaml"))
        cfg.output_dir, cfg.resume = resume, resume
        cfg.wandb = wandb_override
        cfg.wjs = wjs_override
        cfg.resume_epoch = resume_epoch_override

    logger.info("cfg:")
    logger.info(OmegaConf.to_yaml(cfg))

    # wandb?
    if cfg.wandb:
        wandb.init(
            project="voxbind",
            entity="eddy26",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.exp_name,
            dir=cfg.output_dir,
            resume="allow",
            settings=wandb.Settings(code_dir=".")
        )

    # -----------------------------------------------------------
    # create training objects
    # data loaders
    loader_train, loader_val, loader_sampling = create_dataloaders(cfg)
    n_train, n_val = len(loader_train.dataset.data), len(loader_val.dataset.data)
    logger.info(f"training/val set size: {n_train}/{n_val}")

    # model, criterion, optimizer
    device = torch.device("cuda")
    model = create_model(cfg)
    model.to(device)
    criterion = torch.nn.MSELoss(reduction="sum").to("cuda")
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model has {(n_params/1e6):.02f}M parameters")

    # voxelizer
    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    # optionally reload states of model, optimizer
    if cfg.resume is not None:
        logger.info("reloading states of model, optimizer")
        model, optimizer, start_epoch = load_checkpoint(
            model, cfg.output_dir, optimizer, best_model=False
        )
        os.system(f"cp {os.path.join(cfg.output_dir, 'checkpoint.pth.tar')} "
                  + f"{os.path.join(cfg.output_dir, f'checkpoint_{start_epoch}.pth.tar')}")
        logger.info(f"model trained for {start_epoch} epochs")
        if cfg.resume_epoch is not None:
            logger.info(f"overriding start_epoch {start_epoch} → {cfg.resume_epoch} (resume_epoch)")
            start_epoch = cfg.resume_epoch

    # DP/ema (exponential moving average)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    model.to(device)
    model_ema = ModelEma(model, decay=.999)

    # metrics
    metrics_denoise, _ = create_metrics_for_training()

    # pre-compute val voxels (deterministic since aug=False)
    val_cache = precompute_val_voxels(loader_val, voxelizer, cfg)

    # -----------------------------------------------------------
    # start training
    logger.info("start training...")
    for epoch in tqdm(range(start_epoch, start_epoch + cfg.num_epochs), desc="Epochs"):
        t0 = time.time()

        # train
        train_metrics = train(
            cfg, loader_train, voxelizer, model, criterion, optimizer, metrics_denoise, model_ema
        )

        # val
        val_metrics = val(
            cfg, loader_val, voxelizer, model_ema.module, criterion, metrics_denoise,
            val_cache=val_cache,
        )

        # sample
        if epoch > 0 and (epoch % 50 == 0 or epoch == cfg.num_epochs - 1):
            sample(cfg, loader_sampling, voxelizer, model_ema.module, epoch)

        # log and wandb
        log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
        if cfg.wandb:
            wandb.log({"train": train_metrics, "val": val_metrics})

        # save model
        save_checkpoint({
            "epoch": epoch,
            "metrics": {"train": train_metrics, "val": val_metrics},
            "cfg": cfg,
            "state_dict_ema": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, save_dir=cfg.output_dir)


def train(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    criterion: torch.nn,
    optimizer: torch.optim,
    metrics: torchmetrics.MetricCollection,
    model_ema: torch.nn,
) -> torch.Tensor:
    """Train one epoch of the model.

    Args:
        cfg (DictConfig): Configuration parameters for training.
        loader (torch.utils.data.DataLoader): Data loader for training data.
        voxelizer (Voxelizer): Voxelizer object for voxelizing input data.
        model (torch.nn): Model to be trained.
        criterion (torch.nn): Loss criterion for training.
        optimizer (torch.optim): Optimizer for updating model parameters.
        metrics (torchmetrics.MetricCollection): Metrics for evaluating model performance.
        model_ema (torch.nn): Exponential moving average model for model updates.

    Returns:
        torch.Tensor: Computed metrics for the epoch.
    """
    metrics.reset()
    model.train()
    train_miou_interval = max(1, int(cfg.get("train_miou_interval", 8)))

    prefetcher = VoxelPrefetcher(loader, voxelizer, cfg.smooth_sigma, training=True)
    for i, (voxels_lig, smooth_voxels_lig, voxels_poc) in enumerate(prefetcher):
        # fwd
        pred = model(smooth_voxels_lig, voxels_poc)
        loss = criterion(pred, voxels_lig)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # update model EMA
        model_ema.update(model)

        # update metrics
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
) -> torch.Tensor:
    """Evaluate model.

    Args:
        cfg (DictConfig): Configuration dictionary.
        loader (torch.utils.data.DataLoader): Data loader for evaluation dataset.
        voxelizer (Voxelizer): Voxelizer object for voxelizing input data.
        model (torch.nn): Model to be evaluated.
        criterion (torch.nn): Loss criterion for evaluation.
        metrics (torchmetrics.MetricCollection): Collection of metrics for evaluation.
        val_cache (dict, optional): Pre-computed val voxels from precompute_val_voxels().

    Returns:
        float: Computed metrics for evaluation.
    """
    metrics.reset()
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if val_cache is not None:
            voxels_lig_all = val_cache["voxels_lig"]
            voxels_poc_all = val_cache["voxels_poc"]
            n = voxels_lig_all.shape[0]
            for i, start in enumerate(range(0, n, cfg.bsz)):
                voxels_lig = voxels_lig_all[start:start + cfg.bsz].to(device)
                voxels_poc = voxels_poc_all[start:start + cfg.bsz].to(device)
                smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
                pred = model(smooth_voxels_lig, voxels_poc)
                loss = criterion(pred, voxels_lig)
                metrics.update(loss, pred, voxels_lig)
                if cfg.debug and i == 10:
                    break
        else:
            prefetcher = VoxelPrefetcher(loader, voxelizer, cfg.smooth_sigma, training=False)
            for i, (voxels_lig, smooth_voxels_lig, voxels_poc) in enumerate(prefetcher):
                pred = model(smooth_voxels_lig, voxels_poc)
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
    epoch: int
) -> None:
    """Sample ligands with WJS

    Args:
        cfg (DictConfig): Configuration settings for the sampling process.
        loader (torch.utils.data.DataLoader): DataLoader for loading the data.
        voxelizer (Voxelizer): Voxelizer object for voxelizing the input data.
        model (torch.nn): Model for generating samples.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    if torch.cuda.device_count() > 1:
        model = model.module
    model.eval()

    dirname = os.path.join(cfg.output_dir, f"samples_training/{epoch:02d}")
    makedir(dirname)

    for pocket_id, batch in enumerate(loader):
        if pocket_id == cfg.wjs.n_targets:
            break
        logger.info(f"| sampling pocket {pocket_id}")
        target_dirname = os.path.join(dirname, f"target_{pocket_id:02d}")

        # generate samples
        pocket, ligand_gt = batch["pocket"], batch["ligand"]
        sample_molecules(model, pocket, ligand_gt, voxelizer, target_dirname, cfg)

    return None


def _val_cache_path(cfg) -> str:
    """Return a deterministic cache path for val voxels based on voxelizer + dataset params."""
    key = {
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "cubes_around": cfg.vox.cubes_around,
        "ligand_radius": cfg.dset.ligand_radius,
        "pocket_radius": cfg.dset.pocket_radius,
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_{h}.pt")


def precompute_val_voxels(
    loader_val: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    cfg: DictConfig,
) -> dict:
    """Voxelize the entire val set (aug=False, so deterministic) and cache to disk.

    Returns a dict with keys 'voxels_lig' and 'voxels_poc', each a CPU tensor
    of shape (N, C, grid_dim, grid_dim, grid_dim).
    """
    cache_path = _val_cache_path(cfg)
    if os.path.isfile(cache_path):
        logger.info(f"loading pre-computed val voxels from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    logger.info("pre-computing val voxels (will be cached for future runs)...")
    all_lig, all_poc = [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=7).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=4).cpu())

    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
    }
    torch.save(cache, cache_path)
    logger.info(f"saved val voxels to {cache_path}")
    return cache


def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input voxels.

    Args:
        voxels (torch.Tensor): Input voxels.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy voxels.
    """
    if sigma > 0:
        return voxels + torch.empty_like(voxels).normal_(0, sigma)
    return voxels


def log_metrics(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    time: float
) -> None:
    """
    Logs the metrics for each epoch.

    Args:
        logger (logging.Logger): The logger object used for logging.
        epoch (int): The current epoch number.
        train_metrics (dict): The metrics for the training set.
        val_metrics (dict): The metrics for the validation set.
        sample_metrics (dict): The metrics for the sample set.
        time (float): The time taken for the epoch.

    Returns:
        None
    """

    all_metrics = [train_metrics, val_metrics]
    metrics_names = ["train", "val"]

    logger.info(f"epoch: {epoch} ({time:.2f}s)")
    for (split, metric) in zip(metrics_names, all_metrics):
        if metric is None:
            continue
        # str_ += "\n"
        str_ = f"[{split}]"
        for k, v in metric.items():
            if k == "loss":
                str_ += f" | {k}: {v:.2f}"
            else:
                str_ += f" | {k}: {v:.4f}"
        logger.info(str_)


if __name__ == "__main__":
    main()
