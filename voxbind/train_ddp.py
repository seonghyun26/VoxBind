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
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import timedelta

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
from voxbind.models.urepa import (BottleneckTap, UREPAAlignment, build_teacher_input,
                                  load_frozen_cdg, teacher_tokens)
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
    # 2-hour NCCL timeout (default is 10 min). Transient rank-0 stalls (slow
    # disk write, wandb hiccup) won't trip the watchdog and kill training.
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=2)
    )
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class AsyncCheckpointSaver:
    """Atomic + threaded checkpoint writer (rank 0 only).

    Each `save()` deep-copies the in-memory state, kicks off a background
    thread that writes to `<path>.tmp` and then renames to `<path>` (atomic
    on POSIX). The next `save()` first joins the previous thread, so we
    never have two writes stacked, and the latest call always reflects the
    final on-disk file. `wait()` blocks until the in-flight save completes.
    """

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()

    @staticmethod
    def _write(state: dict, save_dir: str, chkp_name: str) -> None:
        tmp = os.path.join(save_dir, chkp_name + ".tmp")
        final = os.path.join(save_dir, chkp_name)
        try:
            torch.save(state, tmp)
            os.replace(tmp, final)  # atomic on POSIX
        except Exception as e:
            logger.warning(f"checkpoint write failed: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def save(self, state: dict, save_dir: str, chkp_name: str = "checkpoint.pth.tar") -> None:
        # Wait for any in-flight save first; ensures linear write order.
        self.wait()
        # Deep-copy so the main thread is free to mutate model/optimizer state
        # while the background thread serializes a frozen snapshot.
        frozen = copy.deepcopy(state)
        with self._lock:
            self._thread = threading.Thread(
                target=self._write,
                args=(frozen, save_dir, chkp_name),
                daemon=False,  # don't drop the write if main exits
            )
            self._thread.start()

    def wait(self) -> None:
        with self._lock:
            t = self._thread
        if t is not None:
            t.join()


class VoxelPrefetcher:
    """Overlaps GPU voxelization of batch N+1 with model compute on batch N.

    Identical to train.py.VoxelPrefetcher — duplicated here only to keep this
    file self-contained for the smoke test (no cross-imports of private
    helpers from train.py).
    """

    def __init__(self, loader, voxelizer, smooth_sigma, training=True, with_density=False,
                 raw_density=False):
        self.loader = loader
        self.voxelizer = voxelizer
        self.smooth_sigma = smooth_sigma
        self.training = training
        self.with_density = with_density
        # U-REPA: the teacher needs the density even though the DENOISER is density-free
        # (model.with_density=False). raw_density decouples "deliver the crop" from "the
        # model consumes it", and keeps the crop UNZEROED — U-REPA must *exclude* samples
        # without a map, not feed them a zero volume. `avail` is yielded so it can.
        self.raw_density = raw_density
        self.stream = torch.cuda.Stream()

    def _voxelize(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                voxels_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                smooth_voxels_lig = add_noise_vox(voxels_lig, self.smooth_sigma)
                voxels_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)
                if self.training:
                    voxels_poc[:math.ceil(.2 * voxels_poc.shape[0])].zero_()
                density, avail = None, None
                if (self.with_density or self.raw_density) and "xray_density" in batch:
                    density = batch["xray_density"].to(
                        self.voxelizer.device, non_blocking=True
                    ).unsqueeze(1)
                    if "xray_available" in batch:
                        avail = batch["xray_available"].to(
                            self.voxelizer.device, non_blocking=True
                        ).bool()
                        if not self.raw_density:
                            density = density * avail.view(-1, 1, 1, 1, 1).float()
        return voxels_lig, smooth_voxels_lig, voxels_poc, density, avail

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

    # -----------------------------------------------------------
    # data loaders (train uses DistributedSampler; val/sampling are not sharded)
    loader_train, loader_val, loader_sampling = create_dataloaders(
        cfg, distributed=True, rank=rank, world_size=world_size
    )
    n_train, n_val = len(loader_train.dataset), len(loader_val.dataset)
    if is_main:
        logger.info(f"training/val set size: {n_train}/{n_val}")
        _accum = max(1, int(cfg.get("accum_steps", 1)))
        logger.info(f"train batches per rank: {len(loader_train)} "
                    f"(effective batch = {cfg.bsz * world_size * _accum}, "
                    f"accum_steps={_accum})")

    # model, criterion, optimizer — identical init across ranks (DDP also
    # broadcasts rank-0 params on wrap, but matching seeds keeps the pre-wrap
    # state identical too).
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    model = create_model(cfg, device=device)

    # ── warm start from a VANILLA (density-free) VoxBind checkpoint ──────────────
    # Loads only the shared denoiser tensors (unet3d / ligand_encoder / pocket_encoder
    # / final_ligand) and leaves the density branch exactly as built: frozen pretrained
    # encoder + ZERO-INIT density_proj. Because density_proj is zero-init, the model
    # starts functionally IDENTICAL to the source checkpoint and grows the density
    # correction from zero on top of an already-trained denoiser.
    #
    # Weights only — no optimizer state, no epoch. This is deliberately NOT `resume`:
    # resume reloads the source run's cfg.yaml (see above), which would restore the
    # vanilla config and delete the density branch. Skipped when resuming, since a
    # resumed checkpoint already carries the full (warm-started) state.
    pretrained_path = cfg.get("pretrained_path", None)
    if pretrained_path:
        if cfg.resume is not None:
            if is_main:
                logger.info(">> resume set — ignoring pretrained_path (checkpoint has full state)")
        else:
            _ck = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            _sd = _ck.get("state_dict_ema") or _ck.get("state_dict")
            if _sd is None:
                raise KeyError(
                    f"{pretrained_path}: no 'state_dict_ema'/'state_dict' key "
                    f"(found {list(_ck.keys())})"
                )
            _missing, _unexpected = model.load_state_dict(_sd, strict=False)
            # everything absent from a vanilla checkpoint MUST be density-branch;
            # anything else means the architectures don't actually match.
            _stray = [k for k in _missing
                      if not k.startswith(("density_encoder.", "density_proj.", "context_proj."))]
            if _unexpected or _stray:
                raise RuntimeError(
                    f"warm start from {pretrained_path} is not architecture-compatible: "
                    f"unexpected={_unexpected[:8]} stray_missing={_stray[:8]}"
                )
            if is_main:
                logger.info(
                    f">> warm start from {pretrained_path} "
                    f"(epoch {_ck.get('epoch')}): loaded {len(_sd)} denoiser tensors, "
                    f"{len(_missing)} density-branch tensors left at init "
                    f"(zero-init density_proj ⇒ output identical to source at step 0)"
                )
            del _ck, _sd

    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    # ── U-REPA: frozen CDG teacher + trainable projector on the U-Net bottleneck ──
    # Additive: with cfg.urepa absent/disabled everything below is skipped and the run is
    # bit-identical to a stock denoiser run. Inference never constructs any of it.
    teacher = align = tap = None
    urepa_stage1 = False
    ucfg = cfg.get("urepa", None)
    if ucfg is not None and bool(ucfg.get("enabled", False)):
        teacher = load_frozen_cdg(ucfg.exp_dir, int(ucfg.get("epoch", 49)), device)
        _bot = model.unet3d.middle.res1.conv1.in_channels      # read, never assume 32*4
        align = UREPAAlignment(
            unet_ch=_bot, cdg_dim=teacher.dim,
            unet_grid=cfg.vox.grid_dim // (2 ** (len(cfg.model.ch_mults) - 1)),
            cdg_grid=teacher.g_p,
            tau=float(ucfg.get("tau", 0.1)), center=bool(ucfg.get("center", True)),
            mode=str(ucfg.get("mode", "relkl")),
            repa_weight=float(ucfg.get("repa_weight", 0.0)),
            ml_weight=float(ucfg.get("ml_weight", 1.0)),
            sampling=str(ucfg.get("sampling", "split")),
            tokens_per_sample=int(ucfg.get("tokens_per_sample", 128)),
            w_intra=float(ucfg.get("w_intra", 1.0)),
            w_inter=float(ucfg.get("w_inter", 1.0)),
        ).to(device)
        tap = BottleneckTap(model.unet3d.middle)
        urepa_stage1 = int(ucfg.get("stage1_epochs", 0)) > 0
        if is_main:
            logger.info(
                f">> U-REPA on: teacher={ucfg.exp_dir} dim={teacher.dim} "
                f"(frozen, {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M) | "
                f"bottleneck={_bot} projector={sum(p.numel() for p in align.parameters())/1e6:.2f}M "
                f"| lam={ucfg.get('lam', 0.5)} sampling={ucfg.get('sampling', 'split')} "
                f"apo={ucfg.get('apo', True)} stage1_epochs={ucfg.get('stage1_epochs', 0)} "
                f"| L_align = {ucfg.get('repa_weight', 0.0)}*REPA + {ucfg.get('ml_weight', 1.0)}*manifold"
            )

    if align is None:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    else:
        # Two groups so Stage 1 can hold the U-Net at lr=0 while the projector trains.
        optimizer = AdamW([
            {"params": list(model.parameters()), "name": "unet",
             "lr": 0.0 if urepa_stage1 else float(cfg.lr)},
            {"params": list(align.parameters()), "name": "urepa_proj",
             "lr": float(ucfg.get("proj_lr", 1e-4))},
        ], lr=cfg.lr, weight_decay=cfg.wd)
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
    global_step = 0
    if cfg.resume is not None:
        if is_main:
            logger.info("reloading states of model, optimizer")
        model, optimizer, start_epoch = load_checkpoint(
            model, cfg.output_dir, optimizer, best_model=False,
            map_location=device,   # avoid ghost CUDA contexts on rank-0's GPU
        )
        # Re-read the raw checkpoint to recover global_step (load_checkpoint
        # only returns model/optimizer/epoch). Falls back to 0 for older ckpts.
        _ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pth.tar")
        try:
            _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
            global_step = int(_ckpt.get("global_step", 0))
            del _ckpt
        except Exception:
            global_step = 0
        if is_main:
            os.system(
                f"cp {os.path.join(cfg.output_dir, 'checkpoint.pth.tar')} "
                f"{os.path.join(cfg.output_dir, f'checkpoint_{start_epoch}.pth.tar')}"
            )
            logger.info(f"model trained for {start_epoch} epochs (global_step={global_step})")
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
    if world_size > 1 and align is not None:
        # The projector lives outside `model`, so it needs its own reducer — otherwise
        # each rank keeps a privately-updated copy and the ranks silently diverge.
        align = torch.nn.parallel.DistributedDataParallel(
            align, device_ids=[local_rank], output_device=local_rank,
        )
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
    ckpt_saver = AsyncCheckpointSaver() if is_main else None
    for epoch in epochs_iter:
        t0 = time.time()

        # DistributedSampler needs set_epoch for proper shuffling
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # train (all ranks)
        if urepa_stage1 and epoch == int(cfg.urepa.get("stage1_epochs", 0)):
            # Stage 1 -> 2: release the U-Net. lr=0 (not requires_grad=False) is what
            # freezes it, so DDP always has trainable params to reduce and never hits
            # "module has no parameter that requires a gradient".
            for g in optimizer.param_groups:
                if g.get("name") == "unet":
                    g["lr"] = float(cfg.lr)
            urepa_stage1 = False
            if is_main:
                logger.info(f">> U-REPA stage 2 at epoch {epoch}: U-Net lr -> {cfg.lr}")

        train_metrics, global_step = train(
            cfg, loader_train, voxelizer, model, criterion, optimizer,
            metrics_denoise, model_ema,
            global_step=global_step,
            align=align, teacher=teacher, tap=tap,
        )

        # val (ALL ranks — torchmetrics auto-syncs at .compute(), so rank-0-only
        # val deadlocks the all-reduce. Each rank reruns the same val_cache;
        # wasteful but the deterministic data + cheap per-batch math make this
        # cost negligible vs the deadlock risk.)
        val_metrics = val(
            cfg, loader_val, voxelizer, model_ema.module, criterion, metrics_denoise,
            val_cache=val_cache,
        )

        # sample + log + save (rank 0 only — no torchmetrics here, no NCCL.)
        if is_main:
            if epoch > 0 and (epoch % 100 == 0 or epoch == cfg.num_epochs - 1):
                sample(cfg, loader_sampling, voxelizer, model_ema.module, epoch)

            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb:
                try:
                    wandb.log(
                        {"train": train_metrics, "val": val_metrics, "epoch": epoch},
                        step=global_step,
                    )
                except Exception as e:
                    logger.warning(f"wandb.log failed: {e}")

            ckpt_saver.save({
                "epoch": epoch,
                "global_step": global_step,
                "metrics": {"train": train_metrics, "val": val_metrics},
                "cfg": cfg,
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                # projector only — the teacher is frozen and reloaded from its own zoo dir,
                # and inference never needs either. Absent for non-U-REPA runs.
                **({"urepa_projector": _unwrap(align).state_dict()} if align is not None else {}),
            }, save_dir=cfg.output_dir)

        # Sync ranks per epoch. With async checkpoint save, rank 0 reaches
        # this barrier almost immediately after kicking off the save thread,
        # so the barrier no longer blocks on disk I/O. The 2-hour NCCL
        # timeout (see _setup_ddp) is the final safety net.
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        torch.cuda.empty_cache()

    # Make sure the final checkpoint is fully on disk before we exit.
    if is_main and ckpt_saver is not None:
        ckpt_saver.wait()

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
    global_step: int = 0,
    align=None,
    teacher=None,
    tap=None,
) -> tuple:
    """Train one epoch. Returns (metrics_dict, new_global_step).

    `global_step` is incremented once per optimizer step so callers can use
    it as a wandb x-axis that stays monotonic across resumes.
    """
    metrics.reset()
    model.train()
    train_miou_interval = max(1, int(cfg.get("train_miou_interval", 8)))
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))

    with_density = bool(cfg.model.get("with_density", False))
    # Leak-removal mask config (read off the model so it tracks the checkpoint).
    _mdl = _unwrap(model)
    enc_sees_ligand = bool(getattr(_mdl, "density_encoder_sees_ligand", False))
    mask_lig = bool(getattr(_mdl, "density_mask_ligand", False))
    mask_thr = float(getattr(_mdl, "density_mask_threshold", 0.2))
    mask_dil = int(getattr(_mdl, "density_mask_dilate", 2))
    urepa_on = align is not None
    prefetcher = VoxelPrefetcher(
        loader, voxelizer, cfg.smooth_sigma,
        training=True, with_density=with_density, raw_density=urepa_on,
    )
    ucfg = cfg.get("urepa", {}) if urepa_on else {}
    u_lam = float(ucfg.get("lam", 0.5))
    u_apo = bool(ucfg.get("apo", True))
    u_amp = bool(ucfg.get("teacher_amp", True))
    align_sum, align_n, ent_sum, ratio_sum = 0.0, 0, 0.0, 0.0
    # Gradient accumulation: forward/backward on `accum_steps` micro-batches,
    # then a single optimizer step -> effective batch = bsz * world_size *
    # accum_steps at the memory cost of one micro-batch. accum_steps=1 (default)
    # steps every batch -> an exact no-op vs. the pre-accumulation loop.
    n_batches = len(prefetcher)
    optimizer.zero_grad(set_to_none=True)
    for i, (voxels_lig, smooth_voxels_lig, voxels_poc, density, avail) in enumerate(prefetcher):
        # last micro-batch of an accumulation group, or of the epoch -> step now
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_batches)

        # On non-step micro-batches, skip the DDP gradient all-reduce: grads
        # accumulate locally; the step micro-batch's backward syncs the sum.
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        with sync_ctx:
            dens_mask = (_mdl.ligand_occupancy_mask(voxels_lig, mask_thr, mask_dil)
                         if (mask_lig and density is not None) else None)
            # U-REPA feeds the teacher raw density, but the DENOISER stays density-free.
            _dens_in = None if urepa_on and not with_density else density
            # REFERENCE mode: the frozen encoder sees the clean available ligand (sampling feeds it the
            # reference ligand, so training must match that distribution — feeding the noisy
            # y here would train under a condition inference never reproduces).
            _enc_lig = voxels_lig if enc_sees_ligand else None
            pred = model(smooth_voxels_lig, voxels_poc, density=_dens_in, dens_mask=dens_mask,
                         enc_ligand=_enc_lig)
            loss = criterion(pred, voxels_lig)

            if urepa_on and density is not None:
                sel = (avail if avail is not None
                       else torch.ones(voxels_lig.shape[0], dtype=torch.bool,
                                       device=voxels_lig.device))
                # Every rank MUST call `align` every step: a rank that skips it leaves its
                # DDP reduction unfinished and the next iteration deadlocks. With
                # dset.subset_xray_only=true `sel` is all-True anyway; the zero-weight
                # fallback just makes a partially-covered dataset safe too.
                n_sel = int(sel.sum())
                if True:
                    scale = 1.0 if n_sel else 0.0
                    idx = (sel.nonzero(as_tuple=True)[0] if n_sel
                           else torch.zeros(1, dtype=torch.long, device=sel.device))
                    # teacher sees the student's OWN augmented frame -> the 8³ token grid
                    # and the 8³ bottleneck correspond voxel-for-voxel, no canonical-cache
                    # rotation problem.
                    t_in = build_teacher_input(voxels_lig[idx], voxels_poc[idx],
                                               density[idx], apo=u_apo)
                    tgt = teacher_tokens(teacher, t_in, amp=u_amp)
                    l_align, a_stats = align(tap.feature[idx], tgt, return_stats=True)
                    loss = loss + (u_lam * scale) * l_align
                    if n_sel:
                        align_sum += float(l_align.detach()); align_n += 1
                        ent_sum += float(a_stats["teacher_entropy"])
                        ratio_sum += float(a_stats.get("entropy_ratio", float("nan")))
            loss.backward()

        if is_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(_unwrap(model))
            global_step += 1

        metrics.update(
            loss.detach(), pred.detach(), voxels_lig,
            update_miou=(i % train_miou_interval == 0)
        )

        if cfg.debug and i == 10:
            break

    out = metrics.compute()
    if align_n:
        out["urepa_align"] = align_sum / align_n
        out["urepa_teacher_entropy"] = ent_sum / align_n
        # ->1.0 means the teacher target is UNIFORM: the loss teaches nothing and the
        # student drives KL->0 by matching uniform. Lower tau and/or keep center=true.
        out["urepa_entropy_ratio"] = ratio_sum / align_n
        out["urepa_frac_batches"] = align_n / max(n_batches, 1)
    return out, global_step


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
    mask_lig = bool(getattr(model, "density_mask_ligand", False))
    mask_thr = float(getattr(model, "density_mask_threshold", 0.2))
    mask_dil = int(getattr(model, "density_mask_dilate", 2))
    _enc_sees_lig = bool(getattr(model, "density_encoder_sees_ligand", False))

    with torch.no_grad():
        if val_cache is not None:
            voxels_lig_all = val_cache["voxels_lig"]
            voxels_poc_all = val_cache["voxels_poc"]
            density_all = val_cache.get("density") if (with_density or _enc_sees_lig) else None
            n = voxels_lig_all.shape[0]
            for i, start in enumerate(range(0, n, cfg.bsz)):
                voxels_lig = voxels_lig_all[start:start + cfg.bsz].to(device)
                voxels_poc = voxels_poc_all[start:start + cfg.bsz].to(device)
                density = density_all[start:start + cfg.bsz].to(device) if density_all is not None else None
                smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
                dens_mask = (model.ligand_occupancy_mask(voxels_lig, mask_thr, mask_dil)
                             if (mask_lig and density is not None) else None)
                pred = model(smooth_voxels_lig, voxels_poc, density=density, dens_mask=dens_mask,
                             enc_ligand=(voxels_lig if _enc_sees_lig else None))
                loss = criterion(pred, voxels_lig)
                metrics.update(loss, pred, voxels_lig)
                if cfg.debug and i == 10:
                    break
        else:
            prefetcher = VoxelPrefetcher(
                loader, voxelizer, cfg.smooth_sigma,
                training=False, with_density=with_density, raw_density=_enc_sees_lig,
            )
            for i, (voxels_lig, smooth_voxels_lig, voxels_poc, density, _av) in enumerate(prefetcher):
                dens_mask = (model.ligand_occupancy_mask(voxels_lig, mask_thr, mask_dil)
                             if (mask_lig and density is not None) else None)
                pred = model(smooth_voxels_lig, voxels_poc, density=density, dens_mask=dens_mask,
                             enc_ligand=(voxels_lig if getattr(model, "density_encoder_sees_ligand", False)
                                         else None))
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

    for pocket_id, batch in enumerate(loader):
        if pocket_id == cfg.wjs.n_targets:
            break
        makedir(dirname)  # after the guard — skip empty dirs when n_targets=0
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
        # part of the key: reference-ligand mode caches density that a with_density=false run
        # would not, so the two must not share a cache file.
        "enc_sees_ligand": bool(cfg.model.get("density_encoder_sees_ligand", False)),
        # val-set identity — without these a changed val subset would silently
        # reuse a stale cache (e.g. the old all-zero-density val voxels).
        # crops_dir is part of that identity: the noise-control run reuses the
        # same subset_* config but a different crop source, so without it the
        # noise run would silently load the real-density val cache.
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_val_n": cfg.dset.get("subset_val_n", None),
        "subset_xray_only": bool(cfg.dset.get("subset_xray_only", False)),
        "crops_dir": cfg.dset.get("crops_dir", ""),
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
            # reference-ligand mode has model.with_density=false but still needs the crop for the
            # frozen encoder, so cache density whenever the config asks for either.
            if (with_density or bool(cfg.model.get("density_encoder_sees_ligand", False))) \
                    and "xray_density" in batch:
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
