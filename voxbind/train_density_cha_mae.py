"""train_density_cha_mae.py — DDP pretraining for ChA-MAEViT (arXiv:2503.19331).

Token-drop multi-channel MAE. The `DensityChaMAE` model masks in TOKEN space
(Dynamic Channel-Patch masking) and reconstructs masked per-group voxel patches
through a lightweight channel-aware transformer decoder (pixel + Fourier loss) —
so masking and the decoder live inside the model, not in the data path. The
multi-channel voxel input is assembled by the SAME `MAEPrefetcher` used by the
ViT-MAE trainer; we consume only its clean `x_clean` (its block-mask output is
ignored). The EMA `encoder.*` slice (a grouped `DensityViT` with memory tokens)
drops into `VoxBind.density_encoder` and the frozen probe exactly like
`train_density_vit_mae.py`.

Launch
------
    cd voxbind
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \\
        train_density_cha_mae.py \\
        --config-name=config_train_atomblob_density_gradmag_cha_mae_40m_v5 \\
        exp_name=260612_cha_mae_gradmag_v5_pretrain \\
        output_dir=exps/260612_cha_mae_gradmag_v5_pretrain
"""

import contextlib
import logging
import os
import re
import time
from datetime import timedelta  # noqa: F401  (kept for parity / future use)

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from voxbind.dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.density_cha_mae import DensityChaMAE
from voxbind.models.density_mae import (
    gaussian_blur3d, gradient_magnitude3d, per_sample_zscore,
)
from voxbind.models.ema import ModelEma
from voxbind.utils.base_utils import create_exp_dir, seed_everything
from voxbind.voxelizer import Voxelizer

# Reuse the ViT-MAE trainer's infrastructure verbatim.
from voxbind.train_density_vit_mae import (
    AsyncCheckpointSaver, MAEPrefetcher, _amp_setup, _build_merged_atoms,
    _channel_layout, _cleanup_ddp, _setup_ddp, _unwrap, maybe_compile_model,
    precompute_val,
)

logger = logging.getLogger("train-density-cha-mae")

_METRIC_KEYS = ("L_recon", "L_pixel", "L_fourier", "mask_ratio")


# ── Model build ──────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig, device) -> DensityChaMAE:
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
    return model


# ── Train / val epochs ───────────────────────────────────────────────────────

def train_epoch_cha(cfg, prefetcher, model, optimizer, model_ema, device,
                    global_step, ema_source_model=None):
    model.train()
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    channels_last = bool(cfg.get("channels_last", False))
    _, amp_ctx_fn = _amp_setup(cfg)
    sums = {k: 0.0 for k in _METRIC_KEYS}
    grad_norm_sum, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    # MAEPrefetcher yields (x_in, x_clean, mask, target_str) — ChA uses x_clean only.
    for i, (_x_in, x_clean, _mask, _target_str) in enumerate(prefetcher):
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
    need_density = input_mode in ("density", "atomblob_density", "atomblob_merged_density")
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
    else:
        raise RuntimeError(f"unexpected input_mode={input_mode!r}")
    if with_gradmag:
        x = torch.cat([x, g_clean], dim=1)
    return x


def val_epoch_cha(cfg, model, val_cache, device):
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


def log_metrics(epoch, train_metrics, val_metrics, dt):
    logger.info(f"epoch: {epoch} ({dt:.2f}s)")
    for split, m in zip(["train", "val"], [train_metrics, val_metrics]):
        if m is None:
            continue
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
        logger.info(f"[{split}] {parts}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config_train_atomblob_density_gradmag_cha_mae_40m_v5",
            version_base=None)
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
            _wandb_name = re.sub(r"^\d{6}_", "", cfg.exp_name)
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

    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct)
    n_in_cfg = int(cfg.model.get("n_in_channels", 1))
    if n_in_cfg != layout["n_in"]:
        raise RuntimeError(
            f"input_mode={input_mode!r} with_gradmag={with_gradmag} expects "
            f"model.n_in_channels={layout['n_in']}, got {n_in_cfg}")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    if bool(cfg.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last_3d)
        if is_main:
            logger.info("channels_last_3d memory format enabled")
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_enc = sum(p.numel() for p in model.encoder.parameters())
        logger.info(
            f"DensityChaMAE {(n_params/1e6):.02f}M params "
            f"(encoder {(n_enc/1e6):.02f}M) — groups={model.channel_groups}, "
            f"n_memory_tokens={model.n_memory_tokens}, dcp={model.dcp_strategy}, "
            f"patch_mask_ratio={model.patch_mask_ratio}, lambda_fourier={model.lambda_fourier}")

    if bool(cfg.get("optimizer", {}).get("fused", False)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
                                      betas=(0.99, 0.999), eps=1e-8, fused=True)
        if is_main:
            logger.info("fused AdamW enabled (betas=(0.99,0.999), eps=1e-8)")
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()

    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

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
        # find_unused_parameters=True: the encoder's to-voxel `decoder_proj` is only
        # exercised on the dense downstream path, NOT by the token-drop reconstruction
        # objective, so it never enters the loss graph during ChA pretraining. It is
        # kept in the module (and state_dict) so the encoder.* slice stays a drop-in
        # DensityViT; it simply trains downstream (diffusion) / is bypassed by the probe.
        model_train = torch.nn.parallel.DistributedDataParallel(
            model_train, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, gradient_as_bucket_view=True,
        )
    model_ema = ModelEma(model, decay=cfg.mae.ema_decay,
                         foreach=bool(cfg.get("ema", {}).get("foreach", False)))

    # Val cache (full set on every rank; same cache as the ViT-MAE trainer)
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
        logger.info("start ChA-MAEViT pre-training...")
    epochs_iter = tqdm(range(start_epoch, start_epoch + cfg.num_epochs),
                       desc="Epochs", disable=not is_main)
    ckpt_saver = AsyncCheckpointSaver() if is_main else None

    for epoch in epochs_iter:
        t0 = time.time()
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # Reuse MAEPrefetcher purely for the clean multi-channel assembly; its
        # block-mask output is ignored (ChA masks in token space inside the model).
        prefetcher = MAEPrefetcher(
            loader_train, voxelizer,
            sigma_blur_vox_range=sigma_blur_vox_range,
            sigma_noise=cfg.mae.sigma_noise,
            block_size=cfg.mae.block_size,
            mask_ratio=cfg.mae.mask_ratio,
            pretext_style="mae",
            density_source=str(cfg.mae.get("density_source", "synthetic")),
            input_mode=input_mode,
            mask_strategy=str(cfg.mae.get("mask_strategy", "uniform")),
            mask_atom_tau=float(cfg.mae.get("mask_atom_tau", 1.0)),
            with_gradmag=with_gradmag,
            gradmag_reconstruct=gradmag_reconstruct,
            gradmag_noise=bool(cfg.mae.get("gradmag_noise", False)),
        )
        train_metrics, global_step = train_epoch_cha(
            cfg, prefetcher, model_train, optimizer, model_ema, device, global_step,
            ema_source_model=model,
        )

        val_metrics = None
        if epoch % max(1, int(cfg.mae.val_every)) == 0:
            val_metrics = val_epoch_cha(cfg, model_ema.module, val_cache, device)

        if is_main:
            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb and wandb.run is not None:
                payload = {"train": train_metrics, "epoch": epoch}
                if val_metrics is not None:
                    payload["val"] = val_metrics
                try:
                    wandb.log(payload, step=global_step)
                except Exception as e:
                    logger.warning(f"wandb.log failed: {e}")

            encoder_sd = {k: v for k, v in model_ema.module.state_dict().items()
                          if k.startswith("encoder.")}
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
