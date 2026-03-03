"""
train_density.py — Finetune VoxBind with simulated EM density conditioning
===========================================================================
Proof-of-concept: adds a density_encoder branch to a pretrained VoxBind model
and finetunes *only* that encoder while freezing everything else.

The density map is generated on-the-fly from the pocket atom coordinates using
the pdb2vol Gaussian-splat algorithm (Wriggers 2010), implemented in
voxbind/dataset/pdb2vol.py.

Usage
-----
    cd voxbind
    python train_density.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --sigma 1.5 \\
        --epochs 30 \\
        --bsz 32

After training the checkpoint is written to:
    exps/density_poc/checkpoint.pth.tar

Evaluation
----------
Run the baseline and density-conditioned model on the test set and compare
metrics (loss, mIoU) to quantify whether density conditioning helps:

    python train_density.py --pretrained_path exps/exp_sig0.9 --eval_only \\
        --density_ckpt exps/density_poc/checkpoint.pth.tar
"""

import argparse
import logging
import math
import os
import time

import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

from voxbind.constants import N_LIGAND_ELEMENTS, N_POCKET_ELEMENTS
from voxbind.dataset import create_dataloaders
from voxbind.dataset.pdb2vol import coords_to_density
from voxbind.metrics import create_metrics_for_training
from voxbind.models.voxbind import VoxBind
from voxbind.utils.base_utils import makedir, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("density_poc")


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma > 0:
        return voxels + torch.empty_like(voxels).normal_(0, sigma)
    return voxels


def load_pretrained_voxbind(pretrained_path: str, with_density: bool, device: torch.device) -> VoxBind:
    """Load the pretrained VoxBind model, then attach density_encoder if requested.

    Uses strict=False so that the density_encoder (not present in the original
    checkpoint) is left with its random initialisation.
    """
    chkp = torch.load(
        os.path.join(pretrained_path, "checkpoint.pth.tar"),
        map_location=device,
        weights_only=False,
    )

    # Defaults matching voxbind/configs/model/voxbind.yaml
    ch_mults = (1, 2, 2, 4)
    is_attn = (False, False, True, True)
    n_blocks = 2
    smooth_sigma = 0.9

    # Infer n_channels from the checkpoint weights directly (reliable)
    sd_peek = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    sd_peek = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd_peek.items()}
    enc_w = sd_peek.get("ligand_encoder.conv2.weight",
                        sd_peek.get("ligand_encoder.block2.weight", None))
    n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
    n_groups   = min(16, n_channels // 2)

    # Override from saved cfg if present
    cfg_saved = chkp.get("cfg", None)
    if cfg_saved is not None:
        try:
            from omegaconf import OmegaConf
            if not isinstance(cfg_saved, dict):
                cfg_saved = OmegaConf.to_container(cfg_saved, resolve=True)
            smooth_sigma = cfg_saved.get("smooth_sigma", smooth_sigma)
            mc = cfg_saved.get("model", {})
            ch_mults   = tuple(mc.get("ch_mults",  list(ch_mults)))
            is_attn    = tuple(mc.get("is_attn",   list(is_attn)))
            n_blocks   = mc.get("n_blocks", n_blocks)
        except Exception:
            pass

    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=n_channels,
        n_groups=n_groups,
        ch_mults=ch_mults,
        is_attn=is_attn,
        n_blocks=n_blocks,
        smooth_sigma=smooth_sigma,
        with_density=with_density,
        verbose=True,
    )

    sd = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    # Strip DataParallel / torch.compile prefixes
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.info(f"  Keys not in checkpoint (will be trained): {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected}")

    return model, smooth_sigma


# ── Prefetcher with density ───────────────────────────────────────────────────

class DensityPrefetcher:
    """Like VoxelPrefetcher but also generates a pocket density map per batch.

    Adds one extra tensor to the yielded tuple:
        (voxels_lig, smooth_voxels_lig, voxels_poc, density)
    where density is (B, 1, G, G, G) float32 from pdb2vol.coords_to_density.
    """

    def __init__(self, loader, voxelizer, smooth_sigma, pdb2vol_sigma, training=True):
        self.loader       = loader
        self.voxelizer    = voxelizer
        self.smooth_sigma = smooth_sigma
        self.pdb2vol_sigma = pdb2vol_sigma
        self.training     = training
        self.stream       = torch.cuda.Stream()

    def _process(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                voxels_lig = self.voxelizer.forward(batch["ligand"], num_channels=7)
                smooth_vox_lig = add_noise_vox(voxels_lig, self.smooth_sigma)
                voxels_poc = self.voxelizer.forward(batch["pocket"], num_channels=4)
                if self.training:
                    voxels_poc[:math.ceil(.2 * voxels_poc.shape[0])].zero_()

                # Generate density from pocket coords (already centred + augmented)
                coords   = batch["pocket"]["coords"].cuda()       # (B, 2000, 3)
                atoms_ch = batch["pocket"]["atoms_channel"].cuda() # (B, 2000)
                density  = coords_to_density(
                    coords, atoms_ch,
                    grid_dim=voxels_poc.shape[-1],
                    sigma=self.pdb2vol_sigma,
                )

        return voxels_lig, smooth_vox_lig, voxels_poc, density

    def __iter__(self):
        loader_it = iter(self.loader)
        try:
            next_batch = next(loader_it)
        except StopIteration:
            return

        next_out = self._process(next_batch)

        for batch in loader_it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur_out = next_out
            next_out = self._process(batch)
            yield cur_out

        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_out

    def __len__(self):
        return len(self.loader)


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, prefetcher, criterion, optimizer, metrics, smooth_sigma, debug=False):
    metrics.reset()
    model.train()
    # Only the density_encoder is in train mode; the rest stay in eval
    if hasattr(model, "density_encoder"):
        model.density_encoder.train()

    for i, (voxels_lig, smooth_vox_lig, voxels_poc, density) in enumerate(prefetcher):
        pred = model(smooth_vox_lig, voxels_poc, density=density)
        loss = criterion(pred, voxels_lig)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.update(loss.detach(), pred.detach(), voxels_lig,
                       update_miou=(i % 8 == 0))
        if debug and i == 10:
            break

    return metrics.compute()


@torch.no_grad()
def eval_epoch(model, prefetcher, criterion, metrics, use_density=True, debug=False):
    metrics.reset()
    model.eval()

    for i, (voxels_lig, smooth_vox_lig, voxels_poc, density) in enumerate(prefetcher):
        density_in = density if use_density else None
        pred = model(smooth_vox_lig, voxels_poc, density=density_in)
        loss = criterion(pred, voxels_lig)
        metrics.update(loss, pred, voxels_lig)
        if debug and i == 10:
            break

    return metrics.compute()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VoxBind density-conditioned finetuning PoC")
    p.add_argument("--pretrained_path", required=True,
                   help="Path to pretrained VoxBind experiment directory")
    p.add_argument("--data_dir", default="dataset/data",
                   help="CrossDocked data directory (default: dataset/data)")
    p.add_argument("--out_dir", default="exps/density_poc",
                   help="Output directory for density PoC checkpoint")
    p.add_argument("--sigma", type=float, default=1.5,
                   help="pdb2vol Gaussian sigma in Å (default: 1.5)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of finetune epochs (default: 30)")
    p.add_argument("--bsz", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for density_encoder only (default: 1e-4)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true",
                   help="Quick smoke test (10 batches per epoch)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; evaluate checkpoint and baseline")
    p.add_argument("--density_ckpt", default=None,
                   help="Path to density PoC checkpoint for --eval_only")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    makedir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("VoxBind Density PoC")
    logger.info(f"  pretrained : {args.pretrained_path}")
    logger.info(f"  out_dir    : {args.out_dir}")
    logger.info(f"  sigma      : {args.sigma} Å")
    logger.info(f"  epochs     : {args.epochs}")
    logger.info(f"  bsz        : {args.bsz}")
    logger.info("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    model, smooth_sigma = load_pretrained_voxbind(
        args.pretrained_path, with_density=True, device=device
    )
    model.to(device)

    if args.eval_only:
        _run_eval_only(args, model, device, smooth_sigma)
        return

    # ── Freeze everything except density_encoder ──────────────────────────────
    n_frozen = 0
    for name, param in model.named_parameters():
        if "density_encoder" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
            n_frozen += param.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Frozen params : {n_frozen / 1e6:.2f}M")
    logger.info(f"  Trainable params (density_encoder) : {n_trainable / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2
    )
    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from voxbind.voxelizer import Voxelizer

    # Minimal config to reuse create_dataloaders
    cfg = OmegaConf.create({
        "dset": {
            "dset_name": "crossdocked",
            "data_dir": args.data_dir,
            "ligand_radius": 0.5,
            "pocket_radius": -1,
        },
        "vox": {"grid_dim": 64, "resolution": 0.25, "cubes_around": 8},
        "wjs": {"split": "val", "n_targets": 10, "n_samples_per_pocket": 10},
        "bsz": args.bsz,
        "num_workers": args.num_workers,
        "aug": True,
        "smooth_sigma": smooth_sigma,
        "debug": args.debug,
    })

    loader_train, loader_val, _ = create_dataloaders(cfg)

    voxelizer = Voxelizer(
        grid_dim=64, resolution=0.25, cubes_around=8, device=device
    )

    # ── Training ──────────────────────────────────────────────────────────────
    metrics_train, _ = create_metrics_for_training()
    metrics_val,   _ = create_metrics_for_training()

    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        t0 = time.time()

        prefetcher_train = DensityPrefetcher(
            loader_train, voxelizer, smooth_sigma, args.sigma, training=True
        )
        train_m = train_epoch(
            model, prefetcher_train, criterion, optimizer, metrics_train,
            smooth_sigma, debug=args.debug
        )

        prefetcher_val = DensityPrefetcher(
            loader_val, voxelizer, smooth_sigma, args.sigma, training=False
        )
        val_m = eval_epoch(model, prefetcher_val, criterion, metrics_val, debug=args.debug)

        elapsed = time.time() - t0
        logger.info(
            f"epoch {epoch:3d} ({elapsed:.1f}s) | "
            f"train loss={train_m['loss']:.2f}  miou={train_m.get('miou', float('nan')):.4f} | "
            f"val   loss={val_m['loss']:.2f}  miou={val_m.get('miou', float('nan')):.4f}"
        )

        # Save checkpoint
        val_loss = val_m["loss"].item() if hasattr(val_m["loss"], "item") else val_m["loss"]
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(state, os.path.join(args.out_dir, "checkpoint.pth.tar"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(args.out_dir, "best_checkpoint.pth.tar"))
            logger.info(f"  ✓ New best val loss: {best_val_loss:.2f}")

    logger.info(f"Done. Best val loss: {best_val_loss:.2f}")
    logger.info(f"Checkpoint: {os.path.join(args.out_dir, 'best_checkpoint.pth.tar')}")


def _run_eval_only(args, model, device, smooth_sigma):
    """Compare baseline (density=None) vs density-conditioned on the val set."""
    from omegaconf import OmegaConf
    from voxbind.voxelizer import Voxelizer

    if args.density_ckpt:
        logger.info(f"Loading density checkpoint: {args.density_ckpt}")
        ckpt = torch.load(args.density_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=True)

    cfg = OmegaConf.create({
        "dset": {
            "dset_name": "crossdocked",
            "data_dir": args.data_dir,
            "ligand_radius": 0.5,
            "pocket_radius": -1,
        },
        "vox": {"grid_dim": 64, "resolution": 0.25, "cubes_around": 8},
        "wjs": {"split": "val", "n_targets": 10, "n_samples_per_pocket": 10},
        "bsz": args.bsz,
        "num_workers": args.num_workers,
        "aug": False,
        "smooth_sigma": smooth_sigma,
        "debug": args.debug,
    })

    _, loader_val, _ = create_dataloaders(cfg)
    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=device)
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    metrics_base, _ = create_metrics_for_training()
    metrics_dens, _ = create_metrics_for_training()

    prefetcher = DensityPrefetcher(
        loader_val, voxelizer, smooth_sigma, args.sigma, training=False
    )

    logger.info("Evaluating baseline (density=None) ...")
    m_base = eval_epoch(model, prefetcher, criterion, metrics_base, use_density=False, debug=args.debug)

    prefetcher2 = DensityPrefetcher(
        loader_val, voxelizer, smooth_sigma, args.sigma, training=False
    )
    logger.info("Evaluating density-conditioned ...")
    m_dens = eval_epoch(model, prefetcher2, criterion, metrics_dens, use_density=True, debug=args.debug)

    print("\n" + "=" * 50)
    print(f"{'Condition':<25s}  {'Loss':>10s}  {'mIoU':>8s}")
    print("-" * 50)
    print(f"{'Baseline (no density)':<25s}  "
          f"{float(m_base['loss']):>10.2f}  "
          f"{float(m_base.get('miou', float('nan'))):>8.4f}")
    print(f"{'+ pdb2vol density':<25s}  "
          f"{float(m_dens['loss']):>10.2f}  "
          f"{float(m_dens.get('miou', float('nan'))):>8.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
