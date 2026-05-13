"""
train_xray.py — Finetune VoxBind with real X-ray electron density conditioning
===============================================================================
Adds an X-ray density conditioning branch to a pretrained VoxBind model and
finetunes *only* that branch (density_encoder) while freezing everything else.

The X-ray density is the 2Fo-Fc CCP4 electron density map cropped and aligned
to the 64³ × 0.25 Å VoxBind pocket grid (downloaded by scripts/04_download_xray.py
and loaded on-the-fly by DatasetCrossDockedXray).

Architecture
------------
VoxBind already supports a density_encoder branch via with_density=True (see
models/voxbind.py). The branch takes a (B, 1, G, G, G) tensor and its output
is summed with the pocket encoding before entering the U-Net. This script
reuses that branch — just feeding real X-ray density instead of simulated
pdb2vol density.

Training strategy
-----------------
- density_encoder parameters are trainable; all others are frozen.
- Samples WITHOUT a CCP4 map (xray_available=False) receive a zero density
  tensor → the model trains both conditionally (with density) and
  unconditionally (without) in the same loop, acting as classifier-free
  guidance pretraining.
- The density map is z-score normalised per-map in the dataset class;
  training does NOT add further noise to it.

Usage
-----
    cd voxbind
    python train_xray.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --ccp4_dir        dataset/data/ccp4 \\
        --epochs          30 \\
        --bsz             32

After training the checkpoint is written to:
    exps/xray_poc/checkpoint.pth.tar   (every epoch)
    exps/xray_poc/best_checkpoint.pth.tar   (best val loss)
"""

import argparse
import logging
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from voxbind.constants import N_LIGAND_ELEMENTS, N_POCKET_ELEMENTS
from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray
from voxbind.metrics import create_metrics_for_training
from voxbind.models.voxbind import VoxBind
from voxbind.utils.base_utils import makedir, seed_everything
from voxbind.voxelizer import Voxelizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xray")


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma > 0:
        return voxels + torch.empty_like(voxels).normal_(0, sigma)
    return voxels


def load_pretrained(pretrained_path: str, device: torch.device):
    """Load pretrained VoxBind and attach density_encoder (with_density=True)."""
    ckpt_path = os.path.join(pretrained_path, "checkpoint.pth.tar")
    chkp = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Infer architecture from checkpoint
    ch_mults   = (1, 2, 2, 4)
    is_attn    = (False, False, True, True)
    n_blocks   = 2
    smooth_sigma = 0.9

    sd_peek = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    sd_peek = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd_peek.items()}
    enc_w = sd_peek.get("ligand_encoder.conv2.weight",
                        sd_peek.get("ligand_encoder.block2.weight", None))
    n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
    n_groups   = min(16, n_channels // 2)

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
        with_density=True,
        verbose=True,
    )

    sd = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.info(f"  New keys (will be trained): {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys ignored : {unexpected}")

    return model, smooth_sigma


# ── Prefetcher ─────────────────────────────────────────────────────────────────

class XrayPrefetcher:
    """Overlap GPU voxelization with model compute via a secondary CUDA stream.

    Yields tuples:
        (vox_lig, smooth_vox_lig, vox_poc, xray_density_gpu, xray_available)

    xray_density_gpu : (B, 1, G, G, G) float32 on GPU — zeros for unavailable maps
    xray_available   : (B,) bool on GPU — True where a real CCP4 map was loaded
    """

    def __init__(self, loader, voxelizer: Voxelizer, smooth_sigma: float, training: bool = True):
        self.loader       = loader
        self.voxelizer    = voxelizer
        self.smooth_sigma = smooth_sigma
        self.training     = training
        self.stream       = torch.cuda.Stream()

    def _process(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                vox_lig  = self.voxelizer(batch["ligand"],  num_channels=N_LIGAND_ELEMENTS)
                smooth   = add_noise_vox(vox_lig, self.smooth_sigma)
                vox_poc  = self.voxelizer(batch["pocket"],  num_channels=N_POCKET_ELEMENTS)
                if self.training:
                    # Zero out 20% of pocket voxels as augmentation (same as train.py)
                    vox_poc[: math.ceil(0.2 * vox_poc.shape[0])].zero_()

                # X-ray density: (B, G, G, G) → unsqueeze to (B, 1, G, G, G)
                xray = batch["xray_density"].to(self.voxelizer.device, non_blocking=True)
                xray = xray.unsqueeze(1)                       # add channel dim
                avail = batch["xray_available"].to(self.voxelizer.device, non_blocking=True)

                # Zero out density for samples without a CCP4 map (already zero from
                # the dataset, but make it explicit for safety)
                xray = xray * avail.view(-1, 1, 1, 1, 1).float()

        return vox_lig, smooth, vox_poc, xray, avail

    def __iter__(self):
        it = iter(self.loader)
        try:
            nxt = next(it)
        except StopIteration:
            return
        nxt_out = self._process(nxt)

        for batch in it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur_out = nxt_out
            nxt_out = self._process(batch)
            yield cur_out

        torch.cuda.current_stream().wait_stream(self.stream)
        yield nxt_out

    def __len__(self):
        return len(self.loader)


# ── Train / eval loops ─────────────────────────────────────────────────────────

def train_epoch(model, prefetcher, criterion, optimizer, metrics, debug=False):
    metrics.reset()
    model.eval()
    if hasattr(model, "density_encoder"):
        model.density_encoder.train()

    for i, (vox_lig, smooth_lig, vox_poc, xray, avail) in enumerate(prefetcher):
        pred = model(smooth_lig, vox_poc, density=xray)
        loss = criterion(pred, vox_lig)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.update(loss.detach(), pred.detach(), vox_lig, update_miou=(i % 8 == 0))
        if debug and i == 10:
            break

    return metrics.compute()


@torch.no_grad()
def eval_epoch(model, prefetcher, criterion, metrics, use_density=True, debug=False):
    metrics.reset()
    model.eval()

    for i, (vox_lig, smooth_lig, vox_poc, xray, avail) in enumerate(prefetcher):
        density_in = xray if use_density else None
        pred = model(smooth_lig, vox_poc, density=density_in)
        loss = criterion(pred, vox_lig)
        metrics.update(loss, pred, vox_lig)
        if debug and i == 10:
            break

    return metrics.compute()


# ── Data loading ───────────────────────────────────────────────────────────────

def make_loaders(args, smooth_sigma: float):
    """Return (loader_train, loader_val) using DatasetCrossDockedXray."""
    common = dict(
        data_dir=args.data_dir,
        ccp4_dir=args.ccp4_dir,
        use_xray=True,
        cache_size=args.cache_size,
        small=args.debug,
    )
    dset_train = DatasetCrossDockedXray(split="train", aug=True,  **common)
    dset_val   = DatasetCrossDockedXray(split="val",   aug=False, **common)

    avail_train = sum(1 for _, _, av, _ in [
        (None, None, dset_train[i]["xray_available"].item(), None)
        for i in range(min(200, len(dset_train)))
    ] if av)
    logger.info(
        f"  Train size : {len(dset_train):,}  "
        f"(~{avail_train/min(200,len(dset_train)):.0%} with X-ray density in first 200)"
    )
    logger.info(f"  Val size   : {len(dset_val):,}")

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 4

    loader_train = torch.utils.data.DataLoader(
        dset_train, batch_size=args.bsz, shuffle=True, drop_last=True, **loader_kw
    )
    loader_val = torch.utils.data.DataLoader(
        dset_val, batch_size=args.bsz, shuffle=False, drop_last=False, **loader_kw
    )
    return loader_train, loader_val


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VoxBind X-ray density-conditioned finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pretrained_path", required=True,
                   help="Pretrained VoxBind experiment directory (contains checkpoint.pth.tar)")
    p.add_argument("--data_dir",   default="dataset/data",
                   help="CrossDocked data directory")
    p.add_argument("--ccp4_dir",   default="dataset/data/ccp4",
                   help="Directory of downloaded .map CCP4 files")
    p.add_argument("--out_dir",    default="exps/xray_poc",
                   help="Output directory for checkpoints")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--bsz",        type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate (density_encoder only)")
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--cache_size", type=int,   default=32,
                   help="CCP4 maps cached per DataLoader worker")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--debug",      action="store_true",
                   help="Smoke test: 10 batches per epoch, small dataset")
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training; compare baseline vs X-ray conditioned on val set")
    p.add_argument("--xray_ckpt",  default=None,
                   help="Path to X-ray finetuned checkpoint for --eval_only")
    p.add_argument("--no_wandb",   action="store_true",
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb_project", default="voxbind",
                   help="W&B project name")
    p.add_argument("--wandb_entity",  default="eddy26",
                   help="W&B entity/username")
    p.add_argument("--exp_name",   default=None,
                   help="W&B run name (defaults to out_dir basename)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    makedir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("VoxBind X-ray density conditioning finetune")
    logger.info(f"  pretrained : {args.pretrained_path}")
    logger.info(f"  ccp4_dir   : {args.ccp4_dir}")
    logger.info(f"  out_dir    : {args.out_dir}")
    logger.info(f"  epochs     : {args.epochs}  bsz={args.bsz}  lr={args.lr}")
    logger.info("=" * 60)

    if not args.no_wandb and not args.eval_only:
        exp_name = args.exp_name or os.path.basename(os.path.abspath(args.out_dir))
        wandb.init(
            project=not args.no_wandb_project,
            entity=not args.no_wandb_entity,
            name=exp_name,
            dir=args.out_dir,
            config=vars(args),
            resume="allow",
            settings=wandb.Settings(code_dir="."),
        )

    model, smooth_sigma = load_pretrained(args.pretrained_path, device)
    model.to(device)

    if args.eval_only:
        _run_eval_only(args, model, device, smooth_sigma)
        return

    # ── Freeze everything except density_encoder ──────────────────────────
    n_frozen = 0
    for name, param in model.named_parameters():
        if "density_encoder" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
            n_frozen += param.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Frozen     : {n_frozen / 1e6:.2f}M params")
    logger.info(f"  Trainable  : {n_trainable / 1e6:.2f}M params  (density_encoder)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )
    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))

    loader_train, loader_val = make_loaders(args, smooth_sigma)

    metrics_train, _ = create_metrics_for_training()
    metrics_val,   _ = create_metrics_for_training()

    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        t0 = time.time()

        pf_train = XrayPrefetcher(loader_train, voxelizer, smooth_sigma, training=True)
        train_m  = train_epoch(model, pf_train, criterion, optimizer, metrics_train,
                               debug=args.debug)

        pf_val  = XrayPrefetcher(loader_val, voxelizer, smooth_sigma, training=False)
        val_m   = eval_epoch(model, pf_val, criterion, metrics_val, debug=args.debug)

        elapsed = time.time() - t0
        logger.info(
            f"epoch {epoch:3d} ({elapsed:.1f}s) | "
            f"train loss={float(train_m['loss']):.2f}  "
            f"miou={float(train_m.get('miou', float('nan'))):.4f} | "
            f"val   loss={float(val_m['loss']):.2f}  "
            f"miou={float(val_m.get('miou', float('nan'))):.4f}"
        )

        val_loss = float(val_m["loss"])
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
            "smooth_sigma": smooth_sigma,
        }
        torch.save(state, os.path.join(args.out_dir, "checkpoint.pth.tar"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(args.out_dir, "best_checkpoint.pth.tar"))
            logger.info(f"  ✓ New best val loss: {best_val_loss:.2f}")

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": float(train_m["loss"]),
                "train/miou": float(train_m.get("miou", float("nan"))),
                "val/loss":   float(val_m["loss"]),
                "val/miou":   float(val_m.get("miou", float("nan"))),
                "epoch_time_s": elapsed,
            }, step=epoch)

    logger.info(f"Done. Best val loss: {best_val_loss:.2f}")
    logger.info(f"Checkpoint: {os.path.join(args.out_dir, 'best_checkpoint.pth.tar')}")

    if not args.no_wandb:
        wandb.finish()


def _run_eval_only(args, model, device, smooth_sigma):
    """Compare baseline vs X-ray-conditioned model on the val set."""
    if args.xray_ckpt:
        logger.info(f"Loading X-ray checkpoint: {args.xray_ckpt}")
        ckpt = torch.load(args.xray_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=True)

    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))
    criterion  = torch.nn.MSELoss(reduction="sum").to(device)

    _, loader_val = make_loaders(args, smooth_sigma)

    metrics_base, _ = create_metrics_for_training()
    metrics_dens, _ = create_metrics_for_training()

    logger.info("Evaluating baseline (density=None) ...")
    pf = XrayPrefetcher(loader_val, voxelizer, smooth_sigma, training=False)
    m_base = eval_epoch(model, pf, criterion, metrics_base, use_density=False,
                        debug=args.debug)

    logger.info("Evaluating X-ray conditioned ...")
    pf2 = XrayPrefetcher(loader_val, voxelizer, smooth_sigma, training=False)
    m_dens = eval_epoch(model, pf2, criterion, metrics_dens, use_density=True,
                        debug=args.debug)

    print("\n" + "=" * 55)
    print(f"{'Condition':<30s}  {'Loss':>10s}  {'mIoU':>8s}")
    print("-" * 55)
    print(f"{'Baseline (no density)':<30s}  "
          f"{float(m_base['loss']):>10.2f}  "
          f"{float(m_base.get('miou', float('nan'))):>8.4f}")
    print(f"{'+ X-ray (2Fo-Fc CCP4)':<30s}  "
          f"{float(m_dens['loss']):>10.2f}  "
          f"{float(m_dens.get('miou', float('nan'))):>8.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
