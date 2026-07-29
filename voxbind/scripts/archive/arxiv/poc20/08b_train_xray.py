"""08b_train_xray.py — Full-dataset X-ray density-conditioned finetuning.

Finetunes VoxBind with differential learning rates on the full CrossDocked
X-ray dataset (all samples, not just one). Uses tqdm progress bars for both
epoch and batch loops, and logs to W&B.

Usage
-----
    cd voxbind
    bash scripts/08b_train_xray.sh
    # or directly:
    python scripts/08b_train_xray.py --pretrained_path exps/exp_sig0.9 --crops_dir dataset/data/xray_crops
"""

import argparse
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_noise(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma > 0:
        return voxels + torch.empty_like(voxels).normal_(0, sigma)
    return voxels


def load_model(pretrained_path: str, device: torch.device):
    ckpt_path = os.path.join(pretrained_path, "checkpoint.pth.tar")
    chkp = torch.load(ckpt_path, map_location=device, weights_only=False)

    sd = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    enc_w = sd.get("ligand_encoder.conv2.weight",
                   sd.get("ligand_encoder.block2.weight", None))
    n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
    n_groups   = min(16, n_channels // 2)

    smooth_sigma = 0.9
    ch_mults, is_attn, n_blocks = (1, 2, 2, 4), (False, False, True, True), 2
    cfg = chkp.get("cfg", None)
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            if not isinstance(cfg, dict):
                cfg = OmegaConf.to_container(cfg, resolve=True)
            smooth_sigma = cfg.get("smooth_sigma", smooth_sigma)
            mc = cfg.get("model", {})
            ch_mults = tuple(mc.get("ch_mults", list(ch_mults)))
            is_attn  = tuple(mc.get("is_attn",  list(is_attn)))
            n_blocks = mc.get("n_blocks", n_blocks)
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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  New keys (will be trained): {missing}")
    if unexpected:
        print(f"  Unexpected keys ignored   : {unexpected}")

    return model, smooth_sigma


# ── Prefetcher ─────────────────────────────────────────────────────────────────

class XrayPrefetcher:
    def __init__(self, loader, voxelizer: Voxelizer, smooth_sigma: float, training: bool = True):
        self.loader       = loader
        self.voxelizer    = voxelizer
        self.smooth_sigma = smooth_sigma
        self.training     = training
        self.stream       = torch.cuda.Stream()

    def _process(self, batch):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                vox_lig = self.voxelizer(batch["ligand"], num_channels=N_LIGAND_ELEMENTS)
                smooth  = add_noise(vox_lig, self.smooth_sigma)
                vox_poc = self.voxelizer(batch["pocket"], num_channels=N_POCKET_ELEMENTS)
                if self.training:
                    vox_poc[: math.ceil(0.2 * vox_poc.shape[0])].zero_()
                xray  = batch["xray_density"].to(self.voxelizer.device, non_blocking=True).unsqueeze(1)
                avail = batch["xray_available"].to(self.voxelizer.device, non_blocking=True)
                xray  = xray * avail.view(-1, 1, 1, 1, 1).float()
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


# ── Train / val loops ──────────────────────────────────────────────────────────

def train_epoch(model, prefetcher, criterion, optimizer, metrics, epoch, debug=False):
    metrics.reset()
    model.eval()
    if hasattr(model, "density_encoder"):
        model.density_encoder.train()

    pbar = tqdm(prefetcher, desc=f"Train {epoch}", leave=False,
                dynamic_ncols=True, total=len(prefetcher))
    for i, (vox_lig, smooth_lig, vox_poc, xray, avail) in enumerate(pbar):
        pred = model(smooth_lig, vox_poc, density=xray)
        loss = criterion(pred, vox_lig)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.update(loss.detach(), pred.detach(), vox_lig, update_miou=(i % 8 == 0))
        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if debug and i == 10:
            break

    return metrics.compute()


@torch.no_grad()
def val_epoch(model, prefetcher, criterion, metrics, epoch, debug=False):
    metrics.reset()
    model.eval()

    pbar = tqdm(prefetcher, desc=f"Val   {epoch}", leave=False,
                dynamic_ncols=True, total=len(prefetcher))
    for i, (vox_lig, smooth_lig, vox_poc, xray, avail) in enumerate(pbar):
        pred = model(smooth_lig, vox_poc, density=xray)
        loss = criterion(pred, vox_lig)
        metrics.update(loss, pred, vox_lig)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if debug and i == 10:
            break

    return metrics.compute()


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-dataset X-ray density-conditioned VoxBind finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pretrained_path", required=True,
                   help="Pretrained VoxBind experiment dir (contains checkpoint.pth.tar)")
    p.add_argument("--data_dir",    default="dataset/data")
    p.add_argument("--ccp4_dir",    default="dataset/data/ccp4")
    p.add_argument("--crops_dir",   default=None,
                   help="Precomputed crops dir (faster than ccp4_dir)")
    p.add_argument("--out_dir",     default="exps/xray_full",
                   help="Output directory for checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--bsz",         type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="LR for density_encoder + density_proj")
    p.add_argument("--lr_backbone", type=float, default=1e-4,
                   help="LR for pretrained backbone (UNet + ligand/pocket encoders)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--cache_size",  type=int,   default=32)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--debug",       action="store_true",
                   help="Smoke test: 10 batches per epoch, small dataset")
    p.add_argument("--no_wandb",    action="store_true",
                   help="Disable W&B logging")
    p.add_argument("--wandb_project", default="voxbind")
    p.add_argument("--wandb_entity",  default="eddy26")
    p.add_argument("--exp_name",    default=None,
                   help="W&B run name (defaults to out_dir basename)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    makedir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = not args.no_wandb
    if use_wandb:
        exp_name = args.exp_name or os.path.basename(os.path.abspath(args.out_dir))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=exp_name,
            dir=args.out_dir,
            config=vars(args),
            resume="allow",
            settings=wandb.Settings(code_dir="."),
        )

    model, smooth_sigma = load_model(args.pretrained_path, device)
    model.to(device)

    # Differential LR: density branch gets higher LR, backbone gets lower
    density_params  = (list(model.density_encoder.parameters()) +
                       list(model.density_proj.parameters()))
    density_ids     = {id(p) for p in density_params}
    backbone_params = [p for p in model.parameters() if id(p) not in density_ids]

    n_density  = sum(p.numel() for p in density_params)
    n_backbone = sum(p.numel() for p in backbone_params)
    print(f"  Density branch : {n_density / 1e6:.2f}M params  lr={args.lr}")
    print(f"  Backbone       : {n_backbone / 1e6:.2f}M params  lr={args.lr_backbone}")

    optimizer = torch.optim.AdamW([
        {"params": density_params,  "lr": args.lr},
        {"params": backbone_params, "lr": args.lr_backbone},
    ], weight_decay=1e-2)
    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))

    # Datasets
    common = dict(
        data_dir=args.data_dir,
        ccp4_dir=args.ccp4_dir,
        crops_dir=args.crops_dir,
        use_xray=True,
        cache_size=args.cache_size,
        small=args.debug,
    )
    dset_train = DatasetCrossDockedXray(split="train", aug=True,  **common)
    dset_val   = DatasetCrossDockedXray(split="val",   aug=False, **common)
    print(f"  Train : {len(dset_train):,}  |  Val : {len(dset_val):,}")

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 4

    loader_train = torch.utils.data.DataLoader(
        dset_train, batch_size=args.bsz, shuffle=True, drop_last=True, **loader_kw)
    loader_val = torch.utils.data.DataLoader(
        dset_val, batch_size=args.bsz, shuffle=False, drop_last=False, **loader_kw)

    metrics_train, _ = create_metrics_for_training()
    metrics_val,   _ = create_metrics_for_training()

    best_val_loss = float("inf")

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.time()

        pf_train = XrayPrefetcher(loader_train, voxelizer, smooth_sigma, training=True)
        train_m  = train_epoch(model, pf_train, criterion, optimizer, metrics_train,
                               epoch, debug=args.debug)

        pf_val = XrayPrefetcher(loader_val, voxelizer, smooth_sigma, training=False)
        val_m  = val_epoch(model, pf_val, criterion, metrics_val, epoch, debug=args.debug)

        elapsed   = time.time() - t0
        val_loss  = float(val_m["loss"])
        train_loss = float(train_m["loss"])
        val_miou  = float(val_m.get("miou", float("nan")))
        train_miou = float(train_m.get("miou", float("nan")))

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.2f}",
            vl_loss=f"{val_loss:.2f}",
            vl_miou=f"{val_miou:.3f}",
        )

        state = {
            "epoch":        epoch,
            "state_dict":   model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_loss":     val_loss,
            "args":         vars(args),
            "smooth_sigma": smooth_sigma,
        }
        torch.save(state, os.path.join(args.out_dir, "checkpoint.pth.tar"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(args.out_dir, "best_checkpoint.pth.tar"))
            tqdm.write(f"  epoch {epoch:3d}  new best val loss: {best_val_loss:.2f}")

        if use_wandb:
            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_loss,
                "train/miou":     train_miou,
                "val/loss":       val_loss,
                "val/miou":       val_miou,
                "epoch_time_s":   elapsed,
            }, step=epoch)

    tqdm.write(f"\nDone. Best val loss: {best_val_loss:.2f}")
    tqdm.write(f"Checkpoint: {os.path.join(args.out_dir, 'best_checkpoint.pth.tar')}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
