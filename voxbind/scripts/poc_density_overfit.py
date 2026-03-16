"""poc_density_overfit.py — Full fine-tune VoxBind on one or more X-ray samples.

Fine-tunes the entire model (U-Net + encoders + density_encoder) on N
(pocket, ligand, xray) triples using differential learning rates:
  - density_encoder  : --lr        (higher, randomly initialised)
  - pretrained backbone : --lr_backbone  (lower, preserve pretrained weights)

Each training iteration picks one sample at random from the N precomputed ones.

Usage
-----
    cd voxbind
    # single sample (default):
    python scripts/poc_density_overfit.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --crops_dir       dataset/data/xray_crops

    # four samples:
    python scripts/poc_density_overfit.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --crops_dir       dataset/data/xray_crops \\
        --n_samples       4
"""

import argparse
import os
import random

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from voxbind.constants import N_LIGAND_ELEMENTS, N_POCKET_ELEMENTS
from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray
from voxbind.models.voxbind import VoxBind
from voxbind.voxelizer import Voxelizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_noise(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    return voxels + torch.empty_like(voxels).normal_(0, sigma)


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
    cfg = chkp.get("cfg", None)
    if isinstance(cfg, dict):
        smooth_sigma = cfg.get("smooth_sigma", smooth_sigma)

    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=n_channels,
        n_groups=n_groups,
        smooth_sigma=smooth_sigma,
        with_density=True,
        verbose=True,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  New keys (randomly init, will be trained): {missing}")
    if unexpected:
        print(f"  Unexpected keys ignored: {unexpected}")

    return model, smooth_sigma


def find_samples_with_density(dset, n: int, max_search: int = 1000) -> list:
    """Return indices of the first n samples where xray_available=True."""
    print(f"Searching for {n} sample(s) with X-ray density (up to {max_search} items)...")
    indices = []
    for i in range(min(max_search, len(dset))):
        if dset[i]["xray_available"].item():
            indices.append(i)
            if len(indices) == n:
                break
    if len(indices) < n:
        raise RuntimeError(
            f"Only found {len(indices)}/{n} samples with X-ray density "
            f"in the first {max_search} items."
        )
    return indices


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PoC: overfit VoxBind density_encoder on a single X-ray sample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pretrained_path", required=True,
                   help="Experiment dir containing checkpoint.pth.tar")
    p.add_argument("--data_dir",    default="dataset/data")
    p.add_argument("--ccp4_dir",    default="dataset/data/ccp4",
                   help="On-the-fly CCP4 loading (used if --crops_dir is not set)")
    p.add_argument("--crops_dir",   default=None,
                   help="Precomputed crop dir (output of scripts/07_process_xray.sh)")
    p.add_argument("--sample_idx",  type=int, default=None,
                   help="Dataset index to use (auto-search if not given). "
                        "When set, n_samples is ignored.")
    p.add_argument("--n_samples",   type=int, default=1,
                   help="Number of samples to overfit on (picked sequentially "
                        "from the dataset, each with xray_available=True)")
    p.add_argument("--normalize",   type=lambda x: x.lower() != "false",
                   default=True, metavar="BOOL",
                   help="Apply local z-score normalization to density crops at load time. "
                        "Set False when crops_dir already contains pre-normalized crops "
                        "(output of 05b_renormalize_crops.py).")
    p.add_argument("--iters",        type=int,   default=2000)
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="LR for density_encoder+density_proj (randomly initialised)")
    p.add_argument("--lr_backbone",  type=float, default=1e-4,
                   help="LR for pretrained backbone (UNet + ligand/pocket encoders)")
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--no_density",   action="store_true",
                   help="Fine-tune without density conditioning (with_density=False). "
                        "All params use lr_backbone. Default out: finetune_ckpt.pth.tar")
    p.add_argument("--out",          default=None,
                   help="Output checkpoint path (auto-set based on --no_density if not given)")
    p.add_argument("--no_wandb",     action="store_true",
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb_project", default="voxbind")
    p.add_argument("--wandb_entity",  default="eddy26")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        args.out = ("exps/poc_xray/finetune_ckpt.pth.tar" if args.no_density
                    else "exps/poc_xray/overfit_ckpt.pth.tar")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {'no-density fine-tune' if args.no_density else 'density-conditioned fine-tune'}\n")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = ("finetune_no_density" if args.no_density else "finetune_xray_density")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(code_dir="."),
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, smooth_sigma = load_model(args.pretrained_path, device)

    if args.no_density:
        # Rebuild with with_density=False and load baseline weights strictly
        ckpt_path = os.path.join(args.pretrained_path, "checkpoint.pth.tar")
        chkp = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        enc_w = sd.get("ligand_encoder.conv2.weight",
                        sd.get("ligand_encoder.block2.weight", None))
        n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
        n_groups = min(16, n_channels // 2)
        model = VoxBind(
            n_channels_ligand=N_LIGAND_ELEMENTS,
            n_channels_pocket=N_POCKET_ELEMENTS,
            n_channels=n_channels,
            n_groups=n_groups,
            smooth_sigma=smooth_sigma,
            with_density=False,
        )
        model.load_state_dict(sd, strict=True)
        print("  Loaded baseline weights (strict=True, no density branch)")

    model.to(device)

    for p in model.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params : {n_trainable:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dset = DatasetCrossDockedXray(
        data_dir=args.data_dir,
        ccp4_dir=args.ccp4_dir,
        crops_dir=args.crops_dir,
        split="train",
        aug=False,      # fixed sample → no augmentation
        use_xray=True,
        normalize=args.normalize,
    )

    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        indices = find_samples_with_density(dset, args.n_samples)

    print(f"Using sample indices : {indices}")

    # ── Voxelize all samples (once; keep fixed on GPU) ─────────────────────────
    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))
    samples_data = []  # list of (vox_lig, vox_poc, xray, pocket_id)

    for idx in indices:
        loader = DataLoader(Subset(dset, [idx]), batch_size=1,
                            shuffle=False, num_workers=0)
        batch = next(iter(loader))

        pocket_id = batch["pocket"]["id"][0]
        xray_ok   = batch["xray_available"].item()
        print(f"  [{idx}] {pocket_id}  xray_available={xray_ok}")
        if not xray_ok and not args.no_density:
            print(f"        WARNING: no X-ray density — density_encoder receives zeros.")

        with torch.no_grad():
            vox_lig = voxelizer(batch["ligand"], num_channels=N_LIGAND_ELEMENTS)
            vox_poc = voxelizer(batch["pocket"], num_channels=N_POCKET_ELEMENTS)
            xray    = batch["xray_density"].unsqueeze(1).to(device)

        samples_data.append((vox_lig, vox_poc, xray, pocket_id))

    print(f"\nPrecomputed {len(samples_data)} sample(s) on {device}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if args.no_density:
        # Uniform LR — no new parameters
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr_backbone, weight_decay=1e-2
        )
        print(f"LR (all params) : {args.lr_backbone}\n")
    else:
        # Differential LR: new density branch gets higher LR
        density_params  = (list(model.density_encoder.parameters()) +
                           list(model.density_proj.parameters()))
        density_ids     = {id(p) for p in density_params}
        backbone_params = [p for p in model.parameters() if id(p) not in density_ids]
        optimizer = torch.optim.AdamW([
            {"params": density_params,  "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone},
        ], weight_decay=1e-2)
        print(f"LR density branch : {args.lr}  |  LR backbone : {args.lr_backbone}\n")

    criterion = torch.nn.MSELoss()

    # ── Fine-tune loop ────────────────────────────────────────────────────────
    n_samples = len(samples_data)
    print(f"Fine-tuning {args.iters} iters  sigma={smooth_sigma}  "
          f"n_samples={n_samples}\n")
    print(f"{'iter':>6}  {'sample':>7}  {'loss':>12}  {'vs_init':>10}")
    print("-" * 42)

    losses = []
    model.train()

    pbar = tqdm(range(1, args.iters + 1), desc="Fine-tuning", dynamic_ncols=True)
    for i in pbar:
        # Pick a random sample each iteration
        s_idx = random.randrange(n_samples)
        vox_lig, vox_poc, xray, _ = samples_data[s_idx]

        smooth_lig = add_noise(vox_lig, smooth_sigma)
        pred = model(smooth_lig, vox_poc, density=xray)
        loss = criterion(pred, vox_lig)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        ratio = losses[0] / loss.item() if loss.item() > 0 else float("inf")
        pbar.set_postfix(loss=f"{loss.item():.4f}", reduction=f"{ratio:.2f}x")

        if (i % args.log_every == 0 or i == 1) and use_wandb:
            wandb.log({"train/loss": loss.item(), "train/loss_reduction": ratio}, step=i)

    print(f"Initial loss : {losses[0]:.6f}")
    print(f"Final loss   : {losses[-1]:.6f}")
    print(f"Reduction    : {losses[0] / max(losses[-1], 1e-12):.1f}x")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict":   model.state_dict(),
        "smooth_sigma": smooth_sigma,
        "args":         vars(args),
        "final_loss":   losses[-1],
        "sample_idx":   indices[0],   # primary index (for compat with compare script)
        "sample_indices": indices,
        "pocket_id":    samples_data[0][3],
        "losses":       losses,
    }, args.out)
    print(f"\nCheckpoint → {args.out}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
