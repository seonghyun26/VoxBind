"""poc_compare_samples.py — Three-way comparison: baseline / fine-tuned / density-conditioned.

Loads up to three checkpoints and runs WJS sampling on the same pocket:
  1. baseline    — original pretrained model, no fine-tuning, no density
  2. finetuned   — fine-tuned on single sample (--finetune_ckpt), no density  [optional]
  3. xray_cond   — fine-tuned with X-ray density conditioning (--xray_ckpt)

Usage
-----
    cd voxbind
    # two-way (original baseline comparison):
    python scripts/poc_compare_samples.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --xray_ckpt      exps/poc_xray/overfit_ckpt.pth.tar \\
        --crops_dir      dataset/data/xray_crops

    # three-way (adds no-density fine-tuned):
    python scripts/poc_compare_samples.py \\
        --pretrained_path exps/exp_sig0.9 \\
        --finetune_ckpt  exps/poc_xray/finetune_ckpt.pth.tar \\
        --xray_ckpt      exps/poc_xray/overfit_ckpt.pth.tar \\
        --crops_dir      dataset/data/xray_crops

Outputs
-------
    exps/poc_xray/compare/
        baseline/   samples.sdf   (original model, no density)
        finetuned/  samples.sdf   (fine-tuned, no density)   [if --finetune_ckpt given]
        xray_cond/  samples.sdf   (fine-tuned + X-ray density)
        pocket.pdb, ligand.sdf    (reference structures)
"""

import argparse
import math
import os
import shutil

import torch
from torch.utils.data import DataLoader, Subset

from voxbind.constants import N_LIGAND_ELEMENTS, N_POCKET_ELEMENTS
from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray
from voxbind.models.voxbind import VoxBind
from voxbind.utils.base_utils import makedir, seed_everything
from voxbind.utils.convert_utils import mol2rdkit_obabel
from voxbind.voxelizer import Voxelizer

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("RDKit is required for molecule output.")


# ── Model loading ──────────────────────────────────────────────────────────────

def _infer_arch(sd):
    enc_w = sd.get("ligand_encoder.conv2.weight",
                   sd.get("ligand_encoder.block2.weight", None))
    n_channels = enc_w.shape[0] * 2 if enc_w is not None else 32
    n_groups   = min(16, n_channels // 2)
    return n_channels, n_groups


def load_baseline(pretrained_path: str, device: torch.device):
    """Load the original pretrained model (with_density=False)."""
    ckpt_path = os.path.join(pretrained_path, "checkpoint.pth.tar")
    chkp = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = chkp.get("state_dict_ema", chkp.get("state_dict", {}))
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    n_channels, n_groups = _infer_arch(sd)
    smooth_sigma = (chkp.get("cfg") or {}).get("smooth_sigma", 0.9) \
        if isinstance(chkp.get("cfg"), dict) else 0.9

    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=n_channels,
        n_groups=n_groups,
        smooth_sigma=smooth_sigma,
        with_density=False,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[baseline]  loaded from {ckpt_path}  (sigma={smooth_sigma})")
    return model, smooth_sigma


def load_finetuned_model(finetune_ckpt_path: str, device: torch.device):
    """Load a fine-tuned model without density conditioning (with_density=False)."""
    chkp = torch.load(finetune_ckpt_path, map_location=device, weights_only=False)
    sd = chkp.get("state_dict", chkp)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    n_channels, n_groups = _infer_arch(sd)
    smooth_sigma = chkp.get("smooth_sigma", 0.9)
    sample_idx   = chkp.get("sample_idx", None)
    pocket_id    = chkp.get("pocket_id", "unknown")

    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=n_channels,
        n_groups=n_groups,
        smooth_sigma=smooth_sigma,
        with_density=False,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[finetuned] loaded from {finetune_ckpt_path}  (sigma={smooth_sigma})")
    print(f"            sample_idx={sample_idx}  pocket_id={pocket_id}")
    return model, smooth_sigma, sample_idx, pocket_id


def load_xray_model(xray_ckpt_path: str, device: torch.device):
    """Load the density-conditioned finetuned model (with_density=True)."""
    chkp = torch.load(xray_ckpt_path, map_location=device, weights_only=False)
    sd = chkp.get("state_dict", chkp)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    n_channels, n_groups = _infer_arch(sd)
    smooth_sigma = chkp.get("smooth_sigma", 0.9)
    sample_idx   = chkp.get("sample_idx", None)
    pocket_id    = chkp.get("pocket_id", "unknown")

    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=n_channels,
        n_groups=n_groups,
        smooth_sigma=smooth_sigma,
        with_density=True,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[xray_cond] loaded from {xray_ckpt_path}  (sigma={smooth_sigma})")
    print(f"            sample_idx={sample_idx}  pocket_id={pocket_id}")
    return model, smooth_sigma, sample_idx, pocket_id


# ── WJS sampling loop (density-aware) ─────────────────────────────────────────

@torch.no_grad()
def wjs_sample(
    model: VoxBind,
    pocket_vox: torch.Tensor,
    ligand_vox: torch.Tensor,
    density: torch.Tensor = None,   # (1,1,G,G,G) or None
    n_chains: int = 10,
    warmup: int = 400,
    steps: int = 100,
    max_steps: int = 100,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Minimal WJS loop; passes density to every forward call when provided."""
    sigma   = model.smooth_sigma
    G       = pocket_vox.shape[-1]
    device  = pocket_vox.device

    # Langevin params (same as VoxBind.wjs_walk_steps)
    gamma  = 1.0
    u      = 1.0
    delta  = sigma / 2
    zeta1  = math.exp(-gamma)
    zeta2  = math.exp(-2 * gamma)

    # Initialise chains
    y = torch.randn(n_chains, N_LIGAND_ELEMENTS, G, G, G, device=device) * sigma
    v = torch.zeros_like(y)

    poc  = pocket_vox.expand(n_chains, -1, -1, -1, -1)
    dens = density.expand(n_chains, -1, -1, -1, -1) if density is not None else None

    # Pocket mask (zero score / noise inside occupied voxels)
    mask = (poc > 0).any(dim=1, keepdim=True).expand_as(y)

    def score(y_):
        xhat = model.forward(y_, poc, density=dens)
        return (xhat - y_) / (sigma ** 2)

    def walk(y_, v_, n):
        for _ in range(n):
            y_ = y_ + delta * v_ / 2
            psi  = score(y_)
            noise = torch.randn_like(y_)
            noise[mask] = 0.
            psi[mask]   = 0.
            v_ = v_ + u * delta * psi / 2
            v_ = zeta1 * v_ + u * delta * psi / 2 \
               + math.sqrt(u * (1 - zeta2)) * noise
            y_ = y_ + delta * v_ / 2
        return y_, v_

    # Warm up
    if warmup > 0:
        y, v = walk(y, v, warmup)

    # Sample: walk → jump → collect
    voxels = []
    for _ in range(0, max_steps, steps):
        y, v = walk(y, v, steps)
        xhat = model.forward(y, poc, density=dens)
        xhat[xhat < threshold] = 0.
        voxels.append(xhat.cpu())

    torch.cuda.empty_cache()
    return torch.cat(voxels, dim=0)


# ── Molecule saving ────────────────────────────────────────────────────────────

def _vox_stats(vox_mols: torch.Tensor, threshold: float = 0.2) -> str:
    flat = vox_mols.detach().float()
    above = flat.gt(threshold).sum().item()
    total = flat.numel()
    return (f"shape={tuple(vox_mols.shape)}  "
            f"mean={flat.mean():.4f}  max={flat.max():.4f}  "
            f"above_thr={above}/{total} ({100*above/total:.1f}%)")


def save_samples(vox_mols: torch.Tensor, voxelizer: Voxelizer,
                 center: torch.Tensor, out_dir: str, n_samples: int):
    print(f"  voxel stats : {_vox_stats(vox_mols)}")
    makedir(out_dir)
    mols, smiles_seen = [], set()
    n_none, n_obabel_fail = 0, 0
    for vox in vox_mols:
        if len(mols) >= n_samples:
            break
        mol_list = voxelizer.vox2mol(
            vox.detach().unsqueeze(0).cuda(), center_coords=center
        )
        if mol_list is None:
            n_none += 1
            continue
        for mol in mol_list:
            sdf_path = os.path.join(out_dir, f"sample_{len(mols):03d}.sdf")
            mol = mol2rdkit_obabel(mol, sdf_path)
            if mol is None:
                n_obabel_fail += 1
                continue
            smi = Chem.MolToSmiles(mol)
            if smi not in smiles_seen:
                smiles_seen.add(smi)
                mols.append(mol)

    if n_none > 0 or n_obabel_fail > 0:
        print(f"  vox2mol=None: {n_none}/{len(vox_mols)}  "
              f"obabel_fail: {n_obabel_fail}")

    if not mols:
        print(f"  WARNING: no valid molecules saved to {out_dir}")
        return 0

    with Chem.SDWriter(os.path.join(out_dir, "samples.sdf")) as w:
        for mol in mols:
            try:
                w.write(mol)
            except Exception:
                pass
    print(f"  Saved {len(mols)} molecules → {out_dir}/samples.sdf")
    return len(mols)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare baseline vs X-ray conditioned VoxBind sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pretrained_path", required=True,
                   help="Original pretrained model dir (contains checkpoint.pth.tar)")
    p.add_argument("--finetune_ckpt",   default=None,
                   help="Fine-tuned checkpoint without density (finetune_ckpt.pth.tar). "
                        "Optional — enables three-way comparison when given.")
    p.add_argument("--xray_ckpt",       required=True,
                   help="Density-finetuned checkpoint (overfit_ckpt.pth.tar)")
    p.add_argument("--data_dir",    default="dataset/data")
    p.add_argument("--ccp4_dir",    default="dataset/data/ccp4")
    p.add_argument("--crops_dir",   default=None,
                   help="Precomputed crops dir (faster than ccp4_dir)")
    p.add_argument("--sample_idx",  type=int, default=None,
                   help="Dataset index to sample from (auto from xray_ckpt if not set)")
    p.add_argument("--out_dir",     default="exps/poc_xray/compare")
    p.add_argument("--n_samples",   type=int, default=10,
                   help="Number of molecules to generate per model")
    p.add_argument("--n_chains",    type=int, default=10,
                   help="WJS chains per iteration")
    p.add_argument("--warmup",      type=int, default=400)
    p.add_argument("--steps",       type=int, default=100)
    p.add_argument("--max_steps",   type=int, default=100)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    model_base, sigma_base  = load_baseline(args.pretrained_path, device)

    model_ft, sigma_ft, ft_idx, ft_pocket = (None, None, None, None)
    if args.finetune_ckpt:
        model_ft, sigma_ft, ft_idx, ft_pocket = load_finetuned_model(
            args.finetune_ckpt, device)

    model_xray, sigma_xray, ckpt_idx, pocket_id = load_xray_model(args.xray_ckpt, device)

    for tag, s in [("baseline", sigma_base), ("xray_cond", sigma_xray)]:
        if s != sigma_base:
            print(f"WARNING: sigma mismatch — baseline={sigma_base}, {tag}={s}")
    if sigma_ft is not None and sigma_ft != sigma_base:
        print(f"WARNING: sigma mismatch — baseline={sigma_base}, finetuned={sigma_ft}")

    # ── Dataset — find the target sample ─────────────────────────────────────
    idx = args.sample_idx if args.sample_idx is not None else ckpt_idx
    if idx is None:
        raise ValueError("Cannot determine sample index. Pass --sample_idx explicitly.")

    dset = DatasetCrossDockedXray(
        data_dir=args.data_dir,
        ccp4_dir=args.ccp4_dir,
        crops_dir=args.crops_dir,
        split="train",
        aug=False,
        use_xray=True,
    )

    loader = DataLoader(Subset(dset, [idx]), batch_size=1,
                        shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"\nSampling pocket : {batch['pocket']['id'][0]}")
    print(f"xray_available  : {batch['xray_available'].item()}")
    if not batch["xray_available"].item():
        print("WARNING: no X-ray density for this sample — xray_cond will use zeros.")

    # ── Voxelize (fixed) ──────────────────────────────────────────────────────
    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))

    with torch.no_grad():
        vox_lig = voxelizer(batch["ligand"], num_channels=N_LIGAND_ELEMENTS)  # (1,7,G,G,G)
        vox_poc = voxelizer(batch["pocket"], num_channels=N_POCKET_ELEMENTS)  # (1,4,G,G,G)
        xray    = batch["xray_density"].unsqueeze(1).to(device)               # (1,1,G,G,G)
        center  = batch["pocket"]["center_coords"]                             # (1,3)

    print(f"\nvox_poc  non-zero : {vox_poc.gt(0).sum().item()}")
    print(f"xray     std      : {xray.std():.4f}")

    # ── Run WJS sampling ──────────────────────────────────────────────────────
    wjs_kw = dict(
        pocket_vox=vox_poc,
        ligand_vox=vox_lig,
        n_chains=args.n_chains,
        warmup=args.warmup,
        steps=args.steps,
        max_steps=args.max_steps,
    )

    n_models = 3 if model_ft is not None else 2

    print(f"\n[1/{n_models}] Sampling with baseline (original, no density) ...")
    vox_base = wjs_sample(model_base, density=None, **wjs_kw)
    print(f"  baseline voxels : {_vox_stats(vox_base)}")

    vox_ft = None
    if model_ft is not None:
        print(f"\n[2/{n_models}] Sampling with finetuned (no density) ...")
        vox_ft = wjs_sample(model_ft, density=None, **wjs_kw)
        print(f"  finetuned voxels: {_vox_stats(vox_ft)}")

    print(f"\n[{n_models}/{n_models}] Sampling with xray_cond (fine-tuned + density) ...")
    vox_xray = wjs_sample(model_xray, density=xray, **wjs_kw)
    print(f"  xray_cond voxels: {_vox_stats(vox_xray)}")

    # ── Save molecules ────────────────────────────────────────────────────────
    makedir(args.out_dir)
    base_dir = os.path.join(args.out_dir, "baseline")
    xray_dir = os.path.join(args.out_dir, "xray_cond")
    ft_dir   = os.path.join(args.out_dir, "finetuned") if model_ft is not None else None

    print(f"\nSaving molecules ...")
    n_base = save_samples(vox_base, voxelizer, center, base_dir, args.n_samples)
    n_ft   = save_samples(vox_ft, voxelizer, center, ft_dir, args.n_samples) \
             if vox_ft is not None else None
    n_xray = save_samples(vox_xray, voxelizer, center, xray_dir, args.n_samples)

    # Save reference pocket/ligand
    data_dir = args.data_dir
    poc_id   = batch["pocket"]["id"][0]
    lig_id   = batch["ligand"]["id"][0]
    for ref_id in [poc_id, lig_id]:
        src = os.path.join(data_dir, "crossdocked_pocket10", ref_id)
        dst = os.path.join(args.out_dir, ref_id.replace("/", "__"))
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    print(f"\n{'='*55}")
    print(f"  {'Condition':<30}  {'Molecules':>9}")
    print(f"  {'-'*43}")
    print(f"  {'1. Baseline (original)':<30}  {n_base:>9}")
    if n_ft is not None:
        print(f"  {'2. Fine-tuned (no density)':<30}  {n_ft:>9}")
    print(f"  {'3. Fine-tuned + X-ray density':<30}  {n_xray:>9}")
    print(f"{'='*55}")
    print(f"\nResults → {args.out_dir}/")
    print(f"  baseline/samples.sdf")
    if n_ft is not None:
        print(f"  finetuned/samples.sdf")
    print(f"  xray_cond/samples.sdf")


if __name__ == "__main__":
    main()
