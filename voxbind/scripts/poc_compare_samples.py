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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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
        with tqdm(total=warmup, desc="  warmup", leave=False, dynamic_ncols=True) as pbar:
            def walk_warmup(y_, v_, n):
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
                    pbar.update(1)
                return y_, v_
            y, v = walk_warmup(y, v, warmup)

    # Sample: walk → jump → collect
    n_jumps = max_steps // steps
    voxels = []
    for jump_i in tqdm(range(n_jumps), desc="  sampling", leave=False, dynamic_ncols=True,
                       unit="jump"):
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


def sample_until_done(
    model: VoxBind,
    voxelizer: Voxelizer,
    center: torch.Tensor,
    out_dir: str,
    n_samples: int,
    wjs_kw: dict,
    density: torch.Tensor = None,
    threshold: float = 0.2,
    max_attempts: int = 500,
) -> int:
    """Mirror the original sample_molecules() while-loop: warmup + walk → reconstruct,
    repeat until n_samples valid unique molecules are collected or max_attempts reached.
    pocket_vox and ligand_vox are passed via wjs_kw."""
    makedir(out_dir)
    mols, smiles_seen = [], set()
    n_no_peaks = n_recon_fail = n_duplicate = n_attempted = 0

    with tqdm(total=n_samples, desc="  valid mols", dynamic_ncols=True) as pbar_valid:
        while len(mols) < n_samples and n_attempted < max_attempts:
            vox_mols = wjs_sample(model, density=density, threshold=threshold, **wjs_kw)
            print(f"  voxel stats : {_vox_stats(vox_mols, threshold)}")

            for vox in vox_mols:
                if len(mols) >= n_samples or n_attempted >= max_attempts:
                    break
                n_attempted += 1
                mol_list = voxelizer.vox2mol(
                    vox.detach().unsqueeze(0).cuda(), center_coords=center
                )
                if mol_list is None:
                    n_no_peaks += 1
                    continue
                for mol in mol_list:
                    sdf_path = os.path.join(out_dir, f"sample_{len(mols):03d}.sdf")
                    mol = mol2rdkit_obabel(mol, sdf_path)
                    if mol is None:
                        n_recon_fail += 1
                        continue
                    smi = Chem.MolToSmiles(mol)
                    if smi not in smiles_seen:
                        smiles_seen.add(smi)
                        mols.append(mol)
                        pbar_valid.update(1)
                    else:
                        n_duplicate += 1

    n_valid = len(mols)
    print(f"  Filter breakdown ({n_attempted} voxels attempted):")
    print(f"    no peaks (vox2mol=None) : {n_no_peaks:4d}  ({100*n_no_peaks/max(n_attempted,1):.1f}%)")
    print(f"    recon fail (obabel/rdkit): {n_recon_fail:4d}  ({100*n_recon_fail/max(n_attempted,1):.1f}%)")
    print(f"    duplicate SMILES        : {n_duplicate:4d}  ({100*n_duplicate/max(n_attempted,1):.1f}%)")
    print(f"    valid & unique          : {n_valid:4d}  ({100*n_valid/max(n_attempted,1):.1f}%)")

    if not mols:
        print(f"  WARNING: no valid molecules saved to {out_dir}")
        return 0, smiles_seen

    with Chem.SDWriter(os.path.join(out_dir, "samples.sdf")) as w:
        for mol in mols:
            try:
                w.write(mol)
            except Exception:
                pass
    print(f"  Saved {n_valid} molecules → {out_dir}/samples.sdf")
    return n_valid, smiles_seen


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_xray_indices(dset: DatasetCrossDockedXray) -> list:
    """Return dataset indices that have X-ray density available, in order."""
    if dset._crops_available is not None:
        return np.where(dset._crops_available)[0].tolist()
    # CCP4 mode: scan the data list
    indices = []
    for i, (pocket_, _) in enumerate(dset.data):
        pdb_id = dset._get_pdb_id(pocket_["id"])
        if pdb_id in dset._available_pdbs:
            indices.append(i)
    return indices


def _sample_pocket(
    idx: int,
    dset: DatasetCrossDockedXray,
    model_base: VoxBind,
    model_ft,
    model_xray: VoxBind,
    model_rand,
    random_densities,
    voxelizer: Voxelizer,
    device: torch.device,
    args,
    out_dir: str,
):
    """Run the three-way comparison for a single pocket and write results to out_dir."""
    loader = DataLoader(Subset(dset, [idx]), batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    pocket_id = batch["pocket"]["id"][0]
    xray_ok   = batch["xray_available"].item()
    print(f"\n  pocket idx={idx}  id={pocket_id}")
    print(f"  xray_available: {xray_ok}")
    if not xray_ok:
        print("  WARNING: no X-ray density — xray_cond will use zeros.")

    with torch.no_grad():
        vox_lig = voxelizer(batch["ligand"], num_channels=N_LIGAND_ELEMENTS)
        vox_poc = voxelizer(batch["pocket"], num_channels=N_POCKET_ELEMENTS)
        xray    = batch["xray_density"].unsqueeze(1).to(device)
        center  = batch["pocket"]["center_coords"]

    print(f"  vox_poc non-zero: {vox_poc.gt(0).sum().item()}  xray std: {xray.std():.4f}")

    wjs_kw = dict(
        pocket_vox=vox_poc,
        ligand_vox=vox_lig,
        n_chains=args.n_chains,
        warmup=args.warmup,
        steps=args.steps,
        max_steps=args.max_steps,
    )
    sample_kw = dict(voxelizer=voxelizer, center=center,
                     n_samples=args.n_samples, wjs_kw=wjs_kw)

    # Load reference ligand SMILES from the source SDF
    lig_id  = batch["ligand"]["id"][0]
    ref_smi = None
    lig_src = os.path.join(args.data_dir, "crossdocked_pocket10", lig_id)
    if os.path.exists(lig_src):
        suppl = Chem.SDMolSupplier(lig_src, removeHs=True)
        ref_mol = next((m for m in suppl if m is not None), None)
        if ref_mol is not None:
            ref_smi = Chem.MolToSmiles(ref_mol)
    if ref_smi:
        print(f"  ref ligand SMILES : {ref_smi}")
    else:
        print(f"  ref ligand SMILES : (could not load {lig_src})")

    n_models = 2  # baseline + xray_cond + random_density (inference)
    n_models += 1  # random_density always runs
    if model_ft is not None:
        n_models += 1
    if model_rand is not None:
        n_models += 1

    step = 0

    step += 1
    print(f"  [{step}/{n_models}] baseline  threshold={args.threshold}")
    n_base, smi_base = sample_until_done(model_base, density=None, threshold=args.threshold,
                                         out_dir=os.path.join(out_dir, "baseline"), **sample_kw)

    n_ft, smi_ft = None, None
    if model_ft is not None:
        step += 1
        print(f"  [{step}/{n_models}] finetuned  threshold={args.threshold}")
        n_ft, smi_ft = sample_until_done(model_ft, density=None, threshold=args.threshold,
                                         out_dir=os.path.join(out_dir, "finetuned"), **sample_kw)

    step += 1
    print(f"  [{step}/{n_models}] xray_cond  threshold={args.threshold_xray}")
    n_xray, smi_xray = sample_until_done(model_xray, density=xray, threshold=args.threshold_xray,
                                         out_dir=os.path.join(out_dir, "xray_cond"), **sample_kw)

    step += 1
    print(f"  [{step}/{n_models}] random_density  threshold={args.threshold_xray}")
    random_dens = torch.randn_like(xray)
    n_rand, smi_rand = sample_until_done(model_xray, density=random_dens, threshold=args.threshold_xray,
                                         out_dir=os.path.join(out_dir, "random_density"), **sample_kw)

    n_rand_trained, smi_rand_trained = None, None
    if model_rand is not None:
        step += 1
        # Use the same fixed random tensor from training if available
        if random_densities is not None:
            # Find the matching density by sample index position
            rand_dens_trained = random_densities[0]  # primary training sample
            if rand_dens_trained.dim() == 3:
                rand_dens_trained = rand_dens_trained.unsqueeze(0).unsqueeze(0)
        else:
            rand_dens_trained = torch.randn_like(xray)
        print(f"  [{step}/{n_models}] random_trained  threshold={args.threshold_xray}")
        n_rand_trained, smi_rand_trained = sample_until_done(
            model_rand, density=rand_dens_trained, threshold=args.threshold_xray,
            out_dir=os.path.join(out_dir, "random_trained"), **sample_kw)

    # Reference SMILES recovery check
    if ref_smi:
        hit_base  = ref_smi in smi_base
        hit_ft    = ref_smi in smi_ft   if smi_ft   is not None else None
        hit_xray  = ref_smi in smi_xray
        hit_rand  = ref_smi in smi_rand
        hit_rt    = ref_smi in smi_rand_trained if smi_rand_trained is not None else None
        print(f"\n  Reference SMILES recovery:")
        print(f"    baseline       : {'HIT' if hit_base else 'miss'}")
        if hit_ft is not None:
            print(f"    finetuned      : {'HIT' if hit_ft else 'miss'}")
        print(f"    xray_cond      : {'HIT' if hit_xray else 'miss'}")
        print(f"    random_density : {'HIT' if hit_rand else 'miss'}")
        if hit_rt is not None:
            print(f"    random_trained : {'HIT' if hit_rt else 'miss'}")

    # Save reference pocket/ligand files
    for ref_id in [pocket_id, lig_id]:
        src = os.path.join(args.data_dir, "crossdocked_pocket10", ref_id)
        dst = os.path.join(out_dir, ref_id.replace("/", "__"))
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    return n_base, n_ft, n_xray, n_rand, n_rand_trained


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
    p.add_argument("--random_ckpt",     default=None,
                   help="Random-density-trained checkpoint (negative control). "
                        "Optional — adds random_trained condition when given.")
    p.add_argument("--data_dir",    default="dataset/data")
    p.add_argument("--ccp4_dir",    default="dataset/data/ccp4")
    p.add_argument("--crops_dir",   default=None,
                   help="Precomputed crops dir (faster than ccp4_dir)")
    p.add_argument("--split",       default="train",
                   choices=["train", "val", "test"],
                   help="Dataset split to draw evaluation pockets from")
    p.add_argument("--sample_idx",  type=int, default=None,
                   help="Dataset index to sample from (auto from xray_ckpt if not set). "
                        "Ignored when --n_pockets > 1.")
    p.add_argument("--n_pockets",   type=int, default=1,
                   help="Number of X-ray available pockets to evaluate. "
                        "When > 1, iterates through the first N pockets that have X-ray data, "
                        "writing results to out_dir/pocket_{idx:04d}/.")
    p.add_argument("--out_dir",     default="exps/poc_xray/compare")
    p.add_argument("--n_samples",       type=int,   default=40,
                   help="Number of molecules to generate per model")
    p.add_argument("--n_chains",        type=int,   default=10,
                   help="WJS chains per iteration")
    p.add_argument("--warmup",          type=int,   default=400)
    p.add_argument("--steps",           type=int,   default=100)
    p.add_argument("--max_steps",       type=int,   default=100)
    p.add_argument("--threshold",       type=float, default=0.2,
                   help="Voxel threshold for baseline and finetuned sampling")
    p.add_argument("--threshold_xray",  type=float, default=0.3,
                   help="Voxel threshold for x-ray density-conditioned sampling")
    p.add_argument("--seed",            type=int,   default=42)
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

    model_rand, random_densities = None, None
    if args.random_ckpt:
        model_rand, sigma_rand, _, _ = load_xray_model(args.random_ckpt, device)
        # Load the fixed random tensors that were used during training
        rand_chkp = torch.load(args.random_ckpt, map_location=device, weights_only=False)
        random_densities = rand_chkp.get("random_densities", None)
        if random_densities is not None:
            random_densities = [d.to(device) for d in random_densities]
            print(f"[random_trained] loaded {len(random_densities)} fixed random density tensor(s)")
        else:
            print("[random_trained] WARNING: no saved random_densities in checkpoint — will use fresh randn")

    for tag, s in [("baseline", sigma_base), ("xray_cond", sigma_xray)]:
        if s != sigma_base:
            print(f"WARNING: sigma mismatch — baseline={sigma_base}, {tag}={s}")
    if sigma_ft is not None and sigma_ft != sigma_base:
        print(f"WARNING: sigma mismatch — baseline={sigma_base}, finetuned={sigma_ft}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dset = DatasetCrossDockedXray(
        data_dir=args.data_dir,
        ccp4_dir=args.ccp4_dir,
        crops_dir=args.crops_dir,
        split=args.split,
        aug=False,
        use_xray=True,
    )
    voxelizer = Voxelizer(grid_dim=64, resolution=0.25, cubes_around=8, device=str(device))

    # ── Determine pocket indices to evaluate ──────────────────────────────────
    if args.n_pockets == 1:
        idx = args.sample_idx if args.sample_idx is not None else ckpt_idx
        if idx is None:
            raise ValueError("Cannot determine sample index. Pass --sample_idx explicitly.")
        pocket_indices = [idx]
    else:
        xray_indices = _get_xray_indices(dset)
        if not xray_indices:
            raise RuntimeError("No X-ray available indices found in dataset.")
        pocket_indices = xray_indices[:args.n_pockets]
        print(f"\nFound {len(xray_indices)} X-ray available pockets "
              f"— evaluating first {len(pocket_indices)}.")

    makedir(args.out_dir)
    results = []  # list of (idx, n_base, n_ft, n_xray, n_rand, n_rt)

    # ── Per-pocket loop ───────────────────────────────────────────────────────
    for p_i, idx in enumerate(pocket_indices):
        print(f"\n{'='*60}")
        print(f"Pocket {p_i+1}/{len(pocket_indices)}  (dataset idx={idx})")
        print(f"{'='*60}")

        # Output dir: flat for n_pockets=1 (backward compat), subdir otherwise
        if args.n_pockets == 1:
            poc_out = args.out_dir
        else:
            poc_out = os.path.join(args.out_dir, f"pocket_{idx:04d}")

        n_base, n_ft, n_xray, n_rand, n_rt = _sample_pocket(
            idx=idx,
            dset=dset,
            model_base=model_base,
            model_ft=model_ft,
            model_xray=model_xray,
            model_rand=model_rand,
            random_densities=random_densities,
            voxelizer=voxelizer,
            device=device,
            args=args,
            out_dir=poc_out,
        )
        results.append((idx, n_base, n_ft, n_xray, n_rand, n_rt))

    # ── Summary ───────────────────────────────────────────────────────────────
    has_rt = model_rand is not None
    hdr = f"  {'idx':>6}  {'Baseline':>10}  {'Finetuned':>10}  {'X-ray cond':>10}  {'Rand dens':>10}"
    if has_rt:
        hdr += f"  {'Rand train':>11}"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for idx, n_base, n_ft, n_xray, n_rand, n_rt in results:
        ft_str = f"{n_ft:>10}" if n_ft is not None else f"{'—':>10}"
        line = f"  {idx:>6}  {n_base:>10}  {ft_str}  {n_xray:>10}  {n_rand:>10}"
        if has_rt:
            rt_str = f"{n_rt:>11}" if n_rt is not None else f"{'—':>11}"
            line += f"  {rt_str}"
        print(line)
    print(f"{'='*len(hdr)}")
    print(f"\nResults → {args.out_dir}/")


if __name__ == "__main__":
    main()
