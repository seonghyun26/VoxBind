"""01e_pdbbind_features.py — Phase 5: frozen-encoder features for PDBbind.

For one pretraining condition at a time (atomblob OR atomblob_density), load
the EMA encoder weights from a checkpoint, freeze it, run every PDBbind refined
complex through, mean-pool the post-norm patch tokens, and save a 512-D
feature vector per complex.

The pre-norm patch tokens come from the encoder forward up to (but not
including) `decoder_proj`. There are 8³ = 512 patch tokens of dim 512; mean
across the 512 tokens yields a single (512,) vector per complex.

Usage
-----
    cd voxbind
    CUDA_VISIBLE_DEVICES=5 python dataset/01e_pdbbind_features.py \
        --condition atomblob          --epoch 99
    CUDA_VISIBLE_DEVICES=5 python dataset/01e_pdbbind_features.py \
        --condition atomblob_density  --epoch 99

Outputs
-------
    dataset/data/pdbbind/features/atomblob_e99.pt
    dataset/data/pdbbind/features/atomblob_density_e99.pt
        Each is a dict-of-tensors: { pdb_id: torch.float32 [512] }
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voxbind.models.density_vit import DensityViT


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR  = Path(__file__).parent / "data" / "pdbbind"
FEAT_DIR     = PDBBIND_DIR / "features"

# Conditions whose encoder never reads the density channel; their features are
# identical across v1/v2/v3 (atoms are symlinked).
ATOM_ONLY_CONDITIONS = {"atomblob", "atomblob_weighted"}

EXPS = {
    "atomblob":          Path("exps") / "260526_atomblob_vit_mae_40m_pretrain",
    "atomblob_density":  Path("exps") / "260526_atomblob_density_vit_mae_40m_pretrain",
    "atomblob_weighted": Path("exps") / "260528_atomblob_vit_mae_40m_weighted_pretrain",
}


def voxel_dir_for(version: str) -> Path:
    """Resolve the voxel directory for a given version tag."""
    if version == "v1":
        return PDBBIND_DIR / "voxels"
    return PDBBIND_DIR / f"voxels_{version}"


def version_suffix(version: str) -> str:
    """Output-filename suffix for the version. v1 is the default → no suffix."""
    return "" if version == "v1" else f"_{version}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_encoder(exp_dir: Path, epoch: int, device: str) -> DensityViT:
    """Instantiate DensityViT from the exp's cfg.yaml and load EMA weights."""
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    m = cfg.model

    encoder = DensityViT(
        grid_dim     = cfg.vox.grid_dim,
        patch_size   = m.patch_size,
        n_in_channels= m.n_in_channels,
        c_out        = m.n_channels // 2,
        dim          = m.dim,
        depth        = m.depth,
        n_heads      = m.heads,
        mlp_ratio    = m.mlp_ratio,
        dropout      = m.dropout,
    )

    ckpt_path = exp_dir / f"checkpoint_e{epoch:04d}.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # encoder_state_dict_ema keys are prefixed 'encoder.' from the wrapper.
    raw = ckpt["encoder_state_dict_ema"]
    stripped = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    missing = encoder.load_state_dict(stripped, strict=True)
    print(f"  loaded {len(stripped)} weights from {ckpt_path.name} (epoch={ckpt.get('epoch')})")
    return encoder.to(device).eval()


@torch.no_grad()
def encode_tokens(encoder: DensityViT, x: torch.Tensor) -> torch.Tensor:
    """(B, n_in, G, G, G) → (B, N=512, D=512). Forward up to encoder.norm."""
    z = encoder.patch_embed(x)            # (B, D, g_p, g_p, g_p)
    z = z.flatten(2).transpose(1, 2)      # (B, N, D)
    z = z + encoder.pos_embed
    for blk in encoder.blocks:
        z = blk(z)
    return encoder.norm(z)


def load_voxels_for(
    pid: str,
    condition: str,
    n_in_channels: int,
    atom_dir: Path,
    dens_dir: Path,
) -> torch.Tensor:
    """Build the (n_in_channels, G, G, G) tensor for one complex."""
    atoms = np.load(atom_dir / f"{pid}.npy")               # (11, G, G, G) float16
    atoms_t = torch.from_numpy(atoms.astype(np.float32))    # promote to float32
    if condition in ATOM_ONLY_CONDITIONS:
        assert n_in_channels == 11
        return atoms_t                                       # (11, G, G, G)
    # atomblob_density
    assert n_in_channels == 12
    dens = np.load(dens_dir / f"{pid}.npy")                 # (G, G, G) float16
    dens_t = torch.from_numpy(dens.astype(np.float32)).unsqueeze(0)  # (1, G, G, G)
    return torch.cat([atoms_t, dens_t], dim=0)              # (12, G, G, G)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frozen-encoder features for PDBbind refined complexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--condition",
                   choices=["atomblob", "atomblob_density", "atomblob_weighted"],
                   required=True)
    p.add_argument("--epoch",      type=int, default=99,
                   help="Checkpoint epoch to use (matched across conditions)")
    p.add_argument("--voxel_version", choices=["v1", "v2", "v3"], default="v1",
                   help="Density-normalisation version: v1 = per-map z-score + "
                        "per-crop ±3σ clip; v2 = pocket-pool z-score; "
                        "v3 = pocket-pool symmetric max-abs. Atom voxels are "
                        "identical across versions (symlinked).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",    default=str(FEAT_DIR))
    p.add_argument("--max_complexes", type=int, default=0,
                   help="Limit to first N (0 = all). Smoke testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = version_suffix(args.voxel_version)
    out_path = out_dir / f"{args.condition}_e{args.epoch}{suffix}.pt"

    vox_dir  = voxel_dir_for(args.voxel_version)
    atom_dir = vox_dir / "atoms"
    dens_dir = vox_dir / "density"
    # availability.csv only exists in v1; v2/v3 reuse v1's since the pid set
    # is the same (they share the same successful-crop universe).
    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = PDBBIND_DIR / "voxels" / "availability.csv"

    print(f"=== PDBbind frozen-encoder features ===")
    print(f"  condition      : {args.condition}")
    print(f"  epoch          : {args.epoch}")
    print(f"  voxel_version  : {args.voxel_version}")
    print(f"  voxel_dir      : {vox_dir}")
    print(f"  device         : {args.device}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  out            : {out_path}")

    # ── Atom-only conditions: features are version-independent, just symlink ──
    if args.voxel_version != "v1" and args.condition in ATOM_ONLY_CONDITIONS:
        src = out_dir / f"{args.condition}_e{args.epoch}.pt"
        if not src.exists():
            print(f"\n[error] v1 baseline {src} doesn't exist.")
            print(f"        Run with --voxel_version v1 first (atom-only conditions "
                  f"don't depend on density).")
            sys.exit(1)
        if not out_path.exists():
            os.symlink(src.resolve(), out_path)
        print(f"\n  [skip] {args.condition} doesn't consume density → features "
              f"identical across versions.")
        print(f"  [symlink] {out_path}  →  {src.name}")
        return

    # ── Distribution-shift warning for density-consuming condition on v2/v3 ──
    if args.voxel_version != "v1" and args.condition == "atomblob_density":
        print(f"\n  [warn] atomblob_density encoder was pretrained on v1 density.")
        print(f"  [warn] --voxel_version {args.voxel_version} feeds the encoder OOD inputs.")
        print(f"  [warn] Probe results reflect distribution shift, not encoder quality.")

    encoder = load_encoder(EXPS[args.condition], args.epoch, args.device)
    n_in = encoder.n_in_channels

    # Pick the right pdb_id pool: density-using conditions need has_density,
    # atom-only conditions (atomblob, atomblob_weighted) just need has_atoms.
    avail = pd.read_csv(avail_csv)
    if args.condition == "atomblob_density":
        pool = avail[avail["has_atoms"] & avail["has_density"]].copy()
    else:
        pool = avail[avail["has_atoms"]].copy()
    if args.max_complexes:
        pool = pool.head(args.max_complexes)
    pids = pool["pdb_id"].tolist()
    print(f"  complexes  : {len(pids):,}")

    features: dict[str, torch.Tensor] = {}
    n_err = 0
    err_log: list[tuple[str, str]] = []

    # Batched forward pass.
    pbar = tqdm(range(0, len(pids), args.batch_size), unit="batch",
                desc=f"encode {args.condition}")
    for start in pbar:
        batch_pids = pids[start : start + args.batch_size]
        tensors, used_pids = [], []
        for pid in batch_pids:
            try:
                tensors.append(load_voxels_for(pid, args.condition, n_in,
                                                atom_dir, dens_dir))
                used_pids.append(pid)
            except Exception as e:
                err_log.append((pid, repr(e)[:160]))
                n_err += 1
        if not tensors:
            continue
        x = torch.stack(tensors, dim=0).to(args.device)            # (B, n_in, G, G, G)
        tokens = encode_tokens(encoder, x)                          # (B, N, D)
        feats = tokens.mean(dim=1).cpu()                            # (B, D)
        for pid, vec in zip(used_pids, feats):
            features[pid] = vec.contiguous().clone()
        pbar.set_postfix(saved=len(features), err=n_err, refresh=False)

    torch.save({
        "condition": args.condition,
        "epoch":     args.epoch,
        "n_in_channels": n_in,
        "feature_dim": encoder.dim,
        "features":  features,
    }, out_path)
    print(f"\n  saved {len(features):,} features → {out_path}  "
          f"({out_path.stat().st_size/1e6:.1f} MB)")
    if err_log:
        err_path = out_dir / f"{args.condition}_e{args.epoch}_errors.txt"
        err_path.write_text("\n".join(f"{p}\t{m}" for p, m in err_log) + "\n")
        print(f"  errors   : {n_err:,}  →  {err_path}")


if __name__ == "__main__":
    main()
