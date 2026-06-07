"""01c_pdbbind_probe.py — Frozen-encoder features + MLP affinity probe for PDBbind.

Consolidated Phase 5 / 6 entry point. Three subcommands:

    cd voxbind
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features --condition atomblob --epoch 99
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py probe --epoch 99 --seeds 3
    CUDA_VISIBLE_DEVICES=6 python dataset/01c_pdbbind_probe.py finetune \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99

──────────────────────────────────────────────────────────────────────────────
features — frozen-encoder features for PDBbind
──────────────────────────────────────────────────────────────────────────────
For one pretraining condition at a time (atomblob OR atomblob_density), load
the EMA encoder weights from a checkpoint, freeze it, run every PDBbind refined
complex through, mean-pool the post-norm patch tokens, and save a 512-D
feature vector per complex.

The pre-norm patch tokens come from the encoder forward up to (but not
including) `decoder_proj`. There are 8³ = 512 patch tokens of dim 512; mean
across the 512 tokens yields a single (512,) vector per complex.

    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features \
        --condition atomblob          --epoch 99
    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py features \
        --condition atomblob_density  --epoch 99
  → dataset/data/pdbbind/features/{condition}_e{epoch}.pt
        Each is a dict-of-tensors: { pdb_id: torch.float32 [512] }

──────────────────────────────────────────────────────────────────────────────
probe — 2-layer MLP probe on frozen-encoder features
──────────────────────────────────────────────────────────────────────────────
Compares pretrained-encoder representations on PDBbind affinity:
  (i)  atomblob          — 11-ch input  (7 lig + 4 pocket atoms)
  (ii) atomblob_density  — 12-ch input  (11 atoms + 1 2Fo-Fc density)

For each condition, the 512-D mean-pooled patch tokens (from `features`) are fed
through an identical 2-layer MLP head:
        Linear(512 → hidden) → SiLU → Dropout → Linear(hidden → 1)

Both conditions are evaluated on the **same complexes** (intersection of
on-disk feature sets across conditions — limited by the density variant
since EDS coverage is the bottleneck). This ensures the Δ in test ρ comes
only from the encoder, not from a different sample pool.

No ligand fingerprint, no other side information — purely the encoder's
512-D representation → pK. The encoder already sees the ligand as
channels 0–6, so its mean-pooled output already encodes ligand identity
implicitly; we test what this representation is worth.

    CUDA_VISIBLE_DEVICES=5 python dataset/01c_pdbbind_probe.py probe \
        --epoch 99 --seeds 3
  → dataset/data/pdbbind/probe_results_e<N>.csv               (this run only)
        cols: condition, seed, n_train, n_val, n_test,
              best_val_spearman, val_pearson, test_spearman, test_pearson,
              test_rmse, epoch_stopped
  → dataset/data/pdbbind/probe_results_consolidated.csv       (every run, appended)
        Same rows + provenance cols source_file, epoch, voxel_version, variant
        (version/variant otherwise live only in the per-run filename). Idempotent
        per source_file. Rebuild from all per-run CSVs any time with:
            python dataset/consolidate_probe_results.py [--summary]

  Optional probe flags:
    --conditions COND ...   default: atomblob atomblob_density atomblob_weighted atomblob_merged_density
    --no_intersect          let each condition use its own pdb_id pool (NOT recommended)
    --no_covalent_filter    keep covalent complexes (default: drop)
    --cl1_only              restrict to LP_PDBBind CL1=True (cleanest subset)
    --max_epochs N          default: 200
    --patience N            default: 30  (epochs of no val-ρ improvement)

──────────────────────────────────────────────────────────────────────────────
finetune — frozen-encoder control vs end-to-end encoder fine-tuning
──────────────────────────────────────────────────────────────────────────────
Quantifies how much *unfreezing the encoder* buys on PDBbind affinity. Both arms
share the identical pdb_id splits and the identical MLP2 head over the mean-pooled
token rep — the ONLY difference is whether encoder gradients flow:

  • frozen   — EMA encoder fixed; encode each complex once (no_grad) → train the
               head on cached features. Reproduces the `probe` baseline.
  • finetune — encoder + head trained end-to-end (voxels re-encoded every step).
               Default full-encoder unfreeze at a low LR (1e-5) + early stopping;
               bf16/fp16 autocast keeps it within a shared GPU's memory.

    CUDA_VISIBLE_DEVICES=6 python dataset/01c_pdbbind_probe.py finetune \
        --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3
  → dataset/data/pdbbind/finetune_results_e<N>_v<ver>.csv
        Probe schema + a `mode` (frozen|finetune), `ft_scope`, `encoder_lr`,
        `head_lr` column, plus a printed frozen-vs-finetune Δ summary.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset/ — for sibling import

from voxbind.models.density_vit import DensityViT
from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore
# Each probe run self-consolidates into one table (see consolidate_probe_results).
# Guarded so a missing/broken aggregator never blocks the probe itself.
try:
    from consolidate_probe_results import append_run
except Exception:  # pragma: no cover
    append_run = None


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
FEAT_DIR    = PDBBIND_DIR / "features"
LP_CSV      = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"
RESULTS_DIR = PDBBIND_DIR

# Conditions whose encoder never reads the density channel; their features are
# identical across v1/v2/v3 (atoms are symlinked).
ATOM_ONLY_CONDITIONS = {"atomblob", "atomblob_weighted"}

# Conditions whose encoder DOES read the density channel; need has_density pids
# AND care about which density-version the model was pretrained on.
DENSITY_CONSUMING = {"atomblob_density", "atomblob_merged_density",
                     "atomblob_merged_density_gradmag", "atomblob_density_gradmag"}

EXPS = {
    "atomblob":               Path("exps") / "260606_atomblob_vit_mae_40m_invfreq_pretrain",   # 260606 v5-ablation (atom-only, version-agnostic)
    "atomblob_density":       Path("exps") / "260526_atomblob_density_vit_mae_40m_pretrain",
    "atomblob_weighted":      Path("exps") / "260528_atomblob_vit_mae_40m_weighted_pretrain",
    "atomblob_merged_density": Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_pretrain",
    # gradmag/ligvdw: merged atoms + density + ‖∇ρ‖ (9-ch), element-wise vdW ligand
    # radii, v4 density. Only a v4 run exists (also in EXPS_OVERRIDE so it's flagged
    # v4-native and skips the OOD warning).
    "atomblob_merged_density_gradmag": Path("exps") / "260603_atomblob_merged_density_vit_mae_40m_weighted_v4_gradmag_ligvdw_pretrain",
    # SEPARATE-channel (non-merged) gradmag/ligvdw encoder (260604): 11 atoms + density
    # + ‖∇ρ‖ (13-ch), element-wise vdW radii, v1 density.
    "atomblob_density_gradmag": Path("exps") / "260604_atomblob_density_vit_mae_40m_weighted_v1_gradmag_ligvdw_pretrain",
}

# Conditions whose encoder was retrained on the v2/v3 density distributions;
# the (condition, version) pair overrides the default exp dir from EXPS.
EXPS_OVERRIDE: dict[tuple[str, str], Path] = {
    ("atomblob_merged_density", "v2"): Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain",
    ("atomblob_merged_density", "v3"): Path("exps") / "260530_atomblob_merged_density_vit_mae_40m_weighted_v3_pretrain",
    ("atomblob_density",        "v2"): Path("exps") / "260531_atomblob_density_vit_mae_40m_v2_pretrain",
    # 260606 v5-ablation (this box): separate-channel atomblob_density on v5 arcsinh density.
    ("atomblob_density",        "v5"): Path("exps") / "260606_atomblob_density_vit_mae_40m_invfreq_v5_pretrain",
    # v4 has two merged_density retrains: 260601 (plain) and 260602 (dual-head,
    # separate atom/density recon heads). We point v4 at the dual-head encoder
    # being evaluated; swap to 260601_..._v4_pretrain to probe the plain variant.
    ("atomblob_merged_density", "v4"): Path("exps") / "260602_atomblob_merged_density_vit_mae_40m_weighted_v4_dualhead_pretrain",
    # gradmag/ligvdw is v4-native (trained on v4 density); mark it so the OOD warning is skipped.
    ("atomblob_merged_density_gradmag", "v4"): Path("exps") / "260603_atomblob_merged_density_vit_mae_40m_weighted_v4_gradmag_ligvdw_pretrain",
    # v1-normalization variant (260604): same gradmag/ligvdw recipe, only the density
    # normalization differs (per-crop ±3σ z-score). Probe with --voxel_version v1.
    ("atomblob_merged_density_gradmag", "v1"): Path("exps") / "260604_atomblob_merged_density_vit_mae_40m_weighted_v1_gradmag_ligvdw_pretrain",
    # SEPARATE-channel gradmag/ligvdw on v5 arcsinh density. 260606 v5-ablation (this box,
    # invfreq recipe, density/gradmag weights 0.1/0.1). [was 260605 svr7 swept-balanced]
    ("atomblob_density_gradmag", "v5"): Path("exps") / "260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain",
}


def resolve_exp(condition: str, version: str) -> Path:
    """Pick the encoder exp dir for a (condition, voxel_version) pair."""
    return EXPS_OVERRIDE.get((condition, version), EXPS[condition])


def voxel_dir_for(version: str) -> Path:
    """Resolve the voxel directory for a given version tag."""
    if version == "v1":
        return PDBBIND_DIR / "voxels"
    return PDBBIND_DIR / f"voxels_{version}"


def version_suffix(version: str) -> str:
    """Output-filename suffix for the version. v1 is the default → no suffix."""
    return "" if version == "v1" else f"_{version}"


# ═══════════════════════════════════════════════════════════════════════════════
# features — frozen-encoder mean-pooled patch tokens
# ═══════════════════════════════════════════════════════════════════════════════

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


def forward_tokens(encoder: DensityViT, x: torch.Tensor) -> torch.Tensor:
    """(B, n_in, G, G, G) → (B, N=512, D=512). Forward up to encoder.norm.

    Grad-capable core (no @no_grad): used both for frozen feature extraction
    (wrapped below) and for end-to-end fine-tuning, where gradients must flow
    back into the encoder. Stops before `decoder_proj` — the MAE recon head is
    irrelevant to the downstream pooled representation.
    """
    z = encoder.patch_embed(x)            # (B, D, g_p, g_p, g_p)
    z = z.flatten(2).transpose(1, 2)      # (B, N, D)
    z = z + encoder.pos_embed
    for blk in encoder.blocks:
        z = blk(z)
    return encoder.norm(z)


@torch.no_grad()
def encode_tokens(encoder: DensityViT, x: torch.Tensor) -> torch.Tensor:
    """No-grad wrapper around `forward_tokens` for frozen feature extraction."""
    return forward_tokens(encoder, x)


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
    # All density-consuming conditions need the density crop loaded.
    dens = np.load(dens_dir / f"{pid}.npy")                 # (G, G, G) float16
    dens_t = torch.from_numpy(dens.astype(np.float32)).unsqueeze(0)  # (1, G, G, G)
    if condition == "atomblob_density":
        assert n_in_channels == 12
        return torch.cat([atoms_t, dens_t], dim=0)          # (12, G, G, G)
    if condition == "atomblob_merged_density":
        # Pocket {C,O,N,S} (channels 7..10) folded into ligand {C,O,N,S} (0..3).
        # Ligand-only halogens + P (channels 4..6) stay as-is.
        assert n_in_channels == 8
        merged = atoms_t[:7].clone()                         # (7, G, G, G)
        merged[:4] += atoms_t[7:11]
        return torch.cat([merged, dens_t], dim=0)            # (8, G, G, G)
    if condition == "atomblob_merged_density_gradmag":
        # merged atoms (7) + density (1) + ‖∇ρ‖ (1) = 9, mirroring training's
        # channel layout. ‖∇ρ‖ is derived from the (already-normalised) density
        # crop exactly as dataset/train: per_sample_zscore(gradient_magnitude3d(ρ)).
        # NB: atoms here come from voxels_ligvdw (element-wise vdW ligand radii) —
        # run_features overrides atom_dir for this condition.
        assert n_in_channels == 9
        merged = atoms_t[:7].clone()                         # (7, G, G, G)
        merged[:4] += atoms_t[7:11]
        g = per_sample_zscore(gradient_magnitude3d(dens_t.unsqueeze(0))).squeeze(0)  # (1, G, G, G)
        return torch.cat([merged, dens_t, g], dim=0)         # (9, G, G, G)
    if condition == "atomblob_density_gradmag":
        # SEPARATE atoms (11: 7 lig + 4 poc, NOT merged) + density (1) + ‖∇ρ‖ (1) = 13.
        # Like atomblob_density but with the trailing gradmag channel; atoms come from
        # voxels_ligvdw (element-wise vdW radii) via the run_features override below.
        assert n_in_channels == 13
        g = per_sample_zscore(gradient_magnitude3d(dens_t.unsqueeze(0))).squeeze(0)  # (1, G, G, G)
        return torch.cat([atoms_t, dens_t, g], dim=0)        # (13, G, G, G)
    raise ValueError(f"unknown condition: {condition!r}")


def run_features(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = version_suffix(args.voxel_version)
    out_path = out_dir / f"{args.condition}_e{args.epoch}{suffix}.pt"

    vox_dir  = voxel_dir_for(args.voxel_version)
    atom_dir = vox_dir / "atoms"
    dens_dir = vox_dir / "density"
    # The gradmag/ligvdw encoder trained on element-wise vdW ligand blobs, so it
    # needs the vdW atom voxels (voxels_ligvdw/atoms), not the uniform-0.5 set the
    # version dirs symlink. Density (and the ‖∇ρ‖ derived from it) still come from
    # the version dir (v4).
    if args.condition in ("atomblob_merged_density_gradmag", "atomblob_density_gradmag"):
        atom_dir = PDBBIND_DIR / "voxels_ligvdw" / "atoms"
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

    # ── Distribution-shift warning when the encoder wasn't retrained on the
    # selected density version. We override the exp dir for (condition, version)
    # combos that have a matching retrain; if no override exists for this combo
    # AND the condition reads density AND version != v1, the encoder is OOD.
    is_density = args.condition in DENSITY_CONSUMING
    has_override = (args.condition, args.voxel_version) in EXPS_OVERRIDE
    if is_density and args.voxel_version != "v1" and not has_override:
        print(f"\n  [warn] {args.condition} encoder was pretrained on v1 density.")
        print(f"  [warn] --voxel_version {args.voxel_version} feeds the encoder OOD inputs.")
        print(f"  [warn] Probe results reflect distribution shift, not encoder quality.")

    exp_dir = resolve_exp(args.condition, args.voxel_version)
    print(f"  exp_dir        : {exp_dir}")
    encoder = load_encoder(exp_dir, args.epoch, args.device)
    n_in = encoder.n_in_channels

    # Pick the right pdb_id pool: density-using conditions need has_density,
    # atom-only conditions (atomblob, atomblob_weighted) just need has_atoms.
    avail = pd.read_csv(avail_csv)
    if args.condition in DENSITY_CONSUMING:
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


# ═══════════════════════════════════════════════════════════════════════════════
# probe — 2-layer MLP on frozen-encoder features
# ═══════════════════════════════════════════════════════════════════════════════

def load_lp_index(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def build_dataset(
    features: dict[str, torch.Tensor],
    lp_df: pd.DataFrame,
    drop_covalent: bool,
    cl1_only: bool,
) -> dict:
    """Build train/val/test splits keyed by LP_PDBBind 'new_split'.

    Returns dict with X, y, pid for each split (numpy arrays).
    """
    df = lp_df.copy()
    if drop_covalent:
        df = df[~df["covalent"].astype(bool)]
    if cl1_only:
        df = df[df["CL1"].astype(bool)]
    df = df[df["pdb_id"].isin(features.keys())]
    df = df[df["new_split"].isin(["train", "val", "test"])]
    df = df.dropna(subset=["pK"])

    split_data: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        sub  = df[df["new_split"] == split]
        pids = sub["pdb_id"].tolist()
        X    = np.stack([features[p].numpy() for p in pids]).astype(np.float32)
        y    = sub["pK"].astype(np.float32).to_numpy()
        split_data[split] = {"X": X, "y": y, "pid": pids}
    return split_data


class MLP2(nn.Module):
    """Probe head: input_dim → hidden → 1."""
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_one(
    data: dict, *, seed: int, device: str, max_epochs: int, patience: int,
    batch_size: int, lr: float, weight_decay: float, hidden: int, dropout: float,
) -> dict:
    """Train a single MLP probe; return metrics dict."""
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr, ytr = torch.from_numpy(data["train"]["X"]), torch.from_numpy(data["train"]["y"])
    Xva, yva = torch.from_numpy(data["val"  ]["X"]), torch.from_numpy(data["val"  ]["y"])
    Xte, yte = torch.from_numpy(data["test" ]["X"]), torch.from_numpy(data["test" ]["y"])
    Xtr, ytr, Xva, yva, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xva, yva, Xte, yte))

    model = MLP2(Xtr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_train = Xtr.shape[0]
    best_val = -np.inf
    best_state = None
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(max_epochs):
        # mini-batch SGD
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()

        # val Spearman
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva).cpu().numpy()
        val_spearman = spearmanr(pred_va, yva.cpu().numpy()).statistic
        if val_spearman > best_val:
            best_val = val_spearman
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva).cpu().numpy()
        pred_te = model(Xte).cpu().numpy()
    yva_np = yva.cpu().numpy()
    yte_np = yte.cpu().numpy()

    return {
        "n_train": int(n_train),
        "n_val":   int(Xva.shape[0]),
        "n_test":  int(Xte.shape[0]),
        "best_val_spearman": float(best_val),
        "val_pearson":       float(pearsonr (pred_va, yva_np).statistic),
        "test_spearman":     float(spearmanr(pred_te, yte_np).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte_np).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte_np) ** 2).mean())),
        "epoch_stopped":     int(best_epoch),
    }


def run_probe(args: argparse.Namespace) -> None:
    suffix = "" if args.voxel_version == "v1" else f"_{args.voxel_version}"
    out_csv = Path(args.out_csv) if args.out_csv else (
        RESULTS_DIR / f"probe_results_e{args.epoch}{suffix}.csv"
    )

    print(f"=== PDBbind frozen-encoder probe (pocket repr only) ===")
    print(f"  conditions    : {args.conditions}")
    print(f"  epoch         : {args.epoch}")
    print(f"  voxel_version : {args.voxel_version}")
    print(f"  seeds         : {args.seeds}")
    print(f"  device        : {args.device}")
    print(f"  intersect     : {not args.no_intersect}")
    print(f"  drop_covalent : {not args.no_covalent_filter}")
    print(f"  cl1_only      : {args.cl1_only}")
    print(f"  out_csv       : {out_csv}")

    lp_df = load_lp_index(LP_CSV)
    print(f"  LP rows       : {len(lp_df):,}")

    # ── Load all feature bundles upfront so we can intersect pdb_ids ─────────
    all_feats: dict[str, dict[str, torch.Tensor]] = {}
    for cond in args.conditions:
        feat_path = FEAT_DIR / f"{cond}_e{args.epoch}{suffix}.pt"
        if not feat_path.exists():
            print(f"\n[error] missing features: {feat_path}")
            print(f"        Run: python dataset/01c_pdbbind_probe.py features "
                  f"--condition {cond} --voxel_version {args.voxel_version}")
            sys.exit(1)
        bundle = torch.load(feat_path, weights_only=False)
        all_feats[cond] = bundle["features"]
        print(f"  loaded {cond:24s}: {len(all_feats[cond]):,} feats (dim={bundle['feature_dim']})")

    if args.no_intersect:
        shared = None
    else:
        shared = set.intersection(*(set(f.keys()) for f in all_feats.values()))
        print(f"  shared pids   : {len(shared):,}  (intersection across conditions)")

    rows = []
    for cond in args.conditions:
        feats = all_feats[cond]
        if shared is not None:
            feats = {p: v for p, v in feats.items() if p in shared}

        print(f"\n── {cond} ──────────────────────────────────────────────────────")
        data = build_dataset(
            feats, lp_df,
            drop_covalent   = not args.no_covalent_filter,
            cl1_only        = args.cl1_only,
        )
        print(f"  split sizes   : train={len(data['train']['pid']):,}  "
              f"val={len(data['val']['pid']):,}  test={len(data['test']['pid']):,}")
        print(f"  input dim     : {data['train']['X'].shape[1]}")

        for seed in range(args.seeds):
            m = train_one(
                data, seed=seed, device=args.device,
                max_epochs=args.max_epochs, patience=args.patience,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden, dropout=args.dropout,
            )
            row = {"condition": cond, "seed": seed, **m}
            rows.append(row)
            print(f"  seed={seed}  ep_stop={m['epoch_stopped']:3d}  "
                  f"val_ρ={m['best_val_spearman']:.4f}  "
                  f"val_r={m['val_pearson']:.4f}  "
                  f"test_ρ={m['test_spearman']:.4f}  "
                  f"test_r={m['test_pearson']:.4f}  "
                  f"test_rmse={m['test_rmse']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n── Summary (mean ± std across seeds) ───────────────────────────────")
    agg = df.groupby("condition")[
        ["test_spearman", "test_pearson", "test_rmse", "best_val_spearman", "val_pearson"]
    ].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    print(f"\n[write] {out_csv}")

    # Fold this run into the consolidated table so results don't stay scattered
    # across per-run CSVs — the filename-encoded version/variant become columns.
    # Idempotent per source_file; rebuild any time with consolidate_probe_results.
    if append_run is not None:
        try:
            cpath = append_run(df, out_csv.name, RESULTS_DIR,
                               epoch=args.epoch, voxel_version=args.voxel_version)
            print(f"[consolidate] +{len(df)} rows → {cpath}")
        except Exception as e:
            print(f"[consolidate] skipped ({e!r})")


# ═══════════════════════════════════════════════════════════════════════════════
# finetune — end-to-end encoder fine-tuning vs frozen control (matched splits)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `probe` freezes the encoder and trains an MLP head on cached, mean-pooled token
# features. `finetune` answers a different question: how much does letting
# gradients flow *into the encoder* buy on PDBbind affinity?
#
# To keep the comparison honest, both arms run over the SAME pdb_id splits and the
# SAME MLP2 head (512 → hidden → 1) on the SAME mean-pooled token rep:
#
#   • mode="frozen"   — encoder EMA weights fixed; encode every complex ONCE
#                       (no_grad) → train the head on cached features. Identical in
#                       expectation to re-running the frozen encoder each epoch,
#                       just far cheaper, and reproduces the `probe` baseline.
#   • mode="finetune" — encoder + head trained end-to-end; voxels are re-encoded
#                       every step so the encoder receives gradient. A low encoder
#                       LR (default 1e-5) + early stopping regularize the 40M-param
#                       backbone against the ~2.7k-complex train set.
#
# The only thing that differs is whether the encoder learns; the Δ in test ρ is the
# value of fine-tuning.


class FineTuneModel(nn.Module):
    """Encoder (DensityViT) → mean-pool token rep → MLP2 affinity head."""

    def __init__(self, encoder: DensityViT, hidden: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = MLP2(encoder.dim, hidden=hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = forward_tokens(self.encoder, x)   # (B, N, D) — grad flows into encoder
        z = z.mean(dim=1)                     # (B, D) — same pooling as `features`
        return self.head(z)                   # (B,)


class VoxelDataset(Dataset):
    """Lazily build the (n_in, G, G, G) input + pK target for each complex.

    Voxels load from disk per __getitem__ (workers + OS page cache make repeat
    epochs cheap). The gradmag channel is derived on the fly inside
    `load_voxels_for`, matching training and frozen feature extraction exactly.
    """

    def __init__(self, pids, ys, condition, n_in, atom_dir, dens_dir):
        self.pids = list(pids)
        self.ys = np.asarray(ys, dtype=np.float32)
        self.condition = condition
        self.n_in = n_in
        self.atom_dir = atom_dir
        self.dens_dir = dens_dir

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int):
        x = load_voxels_for(self.pids[i], self.condition, self.n_in,
                            self.atom_dir, self.dens_dir)
        return x, torch.tensor(self.ys[i], dtype=torch.float32)


def make_encoder_factory(exp_dir: Path, epoch: int):
    """Load the EMA checkpoint ONCE; return a thunk minting fresh encoders.

    Each fine-tune seed needs an independent encoder initialised from the same
    pretrained weights, so we cache the stripped state_dict and re-instantiate
    (cheap) rather than re-reading the ~0.5 GB checkpoint per seed.
    """
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    m = cfg.model
    ckpt_path = exp_dir / f"checkpoint_e{epoch:04d}.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["encoder_state_dict_ema"]
    stripped = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}

    def factory() -> DensityViT:
        enc = DensityViT(
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
        enc.load_state_dict(stripped, strict=True)
        return enc

    print(f"  loaded EMA encoder weights from {ckpt_path.name} "
          f"(epoch={ckpt.get('epoch')}, {len(stripped)} tensors)")
    return factory, cfg


def split_exists_filter(splits: dict, condition: str, atom_dir: Path,
                        dens_dir: Path) -> dict:
    """Drop pids whose voxel files are missing so both arms share one pid pool."""
    needs_density = condition not in ATOM_ONLY_CONDITIONS
    out = {}
    for split, sdf in splits.items():
        keep = [pid for pid in sdf["pdb_id"]
                if (atom_dir / f"{pid}.npy").exists()
                and (not needs_density or (dens_dir / f"{pid}.npy").exists())]
        out[split] = sdf[sdf["pdb_id"].isin(keep)].reset_index(drop=True)
    return out


def build_splits(condition: str, *, drop_covalent: bool, cl1_only: bool,
                 atom_dir: Path, dens_dir: Path, avail_csv: Path) -> dict:
    """Train/val/test pdb_id+pK frames for one condition (LP_PDBBind new_split).

    Single-condition pool: density-using conditions need has_atoms & has_density;
    atom-only need has_atoms. Mirrors `probe`'s pool/filters so a single-condition
    `probe` and this `finetune` see the identical complexes.
    """
    avail = pd.read_csv(avail_csv)
    if condition in DENSITY_CONSUMING:
        pool = avail[avail["has_atoms"] & avail["has_density"]]
    else:
        pool = avail[avail["has_atoms"]]
    pid_set = set(pool["pdb_id"])

    df = load_lp_index(LP_CSV)
    if drop_covalent:
        df = df[~df["covalent"].astype(bool)]
    if cl1_only:
        df = df[df["CL1"].astype(bool)]
    df = df[df["pdb_id"].isin(pid_set)]
    df = df[df["new_split"].isin(["train", "val", "test"])].dropna(subset=["pK"])

    splits = {s: df[df["new_split"] == s][["pdb_id", "pK"]].reset_index(drop=True)
              for s in ("train", "val", "test")}
    return split_exists_filter(splits, condition, atom_dir, dens_dir)


@torch.no_grad()
def encode_split_features(encoder, splits, condition, n_in, atom_dir, dens_dir,
                          device, batch_size):
    """Frozen-encoder mean-pooled features → {split: {X, y, pid}} for train_one."""
    encoder = encoder.to(device).eval()
    data = {}
    for split, sdf in splits.items():
        pk_map = dict(zip(sdf["pdb_id"], sdf["pK"].astype(np.float32)))
        pids = sdf["pdb_id"].tolist()
        feats, used = [], []
        for start in tqdm(range(0, len(pids), batch_size),
                          desc=f"encode {split}", unit="batch", leave=False):
            tensors, ok = [], []
            for pid in pids[start : start + batch_size]:
                try:
                    tensors.append(load_voxels_for(pid, condition, n_in,
                                                   atom_dir, dens_dir)); ok.append(pid)
                except Exception:
                    pass
            if not tensors:
                continue
            x = torch.stack(tensors).to(device)
            feats.append(forward_tokens(encoder, x).mean(dim=1).cpu())
            used += ok
        data[split] = {
            "X": torch.cat(feats).numpy().astype(np.float32),
            "y": np.array([pk_map[p] for p in used], dtype=np.float32),
            "pid": used,
        }
    return data


def set_encoder_trainable(encoder: DensityViT, scope: str, last_k: int) -> list:
    """Flag which encoder params get gradient; return the trainable list.

    scope="full"  — patch_embed + pos_embed + all blocks + norm (decoder_proj is
                    unused downstream, so it stays frozen/excluded).
    scope="lastk" — only the last `last_k` transformer blocks + final norm; the
                    patch embed, positional embedding and earlier blocks freeze.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)
    if scope == "full":
        for n, p in encoder.named_parameters():
            if not n.startswith("decoder_proj"):
                p.requires_grad_(True)
    elif scope == "lastk":
        for p in encoder.norm.parameters():
            p.requires_grad_(True)
        for blk in encoder.blocks[len(encoder.blocks) - last_k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    else:
        raise ValueError(f"unknown ft scope: {scope!r}")
    return [p for n, p in encoder.named_parameters()
            if p.requires_grad and not n.startswith("decoder_proj")]


@torch.no_grad()
def infer_loader(model, loader, device, amp, amp_dtype) -> np.ndarray:
    """Batched no-grad predictions over a DataLoader → (N,) numpy."""
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            preds.append(model(xb).float().cpu())
    return torch.cat(preds).numpy()


def train_finetune(splits, factory, *, seed, device, condition, n_in,
                   atom_dir, dens_dir, max_epochs, patience, batch_size,
                   accum_steps, head_lr, encoder_lr, weight_decay, grad_clip,
                   hidden, dropout, num_workers, amp, ft_scope, last_k) -> dict:
    """Fine-tune encoder + head end-to-end; return metrics (best val ρ → test)."""
    torch.manual_seed(seed); np.random.seed(seed)

    encoder = factory()
    enc_params = set_encoder_trainable(encoder, ft_scope, last_k)
    model = FineTuneModel(encoder, hidden=hidden, dropout=dropout).to(device)

    opt = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        {"params": enc_params,              "lr": encoder_lr, "weight_decay": weight_decay},
    ])
    loss_fn = nn.MSELoss()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def make_loader(split, shuffle):
        sdf = splits[split]
        ds = VoxelDataset(sdf["pdb_id"], sdf["pK"].to_numpy(), condition, n_in,
                          atom_dir, dens_dir)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    train_loader = make_loader("train", True)
    val_loader   = make_loader("val", False)
    test_loader  = make_loader("test", False)
    yva = splits["val"]["pK"].to_numpy().astype(np.float32)
    yte = splits["test"]["pK"].to_numpy().astype(np.float32)

    best_val, best_state, best_epoch, since = -np.inf, None, -1, 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss = loss_fn(model(xb), yb) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        pred_va = infer_loader(model, val_loader, device, amp, amp_dtype)
        val_spearman = spearmanr(pred_va, yva).statistic
        if val_spearman > best_val:
            best_val = val_spearman
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    pred_va = infer_loader(model, val_loader, device, amp, amp_dtype)
    pred_te = infer_loader(model, test_loader, device, amp, amp_dtype)
    return {
        "n_train": len(splits["train"]),
        "n_val":   len(splits["val"]),
        "n_test":  len(splits["test"]),
        "best_val_spearman": float(best_val),
        "val_pearson":       float(pearsonr (pred_va, yva).statistic),
        "test_spearman":     float(spearmanr(pred_te, yte).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte) ** 2).mean())),
        "epoch_stopped":     int(best_epoch),
    }


def run_finetune(args: argparse.Namespace) -> None:
    device = args.device
    amp = (not args.no_amp) and str(device).startswith("cuda")
    exp_dir = resolve_exp(args.condition, args.voxel_version)

    # Distribution-shift warning: density-consuming encoder fed an un-retrained
    # density version (same rationale as run_features).
    is_density = args.condition in DENSITY_CONSUMING
    has_override = (args.condition, args.voxel_version) in EXPS_OVERRIDE
    if is_density and args.voxel_version != "v1" and not has_override:
        print(f"  [warn] {args.condition} encoder pretrained on v1 density; "
              f"--voxel_version {args.voxel_version} is OOD.")

    vox_dir  = voxel_dir_for(args.voxel_version)
    atom_dir = vox_dir / "atoms"
    dens_dir = vox_dir / "density"
    # gradmag/ligvdw encoders trained on element-wise vdW atom blobs (see run_features).
    if args.condition in ("atomblob_merged_density_gradmag", "atomblob_density_gradmag"):
        atom_dir = PDBBIND_DIR / "voxels_ligvdw" / "atoms"
    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = PDBBIND_DIR / "voxels" / "availability.csv"

    suffix = version_suffix(args.voxel_version)
    out_csv = Path(args.out_csv) if args.out_csv else (
        RESULTS_DIR / f"finetune_results_e{args.epoch}{suffix}.csv")

    print(f"=== PDBbind frozen-vs-finetune ({args.condition}) ===")
    print(f"  modes          : {args.modes}")
    print(f"  epoch          : {args.epoch}")
    print(f"  voxel_version  : {args.voxel_version}")
    print(f"  exp_dir        : {exp_dir}")
    print(f"  atom_dir       : {atom_dir}")
    print(f"  dens_dir       : {dens_dir}")
    print(f"  seeds          : {args.seeds}")
    print(f"  device         : {device}  (amp={amp})")
    print(f"  ft_scope       : {args.ft_scope}"
          + (f" (last_k={args.last_k})" if args.ft_scope == "lastk" else ""))
    print(f"  enc_lr/head_lr : {args.encoder_lr} / {args.head_lr}")
    print(f"  bsz x accum    : {args.batch_size} x {args.accum_steps}")
    print(f"  ft max_ep/pat  : {args.max_epochs} / {args.patience}")
    print(f"  out_csv        : {out_csv}")

    factory, cfg = make_encoder_factory(exp_dir, args.epoch)
    n_in = cfg.model.n_in_channels

    splits = build_splits(args.condition,
                          drop_covalent=not args.no_covalent_filter,
                          cl1_only=args.cl1_only, atom_dir=atom_dir,
                          dens_dir=dens_dir, avail_csv=avail_csv)
    if args.max_complexes:
        splits = {s: sdf.head(args.max_complexes) for s, sdf in splits.items()}
        print(f"  [smoke] capped each split to {args.max_complexes} complexes")
    print(f"  split sizes    : train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")

    rows = []

    if "frozen" in args.modes:
        print("\n── frozen control (encode once → MLP head) ─────────────────────")
        enc = factory()
        data = encode_split_features(enc, splits, args.condition, n_in,
                                     atom_dir, dens_dir, device, args.feat_batch_size)
        del enc
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        for seed in range(args.seeds):
            m = train_one(data, seed=seed, device=device,
                          max_epochs=args.frozen_max_epochs,
                          patience=args.frozen_patience,
                          batch_size=args.frozen_batch_size, lr=args.head_lr,
                          weight_decay=args.weight_decay, hidden=args.hidden,
                          dropout=args.dropout)
            rows.append({"condition": args.condition, "mode": "frozen",
                         "ft_scope": "-", "seed": seed,
                         "encoder_lr": 0.0, "head_lr": args.head_lr, **m})
            print(f"  [frozen]   seed={seed} ep_stop={m['epoch_stopped']:3d} "
                  f"val_ρ={m['best_val_spearman']:.4f} "
                  f"test_ρ={m['test_spearman']:.4f} test_r={m['test_pearson']:.4f} "
                  f"rmse={m['test_rmse']:.4f}")

    if "finetune" in args.modes:
        print("\n── finetune (encoder + head end-to-end) ────────────────────────")
        for seed in range(args.seeds):
            m = train_finetune(splits, factory, seed=seed, device=device,
                               condition=args.condition, n_in=n_in,
                               atom_dir=atom_dir, dens_dir=dens_dir,
                               max_epochs=args.max_epochs, patience=args.patience,
                               batch_size=args.batch_size, accum_steps=args.accum_steps,
                               head_lr=args.head_lr, encoder_lr=args.encoder_lr,
                               weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                               hidden=args.hidden, dropout=args.dropout,
                               num_workers=args.num_workers, amp=amp,
                               ft_scope=args.ft_scope, last_k=args.last_k)
            rows.append({"condition": args.condition, "mode": "finetune",
                         "ft_scope": args.ft_scope, "seed": seed,
                         "encoder_lr": args.encoder_lr, "head_lr": args.head_lr, **m})
            print(f"  [finetune] seed={seed} ep_stop={m['epoch_stopped']:3d} "
                  f"val_ρ={m['best_val_spearman']:.4f} "
                  f"test_ρ={m['test_spearman']:.4f} test_r={m['test_pearson']:.4f} "
                  f"rmse={m['test_rmse']:.4f}")

    df = pd.DataFrame(rows)
    df.insert(0, "voxel_version", args.voxel_version)
    df.insert(0, "epoch", args.epoch)
    df.to_csv(out_csv, index=False)

    print("\n── frozen vs finetune (mean ± std across seeds) ────────────────────")
    agg = df.groupby("mode")[["test_spearman", "test_pearson", "test_rmse",
                              "best_val_spearman"]].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    if {"frozen", "finetune"} <= set(df["mode"]):
        fz, ft = df[df["mode"] == "frozen"], df[df["mode"] == "finetune"]
        print(f"\n  Δ finetune − frozen :  "
              f"test_ρ {ft['test_spearman'].mean() - fz['test_spearman'].mean():+.4f}   "
              f"test_r {ft['test_pearson'].mean() - fz['test_pearson'].mean():+.4f}")
    print(f"\n[write] {out_csv}")


# ── CLI ────────────────────────────────────────────────────────────────────────

_CONDITION_CHOICES = ["atomblob", "atomblob_density", "atomblob_weighted",
                      "atomblob_merged_density", "atomblob_merged_density_gradmag",
                      "atomblob_density_gradmag"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDBbind frozen-encoder features + MLP affinity probe (features | probe)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser(
        "features",
        help="Extract frozen-encoder features for PDBbind refined complexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pf.add_argument("--condition", choices=_CONDITION_CHOICES, required=True)
    pf.add_argument("--epoch",      type=int, default=99,
                    help="Checkpoint epoch to use (matched across conditions)")
    pf.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v1",
                    help="Density-normalisation version: v1 = per-map z-score + "
                         "per-crop ±3σ clip; v2 = pocket-pool z-score; "
                         "v3 = pocket-pool symmetric max-abs; v4 = pocket-pool "
                         "clip + z-score. Atom voxels are identical across "
                         "versions (symlinked).")
    pf.add_argument("--batch_size", type=int, default=16)
    pf.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    pf.add_argument("--out_dir",    default=str(FEAT_DIR))
    pf.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N (0 = all). Smoke testing.")
    pf.set_defaults(func=run_features)

    pr = sub.add_parser(
        "probe",
        help="2-layer MLP probe on frozen-encoder pocket features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pr.add_argument("--conditions", nargs="+",
                    default=_CONDITION_CHOICES, choices=_CONDITION_CHOICES)
    pr.add_argument("--epoch",         type=int,   default=99)
    pr.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v1",
                    help="Selects which density-normalisation variant's features "
                         "to probe. Adds matching suffix to feature paths + output CSV.")
    pr.add_argument("--seeds",         type=int,   default=3)
    pr.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    pr.add_argument("--hidden",        type=int,   default=128)
    pr.add_argument("--dropout",       type=float, default=0.1)
    pr.add_argument("--lr",            type=float, default=1e-3)
    pr.add_argument("--weight_decay",  type=float, default=1e-4)
    pr.add_argument("--batch_size",    type=int,   default=64)
    pr.add_argument("--max_epochs",    type=int,   default=200)
    pr.add_argument("--patience",      type=int,   default=30)
    pr.add_argument("--no_intersect",  action="store_true",
                    help="Let each condition use its own pdb_id pool")
    pr.add_argument("--no_covalent_filter", action="store_true",
                    help="Keep covalent complexes (default: drop)")
    pr.add_argument("--cl1_only",      action="store_true",
                    help="Restrict to LP_PDBBind CL1=True (cleanest subset)")
    pr.add_argument("--out_csv",       default=None,
                    help="Override results CSV path")
    pr.set_defaults(func=run_probe)

    pt = sub.add_parser(
        "finetune",
        help="Frozen-encoder control vs end-to-end encoder fine-tuning on PDBbind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pt.add_argument("--condition", choices=_CONDITION_CHOICES,
                    default="atomblob_density_gradmag")
    pt.add_argument("--voxel_version", choices=["v1", "v2", "v3", "v4", "v5"], default="v5",
                    help="Density-normalisation variant; picks the matching encoder "
                         "(EXPS_OVERRIDE) + voxel dir. Default v5 (separate gradmag/ligvdw).")
    pt.add_argument("--epoch",   type=int, default=99, help="Encoder checkpoint epoch")
    pt.add_argument("--modes",   nargs="+", default=["frozen", "finetune"],
                    choices=["frozen", "finetune"],
                    help="Which arms to run. Both → matched frozen-vs-finetune comparison.")
    pt.add_argument("--seeds",   type=int, default=3)
    pt.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    # shared head architecture (identical across arms)
    pt.add_argument("--hidden",  type=int,   default=128)
    pt.add_argument("--dropout", type=float, default=0.1)
    # fine-tune scope + optimisation
    pt.add_argument("--ft_scope", choices=["full", "lastk"], default="full",
                    help="full = whole encoder (low LR); lastk = only last --last_k blocks + norm")
    pt.add_argument("--last_k",   type=int,   default=4, help="blocks to unfreeze when ft_scope=lastk")
    pt.add_argument("--head_lr",     type=float, default=1e-3, help="MLP head LR (both arms)")
    pt.add_argument("--encoder_lr",  type=float, default=1e-5, help="encoder LR (finetune arm)")
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--grad_clip",   type=float, default=1.0, help="0 disables grad clipping")
    pt.add_argument("--batch_size",  type=int,   default=16, help="finetune mini-batch")
    pt.add_argument("--accum_steps", type=int,   default=1,  help="grad-accum (effective bsz = bsz*accum)")
    pt.add_argument("--max_epochs",  type=int,   default=60, help="finetune epoch cap")
    pt.add_argument("--patience",    type=int,   default=15, help="finetune early-stop patience")
    pt.add_argument("--num_workers", type=int,   default=4)
    pt.add_argument("--no_amp",      action="store_true", help="disable autocast (bf16/fp16) mixed precision")
    # frozen-arm head schedule (reproduces the `probe` baseline)
    pt.add_argument("--frozen_max_epochs", type=int, default=200)
    pt.add_argument("--frozen_patience",   type=int, default=30)
    pt.add_argument("--frozen_batch_size", type=int, default=64)
    pt.add_argument("--feat_batch_size",   type=int, default=32, help="frozen-arm encode batch")
    pt.add_argument("--no_covalent_filter", action="store_true",
                    help="Keep covalent complexes (default: drop)")
    pt.add_argument("--cl1_only", action="store_true",
                    help="Restrict to LP_PDBBind CL1=True (cleanest subset)")
    pt.add_argument("--max_complexes", type=int, default=0,
                    help="Head each split to N complexes (0 = all). Smoke testing.")
    pt.add_argument("--out_csv",  default=None, help="Override results CSV path")
    pt.set_defaults(func=run_finetune)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
