"""01f_pdbbind_probe.py — Phase 6: 2-layer MLP probe on frozen-encoder features.

Compares two pretrained-encoder representations on PDBbind affinity:
  (i)  atomblob          — 11-ch input  (7 lig + 4 pocket atoms)
  (ii) atomblob_density  — 12-ch input  (11 atoms + 1 2Fo-Fc density)

For each condition, the 512-D mean-pooled patch tokens (from 01e) are fed
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

Usage
-----
    cd voxbind
    CUDA_VISIBLE_DEVICES=5 python dataset/01f_pdbbind_probe.py \
        --epoch 99 --seeds 3

Optional flags
--------------
    --conditions COND ...   default: atomblob atomblob_density
    --epoch N               default: 99
    --seeds N               default: 3
    --no_intersect          let each condition use its own pdb_id pool (NOT
                            recommended; breaks apples-to-apples)
    --no_covalent_filter    keep covalent complexes (default: drop)
    --cl1_only              restrict to LP_PDBBind CL1=True (cleanest subset)
    --max_epochs N          default: 200
    --patience N            default: 30  (epochs of no val-ρ improvement)

Outputs
-------
    dataset/data/pdbbind/probe_results_e<N>.csv
        cols: condition, seed, n_train, n_val, n_test,
              best_val_spearman, test_spearman, test_pearson, test_rmse,
              epoch_stopped
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Paths ──────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
INDEX_CSV   = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"
FEAT_DIR    = PDBBIND_DIR / "features"
RESULTS_DIR = PDBBIND_DIR


# ── Data assembly ──────────────────────────────────────────────────────────────

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


# ── MLP probe head ─────────────────────────────────────────────────────────────

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
        pred_te = model(Xte).cpu().numpy()
    yte_np = yte.cpu().numpy()

    return {
        "n_train": int(n_train),
        "n_val":   int(Xva.shape[0]),
        "n_test":  int(Xte.shape[0]),
        "best_val_spearman": float(best_val),
        "test_spearman":     float(spearmanr(pred_te, yte_np).statistic),
        "test_pearson":      float(pearsonr (pred_te, yte_np).statistic),
        "test_rmse":         float(np.sqrt(((pred_te - yte_np) ** 2).mean())),
        "epoch_stopped":     int(best_epoch),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-layer MLP probe on frozen-encoder pocket features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--conditions", nargs="+",
                   default=["atomblob", "atomblob_density", "atomblob_weighted",
                            "atomblob_merged_density"],
                   choices=["atomblob", "atomblob_density", "atomblob_weighted",
                            "atomblob_merged_density"])
    p.add_argument("--epoch",         type=int,   default=99)
    p.add_argument("--voxel_version", choices=["v1", "v2", "v3"], default="v1",
                   help="Selects which density-normalisation variant's features "
                        "to probe. Adds matching suffix to feature paths + output CSV.")
    p.add_argument("--seeds",         type=int,   default=3)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--max_epochs",    type=int,   default=200)
    p.add_argument("--patience",      type=int,   default=30)
    p.add_argument("--no_intersect",  action="store_true",
                   help="Let each condition use its own pdb_id pool")
    p.add_argument("--no_covalent_filter", action="store_true",
                   help="Keep covalent complexes (default: drop)")
    p.add_argument("--cl1_only",      action="store_true",
                   help="Restrict to LP_PDBBind CL1=True (cleanest subset)")
    p.add_argument("--out_csv",       default=None,
                   help="Override results CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    lp_df = load_lp_index(INDEX_CSV)
    print(f"  LP rows       : {len(lp_df):,}")

    # ── Load all feature bundles upfront so we can intersect pdb_ids ─────────
    all_feats: dict[str, dict[str, torch.Tensor]] = {}
    for cond in args.conditions:
        feat_path = FEAT_DIR / f"{cond}_e{args.epoch}{suffix}.pt"
        if not feat_path.exists():
            print(f"\n[error] missing features: {feat_path}")
            print(f"        Run: python dataset/01e_pdbbind_features.py "
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
                  f"test_ρ={m['test_spearman']:.4f}  "
                  f"test_r={m['test_pearson']:.4f}  "
                  f"test_rmse={m['test_rmse']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n── Summary (mean ± std across seeds) ───────────────────────────────")
    agg = df.groupby("condition")[
        ["test_spearman", "test_pearson", "test_rmse", "best_val_spearman"]
    ].agg(["mean", "std"]).round(4)
    print(agg.to_string())
    print(f"\n[write] {out_csv}")


if __name__ == "__main__":
    main()
