"""infer_casf.py — run ProFSA inference on CASF-2016 LMDB using a trained head checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=5 python _casf/infer_casf.py \\
        --run_dir _edrscc/runs/lba30 \\
        --ckpt best \\
        --out preds_seed0.csv

Outputs a CSV with columns: pid, pred, label, in_v2train.
Metrics (leaky=all 214, nontrain=124) are printed to stdout.
"""
import argparse
import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from torch.utils.data import DataLoader

logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

# Resolve paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROFSA_ROOT = os.path.dirname(HERE)

# Add profsa to sys.path so `src` is importable
sys.path.insert(0, PROFSA_ROOT)

from src.utils.exptool import register_omegaconf_resolver, Experiment
from src.dataset.profsa import ProFSADataset

register_omegaconf_resolver()


def get_ckpt_path(run_dir: Path, ckpt: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if ckpt == "best":
        ckpts = sorted([p for p in ckpt_dir.glob("*.ckpt") if "last" not in p.name])
        assert ckpts, f"No best checkpoint in {ckpt_dir}"
        return ckpts[-1]
    elif ckpt == "last":
        return ckpt_dir / "last.ckpt"
    else:
        p = ckpt_dir / ckpt
        if not p.suffix:
            p = p.with_suffix(".ckpt")
        return p


def build_casf_dataloader(casf_data_dir: str, batch_size: int = 32) -> DataLoader:
    dataset_cfg = dict(
        data_dir=casf_data_dir,
        data_file="casf2016.lmdb",
        mol_dict_file="dict_mol.txt",
        pocket_dict_file="dict_pkt.txt",
        max_pocket_atoms=256,
        max_seq_len=512,
        shuffle=False,
        seed=0,
        ligand_atoms_key="atoms",
        ligand_coord_key="coordinates",
        pocket_atoms_key="pocket_atoms",
        pocket_coord_key="pocket_coordinates",
        affinity_key="label",
    )
    dataset = ProFSADataset(**dataset_cfg)
    loader = DataLoader(
        dataset,
        collate_fn=dataset.dataset.collater,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    return loader


@torch.no_grad()
def run_inference(pipeline, dataloader, device):
    """Run forward pass and collect (preds, targets, pocket_names)."""
    pipeline.eval()
    pipeline.to(device)

    all_preds = []
    all_targets = []
    all_pockets = []

    for batch in dataloader:
        # Move tensors to device
        net_input = batch["net_input"]
        for k, v in net_input.items():
            if isinstance(v, torch.Tensor):
                net_input[k] = v.to(device)

        outputs = pipeline.model(**net_input)
        logits = outputs["logit"].cpu().float()
        targets = batch["target"]["finetune_target"].float()
        pockets = batch["pocket_name"]

        all_preds.append(logits.numpy())
        all_targets.append(targets.numpy())
        all_pockets.extend(pockets)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return preds, targets, all_pockets


def compute_metrics(preds, targets):
    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return {"pearson": float(r), "spearman": float(rho), "rmse": rmse, "n": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to the training run dir (has config.yaml + checkpoints/)")
    parser.add_argument("--ckpt", default="best", help="'best', 'last', or checkpoint filename stem")
    parser.add_argument("--out", required=True, help="Output CSV path for per-sample predictions")
    parser.add_argument("--casf_data_dir", default=None, help="Override CASF LMDB dir")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    casf_data_dir = args.casf_data_dir or os.path.join(PROFSA_ROOT, "data", "dataset", "casf2016")

    # Load order CSV for in_v2train labels
    order_csv = os.path.join(casf_data_dir, "casf2016_order.csv")
    order_df = pd.read_csv(order_csv)
    pid2meta = {row["pid"]: {"in_v2train": int(row["in_v2train"]), "pK": float(row["pK"])}
                for _, row in order_df.iterrows()}

    # Load experiment and pipeline
    exp = Experiment(run_dir)
    config = exp.config

    ckpt_path = get_ckpt_path(run_dir, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load pipeline
    import tempfile
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    pipeline_cfg = dict(config["pipeline"])
    pipeline_cfg["_target_"] = f"{pipeline_cfg['_target_']}.load_from_checkpoint"
    pipeline_cfg["checkpoint_path"] = ckpt_path
    pipeline_cfg["strict"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))
        f.flush()
        pipeline_cfg["hparams_file"] = f.name
        pipeline = instantiate(pipeline_cfg)

    # Build dataloader
    loader = build_casf_dataloader(casf_data_dir, args.batch_size)

    # Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device} ...")
    preds, targets, pockets = run_inference(pipeline, loader, device)

    print(f"Got {len(preds)} predictions")

    # Build result DataFrame
    rows = []
    for pid, pred, label in zip(pockets, preds, targets):
        meta = pid2meta.get(pid, {"in_v2train": -1, "pK": label})
        rows.append({
            "pid": pid,
            "pred": float(pred),
            "label": float(label),
            "in_v2train": meta["in_v2train"],
        })
    df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")

    # Compute metrics
    leaky_metrics = compute_metrics(df["pred"].values, df["label"].values)
    nontrain_df = df[df["in_v2train"] == 0]
    nontrain_metrics = compute_metrics(nontrain_df["pred"].values, nontrain_df["label"].values)

    print(f"\nLeaky (n={leaky_metrics['n']}):")
    print(f"  Pearson r = {leaky_metrics['pearson']:.4f}")
    print(f"  Spearman ρ = {leaky_metrics['spearman']:.4f}")
    print(f"  RMSE = {leaky_metrics['rmse']:.4f}")
    print(f"\nNon-train (n={nontrain_metrics['n']}):")
    print(f"  Pearson r = {nontrain_metrics['pearson']:.4f}")
    print(f"  Spearman ρ = {nontrain_metrics['spearman']:.4f}")
    print(f"  RMSE = {nontrain_metrics['rmse']:.4f}")

    return leaky_metrics, nontrain_metrics


if __name__ == "__main__":
    main()
