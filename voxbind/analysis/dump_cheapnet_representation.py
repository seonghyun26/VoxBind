#!/usr/bin/env python3
"""Dump CheapNet's fixed-width pre-regression representation on CPU.

The checkpoint was trained on lp_edrscc_v2 train/valid for the repository's
CASF evaluation.  The exported vector is ``l2p + p2l`` immediately before the
``model.fc`` regression head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
CHEAPNET_ROOT = REPO / "base/cheapnet"
DEFAULT_DATA = CHEAPNET_ROOT / "_edrscc/data_lp_edrscc_v2"
DEFAULT_CHECKPOINT = CHEAPNET_ROOT / "_edrscc/ckpts_casf/cheapnet_casf_seed0.pt"
DEFAULT_OUTPUT = CHEAPNET_ROOT / "_edrscc/features/cheapnet_casf_seed0_prehead.pt"

sys.path.insert(0, str(CHEAPNET_ROOT / "cross_dataset"))

from CheapNet import CheapNet  # noqa: E402
from dataset_CheapNet import GraphDataset, PLIDataLoader  # noqa: E402


class TrustedGraphDataset(GraphDataset):
    """Load repository-generated PyG graphs under PyTorch 2.6's safe default."""

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx], map_location="cpu", weights_only=False)


def extract_prehead(model: CheapNet, data) -> torch.Tensor:
    x = model.embedding(data.x)
    model.make_edge_index(data)
    x = model.GIGNBlock1(x, data)
    x = model.GIGNBlock2(x, data)
    x = model.GIGNBlock3(x, data)
    ligand, _ = model.diffpool1(x, data)
    protein, _ = model.diffpool2(x, data)
    ligand_to_protein, _ = model.attblock1(ligand, protein, protein)
    protein_to_ligand, _ = model.attblock2(protein, ligand, ligand)
    return ligand_to_protein + protein_to_ligand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-pids", type=int, default=0, help="smoke-test cap; 0 uses all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be >= 1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    meta = json.loads((data_root / "meta.json").read_text())
    model = CheapNet(
        node_dim=35,
        hidden_dim=256,
        num_clusters=meta["num_clusters_train_median"],
        heads=1,
        drop_rate=0.1,
    )
    max_nodes = max(600, int(meta.get("max_nodes_total", 0)))
    model.diffpool1.max_num = max_nodes
    model.diffpool2.max_num = max_nodes
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    features: dict[str, torch.Tensor] = {}
    predictions: dict[str, float] = {}
    remaining = args.max_pids if args.max_pids > 0 else None
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(data_root / f"{split}.csv")
        if remaining is not None:
            frame = frame.iloc[:remaining]
        dataset = TrustedGraphDataset(
            str(data_root / split),
            frame,
            graph_type=meta["graph_type"],
            dis_threshold=meta["distance"],
            create=False,
        )
        loader = PLIDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                hidden = extract_prehead(model, batch)
                prediction = model.fc(hidden).view(-1)
                batch_ids = dataset.complex_ids[offset : offset + len(hidden)]
                for pid, vector, value in zip(batch_ids, hidden, prediction):
                    key = str(pid).lower()
                    if key in features:
                        raise ValueError(f"duplicate pid across split files: {key}")
                    features[key] = vector.detach().cpu().to(torch.float32)
                    predictions[key] = float(value)
                offset += len(hidden)
        print(f"[{split}] dumped {offset} vectors", flush=True)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "predictions": predictions,
            "metadata": {
                "model": "CheapNet",
                "checkpoint": str(checkpoint),
                "representation": "l2p + p2l immediately before model.fc",
                "dim": 256,
                "n": len(features),
                "cpu_only": True,
                "split": "lp_edrscc_v2",
            },
        },
        output,
    )
    print(f"[done] {len(features)} x 256 -> {output}")


if __name__ == "__main__":
    main()
