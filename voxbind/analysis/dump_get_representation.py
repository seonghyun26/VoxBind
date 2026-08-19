#!/usr/bin/env python3
"""Dump GET's native 64-dimensional graph representation on CPU.

Run this script with the repository's ``get`` environment because the model
depends on torch-scatter.  The selected checkpoint is the best validation
checkpoint from the seed-0 lp_edrscc_v2 run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
GET_ROOT = REPO / "base/get"
DEFAULT_DATA = GET_ROOT / "datasets/edrscc"
DEFAULT_CHECKPOINT = GET_ROOT / (
    "_edrscc/models/GET_v2/version_0/checkpoint/epoch19_step3480.ckpt"
)
DEFAULT_OUTPUT = GET_ROOT / "_edrscc/features/get_v2_seed0_graph_repr.pt"

sys.path.insert(0, str(GET_ROOT))

from data.dataset import PDBBindBenchmark  # noqa: E402
from data.pdb_utils import VOCAB  # noqa: E402
from models.prediction_model import PredictionModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-pids", type=int, default=0, help="smoke-test cap; 0 uses all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be >= 1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    VOCAB.load_tokenizer(None)

    checkpoint = args.checkpoint.resolve()
    model = torch.load(checkpoint, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{checkpoint}: expected a serialized model object")
    model.eval()

    features: dict[str, torch.Tensor] = {}
    predictions: dict[str, float] = {}
    remaining = args.max_pids if args.max_pids > 0 else None
    data_root = args.data_root.resolve()
    for split in ("train", "valid", "test"):
        dataset = PDBBindBenchmark(str(data_root / f"{split}.pkl"))
        if remaining is not None and len(dataset) > remaining:
            dataset.data = dataset.data[:remaining]
            dataset.indexes = dataset.indexes[:remaining]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch.pop("label")
                result = PredictionModel.forward(
                    model,
                    Z=batch["X"],
                    B=batch["B"],
                    A=batch["A"],
                    atom_positions=batch["atom_positions"],
                    block_lengths=batch["block_lengths"],
                    lengths=batch["lengths"],
                    segment_ids=batch["segment_ids"],
                    label=None,
                    return_noise=False,
                )
                batch_ids = dataset.indexes[offset : offset + len(result.graph_repr)]
                for item, vector, energy in zip(batch_ids, result.graph_repr, result.energy):
                    pid = str(item["id"]).lower()
                    if pid in features:
                        raise ValueError(f"duplicate pid across split files: {pid}")
                    features[pid] = vector.detach().cpu().to(torch.float32)
                    predictions[pid] = float(-energy)
                offset += len(result.graph_repr)
        print(f"[{split}] dumped {offset} vectors", flush=True)
        if remaining is not None:
            remaining -= len(dataset)
            if remaining <= 0:
                break

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "predictions": predictions,
            "metadata": {
                "model": "GET",
                "checkpoint": str(checkpoint),
                "representation": "encoder graph_repr (normalized sum of block_repr)",
                "dim": 64,
                "n": len(features),
                "cpu_only": True,
                "split": "lp_edrscc_v2",
            },
        },
        output,
    )
    print(f"[done] {len(features)} x 64 -> {output}")


if __name__ == "__main__":
    main()
