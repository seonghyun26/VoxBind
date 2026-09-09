#!/usr/bin/env python3
"""Dump EGNN's 64-d graph_repr on edrscc train/valid/test + CASF into one file.
Run with the `get` env (EGNN lives in the GET repo). CPU-only. Handles state_dict
checkpoints via namespace.json + models.create_model."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/shpark/prj-denovo/VoxBind"); GET_ROOT = REPO / "base/get"
sys.path.insert(0, str(GET_ROOT))
import models                                            # noqa: E402
from data.dataset import PDBBindBenchmark                # noqa: E402
from data.pdb_utils import VOCAB                          # noqa: E402
from models.prediction_model import PredictionModel       # noqa: E402


def load_model(ckpt: Path):
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    ns = json.load(open(ckpt.parent / "namespace.json"))
    m = models.create_model(argparse.Namespace(**ns))
    m.load_state_dict(obj)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    VOCAB.load_tokenizer(None)
    model = load_model(Path(a.checkpoint).resolve()); model.eval()
    sources = [(GET_ROOT / "datasets/edrscc", ["train", "valid", "test"]),
               (GET_ROOT / "_casf_get/datasets/casf2016", ["test"])]
    feats = {}
    for root, splits in sources:
        for split in splits:
            pkl = root / f"{split}.pkl"
            if not pkl.exists():
                print(f"  skip missing {pkl}"); continue
            ds = PDBBindBenchmark(str(pkl))
            loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    batch.pop("label")
                    res = PredictionModel.forward(
                        model, Z=batch["X"], B=batch["B"], A=batch["A"],
                        atom_positions=batch["atom_positions"], block_lengths=batch["block_lengths"],
                        lengths=batch["lengths"], segment_ids=batch["segment_ids"],
                        label=None, return_noise=False)
                    ids = ds.indexes[offset:offset + len(res.graph_repr)]
                    for item, vec in zip(ids, res.graph_repr):
                        feats[str(item["id"]).lower()] = vec.detach().cpu().to(torch.float32)
                    offset += len(res.graph_repr)
            print(f"[{root.name}/{split}] {offset}", flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "metadata": {"model": "EGNN", "dim": 64, "n": len(feats)}}, a.output)
    print(f"[done] {len(feats)} x 64 -> {a.output}")


if __name__ == "__main__":
    main()
