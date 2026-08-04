"""run_casf_inference.py — run inference on CASF-2016 test.pkl for a given checkpoint,
then emit per-seed prediction jsonl.

Usage:
  python run_casf_inference.py --ckpt <path> --seed_tag <label> --out <jsonl_path> [--gpu 4]
"""
import os
import sys
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
GET_ROOT = os.path.dirname(HERE)
sys.path.insert(0, GET_ROOT)

import models
from train import create_dataset
from data.pdb_utils import VOCAB


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_pkl", default=os.path.join(HERE, "datasets", "casf2016", "test.pkl"))
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=4)
    return ap.parse_args()


def main(args):
    VOCAB.load_tokenizer(None)
    ckpt_dir = os.path.dirname(args.ckpt)
    # resolve relative ckpt path (topk_map uses ./_edrscc/... relative to base/get/)
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(GET_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        # try relative to GET_ROOT
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        weights = model
        ns_path = os.path.join(os.path.dirname(ckpt_path), "namespace.json")
        namespace = json.load(open(ns_path))
        model = models.create_model(argparse.Namespace(**namespace))
        model.load_state_dict(weights)

    device = torch.device("cpu" if args.gpu == -1 else f"cuda:{args.gpu}")
    model.to(device)
    model.eval()

    test_set = create_dataset("PDBBind", args.test_pkl, fragment=None)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=test_set.collate_fn)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results = []
    idx = 0
    for batch in tqdm(test_loader, desc=f"inference gpu={args.gpu}"):
        with torch.no_grad():
            for k in batch:
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)
            gt_labels = batch.pop("label")
            preds = model.infer(batch)
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = preds.tolist() if hasattr(preds, "tolist") else list(preds)
            for pred in preds:
                item = test_set.indexes[idx]
                pid = item["id"]
                gt = float(item["affinity"]["neglog_aff"]) if "affinity" in item else float(item.get("label", 0))
                results.append({"id": pid, "pred": float(pred), "gt": gt})
                idx += 1

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main(parse())
