#!/usr/bin/env python3
"""Dump GET's 64-d graph_repr on the CASF-2016 complexes (for ensemble on CASF cohorts).
Run with the `get` conda env. CPU-only. Mirrors dump_get_representation.py but the single
CASF pickle."""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/shpark/prj-denovo/VoxBind")
GET_ROOT = REPO / "base/get"
sys.path.insert(0, str(GET_ROOT))
from data.dataset import PDBBindBenchmark          # noqa: E402
from data.pdb_utils import VOCAB                    # noqa: E402
from models.prediction_model import PredictionModel # noqa: E402

torch.set_num_threads(4); torch.manual_seed(0)
VOCAB.load_tokenizer(None)
ckpt = (GET_ROOT / "_edrscc/models/GET_v2/version_0/checkpoint/epoch19_step3480.ckpt").resolve()
model = torch.load(ckpt, map_location="cpu"); model.eval()

casf_pkl = GET_ROOT / "_casf_get/datasets/casf2016/test.pkl"
dataset = PDBBindBenchmark(str(casf_pkl))
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
features = {}; offset = 0
with torch.no_grad():
    for batch in loader:
        batch.pop("label")
        result = PredictionModel.forward(
            model, Z=batch["X"], B=batch["B"], A=batch["A"],
            atom_positions=batch["atom_positions"], block_lengths=batch["block_lengths"],
            lengths=batch["lengths"], segment_ids=batch["segment_ids"],
            label=None, return_noise=False)
        batch_ids = dataset.indexes[offset:offset + len(result.graph_repr)]
        for item, vector in zip(batch_ids, result.graph_repr):
            features[str(item["id"]).lower()] = vector.detach().cpu().to(torch.float32)
        offset += len(result.graph_repr)
out = GET_ROOT / "_casf_get/features/get_v2_seed0_casf_repr.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"features": features, "metadata": {"model": "GET", "dim": 64, "split": "casf2016", "n": len(features)}}, out)
print(f"[done] {len(features)} x 64 -> {out}")
print("sample pids:", list(features)[:5])
