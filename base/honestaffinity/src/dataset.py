"""dataset.py — torch Dataset + collate for HonestAffinity-Pocket.

Loads records_{split}.json (from data_prep.py) and the cached ESM float16 embeddings,
tokenizes SMILES, and produces padded batches:
    esm            [B,Lp,1280] float
    prot_mask_pad  [B,Lp]      bool (True==pad)
    pocket_marker  [B,Lp]      long {0,1}
    smi_tokens     [B,Ll]      long
    smi_mask_pad   [B,Ll]      bool
    y              [B]         float (pK)
"""
import os
import json

import torch
from torch.utils.data import Dataset

from data_prep import esm_path
from smiles_tokenizer import encode, PAD_ID


class AffinityDataset(Dataset):
    def __init__(self, records, smi_max_len=200):
        self.recs = records
        self.smi_max_len = smi_max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        esm = torch.load(esm_path(r["seq"]), map_location="cpu").float()  # [L,1280]
        mask = torch.tensor(r["pocket_mask"], dtype=torch.long)
        # guard: sequence and cached embedding length must match
        L = min(esm.shape[0], mask.shape[0])
        esm = esm[:L]
        mask = mask[:L]
        smi = torch.tensor(encode(r["smiles"], self.smi_max_len), dtype=torch.long)
        if smi.numel() == 0:
            smi = torch.tensor([PAD_ID], dtype=torch.long)
        y = torch.tensor(r["pK"], dtype=torch.float)
        return esm, mask, smi, y


def collate(batch):
    esms, masks, smis, ys = zip(*batch)
    B = len(batch)
    Lp = max(e.shape[0] for e in esms)
    Ll = max(s.shape[0] for s in smis)
    d = esms[0].shape[1]

    esm = torch.zeros(B, Lp, d)
    prot_pad = torch.ones(B, Lp, dtype=torch.bool)
    pmark = torch.zeros(B, Lp, dtype=torch.long)
    smi = torch.full((B, Ll), PAD_ID, dtype=torch.long)
    smi_pad = torch.ones(B, Ll, dtype=torch.bool)

    for i, (e, m, s) in enumerate(zip(esms, masks, smis)):
        lp = e.shape[0]
        esm[i, :lp] = e
        prot_pad[i, :lp] = False
        pmark[i, :lp] = m
        ll = s.shape[0]
        smi[i, :ll] = s
        smi_pad[i, :ll] = False

    y = torch.stack(ys)
    return esm, prot_pad, pmark, smi, smi_pad, y


def load_records(split, cache_dir):
    path = os.path.join(cache_dir, f"records_{split}.json")
    with open(path) as f:
        d = json.load(f)
    recs = d["records"]
    buckets = {"train": [], "val": [], "test": []}
    for r in recs:
        buckets.get(r["split"], []).append(r)
    return buckets
