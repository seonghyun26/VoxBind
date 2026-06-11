"""dataset.py — torch Dataset + collate for HBGSA.

Each sample bundles: the cached H-bond graph (with dynamic-KNN edges built here),
the protein-sequence descriptor matrix, and the SMILES token ids. Collate packs
graphs into a single PyG-style (x, edge_index, batch) and pads seq/SMILES with
boolean key-padding masks.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import HB_KNN_K, POCKET_MAX_LEN, SEQ_MAX_LEN, SMILES_MAX_LEN
from featurize import encode_sequence, encode_smiles
from hbonds import cache_path
from pocket import load_pocket_cache


def _knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    """Undirected KNN edge_index (2,E) over node positions; self-loops added by GCNConv."""
    n = pos.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    k_eff = min(k, n - 1)
    nbr = np.argpartition(d, k_eff - 1, axis=1)[:, :k_eff]      # (n, k_eff)
    src = np.repeat(np.arange(n), k_eff)
    dst = nbr.reshape(-1)
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])  # undirected
    return ei.astype(np.int64)


class HBGSADataset(Dataset):
    def __init__(self, df, vocab: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.pocket = load_pocket_cache()                  # {pdb_id: pocket residue seq}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(cache_path(r.pdb_id))
        nf = z["node_feats"].astype(np.float32)            # (N, 9)
        if nf.shape[0] == 0:                               # no H-bond → 1 zero node
            nf = np.zeros((1, 9), np.float32)
        edge_index = _knn_edges(nf[:, 6:9], HB_KNN_K)      # KNN over midpoints
        pkt_seq = self.pocket.get(r.pdb_id, "")
        return {
            "hb_x": torch.from_numpy(nf),
            "hb_edge_index": torch.from_numpy(edge_index),
            "seq": torch.from_numpy(encode_sequence(r.seq, SEQ_MAX_LEN)),
            "pkt": torch.from_numpy(encode_sequence(pkt_seq, POCKET_MAX_LEN)),
            "smi": torch.from_numpy(encode_smiles(r.smiles, self.vocab, SMILES_MAX_LEN)),
            "y": torch.tensor(float(r.pK), dtype=torch.float32),
            "pdb_id": r.pdb_id,
        }


def collate(samples: list[dict]) -> dict:
    # ── H-bond graphs → single batched graph ──
    xs, eis, batch = [], [], []
    offset = 0
    for i, s in enumerate(samples):
        n = s["hb_x"].shape[0]
        xs.append(s["hb_x"])
        eis.append(s["hb_edge_index"] + offset)
        batch.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    hb_x = torch.cat(xs, 0)
    hb_edge_index = torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long)
    hb_batch = torch.cat(batch, 0)

    # ── seq / pocket (B, Lmax, F) + pad mask ──
    F = samples[0]["seq"].shape[1]

    def pad_feat(key):
        L = max(s[key].shape[0] for s in samples)
        x = torch.zeros(len(samples), L, F)
        mask = torch.ones(len(samples), L, dtype=torch.bool)        # True = pad
        for i, s in enumerate(samples):
            li = s[key].shape[0]
            x[i, :li] = s[key]
            mask[i, :li] = False
        return x, mask

    seq, seq_mask = pad_feat("seq")
    pkt, pkt_mask = pad_feat("pkt")

    # ── SMILES (B, Tmax) + pad mask ──
    T = max(s["smi"].shape[0] for s in samples)
    smi = torch.zeros(len(samples), T, dtype=torch.long)            # 0 = PAD
    smi_mask = torch.ones(len(samples), T, dtype=torch.bool)
    for i, s in enumerate(samples):
        ti = s["smi"].shape[0]
        smi[i, :ti] = s["smi"]
        smi_mask[i, :ti] = False

    return {
        "hb_x": hb_x, "hb_edge_index": hb_edge_index, "hb_batch": hb_batch,
        "seq": seq, "seq_mask": seq_mask,
        "pkt": pkt, "pkt_mask": pkt_mask,
        "smi": smi, "smi_mask": smi_mask,
        "y": torch.stack([s["y"] for s in samples]),
        "pdb_id": [s["pdb_id"] for s in samples],
    }
