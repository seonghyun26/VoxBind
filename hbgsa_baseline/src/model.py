"""model.py — HBGSA: full 4-branch multimodal binding-affinity model.

Branches (each → EMB_DIM):
  • H-bond graph : GCN over the H-bond nodes (dynamic-KNN edges built upstream),
                   2 conv layers + residual, global max-pool         [paper core]
  • protein seq  : physicochemical descriptors → multi-scale dilated-conv
                   residual tower → 1D self-attention → masked mean
  • binding pkt  : pocket-residue descriptors → 1D conv (kernel 3) → masked max
  • SMILES       : token embedding → multi-scale dilated-conv residual tower →
                   1D self-attention → masked mean

Fusion: concat(4·EMB) → FC 4·EMB → 128 → 64 → 1.
Loss  : SmoothL1 + λ·(1 − Pearson)  (paper hybrid loss, λ=50).

The dilated-conv residual towers in the seq/SMILES branches mirror the paper's
"multi-scale dilated convolutions with dilated residual blocks, followed by 1D
self-attention" front-end (closing the parameter gap vs the paper's 3.06M model);
set conv_dilations=() to fall back to the attention-only branch. Reference:
HBGSA, arXiv 2604.23115.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

from config import (CONV_CHANNELS, CONV_DILATIONS, CONVS_PER_BLOCK, EMB_DIM,
                    GCN_HIDDEN, PEARSON_LAMBDA)
from featurize import SEQ_FEAT_DIM


class HBondGCN(nn.Module):
    """2-layer GCN on the H-bond graph → EMB_DIM (global max-pool)."""

    def __init__(self, in_dim: int = 9, hidden: int = GCN_HIDDEN, out_dim: int = EMB_DIM):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = F.gelu(self.lin_in(x))
        h = self.norm1(F.gelu(self.conv1(h, edge_index)))
        h = self.norm2(h + F.gelu(self.conv2(h, edge_index)))   # residual
        h = global_max_pool(h, batch)                           # (B, hidden)
        return self.proj(h)


class DilatedConvBlock(nn.Module):
    """Residual block of `n_convs` dilated 1D convs (per-position LayerNorm + GELU).

    Runs in (B, L, C) layout; convs transpose to (B, C, L) internally. The residual
    is added around the whole block. Per-position LayerNorm keeps stats independent
    of padded positions (which the attention mask drops downstream)."""

    def __init__(self, channels: int, dilation: int, n_convs: int = 2):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            for _ in range(n_convs)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(n_convs))

    def forward(self, h):                                       # h: (B, L, C)
        x = h
        for conv, norm in zip(self.convs, self.norms):
            y = conv(x.transpose(1, 2)).transpose(1, 2)         # (B, L, C)
            x = norm(F.gelu(y))
        return h + x                                            # residual around block


class DilatedConvTower(nn.Module):
    """Multi-scale dilated-conv residual tower (paper v_seq/v_smi front-end).

    On (B, L, d_model): per-position channel adapter → one residual block per
    dilation (increasing dilation = multi-scale receptive field) → project back to
    d_model. Padded positions are zeroed on output so the trailing attention/pool
    never sees conv leakage into the pad region."""

    def __init__(self, d_model: int, conv_channels: int = CONV_CHANNELS,
                 dilations=CONV_DILATIONS, convs_per_block: int = CONVS_PER_BLOCK):
        super().__init__()
        self.stem = nn.Linear(d_model, conv_channels)
        self.blocks = nn.ModuleList(
            DilatedConvBlock(conv_channels, d, convs_per_block) for d in dilations
        )
        self.head = nn.Linear(conv_channels, d_model)

    def forward(self, h, key_padding_mask):                     # h: (B, L, d_model)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.head(h)
        return h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class SelfAttnPool(nn.Module):
    """[embed/proj] → dilated-conv residual tower → 1D self-attention → masked mean.

    Matches the paper's v_seq / v_smi branch: multi-scale dilated convolutions with
    dilated residual blocks, followed by 1D self-attention. Pass conv_dilations=()
    to disable the tower (attention-only, the previous behaviour)."""

    def __init__(self, in_dim: int, d_model: int, out_dim: int = EMB_DIM,
                 n_heads: int = 4, n_layers: int = 2, embedding: bool = False,
                 conv_channels: int = CONV_CHANNELS, conv_dilations=CONV_DILATIONS,
                 convs_per_block: int = CONVS_PER_BLOCK):
        super().__init__()
        self.embedding = nn.Embedding(in_dim, d_model, padding_idx=0) if embedding \
            else None
        self.proj_in = None if embedding else nn.Linear(in_dim, d_model)
        self.tower = DilatedConvTower(d_model, conv_channels, conv_dilations,
                                      convs_per_block) if conv_dilations else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, x, key_padding_mask):
        h = self.embedding(x) if self.embedding is not None else self.proj_in(x)
        if self.tower is not None:
            h = self.tower(h, key_padding_mask)                      # dilated-conv front-end
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)   # True = pad
        valid = (~key_padding_mask).unsqueeze(-1).float()            # (B,L,1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)    # masked mean
        return self.proj_out(pooled)


class PocketConv(nn.Module):
    """Binding-pocket branch (paper v_pkt): 1D conv (kernel 3) → masked max-pool → out_dim."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = EMB_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x, key_padding_mask):                     # x (B,L,F), mask True=pad
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)        # (B,L,hidden)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
        pooled = h.max(dim=1).values                            # masked max-pool
        pooled = torch.nan_to_num(pooled, neginf=0.0)           # all-pad → 0
        return self.proj(pooled)


class HBGSA(nn.Module):
    def __init__(self, smiles_vocab_size: int, seq_d_model: int = 128,
                 smi_d_model: int = 128, emb_dim: int = EMB_DIM,
                 n_layers: int = 2, n_heads: int = 4,
                 gcn_hidden: int = GCN_HIDDEN, pocket_hidden: int = 128,
                 head_hidden: int = 128, conv_channels: int = CONV_CHANNELS,
                 conv_dilations=CONV_DILATIONS, convs_per_block: int = CONVS_PER_BLOCK):
        super().__init__()
        self.hb = HBondGCN(in_dim=9, hidden=gcn_hidden, out_dim=emb_dim)
        self.seq = SelfAttnPool(SEQ_FEAT_DIM, seq_d_model, emb_dim,
                                n_heads=n_heads, n_layers=n_layers, embedding=False,
                                conv_channels=conv_channels, conv_dilations=conv_dilations,
                                convs_per_block=convs_per_block)
        self.pkt = PocketConv(SEQ_FEAT_DIM, hidden=pocket_hidden, out_dim=emb_dim)
        self.smi = SelfAttnPool(smiles_vocab_size, smi_d_model, emb_dim,
                                n_heads=n_heads, n_layers=n_layers, embedding=True,
                                conv_channels=conv_channels, conv_dilations=conv_dilations,
                                convs_per_block=convs_per_block)
        self.head = nn.Sequential(
            nn.Linear(4 * emb_dim, head_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_hidden, head_hidden // 2), nn.GELU(),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, batch):
        h_hb = self.hb(batch["hb_x"], batch["hb_edge_index"], batch["hb_batch"])
        h_seq = self.seq(batch["seq"], batch["seq_mask"])
        h_pkt = self.pkt(batch["pkt"], batch["pkt_mask"])
        h_smi = self.smi(batch["smi"], batch["smi_mask"])
        z = torch.cat([h_hb, h_seq, h_pkt, h_smi], dim=-1)
        return self.head(z).squeeze(-1)


def hybrid_loss(pred, target, lam: float = PEARSON_LAMBDA):
    """SmoothL1 + λ·(1 − Pearson r) over the batch."""
    reg = F.smooth_l1_loss(pred, target)
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm() + 1e-8
    r = (p * t).sum() / denom
    return reg + lam * (1.0 - r), reg.detach(), r.detach()
