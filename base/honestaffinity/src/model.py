"""model.py — HonestAffinity-Pocket (PLM-augmented) reimplementation.

Faithful to the SPEC (arXiv 2606.03422, HonestAffinity: Leak-Aware Evaluation of Protein
and Pocket Priors), headline HonestAffinity-Pocket variant. No official code available;
implemented from the paper spec.

Protein branch
  frozen ESM-2-650M per-residue [Lp,1280]  (precomputed & cached; fed in as input)
  -> Linear(1280->256)
  -> + LEARNED pocket-position marker  nn.Embedding(2,256)[marker]   (added to reps)
  -> MultiScaleConv1d(kernels {1,3,5,7}, out ch {32,32,64,128} = 256) + proj + residual
  -> 1x nn.TransformerEncoderLayer(d=256, nhead=8, ff=1024, batch_first)
  => Zprot [Lp,256]

Ligand branch
  SMILES -> 53-token learned vocab -> Embedding(53,256)
  -> MultiScaleConv1d + residual
  -> 1x TransformerEncoderLayer
  => Zsmi [Ll,256]

Compatibility + head
  S = (Zprot @ Zsmi^T) / sqrt(d)              [Lp,Ll]
  cross-attention pooling (documented choice):
    prot_ctx  = softmax(S, dim=lig)  @ Zsmi   -> per-prot-residue context, mean over prot
    smi_ctx   = softmax(S, dim=prot) @ Zprot  -> per-lig-token context,   mean over lig
  pooled = [ prot_mean, prot_attnpooled(=mean of prot_ctx),
             smi_mean,  smi_attnpooled(=mean of smi_ctx) ]  each 256 -> concat 1024
  MLP head 1024 -> 256 -> 64 -> 1  with PReLU + dropout 0.5

All branches are masked (padding) correctly; softmax over masked keys is set to -inf.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


D = 256


class MultiScaleConv1d(nn.Module):
    """Parallel Conv1d kernels {1,3,5,7} -> out channels {32,32,64,128} (=256),
    concat -> Linear(256->256) -> +residual. Operates on [B,L,D] (transposed inside)."""

    KERNELS = (1, 3, 5, 7)
    CHANNELS = (32, 32, 64, 128)

    def __init__(self, d=D):
        super().__init__()
        assert sum(self.CHANNELS) == d
        self.convs = nn.ModuleList([
            nn.Conv1d(d, c, k, padding=k // 2) for k, c in zip(self.KERNELS, self.CHANNELS)
        ])
        self.proj = nn.Linear(d, d)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask=None):
        # x: [B,L,D]; key_padding_mask: [B,L] True where PAD
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        h = x.transpose(1, 2)                       # [B,D,L]
        outs = [conv(h) for conv in self.convs]     # each [B,c,L]
        h = torch.cat(outs, dim=1).transpose(1, 2)  # [B,L,D]
        h = self.act(self.proj(h))
        return self.norm(x + h)                     # residual + norm


class Branch(nn.Module):
    """Multi-scale conv + residual + one Transformer encoder layer."""

    def __init__(self, d=D, nhead=8, ff=1024):
        super().__init__()
        self.msc = MultiScaleConv1d(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x, key_padding_mask=None):
        x = self.msc(x, key_padding_mask=key_padding_mask)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


def masked_mean(x, mask):
    """x:[B,L,D], mask:[B,L] True where PAD -> [B,D] mean over valid positions."""
    valid = (~mask).float().unsqueeze(-1)          # [B,L,1]
    s = (x * valid).sum(1)
    n = valid.sum(1).clamp_min(1.0)
    return s / n


class HonestAffinityPocket(nn.Module):
    def __init__(self, vocab_size=53, d=D, dropout_head=0.5):
        super().__init__()
        self.d = d
        # protein branch
        self.prot_proj = nn.Linear(1280, d)
        self.pocket_marker = nn.Embedding(2, d)     # learned binary marker
        self.prot_branch = Branch(d)
        # ligand branch
        self.smi_embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.smi_branch = Branch(d)
        # head
        self.head = nn.Sequential(
            nn.Linear(4 * d, 256), nn.PReLU(), nn.Dropout(dropout_head),
            nn.Linear(256, 64), nn.PReLU(), nn.Dropout(dropout_head),
            nn.Linear(64, 1),
        )
        self._init_marker()

    def _init_marker(self):
        # small init so an untrained marker barely perturbs reps
        nn.init.normal_(self.pocket_marker.weight, std=0.02)

    def forward(self, esm, prot_mask_pad, pocket_marker, smi_tokens, smi_mask_pad):
        """
        esm            : [B,Lp,1280]  frozen ESM per-residue (padded)
        prot_mask_pad  : [B,Lp] bool  True where PAD
        pocket_marker  : [B,Lp] long  {0,1} pocket indicator (0 at pad, ignored via mask)
        smi_tokens     : [B,Ll] long  token ids (0 == pad)
        smi_mask_pad   : [B,Ll] bool  True where PAD
        """
        # protein
        zp = self.prot_proj(esm)
        zp = zp + self.pocket_marker(pocket_marker)
        Zprot = self.prot_branch(zp, key_padding_mask=prot_mask_pad)   # [B,Lp,D]
        # ligand
        zs = self.smi_embed(smi_tokens)
        Zsmi = self.smi_branch(zs, key_padding_mask=smi_mask_pad)      # [B,Ll,D]

        # compatibility matrix
        S = torch.matmul(Zprot, Zsmi.transpose(1, 2)) / math.sqrt(self.d)  # [B,Lp,Ll]

        # mask invalid keys before softmax
        neg = torch.finfo(S.dtype).min
        prot_key = prot_mask_pad.unsqueeze(2)   # [B,Lp,1] True==pad prot
        smi_key = smi_mask_pad.unsqueeze(1)     # [B,1,Ll] True==pad lig

        # protein-side context: for each prot residue, attend over ligand tokens
        S_lig = S.masked_fill(smi_key, neg)
        A_lig = torch.softmax(S_lig, dim=2)                       # [B,Lp,Ll]
        prot_ctx = torch.matmul(A_lig, Zsmi)                      # [B,Lp,D]

        # ligand-side context: for each lig token, attend over prot residues
        S_prot = S.masked_fill(prot_key, neg)
        A_prot = torch.softmax(S_prot, dim=1)                     # [B,Lp,Ll]
        smi_ctx = torch.matmul(A_prot.transpose(1, 2), Zprot)    # [B,Ll,D]

        prot_mean = masked_mean(Zprot, prot_mask_pad)            # [B,D]
        prot_attn = masked_mean(prot_ctx, prot_mask_pad)        # [B,D]
        smi_mean = masked_mean(Zsmi, smi_mask_pad)              # [B,D]
        smi_attn = masked_mean(smi_ctx, smi_mask_pad)          # [B,D]

        pooled = torch.cat([prot_mean, prot_attn, smi_mean, smi_attn], dim=-1)  # [B,4D]
        return self.head(pooled).squeeze(-1)                     # [B]
