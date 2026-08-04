"""Cross-attention interaction head for the two-tower affinity probe.

Frozen pocket + ligand towers each emit per-patch tokens on the SHARED 8³ patch grid
(N = g_p³ tokens, dim D). This head models their INTERACTION explicitly — bidirectional
cross-attention (pocket↔ligand) with axial 3D RoPE on the shared grid so a ligand token
preferentially attends to the pocket tokens it physically contacts — then pools both sides
and regresses pK. This is the trainable part of the two-tower probe (towers stay frozen).
"""
import torch
import torch.nn as nn

from voxbind.models.density_vit import RoPE3D


class _CrossAttn(nn.Module):
    """Multi-head cross-attention: queries from `x`, keys/values from `y`, both (B,N,D),
    with optional shared-grid 3D RoPE applied to q and k (spatial grounding)."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, y, rope: "RoPE3D" = None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)   # (B,h,N,dh)
        k = self.k(y).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.v(y).view(B, -1, self.h, self.dh).transpose(1, 2)
        if rope is not None:
            q, k = rope.rotate_qk(q, k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.o(out)


class _Block(nn.Module):
    """One bidirectional interaction layer: pocket attends to ligand and vice-versa,
    each with a residual + MLP (pre-norm)."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.np1, self.nl1 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.p2l = _CrossAttn(dim, n_heads)      # pocket queries ligand
        self.l2p = _CrossAttn(dim, n_heads)      # ligand queries pocket
        self.np2, self.nl2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.pmlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.lmlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, p, l, rope):
        pn, ln = self.np1(p), self.nl1(l)
        p = p + self.p2l(pn, ln, rope)
        l = l + self.l2p(ln, pn, rope)
        p = p + self.pmlp(self.np2(p))
        l = l + self.lmlp(self.nl2(l))
        return p, l


class CrossAttnAffinityHead(nn.Module):
    """Two-tower interaction head → scalar pK.

    Args:
      dim       token width of BOTH towers (must match; project first if they differ)
      grid_p    patches per axis (g_p = grid_dim // patch_size, e.g. 64//8 = 8)
      n_layers  interaction blocks · n_heads · mlp_ratio · use_rope
    """
    def __init__(self, dim: int, grid_p: int = 8, n_layers: int = 2, n_heads: int = 8,
                 mlp_ratio: int = 4, use_rope: bool = True, dropout: float = 0.1):
        super().__init__()
        self.rope = RoPE3D(grid_p, self.__head_dim(dim, n_heads)) if use_rope else None
        self.blocks = nn.ModuleList([_Block(dim, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.norm_p, self.norm_l = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # pooled pocket + ligand + their element-wise product (interaction) → MLP → pK
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Dropout(dropout),
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    @staticmethod
    def __head_dim(dim, n_heads):
        return dim // n_heads

    def forward(self, pocket_tokens, ligand_tokens):
        p, l = pocket_tokens, ligand_tokens                 # (B, N, D) each
        for blk in self.blocks:
            p, l = blk(p, l, self.rope)
        p = self.norm_p(p).mean(dim=1)                      # (B, D)
        l = self.norm_l(l).mean(dim=1)
        feat = torch.cat([p, l, p * l], dim=-1)             # (B, 3D)
        return self.head(feat).squeeze(-1)                  # (B,)
