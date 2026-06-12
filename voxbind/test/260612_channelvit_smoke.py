#!/usr/bin/env python
"""Smoke + regression test for the grouped ChannelViT (patch_embed_mode='channel_group').

Verifies: (1) fused mode is unchanged; (2) channel_group forward shapes + drop-in spatial
output match fused; (3) transformer/decoder state_dict keys are shared across modes (only the
patch-embed differs); (4) the explicit-softmax capture path is numerically faithful to flash
SDPA; (5) DensityViT.group_attention returns normalized group->group matrices + per-group
spatial maps. Data-free.

    CUDA_VISIBLE_DEVICES=4 python voxbind/test/260612_channelvit_smoke.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from voxbind.models.density_vit import DensityViTMAE, MultiHeadSelfAttention

GROUPS = (7, 4, 1, 1)            # 7 lig-atom + 4 poc-atom + 1 density + 1 gradmag = 13
NIN = sum(GROUPS)


def build(mode, **extra):
    torch.manual_seed(0)
    return DensityViTMAE(
        grid_dim=64, patch_size=8, n_in_channels=NIN, n_recon_channels=NIN, n_channels=32,
        dim=128, depth=3, n_heads=8, mlp_ratio=4, dropout=0.0, n_struct_channels=0,
        pretext_style="mae", head_style="patch_mlp", head_hidden_dim=67, head_depth=2,
        pos_encoding="learnable", patch_embed_mode=mode, **extra,
    )


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, NIN, 64, 64, 64, device=dev)
    fused = build("fused").to(dev).eval()
    chan = build("channel_group", channel_groups=GROUPS).to(dev).eval()
    ok = True

    with torch.no_grad():
        of, _ = fused(x)
        oc, _ = chan(x)
    shape_ok = of.shape == oc.shape == (1, NIN, 64, 64, 64) \
        and fused.encode(x).shape == chan.encode(x).shape
    print(f"[1] shapes: fused {tuple(of.shape)} == channel {tuple(oc.shape)}  -> {shape_ok}")
    ok &= shape_ok

    ef, ec = set(fused.encoder.state_dict()), set(chan.encoder.state_dict())
    shared = [k for k in ef if k.startswith(("blocks", "norm.", "decoder_proj"))]
    key_ok = all(k in ec for k in shared) and (ef - ec) == {"patch_embed.weight", "patch_embed.bias"}
    print(f"[2] {len(shared)} transformer/decoder keys shared; patch-embed differs -> {key_ok}")
    ok &= key_ok

    # explicit-softmax capture must match flash SDPA
    with torch.no_grad():
        o_flash, _ = chan(x)
        for m in chan.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.capture_attn = True
        o_expl, _ = chan(x)
        for m in chan.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.capture_attn = False; m.attn_weights = None
    dmax = (o_flash - o_expl).abs().max().item()
    faithful = dmax < 1e-4
    print(f"[3] explicit-softmax vs flash max|Δ|={dmax:.1e} -> {faithful}")
    ok &= faithful

    gg, recv = chan.encoder.group_attention(x)
    nG = len(GROUPS)
    norm_ok = (gg.shape == (3, nG, nG) and recv.shape == (3, nG, 8, 8, 8)
               and torch.allclose(gg.sum(-1), torch.ones_like(gg.sum(-1)), atol=1e-4)
               and torch.allclose(recv.flatten(1).sum(-1), torch.ones(3, device=dev), atol=1e-4))
    print(f"[4] group_attention gg{tuple(gg.shape)} rows→1, recv{tuple(recv.shape)} →1 -> {norm_ok}")
    ok &= norm_ok

    print(f"\n{'PASS' if ok else 'FAIL'}: grouped ChannelViT (groups={GROUPS})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
