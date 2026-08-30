"""GPU smoke for fusion='v4': build with a real encoder, one fwd+bwd at the training
batch size, assert the zero-init identity, report peak memory.

Exists because v4 is reached from an unattended chain. A geometry mismatch, a CUDA-only
shape bug, or an OOM from v4's wider 64^3 activations would otherwise surface as a dead
350-epoch launch hours later, so the chain runs this first and refuses to launch on a
non-zero exit.

  python test/v4_gpu_smoke.py --encoder <path/to/checkpoint_e*.pth.tar> [--bsz 32]

Geometry is read from the encoder folder's own cfg.yaml, exactly as the chain does --
model_zoo entries are not interchangeable at fixed dims.
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from voxbind.models.voxbind import VoxBind  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--bsz", type=int, default=32, help="per-rank batch size of the real run")
    ap.add_argument("--grid", type=int, default=64)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("[v4-smoke] FAIL: no CUDA device")
        return 1
    cfg_path = os.path.join(os.path.dirname(a.encoder), "cfg.yaml")
    m = (yaml.safe_load(open(cfg_path)) or {})
    m = m.get("model", m)
    groups = [int(x) for x in (m.get("channel_groups") or [7, 4, 2])]
    print(f"[v4-smoke] encoder={a.encoder}")
    print(f"[v4-smoke] geometry: patch={m.get('patch_size', 8)} dim={m.get('dim')} "
          f"depth={m.get('depth')} heads={m.get('heads')} groups={groups} "
          f"n_in={m.get('n_in_channels', 13)}")

    model = VoxBind(
        n_channels_ligand=7, n_channels_pocket=4, n_channels=32,
        ch_mults=[1, 2, 2, 4], is_attn=[False, False, True, True], n_blocks=2,
        n_groups=16, dropout=0.1, smooth_sigma=0.9,
        with_density=True, density_encoder_type="vit",
        density_grid_dim=int(m.get("grid_dim", a.grid)),
        density_vit_patch=int(m.get("patch_size", 8)),
        density_vit_dim=int(m["dim"]), density_vit_depth=int(m["depth"]),
        density_vit_heads=int(m["heads"]), density_vit_mlp_ratio=int(m.get("mlp_ratio", 4)),
        density_vit_dropout=float(m.get("dropout", 0.1)),
        density_vit_n_in_channels=int(m.get("n_in_channels", 13)),
        density_vit_patch_embed_mode=str(m.get("patch_embed_mode", "channel_group")),
        density_vit_channel_groups=groups,
        density_encoder_amp=True, fusion="v4",
    )

    # strict: a silently partial encoder load is the exact failure this guards against
    ck = torch.load(a.encoder, map_location="cpu", weights_only=False)
    sd = ck.get("encoder_state_dict_ema") or ck.get("state_dict_ema", ck.get("state_dict", {}))
    sd = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    model.density_encoder.load_state_dict(sd, strict=True)
    for p in model.density_encoder.parameters():
        p.requires_grad = False
    model.density_encoder.eval()
    print(f"[v4-smoke] encoder loaded strict; token_trunk[0]="
          f"{model.token_trunk[0]}")

    model = model.cuda()
    G, B = a.grid, a.bsz
    lig = torch.randn(B, 7, G, G, G, device="cuda")
    poc = torch.randn(B, 4, G, G, G, device="cuda")
    den = torch.randn(B, 1, G, G, G, device="cuda")

    model.eval()
    with torch.no_grad():
        d = (model(lig[:2], poc[:2], density=den[:2])
             - model(lig[:2], poc[:2], density=None)).abs().max().item()
    print(f"[v4-smoke] zero-init identity: max|with - without| = {d:.3e}")
    if d != 0.0:
        print("[v4-smoke] FAIL: v4 is not identity at init")
        return 1

    model.train()
    torch.cuda.reset_peak_memory_stats()
    out = model(lig, poc, density=den)
    out.square().mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    grads = {n: (p.grad is not None) for n, p in model.named_parameters()
             if n.startswith(("token_trunk", "token_proj"))}
    print(f"[v4-smoke] fwd+bwd ok at bsz={B}: out={tuple(out.shape)} "
          f"peak={peak:.1f} GiB / {total:.1f} GiB")
    print(f"[v4-smoke] v4 params in graph: {sum(grads.values())}/{len(grads)}")
    if not all(grads.values()):
        print(f"[v4-smoke] FAIL: no grad for {[n for n, g in grads.items() if not g]}")
        return 1
    print("[v4-smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
