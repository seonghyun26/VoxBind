"""00j_urepa_cache_cdg.py — precompute frozen champion-CDG target tokens for U-REPA.

For each U-REPA subset sample (built by 00i), voxelize the holo complex EXACTLY as
VoxBind does (grid_dim 64, same frame) into the champion CDG's 13-ch input
    [ ligand(7), pocket(4), density(1), gradmag(1) ]      (channel_groups [7,4,2])
run the FROZEN champion CDG, and cache its pooled patch tokens (512, 640) keyed by
pid.  Removes the ViT forward from the finetuning loop entirely (plan §4).

Two invariants that MUST hold for the alignment to be meaningful:
  1. FRAME MATCH — run in the finetuning env with the SAME voxelizer + density crops
     VoxBind uses, so the cached 8³ CDG token grid corresponds voxel-for-voxel to the
     U-Net 8³ bottleneck.
  2. NORMALIZATION MATCH — density must use the champion's pretraining recipe (PLINDER
     arcsinh+z); the density source (resample_dir / crops) should already carry it.

OPTIONAL PATH — the default is now a LIVE teacher forward in the training loop
(no_grad + bf16; see urepa_train_integration.md §2-3).  Caching is only worth it for an
un-augmented run, because of the caveat below.

Augmentation caveat: cache at CANONICAL orientation and keep the alignment-loss
samples un-rotated at train time (or use the pool_samples manifold loss, which is
rotation-tolerant).  Rotating the U-Net crop while the cached CDG stays canonical
would break the token↔token spatial correspondence.  Neither escape is free: dropping
rotation for the aligned subset is a distribution mismatch, and pool_samples discards
the intra-sample axis that has the most headroom (bottleneck token CKA 0.254 vs pooled
0.527, test/repa_cka_profile.py).  A live teacher sees the student's own augmented
frame and has neither problem.

    # verify the load→assemble→encode→cache core (no density needed):
    python dataset/00j_urepa_cache_cdg.py --self_test

    # real run (finetuning env), density from the pretraining resample pipeline:
    python dataset/00j_urepa_cache_cdg.py \
        --subset dataset/data/pretrain/urepa_subset.pt \
        --champion_dir model_zoo/champion_100m_v2_mask075 --epoch 49 \
        --out dataset/data/pretrain/urepa_cdg_tokens.pt
      (feed density via --loader; see reference_loader below)
"""
import argparse
import importlib.util
from pathlib import Path

import gemmi  # noqa: F401  (import before torch — silent _load_grid failures otherwise)
import torch

from voxbind.voxelizer import Voxelizer
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore

VOXDATA = Path(__file__).resolve().parent / "data"


def _load_probe_module():
    """01c starts with a digit → import by path to reuse load_encoder()."""
    spec = importlib.util.spec_from_file_location(
        "_probe01c", Path(__file__).resolve().parent / "01c_pdbbind_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── core: assemble the champion's 13-ch input, encode → pooled tokens ────────────
def assemble_cdg_input(v_lig, v_poc, density, gradmag=None):
    """[ ligand(7), pocket(4), density(1), gradmag(1) ] → (B, 13, G,G,G).

    Mirrors train_density's atomblob_density(+gradmag) assembly EXACTLY. gradmag is
    derived from the density if not supplied (per_sample_zscore(‖∇ρ‖)).
    """
    if gradmag is None:
        gradmag = per_sample_zscore(gradient_magnitude3d(density))
    return torch.cat([v_lig, v_poc, density, gradmag], dim=1)


@torch.no_grad()
def encode_tokens(encoder, x):
    """(B, 13, G,G,G) → pooled CDG patch tokens (B, 512, 640)."""
    return encoder._pool_groups(encoder.forward_features(x))


# ── reference density loader (finetuning env) ───────────────────────────────────
def reference_loader(subset, cfg_dataset, split="train"):
    """Yield (pids, ligand, pocket, density, gradmag) batches for the subset pids,
    from the SAME density dataset VoxBind finetuning uses (frame-consistent).

    Left as a thin adapter: on the finetuning server, build the density dataset the
    training config produces (e.g. DatasetCrossDockedXray / resample), filter to
    `subset['pids']`, and yield the raw coord lists + density(+gradmag) crops. The
    voxelization happens in cache_subset() via the shared Voxelizer, so the only
    env-specific choice is WHERE the density crop comes from.
    """
    raise NotImplementedError(
        "wire to the finetuning-env density dataset (see docstring); "
        "the encode+cache core below is env-independent and --self_test-verified")


# ── driver ──────────────────────────────────────────────────────────────────────
def cache_subset(loader, encoder, voxelizer, out, n_lig=7, n_poc=4, half=True):
    cache, n = {}, 0
    for pids, ligand, pocket, density, gradmag in loader:
        v_lig = voxelizer.forward(ligand, num_channels=n_lig)
        v_poc = voxelizer.forward(pocket, num_channels=n_poc)
        x = assemble_cdg_input(v_lig, v_poc, density, gradmag)
        tok = encode_tokens(encoder, x).cpu()
        if half:
            tok = tok.half()
        for i, pid in enumerate(pids):
            cache[pid] = tok[i].clone()
        n += len(pids)
        print(f"  cached {n}", end="\r")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": cache, "dim": 640, "grid": 8, "source": "champion_cdg"}, out)
    print(f"\n→ wrote {len(cache):,} token sets to {out}")


def self_test():
    """load champion → encode random 13-ch inputs → write+reload cache. No density."""
    m = _load_probe_module()
    enc = m.load_encoder(Path("model_zoo/champion_100m_v2_mask075"), 49, "cpu").eval()
    x = torch.randn(3, 13, 64, 64, 64)
    tok = encode_tokens(enc, x)
    assert tok.shape == (3, 512, 640), tok.shape
    # assemble path from separate channels
    x2 = assemble_cdg_input(torch.randn(2, 7, 64, 64, 64), torch.randn(2, 4, 64, 64, 64),
                            torch.randn(2, 1, 64, 64, 64))
    assert x2.shape == (2, 13, 64, 64, 64), x2.shape
    out = VOXDATA / "pretrain" / "_urepa_cache_selftest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": {"a": tok[0].half(), "b": tok[1].half()}, "dim": 640, "grid": 8}, out)
    back = torch.load(out, weights_only=False)["tokens"]
    assert back["a"].shape == (512, 640)
    out.unlink()
    print(f"[self_test] encode {tuple(tok.shape)} | assemble {tuple(x2.shape)} | "
          f"cache round-trip {tuple(back['a'].shape)}  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(VOXDATA / "pretrain" / "urepa_subset.pt"))
    ap.add_argument("--champion_dir", default="model_zoo/champion_100m_v2_mask075")
    ap.add_argument("--epoch", type=int, default=49)
    ap.add_argument("--out", default=str(VOXDATA / "pretrain" / "urepa_cdg_tokens.pt"))
    ap.add_argument("--grid_dim", type=int, default=64)
    ap.add_argument("--resolution", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(); return

    m = _load_probe_module()
    encoder = m.load_encoder(Path(args.champion_dir), args.epoch, args.device).eval()
    voxelizer = Voxelizer(grid_dim=args.grid_dim, resolution=args.resolution,
                          device=args.device)
    subset = torch.load(args.subset, weights_only=False)
    print(f"subset: {len(subset['pids']):,} density-bearing native samples")
    loader = reference_loader(subset, cfg_dataset=None)   # wire in finetuning env
    cache_subset(loader, encoder, voxelizer, args.out)


if __name__ == "__main__":
    main()
