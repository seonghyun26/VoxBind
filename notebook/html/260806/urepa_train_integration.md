# U-REPA training integration (reference patch)

How to wire the alignment loss into VoxBind `train.py` on the finetuning server.
Everything below is **additive** — inference and the base denoise path are untouched.

Prereqs (all built on the analysis server, transfer the two `.pt`):
- `models/urepa.py` — `UREPAAlignment`, `BottleneckTap`, `manifold_alignment_loss`
- `dataset/data/pretrain/urepa_subset.pt` — 6,181 density-bearing native pids (`00i`)
- `dataset/data/pretrain/urepa_cdg_tokens.pt` — `{pid: (512,640) fp16}` frozen CDG targets (`00j`)

Reproduced VoxBind ↔ champion CDG shapes are fixed: U-Net bottleneck `(B,128,8³)`
(`n_channels 32 · ch_mults[-1] 4`, `64/2³`) ↔ CDG tokens `(B,512,640)` — grids match,
projector is a pure 128→640 map.

---

## 1. Dataloader must yield `pid`

The prefetcher (`train.py:64-85`) yields `(voxels_lig, smooth_voxels_lig, voxels_poc,
density)` — add the batch `pid`s so we can (a) gate L_align to the density subset and
(b) look up the cached CDG tokens. In `ForwardWrapper._voxelize` carry `batch["pid"]`
(the pocket `id`) through and append it to the yielded tuple.

## 2. Setup (after `model = create_model(cfg, device)`, `train.py:163`)

```python
from voxbind.models.urepa import UREPAAlignment, BottleneckTap

cdg_tokens = torch.load(cfg.urepa.tokens, weights_only=False)["tokens"]   # {pid: (512,640)}
align = UREPAAlignment(unet_ch=128, cdg_dim=640, unet_grid=8, cdg_grid=8,
                       tau=cfg.urepa.tau, n_sample=cfg.urepa.n_sample,
                       mode=cfg.urepa.mode).to(device)
tap = BottleneckTap(model.unet3d.middle)          # captures (B,128,8,8,8) each forward

# Unfreezing ladder — STAGE 1: projector only, U-Net frozen (plan §4 decision point).
for p in model.parameters():        p.requires_grad_(False)
optimizer = AdamW(align.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
# STAGE 2/3 later: unfreeze model.unet3d.middle (+ neighbours) / all, add to optimizer.
```

## 3. Dual-loss step (replace `train.py:280-286`)

```python
pred = model(smooth_voxels_lig, voxels_poc, density=density)   # density stays None (density-free VoxBind)
L_denoise = criterion(pred, voxels_lig)                        # full batch, always

# L_align only on batch members that have a cached CDG target (the density subset)
have = [(i, pid) for i, pid in enumerate(pids) if pid in cdg_tokens]
if have:
    idx = torch.tensor([i for i, _ in have], device=device)
    tgt = torch.stack([cdg_tokens[p].float() for _, p in have]).to(device)   # (k,512,640)
    L_align = align(tap.feature[idx], tgt)                     # tap.feature: (B,128,8³)
    loss = L_denoise + cfg.urepa.lam * L_align
else:
    loss = L_denoise

loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

`tap.feature` is the live-graph bottleneck tensor, so `L_align` backprops into the
U-Net once it is unfrozen (Stage 2+); in Stage 1 it only trains the projector.

## 4. Augmentation caveat (must handle)

CDG tokens are cached at **canonical orientation**. If VoxBind rotates the crop
(`cfg.aug`), the U-Net 8³ bottleneck no longer corresponds to the canonical CDG
token grid. Options, in order of preference:
1. Disable rotation for the density-subset samples only (denoise loss still augments
   the full set). Cleanest.
2. Use `align.pool_samples = True` (per-sample pooled manifold loss) — rotation-tolerant,
   at the cost of discarding intra-sample spatial structure.
3. Cache K rotations per pid and pick the matching one (heaviest).

## 5. Experiments (plan §5)

- **λ sweep** `cfg.urepa.lam ∈ {0, 0.1, 0.5, 1, 5}` → Vina / QED / diversity / validity.
  The λ→Vina curve is the headline (arch/params/data fixed). λ=0 is the "alignment off"
  control.
- **Control: coords-only CDG target** — rebuild `00j` with the C-only encoder
  (`model_zoo/coords_100m_v2_mask075`, `assemble` without the density/gradmag channels,
  n_in=11) → `urepa_cdg_tokens_coords.pt`. Isolates the density contribution (decides the
  claim). Same architecture/pretraining as champion. (synthetic-density control dropped.)
- **Frozen-U-Net floor**: Stage 1's alignment-loss plateau measures how compatible the
  two representations already are (we pre-measured CKA 0.60, R²(CDG←C) 0.64).

## 6. Inference

Unchanged. Do **not** construct `align`/`tap`; density is never an input. Stock VoxBind
walk-jump — the alignment only reshaped the U-Net's learned representation during training.

## 7. Config block (`cfg.urepa`)

```yaml
urepa:
  tokens: dataset/data/pretrain/urepa_cdg_tokens.pt
  lam: 0.5            # swept
  tau: 0.5
  n_sample: 2048     # manifold similarity-matrix cap; batch size is the real knob
  mode: relkl        # relkl | gram
  pool_samples: false
```
