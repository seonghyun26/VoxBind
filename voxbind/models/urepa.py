"""U-REPA representation alignment: distil a frozen CDG ViT into VoxBind.

Implements the alignment head + manifold loss from the plan
(notebook/html/260806/urepa_density_alignment_plan.md), following
U-REPA (arXiv 2503.18414, "Aligning Diffusion U-Nets to ViTs"):

  • align at the U-Net MIDDLE stage (bottleneck) only — a single alignment point;
    skip connections distort shallow-layer alignment (U-REPA prescription #1).
  • MLP projection FIRST, then upsample — upscale the U-Net feature toward the ViT
    token grid, never downscale the teacher (prescription #2).
  • MANIFOLD loss, not tokenwise cosine — align the relational (inter-sample /
    inter-token) similarity structure, because the U-Net and ViT feature spaces
    have a large gap and strict per-token correspondence is too rigid (#3).

Training-time only. Inference is stock VoxBind (no CDG encoder, no density).

Shapes for the reproduced VoxBind + champion CDG (both on a 64³ grid, same frame):
    U-Net bottleneck : (B, 128, 8, 8, 8)   [n_channels 32 · ch_mults[-1] 4; 64/2³ = 8]
    CDG tokens       : (B, 512, 640)        [64/patch 8 = 8³ tokens, dim 640]
  → grids already match (8³), so the projector is a pure 128→640 channel map and
    the spatial upsample degenerates to identity. The upsample path is kept general
    so a different ch_mults (deeper bottleneck) still works.

The CDG side is a *frozen* target: precompute + cache its tokens once (see the plan's
Phase-A caching step) so no ViT forward runs in the training loop.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Projector: U-Net bottleneck feature → CDG token space ───────────────────────
class UREPAProjector(nn.Module):
    """Per-voxel MLP (1³ convs) mapping the U-Net bottleneck to the CDG token dim,
    then (only if the U-Net grid is coarser) a trilinear upsample to the CDG token
    grid. Returns tokens (B, N_cdg, cdg_dim). Never downscales the teacher.
    """

    def __init__(
        self,
        unet_ch: int,
        cdg_dim: int,
        unet_grid: int,
        cdg_grid: int,
        hidden: Optional[int] = None,
        n_layers: int = 3,
    ):
        super().__init__()
        assert unet_grid <= cdg_grid, (
            f"U-REPA upscales the U-Net feature toward the teacher and never "
            f"downscales the ViT target; got unet_grid={unet_grid} > cdg_grid={cdg_grid}"
        )
        assert n_layers >= 1
        self.unet_grid, self.cdg_grid, self.cdg_dim = unet_grid, cdg_grid, cdg_dim
        hidden = hidden or max(cdg_dim, unet_ch)

        layers, c = [], unet_ch
        for _ in range(n_layers - 1):
            layers += [nn.Conv3d(c, hidden, kernel_size=1), nn.SiLU()]
            c = hidden
        layers += [nn.Conv3d(c, cdg_dim, kernel_size=1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, unet_ch, g, g, g)
        x = self.mlp(feat)                                    # (B, cdg_dim, g, g, g)
        if self.unet_grid != self.cdg_grid:
            x = F.interpolate(
                x, size=self.cdg_grid, mode="trilinear", align_corners=False
            )
        b, d = x.shape[0], x.shape[1]
        return x.reshape(b, d, -1).transpose(1, 2)            # (B, N_cdg, cdg_dim)


# ── Manifold (relational) alignment loss ────────────────────────────────────────
def manifold_alignment_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    tau: float = 0.5,
    n_sample: int = 2048,
    mode: str = "relkl",
    pool_samples: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Relational alignment between projected U-Net tokens and frozen CDG tokens.

    Both are L2-normalised, then a similarity matrix is aligned — the *relations*
    ("what CDG considers similar, you should too"), not absolute token values.

    Args
      student, teacher : (B, N, D)   D must match (projector maps to cdg_dim).
      tau              : similarity temperature.
      n_sample         : cap on the number of points in the similarity matrix
                         (random subset of the B·N batched tokens) for tractability;
                         0/None → use all.
      mode             : 'relkl' — KL(teacher-relations ‖ student-relations),
                                    off-diagonal, scale-free (default);
                         'gram'  — MSE between cosine-similarity matrices.
      pool_samples     : if True, mean-pool each sample's tokens first → the manifold
                         is over the B samples (pure inter-sample, batch-size driven).
                         Default False → over all B·N tokens (inter-sample AND
                         inter-token, since tokens from every sample co-reside).

    Because the relations span samples, BATCH SIZE is a real hyperparameter — the
    similarity matrix needs actual co-resident samples (grad-accum does not help).
    """
    b, n, d = student.shape
    teacher = teacher.to(student.dtype)
    if pool_samples:
        S = student.mean(dim=1)                              # (B, D)
        T = teacher.mean(dim=1)
    else:
        S = student.reshape(b * n, d)                        # (B·N, D)
        T = teacher.reshape(b * n, d)

    S = F.normalize(S, dim=1)
    T = F.normalize(T, dim=1)

    m = S.shape[0]
    if n_sample and 0 < n_sample < m:
        idx = torch.randperm(m, device=S.device, generator=generator)[:n_sample]
        S, T = S[idx], T[idx]
        m = n_sample

    sim_s = (S @ S.t()) / tau                                # (m, m)
    sim_t = (T @ T.t()) / tau

    if mode == "gram":
        # align the raw cosine-similarity matrices (undo the tau scaling).
        return F.mse_loss(sim_s * tau, sim_t * tau)

    if mode != "relkl":
        raise ValueError(f"unknown manifold loss mode={mode!r} (use 'relkl'|'gram')")

    # relational distillation: mask self-relations, teacher defines the target
    # distribution over the OTHER points, student is pulled to match it.
    eye = torch.eye(m, device=S.device, dtype=torch.bool)
    neg = torch.finfo(sim_s.dtype).min
    sim_s = sim_s.masked_fill(eye, neg)
    sim_t = sim_t.masked_fill(eye, neg)
    p = F.softmax(sim_t, dim=1)                              # teacher relations (target)
    log_q = F.log_softmax(sim_s, dim=1)                      # student relations
    return F.kl_div(log_q, p, reduction="batchmean")


# ── Alignment head (projector + loss) ───────────────────────────────────────────
class UREPAAlignment(nn.Module):
    """Trainable projector on the U-Net bottleneck; exposes the alignment loss
    against precomputed frozen CDG tokens. The only thing REPA adds to the graph.
    """

    def __init__(
        self,
        unet_ch: int = 128,
        cdg_dim: int = 640,
        unet_grid: int = 8,
        cdg_grid: int = 8,
        hidden: Optional[int] = None,
        n_layers: int = 3,
        tau: float = 0.5,
        n_sample: int = 2048,
        mode: str = "relkl",
        pool_samples: bool = False,
    ):
        super().__init__()
        self.projector = UREPAProjector(
            unet_ch, cdg_dim, unet_grid, cdg_grid, hidden=hidden, n_layers=n_layers
        )
        self.tau, self.n_sample, self.mode, self.pool_samples = (
            tau, n_sample, mode, pool_samples,
        )

    def forward(
        self,
        unet_feat: torch.Tensor,
        cdg_tokens: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        proj = self.projector(unet_feat)                     # (B, N_cdg, cdg_dim)
        return manifold_alignment_loss(
            proj, cdg_tokens, tau=self.tau, n_sample=self.n_sample,
            mode=self.mode, pool_samples=self.pool_samples, generator=generator,
        )


# ── Bottleneck tap: grab the MiddleBlock output without editing UNet3D ───────────
class BottleneckTap:
    """Forward-hook capture of the U-Net middle-stage (bottleneck) feature — the
    single U-REPA alignment point. Non-invasive; register on `voxbind.unet3d.middle`.

        tap = BottleneckTap(voxbind.unet3d.middle)
        out = voxbind(...)              # normal forward
        feat = tap.feature             # (B, C_bot, g, g, g)
        ...
        tap.remove()

    `retain_graph=True` is implicit — the hooked tensor is part of the live graph,
    so the alignment loss backprops into the U-Net (when unfrozen).
    """

    def __init__(self, module: nn.Module):
        self.feature: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self.feature = out

    def remove(self):
        self._handle.remove()


# ── Integration sketch (training loop lives on the finetune server) ──────────────
# Dual loss  L = L_denoise + λ · L_align  ,  L_align only on density-bearing samples:
#
#   align = UREPAAlignment(unet_ch=128, cdg_dim=640, unet_grid=8, cdg_grid=8)
#   tap   = BottleneckTap(voxbind.unet3d.middle)
#   ...
#   pred      = voxbind(noisy_lig, pocket)          # runs U-Net → tap.feature set
#   L_denoise = denoise_loss(pred, target)
#   if has_density.any():                            # subset with cached CDG tokens
#       L_align = align(tap.feature[has_density], cdg_tokens[has_density])
#       loss = L_denoise + lam * L_align
#   else:
#       loss = L_denoise
#
# Unfreezing ladder: start with only `align.projector` (+ optional adapter/LoRA)
# trainable, U-Net frozen — the frozen-U-Net alignment-loss floor measures how
# compatible the two representations already are (plan §4 decision point).


if __name__ == "__main__":  # shape / grad smoke test
    torch.manual_seed(0)
    B = 8
    unet_feat = torch.randn(B, 128, 8, 8, 8, requires_grad=True)
    cdg = torch.randn(B, 512, 640)                            # frozen target
    head = UREPAAlignment(unet_ch=128, cdg_dim=640, unet_grid=8, cdg_grid=8)

    proj = head.projector(unet_feat)
    assert proj.shape == (B, 512, 640), proj.shape

    for mode in ("relkl", "gram"):
        head.mode = mode
        loss = head(unet_feat, cdg)
        assert loss.ndim == 0 and torch.isfinite(loss), (mode, loss)
        loss.backward(retain_graph=True)
        assert unet_feat.grad is not None and torch.isfinite(unet_feat.grad).all()
        print(f"[{mode:5s}] proj {tuple(proj.shape)}  loss {loss.item():.4f}  grad OK")

    # pooled (pure inter-sample) variant
    head.mode, head.pool_samples = "relkl", True
    loss = head(unet_feat, cdg)
    print(f"[relkl/pool] loss {loss.item():.4f}")

    # coarser bottleneck (e.g. 4³) → trilinear upsample path to the 8³ teacher grid
    up = UREPAProjector(unet_ch=64, cdg_dim=640, unet_grid=4, cdg_grid=8)
    up_out = up(torch.randn(B, 64, 4, 4, 4))
    assert up_out.shape == (B, 512, 640), up_out.shape
    print(f"[upsample] 4³→8³ tokens {tuple(up_out.shape)}  OK")

    # guard: refuse to downscale the ViT teacher (unet_grid > cdg_grid)
    print("[guard] downscale rejected:", end=" ")
    try:
        UREPAProjector(unet_ch=64, cdg_dim=640, unet_grid=16, cdg_grid=8)
        print("NO (BUG)")
    except AssertionError:
        print("yes")
