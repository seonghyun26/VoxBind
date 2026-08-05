"""U-REPA representation alignment: distil a frozen CDG ViT into VoxBind.

Implements the alignment head + manifold loss from the plan
(notebook/html/260806/urepa_density_alignment_plan.md), following
U-REPA (arXiv 2503.18414, "Aligning Diffusion U-Nets to ViTs"):

  • align at the U-Net MIDDLE stage (bottleneck) only — a single alignment point;
    skip connections distort shallow-layer alignment (U-REPA prescription #1).
  • MLP projection FIRST, then upsample — upscale the U-Net feature toward the ViT
    token grid, never downscale the teacher (prescription #2).
  • MANIFOLD loss, not tokenwise cosine — align the relational (inter-sample /
    inter-token) similarity structure (prescription #3). Justified empirically: at the
    bottleneck R²(teacher←student) ≈ 0.08, so a tokenwise cosine/MSE term would chase
    an unreachable target; only relations are learnable.

Training-time only. Inference is stock VoxBind (no CDG encoder, no density).

Shapes (finetune target: VoxBind `exp_sig0.9` ↔ teacher champion `champion_100m_v2_mask075`):
    U-Net bottleneck : (B, 512, 8, 8, 8)   [n_channels 128 · ch_mults[-1] 4; 64/2³ = 8]
    CDG tokens       : (B, 512, 640)        [64/patch 8 = 8³ tokens, dim 640]
  → grids match (8³) so the projector is a pure 512→640 channel map. Fully parameterised
    (an efficient_60m teacher would be 512→512; reproduced 32-ch VoxBind is 128→…).

The CDG side is a *frozen* target: precompute + cache its tokens once (see 00j) so no
ViT forward runs in the training loop.

## Manifold sampling — why the composition is now explicit
With a FLAT random subset of the B·N tokens, the fraction of intra-sample (spatial) pairs
in the similarity matrix is ≈ 1/B — set by the batch size alone, and `n_sample` (a
cost/variance knob) cannot change it. That silently couples WHAT the loss optimises to B,
so a tuned λ does not transfer across batch sizes / DDP world size. The default sampling
here is `split`: two SEPARATE, batch-decoupled terms whose mix is an explicit
(w_intra, w_inter), so λ transfers and one can sweep which axis carries the effect.
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
def _cos_sim(x: torch.Tensor, tau: float) -> torch.Tensor:
    """(..., m, D) → (..., m, m) cosine similarity / tau (rows L2-normalised)."""
    x = F.normalize(x, dim=-1)
    return torch.matmul(x, x.transpose(-1, -2)) / tau


def _relational(sim_s, sim_t, mode, gram_center):
    """Align two similarity matrices (..., m, m). → (loss, teacher_row_entropy[nats])."""
    m = sim_s.shape[-1]
    eye = torch.eye(m, device=sim_s.device, dtype=torch.bool)
    if mode == "relkl":
        neg = torch.finfo(sim_s.dtype).min
        p = F.softmax(sim_t.masked_fill(eye, neg), dim=-1)          # teacher relations (target)
        log_q = F.log_softmax(sim_s.masked_fill(eye, neg), dim=-1)  # student relations
        log_p = p.clamp_min(1e-12).log()
        loss = (p * (log_p - log_q)).sum(-1).mean()                 # KL(p‖q), off-diagonal
        ent = -(p * log_p).sum(-1).mean()                           # teacher row entropy
        return loss, ent
    if mode == "gram":
        s, t = sim_s, sim_t
        if gram_center:                                             # cancel a shared row offset
            s = s - s.mean(-1, keepdim=True)
            t = t - t.mean(-1, keepdim=True)
        off = (~eye).to(s.dtype)
        n_off = off.sum() * (s.numel() // (m * m))
        loss = ((s - t) ** 2 * off).sum() / n_off.clamp_min(1)
        return loss, torch.zeros((), device=s.device)
    raise ValueError(f"unknown manifold loss mode={mode!r} (use 'relkl'|'gram')")


def _sample_per_sample(student, teacher, k, generator):
    """Draw the SAME k random token indices per sample → (B, k, D) student & teacher."""
    b, n, _ = student.shape
    if k >= n:
        return student, teacher
    idx = torch.stack([torch.randperm(n, device=student.device, generator=generator)[:k]
                       for _ in range(b)])                          # (B, k)
    ar = torch.arange(b, device=student.device)[:, None]
    return student[ar, idx], teacher[ar, idx]


def manifold_alignment_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    tau: float = 0.3,
    mode: str = "relkl",
    sampling: str = "split",
    tokens_per_sample: int = 128,
    n_sample: int = 2048,
    w_intra: float = 1.0,
    w_inter: float = 1.0,
    gram_center: bool = True,
    return_stats: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """Relational (manifold) alignment between projected U-Net tokens and frozen CDG
    tokens (B, N, D). Both are L2-normalised; a similarity matrix's RELATIONS are aligned.

      sampling='split' (default): two SEPARATE, batch-decoupled terms —
          L = w_intra · L_intra + w_inter · L_inter
          L_intra: per-sample token relations (spatial manifold), all B samples, mean.
          L_inter: relations between the B mean-pooled sample vectors.
          Composition is (w_intra, w_inter), NOT 1/B. w_intra=0 ≡ pool control,
          w_inter=0 ≡ token-only. Sweep to learn which axis carries the effect.
      sampling='block': one matrix over `tokens_per_sample` tokens PER sample → intra
          fraction (k−1)/(B·k−1), an explicit knob (vs flat's implicit 1/B).
      sampling='flat':  LEGACY flat n_sample subset of B·N; intra fraction ≈ 1/B,
          coupled to batch size. Only for reproducing the old behaviour.
      sampling='pool':  mean-pool per sample → pure inter-sample (B×B) control.

    mode='relkl' (default): KL(teacher ‖ student) relations, off-diagonal, scale-free
    (row-softmax cancels a shared offset). 'gram': CENTERED MSE of the similarity
    matrices (raw gram has no offset protection — a common component would dominate).

    tau: temperature. Too soft → teacher target near-uniform → weak signal. With
    return_stats=True the teacher row entropy is reported; if it sits near log(m·term),
    lower tau.

    return_stats=True → (loss, {intra, inter, teacher_entropy, intra_frac}).
    """
    b, n, d = student.shape
    teacher = teacher.to(student.dtype)
    nan = torch.tensor(float("nan"))

    if sampling == "split":
        S, T = _sample_per_sample(student, teacher, tokens_per_sample, generator)
        l_intra, ent = _relational(_cos_sim(S, tau), _cos_sim(T, tau), mode, gram_center)
        l_inter, _ = _relational(_cos_sim(student.mean(1), tau),
                                 _cos_sim(teacher.mean(1), tau), mode, gram_center)
        loss = w_intra * l_intra + w_inter * l_inter
        stats = dict(intra=l_intra.detach(), inter=l_inter.detach(),
                     teacher_entropy=ent.detach(), intra_frac=nan)
    elif sampling == "pool":
        loss, ent = _relational(_cos_sim(student.mean(1), tau),
                                _cos_sim(teacher.mean(1), tau), mode, gram_center)
        stats = dict(intra=nan, inter=loss.detach(),
                     teacher_entropy=ent.detach(), intra_frac=torch.tensor(0.0))
    elif sampling == "block":
        S, T = _sample_per_sample(student, teacher, tokens_per_sample, generator)
        k = S.shape[1]
        loss, ent = _relational(_cos_sim(S.reshape(b * k, d), tau),
                                _cos_sim(T.reshape(b * k, d), tau), mode, gram_center)
        stats = dict(intra=nan, inter=nan, teacher_entropy=ent.detach(),
                     intra_frac=torch.tensor((k - 1) / max(b * k - 1, 1)))
    elif sampling == "flat":
        S = student.reshape(b * n, d); T = teacher.reshape(b * n, d)
        m = S.shape[0]
        if n_sample and 0 < n_sample < m:
            idx = torch.randperm(m, device=S.device, generator=generator)[:n_sample]
            S, T = S[idx], T[idx]
        loss, ent = _relational(_cos_sim(S, tau), _cos_sim(T, tau), mode, gram_center)
        stats = dict(intra=nan, inter=nan, teacher_entropy=ent.detach(),
                     intra_frac=torch.tensor(1.0 / b))
    else:
        raise ValueError(f"unknown sampling={sampling!r} (split|block|flat|pool)")

    return (loss, stats) if return_stats else loss


# ── Alignment head (projector + loss) ───────────────────────────────────────────
class UREPAAlignment(nn.Module):
    """Trainable projector on the U-Net bottleneck; exposes the manifold alignment loss
    against precomputed frozen CDG tokens. The only thing REPA adds to the graph.
    """

    def __init__(
        self,
        unet_ch: int = 512,
        cdg_dim: int = 640,          # champion CDG teacher token dim (efficient_60m = 512)
        unet_grid: int = 8,
        cdg_grid: int = 8,
        hidden: Optional[int] = None,
        n_layers: int = 3,
        tau: float = 0.3,
        mode: str = "relkl",
        sampling: str = "split",
        tokens_per_sample: int = 128,
        n_sample: int = 2048,
        w_intra: float = 1.0,
        w_inter: float = 1.0,
        gram_center: bool = True,
    ):
        super().__init__()
        self.projector = UREPAProjector(
            unet_ch, cdg_dim, unet_grid, cdg_grid, hidden=hidden, n_layers=n_layers
        )
        self.cfg = dict(tau=tau, mode=mode, sampling=sampling,
                        tokens_per_sample=tokens_per_sample, n_sample=n_sample,
                        w_intra=w_intra, w_inter=w_inter, gram_center=gram_center)

    def forward(self, unet_feat, cdg_tokens, generator=None, return_stats=False):
        proj = self.projector(unet_feat)                            # (B, N_cdg, cdg_dim)
        return manifold_alignment_loss(proj, cdg_tokens, generator=generator,
                                       return_stats=return_stats, **self.cfg)


# ── Bottleneck tap: grab the MiddleBlock output without editing UNet3D ───────────
class BottleneckTap:
    """Forward-hook capture of the U-Net middle-stage (bottleneck) feature — the
    single U-REPA alignment point. Non-invasive; register on `voxbind.unet3d.middle`.

        tap = BottleneckTap(voxbind.unet3d.middle)
        out = voxbind(...)              # normal forward
        feat = tap.feature             # (B, C_bot, g, g, g)
        ...
        tap.remove()

    The hooked tensor is part of the live graph, so the alignment loss backprops into
    the U-Net (once unfrozen).
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
#   align = UREPAAlignment(unet_ch=512, cdg_dim=640)   # exp_sig0.9 ↔ champion CDG
#   tap   = BottleneckTap(voxbind.unet3d.middle)
#   ...
#   pred      = voxbind(noisy_lig, pocket)             # runs U-Net → tap.feature set
#   L_denoise = denoise_loss(pred, target)
#   if has_density.any():                              # subset with cached CDG tokens
#       L_align, stats = align(tap.feature[has_density], cdg_tokens[has_density],
#                              return_stats=True)
#       loss = L_denoise + lam * L_align
#       log(stats["teacher_entropy"], stats["intra"], stats["inter"])   # tau / axis diagnostics
#   else:
#       loss = L_denoise
#
# Unfreezing ladder: start with only `align.projector` trainable, U-Net frozen — the
# frozen-U-Net alignment-loss floor measures how compatible the two reps already are.


if __name__ == "__main__":  # shape / grad / stats smoke test
    torch.manual_seed(0)
    B = 8
    unet_feat = torch.randn(B, 512, 8, 8, 8, requires_grad=True)
    cdg = torch.randn(B, 512, 640)                                  # frozen target (dim 512)
    head = UREPAAlignment(unet_ch=512, cdg_dim=640, unet_grid=8, cdg_grid=8)

    proj = head.projector(unet_feat)
    assert proj.shape == (B, 512, 640), proj.shape

    for sampling in ("split", "block", "flat", "pool"):
        for mode in ("relkl", "gram"):
            head.cfg.update(sampling=sampling, mode=mode)
            loss, st = head(unet_feat, cdg, return_stats=True)
            assert loss.ndim == 0 and torch.isfinite(loss), (sampling, mode, loss)
            loss.backward(retain_graph=True)
            assert unet_feat.grad is not None and torch.isfinite(unet_feat.grad).all()
            ent = float(st["teacher_entropy"]); frac = float(st["intra_frac"])
            print(f"[{sampling:5s}/{mode:5s}] loss {loss.item():.4f}  "
                  f"teacher_entropy {ent:.3f}  intra_frac {frac:.3f}")

    # entropy vs tau — soft tau → near-uniform teacher (entropy → log(m))
    for tau in (1.0, 0.5, 0.3, 0.1):
        head.cfg.update(sampling="block", mode="relkl", tau=tau, tokens_per_sample=64)
        _, st = head(unet_feat, cdg, return_stats=True)
        m = B * 64
        print(f"[tau {tau:>4}] teacher_entropy {float(st['teacher_entropy']):.3f} "
              f"(uniform = log(m-1) = {torch.log(torch.tensor(m - 1.0)).item():.3f})")

    # coarser bottleneck (4³) → trilinear upsample; and downscale guard
    up = UREPAProjector(unet_ch=256, cdg_dim=640, unet_grid=4, cdg_grid=8)
    assert up(torch.randn(B, 256, 4, 4, 4)).shape == (B, 512, 640)
    print("[upsample] 4³→8³ OK ; [guard] downscale rejected:", end=" ")
    try:
        UREPAProjector(unet_ch=256, cdg_dim=640, unet_grid=16, cdg_grid=8); print("NO (BUG)")
    except AssertionError:
        print("yes")
